import copy
from dataclasses import asdict
import math
from pathlib import Path
import random
import time
import torch
from torch.utils.checkpoint import checkpoint
from .io import DeadlineReached, event, read_json, atomic_json, object_hash
from .methods import (ANCHOR, FRESH, GKD, HARD, LAG, SGD, TAIL,
                      allocate_noise_statistics, constant_noise_coefficient,
                      decoder, disagreement_coefficients, gini, kl, name,
                      prefix_weights, project_direction, target)


def completion_inputs(policy, record):
    prompt=record["prompt_ids"]
    completion=record["completion_ids"]
    if not prompt or not completion:
        raise ValueError("empty token sequence; cannot train a completion")
    full=torch.tensor([prompt+completion],device=policy.device)
    return full[:,:-1],full[0,len(prompt):],len(prompt)-1


def _charge(costs, key, amount=1):
    if costs is not None:
        costs[key] = costs.get(key, 0) + amount


def _logits(policy, hidden, costs, kind="reference_logit_chunks"):
    _charge(costs, kind)
    return policy.logits(hidden)


def record_loss(policy, record, method, decode, parameters, chunk=32, retention=False, costs=None):
    """Completion-only loss with O(T) frozen sequence statistics/randomness.

    The round teacher is detached and stays fixed for every update. Mean gates,
    mean prefix weights and noise budgets use the entire completion, not each
    vocabulary chunk. Targets remain dense within a chunk: no sparse speedup is
    implied. ``costs`` counts actual forwards, including checkpoint recomputes.
    """
    method=name(method)
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("vocab_chunk must be a positive integer")
    if method in GKD and not 0 < parameters.get("epsilon", .02) <= 1:
        raise ValueError("GKD epsilon must be in (0,1] for a full-support target")
    ids,labels,start=completion_inputs(policy,record)
    def reference(adapter, kind):
        _charge(costs, kind)
        _charge(costs, "reference_forward_passes")
        return policy.hidden(ids, adapter)[0,start:].detach()
    with torch.no_grad():
        teacher=None if method in HARD or retention else reference("teacher", "teacher_forward_passes")
        anchor=reference("anchor", "anchor_forward_passes") if method in ANCHOR or retention else None
        lag=reference("lag", "lag_forward_passes") if method in LAG and not retention else None
        if method == "temporal_current_twice" and not retention:
            lag=reference("teacher", "duplicate_current_forward_passes")
        weights=torch.ones(len(labels),device=policy.device)
        coefficients=None
        if method in {"prefix_damping", "constant_mean_weight", "prefix_weight_one"} and not retention:
            ratios=[]
            for i in range(0,len(labels),chunk):
                y=labels[i:i+chunk,None]
                lp=_logits(policy,teacher[i:i+chunk],costs,"statistics_logit_chunks").float().log_softmax(-1).gather(-1,y).squeeze(-1)
                la=_logits(policy,anchor[i:i+chunk],costs,"statistics_logit_chunks").float().log_softmax(-1).gather(-1,y).squeeze(-1)
                ratios.append(lp-la)
            weights=prefix_weights(torch.cat(ratios),parameters.get("prefix_cap",5.))
            if method == "constant_mean_weight":
                weights=weights.mean().expand_as(weights)
            elif method == "prefix_weight_one":
                weights=torch.ones_like(weights)
        if method in {"noise_allocation", "fresh_alpha_budget_matched", "constant_mean_gate"} and not retention:
            distances, variances, gates = [], [], []
            for i in range(0,len(labels),chunk):
                logits=_logits(policy,teacher[i:i+chunk],costs,"statistics_logit_chunks")
                p=logits.float().softmax(-1)
                mu=decoder(logits,decode)
                if method == "constant_mean_gate":
                    lm=decoder(_logits(policy,lag[i:i+chunk],costs,"statistics_logit_chunks"),decode)
                    gates.append(disagreement_coefficients(mu,lm,parameters.get("alpha",.5),parameters.get("gate_lambda",.1)))
                else:
                    distances.append((mu-p).square().sum(-1))
                    variances.append(gini(mu))
            if method == "constant_mean_gate":
                coefficients=torch.cat(gates).mean().expand(len(labels))
            elif method == "noise_allocation":
                coefficients=allocate_noise_statistics(torch.cat(distances),torch.cat(variances),parameters.get("noise_budget",.02))
            else:
                coefficients=constant_noise_coefficient(torch.cat(variances),parameters.get("noise_budget",.02))
        # One draw tensor per complete frozen sequence gives identical targets
        # for every vocabulary chunk size, on CPU and CUDA alike.
        uniforms=None
        if not retention and method in FRESH | TAIL:
            draws=parameters.get("draws",4) if method in TAIL else 1
            uniforms=torch.rand((len(labels),draws),device=policy.device)
            _charge(costs,"fresh_categorical_draws",len(labels)*draws)
    # Reference forwards must precede this forward. Keep student active through
    # checkpoint recomputation/backward; changing adapters inside backward is invalid.
    student=policy.hidden(ids,"student",train=True)[0,start:]
    _charge(costs,"student_forward_passes")
    total=student.new_zeros((),dtype=torch.float32)
    for i in range(0,len(labels),chunk):
        stop=min(len(labels),i+chunk)
        y=labels[i:stop]
        w=weights[i:stop]
        th=teacher[i:stop] if teacher is not None else student[i:stop].detach()
        ah=anchor[i:stop] if anchor is not None else th
        lh=lag[i:stop] if lag is not None else th
        a=coefficients[i:stop] if coefficients is not None else None
        u=uniforms[i:stop] if uniforms is not None else None
        def block(sh,th,ah,lh,y,w,a,u):
            log_student=_logits(policy,sh,costs,"student_logit_chunks").float().log_softmax(-1)
            log_q=None
            with torch.no_grad():
                if retention:
                    log_q=_logits(policy,ah,costs).float().log_softmax(-1)
                    q=log_q.exp()
                elif method in HARD:
                    q=None
                else:
                    logits=_logits(policy,th,costs)
                    p=logits.float().softmax(-1)
                    mu=decoder(logits,decode)
                    ap=_logits(policy,ah,costs).float().softmax(-1) if method in ANCHOR else None
                    lm=decoder(_logits(policy,lh,costs),decode) if method in LAG or method == "temporal_current_twice" else None
                    if method in GKD:
                        eps=parameters.get("epsilon",.02)
                        log_p=logits.float().log_softmax(-1)
                        log_q=(log_p if eps == 1 else
                               torch.logaddexp(mu.log()+math.log1p(-eps),log_p+math.log(eps)))
                        q=log_q.exp()
                    else:
                        q=target(method,p,mu,y,anchor=ap,lag=lm,coefficients=a,
                                 uniforms=u,current_again=lm,**parameters)
            if q is None:
                losses=-log_student.gather(-1,y[:,None]).squeeze(-1)
            elif method == "gkd_rkl" and not retention:
                losses=(log_student.exp()*(log_student-log_q)).sum(-1)
            elif method == "gkd_jsd" and not retention:
                log_mid=torch.logaddexp(log_q,log_student)-math.log(2)
                losses=((q*(log_q-log_mid)).sum(-1)+
                        (log_student.exp()*(log_student-log_mid)).sum(-1))/2
            else:
                losses=-(q*log_student).sum(-1)
                if retention or method == "gkd_fkl":
                    losses=losses+(q*log_q).sum(-1)
            return (losses*w).sum()
        total=total+checkpoint(block,student[i:stop],th,ah,lh,y,w,a,u,use_reentrant=False)
    # Equal-sequence mean, then optional equal-prompt allocation correction.
    return total/len(labels)*record.get("loss_weight",1.0)


@torch.no_grad()
def snapshot_hidden(policy, records, costs=None):
    result=[]
    for record in records:
        ids,_,start=completion_inputs(policy,record)
        result.append(policy.hidden(ids,"student")[0,start:].detach())
        _charge(costs,"backtrack_forward_passes")
    return result


@torch.no_grad()
def actual_step_kl(policy, records, old_hidden, chunk, costs=None):
    values=[]
    for record,old in zip(records,old_hidden):
        ids,labels,start=completion_inputs(policy,record)
        now=policy.hidden(ids,"student")[0,start:]
        _charge(costs,"backtrack_forward_passes")
        value=0.
        for i in range(0,len(labels),chunk):
            p=_logits(policy,old[i:i+chunk],costs,"backtrack_logit_chunks").float().softmax(-1)
            q=_logits(policy,now[i:i+chunk],costs,"backtrack_logit_chunks").float().softmax(-1)
            value+=float(kl(p,q).sum())
        values.append(value/len(labels))
    return sum(values)/len(values)


def train_round(policy, records, method, decode, config, folder, deadline, seed, parent_sha,
                rollout_callback=None):
    """Train bounded update groups and persist model, optimizer, RNG and progress.

    ``rollout_callback(group_index, batch)`` returns the same number of fresh
    current-student records. A completed callback batch and its post-rollout RNG
    are checkpointed before optimization, so even an interrupted update can
    replay its exact targets. The caller owns partial generation persistence.

    A training clock admits whole updates. Its first update starts the clock;
    later updates stop at the first boundary at/above the cap. ``max_epochs``
    and optional ``max_updates`` give finite bounds even if updates are cheap.
    Loading a completed arm never extends this budget or reruns callbacks.
    """
    method=name(method)
    parameters=config.get("method_parameters",{})
    wall=config.get("training_wall_seconds")
    if wall is not None and (not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall <= 0):
        raise ValueError("training_wall_seconds must be finite and positive")
    epochs=config.get("max_epochs",config["epochs"]) if wall is not None else config["epochs"]
    accum=config["grad_accum"]
    if not records or not isinstance(epochs,int) or epochs < 1 or not isinstance(accum,int) or accum < 1:
        raise ValueError("training requires records and positive integer epoch/accumulation bounds")
    maximum=config.get("max_updates")
    if maximum is not None and (not isinstance(maximum,int) or maximum < 1):
        raise ValueError("max_updates must be a positive integer")
    optimizer_name=config.get("optimizer","sgd" if method in SGD else "adamw")
    if optimizer_name not in {"adamw","sgd"}:
        raise ValueError("optimizer must be adamw or sgd")
    if method in SGD and optimizer_name != "sgd":
        raise ValueError(f"{method} requires the matched SGD optimizer")
    multiplier=config.get("learning_rate_multiplier",1.)
    lr=config["learning_rate"]*multiplier*(.5 if method == "lower_lr" else 1.)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate and learning_rate_multiplier must give a finite positive rate")
    if not math.isfinite(config["max_grad_norm"]) or config["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    identity_expected=(method == "head_identity" or method in GKD and (
        parameters.get("epsilon",.02) == 1 or
        decode.temperature == 1 and decode.top_k == 0 and decode.top_p == 1))
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/"adapter.pt"
    progress=folder/"training-state.pt"
    timing=folder/"training-timing.json"
    summary_path=folder/"training.json"
    chunk=config["vocab_chunk"]
    configuration=object_hash({"method":method,"decode":asdict(decode),"config":config,
                               "seed":seed,"records":records,"per_update_rollout":rollout_callback is not None})
    trainable=policy.trainable()
    optimizer=(torch.optim.SGD(trainable,lr=lr) if optimizer_name == "sgd" else
               torch.optim.AdamW(trainable,lr=lr,weight_decay=0.0,foreach=False))
    scaler=torch.amp.GradScaler("cuda",enabled=policy.device.type=="cuda")
    order=[]
    for epoch in range(epochs):
        indices=list(range(len(records)))
        random.Random(seed+epoch).shuffle(indices)
        order+=indices
    groups=[order[i:i+accum] for i in range(0,len(order),accum)]
    if maximum is not None:
        groups=groups[:maximum]
    next_group=successful=skipped=zero_updates=0
    elapsed=0.
    pending_batch=None
    costs={key:0 for key in ("student_forward_passes","reference_forward_passes",
        "teacher_forward_passes","anchor_forward_passes","lag_forward_passes",
        "duplicate_current_forward_passes","objective_backward_passes","retention_backward_passes",
        "backtrack_trials","rejected_updates","overflow_updates","rollout_batches",
        "rollout_records","completion_tokens","checkpoints","interrupted_attempts","failed_attempts")}
    costs.update({key:0. for key in ("objective_seconds","retention_seconds","optimizer_seconds",
                                   "backtrack_seconds","rollout_seconds","checkpoint_seconds")})

    def restore_rng(state):
        torch.set_rng_state(state["cpu_rng"])
        random.setstate(state["python_rng"])
        if policy.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    if progress.exists():
        # Validate before loading adapter or optimizer into the live policy.
        state=torch.load(progress,map_location="cpu",weights_only=True)
        if state["parent_sha"] != parent_sha:
            raise ValueError("training checkpoint parent mismatch")
        if state.get("configuration") != configuration:
            raise ValueError("training checkpoint configuration mismatch")
        policy.load_checkpoint(progress)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        next_group=state["next_group"]
        successful=state["successful_updates"]
        skipped=state["skipped_updates"]
        zero_updates=state["zero_gradient_updates"]
        elapsed=state["elapsed_seconds"]
        pending_batch=state.get("pending_batch")
        costs.update(state["costs"])
        if timing.exists():
            measured=read_json(timing)
            if measured["next_group"] == next_group and measured["elapsed_seconds"] >= elapsed:
                elapsed=measured["elapsed_seconds"]
                costs.update(measured["costs"])
        restore_rng(state)
        if final.exists() and summary_path.exists() and read_json(summary_path).get("status") == "complete":
            policy.load_checkpoint(final)
            return final

    def sync():
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)

    sync()
    started=time.monotonic()

    def current_elapsed():
        return elapsed+time.monotonic()-started

    def fields():
        return dict(checkpoint_version=2,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            next_group=next_group,parent_sha=parent_sha,successful_updates=successful,
            skipped_updates=skipped,zero_gradient_updates=zero_updates,elapsed_seconds=current_elapsed(),
            cpu_rng=torch.get_rng_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if policy.device.type == "cuda" else [],
            configuration=configuration,method=method,pending_batch=pending_batch,costs=costs)

    def persist():
        before=time.monotonic()
        policy.save_checkpoint(progress,**fields())
        _charge(costs,"checkpoints")
        _charge(costs,"checkpoint_seconds",time.monotonic()-before)
        # This sidecar captures the just-finished checkpoint cost. A crash
        # between these atomic writes still retains the completed update.
        atomic_json(timing,{"next_group":next_group,"elapsed_seconds":current_elapsed(),"costs":costs})

    if not progress.exists():
        persist()
    stop_reason="max_updates" if maximum is not None and len(groups) == maximum else "epoch_bound"
    for group_index in range(next_group,len(groups)):
        # Admit the initial update, then stop only at update boundaries.
        if wall is not None and next_group > 0 and current_elapsed() >= wall:
            stop_reason="training_wall_seconds"
            break
        deadline.check(45)
        try:
            batch=pending_batch
            if batch is None:
                batch=[records[i] for i in groups[group_index]]
                if rollout_callback is not None:
                    before=time.monotonic()
                    batch=rollout_callback(group_index,batch)
                    sync()
                    _charge(costs,"rollout_seconds",time.monotonic()-before)
                    if not isinstance(batch,list) or len(batch) != len(groups[group_index]):
                        raise ValueError("rollout_callback must return one record per input record")
                    for record in batch:
                        completion_inputs(policy,record)
                    _charge(costs,"rollout_batches")
                    _charge(costs,"rollout_records",len(batch))
                    pending_batch=batch
                    persist()
            optimizer.zero_grad(set_to_none=True)
            old_hidden=snapshot_hidden(policy,batch,costs) if method == "kl_backtrack" else None
            before=time.monotonic()
            loss_value=0.
            for record in batch:
                loss=record_loss(policy,record,method,decode,parameters,chunk,costs=costs)/len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss; experiment invalid")
                loss_value+=float(loss.detach())
                scaler.scale(loss).backward()
                _charge(costs,"objective_backward_passes")
                _charge(costs,"completion_tokens",len(record["completion_ids"]))
            scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,config["max_grad_norm"])
            sync()
            _charge(costs,"objective_seconds",time.monotonic()-before)
            finite=bool(torch.isfinite(grad_norm))
            if not finite and not scaler.is_enabled():
                raise FloatingPointError("nonfinite gradient without loss scaling")
            zero=finite and (float(grad_norm) == 0. or identity_expected and float(grad_norm) <= 1e-7)
            norm_factor=None
            projected_norm=None
            if method in {"gradient_projection","sgd_norm_matched"} and finite and not zero:
                before=time.monotonic()
                h=[torch.zeros_like(p) for p in trainable]
                for record in batch:
                    retention=record_loss(policy,record,method,decode,parameters,chunk,retention=True,costs=costs)/len(batch)
                    if not torch.isfinite(retention):
                        raise FloatingPointError("nonfinite retention loss; experiment invalid")
                    derivatives=torch.autograd.grad(retention,trainable,allow_unused=True)
                    _charge(costs,"retention_backward_passes")
                    for dest,source in zip(h,derivatives):
                        if source is not None:
                            dest.add_(source)
                original=[-p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
                direction=project_direction(original,h)
                raw_norm=sum(d.float().square().sum() for d in original).sqrt()
                projected_norm=sum(d.float().square().sum() for d in direction).sqrt()
                if not torch.isfinite(projected_norm):
                    raise FloatingPointError("nonfinite projected gradient; experiment invalid")
                norm_factor=float((projected_norm/raw_norm.clamp_min(1e-30)).clamp_max(1))
                if method == "sgd_norm_matched":
                    # Same proposed projected norm, unchanged original direction.
                    direction=[norm_factor*d for d in original]
                for p,d in zip(trainable,direction):
                    p.grad=-d
                zero=float(projected_norm) == 0.
                sync()
                _charge(costs,"retention_seconds",time.monotonic()-before)
            old_values=[p.detach().clone() for p in trainable] if old_hidden is not None else None
            old_optimizer=copy.deepcopy(optimizer.state_dict()) if old_hidden is not None else None
            before=time.monotonic()
            old_scale=scaler.get_scale()
            if not zero:
                scaler.step(optimizer)
            scaler.update()
            sync()
            _charge(costs,"optimizer_seconds",time.monotonic()-before)
            overflow=scaler.get_scale() < old_scale
            accepted=finite and not overflow and not zero
            gamma=1. if accepted else 0.
            measured=None
            rejection="zero_gradient" if zero else "overflow" if overflow else None
            if old_hidden is not None and accepted:
                before=time.monotonic()
                proposal=[p.detach().clone()-old for p,old in zip(trainable,old_values)]
                accepted=False
                for trial in range(9):
                    deadline.check(20)
                    gamma=2.**(-trial)
                    with torch.no_grad():
                        for p,old,delta in zip(trainable,old_values,proposal):
                            p.copy_(old+gamma*delta)
                    measured=actual_step_kl(policy,batch,old_hidden,chunk,costs)
                    _charge(costs,"backtrack_trials")
                    if math.isfinite(measured) and measured <= parameters.get("step_kl_budget",.01):
                        accepted=True
                        break
                if not accepted:
                    with torch.no_grad():
                        for p,old in zip(trainable,old_values):
                            p.copy_(old)
                    optimizer.load_state_dict(old_optimizer)
                    gamma=0.
                    rejection="kl_budget"
                    _charge(costs,"rejected_updates")
                sync()
                _charge(costs,"backtrack_seconds",time.monotonic()-before)
            successful+=int(accepted)
            skipped+=int(not accepted and not zero)
            zero_updates+=int(zero)
            _charge(costs,"overflow_updates",int(overflow))
            next_group=group_index+1
            pending_batch=None
            persist()
            event(folder,"optimizer_step",group=group_index,loss=loss_value,
                  grad_norm=float(grad_norm) if finite else None,accepted=accepted,overflow=overflow,
                  zero_gradient=zero,rejection_reason=rejection,backtrack_gamma=gamma,
                  measured_step_kl=measured if measured is None or math.isfinite(measured) else None,
                  norm_matching_factor=norm_factor,
                  projected_gradient_norm=float(projected_norm) if projected_norm is not None else None,
                  elapsed_seconds=current_elapsed())
        except Exception as exc:
            # A deadline during backtracking must not leave a rejected proposal
            # or its optimizer state active. Keep spent work in the cost ledger.
            sync()
            state=policy.load_checkpoint(progress)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            next_group=state["next_group"]
            successful=state["successful_updates"]
            skipped=state["skipped_updates"]
            zero_updates=state["zero_gradient_updates"]
            pending_batch=state.get("pending_batch")
            restore_rng(state)
            _charge(costs,"interrupted_attempts" if isinstance(exc,DeadlineReached) else "failed_attempts")
            persist()
            event(folder,"training_interrupted" if isinstance(exc,DeadlineReached) else "training_failed",
                  group=group_index,error_type=type(exc).__name__,elapsed_seconds=current_elapsed())
            raise
    identity=successful == 0 and zero_updates == next_group and identity_expected and skipped == 0
    summary={"successful_updates":successful,"skipped_updates":skipped,"zero_gradient_updates":zero_updates,
        "attempted_updates":next_group,"planned_updates":len(groups),"parent_sha":parent_sha,"method":method,
        "configuration":configuration,"optimizer":optimizer_name,"learning_rate":lr,
        "elapsed_seconds":current_elapsed(),"training_wall_seconds":wall,"stop_reason":stop_reason,
        "coefficient_scope":"whole_frozen_completion_sequence","costs":costs,
        "optimization_valid":skipped == 0 and costs["failed_attempts"] == 0,
        "trained":successful > 0,"training_outcome":"identity_zero_update" if identity else "trained" if successful else "zero_updates",
        "status":"complete" if successful or identity else "invalid"}
    if successful == 0 and not identity:
        atomic_json(summary_path,summary)
        raise RuntimeError("zero successful optimizer updates; do not count as a trained arm")
    before=time.monotonic()
    policy.save_checkpoint(final,**fields())
    _charge(costs,"checkpoint_seconds",time.monotonic()-before)
    _charge(costs,"checkpoints")
    summary["elapsed_seconds"]=current_elapsed()
    atomic_json(summary_path,summary)
    return final
