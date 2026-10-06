import copy
import math
from pathlib import Path
import random
import time
import torch
from torch.utils.checkpoint import checkpoint
from .io import event, read_json, atomic_json
from .methods import (ANCHOR, HARD, LAG, decoder, kl, name, prefix_weights,
                      project_direction, target)


def completion_inputs(policy, record):
    prompt=record["prompt_ids"]
    completion=record["completion_ids"]
    if not prompt or not completion:
        raise ValueError("empty token sequence; cannot train a completion")
    full=torch.tensor([prompt+completion],device=policy.device)
    return full[:,:-1],full[0,len(prompt):],len(prompt)-1


def record_loss(policy, record, method, decode, parameters, chunk=32, retention=False):
    method=name(method)
    ids,labels,start=completion_inputs(policy,record)
    with torch.no_grad():
        teacher=None if method in HARD and not retention else policy.hidden(ids,"teacher")[0,start:].detach()
        anchor=policy.hidden(ids,"anchor")[0,start:].detach() if method in ANCHOR or retention else None
        lag=policy.hidden(ids,"lag")[0,start:].detach() if method in LAG else None
        weights=torch.ones(len(labels),device=policy.device)
        if method=="prefix_damping":
            ratios=[]
            for i in range(0,len(labels),chunk):
                y=labels[i:i+chunk,None]
                lp=policy.logits(teacher[i:i+chunk]).float().log_softmax(-1).gather(-1,y).squeeze(-1)
                la=policy.logits(anchor[i:i+chunk]).float().log_softmax(-1).gather(-1,y).squeeze(-1)
                ratios.append(lp-la)
            weights=prefix_weights(torch.cat(ratios))
    # Reference forwards must precede this forward. Keep student active through
    # checkpoint recomputation/backward; changing adapters inside backward is invalid.
    student=policy.hidden(ids,"student",train=True)[0,start:]
    total=student.new_zeros((),dtype=torch.float32)
    for i in range(0,len(labels),chunk):
        stop=min(len(labels),i+chunk)
        y=labels[i:stop]
        w=weights[i:stop]
        th=teacher[i:stop] if teacher is not None else student[i:stop].detach()
        ah=anchor[i:stop] if anchor is not None else th
        lh=lag[i:stop] if lag is not None else th
        def block(sh,th,ah,lh,y,w):
            log_student=policy.logits(sh).float().log_softmax(-1)
            with torch.no_grad():
                if retention:
                    q=policy.logits(ah).float().softmax(-1)
                elif method in HARD:
                    q=None
                else:
                    logits=policy.logits(th)
                    p=logits.float().softmax(-1)
                    mu=decoder(logits,decode)
                    ap=policy.logits(ah).float().softmax(-1) if method in ANCHOR else None
                    lm=decoder(policy.logits(lh),decode) if method in LAG else None
                    q=target(method,p,mu,y,anchor=ap,lag=lm,**parameters)
            if q is None:
                losses=-log_student.gather(-1,y[:,None]).squeeze(-1)
            else:
                losses=-(q*log_student).sum(-1)
                if retention:
                    losses=losses+(q*q.clamp_min(1e-30).log()).sum(-1)
            return (losses*w).sum()
        total=total+checkpoint(block,student[i:stop],th,ah,lh,y,w,use_reentrant=False)
    # Equal-sequence mean, then optional equal-prompt allocation correction.
    return total/len(labels)*record.get("loss_weight",1.0)


@torch.no_grad()
def snapshot_hidden(policy, records):
    result=[]
    for record in records:
        ids,_,start=completion_inputs(policy,record)
        result.append(policy.hidden(ids,"student")[0,start:].detach())
    return result


@torch.no_grad()
def actual_step_kl(policy, records, old_hidden, chunk):
    values=[]
    for record,old in zip(records,old_hidden):
        ids,labels,start=completion_inputs(policy,record)
        now=policy.hidden(ids,"student")[0,start:]
        value=0.
        for i in range(0,len(labels),chunk):
            p=policy.logits(old[i:i+chunk]).float().softmax(-1)
            q=policy.logits(now[i:i+chunk]).float().softmax(-1)
            value+=float(kl(p,q).sum())
        values.append(value/len(labels))
    return sum(values)/len(values)


def train_round(policy, records, method, decode, config, folder, deadline, seed, parent_sha):
    method=name(method)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/"adapter.pt"
    progress=folder/"training-state.pt"
    parameters=config.get("method_parameters",{})
    chunk=config["vocab_chunk"]
    trainable=policy.trainable()
    lr=config["learning_rate"]*(0.5 if method=="lower_lr" else 1.0)
    is_sgd=method in {"gradient_projection","full_soft_sgd"}
    optimizer=(torch.optim.SGD(trainable,lr=lr) if is_sgd else
               torch.optim.AdamW(trainable,lr=lr,weight_decay=0.0,foreach=False))
    scaler=torch.amp.GradScaler("cuda",enabled=policy.device.type=="cuda")
    order=[]
    for epoch in range(config["epochs"]):
        indices=list(range(len(records)))
        random.Random(seed+epoch).shuffle(indices)
        order+=indices
    groups=[order[i:i+config["grad_accum"]] for i in range(0,len(order),config["grad_accum"])]
    next_group=0
    successful=0
    skipped=0
    elapsed=0.
    if progress.exists():
        state=policy.load_checkpoint(progress)
        if state["parent_sha"]!=parent_sha:
            raise ValueError("training checkpoint parent mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        next_group=state["next_group"]
        successful=state["successful_updates"]
        skipped=state["skipped_updates"]
        elapsed=state["elapsed_seconds"]
        torch.set_rng_state(state["cpu_rng"])
        if policy.device.type=="cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    started=time.monotonic()
    for group_index in range(next_group,len(groups)):
        deadline.check(45)
        batch=[records[i] for i in groups[group_index]]
        optimizer.zero_grad(set_to_none=True)
        old_hidden=snapshot_hidden(policy,batch) if method=="kl_backtrack" else None
        loss_value=0.
        for record in batch:
            loss=record_loss(policy,record,method,decode,parameters,chunk)/len(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; experiment invalid")
            loss_value+=float(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(trainable,config["max_grad_norm"])
        if not torch.isfinite(grad_norm) and not scaler.is_enabled():
            raise FloatingPointError("nonfinite gradient without loss scaling")
        finite=bool(torch.isfinite(grad_norm))
        if method=="gradient_projection" and finite:
            h=[torch.zeros_like(p) for p in trainable]
            for record in batch:
                retention=record_loss(policy,record,method,decode,parameters,chunk,retention=True)/len(batch)
                derivatives=torch.autograd.grad(retention,trainable,allow_unused=True)
                for dest,source in zip(h,derivatives):
                    if source is not None:
                        dest.add_(source)
            direction=project_direction([-p.grad for p in trainable],h)
            for p,d in zip(trainable,direction):
                p.grad=-d
        old_values=[p.detach().clone() for p in trainable] if old_hidden is not None else None
        old_optimizer=copy.deepcopy(optimizer.state_dict()) if old_hidden is not None else None
        old_scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        overflow=scaler.get_scale()<old_scale
        accepted=finite and not overflow
        gamma=1.
        measured=None
        if old_hidden is not None and accepted:
            proposal=[p.detach().clone()-old for p,old in zip(trainable,old_values)]
            accepted=False
            for trial in range(9):
                deadline.check(20)
                gamma=2.**(-trial)
                with torch.no_grad():
                    for p,old,delta in zip(trainable,old_values,proposal):
                        p.copy_(old+gamma*delta)
                measured=actual_step_kl(policy,batch,old_hidden,chunk)
                if math.isfinite(measured) and measured<=parameters.get("step_kl_budget",0.01):
                    accepted=True
                    break
            if not accepted:
                with torch.no_grad():
                    for p,old in zip(trainable,old_values):
                        p.copy_(old)
                optimizer.load_state_dict(old_optimizer)
                gamma=0.
                measured=0.
        successful+=int(accepted)
        skipped+=int(not accepted)
        elapsed_now=elapsed+time.monotonic()-started
        policy.save_checkpoint(progress,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            next_group=group_index+1,parent_sha=parent_sha,successful_updates=successful,
            skipped_updates=skipped,elapsed_seconds=elapsed_now,cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if policy.device.type=="cuda" else [])
        event(folder,"optimizer_step",group=group_index,loss=loss_value,
              grad_norm=float(grad_norm) if finite else None,accepted=accepted,overflow=overflow,
              backtrack_gamma=gamma,measured_step_kl=measured,elapsed_seconds=elapsed_now)
    if successful==0:
        raise RuntimeError("zero successful optimizer updates; do not count as a trained arm")
    policy.save_checkpoint(final,parent_sha=parent_sha,successful_updates=successful,skipped_updates=skipped,
        method=method,elapsed_seconds=elapsed+time.monotonic()-started)
    atomic_json(folder/"training.json",{"successful_updates":successful,"skipped_updates":skipped,
        "attempted_updates":len(groups),"parent_sha":parent_sha,"method":method,
        "elapsed_seconds":elapsed+time.monotonic()-started,
        "optimization_valid":skipped==0,"status":"complete"})
    return final
