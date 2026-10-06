"""One admitted native training round; no benchmark tests or scoring imports.

The outer harness owns admission, deadlines, one-device policy, and immutable
attempt directories. This worker preserves completed generation and training
inside a copied attempt without granting another round budget.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import time

import torch

from .benchmarks import CLEAN_TRAIN_SHA256, DESIGN_DATA
from .io import atomic_json, digest, jsonl, object_hash, read_json, read_jsonl, stable_seed
from .methods import Decode, allocate_prompts, decoder, gini, name
from .suite_design import MODELS, resolve_arm
from .train import train_round


ROOT=Path(__file__).resolve().parents[1]


def _local_ref(path, root):
    path,root=Path(path).resolve(),Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("output artifact escapes its receipt directory")
    return {"path":str(path.relative_to(root)),"sha256":digest(path)}


def _verify_ref(ref, root):
    if not isinstance(ref,dict) or not isinstance(ref.get("path"),str) or not ref.get("sha256"):
        raise ValueError("missing hashed training artifact reference")
    relative=Path(ref["path"])
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("training artifact reference escapes its receipt directory or is missing")
    if digest(path) != ref["sha256"]:
        raise ValueError(f"training artifact hash changed: {ref['path']}")
    return path


def _clean_ids():
    source=DESIGN_DATA/"train_prompts-clean-v2.jsonl"
    if digest(source) != CLEAN_TRAIN_SHA256:
        raise ValueError("pinned clean-v2 training source changed")
    return [row["task_id"] for row in read_jsonl(source)]


def _validate_records(rows, task_ids, counts):
    expected={(task, sample) for task,count in zip(task_ids,counts) for sample in range(count)}
    if len(counts) != 32 or sum(counts) != 64 or any(type(c) is not int or c < 1 for c in counts):
        raise ValueError("training allocation must cover 32 prompts with 64 sequences")
    actual=[(row.get("task_id"),row.get("sample_id")) for row in rows]
    if len(actual) != 64 or len(set(actual)) != 64 or set(actual) != expected:
        raise ValueError("training generation inventory has missing, duplicate, or foreign sample IDs")
    for row in rows:
        if (not isinstance(row.get("prompt_ids"),list) or not row["prompt_ids"] or
            not isinstance(row.get("completion_ids"),list) or not row["completion_ids"] or
            any(type(token) is not int or token < 0 for token in row["prompt_ids"]+row["completion_ids"]) or
            not isinstance(row.get("loss_weight"),(float,int)) or not math.isfinite(row["loss_weight"]) or row["loss_weight"] <= 0):
            raise ValueError("invalid generated training sequence or weight")


def verify_parent_receipts(paths, trajectory, expected_rounds):
    """Verify ordered R1..Rr receipts, their artifacts, IDs, lineage and costs.

    ``paths`` are receipt.json paths; all artifact refs are relative to each
    receipt's directory. Evaluation may pass its unit identity as ``trajectory``.
    Returned checkpoint/record paths are resolved paths for the next worker.
    """
    if type(expected_rounds) is not int or expected_rounds < 0 or len(paths) != expected_rounds:
        raise ValueError("parent receipts must contain every preceding round exactly once")
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("duplicate parent receipt")
    receipts=[]; checkpoints=[]; records=[]; total=0.
    expected_parent=trajectory["model_revision"]
    for round_index,item in enumerate(paths,1):
        path=Path(item).resolve()
        receipt=read_json(path)
        if receipt.get("status") != "complete" or receipt.get("round") != round_index:
            raise ValueError("parent receipt is incomplete or has the wrong round")
        for key in ("trajectory_id","arm_id","model","model_revision","seed"):
            if key in trajectory and receipt.get(key) != trajectory[key]:
                raise ValueError(f"parent receipt lineage mismatch: {key}")
        if receipt.get("training_lineage") != "clean-v2":
            raise ValueError("parent receipt is not clean-v2")
        resolved={key:_verify_ref(receipt.get(key),path.parent) for key in
                  ("checkpoint_ref","records_ref","training_ref","events_ref")}
        training=read_json(resolved["training_ref"])
        if (training.get("status") != "complete" or training.get("parent_sha") != expected_parent
                or receipt.get("parent_sha") != expected_parent):
            raise ValueError("parent training checkpoint lineage mismatch")
        generation_costs=(read_json(_verify_ref(receipt["generation_costs_ref"],path.parent))
                          if receipt.get("generation_costs_ref") else {})
        failed_rollout=generation_costs.get("failed_rollout_seconds",0.)
        if not isinstance(failed_rollout,(int,float)) or not math.isfinite(failed_rollout) or failed_rollout < 0:
            raise ValueError("invalid bound failed-rollout cost")
        phase_seconds=(training["elapsed_seconds"]-training.get("costs",{}).get("rollout_seconds",0.)
                       -failed_rollout)
        seconds=receipt.get("training_seconds")
        if (not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0 or
                not math.isclose(seconds,phase_seconds,rel_tol=1e-9,abs_tol=1e-8)):
            raise ValueError("parent training seconds do not match the bound training report")
        total+=seconds
        if not math.isclose(receipt.get("cumulative_training_seconds",-1),total,rel_tol=1e-9,abs_tol=1e-8):
            raise ValueError("parent cumulative training cost mismatch")
        counts=receipt.get("allocation",{}).get("counts",[])
        _validate_records(read_jsonl(resolved["records_ref"]),_clean_ids(),counts)
        events=[row for row in read_jsonl(resolved["events_ref"]) if row.get("kind") == "optimizer_step"]
        if ([row.get("group") for row in events] != list(range(training["attempted_updates"])) or
                len(events) != training["attempted_updates"]):
            raise ValueError("optimizer event inventory has missing or duplicate update groups")
        if (sum(row.get("accepted") is True for row in events) != training["successful_updates"] or
                sum(row.get("zero_gradient") is True for row in events) != training.get("zero_gradient_updates",0)):
            raise ValueError("optimizer events disagree with the training report")
        expected_parent=receipt["checkpoint_ref"]["sha256"]
        receipts.append(receipt); checkpoints.append(str(resolved["checkpoint_ref"]))
        records.append(str(resolved["records_ref"]))
    return {"receipts":receipts,"checkpoint_paths":checkpoints,"records_paths":records,
            "last_checkpoint":checkpoints[-1] if checkpoints else None,
            "checkpoint_sha256":expected_parent,"cumulative_training_seconds":total}


def _load_prompts(data):
    data=Path(data)
    manifest=read_json(data/"benchmarks-manifest.json")
    info=manifest.get("training",{})
    if (info.get("lineage") != "clean-v2" or info.get("count") != 32 or
            info.get("fields") != ["task_id","text"] or
            info.get("must_restart_from_base") is not True or info.get("exact_mbpp_plus_overlap") != []):
        raise ValueError("training requires the prompt-only clean-v2 manifest")
    path=data/"train_prompts.jsonl"
    if digest(path) != CLEAN_TRAIN_SHA256 or manifest.get("files",{}).get("train_prompts.jsonl") != CLEAN_TRAIN_SHA256:
        raise ValueError("clean-v2 training prompt hash changed")
    prompts=read_jsonl(path)
    if (len(prompts) != 32 or [row.get("task_id") for row in prompts] != _clean_ids() or
            any(set(row) != {"task_id","text"} for row in prompts)):
        raise ValueError("clean-v2 prompt fields or inventory changed")
    return prompts,digest(data/"benchmarks-manifest.json")


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _adapter_hash(policy, adapter):
    h=hashlib.sha256()
    for key,value in sorted(policy.state(adapter).items()):
        h.update(str((key,str(value.dtype),list(value.shape))).encode())
        h.update(_tensor_bytes(value))
    return h.hexdigest()


def _rng_hash(policy):
    h=hashlib.sha256(_tensor_bytes(torch.get_rng_state()))
    h.update(repr(random.getstate()).encode())
    if policy.device.type == "cuda":
        for state in torch.cuda.get_rng_state_all():
            h.update(_tensor_bytes(state))
    return h.hexdigest()


def _sync(policy):
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)


def train_job(job, data, output, deadline):
    """Execute one exact recursive round and return a bound inner receipt."""
    trajectory=deepcopy(job["trajectory"])
    round_index=job["round"]
    if trajectory.get("arm_id") == "base" or trajectory.get("arm",{}).get("method") == "base":
        raise ValueError("base is evaluation-only and cannot be a training job")
    if (type(round_index) is not int or not 1 <= round_index <= trajectory["rounds"] or
            trajectory.get("training_lineage","clean-v2") != "clean-v2"):
        raise ValueError("invalid clean-v2 recursive training round")
    if MODELS.get(trajectory["model"]) != trajectory["model_revision"]:
        raise ValueError("training model revision is not pinned")
    prompts,manifest_sha=_load_prompts(data)
    parents=verify_parent_receipts(job.get("parent_receipts",[]),trajectory,round_index-1)
    expected_parent=parents["last_checkpoint"]
    parent=job.get("parent_checkpoint")
    if (str(Path(parent).resolve()) if parent else None) != expected_parent:
        raise ValueError("training parent checkpoint differs from the verified prior round")
    expected_lag=parents["checkpoint_paths"][-2] if round_index >= 3 else None
    lag=job.get("lag_checkpoint")
    if (str(Path(lag).resolve()) if lag else None) != expected_lag:
        raise ValueError("lag checkpoint must be round r-2 (base for rounds 1 and 2)")
    arm=resolve_arm(trajectory["arm"],job.get("calibration",{}),round_index)
    for ref in arm.get("calibration_refs",[]):
        source=Path(ref.get("path",""))
        if not source.is_file() or digest(source) != ref.get("sha256"):
            raise ValueError("calibration source reference is missing or its hash changed")
    if arm["arm_id"] != trajectory["arm_id"]:
        raise ValueError("arm settings do not match the trajectory arm_id")
    method=name(arm["method"])
    config=read_json(ROOT/"configs/2080ti_8h.json")
    defaults=dict(config["method_parameters"])
    config.update(deepcopy(job.get("config",{})))
    defaults.update(config.get("method_parameters",{})); defaults.update(arm.get("parameters",{}))
    config.update(arm.get("train",{}))
    config.update(seed=trajectory["seed"],rounds=trajectory["rounds"],method_parameters=defaults)
    clock_tolerance=config.get("clock_match_tolerance_seconds",0.)
    if (isinstance(clock_tolerance,bool) or not isinstance(clock_tolerance,(int,float)) or
            not math.isfinite(clock_tolerance) or clock_tolerance < 0):
        raise ValueError("clock_match_tolerance_seconds must be finite and nonnegative")
    if config["train_prompts"] != 32 or config["samples_per_prompt"] != 2 or config["grad_accum"] != 4:
        raise ValueError("complete-suite training requires 32 prompts, 2 samples and 16 accumulation groups")
    if arm["rollout_refresh"] == "update" and (config["epochs"] != 1 or arm["allocation"] != "uniform"):
        raise ValueError("per-update GKD requires the frozen 64-sequence, one-epoch uniform allowance")
    decode_config=deepcopy(arm["decode"])
    schedule=arm["temperature_schedule"]
    if schedule == "root":
        decode_config["temperature"] **= 1/trajectory["rounds"]
    elif schedule == "once":
        if round_index > 1: decode_config["temperature"]=1.
    elif schedule != "fixed":
        raise ValueError("unknown temperature schedule")
    decode=Decode(**decode_config)
    exploration=arm["generation_exploration"]
    if not 0 <= exploration <= 1 or not math.isfinite(exploration):
        raise ValueError("invalid generation exploration")
    fixed=job.get("fixed_records")
    if method == "fixed_data" and round_index > 1:
        if not fixed or str(Path(fixed).resolve()) != parents["records_paths"][0]:
            raise ValueError("fixed_data requires the verified round-one records")
    elif fixed is not None:
        raise ValueError("fixed records are only valid after round one of fixed_data")
    parent_sha=parents["checkpoint_sha256"]
    identity={"trajectory":trajectory,"round":round_index,"arm":arm,"config":config,
        "data_sha256":CLEAN_TRAIN_SHA256,"manifest_sha256":manifest_sha,"parent_sha":parent_sha,
        "lag_sha256":digest(lag) if lag else trajectory["model_revision"],
        "fixed_records_sha256":digest(fixed) if fixed else None,
        "training_budget_digest":job.get("training_budget_digest")}
    # Bound calibration content, not mutable attempt-local absolute path names.
    identity["arm"]=deepcopy(arm)
    identity["arm"]["calibration_refs"]=[ref["sha256"] for ref in arm.get("calibration_refs",[])]
    signature=object_hash(identity)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    binding=output/"job-binding.json"
    if binding.exists() and read_json(binding).get("signature") != signature:
        raise ValueError("training job resume signature mismatch")
    if not binding.exists(): atomic_json(binding,{"signature":signature,"identity":identity})
    completed=output/"train-receipt.json"
    if completed.exists():
        receipt=read_json(completed)
        if receipt.get("job_signature") != signature:
            raise ValueError("completed training job signature mismatch")
        # Validate this round together with all preceding receipts before reuse.
        verify_parent_receipts([*job.get("parent_receipts",[]),str(completed)],trajectory,round_index)
        return receipt
    state_path=output/"worker-state.json"
    ledger=read_json(state_path) if state_path.exists() else {"signature":signature,"save_seconds":0.,
        "save_operations":0,"model_load_seconds":0.,"active_seconds":0.,"attempts":0}
    if ledger.get("signature") != signature:
        raise ValueError("worker cost ledger signature mismatch")
    for key,default in {"failed_generation_attempts":0,"failed_generation_seconds":0.,
                        "failed_rollout_seconds":0.,"callback_save_seconds":0.,
                        "generation_failures":[]}.items():
        ledger.setdefault(key,default)
    ledger["attempts"]+=1
    attempt_started=time.monotonic()
    def save(path,value,lines=False):
        started=time.monotonic()
        (jsonl if lines else atomic_json)(path,value)
        ledger["save_seconds"]+=time.monotonic()-started
        ledger["save_operations"]+=1
    try:
        deadline.check()
        from .model import Policy
        from .runner import training_prompt
        random.seed(trajectory["seed"])
        torch.manual_seed(trajectory["seed"])
        started=time.monotonic()
        policy=Policy.load(rank=config["lora_rank"],model=trajectory["model"],
                           revision=trajectory["model_revision"],model_path=job.get("model_path"))
        if parent: policy.load_checkpoint(parent)
        policy.copy("student","teacher")
        if lag: policy.load_checkpoint(lag,adapter="lag")
        _sync(policy)
        ledger["model_load_seconds"]+=time.monotonic()-started
        encoded=[policy.encode(training_prompt(row),config["max_prompt_tokens"]) for row in prompts]
        task_ids=[row["task_id"] for row in prompts]
        generation_root=output/"generation"; generation_root.mkdir(exist_ok=True)
        generation_context={"parent_sha":parent_sha,"round":round_index,"model":trajectory["model"],
            "model_revision":trajectory["model_revision"],"decode":asdict(decode),
            "max_new_tokens":config["max_new_tokens"],"max_prompt_tokens":config["max_prompt_tokens"],
            "seed":trajectory["seed"],"exploration":exploration,"encoded":encoded,"task_ids":task_ids}
        counts=[2]*32; proxy_values=[]; proxy_seconds=0.
        if arm["allocation"] in {"adaptive","uniform_proxy"}:
            proxy_dir=generation_root/"proxy"; proxy_dir.mkdir(exist_ok=True)
            proxy_signature=object_hash(generation_context)
            for index,(ids,_) in enumerate(encoded):
                path=proxy_dir/f"prompt-{index:04d}.json"
                if path.exists():
                    row=read_json(path)
                    if row.get("signature") != proxy_signature or row.get("task_id") != task_ids[index]:
                        raise ValueError("allocation proxy cache signature mismatch")
                else:
                    deadline.check(); started=time.monotonic()
                    with torch.no_grad():
                        hidden=policy.hidden(torch.tensor([ids],device=policy.device),"teacher")[:,-1]
                        value=float(gini(decoder(policy.logits(hidden),decode)).sqrt())
                    _sync(policy)
                    row={"signature":proxy_signature,"task_id":task_ids[index],"value":value,
                         "seconds":time.monotonic()-started}
                    save(path,row)
                proxy_values.append(row["value"]); proxy_seconds+=row["seconds"]
            if arm["allocation"] == "adaptive": counts=allocate_prompts(proxy_values,64)
        elif arm["allocation"] != "uniform":
            raise ValueError("unknown generation allocation")
        allocation={"kind":arm["allocation"],"task_ids":task_ids,"counts":counts,"proxy":proxy_values,
                    "weight_correction":arm["weight_correction"],"proxy_seconds":proxy_seconds}
        save(generation_root/"allocation.json",allocation)
        templates=[]
        for prompt,(ids,truncated),count in zip(prompts,encoded,counts):
            for sample in range(count):
                templates.append({"task_id":prompt["task_id"],"sample_id":sample,"prompt_ids":ids,
                    "prompt_truncated":truncated,"completion_ids":[],
                    "seed":stable_seed(trajectory["seed"],"train",round_index,prompt["task_id"],sample),
                    "round":round_index,"parent_sha":parent_sha,
                    "loss_weight":64/(32*count) if arm["weight_correction"] else 1.})

        def generate_batch(batch,folder,adapter,extra=None):
            folder.mkdir(parents=True,exist_ok=True)
            context={**generation_context,"templates":batch,"adapter":adapter,
                     "adapter_sha256":_adapter_hash(policy,adapter),"extra":extra}
            sig=object_hash(context)
            metadata=folder/"generation.json"
            if metadata.exists() and read_json(metadata).get("signature") != sig:
                raise ValueError("generation resume signature mismatch")
            if not metadata.exists(): save(metadata,{"signature":sig,"context":context})
            rows=[]
            for item in batch:
                filename=f"record-{task_ids.index(item['task_id']):04d}-{item['sample_id']:03d}.json"
                path=folder/filename
                if path.exists():
                    row=read_json(path)
                    payload={k:v for k,v in row.items() if k != "record_digest"}
                    if (row.get("generation_signature") != sig or row.get("record_digest") != object_hash(payload) or
                            any(row.get(k) != v for k,v in item.items() if k != "completion_ids")):
                        raise ValueError("completed generation record hash or identity changed")
                else:
                    deadline.check(); started=time.monotonic()
                    try:
                        generated=policy.generate(item["prompt_ids"],decode,config["max_new_tokens"],item["seed"],
                                                  deadline,adapter=adapter,exploration=exploration)
                        _sync(policy)
                        if (not generated.get("completion_ids") or len(generated["completion_ids"]) > config["max_new_tokens"] or
                                generated.get("finish_reason") not in {"eos","length"}):
                            raise ValueError("generation produced an invalid completion")
                    except Exception as exc:
                        # No fabricated completion for interrupted token work.
                        # Its real time is nevertheless charged after a resume.
                        failed_seconds=time.monotonic()-started
                        ledger["failed_generation_attempts"]+=1
                        ledger["failed_generation_seconds"]+=failed_seconds
                        ledger["generation_failures"].append({"task_id":item["task_id"],
                            "sample_id":item["sample_id"],"seed":item["seed"],"round":round_index,
                            "generation_signature":sig,"error_type":type(exc).__name__,"seconds":failed_seconds})
                        raise
                    row={**item,**{key:generated[key] for key in ("completion_ids","text","finish_reason")},
                         "generation_signature":sig,"seconds":time.monotonic()-started}
                    row["record_digest"]=object_hash(row)
                    save(path,row)
                rows.append(row)
            save(folder/"raw.jsonl",rows,True)
            return rows

        if fixed:
            records=read_jsonl(fixed)
            _validate_records(records,task_ids,counts)
            save(generation_root/"raw.jsonl",records,True)
            generation_seconds=0.
        elif arm["rollout_refresh"] == "update":
            records=templates
            save(generation_root/"templates.jsonl",templates,True)
            generation_seconds=0.
        elif arm["rollout_refresh"] == "round":
            records=generate_batch(templates,generation_root,"teacher")
            generation_seconds=sum(row["seconds"] for row in records)+ledger["failed_generation_seconds"]
        else:
            raise ValueError("unknown rollout refresh schedule")

        def rollout(group_index,batch):
            started=time.monotonic(); saved_before=ledger["save_seconds"]
            try:
                return generate_batch(batch,generation_root/"updates"/f"group-{group_index:05d}","student",
                                      {"group":group_index,"rng_sha256":_rng_hash(policy)})
            except Exception:
                ledger["failed_rollout_seconds"]+=time.monotonic()-started
                raise
            finally:
                ledger["callback_save_seconds"]+=ledger["save_seconds"]-saved_before

        effective_config=output/"training-config.json"
        if effective_config.exists():
            stored=read_json(effective_config)
            if stored.get("job_signature") != signature:
                raise ValueError("frozen training configuration signature mismatch")
            train_config=stored["config"]
        else:
            train_config=deepcopy(config)
            total_clock=arm.get("train",{}).get("total_round_wall_seconds")
            if total_clock is not None:
                # Measured save latency reserves the remaining small artifacts;
                # final cost-match eligibility still uses the actual total.
                reserve=8*ledger["save_seconds"]/max(1,ledger["save_operations"])
                remaining=total_clock-generation_seconds-proxy_seconds-ledger["save_seconds"]-reserve
                if remaining <= 0:
                    raise ValueError("total round clock exhausted by generation, proxy, and save costs")
                train_config["training_wall_seconds"]=remaining
                train_config["clock_save_reserve_seconds"]=reserve
            save(effective_config,{"job_signature":signature,"config":train_config})
        # Cache hits must not change label-noise RNG. train_round restores its
        # finer optimizer/RNG checkpoint before invoking any remaining callback.
        training_seed=stable_seed(trajectory["seed"],"optimizer",round_index)
        random.seed(training_seed); torch.manual_seed(training_seed)
        checkpoint=train_round(policy,records,method,decode,train_config,output/"training",deadline,
            training_seed,parent_sha,rollout_callback=rollout if arm["rollout_refresh"] == "update" else None)
        training=read_json(output/"training/training.json")
        if arm["rollout_refresh"] == "update":
            records=[]
            for group in range(training["attempted_updates"]):
                path=generation_root/"updates"/f"group-{group:05d}"/"raw.jsonl"
                records.extend(read_jsonl(path))
            order={task:i for i,task in enumerate(task_ids)}
            records.sort(key=lambda row:(order[row["task_id"]],row["sample_id"]))
            save(generation_root/"raw.jsonl",records,True)
            generation_seconds=sum(row["seconds"] for row in records)
        _validate_records(records,task_ids,counts)
        rollout_seconds=training.get("costs",{}).get("rollout_seconds",0.)
        training_seconds=training["elapsed_seconds"]-rollout_seconds-ledger["failed_rollout_seconds"]
        if training_seconds < 0:
            raise ValueError("rollout cost exceeds measured training elapsed time")
        # Callback validation/cache bookkeeping is generation work too. Saves
        # are a separate measured component and are not counted a second time.
        if arm["rollout_refresh"] == "update":
            generation_seconds=max(generation_seconds+ledger["failed_generation_seconds"],
                rollout_seconds+ledger["failed_rollout_seconds"]-ledger["callback_save_seconds"])
        generation_costs={key:ledger[key] for key in ("failed_generation_attempts","failed_generation_seconds",
            "failed_rollout_seconds","callback_save_seconds","generation_failures")}
        save(generation_root/"costs.json",generation_costs)
        refs={"checkpoint_ref":_local_ref(checkpoint,output),
              "records_ref":_local_ref(generation_root/"raw.jsonl",output),
              "training_ref":_local_ref(output/"training/training.json",output),
              "events_ref":_local_ref(output/"training/events.jsonl",output),
              "allocation_ref":_local_ref(generation_root/"allocation.json",output),
              "generation_costs_ref":_local_ref(generation_root/"costs.json",output),
              "configuration_ref":_local_ref(effective_config,output)}
        costs={"generation_seconds":generation_seconds,"proxy_seconds":proxy_seconds,
            "training_seconds":training_seconds,"save_seconds":ledger["save_seconds"],
            "model_load_seconds":ledger["model_load_seconds"],
            "proxy_forward_passes":len(proxy_values),"generated_sequences":0 if fixed else len(records),
            "completion_tokens":sum(len(row["completion_ids"]) for row in records),
            "rollout_batches":training.get("costs",{}).get("rollout_batches",0),
            **{key:ledger[key] for key in ("failed_generation_attempts","failed_generation_seconds",
                                          "failed_rollout_seconds","callback_save_seconds")},
            "training_clock_includes_rollout_seconds":rollout_seconds,
            "optimizer_costs":training.get("costs",{})}
        costs["total_round_seconds"]=sum(costs[key] for key in ("generation_seconds","proxy_seconds","training_seconds","save_seconds"))
        total_clock=arm.get("train",{}).get("total_round_wall_seconds")
        training_clock=arm.get("train",{}).get("training_wall_seconds")
        clock_basis=("total_round_seconds" if total_clock is not None else
                     "training_seconds" if training_clock is not None else None)
        clock_target=total_clock if total_clock is not None else training_clock
        clock_deviation=costs[clock_basis]-clock_target if clock_basis is not None else None
        overshoot=max(0.,clock_deviation) if clock_deviation is not None else 0.
        receipt={"kind":"train","status":"complete","job_signature":signature,
            **{key:trajectory[key] for key in ("trajectory_id","arm_id","model","model_revision","seed")},
            "round":round_index,"rounds":trajectory["rounds"],"training_lineage":"clean-v2",
            "stage":job.get("stage",trajectory.get("stage","development")),
            "method":method,"arm":arm,"decode":asdict(decode),"parent_sha":parent_sha,
            "training_budget_digest":job.get("training_budget_digest"),
            "training_data_sha256":CLEAN_TRAIN_SHA256,"allocation":allocation,
            "training_seconds":training_seconds,
            "cumulative_training_seconds":parents["cumulative_training_seconds"]+training_seconds,
            "optimization_valid":training["optimization_valid"],"trained":training["trained"],
            "training_outcome":training["training_outcome"],"costs":costs,
            "clock_basis":clock_basis,"clock_target_seconds":clock_target,
            "clock_deviation_seconds":clock_deviation,"clock_match_tolerance_seconds":clock_tolerance,
            "clock_overshoot_seconds":overshoot,"cost_match_requested":clock_basis is not None,
            "cost_match_valid":abs(clock_deviation) <= clock_tolerance if clock_deviation is not None else None,
            "scientific_result_verified":False,**refs}
        save(completed,receipt)
        return receipt
    finally:
        ledger["active_seconds"]+=time.monotonic()-attempt_started
        atomic_json(state_path,ledger)
