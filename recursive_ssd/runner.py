"""Bounded interleaved recursive experiment, raw outputs and lineage retained."""
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
from importlib.resources import files
from jinja2 import Template
import shutil
import signal
import subprocess
import time
import traceback
import torch
from .data import MODEL_REV, verify_data
from .evaluation import evaluator_info, official_score, paired_interval
from .io import (Deadline, DeadlineReached, atomic_json, digest, event, jsonl,
                 object_hash, read_json, read_jsonl, source_manifest, stable_seed)
from .methods import Decode, allocate_prompts, decoder, gini, kl, name, round_decode
from .model import Policy
from .train import train_round


def validate_config(config):
    for key in ("rounds","train_prompts","samples_per_prompt","eval_tasks","eval_samples",
                "max_prompt_tokens","max_new_tokens","eval_max_new_tokens","grad_accum","epochs","vocab_chunk","lora_rank"):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError(f"{key} must be a positive integer")
    names=[name(a) for a in config["arms"]]
    if len(set(names))!=len(names):
        raise ValueError("duplicate arm aliases")
    if not {"hard","full_soft"}.issubset(names):
        raise ValueError("native hard and full_soft baselines are mandatory")
    if config["eval_tasks"]>=164:
        raise ValueError("this pilot reserves a separate confirmation split")
    if config["learning_rate"]<=0 or config["max_grad_norm"]<=0:
        raise ValueError("invalid optimizer settings")
    Decode(config["temperature"],config["top_k"],config["top_p"])
    Decode(config["eval_temperature"],config["eval_top_k"],config["eval_top_p"])
    params=config.get("method_parameters",{})
    for key in ("floor","epsilon","exploration"):
        if key in params and not 0<=params[key]<1:
            raise ValueError(f"invalid {key}")
    if not 0<params.get("alpha",.5)<1 or params.get("gate_lambda",.1)<=0:
        raise ValueError("invalid gate parameters")
    if not 0<=params.get("head_transfer",.5)<=1 or not 0<=params.get("diversity_retention",.9)<=1:
        raise ValueError("invalid probability/diversity coefficient")
    if any(params.get(k,0)<0 for k in ("noise_budget","ratio_bound","step_kl_budget")):
        raise ValueError("budgets and bounds must be nonnegative")
    return config


def hardware():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for real run; CPU tests are a separate command")
    if torch.cuda.device_count()!=1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")
    props=torch.cuda.get_device_properties(0)
    if props.total_memory<10*1024**3:
        raise RuntimeError("default profile requires approximately 11 GiB VRAM")
    if (props.major,props.minor)<(7,5):
        raise RuntimeError("this profile targets Turing compute capability 7.5 or newer")
    return {"name":props.name,"capability":[props.major,props.minor],"vram_bytes":props.total_memory,
            "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"dtype":"float16"}


def training_prompt(row):
    # Actual upstream prompt template; no starter solution/tests are supplied.
    template=files("recursive_ssd").joinpath("templates/self_distillation_prompt_function.j2").read_text()
    return Template(template).render(question=row["text"],starter_code="")


def evaluation_prompt(row):
    return "Complete the following Python function. Return the complete code including imports and the function definition.\n\n"+row["prompt"]


def generate_records(policy, prompts, decode, config, folder, deadline, round_index,
                     parent_sha, method="hard", evaluation=False):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    method=name(method)
    counts=[config["eval_samples"] if evaluation else config["samples_per_prompt"]]*len(prompts)
    max_new=config["eval_max_new_tokens"] if evaluation else config["max_new_tokens"]
    exploration=config["method_parameters"].get("exploration",.1) if method=="ancestral_exploration" and not evaluation else 0.
    encoded=[policy.encode(evaluation_prompt(p) if evaluation else training_prompt(p),config["max_prompt_tokens"]) for p in prompts]
    if method=="prompt_allocation" and not evaluation:
        proxy=[]
        with torch.no_grad():
            for prompt_ids,_ in encoded:
                deadline.check()
                ids=torch.tensor([prompt_ids],device=policy.device)
                mu=decoder(policy.logits(policy.hidden(ids,"teacher")[:,-1]),decode)
                proxy.append(float(gini(mu).sqrt()))
        counts=allocate_prompts(proxy,sum(counts))
        atomic_json(folder/"allocation.json",{"counts":counts,"proxy":proxy,"objective":"equal prompt"})
    signature=object_hash({"policy":parent_sha,"decode":asdict(decode),"counts":counts,
        "max_new":max_new,"encoded":encoded,"seed":config["seed"],"round":round_index,
        "exploration":exploration,"evaluation":evaluation})
    signature_file=folder/"generation.json"
    if signature_file.exists() and read_json(signature_file)["signature"]!=signature:
        raise ValueError("generation resume signature mismatch")
    atomic_json(signature_file,{"signature":signature,"parent_sha":parent_sha,"round":round_index,
        "decode":asdict(decode),"counts":counts,"evaluation":evaluation,"exploration":exploration})
    rows=[]
    for i,(prompt,(prompt_ids,truncated),count) in enumerate(zip(prompts,encoded,counts)):
        for sample in range(count):
            path=folder/f"record-{i:04d}-{sample:03d}.json"
            if path.exists():
                row=read_json(path)
                if row["generation_signature"]!=signature:
                    raise ValueError("record signature mismatch")
            else:
                deadline.check()
                seed=stable_seed(config["seed"],"eval" if evaluation else "train",round_index,prompt["task_id"],sample)
                started=time.monotonic()
                generated=policy.generate(prompt_ids,decode,max_new,seed,deadline,
                    adapter="student" if evaluation else "teacher",exploration=exploration)
                row={"task_id":prompt["task_id"],"sample_id":sample,"prompt_ids":prompt_ids,
                    "prompt_truncated":truncated,"seed":seed,"round":round_index,
                    "parent_sha":parent_sha,"generation_signature":signature,**generated,
                    "seconds":time.monotonic()-started,
                    "loss_weight":sum(counts)/(len(prompts)*count)}
                atomic_json(path,row)
                event(folder,"generation",task_id=prompt["task_id"],sample=sample,
                    tokens=len(row["completion_ids"]),finish_reason=row["finish_reason"],seconds=row["seconds"])
            rows.append(row)
    jsonl(folder/"raw.jsonl",rows)
    atomic_json(folder/"inventory.json",{"count":len(rows),"completion_tokens":sum(len(x["completion_ids"]) for x in rows),
        "length_capped":sum(x["finish_reason"]=="length" for x in rows),
        "prompt_truncated":sum(x["prompt_truncated"] for x in rows),
        "generated_seconds":sum(x["seconds"] for x in rows),"raw_sha256":digest(folder/"raw.jsonl")})
    return rows


def evaluate_policy(policy, config, artifacts, folder, deadline, checkpoint_sha):
    folder=Path(folder)
    if (folder/"metrics.json").exists():
        return read_json(folder/"metrics.json")
    decode=Decode(config["eval_temperature"],config["eval_top_k"],config["eval_top_p"])
    # Common evaluation random numbers across arms/rounds; no result-based selection.
    generate_records(policy,read_jsonl(artifacts/"eval_prompts-dev.jsonl"),decode,config,
        folder/"generation",deadline,0,checkpoint_sha,evaluation=True)
    metrics=official_score(folder/"generation/raw.jsonl",artifacts/"HumanEvalPlus-dev.jsonl",
        folder,config["eval_samples"],deadline)
    atomic_json(folder/"metrics.json",metrics)
    return metrics


@torch.no_grad()
def distribution_diagnostics(policy, records, config, deadline):
    """Same base-generated prefixes for all arms; descriptive, not correctness."""
    from .train import completion_inputs
    sums={"kl_initial_to_student":0.,"kl_student_to_initial":0.,"entropy":0.,
          "gini":0.,"floor_violation_mass":0.,"initial_decoder_outside_mass":0.}
    count=0
    decode=Decode(config["temperature"],config["top_k"],config["top_p"])
    c=config["method_parameters"]["floor"]
    for row in records:
        deadline.check()
        # Bound diagnostics to the first 64 generated prefixes per fixed record.
        row=dict(row,completion_ids=row["completion_ids"][:64])
        ids,labels,start=completion_inputs(policy,row)
        anchor=policy.hidden(ids,"anchor")[0,start:]
        current=policy.hidden(ids,"student")[0,start:]
        for i in range(0,len(labels),config["vocab_chunk"]):
            stop=i+config["vocab_chunk"]
            al=policy.logits(anchor[i:stop])
            a=al.float().softmax(-1)
            p=policy.logits(current[i:stop]).float().softmax(-1)
            values={"kl_initial_to_student":kl(a,p),"kl_student_to_initial":kl(p,a),
                "entropy":-(p*p.clamp_min(1e-30).log()).sum(-1),"gini":gini(p),
                "floor_violation_mass":(c*a-p).clamp_min(0).sum(-1),
                "initial_decoder_outside_mass":(p*(decoder(al,decode)==0)).sum(-1)}
            for key,value in values.items():
                sums[key]+=float(value.sum())
            count+=len(p)
    return {**{k:v/count for k,v in sums.items()},"prefixes":count,"floor":c,
        "scope":"fixed base-generated development prefixes; not global retention or correctness"}


def preflight(config, artifacts, output):
    validate_config(config)
    verify_data(artifacts,config)
    info=hardware()
    evaluator=evaluator_info()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config["seed"])
    policy=Policy.load(config["lora_rank"])
    prompt=read_jsonl(Path(artifacts)/"train_prompts.jsonl")[:1]
    canary=dict(config,samples_per_prompt=1,max_new_tokens=64,epochs=1,grad_accum=1)
    deadline=Deadline(time.time()+900)
    torch.cuda.reset_peak_memory_stats()
    records=generate_records(policy,prompt,Decode(),canary,output/"generation",deadline,0,MODEL_REV)
    # Actual max-length canary checks the configured training shape; repeat only
    # a real generated sequence for a MEMORY check, never report it as benchmark data.
    sample=dict(records[0])
    sample["prompt_ids"]=(sample["prompt_ids"]*config["max_prompt_tokens"])[:config["max_prompt_tokens"]]
    sample["completion_ids"]=(sample["completion_ids"]*config["max_new_tokens"])[:config["max_new_tokens"]]
    train_round(policy,[sample],"floor_projection",Decode(),canary,output/"training",deadline,0,MODEL_REV)
    peak=torch.cuda.max_memory_allocated()
    if peak>info["vram_bytes"]*.93:
        raise RuntimeError("insufficient VRAM headroom; reduce sequence length/rank in a new frozen configuration")
    report={"hardware":info,"evaluator":evaluator,"peak_allocated_bytes":peak,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "probe":"engineering resource canary on a real MBPP prompt; no accuracy measurement",
        "config_sha256":object_hash(config),
        "source_sha256":source_manifest(Path(__file__).resolve().parents[1])["sha256"],"status":"passed"}
    atomic_json(output/"preflight.json",report)
    return report


def worker(run, artifacts):
    run,artifacts=Path(run),Path(artifacts)
    state=read_json(run/"run.json")
    config=validate_config(state["config"])
    deadline=Deadline(state["end_epoch"])
    deadline.check(60)
    if source_manifest(Path(__file__).resolve().parents[1])["sha256"]!=state["source"]["sha256"]:
        raise ValueError("source changed since run was frozen; do not mix implementations on resume")
    verify_data(artifacts,config)
    hardware()
    if evaluator_info()!=state.get("evaluator"):
        raise ValueError("native evaluator identity changed; legacy runs need a separate versioned run")
    # Stop signals leave the last atomic optimizer-step checkpoint intact.
    def stop(_signal,_frame):
        raise DeadlineReached("supervisor requested bounded stop")
    signal.signal(signal.SIGTERM,stop)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    policy=Policy.load(config["lora_rank"])
    # Qualify the exact official native scorer on released canonical solutions.
    qualification=run/"qualification"
    qualification.mkdir(exist_ok=True)
    qproblems=qualification/"problems.jsonl"
    jsonl(qproblems,read_jsonl(artifacts/"HumanEvalPlus-dev.jsonl")[:2])
    official_score(None,qproblems,qualification,1,deadline,qualify=True)
    evaluate_policy(policy,config,artifacts,run/"base",deadline,MODEL_REV)
    diagnostic_records=read_jsonl(run/"base/generation/raw.jsonl")[:2]
    if not (run/"base/distribution.json").exists():
        atomic_json(run/"base/distribution.json",distribution_diagnostics(policy,diagnostic_records,config,deadline))
    train_prompts=read_jsonl(artifacts/"train_prompts.jsonl")
    completed=[]
    try:
        for r in range(1,config["rounds"]+1):
            for arm in config["arms"]:
                deadline.check(90)
                method=name(arm)
                folder=run/"arms"/method/f"round-{r:02d}"
                if (folder/"complete.json").exists():
                    receipt=read_json(folder/"complete.json")
                    prior=folder.parent/f"round-{r-1:02d}"/"adapter.pt"
                    expected_parent=digest(prior) if r>1 else MODEL_REV
                    if receipt["checkpoint_sha"]!=digest(folder/"adapter.pt") or receipt["parent_sha"]!=expected_parent:
                        raise ValueError("completed checkpoint lineage changed")
                    completed.append(f"{method}/{r}")
                    continue
                previous=folder.parent/f"round-{r-1:02d}"/"adapter.pt" if r>1 else None
                policy.reset()
                if previous is not None:
                    policy.load_checkpoint(previous)
                policy.copy("student","teacher")
                if r>2:
                    policy.load_checkpoint(folder.parent/f"round-{r-2:02d}"/"adapter.pt","lag")
                elif r==1:
                    policy.copy("teacher","lag")
                parent=digest(previous) if previous is not None else MODEL_REV
                decode=round_decode(method,Decode(config["temperature"],config["top_k"],config["top_p"]),config["rounds"])
                folder.mkdir(parents=True,exist_ok=True)
                if method=="fixed_data" and r>1:
                    records=read_jsonl(folder.parent/"round-01/generation/raw.jsonl")
                    jsonl(folder/"generation/raw.jsonl",records)
                    atomic_json(folder/"generation/reused.json",{"from":"round-01","control":"fixed-data repeated training"})
                else:
                    records=generate_records(policy,train_prompts,decode,config,folder/"generation",deadline,r,parent,method)
                final=folder/"adapter.pt"
                torch.cuda.reset_peak_memory_stats()
                if final.exists():
                    policy.load_checkpoint(final)
                else:
                    target_seed=stable_seed(config["seed"],"training-targets",r)
                    torch.manual_seed(target_seed)
                    torch.cuda.manual_seed_all(target_seed)
                    train_round(policy,records,method,decode,config,folder,deadline,
                                config["seed"]+r,parent)
                evaluate_policy(policy,config,artifacts,folder/"evaluation",deadline,digest(final))
                atomic_json(folder/"distribution.json",distribution_diagnostics(policy,diagnostic_records,config,deadline))
                atomic_json(folder/"complete.json",{"round":r,"method":method,"parent_sha":parent,
                    "checkpoint_sha":digest(final),"peak_vram_bytes":torch.cuda.max_memory_allocated(),
                    "train_completion_tokens":sum(len(x["completion_ids"]) for x in records),
                    "scientific_verdict":"INCONCLUSIVE; pilot only"})
                completed.append(f"{method}/{r}")
                report(run)
    except DeadlineReached as exc:
        event(run,"budget_stop",reason=str(exc),completed=completed)
    finally:
        report(run)


def report(run):
    run=Path(run)
    rows=[]
    base=read_json(run/"base/metrics.json") if (run/"base/metrics.json").exists() else None
    state=read_json(run/"run.json")
    for r in range(1,state["config"]["rounds"]+1):
        for arm in state["config"]["arms"]:
            method=name(arm)
            folder=run/"arms"/method/f"round-{r:02d}"
            row={"method":method,"round":r,"status":"pending"}
            if (folder/"adapter.pt").exists():
                row["status"]="trained; evaluation incomplete"
            elif (folder/"training-state.pt").exists():
                row["status"]="training incomplete"
            elif (folder/"generation").exists():
                row["status"]="generation incomplete"
            if (folder/"complete.json").exists():
                metrics=read_json(folder/"evaluation/metrics.json")
                row.update(status="complete",plus_pass1=metrics["plus_pass1"],base_pass1=metrics["base_pass1"])
                if base:
                    row["vs_base"]=paired_interval(metrics,base)
                for comparator in ("hard","full_soft","arithmetic_anchor"):
                    other=run/"arms"/comparator/f"round-{r:02d}"/"evaluation/metrics.json"
                    if other.exists() and comparator!=method:
                        row[f"vs_{comparator}"]=paired_interval(metrics,read_json(other))
                row["optimization"]=read_json(folder/"training.json")
                row["distribution"]=read_json(folder/"distribution.json")
            rows.append(row)
    result={"status":"INCONCLUSIVE", "reason":"development subset and one seed; no confirmation or complete multi-seed comparison",
        "base":{k:v for k,v in base.items() if k!="per_task"} if base else None,
        "inventory":rows,"end_epoch":state["end_epoch"],"source":state["source"],
        "budget_expired":time.time()>=state["end_epoch"]}
    atomic_json(run/"report.json",result)
    lines=["# Recursive SSD pilot", "", "Status: **INCONCLUSIVE**. This is an exploratory development subset, not a novelty or improvement certificate.","",
        "| Method | Round | Status | HumanEval+ pass@1 |", "|---|---|---|---|"]
    for row in rows:
        score=f"{row['plus_pass1']:.3f}" if "plus_pass1" in row else "—"
        lines.append(f"| {row['method']} | {row['round']} | {row['status']} | {score} |")
    lines += ["","Raw generations, evaluator outputs, optimizer events, paired task-bootstrap intervals and incomplete inventory remain in the run directory.",
        "No absent method is counted as a failed experiment. Read report.json for diagnostics and matched-baseline comparisons."]
    (run/"REPORT.md").write_text("\n".join(lines)+"\n")
    return result
