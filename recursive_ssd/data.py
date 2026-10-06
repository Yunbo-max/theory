"""Pinned public data. Training export contains only identifiers and prompts."""
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
from .io import atomic_json, digest, jsonl, read_json, read_jsonl, stable_seed

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REV="2e1fd397ee46e1388853d2af2c993145b0f1098a"
MBPP="google-research-datasets/mbpp"
MBPP_REV="4bb6404fdc6cacfda99d4ac4205087b89d32030c"
EVAL_URL="https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
EVAL_SHA256="42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"


def prepare(root, config, download_model=True):
    from datasets import load_dataset
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset(MBPP,"full",split="train",revision=MBPP_REV)
    # Explicit field allowlist: never export code, test_list or assertions.
    prompts=[{"task_id":str(row["task_id"]),"text":row["text"]} for row in dataset]
    prompts.sort(key=lambda x:stable_seed("train-subset-v1",x["task_id"]))
    prompts=prompts[:config["train_prompts"]]
    jsonl(root/"train_prompts.jsonl",prompts)
    request=urllib.request.Request(EVAL_URL,headers={"User-Agent":"recursive-ssd-research"})
    with urllib.request.urlopen(request,timeout=120) as response:
        compressed=response.read()
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest()!=EVAL_SHA256:
        raise ValueError("official HumanEval+ release bytes changed from the inspected snapshot")
    (root/"HumanEvalPlus-full.jsonl").write_bytes(raw)
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows)!=164 or len({r["task_id"] for r in rows})!=164:
        raise ValueError("unexpected HumanEval+ inventory")
    rows.sort(key=lambda r:stable_seed("eval-split-v1",r["task_id"]))
    n=config["eval_tasks"]
    if not 1<=n<164:
        raise ValueError("pilot requires a nonempty disjoint confirmation split")
    jsonl(root/"HumanEvalPlus-dev.jsonl",rows[:n])
    jsonl(root/"HumanEvalPlus-confirm.jsonl",rows[n:])
    # Prompt-only mirror is the only evaluation file that GPU generation reads.
    for split,problems in (("dev",rows[:n]),("confirm",rows[n:])):
        jsonl(root/f"eval_prompts-{split}.jsonl",[
            {"task_id":p["task_id"],"prompt":p["prompt"],"entry_point":p["entry_point"]}
            for p in problems])
    model_path=None
    if download_model:
        from huggingface_hub import snapshot_download
        model_path=snapshot_download(MODEL,revision=MODEL_REV,
            allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"])
    files={p.name:digest(p) for p in root.glob("*.jsonl")}
    manifest={"model":MODEL,"model_revision":MODEL_REV,"model_path":model_path,
        "dataset":MBPP,"dataset_revision":MBPP_REV,"train_split":"full/train",
        "train_fields":["task_id","text"],"eval_url":EVAL_URL,"files":files,
        "dev_ids":[r["task_id"] for r in rows[:n]],"confirm_ids":[r["task_id"] for r in rows[n:]],
        "train_count":len(prompts),"config_data_fields":{"train_prompts":config["train_prompts"],"eval_tasks":n}}
    atomic_json(root/"manifest.json",manifest)
    return manifest


def verify_data(root, config):
    root=Path(root)
    manifest=read_json(root/"manifest.json")
    if manifest["config_data_fields"]!={k:config[k] for k in ("train_prompts","eval_tasks")}:
        raise ValueError("prepared data differs from run config; prepare again in another artifact directory")
    for name,sha in manifest["files"].items():
        if digest(root/name)!=sha:
            raise ValueError(f"data hash mismatch: {name}")
    train=read_jsonl(root/"train_prompts.jsonl")
    if any(set(row)!={"task_id","text"} for row in train):
        raise ValueError("training prompt field allowlist violated")
    return manifest
