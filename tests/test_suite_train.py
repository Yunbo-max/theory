"""Inner-worker engineering tests with a categorical torch learner, no scorer."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import time

import pytest
import torch

from recursive_ssd import model
from recursive_ssd.benchmarks import CLEAN_TRAIN_SHA256, DESIGN_DATA
from recursive_ssd.io import Deadline, DeadlineReached, atomic_json, digest, read_json, read_jsonl
from recursive_ssd.methods import decoder
from recursive_ssd.suite_design import MODEL, MODELS, arm_registry


class CategoricalPolicy:
    """Real autograd/optimizer learner; replace only external model acquisition."""
    def __init__(self):
        self.device=torch.device("cpu")
        initial=torch.tensor([2., .5, -.2, -1., -2.])
        self.student=torch.nn.Parameter(initial.clone())
        self.teacher=initial.clone()
        self.lag=initial.clone()
        self.anchor=initial.clone()
        self.generations=[]

    def state(self, adapter="student"):
        return {"logits":getattr(self,adapter)}

    def set_state(self, state, adapter="student"):
        with torch.no_grad():
            getattr(self,adapter).copy_(state["logits"])

    def copy(self, source, destination):
        self.set_state(self.state(source),destination)

    def trainable(self):
        return [self.student]

    def hidden(self, ids, adapter="student", train=False):
        return getattr(self,adapter)[None,None,:].expand(ids.shape[0],ids.shape[1],-1)

    def logits(self, hidden):
        return hidden

    def encode(self, text, max_prompt):
        assert "Task" in text or "Write" in text or "write" in text
        return [0,1],False

    def generate(self,prompt_ids,config,max_new,seed,deadline,adapter="teacher",exploration=0.):
        deadline.check()
        logits=getattr(self,adapter).detach()
        p=decoder(logits,config)
        if exploration:
            p=(1-exploration)*p+exploration*decoder(self.anchor,config)
        token=int(torch.multinomial(p,1,generator=torch.Generator().manual_seed(seed)))
        self.generations.append({"adapter":adapter,"seed":seed,"temperature":config.temperature,
            "exploration":exploration,"student":self.student.detach().clone(),"teacher":self.teacher.clone()})
        return {"completion_ids":[token,1],"text":f"token {token}","finish_reason":"length"}

    def save_checkpoint(self,path,**fields):
        torch.save({"adapter":{"logits":self.student.detach().clone()},**fields},path)

    def load_checkpoint(self,path,adapter="student"):
        state=torch.load(path,map_location="cpu",weights_only=True)
        self.set_state(state["adapter"],adapter)
        return state


@pytest.fixture
def assets(tmp_path):
    folder=tmp_path/"data"
    folder.mkdir()
    shutil.copyfile(DESIGN_DATA/"train_prompts-clean-v2.jsonl",folder/"train_prompts.jsonl")
    atomic_json(folder/"benchmarks-manifest.json",{
        "version":"clean-v2-native-benchmarks-1","files":{"train_prompts.jsonl":CLEAN_TRAIN_SHA256},
        "training":{"lineage":"clean-v2","count":32,"fields":["task_id","text"],
                    "must_restart_from_base":True,"exact_mbpp_plus_overlap":[]}})
    return folder


@pytest.fixture
def policies(monkeypatch):
    loaded=[]
    def load(**kwargs):
        policy=CategoricalPolicy()
        loaded.append((policy,kwargs))
        return policy
    monkeypatch.setattr(model.Policy,"load",load)
    return loaded


def job(arm="hard",round_index=1):
    return {"trajectory":{"trajectory_id":f"{arm}-test-lineage","arm_id":arm,
        "arm":deepcopy(arm_registry()[arm]),"model":MODEL,"model_revision":MODELS[MODEL],
        "seed":17,"rounds":3,"training_lineage":"clean-v2"},
        "round":round_index,"parent_checkpoint":None,"parent_receipts":[],
        "lag_checkpoint":None,"fixed_records":None,"calibration":{},
        "config":{"max_new_tokens":2,"lora_rank":2,"learning_rate":.02,"vocab_chunk":2}}


def worker():
    return importlib.import_module("recursive_ssd.suite_train")


def run(job_spec,data,output):
    receipt=worker().train_job(job_spec,data,output,Deadline(time.time()+120))
    atomic_json(output/"receipt.json",receipt)
    return receipt


def test_base_and_dirty_training_assets_reject_before_loading_model(assets,policies,tmp_path):
    with pytest.raises(ValueError,match="base"):
        run(job("base"),assets,tmp_path/"base")
    manifest=read_json(assets/"benchmarks-manifest.json")
    manifest["training"]["lineage"]="old-data"
    atomic_json(assets/"benchmarks-manifest.json",manifest)
    with pytest.raises(ValueError,match="clean-v2"):
        run(job(),assets,tmp_path/"dirty")
    assert policies == []


def test_worker_trains_clean_32_by_2_and_returns_bound_actual_receipts(assets,policies,tmp_path):
    output=tmp_path/"round"
    receipt=run(job(),assets,output)
    rows=read_jsonl(output/receipt["records_ref"]["path"])
    assert len(rows)==64 and len({r["task_id"] for r in rows})==32
    assert all(r["completion_ids"] and r["loss_weight"]==1. for r in rows)
    assert receipt["training_lineage"]=="clean-v2" and receipt["status"]=="complete"
    assert receipt["cumulative_training_seconds"]==receipt["training_seconds"]>0
    assert len(policies[0][0].generations)==64
    assert {g["adapter"] for g in policies[0][0].generations}=={"teacher"}
    assert policies[0][1]["model"]==MODEL and policies[0][1]["revision"]==MODELS[MODEL]
    for field in ("checkpoint_ref","records_ref","training_ref","events_ref"):
        ref=receipt[field]
        assert not Path(ref["path"]).is_absolute()
        assert digest(output/ref["path"])==ref["sha256"]
    training=read_json(output/receipt["training_ref"]["path"])
    assert training["attempted_updates"]==16
    assert receipt["costs"]["generation_seconds"]>0


def test_gkd_update_generates_only_current_student_batches_and_keeps_round_teacher(assets,policies,tmp_path):
    output=tmp_path/"update"
    receipt=run(job("gkd_fkl_update"),assets,output)
    generated=policies[0][0].generations
    assert len(generated)==64
    assert {g["adapter"] for g in generated}=={"student"}
    assert not torch.equal(generated[0]["student"],generated[-1]["student"])
    assert all(torch.equal(generated[0]["teacher"],g["teacher"]) for g in generated)
    assert len(read_jsonl(output/receipt["records_ref"]["path"]))==64
    assert receipt["costs"]["rollout_batches"]==16


def test_round_parent_lag_and_cumulative_receipts_are_verified(assets,policies,tmp_path):
    first_job=job("M08")
    first=run(first_job,assets,tmp_path/"r1")
    second_job=deepcopy(first_job)
    second_job.update(round=2,parent_checkpoint=str((tmp_path/"r1"/first["checkpoint_ref"]["path"]).resolve()),
                      parent_receipts=[str((tmp_path/"r1/receipt.json").resolve())])
    second=run(second_job,assets,tmp_path/"r2")
    assert second["cumulative_training_seconds"]==first["training_seconds"]+second["training_seconds"]
    torch.testing.assert_close(policies[1][0].lag,policies[1][0].anchor)
    torch.testing.assert_close(policies[1][0].teacher,policies[0][0].student)
    verified=worker().verify_parent_receipts([str(tmp_path/"r1/receipt.json"),str(tmp_path/"r2/receipt.json")],
                                           second_job["trajectory"],2)
    assert verified["cumulative_training_seconds"]==second["cumulative_training_seconds"]
    (tmp_path/"r1"/first["training_ref"]["path"]).write_text("{}")
    with pytest.raises(ValueError,match="hash|changed"):
        worker().verify_parent_receipts([str(tmp_path/"r1/receipt.json")],first_job["trajectory"],1)


def test_fixed_data_reuses_r1_records_without_new_generation(assets,policies,tmp_path):
    first_spec=job("fixed_data")
    first=run(first_spec,assets,tmp_path/"r1")
    second_spec=deepcopy(first_spec)
    second_spec.update(round=2,parent_checkpoint=str((tmp_path/"r1"/first["checkpoint_ref"]["path"]).resolve()),
        parent_receipts=[str((tmp_path/"r1/receipt.json").resolve())],
        fixed_records=str((tmp_path/"r1"/first["records_ref"]["path"]).resolve()))
    second=run(second_spec,assets,tmp_path/"r2")
    assert policies[1][0].generations==[]
    assert second["records_ref"]["sha256"]==first["records_ref"]["sha256"]
    assert second["costs"]["generation_seconds"]==0


def test_proxy_controls_record_real_forward_cost_and_correction(assets,policies,tmp_path):
    adaptive_spec=job("M14")
    uniform_spec=job("uniform_counts_weighted")
    adaptive=run(adaptive_spec,assets,tmp_path/"adaptive")
    uniform=run(uniform_spec,assets,tmp_path/"uniform")
    for receipt in (adaptive,uniform):
        assert receipt["costs"]["proxy_forward_passes"]==32
        assert receipt["costs"]["proxy_seconds"]>0
        assert sum(receipt["allocation"]["counts"])==64
    assert uniform["allocation"]["counts"]==[2]*32


def test_clock_calibration_is_required_and_total_clock_cannot_hide_generation(assets,policies,tmp_path):
    spec=job("hard_equal_clock")
    with pytest.raises(ValueError,match="calibration"):
        run(spec,assets,tmp_path/"missing")
    source=tmp_path/"source.json"
    atomic_json(source,{"measured":True})
    spec["calibration"]={"clock_m15":{"value":1e-12,"source_refs":[{"path":str(source),"sha256":digest(source)}]}}
    with pytest.raises(ValueError,match="generation|remaining|clock"):
        run(spec,assets,tmp_path/"exhausted")
    assert not (tmp_path/"exhausted/training/adapter.pt").exists()


def test_clock_control_does_not_claim_match_when_epoch_bound_stops_early(assets,policies,tmp_path):
    spec=job("full_soft_equal_clock")
    spec["trajectory"]["arm"]["train"]["max_epochs"]=1
    spec["config"]["clock_match_tolerance_seconds"]=.1
    source=tmp_path/"clock-source.json"
    atomic_json(source,{"measured_seconds":60.})
    spec["calibration"]={"clock_m13":{"value":60.,"source_refs":[{"path":str(source),"sha256":digest(source)}]}}
    receipt=run(spec,assets,tmp_path/"clock")
    assert receipt["cost_match_valid"] is False
    assert receipt["clock_deviation_seconds"] < -1.
    assert receipt["clock_match_tolerance_seconds"] == .1


def test_generation_resume_preserves_completed_samples_and_common_random_seeds(assets,policies,tmp_path):
    class StopGeneration:
        calls=0
        def check(self,*args):
            self.calls+=1
            if self.calls>12:
                raise DeadlineReached()
    spec=job()
    partial=tmp_path/"partial"
    with pytest.raises(DeadlineReached):
        worker().train_job(spec,assets,partial,StopGeneration())
    generated_before=len(policies[0][0].generations)
    assert 0<generated_before<64
    failed_state=read_json(partial/"worker-state.json")
    assert failed_state["failed_generation_attempts"]==1
    assert failed_state["failed_generation_seconds"]>0
    resumed=run(spec,assets,partial)
    assert generated_before+len(policies[1][0].generations)==64
    control=run(job("full_soft"),assets,tmp_path/"control")
    first_rows=read_jsonl(partial/resumed["records_ref"]["path"])
    assert resumed["costs"]["generation_seconds"] >= sum(r["seconds"] for r in first_rows)+failed_state["failed_generation_seconds"]
    control_rows=read_jsonl(tmp_path/"control"/control["records_ref"]["path"])
    assert [r["seed"] for r in first_rows]==[r["seed"] for r in control_rows]
    before=(partial/resumed["checkpoint_ref"]["path"]).read_bytes()
    rerun=run(spec,assets,partial)
    assert (partial/rerun["checkpoint_ref"]["path"]).read_bytes()==before
    assert len(policies)==3  # completed worker returns verified receipt before model load
