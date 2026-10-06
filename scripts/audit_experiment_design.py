"""Check the retained design/data bindings; does not certify scientific gates."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/"research/design-v2"


def read(path):
    return json.loads(path.read_text())


def check_ref(reference):
    path=(ROOT/reference["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("reference escapes project")
    assert hashlib.sha256(path.read_bytes()).hexdigest()==reference["sha256"], str(path)


def main():
    selection=read(ROOT/"research/selection.json")
    packet=read(DESIGN/"candidate-drafts.json")
    methods=packet["methods"]
    assert [m["candidate_id"] for m in methods]==selection["selected_ids"]
    assert len({m["candidate_id"] for m in methods})==15
    blockers={x["id"] for x in read(DESIGN/"implementation-gaps.json")["blockers"]}
    for method in methods:
        for key in ("math_card_ref","selection_ref","historical_implementation_ref",
                    "common_design_ref","benchmark_contracts_ref"):
            check_ref(method[key])
        assert method["baselines"] and method["decisive_controls"]
        assert method["prediction"] and method["falsifier"] and method["semantic_review"]
        assert set(method["blocker_ids"])<=blockers
        assert method["development_seed"] not in method["confirmation_seeds_proposed"]
    he=read(DESIGN/"data/humaneval-manifest.json")
    assert len(set(he["dev_ids"]))==32 and len(set(he["confirm_ids"]))==132
    assert not set(he["dev_ids"])&set(he["confirm_ids"])
    mbpp=read(DESIGN/"data/mbpp-plus-audit.json")
    assert len(set(mbpp["all_task_ids"]))==mbpp["count"]==378
    prompts_path=DESIGN/"data/train_prompts-clean-v2.jsonl"
    prompts=[json.loads(line) for line in prompts_path.read_text().splitlines()]
    assert len(prompts)==32 and all(set(p)=={"task_id","text"} for p in prompts)
    ids={p["task_id"] for p in prompts}
    assert len(ids)==32 and not ids&{x.split("/")[-1] for x in mbpp["all_task_ids"]}
    train=read(DESIGN/"data/clean-training-manifest.json")
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest()==train["selected_sha256"]
    assert [p["task_id"] for p in prompts]==train["selected_ids"]
    inventory=read(DESIGN/"run-inventory.json")
    assert {b["candidate_id"] for b in inventory["method_bundles"]}==set(selection["selected_ids"])
    assert not inventory["executable"]
    report={"status":"passed","scope":"design coverage, source-reference identity and exact task-ID data separation only",
            "design_drafts":len(methods),"human_eval_development":32,"human_eval_confirmation":132,
            "mbpp_plus_tasks_observed":378,"old_train_prompt_overlap":len(mbpp["current_train_overlap"]),
            "new_train_prompt_overlap":0,"formal_design_verified":0,"gpu_results_verified":0}
    (DESIGN/"audit-result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
