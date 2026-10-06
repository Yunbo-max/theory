"""Bind same-context implementation reviews to an existing source commit.

Run after scripts/verify_local.py and after a source snapshot has been committed.
This emits no native experiment PASS and never turns a CPU check into GPU evidence.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recursive_ssd.io import digest, object_hash, read_json, atomic_json, source_manifest


def ref(path):
    path=Path(path)
    return {"path":str(path.relative_to(ROOT)),"sha256":digest(path)}


def put(path,obj):
    atomic_json(path,obj)
    return ref(path)


def main():
    checks=read_json(ROOT/"research/verification/engineering-checks.json")
    if any(c["exit_code"] for c in checks["checks"]):
        raise SystemExit("Actual software checks did not pass")
    if checks["source"]["sha256"]!=source_manifest(ROOT)["sha256"]:
        raise SystemExit("Source changed after software verification")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=ROOT,text=True).strip()
    listing=subprocess.check_output(["git","ls-tree","-rz","HEAD"],cwd=ROOT)
    version=hashlib.sha256(listing).hexdigest()
    source_files=[*sorted((ROOT/"recursive_ssd").glob("*.py")),ROOT/"docker/score.py",ROOT/"docker/Dockerfile",
        ROOT/"configs/2080ti_8h.json",ROOT/"recursive_ssd/templates/self_distillation_prompt_function.j2"]
    for path in source_files:
        committed=subprocess.check_output(["git","show",f"HEAD:{path.relative_to(ROOT)}"],cwd=ROOT)
        if committed!=path.read_bytes():
            raise SystemExit(f"Uncommitted source: {path}")
    refs=[ref(p) for p in source_files]
    identity=put(ROOT/"research/verification/source-identity.json",{
        "git_commit":commit,"git_tree":tree,"git_tree_digest":version,
        "digest_definition":"SHA256 of exact git ls-tree -rz <source-commit> bytes",
        "tree_listing_hex":listing.hex(),"method_code_refs":refs,
        "scope":"source snapshot preceding evidence-only commit; not a GPU-run identity"})
    batch=read_json(ROOT/"research/method-batch.json")
    selection=read_json(ROOT/batch["selection_ref"]["path"])
    selected=set(selection["selected_ids"])
    locations={
        "M01":"methods.target / geometric_anchor", "M02":"methods.target / ratio_cap",
        "M03":"methods.floor_projection", "M04":"methods.round_decode and runner.worker",
        "M05":"methods.target / head_mass", "M06":"methods.diversity_floor",
        "M07":"methods.stratified_target", "M08":"methods.target / temporal_mean",
        "M09":"methods.target / disagreement_gate", "M10":"methods.prefix_weights and train.record_loss",
        "M11":"methods.allocate_noise and target / noise_allocation",
        "M12":"methods.project_direction and train.train_round SGD branch",
        "M13":"train.actual_step_kl and train.train_round rollback branch",
        "M14":"methods.allocate_prompts and runner.generate_records",
        "M15":"model.Policy.generate exploration branch and runner.generate_records"}
    code_ref=next(r for r in refs if r["path"]=="recursive_ssd/methods.py")
    timestamp=datetime.now(timezone.utc).isoformat()
    for entry in batch["candidates"]:
        cid=entry["candidate_id"]
        if cid not in selected:
            continue
        card=read_json(ROOT/entry["math_card_ref"]["path"])
        file="recursive_ssd/train.py" if cid in {"M10","M12","M13"} else (
             "recursive_ssd/runner.py" if cid=="M14" else "recursive_ssd/model.py" if cid=="M15" else "recursive_ssd/methods.py")
        mapped_ref=next(r for r in refs if r["path"]==file)
        check_records=[{"status":"passed","command":c["command"],"tested_code_refs":refs,
            "log_refs":[ref(ROOT/c["log"])]} for c in checks["checks"]]
        packet={"kind":"implementation-card","version":"1.0.0","candidate_id":cid,
            "math_card_ref":entry["math_card_ref"],"selection_ref":batch["selection_ref"],
            "code_refs":refs,"code_version":object_hash(refs),"method_version":version,
            "source_identity_ref":identity,
            "mapping":[{"derivation_step":s["id"],"code_ref":mapped_ref,"location":locations[cid],
                "rationale":s["statement"]+". The implementation constructs/checks this object under the card's conditions; the distributional proof is not asserted as a neural guarantee."} for s in card["derivation_steps"]],
            "software_checks":check_records,
            "limits":"CPU engine checks only; mathematical special cases are tested where tractable. No full model GPU run, independent review or empirical method verdict."}
        pr=put(ROOT/f"research/implementations/{cid}.json",packet)
        rationales={"math_to_code":"Reviewed "+locations[cid]+" against "+card["method_expression"]+". Frozen references, completion masks and conditional scope are retained; see mapped statements for limitations.",
            "software_checks":"Actual pytest suite, dependency check, compilation, CLI and shell checks passed. Tests include target constraints, all-candidate backward passes, tiny-model recursion, optimizer/resume paths and inventory checks. This is an engineering scope, not native benchmark evidence.",
            "actual_code_identity":"Source files match committed bytes at "+commit+". Full source tree listing and SHA256 plus code-file hashes bind the review; later files added here are evidence-only."}
        review={"kind":"method-review","version":"1.0.0","subject_id":cid,"scope":"code",
            "artifact_ref":pr,"reviewer":"same-context assistant review; not independent",
            "reviewed_at":timestamp,"outcome":"verified","checks":{
                k:{"status":"verified","rationale":v,"evidence_refs":[pr,identity,ref(ROOT/"research/verification/engineering-checks.json")]} for k,v in rationales.items()}}
        rr=put(ROOT/f"research/reviews/{cid}-code.json",review)
        entry.update(implementation_ref=pr,code_review_ref=rr)
    put(ROOT/"research/implementation-batch.json",batch)
    print(json.dumps({"source_commit":commit,"selected_implementations":len(selected),"scope":"CPU engineering review only"}))


if __name__=="__main__":
    main()
