import math
import time
import pytest
from recursive_ssd.evaluation import inventory, pass_at_k, summarize
from recursive_ssd.io import Deadline, DeadlineReached, atomic_json, digest, read_json
from recursive_ssd.runner import validate_config


def test_no_missing_task_score_inflation():
    problems=[{"task_id":"A"},{"task_id":"B"}]
    with pytest.raises(ValueError,match="incomplete"):
        inventory([{"task_id":"A","sample_id":0}],problems,1)
    with pytest.raises(ValueError,match="duplicate"):
        inventory([{"task_id":"A","sample_id":0}]*2,problems,1)
    assert inventory([{"task_id":"A","sample_id":0},{"task_id":"B","sample_id":0}],problems,1)=={"A":1,"B":1}


def test_native_estimator_combinatorial_identity():
    for n in range(1,10):
        for c in range(n+1):
            for k in range(1,n+1):
                expected=1-(math.comb(n-c,k)/math.comb(n,k) if n-c>=k else 0)
                assert abs(pass_at_k(n,c,k)-expected)<1e-12


def test_native_summary_requires_base_and_plus(tmp_path):
    f=tmp_path/"eval.json"
    atomic_json(f,{"eval":{"A":[{"base_status":"fail","plus_status":"pass"},
                                  {"base_status":"pass","plus_status":"pass"}]}})
    result=summarize(f,2)
    assert result["plus_pass1"]==.5
    assert result["plus_pass2"]==1.
    with pytest.raises(ValueError,match="missing"):
        summarize(f,3)


def test_deadline_and_atomic_persisted_state(tmp_path):
    path=tmp_path/"run.json"
    end=time.time()-1
    atomic_json(path,{"end_epoch":end})
    before=digest(path)
    with pytest.raises(DeadlineReached):
        Deadline(read_json(path)["end_epoch"]).check()
    assert digest(path)==before


def test_default_config_has_required_controls():
    config=read_json("configs/2080ti_8h.json")
    validate_config(config)
    config["arms"]=["M03"]
    with pytest.raises(ValueError,match="mandatory"):
        validate_config(config)


def test_actual_upstream_template_used_without_answers():
    from recursive_ssd.runner import training_prompt
    text=training_prompt({"task_id":"fixture","text":"Engineering test prompt."})
    assert "Question: Engineering test prompt." in text
    assert "{{" not in text
    assert "```python\n\n```" in text
