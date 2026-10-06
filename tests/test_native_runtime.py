"""Runtime/metadata fixtures only; these are not benchmark accuracy tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from recursive_ssd import evaluation as ev
from recursive_ssd.io import Deadline, atomic_json, jsonl


def test_cached_score_without_provenance_is_rejected(tmp_path):
    # A cached result must not bypass checking its native input/code identity.
    problems=tmp_path/'problems.jsonl'
    raw=tmp_path/'raw.jsonl'
    jsonl(problems,[{'task_id':'metadata-fixture'}])
    jsonl(raw,[{'task_id':'metadata-fixture','sample_id':0,'text':'not executed'}])
    atomic_json(tmp_path/'samples_eval_results.json',{'eval':{'metadata-fixture':[
        {'base_status':'pass','plus_status':'pass'}]}})
    with pytest.raises(ValueError,match='unbound|provenance'):
        ev.official_score(raw,problems,tmp_path,1,Deadline(time.time()+60))


def test_inventory_rejects_empty_or_duplicate_problem_manifest():
    with pytest.raises(ValueError):
        ev.inventory([],[],1)
    with pytest.raises(ValueError):
        ev.inventory([{'task_id':'A','sample_id':0}],[{'task_id':'A'},{'task_id':'A'}],1)
    with pytest.raises(ValueError):
        ev.inventory([], [{'task_id':'A'}],0)


def test_native_child_does_not_inherit_credentials_or_gpu(tmp_path,monkeypatch):
    run=getattr(ev,'run_native_process',None)
    assert callable(run), 'native bounded process runner is required'
    monkeypatch.setenv('OPENAI_API_KEY','engineering-placeholder')
    monkeypatch.setenv('HF_TOKEN','engineering-placeholder')
    output=tmp_path/'probe.json'
    code='import json,os,pathlib; pathlib.Path("probe.json").write_text(json.dumps({k:os.getenv(k) for k in ["OPENAI_API_KEY","HF_TOKEN","CUDA_VISIBLE_DEVICES"]}))'
    run([sys.executable,'-I','-c',code],tmp_path,10)
    assert json.loads(output.read_text()) == {'OPENAI_API_KEY':None,'HF_TOKEN':None,'CUDA_VISIBLE_DEVICES':''}


def test_native_timeout_stops_descendants(tmp_path):
    run=getattr(ev,'run_native_process',None)
    assert callable(run), 'native bounded process runner is required'
    code='import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); pathlib.Path("child.pid").write_text(str(p.pid)); time.sleep(30)'
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable,'-I','-c',code],tmp_path,1)
    pid=int((tmp_path/'child.pid').read_text())
    if psutil.pid_exists(pid):
        assert psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
    assert json.loads((tmp_path/'execution.json').read_text())['status']=='timeout'
