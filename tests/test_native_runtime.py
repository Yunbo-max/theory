"""Runtime/metadata fixtures only; these are not benchmark accuracy tests."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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


@pytest.mark.parametrize('ignore_sigterm',[False,True])
def test_native_timeout_stops_descendants(tmp_path,ignore_sigterm):
    run=getattr(ev,'run_native_process',None)
    assert callable(run), 'native bounded process runner is required'
    handler='signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_sigterm else ''
    grandchild='import signal,time,pathlib; '+handler+'pathlib.Path("child.ready").touch(); time.sleep(30)'
    code='\n'.join([
        'import json,os,pathlib,signal,subprocess,sys,time',
        handler,
        f'p=subprocess.Popen([sys.executable,"-I","-c",{grandchild!r}])',
        'pathlib.Path("pids.json").write_text(json.dumps({"parent":os.getpid(),"child":p.pid,"pgid":os.getpgrp()}))',
        'while not pathlib.Path("child.ready").exists(): time.sleep(.01)',
        'time.sleep(30)',
    ])
    libc=ctypes.CDLL(None,use_errno=True)
    before=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(before),0,0,0)==0
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable,'-I','-c',code],tmp_path,1)
    pids=json.loads((tmp_path/'pids.json').read_text())
    # Syscall PIDs can differ from /proc mount PIDs in a nested PID namespace.
    # kill(0) also sees zombies, so this requires termination *and* reaping.
    for pid in (pids['parent'],pids['child']):
        with pytest.raises(ProcessLookupError):
            os.kill(pid,0)
    assert pids['pgid']==os.getpgrp(), 'native descendants must inherit the harness worker group'
    receipt=json.loads((tmp_path/'execution.json').read_text())
    assert receipt['status']=='timeout'
    assert receipt['returncode']==(-9 if ignore_sigterm else -15)
    assert receipt['process_cleanup']['status']=='completed'
    after=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(after),0,0,0)==0
    assert after.value==before.value, 'restore process-wide subreaper ownership'


def test_native_exit_reaps_orphans_without_stopping_existing_children(tmp_path):
    # An already-running child belongs to the caller, not this evaluator call.
    sibling=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(30)'])
    try:
        code='\n'.join([
            'import pathlib,subprocess,sys,time',
            'p=subprocess.Popen([sys.executable,"-I","-c","import pathlib,time; pathlib.Path(\'child.ready\').touch(); time.sleep(30)"])',
            'pathlib.Path("child.pid").write_text(str(p.pid))',
            'while not pathlib.Path("child.ready").exists(): time.sleep(.01)',
        ])
        assert ev.run_native_process([sys.executable,'-I','-c',code],tmp_path,10)==0
        pid=int((tmp_path/'child.pid').read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid,0)
        assert sibling.poll() is None
        receipt=json.loads((tmp_path/'execution.json').read_text())
        assert receipt['status']=='completed'
        assert receipt['returncode']==0
        assert receipt['process_cleanup']['orphan_cleanup'] is True
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)
