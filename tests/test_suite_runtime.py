"""Workload orchestration fixtures, never scientific benchmark evidence."""
import importlib
from pathlib import Path
import time

import pytest

from recursive_ssd.io import Deadline, DeadlineReached, atomic_json, digest


def api():
    return importlib.import_module('recursive_ssd.suite')


class TokenPolicy:
    """Deterministic engineering stand-in for expensive GPU generation only."""
    def __init__(self, fail_at=None):
        self.calls=[]
        self.fail_at=fail_at

    def encode(self, text, maximum):
        assert 'SECRET' not in text
        return [1,2], False

    def generate(self, ids, decode, maximum, seed, deadline, **kwargs):
        self.calls.append(seed)
        if self.fail_at == len(self.calls):
            raise DeadlineReached('engineering interruption')
        return {'completion_ids':[3,4], 'text':'engineering fixture', 'finish_reason':'eos'}


def unit(arm='M03'):
    return {'cell_id':arm+'-cell','arm_id':arm,'model':'fixture-model',
            'benchmark':'humaneval','split':'dev','seed':17,'round':3,
            'expected_samples':2, 'sampling':{'samples':2,'temperature':.8,
            'top_k':0,'top_p':.95,'max_prompt_tokens':1024,'max_new_tokens':10}}


def test_eval_resume_preserves_samples_and_common_random_seeds(tmp_path):
    first=TokenPolicy(fail_at=2)
    prompts=[{'task_id':'fixture/0','prompt':'function prompt'}]
    with pytest.raises(DeadlineReached):
        api().generate_evaluation(first,prompts,unit(),tmp_path/'a',Deadline(time.time()+120),'checkpoint-sha')
    second=TokenPolicy()
    rows=api().generate_evaluation(second,prompts,unit(),tmp_path/'a',Deadline(time.time()+120),'checkpoint-sha')
    assert len(second.calls)==1 and len(rows)==2
    assert [r['sample_id'] for r in rows]==[0,1]
    assert rows[0]['seed']==first.calls[0]
    other=api().generate_evaluation(TokenPolicy(),prompts,unit('hard'),tmp_path/'b',Deadline(time.time()+120),'other-sha')
    assert [r['seed'] for r in rows]==[r['seed'] for r in other]


def test_eval_cache_is_bound_to_checkpoint_decoder_and_prompt(tmp_path):
    prompts=[{'task_id':'fixture/0','prompt':'function prompt'}]
    api().generate_evaluation(TokenPolicy(),prompts,unit(),tmp_path,Deadline(time.time()+120),'original')
    with pytest.raises(ValueError,match='identity'):
        api().generate_evaluation(TokenPolicy(),prompts,unit(),tmp_path,Deadline(time.time()+120),'changed')
    changed=unit(); changed['sampling']['temperature']=1.
    with pytest.raises(ValueError,match='identity'):
        api().generate_evaluation(TokenPolicy(),prompts,changed,tmp_path,Deadline(time.time()+120),'original')


def test_generation_rejects_private_fields_and_duplicate_inventory(tmp_path):
    for rows in ([{'task_id':'t','prompt':'p','private_test_cases':'SECRET'}],
                 [{'task_id':'t','prompt':'p'},{'task_id':'t','prompt':'p'}]):
        with pytest.raises(ValueError,match='prompt|inventory'):
            api().generate_evaluation(TokenPolicy(),rows,unit(),tmp_path,Deadline(time.time()+120),'sha')


def test_model_manifest_binds_physical_snapshot_files(tmp_path):
    model=tmp_path/'model'; model.mkdir()
    weights=model/'model.safetensors'; weights.write_bytes(b'engineering fixture')
    cfg=model/'config.json'; cfg.write_text('{}')
    manifest=tmp_path/'model-manifest.json'
    atomic_json(manifest,{'schema':'recursive-ssd-model-v1','model':'fixture','revision':'fixed',
        'snapshot_path':'model','files':[{'path':'model/model.safetensors','sha256':digest(weights)},
                                       {'path':'model/config.json','sha256':digest(cfg)}]})
    assert api().verify_model_manifest(manifest,'fixture','fixed')==model
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='model'):
        api().verify_model_manifest(manifest,'fixture','fixed')


def test_staged_job_paths_are_remapped_without_reading_original_tree(tmp_path):
    workspace=tmp_path/'workspace'; workspace.mkdir()
    target=workspace/'inputs/model.pt'; target.parent.mkdir(); target.write_bytes(b'x')
    job={'project_root':'/old/project','parent_checkpoint':'/old/project/inputs/model.pt',
         'nested':{'refs':[{'path':'inputs/model.pt','sha256':digest(target)}]}}
    actual=api().stage_paths(job,workspace)
    assert actual['parent_checkpoint']==str(target)
    assert actual['nested']['refs'][0]['path']=='inputs/model.pt'
    assert actual['project_root']==str(workspace)


def test_per_round_calibration_is_preserved_until_each_round_is_materialized():
    from recursive_ssd.suite_design import make_suite,resolve_arm
    calibration={'clock_m13':{'rounds':{str(r):{'value':r*2.,'source_refs':[{'path':f'r{r}.json','sha256':'a'*64}]} for r in [1,2,3]}}}
    # An incomplete measured set may be compiled but cannot be dispatched.
    suite=make_suite('development',calibration=calibration)
    trajectory=next(x for x in suite['trajectories'] if x['arm_id']=='full_soft_equal_clock')
    assert resolve_arm(trajectory['arm'],calibration,round_index=2)['train']['training_wall_seconds']==4.
    assert suite['scientific_dispatch_ready'] is False


def test_legacy_entrypoint_cannot_bypass_the_authorized_harness():
    import subprocess, sys
    result=subprocess.run([sys.executable,'-m','recursive_ssd.cli','report'],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'scripts/research.py' in result.stderr


def test_clock_invalid_parent_is_ineligible_and_full_training_cost_is_charged():
    receipts=[{'training_seconds':3.,'training_budget_digest':'same',
        'optimization_valid':True,'cost_match_requested':False,'cost_match_valid':None,
        'costs':{'total_round_seconds':5.,'model_load_seconds':1.}},
        {'training_seconds':4.,'training_budget_digest':'same','optimization_valid':True,
         'cost_match_requested':True,'cost_match_valid':False,
         'costs':{'total_round_seconds':7.,'model_load_seconds':2.}}]
    result=api().training_cost_summary(receipts)
    assert result['training_seconds']==15. and result['optimization_seconds']==7.
    assert result['optimization_valid'] is True and result['cost_matching_valid'] is False
    assert result['training_budget_digest']=='same'
