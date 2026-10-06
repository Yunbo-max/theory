"""Inner native workloads. Launch using scripts/research.py, owned by the skill harness.

This module writes observations and provenance, never a research gate verdict.
Its pure helpers are also exercised with explicitly labelled engineering fixtures.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import time

from .io import (Deadline, DeadlineReached, atomic_json, digest, jsonl, object_hash,
                 read_json, read_jsonl, source_manifest, stable_seed)
from .methods import Decode


ROOT=Path(__file__).resolve().parents[1]


def local_ref(path, root):
    path=Path(path).resolve(); root=Path(root).resolve()
    return {'path':str(path.relative_to(root)),'sha256':digest(path)}


def stage_paths(job, workspace):
    """Map bound original-root paths to the harness's isolated input snapshot."""
    old=job.get('project_root')
    current=str(Path(workspace).resolve())
    def visit(value):
        if isinstance(value,dict): return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list): return [visit(v) for v in value]
        if isinstance(value,str) and old and (value==old or value.startswith(old.rstrip('/')+'/')):
            return current+value[len(old.rstrip('/')):]
        return value
    return visit(job)


def verify_model_manifest(path, model, revision):
    path=Path(path).resolve(); manifest=read_json(path)
    if manifest.get('model')!=model or manifest.get('revision')!=revision:
        raise ValueError('model manifest identity mismatch')
    snapshot=(path.parent/manifest['snapshot_path']).resolve()
    if not snapshot.is_relative_to(path.parent) or not snapshot.is_dir():
        raise ValueError('model snapshot must be physically inside the prepared artifact')
    refs=manifest.get('files',[])
    if not refs or len({ref['path'] for ref in refs})!=len(refs):
        raise ValueError('model file inventory missing or duplicated')
    expected=set()
    for ref in refs:
        target=(path.parent/ref['path']).resolve()
        if (not target.is_relative_to(snapshot) or target.is_symlink()
                or not target.is_file() or digest(target)!=ref['sha256']):
            raise ValueError('model snapshot file digest mismatch')
        expected.add(target)
    actual={p.resolve() for p in snapshot.rglob('*') if p.is_file() and '.cache' not in p.parts}
    if actual!=expected: raise ValueError('model snapshot file inventory changed')
    return snapshot


def load_policy(job, identity):
    import torch
    from .model import Policy
    from .runner import hardware
    hardware()
    seed=identity['seed']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=identity['model']; revision=identity.get('model_revision')
    if not revision:
        from .suite_design import MODELS
        revision=MODELS[model]
    if not job.get('model_manifest'):
        raise ValueError('pinned model_manifest is required for an executable workload')
    model_path=verify_model_manifest(job['model_manifest'],model,revision)
    return Policy.load(rank=job.get('config',{}).get('lora_rank',16),model=model,
                       revision=revision,model_path=model_path)


def prepare_model(job, data, output, deadline):
    from huggingface_hub import snapshot_download
    from .suite_design import MODELS
    model=job.get('model','Qwen/Qwen2.5-Coder-1.5B-Instruct')
    if model not in MODELS: raise ValueError('model has no pinned revision')
    revision=MODELS[model]
    if job.get('revision',revision)!=revision: raise ValueError('model revision changed')
    destination=Path(output)/'models'/revision
    deadline.check()
    started=time.monotonic()
    # local_dir writes real files, avoiding links to an unbound global HF cache.
    snapshot_download(model,revision=revision,local_dir=destination,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken'],max_workers=2)
    refs=[local_ref(p,output) for p in sorted(destination.rglob('*'))
          if p.is_file() and '.cache' not in p.parts]
    if not any(r['path'].endswith('.safetensors') for r in refs):
        raise ValueError('model weights absent from downloaded snapshot')
    manifest={'schema':'recursive-ssd-model-v1','model':model,'revision':revision,
        'snapshot_path':str(destination.relative_to(output)),'files':refs,
        'source':'https://huggingface.co/'+model+'/tree/'+revision,
        'trust_remote_code':False}
    atomic_json(Path(output)/'model-manifest.json',manifest)
    verify_model_manifest(Path(output)/'model-manifest.json',model,revision)
    return {'kind':'prepare-model','status':'complete','model_manifest_ref':local_ref(
        Path(output)/'model-manifest.json',output),'costs':{'download_seconds':time.monotonic()-started}}


def generate_evaluation(policy, prompts, unit, output, deadline, checkpoint_sha):
    """Resumable public-prompt generation, common random numbers across arms/rounds."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    allowed={'task_id','prompt','entry_point'}
    if (not prompts or any(not isinstance(p,dict) or set(p)-allowed or
            not isinstance(p.get('task_id'),str) or not isinstance(p.get('prompt'),str) for p in prompts)
            or len({p['task_id'] for p in prompts})!=len(prompts)):
        raise ValueError('invalid public prompt inventory')
    setting=unit['sampling']; count=unit['expected_samples']
    if type(count) is not int or count<1 or count!=setting['samples']:
        raise ValueError('evaluation sample inventory mismatch')
    decode=Decode(setting['temperature'],setting['top_k'],setting['top_p'])
    identity={'checkpoint_sha256':checkpoint_sha,'model':unit['model'],
        'benchmark':unit['benchmark'],'split':unit['split'],'seed':unit['seed'],
        'sampling':setting,'prompts_sha256':object_hash(prompts)}
    signature=object_hash(identity)
    state=output/'generation.json'
    if state.exists() and read_json(state).get('signature')!=signature:
        raise ValueError('evaluation generation identity changed on resume')
    atomic_json(state,{'identity':identity,'signature':signature})
    records=[]
    for index,prompt in enumerate(prompts):
        ids,truncated=policy.encode(prompt['prompt'],setting['max_prompt_tokens'])
        for sample in range(count):
            path=output/f'record-{index:04d}-{sample:02d}.json'
            seed=stable_seed(unit['seed'],'suite-eval-v3',unit['benchmark'],unit['split'],prompt['task_id'],sample)
            if path.exists():
                row=read_json(path)
                if (row.get('generation_signature')!=signature or row.get('task_id')!=prompt['task_id']
                        or row.get('sample_id')!=sample or row.get('seed')!=seed):
                    raise ValueError('evaluation cached sample identity changed')
            else:
                deadline.check(45); started=time.monotonic()
                result=policy.generate(ids,decode,setting['max_new_tokens'],seed,deadline,adapter='student')
                row={'task_id':prompt['task_id'],'sample_id':sample,'seed':seed,
                    'prompt_ids':ids,'prompt_truncated':truncated,'generation_signature':signature,
                    'checkpoint_sha256':checkpoint_sha,'seconds':time.monotonic()-started,**result}
                atomic_json(path,row)
            records.append(row)
    jsonl(output/'raw.jsonl',records)
    atomic_json(output/'inventory.json',{'count':len(records),'expected_count':len(prompts)*count,
        'task_ids':[p['task_id'] for p in prompts],'raw_sha256':digest(output/'raw.jsonl'),
        'generation_seconds':sum(r['seconds'] for r in records),
        'completion_tokens':sum(len(r['completion_ids']) for r in records),
        'length_capped':sum(r['finish_reason']=='length' for r in records),
        'prompt_truncated':sum(r['prompt_truncated'] for r in records)})
    return records


def training_cost_summary(receipts):
    """Full paid recursive path and clock eligibility, never optimizer time alone."""
    budgets={row.get('training_budget_digest') for row in receipts}
    if len(budgets)>1: raise ValueError('recursive training budget identity changed')
    return {'optimization_seconds':sum(row['training_seconds'] for row in receipts),
        'training_seconds':sum(row['costs']['total_round_seconds']+row['costs']['model_load_seconds']
                               for row in receipts),
        'optimization_valid':all(row.get('optimization_valid') is True for row in receipts),
        'cost_matching_valid':all(row.get('cost_match_valid') is True
            for row in receipts if row.get('cost_match_requested',False)),
        'training_budget_digest':next(iter(budgets)) if budgets else None}


def evaluate_job(job,data,output,deadline):
    from .benchmarks import benchmark_prompts_path, benchmark_tasks_path
    from .suite_score import score_benchmark
    from .suite_train import verify_parent_receipts
    from .suite_design import MODELS, SEEDS
    unit=job['unit']; output=Path(output); data=Path(data)
    identity={**unit,'model_revision':MODELS[unit['model']]}
    parents=job.get('training_receipts',[])
    if unit['round']:
        if not job.get('checkpoint'): raise ValueError('trained evaluation requires an exact checkpoint')
        # Reuse the same ordered lineage validator as the next recursive update.
        lineage=verify_parent_receipts(parents,{**identity,'training_lineage':'clean-v2'},unit['round'])
        last=read_json(parents[-1])
        expected=(Path(parents[-1]).parent/last['checkpoint_ref']['path']).resolve()
        if Path(job['checkpoint']).resolve()!=expected: raise ValueError('evaluation checkpoint differs from final training receipt')
        costs=training_cost_summary(lineage['receipts'])
    else:
        if parents or job.get('checkpoint') or unit['arm_id']!='base':
            raise ValueError('round zero is reserved for the original base model')
        costs=training_cost_summary([])
    policy=load_policy(job,identity)
    if job.get('checkpoint'): policy.load_checkpoint(job['checkpoint'])
    checkpoint_sha=digest(job['checkpoint']) if job.get('checkpoint') else MODELS[unit['model']]
    prompt_file=benchmark_prompts_path(data,unit['benchmark'],unit['split'])
    prompts=read_jsonl(prompt_file)
    rows=generate_evaluation(policy,prompts,unit,output/'generation',deadline,checkpoint_sha)
    # Release CUDA weights before the native CPU scorer starts subprocesses.
    del policy
    import torch
    torch.cuda.empty_cache()
    started=time.monotonic()
    task_file=benchmark_tasks_path(data,unit['benchmark'],unit['split'])
    metrics=score_benchmark(output/'generation/raw.jsonl',task_file,output/'scoring',unit['benchmark'],
        unit['expected_samples'],deadline,data_root=data,max_seconds=max(1,deadline.remaining()-30))
    elapsed=time.monotonic()-started
    shared={'decoder':unit['sampling'],'benchmark_inventory':{'task_ids':[p['task_id'] for p in prompts],
            'tasks_sha256':digest(task_file)},
        'evaluation_seeds':{'scheme':'suite-eval-v3','training_seeds':[17] if unit['seed']==17 else SEEDS},
        'model_base':{'model':unit['model'],'revision':MODELS[unit['model']]},
        'training_data':{'sha256':digest(data/'train_prompts.jsonl'),'lineage':'clean-v2'},
        'evaluation_protocol':{'benchmark':unit['benchmark'],'split':unit['split'],'sampling':unit['sampling']}}
    # Scorer identity must be common across arms; the per-cell raw result receipt
    # is a separate ref so pairing does not confuse provenance with equality.
    native_receipt=read_json(output/'scoring/native_receipt.json')
    shared['scorer']=native_receipt['identity']['evaluator']
    refs={}
    for key,value in shared.items():
        path=output/'identity'/f'{key}.json'; atomic_json(path,value)
        refs[key]={'path':str(path.resolve()),'sha256':digest(path)}
    refs['native_result']={'path':str((output/'scoring/official_results.json').resolve()),
                           'sha256':digest(output/'scoring/official_results.json')}
    observation={**unit,'status':'completed','per_task':metrics['per_task'],
        **costs,
        'training_cost_scope':'cumulative generation, proxy, optimization, save and model load through endpoint',
        'identity_refs':refs,
        'scientific_result_verified':False}
    atomic_json(output/'observation.json',observation)
    return {'kind':'evaluate','status':'complete','unit':unit,'observation':observation,
        'observation_ref':local_ref(output/'observation.json',output),
        'raw_ref':local_ref(output/'generation/raw.jsonl',output),
        'score_ref':local_ref(output/'scoring/metrics.json',output),
        'native_receipt_ref':local_ref(output/'scoring/native_receipt.json',output),
        'costs':{'generation_seconds':sum(r['seconds'] for r in rows),'scoring_seconds':elapsed}}


def preflight(job,data,output,deadline):
    """Actual one-GPU resource probe. It never supplies benchmark scores."""
    import torch
    from .runner import hardware, training_prompt
    from .suite_design import MODELS
    from .train import train_round
    from .methods import name
    config=read_json(ROOT/'configs/2080ti_8h.json'); config.update(job.get('config',{}))
    model=job.get('model','Qwen/Qwen2.5-Coder-1.5B-Instruct')
    identity={'model':model,'model_revision':MODELS[model],'seed':17}
    info=hardware(); policy=load_policy(job,identity)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    prompt=read_jsonl(Path(data)/'train_prompts.jsonl')[0]
    ids,truncated=policy.encode(training_prompt(prompt),config['max_prompt_tokens'])
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    generated=policy.generate(ids,Decode(),min(64,config['max_new_tokens']),17,deadline)
    generation_seconds=time.monotonic()-started
    if not generated['completion_ids']: raise ValueError('resource probe produced no tokens')
    # Explicit shape-only probe based on real generated tokens. Repeated tokens
    # are never used for evaluation, learning claims or baseline qualification.
    measured=dict(generated,prompt_ids=(ids*config['max_prompt_tokens'])[:config['max_prompt_tokens']],
        completion_ids=(generated['completion_ids']*config['max_new_tokens'])[:config['max_new_tokens']],
        task_id=prompt['task_id'],sample_id=0,scope='engineering shape probe')
    profiles={}
    methods=job.get('methods',['hard','full_soft'])
    if not methods or any(m not in {'hard','full_soft'} for m in methods):
        raise ValueError('resource preflight is restricted to native baseline controls')
    for method in methods:
        deadline.check(60); policy.reset()
        folder=output/method
        train_round(policy,[measured],name(method),Decode(),dict(config,epochs=1,grad_accum=1),
                    folder,deadline,17,MODELS[model])
        profiles[method]=read_json(folder/'training.json')
    torch.cuda.synchronize()
    allocated=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
    if reserved>=info['vram_bytes']*.93:
        raise RuntimeError('insufficient measured VRAM headroom for configured training shape')
    report={'schema':'recursive-ssd-host-profile-v1','status':'measured','hardware':info,
        'model':model,'model_revision':MODELS[model],'generation_seconds':generation_seconds,
        'generation_tokens':len(generated['completion_ids']),'profiles':profiles,
        'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
        'config':config,'source':source_manifest(ROOT),'baseline_qualification':False,
        'scope':'baseline training shape/resource probe; full LCB generation length and candidate costs remain unmeasured'}
    atomic_json(output/'preflight.json',report)
    return {'kind':'preflight','status':'complete','profile_ref':local_ref(output/'preflight.json',output)}


def execute(job,data,output,seconds):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if not 0<float(seconds)<=8*3600: raise ValueError('finite workload deadline required')
    deadline=Deadline(time.time()+float(seconds))
    resume=job.get('resume_from')
    if resume:
        source=Path(resume).resolve()
        if not source.is_dir() or source==output or output.is_relative_to(source):
            raise ValueError('invalid immutable prior-attempt resume directory')
        if any(output.iterdir()): raise ValueError('new attempt output must be empty before restore')
        shutil.copytree(source,output,dirs_exist_ok=True,ignore=shutil.ignore_patterns('receipt.json'))
    def stopped(signum,frame): raise DeadlineReached('outer execution owner requested stop')
    old=signal.signal(signal.SIGTERM,stopped)
    started=time.monotonic()
    try:
        kind=job['kind']
        if kind=='train':
            from .suite_train import train_job
            # Model path is verified here and consumed by the inner trainer.
            t=job['trajectory']; job=deepcopy(job)
            job['model_path']=str(verify_model_manifest(job['model_manifest'],t['model'],t['model_revision']))
            receipt=train_job(job,Path(data),output,deadline)
        elif kind=='evaluate': receipt=evaluate_job(job,Path(data),output,deadline)
        elif kind=='prepare-model': receipt=prepare_model(job,Path(data),output,deadline)
        elif kind=='preflight': receipt=preflight(job,Path(data),output,deadline)
        elif kind in {'derive-calibration','head_diagnostics'}:
            from .suite_calibration import derive_calibration
            if any('records_ref' in request for request in job.get('requests',[])):
                job=deepcopy(job)
                model_path=verify_model_manifest(job['model_manifest'],job['model'],job['model_revision'])
                for request in job['requests']:
                    if 'records_ref' in request:
                        request['model_path']=str(model_path)
                        request.setdefault('source_refs',[]).append({'path':str(Path(job['model_manifest']).resolve()),
                            'sha256':digest(job['model_manifest'])})
            receipt=derive_calibration(job,Path(data),output,deadline)
        else: raise ValueError('unknown native workload kind: '+str(kind))
        receipt.setdefault('kind',kind)
        receipt.update(workload_seconds=time.monotonic()-started,scientific_result_verified=False)
        atomic_json(output/'receipt.json',receipt)
        return receipt
    except Exception as exc:
        # Do not publish success or erase partial records after interruption.
        atomic_json(output/'failure.json',{'kind':job.get('kind'),'error_type':type(exc).__name__,
            'message':str(exc),'seconds':time.monotonic()-started,'status':'incomplete'})
        raise
    finally: signal.signal(signal.SIGTERM,old)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['execute'])
    parser.add_argument('--job',required=True); parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--seconds',required=True,type=float)
    args=parser.parse_args()
    # This prevents accidental use of the inner worker as an independent runner.
    if ROOT.name!='workspace' or not (ROOT.parent.parent/'plan.json').is_file():
        raise RuntimeError('launch workloads with scripts/research.py through research-autopilot')
    job=stage_paths(read_json(args.job),ROOT)
    result=execute(job,args.data,args.output,args.seconds)
    print(json.dumps({'kind':result['kind'],'status':result['status'],'scope':'workload only; no gate advanced'}))


if __name__=='__main__': main()
