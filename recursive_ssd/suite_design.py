"""Complete comparison catalog. Compilation is not scientific admission."""
from copy import deepcopy
import math
from pathlib import Path
from .data import MODEL, MODEL_REV
from .io import object_hash, read_json

ROOT=Path(__file__).resolve().parents[1]
ORDER=['M03','M04','M05','M01','M07','M02','M06','M08','M09','M10','M11','M12','M13','M14','M15']
SEEDS=[23,47,71,101,131]
MODELS={MODEL:MODEL_REV,'Qwen/Qwen2.5-Coder-0.5B-Instruct':'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'}
GKD=[f'gkd_{loss}_{refresh}' for loss in ('fkl','rkl','jsd') for refresh in ('cached','update')]
BENCHMARK_SETTINGS={
    'humaneval':{'samples':10,'temperature':.8,'top_k':0,'top_p':.95,'max_prompt_tokens':1024,'max_new_tokens':384,'coverage':'pass@10'},
    'mbpp':{'samples':10,'temperature':.8,'top_k':0,'top_p':.95,'max_prompt_tokens':1024,'max_new_tokens':384,'coverage':'pass@10'},
    'livecodebench':{'samples':10,'temperature':.2,'top_k':0,'top_p':.95,'max_prompt_tokens':4096,'max_new_tokens':2000,'coverage':'pass@5'},
}


def arm_registry():
    registry={}
    def add(identifier,method=None,**settings):
        item={'arm_id':identifier,'method':method or identifier,'parameters':{},'train':{},
              'decode':{'temperature':1.5,'top_k':20,'top_p':.8},'rollout_refresh':'round',
              'temperature_schedule':'fixed','generation_exploration':0.,'allocation':'uniform',
              'weight_correction':True,'required_calibration':[]}
        item.update(settings)
        registry[identifier]=item
    add('base',method='base')
    for candidate in ORDER: add(candidate)
    for control in ('hard','full_soft','arithmetic_anchor','lower_lr','fixed_data','fresh_alpha',
                    'soft_current_mix','head_shape_only','mass_transfer_only','head_identity',
                    'arithmetic_same_smoothing','geometric_weight_one','ratio_bound_unlimited',
                    'head_exact_tail_iid','fresh_hard','temporal_current_twice','lag_only',
                    'constant_mean_gate','constant_mean_weight','prefix_weight_one',
                    'fresh_alpha_budget_matched'):
        add(control)
    add('full_soft_lr_half','full_soft',train={'learning_rate_multiplier':.5})
    add('floor_zero','M03',parameters={'floor':0.})
    add('arithmetic_anchor_tuned','arithmetic_anchor',required_calibration=['arithmetic_floor'])
    registry['M04']['temperature_schedule']='root'
    for identifier,schedule,method in (
        ('temperature_fixed_untruncated','fixed','full_soft'),
        ('temperature_budget_untruncated','root','full_soft'),
        ('temperature_once_then_one','once','full_soft'),
        ('hard_untruncated','fixed','hard')):
        add(identifier,method,decode={'temperature':1.5,'top_k':0,'top_p':1.},temperature_schedule=schedule)
    add('head_retention_zero','M06',parameters={'diversity_retention':0.})
    add('temperature_matched_head',required_calibration=['head_temperature'])
    for identifier,method in (('full_soft_sgd','full_soft_sgd'),('arithmetic_anchor_sgd','arithmetic_anchor_sgd'),
                              ('sgd_norm_matched','sgd_norm_matched'),('projection_disabled','full_soft_sgd')):
        add(identifier,method,train={'optimizer':'sgd'})
    registry['M12']['train']['optimizer']='sgd'
    add('backtrack_disabled','full_soft')
    add('full_soft_lr_matched','full_soft',required_calibration=['m13_step_scale'])
    add('full_soft_equal_clock','full_soft',train={'max_epochs':1000},required_calibration=['clock_m13'])
    add('uniform_counts_weighted','hard',allocation='uniform_proxy')
    add('allocation_without_weight_correction','M14',allocation='adaptive',weight_correction=False)
    registry['M14']['allocation']='adaptive'
    registry['M15']['generation_exploration']=.1
    add('ancestral_mix_zero','hard',generation_exploration=0.)
    add('ancestral_soft_anchor','arithmetic_anchor',generation_exploration=.1)
    add('hard_equal_clock','hard',train={'max_epochs':1000},required_calibration=['clock_m15'])
    for loss in ('fkl','rkl','jsd'):
        for refresh in ('cached','update'):
            add(f'gkd_{loss}_{refresh}',f'gkd_{loss}',parameters={'epsilon':.02},
                rollout_refresh='update' if refresh=='update' else 'round')
    for family,key,values in (('M03','floor',[.03,.1,.3]),('arithmetic_anchor','floor',[.03,.1,.3]),
                              ('M01','alpha',[.25,.5,.75]),('arithmetic_same_smoothing','alpha',[.25,.5,.75])):
        for value in values:
            identifier=f'{family}__{key}_{str(value).replace(".","p")}'
            add(identifier,family,parameters={key:value})
    return registry


def resolve_arm(arm,calibration,round_index=None):
    result=deepcopy(arm)
    mapping={'head_temperature':('parameters','temperature_beta'),
             'arithmetic_floor':('parameters','floor'),
             'm13_step_scale':('train','learning_rate_multiplier'),
             'clock_m13':('train','training_wall_seconds'),
             'clock_m15':('train','total_round_wall_seconds')}
    result['calibration_refs']=[]
    for key in result['required_calibration']:
        record=calibration.get(key)
        if isinstance(record,dict) and 'rounds' in record:
            record=record['rounds'].get(str(round_index))
        if not isinstance(record,dict) or not record.get('source_refs'):
            raise ValueError(f'measured calibration required: {key}')
        value=record.get('value')
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError(f'invalid calibration value: {key}')
        if (key.startswith('clock_') and value<=0 or key=='m13_step_scale' and not 0<value<=1
                or key=='head_temperature' and not 0<=value<=1):
            raise ValueError(f'out-of-range calibration: {key}')
        if key=='arithmetic_floor' and not 0<=value<1:
            raise ValueError('invalid calibrated arithmetic floor')
        group,field=mapping[key]
        result[group][field]=value
        result['calibration_refs']+=record['source_refs']
    return result


def comparison_bundles():
    specs=read_json(ROOT/'research/design-v2/method-specs.json')
    result={}
    extras={'M04':['hard_untruncated'],'M05':['head_identity'],'M13':['full_soft_equal_clock']}
    for values in specs['methods']:
        row=dict(zip(specs['columns'],values))
        candidate=row['candidate_id']
        arms=list(dict.fromkeys(['base','hard','full_soft','fixed_data',candidate,*row['baselines'],
                                 *row['decisive_controls'],*extras.get(candidate,[]),*GKD]))
        # Conditions with changed decoder are compared within that condition.
        comparisons=[{'candidate_arm':candidate,'control_arm':a,
                      'control_role':'initial_model' if a=='base' else 'baseline'}
                     for a in arms if a!=candidate and not (candidate=='M04' and 'untruncated' in a)
                     and not (candidate=='M04' and a=='temperature_once_then_one')]
        if candidate=='M04':
            comparisons += [{'candidate_arm':'temperature_budget_untruncated','control_arm':a,'control_role':'baseline'}
                            for a in ['temperature_fixed_untruncated','temperature_once_then_one','hard_untruncated']]
        if candidate=='M05':
            comparisons += [{'factorial_arms':['head_identity','head_shape_only','mass_transfer_only','M05']}]
        if candidate=='M15':
            comparisons += [{'factorial_arms':['hard','arithmetic_anchor','M15','ancestral_soft_anchor']}]
        result[candidate]={**row,'arms':arms,'comparisons':comparisons}
    return result


def _digest_suite(value):
    return object_hash({k:v for k,v in value.items() if k!='suite_digest'})


def make_suite(stage,*,finalists=None,model=MODEL,calibration=None,tuning_selection=None):
    if stage not in {'development','tuning','confirmation','boundary','model_boundary'}:
        raise ValueError('unknown experiment stage')
    confirmation=stage in {'confirmation','boundary','model_boundary'}
    if confirmation:
        if not finalists or len(finalists)>2 or len(set(finalists))!=len(finalists) or not set(finalists)<=set(ORDER):
            raise ValueError('confirmation requires one or two distinct selected candidates')
        candidates=list(finalists)
    else: candidates=list(ORDER)
    if stage=='model_boundary': model='Qwen/Qwen2.5-Coder-0.5B-Instruct'
    if model not in MODELS: raise ValueError('model revision is not pinned')
    registry=arm_registry(); bundles=comparison_bundles()
    if tuning_selection is not None:
        values=tuning_selection.get('selected_values',{})
        refs=tuning_selection.get('source_refs',[])
        grids={'M03':('floor',[.03,.1,.3]),'arithmetic_anchor':('floor',[.03,.1,.3]),
               'M01':('alpha',[.25,.5,.75]),'arithmetic_same_smoothing':('alpha',[.25,.5,.75])}
        if (set(values)!=set(grids) or not refs or any(not isinstance(ref,dict) or
                not ref.get('path') or len(ref.get('sha256',''))!=64 for ref in refs)):
            raise ValueError('complete source-bound tuning selection required')
        for family,(key,grid) in grids.items():
            if set(values[family])!={key} or values[family][key] not in grid:
                raise ValueError('tuning selection lies outside the frozen three-value grid')
            registry[family]['parameters'].update(values[family])
        for arm_id in ('arithmetic_anchor_tuned','arithmetic_anchor_sgd','ancestral_soft_anchor'):
            registry[arm_id]['parameters'].update(values['arithmetic_anchor'])
        registry['arithmetic_anchor_tuned']['required_calibration']=[]
    arm_ids=list(dict.fromkeys(a for candidate in candidates for a in bundles[candidate]['arms']))
    tuning_groups={family:[a for a in registry if a.startswith(family+'__')] for family in
                   ('M03','arithmetic_anchor','M01','arithmetic_same_smoothing')}
    if stage=='tuning':
        candidates=['M03','M01']
        arm_ids=['base',*[a for group in tuning_groups.values() for a in group]]
    seeds=SEEDS if confirmation else [17]
    rounds=5 if stage=='boundary' else 3
    benchmarks=list(BENCHMARK_SETTINGS) if confirmation else ['humaneval']
    trajectories=[]; evaluations=[]; aliases={}
    # Baselines can be shared only for byte-identical resolved settings, model,
    # training seed and recursive depth. Arm labels alone are never identity.
    for identifier in arm_ids:
        arm=deepcopy(registry[identifier])
        # Measured clock/step controls can differ by recursive round. Keep the
        # frozen source records here; resolve them only for an actual round job.
        arm_calibration={k:deepcopy(calibration[k]) for k in arm['required_calibration']
                         if calibration is not None and k in calibration}
        for seed in seeds:
            identity={'arm':arm,'model':model,'model_revision':MODELS[model],'seed':seed,
                      'rounds':rounds,'training_lineage':'clean-v2','calibration':arm_calibration}
            tid=identifier+'-'+object_hash(identity)[:16]
            trajectories.append({'trajectory_id':tid,'arm_id':identifier,'arm':arm,
                                 'model':model,'model_revision':MODELS[model],'seed':seed,
                                 'rounds':rounds,'training_lineage':'clean-v2'})
            endpoints=[0] if identifier=='base' else ([rounds] if confirmation or stage=='tuning' else list(range(1,rounds+1)))
            for benchmark in benchmarks:
                split=('confirm' if confirmation else 'dev') if benchmark=='humaneval' else 'full'
                for r in endpoints:
                    setting=deepcopy(BENCHMARK_SETTINGS[benchmark])
                    key={'trajectory_id':tid,'benchmark':benchmark,'split':split,'round':r,'sampling':setting}
                    evaluations.append({'cell_id':identifier+'-'+object_hash(key)[:20],
                        'trajectory_id':tid,'arm_id':identifier,'model':model,'benchmark':benchmark,
                        'split':split,'seed':seed,'round':r,'expected_samples':setting['samples'],
                        'sampling':setting})
    comparisons=[]
    if stage!='tuning':
        for candidate in candidates:
            for benchmark in benchmarks:
                for index,comparison in enumerate(bundles[candidate]['comparisons']):
                    comparisons.append({**deepcopy(comparison),'candidate_id':candidate,
                        'comparison_id':f'{candidate}-{benchmark}-{index:02d}',
                        'model':model,'benchmark':benchmark,'split':'confirm' if confirmation and benchmark=='humaneval' else 'dev' if benchmark=='humaneval' else 'full',
                        'round':rounds,'candidate_round':rounds,
                        'control_round':0 if comparison.get('control_arm')=='base' else rounds,
                        'metrics':['pass@1',BENCHMARK_SETTINGS[benchmark]['coverage']] +
                                  (['training_seconds_relative'] if candidate=='M07' and comparison.get('control_arm')=='full_soft' else [])})
    family_size=sum(len(c['metrics']) for c in comparisons)
    result={'schema':'recursive-ssd-complete-suite-v3','stage':stage,'candidate_ids':candidates,
            'model':model,'model_revision':MODELS[model],'seeds':seeds,'rounds':rounds,
            'training_lineage':'clean-v2','trajectories':trajectories,'evaluation_units':evaluations,
            'comparisons':comparisons,'method_bundles':{c:bundles[c] for c in candidates},
            'tuning_groups':tuning_groups if stage=='tuning' else {},
            'analysis':{'family_size':family_size,'alpha':.05,'bootstrap_seed':20261006,
                        'bootstrap_repeats':max(20000,math.ceil(2*max(1,family_size)*20/.05)),
                        'aggregation':'crossed task and training seed; paired arms; separate benchmarks',
                        'endpoint_round':rounds,'development_only':not confirmation},
            'calibration':deepcopy(calibration or {}),'tuning_selection':deepcopy(tuning_selection),
            'scientific_dispatch_ready':False,
            'admission':'requires exact native protocol, current method evidence, host qualification and remaining budget',
            'unresolved_calibration':sorted({k for a in arm_ids for k in registry[a]['required_calibration']
                                             if k not in (calibration or {})})}
    result['suite_digest']=_digest_suite(result)
    return result


def verify_suite(suite):
    if suite.get('suite_digest')!=_digest_suite(suite): raise ValueError('suite definition changed')
    if len({c['cell_id'] for c in suite['evaluation_units']})!=len(suite['evaluation_units']):
        raise ValueError('duplicate evaluation cell')
    if suite['analysis']['family_size']!=sum(len(c['metrics']) for c in suite['comparisons']):
        raise ValueError('frozen comparison family changed')
    return suite


def expected_cells(suite,benchmark_manifest):
    verify_suite(suite)
    cells=[]
    for unit in suite['evaluation_units']:
        info=benchmark_manifest['benchmarks'][unit['benchmark']]['splits'][unit['split']]
        cells.append({**unit,'task_ids':info['task_ids']})
    return cells
