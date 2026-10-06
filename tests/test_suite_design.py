"""Queue/dependency fixtures only; no fabricated benchmark evaluation."""
import importlib
import pytest


def api():
    return importlib.import_module('recursive_ssd.suite_design')


def test_all_fifteen_complete_bundles_resolve_real_arms():
    registry=api().arm_registry()
    bundles=api().comparison_bundles()
    assert len(bundles)==15
    for candidate,bundle in bundles.items():
        assert candidate in registry
        assert {'base','hard','full_soft'}<=set(bundle['arms'])
        assert set(bundle['arms'])<=set(registry)
        assert bundle['comparisons'] and bundle['question'] and bundle['falsifier']
    assert len([a for a in registry if a.startswith('gkd_')])==6
    assert {'head_identity','head_shape_only','mass_transfer_only','M05'}<=set(bundles['M05']['arms'])
    assert {'hard','M15','arithmetic_anchor','ancestral_soft_anchor'}<=set(bundles['M15']['arms'])


def test_development_is_shared_complete_controls_not_fifteen_isolated_pilots():
    suite=api().make_suite('development')
    assert suite['candidate_ids']==['M03','M04','M05','M01','M07','M02','M06','M08','M09','M10','M11','M12','M13','M14','M15']
    assert {x['seed'] for x in suite['trajectories']}=={17}
    ids=[x['trajectory_id'] for x in suite['trajectories']]
    assert len(ids)==len(set(ids))
    assert {x['benchmark'] for x in suite['evaluation_units']}=={'humaneval'}
    assert {x['split'] for x in suite['evaluation_units']}=={'dev'}
    assert all(x['expected_samples']==10 for x in suite['evaluation_units'])
    assert suite['scientific_dispatch_ready'] is False


def test_confirmation_covers_all_benchmarks_and_five_fresh_training_seeds():
    suite=api().make_suite('confirmation',finalists=['M03','M09'])
    assert {x['benchmark'] for x in suite['evaluation_units']}=={'humaneval','mbpp','livecodebench'}
    assert {x['seed'] for x in suite['trajectories']}=={23,47,71,101,131}
    assert all(x['round'] in {0,3} for x in suite['evaluation_units'])
    assert all(x['split']=='confirm' if x['benchmark']=='humaneval' else x['split']=='full' for x in suite['evaluation_units'])
    assert all(x['model']=='Qwen/Qwen2.5-Coder-1.5B-Instruct' for x in suite['trajectories'])
    assert suite['analysis']['family_size']==sum(len(c['metrics']) for c in suite['comparisons'])


@pytest.mark.parametrize('finalists',[[],['M03','M09','M01'],['M16'],['M03','M03']])
def test_confirmation_rejects_ineligible_finalist_inventory(finalists):
    with pytest.raises(ValueError): api().make_suite('confirmation',finalists=finalists)


def test_temperature_conditions_and_gkd_refresh_are_explicit():
    arms=api().arm_registry()
    assert arms['temperature_fixed_untruncated']['decode']['top_k']==0
    assert arms['temperature_fixed_untruncated']['decode']['top_p']==1
    assert arms['temperature_once_then_one']['temperature_schedule']=='once'
    assert arms['gkd_fkl_update']['rollout_refresh']=='update'
    assert arms['gkd_fkl_cached']['rollout_refresh']=='round'
    assert arms['full_soft_lr_half']['method']=='full_soft'
    assert arms['full_soft_lr_half']['train']['learning_rate_multiplier']==.5
    assert arms['soft_current_mix']['method']=='soft_current_mix'


def test_unmeasured_match_is_not_filled_with_a_convenient_constant():
    arm=api().arm_registry()['temperature_matched_head']
    with pytest.raises(ValueError,match='calibration'): api().resolve_arm(arm,{})
    result=api().resolve_arm(arm,{'head_temperature':{'value':.72,'source_refs':[{'path':'real-record.json','sha256':'a'*64}]}})
    assert result['parameters']['temperature_beta']==.72


def test_frozen_comparison_identity_changes_with_model_round_or_settings():
    first=api().make_suite('confirmation',finalists=['M03'])
    boundary=api().make_suite('boundary',finalists=['M03'])
    assert first['suite_digest']!=boundary['suite_digest']
    assert {u['round'] for u in boundary['evaluation_units']}=={0,5}
    assert all(u['rounds']==5 for u in boundary['trajectories'])


def test_fair_tuning_inventory_has_equal_declared_options():
    suite=api().make_suite('tuning')
    choices=suite['tuning_groups']
    assert len(choices['M03'])==len(choices['arithmetic_anchor'])==3
    assert len(choices['M01'])==len(choices['arithmetic_same_smoothing'])==3
    assert all(u['split']=='dev' for u in suite['evaluation_units'])


def test_selected_development_hyperparameters_reach_confirmation_trajectories():
    selected={'selected_values':{'M03':{'floor':.3},'arithmetic_anchor':{'floor':.03},
        'M01':{'alpha':.75},'arithmetic_same_smoothing':{'alpha':.25}},
        'source_refs':[{'path':'development-selection.json','sha256':'a'*64}]}
    suite=api().make_suite('confirmation',finalists=['M03','M01'],tuning_selection=selected)
    arms={t['arm_id']:t['arm'] for t in suite['trajectories']}
    assert arms['M03']['parameters']['floor']==.3
    assert arms['arithmetic_anchor']['parameters']['floor']==.03
    assert arms['arithmetic_anchor_tuned']['parameters']['floor']==.03
    assert arms['arithmetic_anchor_tuned']['required_calibration']==[]
    assert arms['M01']['parameters']['alpha']==.75
    assert arms['arithmetic_same_smoothing']['parameters']['alpha']==.25
    assert suite['tuning_selection']==selected
    assert suite['suite_digest']!=api().make_suite('confirmation',finalists=['M03','M01'])['suite_digest']


def test_out_of_grid_or_unbound_tuning_choices_are_rejected():
    packet={'selected_values':{'M03':{'floor':.99}},'source_refs':[]}
    with pytest.raises(ValueError,match='tuning'):
        api().make_suite('development',tuning_selection=packet)


def test_fixed_first_round_data_is_a_shared_core_control_in_every_bundle():
    assert all('fixed_data' in bundle['arms'] for bundle in api().comparison_bundles().values())
