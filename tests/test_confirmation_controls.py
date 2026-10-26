"""Scaled CPU learning fixtures for actual observation, not scientific evidence."""
import copy
import hashlib
import json
from pathlib import Path
import time

import pytest

from orchestrator.confirmation import normal_interval,verify_pairs
from orchestrator.experiment_plan import digest,validate_plan
from orchestrator.planning_contract import comparison_controls,constrain_output_schema
from orchestrator.codex_runner import provider_output_schema
from scripts.training_phase import run_training_phases
from test_training_loop import timed_module,loop_config
from test_required_training import plan3


def future_plan():
    p=plan3();p['schema_version']='4.0';p['confirmation']=False
    p['training'].update(sampling_protocol='fixture',sampling_seed=42)
    p['controlled_variables']+=['sampling_protocol','sampling_seed']
    p['confirmation_requirements']={'normal_images':47,'max_additional_false_positives':1,
        'paired_controls':['initial_state','optimizer','input:0'],
        'uncertainty':'Independent paired training seeds; paired/grouped source intervals; marginal binomial normals; repeated development selection remains biased.'}
    return p


def test_counts_have_explicit_granularity_and_uncertainty():
    one=normal_interval(1,47);zero=normal_interval(0,47)
    assert one['rate']==pytest.approx(1/47) and one['rate']>.02
    assert zero['lower']==0 and zero['upper']>0
    assert one['lower']<one['rate']<one['upper'] and 'paired difference' in one['limitation']


def test_future_requirements_use_provider_compatible_schema_without_rewriting_old_plans():
    repo=Path(__file__).parents[1]
    schema=constrain_output_schema(json.loads((repo/'schemas/experiment_plan_agent.schema.json').read_text()))
    compiled=provider_output_schema(schema)
    assert 'confirmation_requirements' in compiled['required']
    p=future_plan();before=copy.deepcopy(p);validate_plan(p);assert p==before
    p['confirmation_requirements']['paired_controls']=['input:garbage']
    with pytest.raises(ValueError,match='prepared input'):validate_plan(p)


def test_actual_state_inputs_and_optimizer_join_and_detect_mismatch(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    plan=future_plan();plan['seeds']=[3];rows=[]
    for arm in plan['arms']:
        root=tmp_path/arm['arm_id'];root.mkdir()
        cfg=loop_config();cfg['paired_control_requirements']=plan['confirmation_requirements']['paired_controls']
        run_training_phases(timed_module(now),cfg,tmp_path,root/'checkpoint',budget=.9)
        audit=json.loads((root/'training_timing.json').read_text())
        (root/'run_manifest.json').write_text(json.dumps({'artifact_hashes':{
            path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [root/'training_timing.json',root/'paired_input_0.sha256']}}))
        assert audit['loop_elapsed_seconds']==pytest.approx(1.)
        assert audit['learning_seconds']==pytest.approx(.2)
        assert audit['paired_controls']['observed']['input:0']['updates']==2
        rows.append({'artifact_dir':str(root),'config_json':json.dumps(dict(plan['training'],seed=3,
            plan_sha256=digest(plan),plan_arm=arm['arm_id'])),'metrics_json':json.dumps({'protocol/optimizer_updates':2,'protocol/source_exposures':4})})
    result=verify_pairs(plan,rows)
    assert result['status']=='VERIFIED' and result['pairs'][0]['coverage']['input:0']['shared_updates_verified']==2
    assert comparison_controls(plan,rows)['sampling_stream_status']=='VERIFIED'
    # Tampering fails hash checks before it can support any confirmation.
    path=Path(rows[1]['artifact_dir'])/'paired_input_0.sha256';path.write_bytes(b'x'*64)
    assert verify_pairs(plan,rows)['status']=='MISMATCHED'


def test_missing_observations_never_become_verified_by_equal_seeds(tmp_path):
    plan=future_plan();plan['seeds']=[3]
    rows=[{'artifact_dir':str(tmp_path),'config_json':json.dumps(dict(plan['training'],plan_sha256=digest(plan),
        plan_arm=a['arm_id'],seed=3)),'metrics_json':'{}'} for a in plan['arms']]
    controls=comparison_controls(plan,rows)
    assert not controls['causal_claim_eligible'] and controls['sampling_stream_status']=='NOT_MEASURED'
