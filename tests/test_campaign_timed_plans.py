"""No agents or CUDA: exercise the production plan compiler and guarded dispatch."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import validate
from plan_fixtures import experiment_plan, agent_plan
from test_experiment_plan import laboratory
from test_campaign_recovery import lab
from orchestrator import autonomy, controller
from orchestrator.budgets import BudgetExceeded, enforce, max_job_seconds
from orchestrator.campaign_config import plan_seconds
from orchestrator.codex_runner import provider_output_schema
from orchestrator.experiment_plan import validate_plan, execute_plan, digest
from orchestrator.planning_contract import constrain_output_schema, inspect_plan

REPO=Path(__file__).resolve().parents[1]


def timed_plan():
    plan=experiment_plan()
    plan['schema_version']='2.0'
    del plan['resource_budget']
    return plan


@pytest.mark.parametrize('file',['experiment_plan_agent.schema.json','proposals.schema.json'])
def test_actual_generated_schema_has_no_independent_deadlines(file):
    schema=constrain_output_schema(json.loads((REPO/'schemas'/file).read_text()),target_training=False)
    generated=provider_output_schema(schema)
    text=json.dumps(generated)
    for key in ('max_job_seconds','max_gpu_seconds','max_wall_seconds','max_jobs','resource_budget'):
        assert key not in text
    if file.startswith('experiment_plan'):
        validate(agent_plan(timed_plan()),schema)
    else:
        wire=schema['$defs']['proposal']['properties']['experiment_plan']
        validate(agent_plan(timed_plan()),wire)


def test_new_matrix_needs_no_resource_reservations_and_keeps_controls():
    plan=timed_plan();before=copy.deepcopy(plan)
    assert validate_plan(plan)==plan and inspect_plan(plan)['valid']
    assert plan==before
    assert plan_seconds(plan,{})==0
    # Legacy fields cannot be smuggled into scientific training settings.
    for field in ('max_job_seconds','max_gpu_seconds','research_deadline'):
        bad=copy.deepcopy(plan);bad['training'][field]=1
        assert not inspect_plan(bad)['valid']


def test_complete_new_matrix_ignores_historical_gpu_and_environment_limits(laboratory,monkeypatch):
    store,c,cid,runtime,repo,calls,_=laboratory
    history=store.create_run('history','toy_normal_ae','juice_bottle')
    store.account_gpu_usage(history,100000)
    monkeypatch.setenv('LOCO_GPU_BUDGET_HOURS','0')
    monkeypatch.setenv('LOCO_JOB_TIMEOUT_SECONDS','1')
    plan=timed_plan();original=digest(plan);created=c.get(cid)['created_at']
    result=execute_plan(store,c,cid,runtime,plan)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert result['expected_jobs']==6 and len(calls)==6
    assert [(x['plan_arm'],x['seed']) for x in calls]==[(a,s) for a in ('control','intervention') for s in (3,17,29)]
    for cfg in calls:
        assert 'timeout_seconds' not in cfg
        assert cfg['research_deadline']==created+4*3600
        assert cfg['training_budget_seconds']==15*60
        assert cfg['steps']==2 and cfg['expected_source_exposures']==4
        assert max_job_seconds(cfg)>10000
    assert digest(plan)==original and c.get(cid)['status']=='RUNNING'
    assert execute_plan(store,c,cid,runtime,plan)==result and len(calls)==6
    assert store.get_run(history)['gpu_seconds']==100000


def test_queue_claim_does_not_reserve_whole_remaining_campaign(laboratory,monkeypatch):
    store,c,cid,runtime,repo,calls,worker=laboratory
    observed=[]
    # The matrix fixture normally replaces the worker. Inspect the production
    # claim argument before routing to that deterministic CPU fixture.
    from orchestrator import gpu_queue
    def claim(target,max_duration,expected_run_id):
        observed.append(max_duration)
        gpu_queue.enforce(target,next_gpu_seconds=max_duration,campaign_id=cid,runtime=runtime)
        return None
    import importlib.util
    spec=importlib.util.spec_from_file_location('production_controller',REPO/'orchestrator/controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'claim_next',claim)
    def inspect_worker(*args,**kwargs):
        assert module.run_worker_once(*args,**kwargs)=={'status':'IDLE'}
        return worker(*args,**kwargs)
    monkeypatch.setattr(controller,'run_worker_once',inspect_worker)
    result=execute_plan(store,c,cid,runtime,timed_plan())
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert observed==[0]*6


def test_plan_validation_failure_recovers_without_limit_reached(lab,monkeypatch):
    store,c,cid,runtime=lab;attempts=[];created=c.get(cid)['created_at']
    monkeypatch.setattr(autonomy,'preflight',lambda *a,**k:None)
    def cycle(*args):
        attempts.append(c.get(cid)['consecutive_failures'])
        if len(attempts)<=10:
            bad=timed_plan();bad['training']['device']='cuda'
            validate_plan(bad)
        assert c.get(cid)['created_at']==created
        assert any(e['kind']=='RECOVERY_RESTORED' for e in c.events(cid,100))
        with store.connect() as con:
            con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    monkeypatch.setattr(autonomy,'run_cycle',cycle)
    autonomy.run_forever(cid,runtime,store.path,idle_seconds=0)
    assert attempts==list(range(10))+[0]
    assert c.get(cid)['status']=='LIMIT_REACHED'
    assert c.get(cid)['reason']=='wall-time budget reached'
    with pytest.raises(BudgetExceeded,match='campaign is not running'):
        enforce(store,campaign_id=cid,runtime=runtime)


def test_frozen_legacy_plan_preserves_original_restrictions(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory
    plan=experiment_plan();plan['resource_budget']['max_gpu_seconds']=1
    before=copy.deepcopy(plan)
    result=execute_plan(store,c,cid,runtime,plan)
    assert result['status']=='INCOMPLETE' and not calls
    assert plan==before and c.get(cid)['status']=='RUNNING'
    assert 'exact planned job allowance' in ' '.join(result['problems'])


def test_expired_deadline_rejects_zero_reservation_without_legacy_limit(lab):
    store,c,cid,runtime=lab
    with store.connect() as con:
        con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    with pytest.raises(BudgetExceeded,match='research duration'):
        enforce(store,next_gpu_seconds=0,campaign_id=cid,runtime=runtime)


def test_new_plan_evaluation_uses_original_campaign_deadline(laboratory,monkeypatch):
    store,c,cid,runtime,repo,calls,worker=laboratory
    with store.connect() as con:
        cfg=json.loads(c.get(cid)['config_json']);cfg['research']['real_development_feedback']=True
        # Obsolete per-feedback limit must not truncate evaluation.
        cfg['research']['feedback_cpu_timeout_seconds']=1
        con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    from orchestrator import development_feedback
    seen=[]
    def evaluate(*args,**kwargs):
        seen.append(kwargs)
        return {'status':'COMPLETED'}
    monkeypatch.setattr(development_feedback,'evaluate_run',evaluate)
    plan=timed_plan();plan['evaluation']['real_development_feedback']=True
    execute_plan(store,c,cid,runtime,plan)
    assert len(seen)==6
    deadline=c.get(cid)['created_at']+4*3600
    assert all(item['timeout']>10000 and item['deadline']==deadline for item in seen)
    # No synthetic successful evaluation was persisted by this fixture.
    assert all(development_feedback.latest_summary(store,r['run_id']) is None for r in c.runs(cid))


def test_sandbox_planner_reads_the_actual_generated_contract(tmp_path,monkeypatch):
    """Intercept the command; never invoke a CLI, API or model."""
    from types import SimpleNamespace
    from orchestrator import codex_runner,codex_sandbox
    from plan_fixtures import agent_plan
    schema=constrain_output_schema(json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text()),target_training=False)
    path=tmp_path/'generated.json';path.write_text(json.dumps(schema))
    response={}
    def command(root,role,provider,output,task,candidate_dir,**kwargs):
        reference=json.loads(kwargs['planning_reference'].read_text())
        assert reference==json.loads(provider.read_text())
        assert reference['properties']['schema_version']['const']=='2.0'
        assert 'resource_budget' not in json.dumps(reference)
        response['path']=output
        return ['intercepted-offline-fixture']
    def run(*args,**kwargs):
        response['path'].write_text(json.dumps(agent_plan(timed_plan())))
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(codex_sandbox,'command',command)
    monkeypatch.setattr(codex_runner.subprocess,'run',run)
    result=codex_runner.invoke('researcher',REPO,path,'offline schema transport fixture',personal_mode=True)
    assert result['response']['schema_version']=='2.0'


@pytest.mark.parametrize('missing',['arm','diagnostic'])
def test_new_plan_keeps_incomplete_scientific_comparisons_ineligible(laboratory,missing):
    from orchestrator.experiment_plan import comparison_result
    store,c,cid,runtime,repo,calls,_=laboratory
    plan=timed_plan();execute_plan(store,c,cid,runtime,plan)
    rows=copy.deepcopy(c.runs(cid))
    if missing=='arm':rows=rows[:3]
    else:
        metrics=json.loads(rows[-1]['metrics_json'])
        del metrics['diagnostic/fixture_check/mean_score']
        rows[-1]['metrics_json']=json.dumps(metrics)
    result=comparison_result(plan,rows)
    assert result['status']=='INCOMPLETE'
    assert result['scientific_eligibility']['promotion_eligible'] is False
