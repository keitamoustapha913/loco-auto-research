"""Deterministic configuration/dispatch integration; no research agents or GPU work."""
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orchestrator.campaign_config import normalize, starting_context, time_based
from orchestrator.autonomy import load_campaign_config, stop_reason
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from orchestrator.budgets import enforce, CodexBudgetExhausted, max_job_seconds
from orchestrator.experiment_plan import execute_plan, digest
from scripts.run_candidate import train_measured
from legacy_campaign import legacy_config
from plan_fixtures import experiment_plan
from test_experiment_plan import laboratory
from test_model_registry import lab as registry_lab


def simple(mode='new', hours=4, minutes=15):
    config=legacy_config()
    config['campaign']['start_from']=mode
    config['limits']={'research_hours':hours,'training_minutes_per_experiment':minutes}
    return normalize(config)


@pytest.mark.parametrize('mode',['new','previous_best'])
def test_two_time_values_load_defaults_without_authorizing(tmp_path,mode):
    cfg=simple(mode)
    public={'campaign':cfg['campaign'],'limits':{'research_hours':4,'training_minutes_per_experiment':15}}
    path=tmp_path/'operator.yaml';path.write_text(yaml.safe_dump(public))
    got=load_campaign_config(path)
    assert got['campaign']['start_from']==mode and time_based(got)
    assert got['limits']['max_codex_calls'] is None and got['limits']['max_experiments'] is None
    assert got['limits']['max_job_seconds']>900
    assert got['security']['operator_accepts_shared_uid_risk'] is False
    assert got['research']['official_test_access'] is False


@pytest.mark.parametrize('bad',[0,-1,True,float('inf'),float('nan'),'4'])
def test_invalid_time_rejected(bad):
    cfg=legacy_config();cfg['limits']={'research_hours':bad,'training_minutes_per_experiment':15}
    with pytest.raises(ValueError,match='finite positive'):normalize(cfg)


def test_mixed_configuration_and_unknown_mode_rejected():
    cfg=legacy_config();cfg['limits'].update(research_hours=4,training_minutes_per_experiment=15)
    with pytest.raises(ValueError,match='mix'):normalize(cfg)
    with pytest.raises(ValueError,match='start_from'):simple('latest')


def test_legacy_migration_is_lossless_and_does_not_rewrite_file(tmp_path):
    cfg=legacy_config();path=tmp_path/'legacy.yaml';original=yaml.safe_dump(cfg);path.write_text(original)
    loaded=load_campaign_config(path)
    assert loaded['limits']==cfg['limits'] and not time_based(loaded)
    assert path.read_text()==original


def test_new_campaign_has_own_clock_and_ignores_exhausted_legacy_globals(tmp_path,monkeypatch):
    for variable in ['LOCO_GPU_BUDGET_HOURS','LOCO_JOB_TIMEOUT_SECONDS','LOCO_CODEX_MAX_CALLS']:
        monkeypatch.setenv(variable,'0')
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store)
    old=store.create_run('history','toy_normal_ae','juice_bottle');store.account_gpu_usage(old,100000)
    # Historical record is preserved; campaign accounting does not reset it.
    cid=campaigns.create(simple());campaigns.reserve_codex_call(cid)
    enforce(store,next_gpu_seconds=120,campaign_id=cid,runtime=tmp_path)
    assert campaigns.call_budget(cid)['global']['limit'] is None
    assert campaigns.call_budget(cid)['campaign']['remaining'] is None
    assert store.get_run(old)['gpu_seconds']==100000
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    assert stop_reason(campaigns,cid)==('LIMIT_REACHED','wall-time budget reached')
    with pytest.raises(CodexBudgetExhausted,match='research duration'):campaigns.reserve_codex_call(cid)
    assert campaigns.usage(cid)['codex_calls']==1
    campaigns.set_status(cid,'LIMIT_REACHED')
    fresh=campaigns.create(simple())
    assert campaigns.usage(fresh)['wall_seconds']<2 and campaigns.usage(fresh)['codex_calls']==0
    assert campaigns.usage(cid)['wall_seconds']>14400


def test_training_time_is_distinct_from_validation_and_job_timeout(tmp_path):
    cfg={'device':'cpu','training_budget_seconds':.2}
    from training_phase_fixtures import phase_module
    module=phase_module(training_delay=.015)
    _,duration=train_measured(module,cfg,tmp_path,tmp_path/'checkpoint')
    time.sleep(.04)  # Represents post-training validation, excluded from learning time.
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert .01<duration<.2 and audit['status']=='COMPLETED'
    assert audit['budget_seconds']==.2
    assert max_job_seconds({'campaign_id':'fixture','campaign_configuration_version':'campaign-time-v1','timeout_seconds':10000})==10000


def test_training_deadline_does_not_continue_to_validation(tmp_path):
    from training_phase_fixtures import phase_module
    module=phase_module(training_delay=.3)
    with pytest.raises(TimeoutError,match='training-time budget'):
        train_measured(module,{'device':'cpu','training_budget_seconds':.02},tmp_path,tmp_path/'checkpoint')
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['status']=='TIMED_OUT' and .015<=audit['duration_seconds']<.2


def test_cuda_training_timing_synchronizes_without_real_cuda(tmp_path,monkeypatch):
    import sys
    calls=[]
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:calls.append('sync'))))
    train_measured(SimpleNamespace(train=lambda *a:calls.append('train')),{'device':'cuda'},tmp_path,tmp_path/'checkpoint')
    assert calls==['sync','train','sync']


def test_full_six_job_plan_not_shortened_by_legacy_timeout(laboratory,monkeypatch):
    store,c,cid,runtime,repo,calls,_=laboratory
    monkeypatch.setenv('LOCO_JOB_TIMEOUT_SECONDS','1');monkeypatch.setenv('LOCO_GPU_BUDGET_HOURS','0')
    plan=experiment_plan(seeds=(7,19,31));before=digest(plan)
    result=execute_plan(store,c,cid,runtime,plan)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert [(x['plan_arm'],x['seed']) for x in calls]==[(a,s) for a in ['control','intervention'] for s in [7,19,31]]
    assert all(x['timeout_seconds']==5 and x['steps']==2 and x['training_budget_seconds']==900 for x in calls)
    assert digest(plan)==before


def test_complete_plan_admission_includes_evaluation_and_critique(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory
    cfg=json.loads(c.get(cid)['config_json']);cfg['limits']['max_wall_hours']=.04
    with store.connect() as con:con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    plan=experiment_plan();plan['resource_budget']['max_wall_seconds']=144
    result=execute_plan(store,c,cid,runtime,plan)
    assert result['status']=='INCOMPLETE' and calls==[]
    assert c.get(cid)['status']=='RUNNING'  # A rejected plan is not the campaign deadline.
    assert any('complete frozen comparison' in p for p in result['problems'])
    assert result['scientific_eligibility']['promotion_eligible'] is False


@pytest.mark.parametrize('mode',['previous_best','new'])
def test_start_modes_preserve_incumbent_and_all_evidence(registry_lab,mode):
    from orchestrator import model_registry as registry
    store,c,cid,repo,add=registry_lab
    old=add(seed=7);registry.initialize(store,repo)
    before=registry.states(store);archives=registry.retained(store)
    context=starting_context(store,simple(mode),strict=True)
    assert bool(context['models'])==(mode=='previous_best')
    if mode=='previous_best':assert context['models'][0]['run_id']==old
    c.set_status(cid,'COMPLETED');fresh=c.create(simple(mode))
    assert CampaignState(StateStore(store.path)).get(fresh)['status']=='RUNNING'
    assert registry.states(store)==before
    for model in archives.values():registry.verify_archive(model)
    events=c.events(fresh)
    assert next(json.loads(e['payload_json']) for e in events if e['kind']=='STARTING_CONTEXT')==context


def test_previous_best_requires_an_eligible_retained_reference(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite')
    with pytest.raises(ValueError,match='No eligible retained'):starting_context(store,simple('previous_best'),strict=True)


def test_cli_report_prefers_research_and_training_times(tmp_path):
    from orchestrator.lab_cli import research_summary
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    report=research_summary(c,c.get(cid),None)
    assert report['training_seconds_per_experiment']==900
    assert report['actual_training_seconds']==0 and report['research_duration_seconds']>=0
    assert 'gpu_seconds' not in report['usage'] and 'codex_budget' not in report
    assert report['technical_accounting']['usage']['gpu_seconds']==0


def test_removed_globals_need_not_even_parse_for_new_campaigns(tmp_path,monkeypatch):
    for name in ['LOCO_GPU_BUDGET_HOURS','LOCO_JOB_TIMEOUT_SECONDS','LOCO_CODEX_MAX_CALLS']:
        monkeypatch.setenv(name,'obsolete')
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    c.reserve_codex_call(cid)
    enforce(store,next_gpu_seconds=1,campaign_id=cid,runtime=tmp_path)
    assert c.call_budget(cid)['global']['remaining'] is None


def test_paused_campaign_clock_does_not_reset_or_wait_past_deadline(tmp_path):
    from orchestrator.autonomy import await_authorized_stage,CampaignHalted
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    c.set_status(cid,'PAUSED')
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    with pytest.raises(CampaignHalted,match='wall-time'):await_authorized_stage(c,cid)
    assert c.get(cid)['status']=='LIMIT_REACHED'


def test_expired_campaign_cannot_start_or_repair_agent_work(tmp_path,monkeypatch):
    from orchestrator import controller
    from orchestrator.autonomy import CampaignHalted
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    monkeypatch.setattr(controller,'invoke',lambda *a,**k:pytest.fail('agent must not run'))
    with pytest.raises(CampaignHalted):
        controller._budgeted_invoke_once(store,'scout',Path(__file__).parents[1],None,'fixture',campaign=(c,cid))
    assert c.usage(cid)['codex_calls']==0 and c.get(cid)['consecutive_failures']==0


@pytest.mark.parametrize('mode',['previous_best','new'])
def test_existing_cli_start_pause_resume_report_stop_with_simple_config(registry_lab,tmp_path,monkeypatch,mode):
    from orchestrator import model_registry as registry,autonomy,lab_cli
    store,c,old_cid,repo,add=registry_lab
    add(seed=7);registry.initialize(store,repo);before=registry.states(store)
    c.set_status(old_cid,'COMPLETED')
    cfg={'campaign':simple(mode)['campaign'],'limits':{'research_hours':4,'training_minutes_per_experiment':15},
         'security':{'operator_accepts_shared_uid_risk':True}}
    path=tmp_path/'approved.yaml';path.write_text(yaml.safe_dump(cfg));original=path.read_bytes()
    monkeypatch.setattr(autonomy,'preflight',lambda *a,**k:{'images':389})
    monkeypatch.setattr(lab_cli,'spawn_research',lambda runtime,campaigns,cid:lab_cli.research_summary(campaigns,campaigns.get(cid),None))
    monkeypatch.setattr(lab_cli,'owned_process',lambda *a:None)
    runtime=store.path.parent.parent
    started=lab_cli.research_command('start',runtime,store.path,path)
    assert started['status']=='RUNNING' and started['start_from']==mode
    again=lab_cli.research_command('start',runtime,store.path,path)
    assert again['campaign_id']==started['campaign_id']
    assert lab_cli.research_command('pause',runtime,store.path)['status']=='PAUSED'
    assert lab_cli.research_command('resume',runtime,store.path)['status']=='RUNNING'
    report=lab_cli.research_command('report',runtime,store.path)
    assert report['training_seconds_per_experiment']==900
    assert set(report['limits'])=={'research_hours','training_minutes_per_experiment'}
    assert 'max_job_seconds' in report['technical_accounting']['limits']
    assert lab_cli.research_command('stop',runtime,store.path)['status']=='STOPPED'
    assert registry.states(store)==before and path.read_bytes()==original


def test_implementation_does_not_start_if_complete_plan_will_not_fit(tmp_path,monkeypatch):
    from orchestrator import controller
    from orchestrator.autonomy import CampaignHalted
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    store.add_proposal({'hypothesis_id':'fixture','experiment_plan':experiment_plan()})
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14300 WHERE id=?',(cid,))
    monkeypatch.setattr(controller,'create_candidate_worktree',lambda *a:pytest.fail('must not start implementation'))
    with pytest.raises(CampaignHalted,match='complete plan'):
        controller.implement_proposal(store,'fixture',campaign=(c,cid))
    assert c.get(cid)['status']=='LIMIT_REACHED' and c.usage(cid)['codex_calls']==0


def test_training_display_does_not_depend_on_optional_recent_ui_changes():
    from dashboard.campaign_timing import campaign_training_seconds
    campaign={'id':'selected'}
    runs=[{'run_id':'a','config':{'campaign_id':'selected'},'metrics':{'training/loop_elapsed_seconds':12,'training/learning_seconds':2}},
          {'run_id':'b','config':{'campaign_id':'other'},'metrics':{'training/duration_seconds':99}}]
    assert campaign_training_seconds(campaign,runs)==12


def test_training_timing_reader_rejects_unsafe_or_invalid_artifacts(tmp_path):
    from orchestrator.experiment_runner import read_training_timing
    target=tmp_path/'timing.json';alias=tmp_path/'alias';alias.symlink_to(target)
    for value in [{'status':'RUNNING','started_at':float('nan')},
                  {'status':'COMPLETED','started_at':1,'duration_seconds':-1},[],{'junk':'x'*5000}]:
        target.write_text(json.dumps(value));assert read_training_timing(target)=={}
    target.write_text(json.dumps({'status':'COMPLETED','started_at':1,'duration_seconds':2}))
    assert read_training_timing(alias)=={}
    assert read_training_timing(target)['duration_seconds']==2


def test_results_page_shows_training_and_research_duration(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    rid=store.create_run('fixture-hypothesis','fixture-candidate','juice_bottle',config={'campaign_id':cid})
    for status in ['IMPLEMENTED','VALIDATED','QUEUED','RUNNING']:store.transition(rid,status)
    store.set_run_result(rid,{'training/loop_elapsed_seconds':12.5,'training/duration_seconds':12.5},gpu_seconds=99,artifact_dir=tmp_path/'artifact')
    store.transition(rid,'COMPLETED');c.attach_run(cid,rid)
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path));monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    page=Path(__file__).parents[1]/'dashboard/pages/03_Results.py'
    rendered=AppTest.from_file(str(page),default_timeout=10).run()
    assert not rendered.exception
    metrics={item.label:item.value for item in rendered.metric}
    assert 'Research duration' in metrics and 'Training-loop duration' in metrics
    assert 'GPU-slot time' not in metrics and 'Codex calls' not in metrics


def test_expiry_finalizes_registered_comparison_without_replaying(laboratory):
    from orchestrator.experiment_plan import register_plan
    from orchestrator.autonomy import close_unfinished_comparisons
    store,c,cid,runtime,repo,calls,_=laboratory
    p=experiment_plan();register_plan(store,c,cid,p)
    close_unfinished_comparisons(store,c,cid,runtime,'research duration exhausted before implementation')
    results=[json.loads(e['payload_json'])['result'] for e in c.events(cid) if e['kind']=='EXPERIMENT_PLAN_RESULT']
    assert len(results)==1 and results[0]['status']=='INCOMPLETE' and calls==[]
    close_unfinished_comparisons(store,c,cid,runtime,'same deadline')
    assert len([e for e in c.events(cid) if e['kind']=='EXPERIMENT_PLAN_RESULT'])==1
