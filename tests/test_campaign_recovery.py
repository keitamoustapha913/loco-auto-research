"""Offline stage/deadline/recovery integration: no agents, datasets or CUDA."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from orchestrator import autonomy,controller,codex_runner,development_feedback
from orchestrator.campaign_config import operation_seconds,operation_cancelled
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from test_time_campaigns import simple
from test_model_registry import lab as registry_lab


@pytest.fixture
def lab(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store)
    cid=campaigns.create(simple('new'))
    return store,campaigns,cid,tmp_path


def test_observed_sequence_resets_only_on_validated_success(lab):
    store,c,cid,_=lab;created=c.get(cid)['created_at']
    c.recoverable_failure(cid,'Scout timeout');assert c.get(cid)['consecutive_failures']==1
    c.validated_stage(cid,'scout');assert c.get(cid)['consecutive_failures']==0
    c.recoverable_failure(cid,'Researcher timeout');c.recoverable_failure(cid,'Scout timeout')
    assert c.get(cid)['consecutive_failures']==2
    assert autonomy.stop_reason(c,cid)==(None,'') and c.get(cid)['created_at']==created


def test_ten_failures_recover_durably_without_resetting_clock(lab):
    store,c,cid,runtime=lab;created=c.get(cid)['created_at']
    for index in range(10):c.recoverable_failure(cid,f'failure evidence {index}')
    assert autonomy.stop_reason(c,cid)==(None,'')
    restarted=CampaignState(StateStore(store.path))
    assert autonomy.recover_research_attempt(restarted.store,restarted,cid,runtime)
    assert restarted.get(cid)['status']=='RUNNING' and restarted.get(cid)['consecutive_failures']==0
    assert restarted.get(cid)['created_at']==created
    assert restarted.get(cid)['next_hypothesis'].startswith('recovery:')
    assert not autonomy.recover_research_attempt(store,c,cid,runtime)
    events=c.events(cid,100)
    assert len([e for e in events if e['kind']=='RECOVERY_RESTORED'])==1
    assert len([e for e in events if e['kind']=='ATTEMPT_FAILED'])==10


def test_retry_loop_survives_ten_failures_and_expires_on_original_deadline(lab,monkeypatch):
    store,c,cid,runtime=lab;called=[];created=c.get(cid)['created_at']
    monkeypatch.setattr(autonomy,'preflight',lambda *a,**k:None)
    def cycle(target,campaigns,ident,folder):
        called.append(campaigns.get(ident)['consecutive_failures'])
        if len(called)<=10:raise RuntimeError('recoverable fixture transport failure')
        assert campaigns.get(ident)['created_at']==created
        assert any(e['kind']=='RECOVERY_RESTORED' for e in campaigns.events(ident))
        with target.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(ident,))
    monkeypatch.setattr(autonomy,'run_cycle',cycle)
    autonomy.run_forever(cid,runtime,store.path,idle_seconds=0)
    assert called==list(range(10))+[0]
    assert c.get(cid)['status']=='LIMIT_REACHED' and c.get(cid)['reason']=='wall-time budget reached'


def test_previous_campaign_failures_do_not_affect_new_campaign(lab):
    store,c,cid,_=lab
    for _ in range(10):c.recoverable_failure(cid,'old failure')
    c.set_status(cid,'STOPPED');fresh=c.create(simple('new'))
    assert c.get(fresh)['consecutive_failures']==0 and c.usage(fresh)['wall_seconds']<2


def test_all_agent_roles_receive_remaining_deadline_not_legacy_cap(lab,monkeypatch):
    store,c,cid,_=lab;received=[]
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS','0');monkeypatch.setenv('LOCO_JOB_TIMEOUT_SECONDS','1')
    def fake(*a,**kw):received.append(kw);return {'response':{}}
    monkeypatch.setattr(controller,'invoke',fake)
    for role in ('scout','researcher','implementer','critic'):
        controller._budgeted_invoke_once(store,role,Path('.'),None,'fixture',campaign=(c,cid),timeout=1)
    assert len(received)==4
    for kw in received:
        assert 14300<kw['timeout']<=14400 and not kw['cancel']()
        assert kw['deadline']==c.get(cid)['created_at']+14400
    c.set_status(cid,'STOPPING');assert all(kw['cancel']() for kw in received)


def test_semantic_validation_not_raw_agent_completion_resets_counter(lab):
    store,c,cid,_=lab;c.recoverable_failure(cid,'old failure')
    with pytest.raises(ValueError):
        controller._with_agent_correction(store,(c,cid),'researcher',lambda *a:{},
                                         lambda response:(_ for _ in ()).throw(ValueError('bad plan')))
    assert c.get(cid)['consecutive_failures']==1
    controller._with_agent_correction(store,(c,cid),'researcher',lambda *a:{},lambda response:True)
    assert c.get(cid)['consecutive_failures']==0


def test_feedback_uses_original_campaign_deadline_not_legacy_settings(lab,monkeypatch):
    store,c,cid,runtime=lab;(runtime/'controller').mkdir()
    run=store.create_run('fixture','fixture','fixture',config={'campaign_id':cid})
    captured={}
    def evaluate(*args):captured.update(timeout=args[3],cancel=args[4]);return {'fixture':True}
    monkeypatch.setattr(development_feedback,'_evaluate',evaluate)
    assert development_feedback.evaluate_run(runtime,store,run,timeout=1,personal_mode_authorized=True)=={'fixture':True}
    assert captured['timeout']>14300 and not captured['cancel']()
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14401 WHERE id=?',(cid,))
    assert captured['cancel']()
    with pytest.raises(TimeoutError):development_feedback.evaluate_run(runtime,store,run,personal_mode_authorized=True)


def test_cancellable_subprocess_honors_deadline_and_reaps_child(tmp_path):
    pidfile=tmp_path/'pid'
    code=f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(10)'
    start=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        codex_runner.run_cancellable([sys.executable,'-c',code],timeout=10,deadline=time.time()+.2,text=True)
    assert time.monotonic()-start<2
    import os
    with pytest.raises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)


def test_safety_error_stops_instead_of_retrying(lab,monkeypatch):
    store,c,cid,runtime=lab;calls=[]
    monkeypatch.setattr(autonomy,'preflight',lambda *a,**k:None)
    def fail(*a):calls.append(1);raise PermissionError('checkpoint hash changed')
    monkeypatch.setattr(autonomy,'run_cycle',fail)
    autonomy.run_forever(cid,runtime,store.path,idle_seconds=0)
    assert len(calls)==1 and c.get(cid)['status']=='FAILED'


def test_recovery_abandons_plan_as_incomplete_without_mutating_plan(lab):
    from plan_fixtures import experiment_plan
    from orchestrator.experiment_plan import register_plan,digest
    store,c,cid,runtime=lab;plan=experiment_plan('fixture')
    before=digest(plan);store.add_proposal({'hypothesis_id':'fixture','experiment_plan':plan})
    register_plan(store,c,cid,plan);c.update(cid,current_hypothesis='fixture')
    for _ in range(10):c.recoverable_failure(cid,'invalid implementation')
    autonomy.recover_research_attempt(store,c,cid,runtime)
    assert digest(plan)==before
    results=[json.loads(e['payload_json']) for e in c.events(cid) if e['kind']=='EXPERIMENT_PLAN_RESULT']
    assert results[-1]['result']['status']=='INCOMPLETE'
    assert results[-1]['result']['scientific_eligibility']['promotion_eligible'] is False
    assert any(e['kind']=='PLAN_ABANDONED' for e in c.events(cid))


def test_recovery_preserves_realistic_retained_archive(registry_lab,monkeypatch):
    from orchestrator import model_registry as registry
    store,c,cid,repo,add=registry_lab
    rid=add();registry.initialize(store,repo);before=registry.states(store)
    c.set_status(cid,'COMPLETED');cid=c.create(simple('previous_best'))
    monkeypatch.setattr(autonomy,'REPO',repo)
    for _ in range(10):c.recoverable_failure(cid,'fixture failure')
    autonomy.recover_research_attempt(store,c,cid,store.path.parent)
    recovery=next(json.loads(e['payload_json']) for e in c.events(cid) if e['kind']=='RECOVERY_RESTORED')
    assert recovery['verified_state']['models'][0]['run_id']==rid
    assert registry.states(store)==before
    for archived in registry.retained(store).values():registry.verify_archive(archived)


def test_newer_verified_state_wins_and_corruption_halts(registry_lab,monkeypatch):
    from orchestrator import model_registry as registry
    from orchestrator.experiment_plan import baseline_reference
    store,c,cid,repo,add=registry_lab
    rid=add();reference,_,_=baseline_reference(store,rid,repo)
    c.set_status(cid,'COMPLETED');cid=c.create(simple('new'))
    c.event(cid,'VERIFIED_RESEARCH_STATE',references=[reference],plan_id='completed-study')
    monkeypatch.setattr(autonomy,'REPO',repo)
    for _ in range(10):c.recoverable_failure(cid,'fixture failure')
    autonomy.recover_research_attempt(store,c,cid,store.path.parent)
    recovery=next(json.loads(e['payload_json']) for e in c.events(cid) if e['kind']=='RECOVERY_RESTORED')
    assert recovery['verified_state']['references']==[reference]
    with store.connect() as con:con.execute("UPDATE runs SET git_sha='invalid' WHERE run_id=?",(rid,))
    for _ in range(10):c.recoverable_failure(cid,'fixture failure')
    with pytest.raises(PermissionError,match='integrity unavailable'):
        autonomy.recover_research_attempt(store,c,cid,store.path.parent)


def test_internal_research_retry_does_not_force_scout_even_with_legacy_preference(lab,monkeypatch):
    store,c,cid,runtime=lab
    cfg=json.loads(c.get(cid)['config_json']);cfg['research'].update(fresh_literature=True,fresh_hypotheses=True)
    with store.connect() as con:con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    scouts=[];prompts=[]
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**kw:scouts.append(1) or {'sources':['verified fixture lead']})
    def propose(*a,**kw):prompts.append(a[1]);raise RuntimeError('transport fixture failure')
    monkeypatch.setattr(controller,'add_research_proposals',propose)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='transport fixture'):autonomy.run_cycle(store,c,cid,runtime)
    assert len(scouts)==0 and len(prompts)==2
    assert all('existing models' in p for p in prompts)


def test_cancellation_kills_owned_subprocess_group_and_retains_diagnostics(tmp_path):
    pidfile=tmp_path/'grandchild'
    child=f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(10)'
    code=f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);print("fixture stderr evidence",file=sys.stderr,flush=True);time.sleep(10)'
    with pytest.raises(subprocess.TimeoutExpired) as stopped:
        codex_runner.run_cancellable([sys.executable,'-c',code],timeout=10,
                                     cancel=lambda:pidfile.exists(),text=True)
    assert b'fixture stderr evidence' in stopped.value.stderr
    import psutil
    try:assert psutil.Process(int(pidfile.read_text())).status()==psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:pass


def test_runner_budget_admission_ignores_obsolete_global_limits(lab,monkeypatch):
    from orchestrator.experiment_runner import execute_run
    from benchmark import data_guard
    store,c,cid,runtime=lab
    historical=store.create_run('old','old','fixture');store.account_gpu_usage(historical,100000)
    for key in ('LOCO_GPU_BUDGET_HOURS','LOCO_JOB_TIMEOUT_SECONDS','LOCO_CODEX_MAX_CALLS'):
        monkeypatch.setenv(key,'obsolete')
    rid=store.create_run('fixture','fixture','fixture',config={'campaign_id':cid})
    class ReachedManifest(Exception):pass
    def stop_before_any_execution(*a,**kw):raise ReachedManifest()
    monkeypatch.setattr(data_guard,'load_manifest',stop_before_any_execution)
    with pytest.raises(ReachedManifest):
        execute_run(store.get_run(rid),store,Path('.'),runtime/'fixture_manifest',runtime)
    assert store.get_run(historical)['gpu_seconds']==100000


def test_verified_implementation_reused_only_with_matching_hash(lab,monkeypatch):
    from orchestrator import experiment_runner
    from plan_fixtures import experiment_plan
    store,c,cid,runtime=lab;source=runtime/'fixture_source.py';source.write_text('# artificial fixture')
    c.event(cid,'IMPLEMENTED',hypothesis_id='fixture',source_sha256=experiment_runner.sha256(source))
    monkeypatch.setattr(experiment_runner,'candidate_file',lambda *a:source)
    monkeypatch.setattr(controller,'implement_proposal',lambda *a,**kw:pytest.fail('validated implementation rerun'))
    autonomy.prepare_plan_candidate(store,c,cid,runtime,experiment_plan('fixture'),{})
    source.write_text('# changed fixture')
    with pytest.raises(PermissionError,match='source integrity changed'):
        autonomy.prepare_plan_candidate(store,c,cid,runtime,experiment_plan('fixture'),{})
