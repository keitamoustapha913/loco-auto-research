"""Deterministic local processes only: no Codex, CUDA, or development inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from orchestrator import codex_runner,controller,autonomy
from orchestrator.agent_diagnostics import save,prune
from orchestrator.campaign_config import normalize,AGENT_MINUTES
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from test_time_campaigns import simple
from test_experiment_plan import laboratory
from plan_fixtures import experiment_plan


@pytest.mark.parametrize('value',[0,-1,31,True,'30',float('nan')])
def test_invalid_agent_cap(value):
    config=simple();config['limits'][AGENT_MINUTES]=value
    with pytest.raises(ValueError,match=AGENT_MINUTES):normalize(config)


@pytest.mark.parametrize('role',['scout','researcher','implementer','critic'])
@pytest.mark.parametrize('remaining',[25,7200])
def test_each_role_gets_earlier_deadline(tmp_path,monkeypatch,role,remaining):
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple())
    with store.connect() as con:
        con.execute('UPDATE campaigns SET created_at=? WHERE id=?',(time.time()-14400+remaining,cid))
    got={}
    def fake(*args,**kw):got.update(kw);return {'response':{}}
    monkeypatch.setattr(controller,'invoke',fake)
    created=c.get(cid)['created_at']
    controller._budgeted_invoke_once(store,role,Path('.'),None,'fixture',campaign=(c,cid),timeout=1)
    assert min(1800,remaining)-2 < got['timeout']<=min(1800,remaining)
    assert got['deadline']==created+14400 and len(got['invocation_id'])==32
    assert c.get(cid)['created_at']==created


def test_successful_process_progress_and_bounded_output():
    observed={}
    done=codex_runner.run_cancellable([sys.executable,'-c',
        'import json;print(json.dumps({"type":"turn.completed"}));print("x"*200000)'],
        timeout=3,text=True,observation=observed)
    assert done.returncode==0 and len(done.stdout)<=65536
    assert observed['progress_events'][0]['type']=='turn.completed'
    assert observed['output_bytes']['stdout']>200000 and observed['termination_reason']=='completed'


@pytest.mark.parametrize('reason',['call_deadline','campaign_deadline'])
def test_alive_nonreturning_process_is_reaped_with_partial_evidence(tmp_path,reason):
    observation={};pidfile=tmp_path/'pid'
    code=f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));print("partial",flush=True);time.sleep(20)'
    deadline=time.time()+.3 if reason=='campaign_deadline' else None
    with pytest.raises(subprocess.TimeoutExpired):
        codex_runner.run_cancellable([sys.executable,'-c',code],timeout=.3 if deadline is None else 20,
                                    deadline=deadline,text=True,observation=observation)
    assert observation['termination_reason']==reason and 'partial' in observation['stdout']
    assert observation['exit_code'] is not None and observation['last_output_at'] is not None
    with pytest.raises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)


def test_structured_terminal_failure_recovers_early_not_silence():
    evidence={};start=time.monotonic()
    with pytest.raises(codex_runner.AgentFailure,match='turn failed'):
        codex_runner.run_cancellable([sys.executable,'-c',
            'import time;print(\'{"type":"turn.failed","error":{"message":"fixture"}}\',flush=True);time.sleep(20)'],
            timeout=10,text=True,observation=evidence)
    assert time.monotonic()-start<2 and evidence['termination_reason']=='structured_terminal_failure'


def test_invocation_saves_failure_evidence_before_temp_cleanup(tmp_path,monkeypatch):
    from orchestrator import codex_sandbox
    schema=tmp_path/'schema.json';schema.write_text('{"type":"object","properties":{},"required":[],"additionalProperties":false}')
    output={}
    def command(root,role,provider,response,*args,**kw):
        output['temporary']=response.parent
        return [sys.executable,'-c','import sys,time;print("partial stdout",flush=True);print("Bearer secret_token",file=sys.stderr,flush=True);time.sleep(20)']
    monkeypatch.setattr(codex_sandbox,'command',command)
    with pytest.raises(codex_runner.AgentTimedOut):
        codex_runner.invoke('critic',Path(__file__).parents[1],schema,'fixture',personal_mode=True,
                            timeout=.3,diagnostics_dir=tmp_path/'diagnostics')
    record=json.loads(next((tmp_path/'diagnostics').glob('*.json')).read_text())
    assert record['role']=='critic' and record['finished_at']>=record['started_at']
    assert 'partial stdout' in record['stdout'] and 'secret_token' not in json.dumps(record)
    assert not record['response_file_created'] and not output['temporary'].exists()
    assert record['traceback'] and record['exit_code'] is not None


def test_transcript_retention_and_symlink_protection(tmp_path):
    for i in range(131):save(tmp_path,f'{i:032x}',{'stderr':'api_key=privatevalue'})
    assert len(list(tmp_path.glob('*.json')))==128 and prune(tmp_path)==0
    assert 'privatevalue' not in next(tmp_path.glob('*.json')).read_text()
    link=tmp_path/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(PermissionError):save(link,'a'*32,{})
    stale=tmp_path/('b'*32+'.tmp');stale.write_text('interrupted publication')
    os.utime(stale,(time.time()-7200,time.time()-7200))
    active=tmp_path/('c'*32+'.tmp');active.write_text('active publication')
    assert prune(tmp_path)>0 and not stale.exists() and active.exists()


def test_deadline_reaps_owned_descendants_even_if_they_ignore_sigterm(tmp_path):
    import psutil
    pid=tmp_path/'child'
    child='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'
    code=f'import subprocess,sys,time;p=subprocess.Popen([sys.executable,"-c",{child!r}]);open({str(pid)!r},"w").write(str(p.pid));time.sleep(20)'
    with pytest.raises(subprocess.TimeoutExpired):
        codex_runner.run_cancellable([sys.executable,'-c',code],timeout=.4)
    child_pid=int(pid.read_text())
    for _ in range(50):
        if not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status()==psutil.STATUS_ZOMBIE:break
        time.sleep(.01)
    assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status()==psutil.STATUS_ZOMBIE


def test_completed_review_survives_retry_exhaustion_and_restart(laboratory,monkeypatch):
    from orchestrator import experiment_plan as ep
    store,c,cid,runtime,repo,calls,_=laboratory;p=experiment_plan()
    result=ep.execute_plan(store,c,cid,runtime,p);before=json.dumps(result,sort_keys=True)
    jobs=len(calls)
    def timeout(*args,**kw):raise codex_runner.AgentTimedOut('fixture call_deadline')
    monkeypatch.setattr(controller,'_budgeted_invoke_once',timeout)
    with pytest.raises(codex_runner.AgentTimedOut):ep.review_plan(store,c,cid,p,result)
    assert all(r['status']=='COMPLETED' for r in c.runs(cid))
    for _ in range(10):c.recoverable_failure(cid,'review timeout')
    assert autonomy.recover_research_attempt(store,c,cid,runtime)
    assert any(e['kind']=='REVIEW_DEFERRED' for e in c.events(cid))
    assert not any(e['kind']=='PLAN_ABANDONED' for e in c.events(cid))
    pending=ep.pending_reviews(StateStore(store.path),'juice_bottle')
    assert len(pending)==1 and json.dumps(pending[0]['result'],sort_keys=True)==before
    rows,availability=ep.reconstruct_review(store,c,cid,p,result)
    assert len(rows)==6 and all(x['checkpoint_available'] for x in availability)
    assert len(calls)==jobs  # No training or scoring replayed.


def test_cross_campaign_review_uses_original_rows_and_does_not_execute_jobs(laboratory,monkeypatch):
    from orchestrator import experiment_plan as ep
    store,c,cid,runtime,repo,calls,_=laboratory;p=experiment_plan()
    result=ep.execute_plan(store,c,cid,runtime,p);c.set_status(cid,'COMPLETED')
    import orchestrator.campaign_config as configuration
    monkeypatch.setattr(configuration,'starting_context',lambda *a,**kw:{'models':[],'start_from':'previous_best'})
    cfg=simple('previous_best');fresh=c.create(cfg);captured=[]
    def review(*args,**kw):captured.append((args[3],args[4],kw))
    monkeypatch.setattr(ep,'review_plan',review)
    monkeypatch.setattr(ep,'execute_plan',lambda *a:pytest.fail('completed evidence must not execute again'))
    autonomy.run_cycle(store,c,fresh,runtime)
    assert captured[0][0]==p and captured[0][1]==result
    assert captured[0][2]['evidence_campaign_id']==cid and len(calls)==6
