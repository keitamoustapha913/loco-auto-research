"""New elapsed-loop semantics, with scaled CPU fixtures; no CUDA experiments."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jsonschema import validate

from scripts.training_phase import (TrainingStep,TrainingLoopStart,LOOP_VERSION,TARGET_VERSION,
    TrainingClock,TrainingTargetIncomplete,TrainingTimeUnverifiable,TrainingBudgetExceeded,run_training_phases)
from orchestrator.campaign_config import training_execution_estimate
from orchestrator.experiment_plan import execute_plan,comparison_result
from orchestrator.planning_contract import constrain_output_schema
from test_experiment_plan import laboratory
from test_required_training import plan3,prepare_matrix
from test_training_throughput import config,profile
from orchestrator.state_store import StateStore
from plan_fixtures import agent_plan

REPO=Path(__file__).parents[1]


def timed_module(now, *, finite=False, marker=True, idle=False, prep=.4):
    def steps(config,manifest,output):
        class Objective(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,x):
                loss=(self.p*x).square().mean();now[0]+=.1;return loss
        model=Objective();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        now[0]+=.5  # One-time model/setup fixture cost, before the explicit boundary.
        if marker:yield TrainingLoopStart(model)
        updates=0
        while True:
            if idle:time.sleep(.001)
            now[0]+=prep  # Deterministic fixture for actual in-loop batch work.
            progress=yield TrainingStep(model,optimizer,(torch.ones(2),),source_exposures=2)
            updates+=1
            if finite or progress['target_reached']:break
        now[0]+=.7  # Final saving cost, after the stopping update.
        return {'updates':updates,'weight':model.p.item()}
    return SimpleNamespace(training_steps=steps)


def loop_config(target=.9):
    return {'device':'cpu','training_completion_policy':LOOP_VERSION,'training_budget_seconds':target}


def test_loop_counts_first_batch_preparation_and_learning_not_setup_saving_or_evaluation(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    meta,duration=run_training_phases(timed_module(now),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.9)
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert meta['updates']==2 and meta['weight']<.5
    assert duration==pytest.approx(1.) and audit['learning_seconds']==pytest.approx(.2)
    assert audit['loop_elapsed_seconds']==duration and audit['duration_seconds']==duration
    assert audit['overshoot_seconds']==pytest.approx(.1)
    assert audit['overshoot_seconds']<=audit['last_iteration_seconds']
    assert audit['completion_policy']==LOOP_VERSION and audit['verified']
    assert now[0]==pytest.approx(.5+1+.7)
    before=(tmp_path/'training_timing.json').read_bytes();now[0]+=.9
    assert (tmp_path/'training_timing.json').read_bytes()==before  # Post-training evaluation cannot change it.


def test_no_finite_schedule_or_sleep_can_manufacture_success(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    with pytest.raises(TrainingTargetIncomplete,match='prematurely'):
        run_training_phases(timed_module(now,finite=True),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.9)
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['duration_seconds']==pytest.approx(.5) and not audit['verified']
    with pytest.raises(TrainingTimeUnverifiable,match='sleep'):
        run_training_phases(timed_module(now,idle=True),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.9)


def test_missing_explicit_boundary_fails_closed(tmp_path):
    with pytest.raises(TrainingTimeUnverifiable,match='TrainingLoopStart'):
        run_training_phases(timed_module([0.],marker=False),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.02)


def test_original_campaign_deadline_includes_initialization_and_preparation(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0]);monkeypatch.setattr(time,'time',lambda:now[0])
    cfg=loop_config();cfg['research_deadline']=.8
    with pytest.raises(TrainingBudgetExceeded,match='research deadline'):
        run_training_phases(timed_module(now),cfg,tmp_path,tmp_path/'checkpoint',budget=.9)
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['completed_steps']==0 and not audit['verified']


def test_disposable_preflight_never_spends_full_training_target(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    meta,duration=run_training_phases(timed_module(now),loop_config(900),tmp_path,tmp_path/'checkpoint',preflight_steps=2)
    assert meta['updates']==2 and duration==pytest.approx(1.)


def test_new_provider_schema_and_old_semantics_remain_separate():
    plan=plan3();plan['schema_version']='4.0'
    schema=constrain_output_schema(json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text()))
    validate(agent_plan(plan),schema)
    old=plan3();before=copy.deepcopy(old)
    from orchestrator.experiment_plan import validate_plan
    validate_plan(old);assert before==old
    assert schema['properties']['schema_version']['const']=='4.0'


def test_loop_matrix_preserves_all_arms_seeds_and_rejects_missing_evidence(laboratory,monkeypatch):
    store,campaigns,cid,runtime,calls=prepare_matrix(laboratory,monkeypatch)
    source=runtime/'repo/candidates/fixture/model.py'
    source.write_text(source.read_text().replace(TARGET_VERSION,LOOP_VERSION))
    original=store.set_run_result
    def record(run_id,metrics,*args,**kw):
        m=dict(metrics)
        m.update({'training/loop_elapsed_seconds':m['training/duration_seconds'],'training/timing_verified':1})
        return original(run_id,m,*args,**kw)
    monkeypatch.setattr(store,'set_run_result',record)
    p=plan3();p['schema_version']='4.0';result=execute_plan(store,campaigns,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert [(x['plan_arm'],x['seed']) for x in calls]==[(a,s) for a in ('control','intervention') for s in (3,17,29)]
    assert all(x['training_completion_policy']==LOOP_VERSION for x in calls)
    rows=copy.deepcopy(campaigns.runs(cid));m=json.loads(rows[0]['metrics_json']);m.pop('training/loop_elapsed_seconds');rows[0]['metrics_json']=json.dumps(m)
    assert comparison_result(p,rows)['status']=='INCOMPLETE'


def test_six_loop_targets_are_90_minutes_not_scaled_by_old_gpu_arithmetic(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite');cfg=config();cfg['limits']['research_hours']=4
    p=profile(ratio=118);p.update(batch_size=8,size=128)
    estimate=training_execution_estimate(store,cfg,jobs=6,newly_trained=6,source_sha256='new',throughput=p,loop_timed=True,critique=True)
    assert estimate['training_seconds_per_model']==900
    assert estimate['projected_training_execution_seconds_per_model']==900
    assert estimate['required_seconds']==5400+6*240+90 and estimate['required_seconds']<4*3600


def test_dashboard_distinguishes_new_loop_time_from_unchanged_historical_measurements(tmp_path,monkeypatch):
    from dashboard.campaign_timing import training_seconds,campaign_training_seconds
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    old={'run_id':'old','config':{'campaign_id':'c'},'metrics':{'training/duration_seconds':237.7848}}
    new={'run_id':'new','config':{'campaign_id':'c'},'metrics':{'training/loop_elapsed_seconds':900,'training/learning_seconds':237}}
    before=copy.deepcopy(old)
    assert training_seconds(old) is None and training_seconds(new)==900
    assert campaign_training_seconds({'id':'c'},[old,new])==900 and old==before
    live={'run_id':'live','status':'RUNNING','config':{'campaign_id':'c','training_completion_policy':LOOP_VERSION},'metrics':{}}
    folder=tmp_path/'jobs/live';folder.mkdir(parents=True)
    (folder/'training_timing.json').write_text(json.dumps({'completion_policy':LOOP_VERSION,'loop_elapsed_seconds':120}))
    assert training_seconds(live)==120
    live['run_id']='../private';assert training_seconds(live) is None


def test_cuda_sync_accounting_keeps_transfers_in_loop_and_learning_completion_in_computation(tmp_path,monkeypatch):
    now=[10.];monkeypatch.setattr(time,'perf_counter',lambda:now[0]);calls=[]
    def synchronize():calls.append(1);now[0]+=.2
    clock=TrainingClock(tmp_path/'timing.json',1,synchronize,target=True,loop=True)
    clock.loop_started=now[0]
    def real_operation_fixture():now[0]+=.1
    clock.measure(real_operation_fixture)
    assert len(calls)==2
    assert clock.duration()==pytest.approx(.5) and clock.elapsed==pytest.approx(.3)
    assert clock.costs['setup_sync_seconds']==pytest.approx(.2)
    assert clock.costs['learning_sync_seconds']==pytest.approx(.2)


@pytest.mark.parametrize('kind',['model','optimizer'])
def test_initialization_cannot_be_misclassified_as_loop_work(tmp_path,kind):
    def provider(*args):
        model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        yield TrainingLoopStart(model)
        if kind=='model':torch.nn.Linear(1,1)
        else:torch.optim.SGD(model.parameters(),lr=.01)
        yield TrainingStep(model,optimizer,(torch.ones(1,1),),source_exposures=1)
    with pytest.raises(TrainingTimeUnverifiable,match='before TrainingLoopStart'):
        run_training_phases(SimpleNamespace(training_steps=provider),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.02)
