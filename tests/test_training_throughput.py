"""Deterministic CPU throughput/accounting fixtures. Never launch CUDA or agents."""
import copy
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.train_adapter import DecodedTrainingCache
from orchestrator.campaign_config import (normalize,training_execution_estimate,
    throughput_ratio,remaining_plan_work)
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.experiment_plan import register_plan,digest
from scripts.training_phase import TrainingStep,run_training_phases,TARGET_VERSION
from plan_fixtures import experiment_plan
from test_experiment_plan import laboratory


def test_decoded_pixels_identical_immutable_and_reader_runs_once():
    cache=DecodedTrainingCache();calls=[]
    original=np.arange(48,dtype=np.float32).reshape(3,4,4)/255
    def read():calls.append(1);return original.copy()
    a=cache.get(('source','verified-sha',128),read)
    b=cache.get(('source','verified-sha',128),read)
    assert len(calls)==1 and a is b and np.array_equal(a,original)
    with pytest.raises(ValueError):a.setflags(write=True)
    batch=np.stack([a,b]);batch[:]=0
    assert np.array_equal(cache.get(('source','verified-sha',128),read),original)
    cache.get(('source','changed-sha',128),read)
    assert len(calls)==2


def test_cache_is_bounded_and_preserves_indexed_stochastic_batches():
    capacity=3*3*4*4*4;cache=DecodedTrainingCache(capacity)
    originals={i:np.full((3,4,4),i,dtype=np.float32) for i in range(5)}
    for seed in (367,373):
        for index in range(12):
            order=np.random.default_rng(np.random.SeedSequence([seed,index,11])).integers(0,5,8)
            old=np.stack([originals[i] for i in order])
            new=np.stack([cache.get(int(i),lambda i=i:originals[i]) for i in order])
            assert np.array_equal(old,new) and cache.bytes<=capacity
            # Cache misses/hits do not consume the scientific random stream.
            for arm in ('full','random','censored'):
                rng=np.random.default_rng(np.random.SeedSequence([seed,index,23]))
                mask=rng.integers(0,2,old.shape,dtype=np.uint8)
                assert np.array_equal(old*mask,new*mask)
    large=DecodedTrainingCache(1);large.get('large',lambda:originals[0])
    assert large.bytes==0 and not large.entries


def profile(ratio=2,source='new',device='cuda'):
    return dict(version='training-throughput-v1',sampled_updates=28,
        sampled_learning_seconds=1.,sampled_step_wall_seconds=ratio,
        initial_preparation_seconds=2.,source_sha256=source,device=device)


@pytest.mark.parametrize('field,value',[
    ('sampled_updates',2),('sampled_updates',True),('sampled_learning_seconds',0),
    ('sampled_learning_seconds',math.nan),('sampled_step_wall_seconds',math.inf),
    ('sampled_step_wall_seconds',.5),('version','agent-claimed')])
def test_invalid_or_two_step_profiles_cannot_admit_long_training(field,value):
    p=profile();p[field]=value
    with pytest.raises(ValueError):throughput_ratio(p)


def config():
    return normalize({'campaign':{'name':'fixture','dataset':'juice_bottle','gpu_device':'cuda','start_from':'new'},
        'limits':{'research_hours':8,'training_minutes_per_experiment':15},'research':{}})


def test_cancelled_historical_run_scales_to_full_target_and_new_sources_require_preflight(tmp_path):
    store=StateStore(tmp_path/'fixture.sqlite');rid=store.create_run('fixture','fixture','juice_bottle',config={
        'execution_kind':'train','device':'cuda','model_sha256':'old'})
    with store.connect() as c:
        c.execute('UPDATE runs SET status=?,started_at=?,finished_at=?,metrics_json=? WHERE run_id=?',
            ('CANCELLED',1.,27983.15626605414,json.dumps({'training/duration_seconds':237.78479909570888}),rid))
    before=store.get_run(rid)
    estimated=training_execution_estimate(store,config(),jobs=6,newly_trained=6,source_sha256='old',critique=True)
    assert estimated['required_seconds']>8*3600 and estimated['execution_to_learning_ratio']>100
    new=training_execution_estimate(store,config(),jobs=6,newly_trained=6,source_sha256='new')
    assert not new['throughput_observed'] and 'queued throughput admission required' in new['basis']
    measured=training_execution_estimate(store,config(),jobs=6,newly_trained=6,source_sha256='new',throughput=profile())
    assert measured['required_seconds']==6*900*2.5+6*240
    assert measured['training_seconds_per_model']==900 and store.get_run(rid)==before
    # Scope follows the real device; a CPU profile cannot certify CUDA feasibility.
    with pytest.raises(ValueError,match='source/device'):
        training_execution_estimate(store,config(),source_sha256='new',throughput=profile(device='cpu'))
    with pytest.raises(ValueError,match='source/device'):
        training_execution_estimate(store,config(),source_sha256='wrong',throughput=profile())
    mismatched=profile();mismatched.update(batch_size=1,size=128)
    with pytest.raises(ValueError,match='batch size/resolution'):
        training_execution_estimate(store,config(),source_sha256='new',throughput=mismatched,
            training_settings={'batch_size':8,'size':128})


def test_remaining_work_counts_all_arms_and_seeds_not_just_current_job(tmp_path):
    store=StateStore(tmp_path/'fixture.sqlite');campaigns=CampaignState(store);cid=campaigns.create(config())
    p=experiment_plan();register_plan(store,campaigns,cid,p)
    effective={'campaign_id':cid,'plan_id':p['plan_id'],'plan_sha256':digest(p),'plan_arm':p['arms'][0]['arm_id'],'seed':p['seeds'][0]}
    assert remaining_plan_work(store,effective)=={'jobs':6,'newly_trained':6}
    rid=store.create_run('fixture','fixture','juice_bottle',config=effective)
    with store.connect() as con:con.execute("UPDATE runs SET status='COMPLETED' WHERE run_id=?",(rid,))
    assert remaining_plan_work(store,effective)=={'jobs':5,'newly_trained':5}
    effective['plan_id']='not-registered'
    with pytest.raises(ValueError,match='registered complete'):remaining_plan_work(store,effective)


def test_profiling_keeps_preparation_and_integrity_outside_learning(tmp_path):
    def steps(config,manifest,output):
        class Loss(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,x):return (self.p*x).square().mean()
        loss=Loss();opt=torch.optim.SGD(loss.parameters(),lr=.001)
        while True:
            time.sleep(.002)  # Fixture setup only; never inside the learning phase.
            progress=yield TrainingStep(loss,opt,(torch.ones(4),),source_exposures=4)
            if progress['target_reached']:break
        time.sleep(.002)
        return {}
    run_training_phases(SimpleNamespace(training_steps=steps),{'device':'cpu','training_completion_policy':TARGET_VERSION},
        tmp_path,tmp_path/'checkpoint',budget=.02)
    timing=json.loads((tmp_path/'training_timing.json').read_text());p=timing['execution_profile']
    assert timing['duration_seconds']>=.02 and timing['verified']
    assert p['costs']['preparation_seconds']>.002 and p['finalization_seconds']>=.002
    assert p['sampled_step_wall_seconds']>p['sampled_learning_seconds']>0
    assert p['costs']['integrity_seconds']>0 and p['costs']['audit_io_seconds']>0


def test_impossible_six_cell_plan_is_recoverable_and_queues_nothing(laboratory,monkeypatch):
    from test_required_training import prepare_matrix,plan3
    from orchestrator.experiment_plan import execute_plan
    from orchestrator.experiment_runner import sha256
    store,campaigns,cid,runtime,calls=prepare_matrix(laboratory,monkeypatch,minutes=15)
    source=runtime/'repo/candidates/fixture/model.py'
    rid=store.create_run('old','fixture','juice_bottle',config={'device':'cuda','model_sha256':sha256(source)})
    with store.connect() as con:
        con.execute("UPDATE runs SET status='CANCELLED',started_at=1,finished_at=3001,metrics_json=? WHERE run_id=?",
            (json.dumps({'training/duration_seconds':10}),rid))
    result=execute_plan(store,campaigns,cid,runtime,plan3())
    assert result['status']=='INCOMPLETE' and not calls
    assert campaigns.get(cid)['status']=='RUNNING'
    assert any('required-duration matrix' in x for x in result['problems'])


@pytest.mark.parametrize('version',['3.0','4.0'])
def test_disposable_throughput_probe_uses_pilot_batch_and_full_training_split(tmp_path,monkeypatch,version):
    from scripts import candidate_smoke,training_phase
    from scripts.demo_run import make_fixture
    from benchmark.data_guard import build_manifest
    make_fixture(tmp_path/'good');manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(build_manifest(tmp_path/'good')))
    module=tmp_path/'model.py';module.write_text('# deterministic offline fixture, not CUDA evidence')
    captured=[]
    def learning(module,config,manifest,output,**kwargs):
        entries=json.loads(manifest.read_text())['images'];captured.append((config,entries,kwargs))
        output.mkdir();(output/'model.pt').write_bytes(b'dummy')
        (output/'checkpoint_manifest.json').write_text('{}')
        (output.parent/'training_timing.json').write_text(json.dumps({'execution_profile':profile(),'loop_elapsed_seconds':1.}))
    class Predictor:
        def predict(self,x):
            return {'anomaly_maps':np.zeros((len(x),1,x.shape[-2],x.shape[-1]),np.float32),'image_scores':np.zeros(len(x))}
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:1)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(training_phase,'run_training_phases',learning)
    monkeypatch.setattr(candidate_smoke,'load_candidate',lambda *a:SimpleNamespace(load=lambda *a,**k:Predictor()))
    cfg={'device':'cuda','category':'juice_bottle','size':128,'batch_size':8,'seed':367,
        'training_budget_seconds':900,'plan_contract_version':'experiment-plan-v'+version}
    result=candidate_smoke.run(module,manifest,cfg,tmp_path/'smoke')
    effective,entries,kwargs=captured[0]
    assert len(entries)==4 and all(e['split']=='train/good' for e in entries)
    assert effective['batch_size']==8 and effective['size']==128 and effective['seed']==367
    assert kwargs=={'budget':None,'preflight_steps':32}
    assert result['training_throughput']['batch_size']==8 and result['optimizer_step_request']==32


@pytest.mark.parametrize('bad',['symlink','parent_symlink','oversized'])
def test_host_throughput_summary_cannot_escape_assigned_output(tmp_path,bad):
    from orchestrator.experiment_runner import read_training_throughput
    job=tmp_path/'job';job.mkdir();outside=tmp_path/'private';outside.mkdir()
    (outside/'summary.json').write_text('{}')
    if bad=='parent_symlink':(job/'cuda_smoke').symlink_to(outside,target_is_directory=True)
    else:
        (job/'cuda_smoke').mkdir();p=job/'cuda_smoke/summary.json'
        if bad=='symlink':p.symlink_to(outside/'summary.json')
        else:p.write_text(' '*16385)
    with pytest.raises(ValueError,match='unsafe or oversized'):read_training_throughput(job)
