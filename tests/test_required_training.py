"""Short CPU/deterministic fixtures only; no agents, CUDA experiments or live data."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch
from jsonschema import validate
from benchmark.train_adapter import validate_diagnostic_output,CandidateContractError
from orchestrator.experiment_plan import validate_plan,execute_plan,comparison_result,baseline_reference,digest
from orchestrator.planning_contract import inspect_plan,constrain_output_schema
from orchestrator.codex_runner import provider_output_schema
from orchestrator.state_store import actionable_error
from orchestrator.experiment_runner import verify_completed_learning
from scripts.training_phase import (TrainingStep,TrainingClock,TrainingTargetIncomplete,TrainingTimeUnverifiable,
                                    run_training_phases,TARGET_VERSION)
from scripts.run_candidate import train_measured
from plan_fixtures import experiment_plan,agent_plan
from test_experiment_plan import laboratory
from test_model_registry import lab as registry_lab

REPO=Path(__file__).parents[1]


def plan3():
    p=experiment_plan();p['schema_version']='3.0';p['confirmation']=False;del p['resource_budget']
    for k in ('steps','epochs','expected_optimizer_updates','expected_source_exposures'):p['training'].pop(k)
    p['controlled_variables']=['batch_size','size']
    return p


def target_module(preparation=0,finite=False,padding=False):
    def provider(config,manifest,output):
        time.sleep(preparation)
        class Objective(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(8,8)*.1)
            def forward(self,batch):
                if padding:time.sleep(.01)
                return (batch@self.weight).sin().square().mean()
        model=Objective();optimizer=torch.optim.SGD(model.parameters(),lr=.0001);updates=0
        while True:
            time.sleep(preparation)
            progress=yield TrainingStep(model,optimizer,(torch.ones(2,8),),source_exposures=2)
            updates+=1
            if finite or progress['target_reached']:break
        time.sleep(preparation)
        return {'updates':updates}
    return SimpleNamespace(training_steps=provider)


def cfg(seconds=.02):
    return {'device':'cpu','training_completion_policy':TARGET_VERSION,'training_budget_seconds':seconds}


def test_actual_cpu_learning_reaches_target_and_excludes_preparation(tmp_path):
    start=time.perf_counter()
    meta,elapsed=train_measured(target_module(preparation=.002),cfg(),tmp_path,tmp_path/'checkpoint')
    wall=time.perf_counter()-start;audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert elapsed>=.02 and audit['verified'] and audit['status']=='COMPLETED'
    assert meta['updates']==audit['completed_steps'] and audit['source_exposures']==2*meta['updates']
    assert wall>elapsed+.004
    assert 0<=audit['overshoot_seconds']<=audit['last_step_seconds']+1e-6


def test_trusted_progress_continues_beyond_short_initial_block(tmp_path,monkeypatch):
    # Real CPU gradients/updates, with a deterministic elapsed-time fixture.
    now=[0.]
    monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    class Objective(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.5))
        def forward(self,x):
            loss=(self.w*x).square().mean();now[0]+=.01;return loss
    def provider(config,manifest,output):
        model=Objective();optimizer=torch.optim.SGD(model.parameters(),lr=.001);count=0
        while True:
            progress=yield TrainingStep(model,optimizer,(torch.ones(2),),source_exposures=2)
            count+=1
            if progress['target_reached']:break
        return {'updates':count}
    result,elapsed=train_measured(SimpleNamespace(training_steps=provider),cfg(.055),tmp_path,tmp_path/'checkpoint')
    assert result['updates']==6 and elapsed==pytest.approx(.06)


def test_premature_or_unverifiable_training_never_succeeds(tmp_path):
    with pytest.raises(TrainingTargetIncomplete,match='prematurely'):
        train_measured(target_module(finite=True),cfg(900),tmp_path,tmp_path/'checkpoint')
    assert json.loads((tmp_path/'training_timing.json').read_text())['verified'] is False
    with pytest.raises(TrainingTimeUnverifiable,match='sleep'):
        train_measured(target_module(padding=True),cfg(),tmp_path,tmp_path/'checkpoint')


def test_cuda_synchronization_is_inside_learning_but_setup_drain_is_excluded(tmp_path,monkeypatch):
    now=[0.];calls=[]
    monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    def synchronize():
        calls.append(1);now[0]+=5 if len(calls)==1 else .02
    clock=TrainingClock(tmp_path/'timing.json',.015,synchronize,target=True)
    clock.measure(lambda:None)
    assert len(calls)==2 and clock.elapsed==pytest.approx(.02)


def test_v3_provider_schema_and_confirmation_preconditions():
    p=plan3();schema=constrain_output_schema(json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text()))
    validate(agent_plan(p),schema);provider_output_schema(schema)
    assert inspect_plan(p)['valid']
    p['training']['steps']=2
    assert not inspect_plan(p)['valid']
    p=plan3();p['confirmation']=True
    report=inspect_plan(p)
    assert any(e['code']=='CONFIRMATION_PREREQUISITES' for e in report['errors'])
    from orchestrator.model_registry import SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY
    p['required_measurements'] += [{'source':'real','key':k} for k in (SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY)]
    p['evaluation']['real_development_feedback']=True
    assert inspect_plan(p)['valid']
    p['seeds']=[3,17]
    assert not inspect_plan(p)['valid']
    p['confirmation']=False
    assert inspect_plan(p)['valid']  # Exploration need not meet confirmation policy.


def prepare_matrix(laboratory,monkeypatch,minutes=.001):
    store,c,cid,runtime,repo,calls,_=laboratory
    source=repo/'candidates/fixture/model.py';source.write_text("TRAINING_PROTOCOL='required-training-target-v1'\n"+source.read_text())
    config=json.loads(c.get(cid)['config_json']);config['limits']['training_minutes_per_experiment']=minutes
    with store.connect() as con:con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(config),cid))
    def worker(*args,**kwargs):
        rid=kwargs['expected_run_id'];r=store.get_run(rid);config=json.loads(r['config_json']);calls.append(config)
        store.transition(rid,'RUNNING');folder=runtime/'jobs'/rid;folder.mkdir(parents=True)
        value=float(config['seed'])+(config['plan_arm']=='intervention')
        metrics={'normal/image_false_positives':0,'diagnostic/fixture_check/mean_score':value,
            'protocol/seed':config['seed'],'protocol/optimizer_updates':12,'protocol/source_exposures':24,
            'training/duration_seconds':config['training_budget_seconds'], 'training/target_reached':1}
        store.set_run_result(rid,metrics,artifact_dir=str(folder));store.transition(rid,'COMPLETED')
        return {'run_id':rid,'status':'COMPLETED'}
    monkeypatch.setattr(__import__('orchestrator.controller',fromlist=['run_worker_once']),'run_worker_once',worker)
    return store,c,cid,runtime,calls


def test_two_arms_three_seeds_each_receive_full_target(laboratory,monkeypatch):
    store,c,cid,runtime,calls=prepare_matrix(laboratory,monkeypatch)
    p=plan3();result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert [(x['plan_arm'],x['seed']) for x in calls]==[(a,s) for a in ('control','intervention') for s in (3,17,29)]
    assert len(calls)==6 and all(x['training_completion_policy']==TARGET_VERSION and x['training_budget_seconds']==.06 for x in calls)
    assert all('steps' not in x and 'expected_optimizer_updates' not in x for x in calls)
    rows=copy.deepcopy(c.runs(cid));m=json.loads(rows[0]['metrics_json']);m['training/duration_seconds']=.01;rows[0]['metrics_json']=json.dumps(m)
    assert comparison_result(p,rows)['status']=='INCOMPLETE'


def test_insufficient_original_campaign_time_admits_no_partial_matrix(laboratory,monkeypatch):
    store,c,cid,runtime,calls=prepare_matrix(laboratory,monkeypatch,minutes=15)
    with store.connect() as con:con.execute('UPDATE campaigns SET created_at=created_at-14000 WHERE id=?',(cid,))
    result=execute_plan(store,c,cid,runtime,plan3())
    assert result['status']=='INCOMPLETE' and not calls and c.get(cid)['status']=='RUNNING'
    assert any('required-duration matrix' in p for p in result['problems'])


def valid_diagnostic():
    declaration={'diagnostic_id':'check','required_metrics':['ap']}
    return declaration,{'metrics':{'ap':.5},'per_image':[{'source_id':'a','metrics':{'ap':.4}}]}


@pytest.mark.parametrize('defect',['unknown','missing','flat','missing_per_image','wrong_ids','nan'])
def test_diagnostic_preflight_payload_rejects_previous_failures(defect):
    d,p=valid_diagnostic()
    if defect=='unknown':p['metrics']['active_classes']=512
    elif defect=='missing':p['metrics']={}
    elif defect=='flat':p['per_image'][0]={'source_id':'a','ap':.4}
    elif defect=='missing_per_image':del p['per_image']
    elif defect=='wrong_ids':p['per_image'][0]['source_id']='wrong'
    elif defect=='nan':p['metrics']['ap']=float('nan')
    with pytest.raises(CandidateContractError):validate_diagnostic_output(d,p,['a'],['ap'])


def test_diagnostic_preflight_accepts_exact_nested_contract():
    d,p=valid_diagnostic();assert validate_diagnostic_output(d,p,['a'],['ap'])==p


def test_actionable_exception_survives_long_training_logs():
    decisive='ValueError: diagnostic returned undeclared metric: check/active_classes'
    full='Candidate FAILED\n'+('optimizer update successful\n'*1000)+'Traceback (most recent call last):\n  File "model.py", line 23, in diagnose\n'+decisive
    message=actionable_error(full)
    assert len(message)<=2000 and decisive in message and 'model.py' in message
    assert message.index(decisive)<message.index('Relevant traceback')


def certified_failed_checkpoint(registry_lab):
    store,c,cid,repo,add=registry_lab;rid=add();row=store.get_run(rid);folder=Path(row['artifact_dir']);config=json.loads(row['config_json'])
    config.update(training_budget_seconds=.05,training_completion_policy=TARGET_VERSION)
    timing={'protocol':'prepared-training-step-v1','completion_policy':TARGET_VERSION,'status':'COMPLETED','verified':True,
            'duration_seconds':.06,'completed_steps':2,'source_exposures':4}
    receipt={'version':'completed-learning-v1','phase':'diagnostics_ready','config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'model_module_sha256':config['model_sha256'],'good_manifest_sha256':row['manifest_sha256'],'training_timing':timing,
        'training_audit':{'seed':config['seed'],'optimizer_updates':2,'observed_optimizer_step_calls':2,'source_exposures':4},
        'checkpoint_sha256':hashlib.sha256((folder/'checkpoint/model.pt').read_bytes()).hexdigest(),
        'checkpoint_manifest_sha256':hashlib.sha256((folder/'checkpoint/checkpoint_manifest.json').read_bytes()).hexdigest()}
    for name,data in [('training_timing.json',timing),('training_receipt.json',receipt),('host_verified_training.json',receipt)]:
        (folder/name).write_text(json.dumps(data))
    with store.connect() as con:
        con.execute('UPDATE runs SET status=?,config_json=? WHERE run_id=?',('FAILED',json.dumps(config),rid))
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',(rid,time.time(),'TRAINING_CHECKPOINT_VERIFIED',json.dumps({'receipt_sha256':digest(receipt),'checkpoint_sha256':receipt['checkpoint_sha256']})))
    return store,repo,rid,folder


def test_exact_failed_diagnostic_checkpoint_can_be_explicitly_reused(registry_lab):
    store,repo,rid,folder=certified_failed_checkpoint(registry_lab)
    before=(folder/'checkpoint/model.pt').read_bytes()
    reference,source,row=baseline_reference(store,rid,repo)
    assert reference['run_id']==rid and row['status']=='FAILED'
    assert (folder/'checkpoint/model.pt').read_bytes()==before
    assert verify_completed_learning(row,folder)['training_audit']['optimizer_updates']==2
    # Never reinterpret the failed experiment as a valid scientific result.
    assert store.get_run(rid)['status']=='FAILED'


@pytest.mark.parametrize('tamper',['checkpoint','config','timing','certification'])
def test_checkpoint_reuse_rejects_integrity_or_incomplete_learning(registry_lab,tamper):
    store,repo,rid,folder=certified_failed_checkpoint(registry_lab)
    if tamper=='checkpoint':(folder/'checkpoint/model.pt').write_bytes(b'changed')
    elif tamper=='config':
        row=store.get_run(rid);config=json.loads(row['config_json']);config['seed']=19
        with store.connect() as con:con.execute('UPDATE runs SET config_json=? WHERE run_id=?',(json.dumps(config),rid))
    elif tamper=='timing':
        timing=json.loads((folder/'training_timing.json').read_text());timing['duration_seconds']=.001;(folder/'training_timing.json').write_text(json.dumps(timing))
    else:
        with store.connect() as con:con.execute("DELETE FROM events WHERE run_id=? AND kind='TRAINING_CHECKPOINT_VERIFIED'",(rid,))
    with pytest.raises(ValueError):baseline_reference(store,rid,repo)


def test_original_research_deadline_includes_preparation(tmp_path,monkeypatch):
    from scripts.training_phase import TrainingBudgetExceeded
    now=[0.];monkeypatch.setattr(time,'time',lambda:now[0])
    def provider(config,manifest,output):
        model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        now[0]=2.  # Preparation exhausts research time without learning time.
        yield TrainingStep(model,optimizer,(torch.ones(1,1),),source_exposures=1)
    config=dict(cfg(),research_deadline=1.)
    with pytest.raises(TrainingBudgetExceeded,match='research deadline'):
        train_measured(SimpleNamespace(training_steps=provider),config,tmp_path,tmp_path/'checkpoint')
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['completed_steps']==0 and audit['duration_seconds']==0 and not audit['verified']


@pytest.mark.parametrize('defect',['unknown','flat','shape'])
def test_disposable_smoke_checks_real_diagnostic_hook_before_full_training(tmp_path,monkeypatch,defect):
    """CUDA functions intercepted; only artificial CPU/NumPy fixture data is used."""
    import numpy as np
    from scripts import candidate_smoke,training_phase
    from scripts.demo_run import make_fixture
    from benchmark.data_guard import build_manifest
    data=tmp_path/'good';make_fixture(data)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(build_manifest(data)))
    calls=[]
    class Predictor:
        def predict(self,images):
            maps=np.zeros((len(images),1,images.shape[-2],images.shape[-1]),dtype=np.float32)
            return {'anomaly_maps':maps,'image_scores':np.zeros(len(images))}
    def learning(module,config,manifest,output,**kwargs):
        calls.append(kwargs);output.mkdir(parents=True)
        (output/'model.pt').write_bytes(b'artificial fixture')
        (output/'checkpoint_manifest.json').write_text('{}')
    def diagnose(predictor,images,config):
        ids=config['assessment_source_ids']
        result={'metrics':{'ap':.5},'per_image':[{'source_id':i,'metrics':{'ap':.5}} for i in ids]}
        if defect=='unknown':result['metrics']['undeclared']=1
        elif defect=='flat':result['per_image'][0]={'source_id':ids[0],'ap':.5}
        else:
            from benchmark.metrics_v2 import pixel_ap_report
            pixel_ap_report(np.zeros((1,16)),np.zeros((1,16)))
        return result
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:1)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(training_phase,'run_training_phases',learning)
    monkeypatch.setattr(candidate_smoke,'load_candidate',lambda *a:SimpleNamespace(load=lambda *a,**k:Predictor(),diagnose=diagnose))
    config=dict(cfg(900),device='cuda',category='juice_bottle',seed=3,size=16,
        plan_contract_version='experiment-plan-v3.0',diagnostic_preflight_required=True,
        diagnostics=[{'diagnostic_id':'check','config':{},'required_metrics':['ap']}],paired_diagnostic_metrics={'check':['ap']})
    with pytest.raises(ValueError):candidate_smoke.run(Path('fixture'),manifest,config,tmp_path/'smoke')
    assert calls==[{'budget':None,'preflight_steps':2}]


def test_host_certifies_post_training_failure_idempotently_without_retraining(registry_lab):
    from orchestrator.experiment_runner import certify_completed_learning
    store,repo,rid,folder=certified_failed_checkpoint(registry_lab)
    (folder/'host_verified_training.json').unlink()
    with store.connect() as con:con.execute("DELETE FROM events WHERE run_id=? AND kind='TRAINING_CHECKPOINT_VERIFIED'",(rid,))
    before=(folder/'checkpoint/model.pt').read_bytes()
    result=certify_completed_learning(store,store.get_run(rid),folder)
    assert certify_completed_learning(store,store.get_run(rid),folder)==result
    assert baseline_reference(store,rid,repo)[0]['checkpoint_sha256']==result['checkpoint_sha256']
    assert store.get_run(rid)['status']=='FAILED' and (folder/'checkpoint/model.pt').read_bytes()==before
    with store.connect() as con:assert con.execute("SELECT count(*) FROM events WHERE run_id=? AND kind='TRAINING_CHECKPOINT_VERIFIED'",(rid,)).fetchone()[0]==1


def test_host_certification_cannot_write_through_a_candidate_symlink(registry_lab,tmp_path):
    from orchestrator.experiment_runner import certify_completed_learning
    store,repo,rid,folder=certified_failed_checkpoint(registry_lab)
    target=tmp_path/'protected_dummy';target.write_text('unchanged')
    (folder/'host_verified_training.json').unlink()
    (folder/'host_verified_training.json').symlink_to(target)
    with pytest.raises(PermissionError):certify_completed_learning(store,store.get_run(rid),folder)
    assert target.read_text()=='unchanged'


def test_generated_instructions_match_new_and_frozen_legacy_contracts():
    from orchestrator.planning_contract import planning_instructions
    new=json.loads(planning_instructions());old=json.loads(planning_instructions(time_only=False))
    assert new['contract']=='controller-planning-v3' and new['required_positive_training_keys']==['batch_size','size']
    assert 'No fixed steps' in new['required_training']
    assert 'Preserve the exact authorized legacy training steps' in old['required_training']


def test_target_plan_cannot_downgrade_to_maximum_only_at_queue_admission(laboratory):
    from orchestrator.experiment_plan import register_plan
    from orchestrator.controller import queue_candidate
    store,c,cid,runtime,repo,_,_=laboratory
    plan=plan3();register_plan(store,c,cid,plan)
    with pytest.raises(ValueError,match='timing contract differs'):
        queue_candidate(store,'fixture','fixture','juice_bottle','pilot','cuda',runtime/'good_manifest.json',
            personal_mode_authorized=True,campaign_id=cid,effective_config={'plan_sha256':digest(plan),
                'plan_contract_version':'experiment-plan-v2.0'})
