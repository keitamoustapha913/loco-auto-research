"""Offline research routing: no live agents, checkpoint execution or GPU work."""
import copy,json
from pathlib import Path
import pytest
from jsonschema import validate
from orchestrator import controller,autonomy
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.codex_runner import provider_output_schema
from orchestrator.planning_contract import constrain_output_schema
from orchestrator.experiment_plan import investigation_observations,baseline_reference
from test_research_loop_interfaces import proposal
from test_model_registry import lab as model_lab
from test_campaign_recovery import lab


def batch():return [proposal(i) for i in (1,2,3)]


def test_generated_request_schema_passes_actual_provider_preflight():
    schema=constrain_output_schema(json.loads((controller.REPO/'schemas/proposals.schema.json').read_text()),research_request=True)
    compiled=provider_output_schema(schema)
    validate({'schema_version':'1.0','literature_request':'Which published evidence addresses the measured gap?','proposals':[]},compiled)
    assert compiled['properties']['proposals']['minItems']==0
    assert 'literature_request' in compiled['required']
    # Legacy/public plan contracts are not changed to accept empty experiments.
    original=constrain_output_schema(json.loads((controller.REPO/'schemas/proposals.schema.json').read_text()))
    assert original['properties']['proposals']['minItems']==3


def test_internal_same_family_distinct_questions_need_no_scout(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite');items=batch()
    for p in items:p['candidate_family']='same_existing_family';p['references']=['run:internal-observation']
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**kw:{'response':{'literature_request':'','proposals':items}})
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**kw:pytest.fail('no external search requested'))
    accepted=controller.add_research_proposals(store,'Investigate existing training',personal_mode=True)
    assert len(accepted)==3 and {p['candidate_family'] for p in accepted}=={'same_existing_family'}
    with pytest.raises(ValueError,match='duplicate'):controller.add_research_proposals(store,'repeat',personal_mode=True)


def test_researcher_requests_targeted_scout_only_when_useful(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite');prompts=[];queries=[]
    outputs=iter([{'literature_request':'Resolve the observed representation-collapse failure','proposals':[]},
                  {'literature_request':'','proposals':batch()}])
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**kw:prompts.append(a[4]) or {'response':next(outputs)})
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**kw:queries.append(kw['question']) or {'sources':['unverified fixture source']})
    controller.add_research_proposals(store,'Research independently',personal_mode=True)
    assert queries==['Resolve the observed representation-collapse failure']
    assert len(prompts)==2 and 'unverified fixture source' in prompts[1]


def test_requested_scout_findings_survive_transport_failure_and_restart(lab,monkeypatch):
    store,c,cid,_=lab;queries=[];created=c.get(cid)['created_at'];calls=[0]
    def invoke(*a,**kw):
        calls[0]+=1
        if calls[0] in (1,3):return {'response':{'literature_request':'A specific knowledge gap','proposals':[]}}
        if calls[0]==2:raise RuntimeError('transport fixture')
        return {'response':{'literature_request':'','proposals':batch()}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**kw:queries.append(kw['question']) or {'sources':['cached source']})
    with pytest.raises(RuntimeError,match='transport fixture'):
        controller.add_research_proposals(store,'Investigate',personal_mode=True,campaign=(c,cid))
    restarted=CampaignState(StateStore(store.path))
    controller.add_research_proposals(store,'Investigate',personal_mode=True,campaign=(restarted,cid))
    assert queries==['A specific knowledge gap'] and restarted.get(cid)['created_at']==created
    assert len([e for e in restarted.events(cid) if e['kind']=='REQUESTED_LITERATURE_COMPLETED'])==1


def test_disabled_scout_is_not_bypassed(lab,monkeypatch):
    store,c,cid,_=lab;cfg=json.loads(c.get(cid)['config_json']);cfg['research']['literature_scout']=False
    with store.connect() as con:con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**kw:{'response':{'literature_request':'External gap','proposals':[]}})
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**kw:pytest.fail('disabled Scout must not run'))
    with pytest.raises(Exception,match='Scout is disabled'):
        controller.add_research_proposals(store,'Investigate',personal_mode=True,campaign=(c,cid))
    assert controller.list_proposals(store)==[]


def test_model_investigation_observations_redact_real_case_identities(model_lab):
    store,_,_,_,add=model_lab;rid=add();row=store.get_run(rid);folder=Path(row['artifact_dir'])
    feedback=__import__('orchestrator.development_feedback',fromlist=['latest_summary']).latest_summary(store,rid)
    (folder/'training_timing.json').write_text(json.dumps({'duration_seconds':900,'completed_steps':100,'status':'COMPLETED'}))
    (folder/'development_case_metrics.json').write_text(json.dumps({'cases':[{'primary_ap':.4,'source_id':'normal-source'}]}))
    Path(feedback['artifact_dir'],'case_scores.json').write_text(json.dumps([
        {'category':'good','image_score':.1,'source_sha256':'private-development-case'},
        {'category':'logical_anomalies','image_score':.8,'path':'private-image.png','masks':['private-mask.png']},
        {'category':'structural_anomalies','image_score':.7}]))
    observed=investigation_observations(store,row,feedback);encoded=json.dumps(observed)
    assert 'private' not in encoded
    assert observed['artifacts']['training_timing.json']['duration_seconds']==900
    assert observed['artifacts']['development_case_metrics.json']['per_image_measurements']==[{'primary_ap':.4}]
    assert observed['real_development_aggregate_diagnostics']['logical_anomalies']['count']==1
    assert 'No recorded loss/gradient trajectory' in observed['unavailable'][0]
    target=folder/'training_audit.json';target.unlink(missing_ok=True);target.symlink_to(folder/'checkpoint/model.pt')
    with pytest.raises(PermissionError,match='unsafe research observation'):
        investigation_observations(store,row,feedback)


def test_cuda_learning_rule_allows_genuinely_frozen_analysis(model_lab,monkeypatch):
    store,_,_,repo,add=model_lab;rid=add();ref,_,_=baseline_reference(store,rid,repo)
    monkeypatch.setattr(controller,'REPO',repo)
    items=batch()
    for p in items:
        plan=__import__('plan_fixtures').experiment_plan(p['hypothesis_id'],seeds=(3,),arms=('control',))
        plan['baselines']=[ref]
        plan['arms'][0].update(execution='frozen',baseline_by_seed={'3':ref['baseline_id']})
        p['experiment_plan']=__import__('plan_fixtures').agent_plan(plan)
        p['estimated_gpu_minutes']=0;p['evaluation_plan']='Frozen readout; no learning'
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**kw:{'response':{'literature_request':'','proposals':items}})
    accepted=controller.add_research_proposals(store,'Analyze frozen models',personal_mode=True,require_gpu_pilot=True)
    assert all(a['execution']=='frozen' for p in accepted for a in p['experiment_plan']['arms'])


def test_recorded_normal_learning_curves_are_available_not_invented(model_lab):
    store,_,_,_,add=model_lab;row=store.get_run(add());folder=Path(row['artifact_dir'])
    (folder/'training_audit.json').write_text(json.dumps({'optimizer_updates':3,'loss_curve':[1.,.7,.4],'gradient_norms':[.8,.6,.3],'untrusted_path':'private-fixture'}))
    observed=investigation_observations(store,row,None)
    assert observed['artifacts']['training_audit.json']['loss_curve']['samples']==[1.,.7,.4]
    assert observed['artifacts']['training_audit.json']['gradient_norms']['samples']==[.8,.6,.3]
    assert not observed['unavailable'] and 'private-fixture' not in json.dumps(observed)


def test_invalid_optional_literature_cache_does_not_block_internal_research(lab,monkeypatch,tmp_path):
    _,campaigns,cid,_=lab
    (tmp_path/'literature').mkdir();(tmp_path/'literature/scout_notes.json').write_text('{invalid fixture')
    monkeypatch.setattr(autonomy,'REPO',tmp_path)
    assert autonomy.existing_literature(campaigns,cid)=={}
    assert any(e['kind']=='LITERATURE_CACHE_UNAVAILABLE' for e in campaigns.events(cid))
