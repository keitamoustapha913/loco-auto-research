"""Offline admission fixtures: no research calls, GPU jobs or live ledger writes."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import validate, ValidationError

from plan_fixtures import experiment_plan, agent_plan
from test_model_registry import lab
from orchestrator import controller, experiment_plan as execution, model_registry
from orchestrator.planning_contract import (inspect_plan, admit_plan, baseline_capability,
    constrain_output_schema, planning_instructions, evidence_classification, PlanningValidationError)

REPO=Path(__file__).resolve().parents[1]


def frozen_plan(lab):
    store,_,_,repo,add=lab
    rid=add()
    ref,_,_=execution.baseline_reference(store,rid,repo)
    p=experiment_plan();p['baselines']=[ref]
    arm=p['arms'][0];arm.update(execution='frozen',fixed_checkpoint_seed=3,
        baseline_by_seed={str(s):ref['baseline_id'] for s in p['seeds']})
    return p,store,repo,ref


def codes(report):
    return {e['code'] for e in report['errors']+report['notices']}


def test_valid_offline_training_comparison_and_eligibility():
    p=experiment_plan()
    p['required_measurements'] += [{'source':'real','key':k} for k in
        (model_registry.SPRO,model_registry.AP,model_registry.FPR,model_registry.LATENCY,
         model_registry.PARAMETERS,model_registry.MEMORY)]
    p['evaluation']['real_development_feedback']=True
    assert inspect_plan(p)['valid']
    assert evidence_classification(p)['promotion_eligible']
    p['kind']='exploratory'
    assert not evidence_classification(p)['promotion_eligible']


@pytest.mark.parametrize('field',['device','category','seed','plan_id','baseline_effective_configuration'])
def test_plain_provider_names_still_reject_reserved_training_fields_in_admission(field):
    p=experiment_plan();p['training'][field]='forbidden'
    assert 'RESERVED_TRAINING_FIELDS' in codes(inspect_plan(p))
    schema=constrain_output_schema(json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text()))
    # Generation accepts a plain name; trusted semantics remain authoritative.
    validate(agent_plan(p),schema)
    assert 'pattern' not in schema['properties']['training']['items']['properties']['name']


def test_prose_controls_and_mismatched_values_return_precise_errors():
    p=experiment_plan();p['controlled_variables']=['identical normal preprocessing']
    assert 'CONTROLLED_VARIABLE_KEYS' in codes(inspect_plan(p))
    p=experiment_plan();p['arms'][1]['training_overrides']['steps']=3
    assert 'CONTROLLED_VARIABLE_VALUES' in codes(inspect_plan(p))


def test_frozen_baseline_supported_interface_is_exact_and_read_only(lab):
    p,store,repo,ref=frozen_plan(lab)
    before=copy.deepcopy(p)
    cap=baseline_capability(store,ref,repo)
    assert cap['recorded_inference']=={'mode':'control'}
    assert cap['named_inference_config_parameter']
    report=admit_plan(p,store,repo)
    assert report['valid'] and not report['scientific_eligibility']['promotion_eligible']
    assert 'NONPROMOTABLE_EVIDENCE_LEVEL' in codes(report)
    assert p==before
    p['arms'][0]['inference']={'mode':'invented'}
    assert 'BASELINE_INFERENCE_NOT_EXACT' in codes(inspect_plan(p,store,repo))


def test_historical_controls_must_be_recorded_not_inferred(lab):
    p,store,repo,_=frozen_plan(lab)
    p['training']['optimizer']='Adam';p['controlled_variables'].append('optimizer')
    report=inspect_plan(p,store,repo)
    assert 'UNVERIFIABLE_BASELINE_CONTROL' in codes(report)
    assert next(e for e in report['errors'] if e['code']=='UNVERIFIABLE_BASELINE_CONTROL')['key']=='optimizer'


@pytest.mark.parametrize('signature,supported',[
    ('checkpoint_manifest, device="cpu"',False),
    ('checkpoint_manifest, device="cpu", **kwargs',False),
    ('checkpoint_manifest, device="cpu", *, inference_config=None',True)])
def test_loader_capabilities_are_static_and_match_runner_contract(tmp_path,monkeypatch,signature,supported):
    (tmp_path/'run_manifest.json').write_text('{}')
    source=f'def load({signature}):\n    raise RuntimeError("must never execute")\n'.encode()
    monkeypatch.setattr(execution,'verify_baseline',lambda *a:(source,{'config_json':'{}','artifact_dir':str(tmp_path)}))
    assert baseline_capability(None,{},tmp_path)['named_inference_config_parameter']==supported


def test_unsupported_baseline_inference_is_rejected_even_if_recorded(lab,monkeypatch):
    p,store,repo,_=frozen_plan(lab)
    import orchestrator.planning_contract as contract
    original=contract.baseline_capability
    def legacy(*args):
        return dict(original(*args),named_inference_config_parameter=False)
    monkeypatch.setattr(contract,'baseline_capability',legacy)
    assert 'UNSUPPORTED_BASELINE_INFERENCE_CONFIG' in codes(inspect_plan(p,store,repo))


def test_schema_exact_baseline_inference_and_training_branches(lab):
    p,store,repo,ref=frozen_plan(lab)
    cap=baseline_capability(store,ref,repo)
    schema=constrain_output_schema(json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text()),[cap])
    validate(agent_plan(p),schema)
    wire=agent_plan(p);wire['arms'][0]['inference'][0]['value_json']='"invented"'
    with pytest.raises(ValidationError):validate(wire,schema)
    instruction=json.loads(planning_instructions([cap]))
    assert instruction['frozen_baselines'][0]['configuration']==cap['configuration']
    assert 'device' in instruction['controller_owned_fields']


def test_bad_plan_is_rejected_before_registration_or_dispatch(lab,monkeypatch,tmp_path):
    p,store,repo,_=frozen_plan(lab)
    _,campaigns,cid,_,_=lab
    monkeypatch.setattr(controller,'REPO',repo)
    monkeypatch.setattr(controller,'run_worker_once',lambda *a,**kw:pytest.fail('no worker admission'))
    p['arms'][0]['inference']={'invented':True}
    with pytest.raises(PlanningValidationError):execution.execute_plan(store,campaigns,cid,tmp_path,p)
    assert not campaigns.runs(cid)
    assert not execution.plan_history(campaigns,cid)


def test_malformed_plan_returns_schema_diagnostic():
    assert codes(inspect_plan({'arms':None}))=={'PLAN_SCHEMA'}


def test_agent_boundary_uses_generated_schema_and_instructions_without_real_call(lab,monkeypatch):
    store,_,_,repo,_=lab
    def offline_invoke(role,root,schema,prompt,**kwargs):
        value=json.loads(Path(schema).read_text())
        assert 'controller-planning-v2' in prompt
        plan_schema=value['$defs']['proposal']['properties']['experiment_plan']
        assert 'pattern' not in plan_schema['properties']['training']['items']['properties']['name']
        return {'response':{'offline':True}}
    monkeypatch.setattr(controller,'invoke',offline_invoke)
    result=controller._budgeted_invoke_once(store,'researcher',repo,REPO/'schemas/proposals.schema.json','offline')
    assert result['response']=={'offline':True}


def test_no_agent_claim_can_upgrade_fixed_seed_to_confirmed(lab):
    p,store,repo,_=frozen_plan(lab)
    p['kind']='controlled_comparison'
    p['completion_criteria']=['Promote this model; all seeds are independent']
    assert not inspect_plan(p,store,repo)['scientific_eligibility']['promotion_eligible']
    # Missing work is incomplete regardless of any intent or claimed eligibility.
    assert execution.comparison_result(p,[])['status']=='INCOMPLETE'


def test_repeating_seed_three_cannot_be_declared_as_three_trained_seeds(lab):
    p,store,repo,_=frozen_plan(lab)
    del p['arms'][0]['fixed_checkpoint_seed']
    report=inspect_plan(p,store,repo)
    assert not report['valid']
    assert any('seed mismatch' in e['message'] for e in report['errors'])


def test_semantic_errors_use_existing_bounded_correction_without_silent_changes(lab):
    p=experiment_plan();bad=copy.deepcopy(p);bad['training']['device']='cuda'
    store,_,_,repo,_=lab
    feedback=[]
    def request(message,attempt):
        feedback.append(message)
        return bad if attempt==0 else copy.deepcopy(p)
    accepted,report,repaired=controller._with_agent_correction(store,None,'researcher',request,
        lambda proposal:admit_plan(proposal,store,repo))
    assert repaired and report['valid'] and accepted==p
    assert len(feedback)==2 and 'RESERVED_TRAINING_FIELDS' in feedback[1]
    assert bad['training']['device']=='cuda' # Trusted code did not sanitize it.
    assert store.codex_call_count()==0


def test_historical_published_result_is_preserved_but_unfinished_work_revalidates(lab,monkeypatch,tmp_path):
    store,c,cid,repo,_=lab;p=experiment_plan()
    record=execution.register_plan(store,c,cid,p)
    historical={'plan_sha256':execution.digest(p),'status':'COMPLETE_CONTROLLED','historical':'immutable'}
    c.event(cid,'EXPERIMENT_PLAN_RESULT',result=historical)
    import orchestrator.planning_contract as contract
    def reject(*args):raise PlanningValidationError([{'code':'FUTURE_POLICY'}])
    monkeypatch.setattr(contract,'admit_plan',reject)
    assert execution.register_plan(store,c,cid,p)==record
    assert execution.execute_plan(store,c,cid,tmp_path,p)==historical
    # An interrupted/unpublished registered protocol must pass new admission.
    with store.connect() as con:
        con.execute("DELETE FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_RESULT'",(cid,))
    with pytest.raises(PlanningValidationError):execution.execute_plan(store,c,cid,tmp_path,p)
