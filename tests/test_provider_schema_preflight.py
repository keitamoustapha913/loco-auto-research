"""Actual generated schemas checked offline; no CLI/model/network/GPU execution."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import validate

from orchestrator import controller
from orchestrator.codex_runner import provider_output_schema,preflight_provider_schema,ProviderSchemaError
from orchestrator.experiment_plan import decode_agent_plan
from orchestrator.planning_contract import constrain_output_schema,inspect_plan,baseline_capability
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.autonomy import load_campaign_config
from test_planning_contract import frozen_plan
from test_model_registry import lab
from plan_fixtures import agent_plan,experiment_plan

REPO=Path(__file__).resolve().parents[1]


def generated(capabilities=()):
    source=json.loads((REPO/'schemas/proposals.schema.json').read_text())
    return constrain_output_schema(source,capabilities)


@pytest.mark.parametrize('single',[False,True])
def test_complete_generated_provider_schema_passes_preflight_with_frozen_branches(lab,single):
    p,store,repo,ref=frozen_plan(lab)
    schema=generated([baseline_capability(store,ref,repo)])
    if single:schema=schema['$defs']['proposal']
    original=copy.deepcopy(schema)
    compiled=provider_output_schema(schema)
    assert preflight_provider_schema(compiled)==compiled
    assert schema==original
    serialized=json.dumps(compiled)
    assert '(?!' not in serialized and '(?<=' not in serialized
    if single:
        validate(agent_plan(p),compiled['properties']['experiment_plan'])


@pytest.mark.parametrize('field',['device','category','seed','plan_id','trusted_source_sha256'])
@pytest.mark.parametrize('location',['common','override'])
def test_real_generated_schema_keeps_trusted_reserved_field_rejection(field,location):
    p=experiment_plan()
    target=p['training'] if location=='common' else p['arms'][1]['training_overrides']
    target[field]='forbidden'
    wire=agent_plan(p)
    schema=provider_output_schema(generated())['$defs']['proposal']['properties']['experiment_plan']
    validate(wire,schema) # A plain provider string is deliberately insufficient.
    report=inspect_plan(decode_agent_plan(wire))
    assert not report['valid']
    assert any(e['code']=='RESERVED_TRAINING_FIELDS' for e in report['errors'])


def test_actual_generated_schema_defers_duplicates_and_scientific_semantics_to_trusted_checks(lab):
    p,store,repo,ref=frozen_plan(lab)
    cap=baseline_capability(store,ref,repo)
    schema=provider_output_schema(generated([cap]))['$defs']['proposal']['properties']['experiment_plan']
    validate(agent_plan(p),schema)
    assert not inspect_plan(p,store,repo)['scientific_eligibility']['promotion_eligible']
    duplicate=copy.deepcopy(p);duplicate['seeds']=[3,3,29]
    validate(agent_plan(duplicate),schema) # uniqueItems omitted for provider only.
    assert not inspect_plan(duplicate,store,repo)['valid']
    prose=copy.deepcopy(p);prose['controlled_variables']=['identical preprocessing']
    validate(agent_plan(prose),schema)
    assert any(e['code']=='CONTROLLED_VARIABLE_KEYS' for e in inspect_plan(prose,store,repo)['errors'])
    # Another known baseline's valid inference option cannot replace this one's.
    inconsistent=copy.deepcopy(p);inconsistent['arms'][0]['inference']={'mode':'alternate'}
    other=dict(cap,recorded_inference={'mode':'alternate'})
    alternate_schema=provider_output_schema(generated([cap,other]))['$defs']['proposal']['properties']['experiment_plan']
    validate(agent_plan(inconsistent),alternate_schema)
    assert any(e['code']=='BASELINE_INFERENCE_NOT_EXACT' for e in inspect_plan(inconsistent,store,repo)['errors'])


@pytest.mark.parametrize('pattern',[r'^(?!device$).*$',r'(?=cuda)',r'(?<=device)cuda',r'(?<!category)cuda',r'(x)\1'])
def test_nested_generated_lookarounds_and_backreferences_are_rejected(pattern):
    schema=generated()
    node=schema['$defs']['proposal']['properties']['experiment_plan']['properties']['arms']['items']
    node['properties']['training_overrides']['items']['properties']['name']['pattern']=pattern
    with pytest.raises(ProviderSchemaError,match='training_overrides.*pattern'):
        provider_output_schema(schema)


@pytest.mark.parametrize('keyword,value',[
    ('allOf',[{'type':'string'}]),('oneOf',[{'type':'string'}]),('not',{'type':'string'}),
    ('if',{'type':'string'}),('then',{'type':'string'}),('else',{'type':'string'}),
    ('dependentRequired',{'x':['y']}),('dependentSchemas',{'x':{'type':'string'}}),
    ('patternProperties',{'x':{'type':'string'}}),('unknownProviderExtension',True)])
def test_unsupported_keywords_fail_in_nested_real_generated_schema(keyword,value):
    schema=generated();node=schema['$defs']['proposal']['properties']['experiment_plan']['properties']['training']['items']
    node[keyword]=value
    with pytest.raises(ProviderSchemaError,match=keyword):provider_output_schema(schema)


def test_literal_values_are_not_mistaken_for_schema_features():
    schema={'type':'object','properties':{'literal':{'const':{'pattern':'(?!x)','uniqueItems':True,'allOf':[]}}},
            'required':['literal'],'additionalProperties':False}
    assert provider_output_schema(schema)==schema


@pytest.mark.parametrize('campaign_mode',[False,True])
def test_preflight_rejection_never_charges_or_retries_agent_call(tmp_path,monkeypatch,campaign_mode):
    store=StateStore(tmp_path/'ledger.sqlite');campaign=None
    if campaign_mode:
        campaigns=CampaignState(store);config=load_campaign_config(REPO/'configs/autonomy.yaml');config['campaign']['start_from']='new';cid=campaigns.create(config)
        campaign=(campaigns,cid)
    # This goes through the actual researcher schema-generation code path.
    schema=generated();schema['$defs']['proposal']['properties']['research_question']['pattern']='(?!bad)'
    path=tmp_path/'invalid.schema.json';path.write_text(json.dumps(schema))
    monkeypatch.setattr(controller,'invoke',lambda *a,**kw:pytest.fail('CLI invocation prohibited'))
    with pytest.raises(ProviderSchemaError,match='lookaround'):
        controller._with_agent_correction(store,campaign,'researcher',
            lambda feedback,attempt:controller._budgeted_invoke_once(store,'researcher',REPO,path,'offline',campaign=campaign),
            lambda output:None)
    assert store.codex_call_count()==0
    assert not any(e['kind'] in ('REPAIR_REQUESTED','VALIDATION_FAILED','CODEX_CALL') for e in store.events())
    if campaign:
        assert campaigns.get(cid)['codex_calls']==0
        assert not any(e['kind'].startswith('AGENT_') for e in campaigns.events(cid))


@pytest.mark.parametrize('change', ['open_object','missing_required','nonlocal_ref','unknown_ref','root_anyof'])
def test_strict_shape_and_reference_errors_are_caught_locally(change):
    schema=generated()
    if change=='open_object':schema['additionalProperties']=True
    elif change=='missing_required':schema['required']=[]
    elif change=='nonlocal_ref':schema['properties']['schema_version']={'$ref':'https://invalid.example/schema'}
    elif change=='unknown_ref':schema['properties']['schema_version']={'$ref':'#/$defs/missing'}
    else:schema={'anyOf':[schema]}
    with pytest.raises(ProviderSchemaError):provider_output_schema(schema)


def test_all_checked_in_agent_output_schemas_pass_available_local_checks():
    for name in ('proposals','experiment_plan_agent','implementation','literature','plan_review','review','ablation_decision'):
        provider_output_schema(json.loads((REPO/'schemas'/f'{name}.schema.json').read_text()))


@pytest.mark.parametrize('name',['',' prose setting','setting with spaces','setting\tname'])
def test_plain_provider_names_cannot_escape_trusted_key_validation(name):
    p=experiment_plan();p['training'][name]='invalid'
    assert any(e['code']=='INVALID_TRAINING_FIELD_NAMES' for e in inspect_plan(p)['errors'])


def test_valid_generated_plan_remains_eligible_only_with_matched_evidence_requirements():
    from orchestrator.model_registry import SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY
    p=experiment_plan();p['evaluation']['real_development_feedback']=True
    p['required_measurements'] += [{'source':'real','key':k} for k in (SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY)]
    compiled=provider_output_schema(generated())['$defs']['proposal']['properties']['experiment_plan']
    wire=agent_plan(p);validate(wire,compiled)
    report=inspect_plan(decode_agent_plan(wire))
    assert report['valid'] and report['scientific_eligibility']['promotion_eligible']
    p['seeds']=[3,17]
    assert not inspect_plan(p)['scientific_eligibility']['promotion_eligible']
