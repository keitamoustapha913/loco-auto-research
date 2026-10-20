import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from plan_fixtures import experiment_plan
from orchestrator import controller
from orchestrator.autonomy import load_campaign_config, choose_proposal
from orchestrator.campaign_state import CampaignState
from orchestrator.experiment_plan import (validate_plan, execute_plan, comparison_result, register_plan,
    digest, baseline_reference, verify_baseline, execution_source, review_plan, research_evidence, decode_agent_plan)
from orchestrator.state_store import StateStore
from benchmark.data_guard import build_manifest
from scripts.demo_run import make_fixture

REPO=Path(__file__).resolve().parents[1]


def test_agent_wire_schema_is_closed_and_round_trips_all_scientific_settings():
    from jsonschema import validate
    from plan_fixtures import agent_plan
    p=experiment_plan();p['training']['optimizer']={'name':'SGD','lr':.01}
    wire=agent_plan(p)
    schema=json.loads((REPO/'schemas/experiment_plan_agent.schema.json').read_text())
    validate(wire,schema)
    assert decode_agent_plan(wire)==p
    assert json.loads((REPO/'schemas/proposals.schema.json').read_text())['$defs']['proposal']['properties']['experiment_plan']==schema
    def check(value):
        if isinstance(value,dict):
            if value.get('type')=='object':assert value['additionalProperties'] is False
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    check(schema)


@pytest.fixture
def laboratory(tmp_path, monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    candidate=repo/'candidates/fixture';candidate.mkdir(parents=True)
    (candidate/'model.py').write_text('# committed fixture implementation\ndef training_steps(*args):\n    raise RuntimeError("fixture protocol only; never executed")\n')
    for args in (['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@localhost','commit','-qm','fixture']):
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    make_fixture(tmp_path/'good')
    (tmp_path/'good_manifest.json').write_text(json.dumps(build_manifest(tmp_path/'good')))
    store=StateStore(tmp_path/'controller/experiments.sqlite');campaigns=CampaignState(store)
    cfg=load_campaign_config(REPO/'configs/autonomy.yaml');cfg['research']['real_development_feedback']=False
    cfg['campaign']['start_from']='new'
    cfg['limits']['max_experiments']=20
    cid=campaigns.create(cfg);store.add_proposal({'hypothesis_id':'fixture'})
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path));monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    monkeypatch.setattr(controller,'REPO',repo)
    calls=[]
    def worker(*args,**kwargs):
        rid=kwargs['expected_run_id'];r=store.get_run(rid);c=json.loads(r['config_json']);calls.append(c)
        store.transition(rid,'RUNNING')
        folder=tmp_path/'jobs'/rid;cp=folder/'checkpoint';cp.mkdir(parents=True)
        weights=b'fixture-frozen-weights'
        (cp/'model.pt').write_bytes(weights);(cp/'checkpoint_manifest.json').write_text('{}')
        (folder/'run_manifest.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(weights).hexdigest()}))
        frozen=c['execution_kind']=='frozen'
        value=float(c['seed'])+(c['plan_arm']=='intervention')
        (folder/'diagnostic_fixture_check.json').write_text(json.dumps({'metrics':{'mean_score':value},
            'per_image':[{'source_id':f'fixture-source-{i}','metrics':{'mean_score':value}} for i in range(3)]}))
        m={'normal/image_false_positives':0,'diagnostic/fixture_check/mean_score':value,
           'protocol/optimizer_updates':0 if frozen else c['expected_optimizer_updates'],
           'protocol/source_exposures':0 if frozen else c['expected_source_exposures'], 'protocol/seed':c['seed']}
        store.set_run_result(rid,m,gpu_seconds=.1,artifact_dir=str(folder));store.transition(rid,'COMPLETED')
        return {'run_id':rid,'status':'COMPLETED'}
    monkeypatch.setattr(controller,'run_worker_once',worker)
    return store,campaigns,cid,tmp_path,repo,calls,worker


def test_two_arms_three_seeds_dispatch_exactly_six_jobs_and_complete(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory;p=experiment_plan()
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert [(x['plan_arm'],x['seed']) for x in calls]==[(a,s) for a in ('control','intervention') for s in (3,17,29)]
    assert all(x['epochs']==4 and x['steps']==2 and x['size']==16 for x in calls)
    assert all(x['inference']=={'mode':x['plan_arm']} and x['diagnostics']==p['diagnostics'] for x in calls)
    assert len(result['paired_seed_differences'])==6
    assert all((runtime/'jobs'/r['run_id']/'plan_comparison.json').is_file() for r in c.runs(cid))
    # Reopening both ledger and campaign does not duplicate any successful cell.
    recovered=CampaignState(StateStore(store.path))
    again=execute_plan(recovered.store,recovered,cid,runtime,p)
    assert again['status']=='COMPLETE_CONTROLLED' and len(calls)==6 and len(recovered.runs(cid))==6


@pytest.mark.parametrize('missing',['arm','diagnostic','setting','duplicate'])
def test_missing_or_invalid_coverage_is_incomplete(laboratory,missing):
    store,c,cid,runtime,_,_,_=laboratory;p=experiment_plan();execute_plan(store,c,cid,runtime,p)
    rows=copy.deepcopy(c.runs(cid))
    if missing=='arm':rows=rows[:3]
    elif missing=='diagnostic':
        m=json.loads(rows[-1]['metrics_json']);m.pop('diagnostic/fixture_check/mean_score');rows[-1]['metrics_json']=json.dumps(m)
    elif missing=='setting':
        cfg=json.loads(rows[-1]['config_json']);cfg['steps']=1;rows[-1]['config_json']=json.dumps(cfg)
    else:rows.append(rows[-1])
    result=comparison_result(p,rows)
    assert result['status']=='INCOMPLETE' and result['problems']


def test_failed_cell_preserved_and_not_replayed(laboratory,monkeypatch):
    store,c,cid,runtime,_,calls,normal=laboratory;p=experiment_plan()
    def fail(*a,**kw):
        r=store.get_run(kw['expected_run_id']);cfg=json.loads(r['config_json'])
        if cfg['plan_arm']=='intervention' and cfg['seed']==17:
            calls.append(cfg);store.transition(r['run_id'],'RUNNING');store.account_gpu_usage(r['run_id'],.2)
            store.transition(r['run_id'],'FAILED',reason='EXACT fixture diagnostic failure')
            return {'run_id':r['run_id'],'status':'FAILED'}
        return normal(*a,**kw)
    monkeypatch.setattr(controller,'run_worker_once',fail)
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='INCOMPLETE' and result['recorded_jobs']==6
    assert any('EXACT fixture diagnostic failure' in x for x in result['problems'])
    execute_plan(store,c,cid,runtime,p)
    assert len(calls)==6


def test_plan_is_immutable_and_respects_authorization(laboratory):
    store,c,cid,runtime,_,calls,_=laboratory;p=experiment_plan();register_plan(store,c,cid,p)
    changed=copy.deepcopy(p);changed['seeds']=[1,2,3]
    with pytest.raises(ValueError,match='immutable'):register_plan(store,c,cid,changed)
    with store.connect() as con:
        cfg=json.loads(c.get(cid)['config_json']);cfg['limits']['max_experiments']=5
        con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    assert execute_plan(store,c,cid,runtime,p)['status']=='INCOMPLETE' and not calls


def test_changing_order_during_comparison_stops_before_mixing_assessment_sources(laboratory,monkeypatch):
    store,c,cid,runtime,_,calls,normal=laboratory;p=experiment_plan()
    def change(*a,**kw):
        result=normal(*a,**kw)
        if len(calls)==1:
            path=runtime/'good_manifest.json';m=json.loads(path.read_text());m['images'].reverse();path.write_text(json.dumps(m))
        return result
    monkeypatch.setattr(controller,'run_worker_once',change)
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='INCOMPLETE' and len(calls)==1
    assert any('source order changed' in x for x in result['problems'])
    assert execute_plan(store,c,cid,runtime,p)==result and len(calls)==1


def test_job_admission_is_atomic_under_concurrent_campaign_count_limit(laboratory):
    from concurrent.futures import ThreadPoolExecutor
    from orchestrator.budgets import BudgetExceeded
    store,c,cid,_,_,_,_=laboratory
    cfg=json.loads(c.get(cid)['config_json']);cfg['limits']['max_experiments']=1
    with store.connect() as con:con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    def reserve(index):
        try:return store.create_run('fixture','fixture','juice_bottle',config={'campaign_id':cid,'plan_sha256':'a'*64})
        except BudgetExceeded:return None
    with ThreadPoolExecutor(max_workers=4) as workers:ids=list(workers.map(reserve,range(4)))
    assert sum(bool(r) for r in ids)==1 and len(c.runs(cid))==1
    # Ownership survives a crash between creation and the caller's attach step.
    reopened=CampaignState(StateStore(store.path));assert len(reopened.runs(cid))==1


def test_explicit_followup_allows_completion_without_renaming_away_the_parent(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite')
    parent={'hypothesis_id':'parent','claim':'Same falsifiable scientific mechanism under a fixed protocol',
            'controlled_change':'Only the declared intervention changes'}
    store.add_proposal(parent)
    child=dict(parent,hypothesis_id='child',priority_rank=1,new_dependencies=[],candidate_family='same_family',
               estimated_gpu_minutes=1,evaluation_plan='Permitted good-only controls',followup_of='parent',
               followup_kind='complete_incomplete',experiment_plan=experiment_plan('child'))
    controller._validate_research_proposals(store,[child],1,False)
    child['followup_kind']='new';child['followup_of']=''
    with pytest.raises(ValueError,match='duplicate'):controller._validate_research_proposals(store,[child],1,False)


def test_legacy_default_queue_cannot_silently_drop_a_structured_plan(laboratory):
    store,c,cid,runtime,_,_,_=laboratory
    with store.connect() as con:
        con.execute('UPDATE hypotheses SET proposal_json=? WHERE hypothesis_id=?',
                    (json.dumps({'hypothesis_id':'fixture','experiment_plan':experiment_plan()}),'fixture'))
    with pytest.raises(ValueError,match='refusing default queue'):
        controller.queue_candidate(store,'fixture','fixture','juice_bottle','pilot','cuda',runtime/'good_manifest.json',
                                   personal_mode_authorized=True)
    assert store.list_runs()==[]


def test_frozen_arm_uses_exact_same_seed_checkpoint_and_committed_source(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory;p=experiment_plan()
    arm=p['arms'][1];arm.update(execution='frozen',source_arm_id='control')
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    frozen=[r for r in c.runs(cid) if json.loads(r['config_json'])['execution_kind']=='frozen']
    assert len(frozen)==3
    original_source=(repo/'candidates/fixture/model.py').read_text()
    (repo/'candidates/fixture/model.py').write_text('# later unrelated source change\n')
    for row in frozen:
        cfg=json.loads(row['config_json']);assert cfg['frozen_baseline']['seed']==cfg['seed']
        source=execution_source(row,store,runtime,repo)
        assert source.read_text()==original_source
        assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['model_sha256']
        bad=dict(cfg['frozen_baseline'],checkpoint_sha256='0'*64)
        with pytest.raises(ValueError,match='changed'):verify_baseline(store,bad,repo)


def test_history_exposes_committed_code_and_aggregate_evidence_only(laboratory):
    store,c,cid,runtime,repo,_,_=laboratory;p=experiment_plan();execute_plan(store,c,cid,runtime,p)
    context,files=research_evidence(store,repo,requested=[c.runs(cid)[0]['run_id']])
    assert context and files and context[0]['frozen_reference']['source_sha256']
    assert all(Path(x).suffix in {'.py','.json'} for x in files.values())
    assert all(p.stat().st_mode & 0o222==0 for p in files.values())
    # Selection must be scoped, including when old proposals have a better rank.
    store.add_proposal({'hypothesis_id':'old','priority_rank':1})
    assert choose_proposal(store,allowed_ids={'unrelated'}) is None


def test_critic_cannot_certify_incomplete_comparison(laboratory,monkeypatch):
    store,c,cid,runtime,_,_,_=laboratory;p=experiment_plan();result=comparison_result(p,[]);calls=[]
    def critic(*a,**kw):
        calls.append(a[4]);return {'response':{'plan_id':p['plan_id'],'comparison_status':'INCOMPLETE',
            'hypothesis_outcome':'supported' if len(calls)==1 else 'inconclusive','causal_claim':len(calls)==1,
            'recommended_next_action':'ablate','objective_measured_results':'Missing all declared cells.'}}
    monkeypatch.setattr(controller,'_budgeted_invoke_once',critic)
    review=review_plan(store,c,cid,p,result)
    assert len(calls)==2 and not review['causal_claim']
    assert store.list_runs()==[] and c.events(cid)[0]['kind']=='NEXT_DECISION'


def test_paired_image_intervals_use_identical_units_and_stay_separate_from_seed_variation(laboratory):
    store,c,cid,runtime,_,_,_=laboratory;p=experiment_plan()
    p['paired_comparisons']=[{'comparison_id':'fixture_delta','left_arm':'intervention','right_arm':'control',
        'diagnostic_id':'fixture_check','metric':'mean_score','bootstrap_seed':991,'repeats':100}]
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert len(result['paired_image_intervals'])==3
    assert all(r['mean_difference']==1 and r['ci95']==[1,1] and r['unit']=='image' for r in result['paired_image_intervals'])
    last=c.runs(cid)[-1];path=Path(last['artifact_dir'])/'diagnostic_fixture_check.json'
    bad=json.loads(path.read_text());bad['per_image'][0]['source_id']='different-source';path.write_text(json.dumps(bad))
    assert comparison_result(p,c.runs(cid))['status']=='INCOMPLETE'


@pytest.mark.parametrize('mutation',['reserved','controlled','baseline_seed','disabled_real','missing_counts'])
def test_protocol_rejects_silent_substitution(mutation):
    p=experiment_plan()
    if mutation=='reserved':p['training']['device']='cpu'
    elif mutation=='controlled':p['arms'][1]['training_overrides']['steps']=1
    elif mutation=='baseline_seed':p['arms'][1].update(execution='frozen',baseline_by_seed={'3':'absent'})
    elif mutation=='disabled_real':p['required_measurements'].append({'source':'real','key':'dev_real/logical_image_auroc'})
    else:p['training'].pop('expected_optimizer_updates')
    with pytest.raises(ValueError):validate_plan(p)
