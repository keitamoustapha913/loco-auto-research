"""Recorded campaign replay and deterministic prompt plumbing, never live AI."""
import copy
import json
from pathlib import Path

import pytest

from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.model_registry import research_decisions, scientific_memory, DECISION_GUIDANCE
from test_model_registry import lab as model_lab


@pytest.fixture
def replay(tmp_path):
    fixture=Path(__file__).parent/'fixtures/research_decisions_campaign.json'
    data=json.loads(fixture.read_text());store=StateStore(tmp_path/'controller/experiments.sqlite');CampaignState(store)
    for p in data['proposals']:store.add_proposal(p)
    for row in data['runs']:
        rid=store.create_run('recorded','recorded',row['category'],config=row['config'],run_id=row['run_id'])
        with store.connect() as con:
            con.execute("UPDATE runs SET status='REVIEWED',started_at=?,finished_at=? WHERE run_id=?",(row['started_at'],row['finished_at'],rid))
    with store.connect() as con:
        for row in data['events']:
            con.execute('INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (data['campaign_id'],row['time'],row['kind'],json.dumps(row['payload'])))
    return store,data,fixture


def test_recorded_campaign_replay_surfaces_repeat_cost_and_tested_explanations(replay):
    store,data,fixture=replay;before=fixture.read_bytes()
    context=research_decisions(store)
    stats=context['campaign_activity'][data['campaign_id']]
    assert stats['new_training_runs']==0 and stats['frozen_runs']==15
    assert stats['recorded_job_seconds']==pytest.approx(3615.504630088806)
    assert stats['reviewed_questions']==5
    history=context['investigated_questions']
    assert len(history)==6 and history[-1]['execution_status']=='INCOMPLETE'
    rounding=next(x for x in history if 'rounding' in x['hypothesis_id'])
    assert rounding['hypothesis_outcome']=='rejected' and '3.6800e-08' in rounding['finding']
    assert rounding['suggested_next_investigation']
    groups=context['checkpoint_series'];assert len(groups)==1
    assert len(groups[0]['studies'])==6 and len(groups[0]['checkpoint_hashes'])==3
    assert 'Compare the marginal value' in groups[0]['reconsideration']
    assert 'not a diagnostic limit' in groups[0]['reconsideration']
    assert fixture.read_bytes()==before


def test_memory_exposes_whole_decision_sequence_without_mutating_records(replay):
    store,data,_=replay
    with store.connect() as con:
        before=[tuple(r) for r in con.execute('SELECT * FROM campaign_events')]
        settings=[tuple(r) for r in con.execute('SELECT * FROM settings')]
    memory=scientific_memory(store)
    assert len(memory['research_decisions']['investigated_questions'])==6
    with store.connect() as con:
        assert [tuple(r) for r in con.execute('SELECT * FROM campaign_events')]==before
        assert [tuple(r) for r in con.execute('SELECT * FROM settings')]==settings


def test_renamed_study_same_checkpoints_cannot_hide_previous_findings(replay):
    store,data,_=replay
    copied=copy.deepcopy(data['events'][0]['payload']['plan']);copied['plan_id']='new_name_same_checkpoints';copied['hypothesis_id']='new_alias'
    with store.connect() as con:
        con.execute('INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (data['campaign_id'],99,'EXPERIMENT_PLAN_REGISTERED',json.dumps({'plan':copied})))
    context=research_decisions(store)
    assert any(x['hypothesis_id']=='new_alias' for x in context['checkpoint_series'][0]['studies'])
    assert context['checkpoint_series'][0]['reconsideration']


def test_planner_and_critic_guidance_requires_action_branches_without_quotas():
    for token in ('different model/training/scoring','each plausible outcome','same next action',
                  'total elapsed research cost','real logical pixel localization','no diagnostic quota',
                  'diagnostic-only campaigns remain permitted','no architecture family is preferred'):
        assert token in DECISION_GUIDANCE
    repo=Path(__file__).parents[1]
    for role in ('researcher','critic'):
        text=(repo/f'agents/{role}.md').read_text()
        assert 'advisory' in text
        assert ('outcome-to-action' if role=='critic' else 'outcome-to-decision') in text


def test_proposals_receive_decision_guidance_without_new_schema(tmp_path,monkeypatch):
    from orchestrator import controller
    from test_internal_research import batch
    prompts=[];proposals=batch()
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**k:prompts.append(a[4]) or {'response':{'literature_request':'','proposals':proposals}})
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**k:pytest.fail('external literature is optional'))
    accepted=controller.add_research_proposals(StateStore(tmp_path/'ledger.sqlite'),'Choose research',personal_mode=True)
    assert len(accepted)==3 and all(any(a['execution']=='train' for a in p['experiment_plan']['arms']) for p in accepted)
    assert DECISION_GUIDANCE in prompts[0]
    assert 'existing fields rather than a new contract' in prompts[0]


@pytest.mark.parametrize('fresh_campaign',[False,True])
def test_actual_agent_boundary_mounts_whole_history_context_offline(replay,monkeypatch,fresh_campaign):
    from orchestrator import controller,experiment_plan,model_registry
    store,data,_=replay;seen=[]
    monkeypatch.setattr(model_registry,'initialize',lambda *a:None)
    monkeypatch.setattr(model_registry,'recover_comparisons',lambda *a:None)
    monkeypatch.setattr(experiment_plan,'research_evidence',lambda *a:([],{}))
    def offline_invoke(role,root,schema,prompt,**kw):
        seen.append(prompt)
        memory=json.loads(kw['reference_files']['cumulative/evidence.json'].read_text())
        assert len(memory['research_decisions']['investigated_questions'])==6
        assert memory['research_decisions']['campaign_activity'][data['campaign_id']]['frozen_runs']==15
        assert 'SCIENTIFIC DECISION CHECK' in prompt and 'advisory' in prompt
        return {'response':{'offline_fixture':True}}
    monkeypatch.setattr(controller,'invoke',offline_invoke)
    context=None
    if fresh_campaign:
        from test_time_campaigns import simple
        campaigns=CampaignState(store);context=(campaigns,campaigns.create(simple('new')))
    controller._budgeted_invoke_once(store,'researcher',Path(__file__).parents[1],None,'offline boundary test',personal_mode=True,campaign=context)
    assert len(seen)==1
    assert 'dense_sigmoid_rounding_dice_gap_v1' in seen[0]


def test_complete_comparison_does_not_end_campaign(tmp_path,monkeypatch):
    from orchestrator import autonomy,storage_cleanup
    from test_time_campaigns import simple
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store);cid=campaigns.create(simple('new'))
    created=campaigns.get(cid)['created_at'];cycles=[]
    monkeypatch.setattr(autonomy,'preflight',lambda *a:None)
    monkeypatch.setattr(autonomy,'reconcile_unfinished_runs',lambda *a:None)
    monkeypatch.setattr(autonomy,'recover_research_attempt',lambda *a:False)
    monkeypatch.setattr(autonomy,'stop_reason',lambda *a:(None,''))
    monkeypatch.setattr(storage_cleanup,'maybe_cleanup',lambda *a,**k:None)
    def cycle(*a):
        cycles.append('comparison complete')
        if len(cycles)==2:campaigns.set_status(cid,'COMPLETED','offline fixture done')
    monkeypatch.setattr(autonomy,'run_cycle',cycle)
    autonomy.run_forever(cid,tmp_path,store.path,idle_seconds=0)
    assert len(cycles)==2 and campaigns.get(cid)['created_at']==created


def test_valuable_frozen_studies_remain_permitted_without_training_quota(model_lab,monkeypatch):
    from orchestrator import controller,experiment_plan
    from test_internal_research import batch
    from plan_fixtures import experiment_plan as plan_fixture,agent_plan
    store,_,_,repo,add=model_lab;rid=add();reference,_,_=experiment_plan.baseline_reference(store,rid,repo)
    monkeypatch.setattr(controller,'REPO',repo)
    items=batch()
    for proposal in items:
        plan=plan_fixture(proposal['hypothesis_id'],seeds=(3,),arms=('control',))
        plan['baselines']=[reference]
        plan['arms'][0].update(execution='frozen',baseline_by_seed={'3':reference['baseline_id']})
        proposal.update(experiment_plan=agent_plan(plan),estimated_gpu_minutes=0,
            evaluation_plan='If coverage is retained, revise readout; if coverage is lost, abandon this readout. Reuse frozen weights.')
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**k:{'response':{'literature_request':'','proposals':items}})
    accepted=controller.add_research_proposals(store,'Decision-changing frozen investigations',personal_mode=True,require_gpu_pilot=True)
    assert all(a['execution']=='frozen' for p in accepted for a in p['experiment_plan']['arms'])
    experiment_plan.verify_baseline(store,reference,repo)


def test_saved_nested_diagnostics_are_identified_without_claiming_analysis_complete(model_lab):
    from orchestrator.experiment_plan import investigation_observations
    store,_,_,_,add=model_lab;rid=add();row=store.get_run(rid);folder=Path(row['artifact_dir'])
    (folder/'diagnostic_existing.json').write_text(json.dumps({'metrics':{},'per_image':[],
        'evidence':{'variant_records':[{'source_id':'do-not-export-record-values','query_sha256':'opaque'}],
                    'factorial_summaries':[{'side':16,'metrics':{}}],
                    'joint_analysis_status':'INCOMPLETE until trusted cross-arm joins'}}))
    observed=investigation_observations(store,row,None)
    inventory=observed['artifacts']['diagnostic_existing.json']['saved_evidence_inventory']
    assert inventory['variant_records']['recorded_count']==1
    assert 'query_sha256' in inventory['variant_records']['record_fields']
    assert inventory['joint_analysis_status']=={'declared_status':'INCOMPLETE','verified_analysis':False}
    assert 'do-not-export-record-values' not in json.dumps(observed)


def test_implementer_is_not_instructed_to_change_the_scientific_direction(replay,monkeypatch):
    from orchestrator import controller,experiment_plan,model_registry
    store,_,_=replay;prompts=[]
    monkeypatch.setattr(model_registry,'initialize',lambda *a:None)
    monkeypatch.setattr(model_registry,'recover_comparisons',lambda *a:None)
    monkeypatch.setattr(experiment_plan,'research_evidence',lambda *a:([],{}))
    monkeypatch.setattr(controller,'invoke',lambda *a,**k:prompts.append(a[3]) or {'response':{}})
    controller._budgeted_invoke_once(store,'implementer',Path(__file__).parents[1],None,'Execute frozen plan',personal_mode=True)
    assert 'SCIENTIFIC DECISION CHECK' not in prompts[0]
    assert 'Implement the exact approved scientific plan' in prompts[0]
