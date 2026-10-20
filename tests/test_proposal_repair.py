"""Rejected proposals receive a bounded, budgeted correction without partial writes."""
from pathlib import Path
import pytest
from orchestrator import controller
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.autonomy import load_campaign_config
from orchestrator.budgets import CodexBudgetExhausted


def proposal(hid,claim,family):
    return dict(hypothesis_id=hid,claim=claim,controlled_change=claim,research_question=claim,
                candidate_family=family,priority_rank=1,priority_rationale='Information gain and feasibility',
                new_dependencies=[],estimated_gpu_minutes=1,evaluation_plan='normal-only CUDA pilot',
                experiment_plan=__import__('plan_fixtures').experiment_plan(hid))


def fresh():
    values=[proposal('semantic','semantic categorical context predicts components','tokens'),
            proposal('relations','spatial graph learns pairwise geometric compatibility','graph'),
            proposal('photometric','illumination normalization removes exposure nuisances','normalization')]
    for rank,item in enumerate(values,1):item['priority_rank']=rank
    return values


def campaign(tmp_path,limit=8):
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store)
    config=load_campaign_config(Path(__file__).parents[1]/'configs/autonomy.yaml')
    config['campaign']['start_from']='new'
    config['limits']['max_codex_calls']=limit
    cid=campaigns.create(config)
    return store,campaigns,cid


def test_duplicate_corrected_before_any_partial_insertion(tmp_path,monkeypatch):
    store,campaigns,cid=campaign(tmp_path)
    prior=proposal('old','reconstruct an image from nearby pixels','predictive')
    store.add_proposal(prior)
    bad=fresh();bad[0]=dict(prior)
    prompts=[]
    def invoke(*a,**kw):
        prompts.append(a[3])
        if len(prompts)==1:return {'response':{'proposals':bad}}
        assert 'duplicate research proposal: old' in a[3]
        assert 'renaming the same idea is insufficient' in a[3]
        assert len(controller.list_proposals(store))==1
        return {'response':{'proposals':fresh()}}
    monkeypatch.setattr(controller,'invoke',invoke)
    result=controller.add_research_proposals(store,'propose',campaign=(campaigns,cid))
    assert len(result)==3 and len(prompts)==2
    assert campaigns.usage(cid)['codex_calls']==store.codex_call_count()==2
    assert campaigns.get(cid)['consecutive_failures']==0
    assert len(controller.list_proposals(store))==4
    assert len([e for e in campaigns.events(cid) if e['kind']=='REPAIR_REQUESTED'])==1


def test_renamed_duplicate_and_second_rejection_do_not_loop(tmp_path,monkeypatch):
    store,campaigns,cid=campaign(tmp_path)
    prior=proposal('old','reconstruct an image from nearby pixels','predictive')
    store.add_proposal(prior)
    bad=fresh();bad[0]=dict(prior,hypothesis_id='renamed')
    calls=[]
    def invoke(*a,**kw):
        calls.append(1)
        return {'response':{'proposals':bad}}
    monkeypatch.setattr(controller,'invoke',invoke)
    with pytest.raises(ValueError,match='matches old'):
        controller.add_research_proposals(store,'fixture',campaign=(campaigns,cid))
    assert len(calls)==2 and len(controller.list_proposals(store))==1
    assert campaigns.usage(cid)['codex_calls']==2


def test_repair_cannot_bypass_call_budget(tmp_path,monkeypatch):
    store,campaigns,cid=campaign(tmp_path,1)
    bad=fresh();bad[0]['priority_rank']=2
    calls=[]
    def invoke(*a,**kw):
        calls.append(1)
        return {'response':{'proposals':bad}}
    monkeypatch.setattr(controller,'invoke',invoke)
    with pytest.raises(CodexBudgetExhausted):
        controller.add_research_proposals(store,'fixture',campaign=(campaigns,cid))
    assert len(calls)==1 and store.codex_call_count()==1
    assert campaigns.get(cid)['consecutive_failures']==0
    assert controller.list_proposals(store)==[]
