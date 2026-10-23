"""Research ranking never substitutes for scientific confirmation or deployment."""
import copy
import json
from pathlib import Path
import pytest
from test_model_registry import lab
from orchestrator import model_registry as registry
from orchestrator.campaign_config import starting_context
from orchestrator.campaign_state import CampaignState
from orchestrator.development_feedback import latest_summary


def test_localization_eighty_percent_and_moderate_fpr_not_veto(lab):
    store,_,_,repo,add=lab
    old=add(spro=.25,ap=.2);registry.initialize(store,repo)
    scientific=copy.deepcopy(registry.states(store));snapshot=store.get_run(old)
    high=add(name='challenger',spro=.76,ap=.57,metrics={registry.FPR:.5,'dev_real/normal_false_positives':1,registry.LATENCY:150})
    registry.initialize(store,repo)
    head=next(iter(registry.research_states(store).values()));model=registry.retained(store)[head['retention_id']]
    assert model['run_id']==high
    assert head['selection']['localization_weight']==.8
    assert head['selection']['weighted']['logical_spro']==pytest.approx(.6*.76)
    assert head['selection']['weighted']['logical_pixel_ap']==pytest.approx(.2*.57)
    assert not head['deployment_readiness']['meets_existing_fpr_and_size_targets']
    assert registry.states(store)==scientific and store.get_run(old)==snapshot
    assert head['confirmation']=='EXPLORATORY_RESEARCH_INCUMBENT'


def test_previous_best_selects_research_head_new_does_not_reuse(lab):
    store,c,cid,repo,add=lab
    add(spro=.2);registry.initialize(store,repo)
    best=add(name='challenger',spro=.8,ap=.6);registry.initialize(store,repo)
    cfg=json.loads(c.get(cid)['config_json']);cfg['campaign']['start_from']='previous_best'
    assert starting_context(store,cfg,strict=True)['models'][0]['run_id']==best
    cfg['campaign']['start_from']='new'
    assert starting_context(store,cfg,strict=True)['models']==[]
    before=registry.research_states(store)
    assert registry.research_states(store)==before


def test_ranking_idempotent_lower_scores_cannot_replace_and_archive_tamper_excluded(lab):
    store,_,_,repo,add=lab
    best=add(spro=.8,ap=.7);registry.initialize(store,repo)
    before=registry.research_states(store);n=len(registry.records(store,'RESEARCH_INCUMBENT_SELECTED'))
    add(name='challenger',spro=.1,ap=.1);registry.initialize(store,repo)
    assert registry.research_states(store)==before
    assert len(registry.records(store,'RESEARCH_INCUMBENT_SELECTED'))==n
    m=next(m for m in registry.retained(store).values() if m['run_id']==best)
    p=Path(m['archive'])/'model.pt';p.chmod(0o600);p.write_bytes(b'corrupt')
    result=registry.refresh_research_incumbents(store)
    assert result['excluded'][0]['run_id']==best
    assert next(iter(registry.research_states(store).values()))['retention_id']!=m['retention_id']


def test_incompatible_cohorts_never_rank_together(lab):
    store,_,_,repo,add=lab
    add(spro=.2);add(name='challenger',spro=.9,evaluator_version='different')
    registry.initialize(store,repo)
    assert len(registry.research_states(store))==2


def test_stronger_exploratory_does_not_downgrade_confirmed_evidence(lab):
    store,_,_,repo,add=lab
    add(spro=.3);registry.initialize(store,repo)
    cid,confirmed=next(iter(registry.states(store).items()));confirmed['confirmation']='REPEATED_SEED_SUPPORTED'
    with store.connect() as con:con.execute('UPDATE settings SET value=? WHERE name=?',(json.dumps(confirmed),registry.PREFIX+cid))
    add(name='challenger',spro=.8,ap=.6);registry.initialize(store,repo)
    assert registry.states(store)[cid]==confirmed
    g=registry.progress(store)['cohorts'][0]
    assert g['research_incumbent']['metrics'][registry.SPRO]==.8 and g['best_supported_spro']==.3


def test_high_fpr_only_cohort_has_visible_research_reference_without_confirmation(lab):
    store,_,_,repo,add=lab
    best=add(spro=.9,ap=.8,metrics={registry.FPR:.5,'dev_real/normal_false_positives':1})
    registry.initialize(store,repo)
    assert registry.states(store)=={}
    g=registry.progress(store)['cohorts'][0]
    assert g['research_incumbent']['run_id']==best
    assert g['state']['confirmation']=='NO_CONFIRMED_CHAMPION'
    assert g['best_supported_spro'] is None


def test_dashboard_reports_research_score_separately_from_confirmation(lab):
    from streamlit.testing.v1 import AppTest
    store,_,_,repo,add=lab;add();registry.initialize(store,repo)
    add(name='challenger',spro=.8,ap=.7);registry.initialize(store,repo)
    summary=registry.progress(store)
    app=AppTest.from_string('from dashboard.cumulative import progress_card\nprogress_card('+repr(summary)+')').run()
    assert not app.exception
    assert any('80% localization' in m.label for m in app.metric)
    assert any('Active exploratory research incumbent' in m.value for m in app.markdown)


def test_stale_concurrent_scan_cannot_replace_newer_stronger_reference(lab,monkeypatch):
    store,_,_,repo,add=lab
    add(spro=.2);registry.initialize(store,repo)
    snapshot=registry.retained(store)
    add(name='challenger',spro=.8,ap=.7);registry.initialize(store,repo)
    stronger=registry.research_states(store)
    monkeypatch.setattr(registry,'retained',lambda store:snapshot)
    registry.refresh_research_incumbents(store)
    assert registry.research_states(store)==stronger
