"""Atomic campaign/global admission, including denied calls and recovery."""
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestrator.budgets import CodexBudgetExhausted
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from orchestrator.autonomy import stop_reason, run_forever
from orchestrator.controller import budgeted_invoke
from orchestrator.lab_cli import research_summary

REPO = Path(__file__).resolve().parents[1]


def setup_campaign(tmp_path, calls=8):
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    from legacy_campaign import legacy_config
    config = legacy_config()
    config['limits']['max_codex_calls'] = calls
    return store, campaigns, campaigns.create(config)


def test_historical_calls_do_not_consume_campaign_allowance(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', '30')
    store, campaigns, cid = setup_campaign(tmp_path)
    for _ in range(22):
        store.reserve_codex_call('historical')
    for _ in range(8):
        campaigns.reserve_codex_call(cid, 'scout')
    assert campaigns.usage(cid)['codex_calls'] == 8
    assert store.codex_call_count() == 30
    with pytest.raises(CodexBudgetExhausted, match='campaign'):
        campaigns.reserve_codex_call(cid)
    assert store.codex_call_count() == 30
    assert stop_reason(campaigns, cid)[0] == 'BUDGET_EXHAUSTED'


def test_global_denial_has_no_campaign_charge_or_agent_events(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', '1')
    store, campaigns, cid = setup_campaign(tmp_path)
    store.reserve_codex_call('historical')
    monkeypatch.setattr('orchestrator.controller.invoke', lambda *a, **k: pytest.fail('denied call invoked'))
    with pytest.raises(CodexBudgetExhausted, match='global'):
        budgeted_invoke(store, 'scout', REPO, None, 'fixture', campaign=(campaigns, cid))
    assert campaigns.usage(cid)['codex_calls'] == 0
    assert campaigns.get(cid)['consecutive_failures'] == 0
    assert [e['kind'] for e in campaigns.events(cid)] == ['STARTED']
    status = research_summary(campaigns, campaigns.get(cid), None)
    assert status['codex_budget']['campaign']['remaining'] == 8
    assert status['codex_budget']['global']['remaining'] == 0
    assert status['codex_budget']['global']['used'] == 1


@pytest.mark.parametrize('campaign_limit,global_limit', [(1, 30), (8, 1)])
def test_concurrent_reservations_admit_only_one(tmp_path, monkeypatch, campaign_limit, global_limit):
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', str(global_limit))
    store, campaigns, cid = setup_campaign(tmp_path, campaign_limit)
    def reserve(_):
        try:
            campaigns.reserve_codex_call(cid, 'scout')
            return True
        except CodexBudgetExhausted:
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(reserve, range(6))) == 1
    assert store.codex_call_count() == campaigns.get(cid)['codex_calls'] == 1


def test_restart_retains_admission_and_terminal_budget_stop(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', '30')
    store, campaigns, cid = setup_campaign(tmp_path, 1)
    campaigns.reserve_codex_call(cid, 'scout')  # Crash before response: never refund/replay.
    recovered = CampaignState(StateStore(store.path))
    assert recovered.call_budget(cid)['campaign']['completed'] == 0
    assert recovered.call_budget(cid)['campaign']['remaining'] == 0
    monkeypatch.setattr('orchestrator.autonomy.preflight', lambda *a: None)
    monkeypatch.setattr('orchestrator.autonomy.run_cycle', lambda *a: pytest.fail('exhausted campaign retried'))
    run_forever(cid, tmp_path, store.path, idle_seconds=0)
    assert recovered.get(cid)['status'] == 'BUDGET_EXHAUSTED'
    assert recovered.get(cid)['consecutive_failures'] == 0
    run_forever(cid, tmp_path, store.path, idle_seconds=0)
    assert store.codex_call_count() == 1


def test_denial_during_cycle_stops_once_without_research_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', '1')
    store, campaigns, cid = setup_campaign(tmp_path)
    monkeypatch.setattr('orchestrator.autonomy.preflight', lambda *a: None)
    calls = []
    def cycle(*args):
        calls.append(1)
        store.reserve_codex_call('another-controller')
        budgeted_invoke(store, 'scout', REPO, None, 'fixture', campaign=(campaigns, cid))
    monkeypatch.setattr('orchestrator.autonomy.run_cycle', cycle)
    run_forever(cid, tmp_path, store.path, idle_seconds=0)
    assert calls == [1]
    row = campaigns.get(cid)
    assert row['status'] == 'BUDGET_EXHAUSTED'
    assert row['consecutive_failures'] == row['codex_calls'] == 0
    assert 'global' in row['reason']
    assert {e['kind'] for e in campaigns.events(cid)} == {'STARTED', 'BUDGET_EXHAUSTED'}


def test_success_and_failed_invocation_are_charged_once(tmp_path, monkeypatch):
    store, campaigns, cid = setup_campaign(tmp_path)
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS', '30')
    monkeypatch.setattr('orchestrator.controller.invoke', lambda *a, **k: {'response': {}})
    budgeted_invoke(store, 'scout', REPO, None, 'fixture', campaign=(campaigns, cid))
    def fail(*a, **k):
        raise RuntimeError('CLI timeout')
    monkeypatch.setattr('orchestrator.controller.invoke', fail)
    with pytest.raises(RuntimeError, match='CLI timeout'):
        budgeted_invoke(store, 'scout', REPO, None, 'fixture', campaign=(campaigns, cid))
    budget = campaigns.call_budget(cid)['campaign']
    assert budget['used'] == 2 and budget['completed'] == budget['failed_invocations'] == 1
    assert store.codex_call_count() == 2


def test_legacy_reservations_remain_visible_and_unchanged(tmp_path):
    store, campaigns, cid = setup_campaign(tmp_path)
    with store.connect() as con:
        con.execute('UPDATE campaigns SET codex_calls=6 WHERE id=?', (cid,))
    for _ in range(4):
        campaigns.event(cid, 'AGENT_COMPLETED', role='scout')
    before = campaigns.get(cid)
    status = research_summary(campaigns, before, None)
    budget = status['codex_budget']['campaign']
    assert budget['used'] == 6 and budget['completed'] == 4
    assert 'legacy' in budget['accounting']
    assert campaigns.get(cid) == before
