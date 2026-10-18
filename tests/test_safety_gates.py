import pytest
from orchestrator.state_store import StateStore
from orchestrator.controller import run_research_campaign
from orchestrator.experiment_runner import safe_candidate_id

def test_unattended_requires_operator_os_sandbox(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCO_ALLOW_UNATTENDED', raising=False)
    monkeypatch.delenv('LOCO_SANDBOX_VERIFIED', raising=False)
    with pytest.raises(PermissionError):
        run_research_campaign(StateStore(tmp_path/'ledger.db'), None)

def test_codex_call_ceiling(tmp_path, monkeypatch):
    store=StateStore(tmp_path/'ledger.db')
    monkeypatch.setenv('LOCO_CODEX_MAX_CALLS','1')
    store.reserve_codex_call('scout')
    assert store.codex_call_count()==1
    with pytest.raises(RuntimeError):
        store.reserve_codex_call('researcher')

@pytest.mark.parametrize('bad',['../../etc', 'with space','name;rm','', 'unicode-á'])
def test_untrusted_candidate_id_rejected(bad):
    with pytest.raises(ValueError):
        safe_candidate_id(bad)
