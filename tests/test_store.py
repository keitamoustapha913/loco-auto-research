import json
import pytest
from orchestrator.state_store import StateStore
from orchestrator.budgets import enforce, BudgetExceeded

@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path/'experiments.sqlite')

def add_run(store):
    return store.create_run('H1','toy_normal_ae','fixture')

def test_run_lifecycle(store):
    rid = add_run(store)
    for status in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'):
        store.transition(rid,status)
    store.set_run_result(rid,{'normal/image_fpr_at_threshold':0.0,'dev_synthetic/pixel_auroc':0.7,'inference/e2e_ms_p95':11.0},gpu_seconds=12)
    for status in ('COMPLETED','REVIEWED','PROMOTED'):
        store.transition(rid,status)
    assert store.get_run(rid)['status'] == 'PROMOTED'
    assert store.total_gpu_seconds() == 12
    assert len(store.events()) >= 7

def test_invalid_transition(store):
    rid = add_run(store)
    with pytest.raises(ValueError):
        store.transition(rid,'PROMOTED')

def test_budgets_paused(store, monkeypatch):
    store.set_paused(True)
    with pytest.raises(BudgetExceeded):
        enforce(store)
    store.set_paused(False)
    monkeypatch.setenv('LOCO_GPU_BUDGET_HOURS','0')
    with pytest.raises(BudgetExceeded):
        enforce(store, next_gpu_seconds=1)

def test_outbox_idempotent(store):
    rid=add_run(store)
    store.enqueue_tracking('x',rid,{'a':1})
    store.enqueue_tracking('x',rid,{'a':1})
    assert len(store.pending_tracking()) == 1
    store.tracking_delivered(store.pending_tracking()[0]['id'])
    assert not store.pending_tracking()

def test_crash_marks_interrupted_without_rerun(store):
    rid=add_run(store)
    for state in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'):
        store.transition(rid,state)
    assert store.reconcile_after_restart() == [rid]
    assert store.get_run(rid)['status'] == 'INTERRUPTED'
    assert store.reconcile_after_restart() == []

def test_readonly_db_denies_writes(store):
    ro=StateStore(store.path, read_only=True)
    with pytest.raises(Exception):
        with ro.connect() as conn:
            conn.execute("INSERT INTO settings(name,value) VALUES('hack','1')")
