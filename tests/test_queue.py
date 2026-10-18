import pytest
from orchestrator.state_store import StateStore
from orchestrator.gpu_queue import gpu_lock, GpuBusy, claim_next


def test_exclusive_gpu_lock(tmp_path):
    with gpu_lock(tmp_path):
        with pytest.raises(GpuBusy):
            with gpu_lock(tmp_path):
                pass
    with gpu_lock(tmp_path):
        pass


def test_queue_claim_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    store = StateStore(tmp_path/'db.sqlite')
    run_ids=[]
    for _ in range(2):
        rid=store.create_run('hyp','toy_normal_ae','fixture')
        for state in ('IMPLEMENTED','VALIDATED','QUEUED'):
            store.transition(rid,state)
        run_ids.append(rid)
    with gpu_lock(tmp_path):
        first=claim_next(store,max_duration=1)
        assert first['run_id'] == run_ids[0]
        assert claim_next(store,max_duration=1) is None
