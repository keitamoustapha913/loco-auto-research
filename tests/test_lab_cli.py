import json
import os
from pathlib import Path
import types
import pytest

from orchestrator import lab_cli
from orchestrator.state_store import StateStore
from dashboard.data_access import list_runs, ledger_summary


def configured(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    good = runtime / 'good_data'
    good.mkdir(parents=True)
    (runtime / 'good_manifest.json').write_text(json.dumps({'schema_version': '1.0', 'type': 'good_only', 'root': str(good), 'images': []}))
    ledger = runtime / 'controller' / 'experiments.sqlite'
    monkeypatch.setenv('LOCO_RUNTIME', str(runtime))
    monkeypatch.setenv('LOCO_LEDGER', str(ledger))
    monkeypatch.setenv('LOCO_GOOD_ROOT', str(good))
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:56000')
    return runtime, ledger


def test_safe_environment_loading(tmp_path, monkeypatch):
    for name in ('LOCO_RUNTIME', 'LOCO_LEDGER', 'LOCO_GOOD_ROOT', 'MLFLOW_TRACKING_URI'):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / 'runtime'
    env = tmp_path / 'research.env'
    env.write_text(f'export LOCO_RUNTIME="{root}"\nexport LOCO_LEDGER="$LOCO_RUNTIME/controller/experiments.sqlite"\n')
    runtime, ledger = lab_cli.load_settings(env)
    assert runtime == root and ledger == root / 'controller' / 'experiments.sqlite'
    env.write_text('export LOCO_RUNTIME="$(touch /tmp/unsafe)"\n')
    monkeypatch.delenv('LOCO_RUNTIME', raising=False)
    with pytest.raises(ValueError):
        lab_cli.load_settings(env)


def test_start_idempotent_stop_and_preserve_ledger(tmp_path, monkeypatch):
    runtime, ledger = configured(tmp_path, monkeypatch)
    store = StateStore(ledger)
    run = store.create_run('h', 'toy_normal_ae', 'fixture')
    monkeypatch.setattr(lab_cli.importlib.util, 'find_spec', lambda name: True)
    monkeypatch.setattr('benchmark.data_guard.load_manifest', lambda path, verify_hashes=False: ({'root': str(runtime / 'good_data'), 'images': []}, []))
    active = set()
    calls = []
    def spawn(_runtime, name):
        calls.append(name)
        if name in active:
            return False
        active.add(name)
        return True
    monkeypatch.setattr(lab_cli, 'spawn_service', spawn)
    monkeypatch.setattr(lab_cli, 'stop_service', lambda _runtime, name: active.discard(name))
    monkeypatch.setattr(lab_cli, 'status', lambda *_: {'active': sorted(active)})
    assert lab_cli.start(runtime, ledger)['active'] == ['dashboard', 'mlflow', 'tracking']
    lab_cli.start(runtime, ledger)
    assert calls == ['mlflow', 'dashboard', 'tracking'] * 2
    assert store.get_run(run) is not None
    for name in reversed(lab_cli.SERVICES):
        lab_cli.stop_service(runtime, name)
    assert not active


def test_partial_startup_failure_rolls_back(tmp_path, monkeypatch):
    runtime, ledger = configured(tmp_path, monkeypatch)
    monkeypatch.setattr(lab_cli.importlib.util, 'find_spec', lambda name: True)
    monkeypatch.setattr('benchmark.data_guard.load_manifest', lambda path, verify_hashes=False: ({'root': str(runtime / 'good_data'), 'images': []}, []))
    stopped = []
    def spawn(_runtime, name):
        if name == 'dashboard':
            raise RuntimeError('failed health check')
        return True
    monkeypatch.setattr(lab_cli, 'spawn_service', spawn)
    monkeypatch.setattr(lab_cli, 'stop_service', lambda _runtime, name: stopped.append(name))
    with pytest.raises(RuntimeError):
        lab_cli.start(runtime, ledger)
    assert stopped == ['mlflow']


def test_unmanaged_occupied_port(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_cli, 'owned_process', lambda *_: None)
    monkeypatch.setattr(lab_cli, 'port_open', lambda port: True)
    with pytest.raises(RuntimeError, match='unmanaged'):
        lab_cli.spawn_service(tmp_path, 'mlflow')


def test_stale_pid_and_read_only_dashboard(tmp_path, monkeypatch):
    runtime, ledger = configured(tmp_path, monkeypatch)
    state = lab_cli.state_path(runtime, 'tracking')
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'pid': 999999, 'created': 0, 'signature': 'tracking'}))
    assert lab_cli.owned_process(runtime, 'tracking') is None
    store = StateStore(ledger)
    rid = store.create_run('h', 'toy_normal_ae', 'fixture')
    assert any(r['run_id'] == rid for r in list_runs())
    assert ledger_summary()['counts']['PROPOSED'] == 1


def test_smoke_rejects_unavailable_cuda(tmp_path, monkeypatch):
    runtime, ledger = configured(tmp_path, monkeypatch)
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        lab_cli.smoke(runtime, ledger, 'juice_bottle', 'cuda')


def test_unattended_candidate_rejected_without_isolation(tmp_path, monkeypatch):
    from orchestrator.controller import queue_candidate
    monkeypatch.delenv('LOCO_SANDBOX_VERIFIED', raising=False)
    with pytest.raises(PermissionError):
        queue_candidate(StateStore(tmp_path / 'db.sqlite'), 'h', 'untrusted', 'juice_bottle', 'smoke', 'cuda', tmp_path / 'manifest.json')


@pytest.mark.parametrize('terminal', ['FAILED', 'TIMED_OUT'])
def test_smoke_records_guarded_failure_or_timeout(tmp_path, monkeypatch, terminal):
    runtime, ledger = configured(tmp_path, monkeypatch)
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr('benchmark.data_guard.load_manifest', lambda path: ({'root': str(runtime / 'good_data'), 'images': [{'category': 'juice_bottle'}]}, []))
    def queue(store, hypothesis, candidate, category, stage, device, manifest, seconds):
        rid = store.create_run(hypothesis, candidate, category, stage, {'device': device})
        for state in ('IMPLEMENTED', 'VALIDATED', 'QUEUED'):
            store.transition(rid, state)
        return rid
    monkeypatch.setattr('orchestrator.controller.queue_candidate', queue)
    def execute(run, store, *args):
        store.account_gpu_usage(run['run_id'], 1)
        store.transition(run['run_id'], terminal, reason='bounded toy job did not finish')
        return terminal
    monkeypatch.setattr('orchestrator.experiment_runner.execute_run', execute)
    monkeypatch.setattr('orchestrator.mlflow_tracker.sync_outbox', lambda store: {'delivered': 1})
    outcome = lab_cli.smoke(runtime, ledger, 'juice_bottle', 'cuda')
    assert outcome['status'] == terminal
    assert StateStore(ledger).get_run(outcome['run_id'])['error']
    assert len(StateStore(ledger).pending_tracking()) == 1
