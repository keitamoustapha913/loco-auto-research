import json
import sys
import types
from pathlib import Path
from orchestrator.state_store import StateStore
from orchestrator.mlflow_tracker import sync_outbox


def finished_store(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    rid = store.create_run('smoke', 'toy_normal_ae', 'fixture', config={'seed': 7})
    for state in ('IMPLEMENTED', 'VALIDATED', 'QUEUED', 'RUNNING'):
        store.transition(rid, state)
    store.set_run_result(rid, {'dev_synthetic/pixel_auroc': 0.5})
    store.transition(rid, 'COMPLETED')
    return store, rid


def test_terminal_transition_atomically_creates_outbox(tmp_path):
    store, rid = finished_store(tmp_path)
    pending = store.pending_tracking()
    assert len(pending) == 1
    assert pending[0]['run_id'] == rid
    assert json.loads(pending[0]['payload_json'])['status'] == 'COMPLETED'


def test_tracking_outage_retry_and_remote_run_recovery(tmp_path, monkeypatch):
    store, rid = finished_store(tmp_path)
    remote = {}
    calls = {'create': 0, 'fail': True}
    class Client:
        def __init__(self, tracking_uri):
            pass
        def get_experiment_by_name(self, name):
            return types.SimpleNamespace(experiment_id='1')
        def search_runs(self, ids, filter_string, max_results):
            return [types.SimpleNamespace(info=types.SimpleNamespace(run_id=remote[rid]))] if rid in remote else []
        def create_run(self, experiment_id, tags, start_time=None):
            calls['create'] += 1
            remote[rid] = 'remote-one'
            if calls['fail']:
                raise ConnectionError('response lost after remote creation')
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id='remote-one'))
        def log_param(self, *args): pass
        def log_metric(self, *args): pass
        def set_tag(self, *args): pass
        def set_terminated(self, *args, **kwargs): pass
    ml = types.ModuleType('mlflow')
    ml.set_tracking_uri = lambda uri: None
    tracking = types.ModuleType('mlflow.tracking')
    tracking.MlflowClient = Client
    monkeypatch.setitem(sys.modules, 'mlflow', ml)
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
    assert sync_outbox(store, 'http://127.0.0.1:56000')['failed'] == 1
    assert len(store.pending_tracking()) == 1
    assert store.pending_tracking()[0]['attempts'] == 1
    calls['fail'] = False
    assert sync_outbox(store, 'http://127.0.0.1:56000')['delivered'] == 1
    assert store.get_run(rid)['mlflow_run_id'] == 'remote-one'
    assert calls['create'] == 1
    assert sync_outbox(store, 'http://127.0.0.1:56000')['delivered'] == 0


def test_dashboard_empty_and_recorded_pages(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LOCO_RUNTIME', str(tmp_path))
    monkeypatch.setenv('LOCO_LEDGER', str(tmp_path / 'ledger.sqlite'))
    page = Path(__file__).resolve().parents[1] / 'dashboard/pages/03_Results.py'
    empty = AppTest.from_file(page).run()
    assert not empty.exception
    assert empty.info
    finished_store(tmp_path)
    # This page is backed by the configured SQLite ledger, including failures and metrics.
    monkeypatch.setenv('LOCO_LEDGER', str(tmp_path / 'ledger.sqlite'))
    recorded = AppTest.from_file(page).run()
    assert not recorded.exception
    assert recorded.info  # Toy runs are excluded from research results by default.
    recorded.toggle[0].set_value(True).run()
    assert recorded.dataframe


def test_tracking_adds_research_sources_and_critique(tmp_path, monkeypatch):
    store, rid = finished_store(tmp_path)
    store.add_proposal({'hypothesis_id': 'smoke', 'references': ['https://example.org/source'], 'priority_rank': 1})
    with store.connect() as con:
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                    (rid, 1.0, 'CRITIC_REVIEW', json.dumps({'critic_concerns': ['shortcut']})))
    tags = {}
    class Client:
        def __init__(self, tracking_uri): pass
        def get_experiment_by_name(self, name): return types.SimpleNamespace(experiment_id='1')
        def search_runs(self, ids, filter_string, max_results): return []
        def create_run(self, experiment_id, tags, start_time=None):
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id='remote'))
        def log_param(self, *args): pass
        def log_metric(self, *args): pass
        def set_tag(self, run_id, key, value): tags[key] = value
        def set_terminated(self, *args, **kwargs): pass
    ml = types.ModuleType('mlflow'); ml.set_tracking_uri = lambda uri: None
    tracking = types.ModuleType('mlflow.tracking'); tracking.MlflowClient = Client
    monkeypatch.setitem(sys.modules, 'mlflow', ml)
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
    assert sync_outbox(store, 'http://127.0.0.1:56000')['delivered'] == 1
    assert 'example.org/source' in tags['research.source_references']
    assert 'shortcut' in tags['research.critic_assessment']
