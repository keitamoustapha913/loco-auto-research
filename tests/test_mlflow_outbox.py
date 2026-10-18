import sys
import types
from orchestrator.state_store import StateStore
from orchestrator.mlflow_tracker import enqueue_run_tracking, sync_outbox


def test_mlflow_outbox_sync_with_fake_client(tmp_path, monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite')
    rid=store.create_run('h','toy_normal_ae','fixture',config={'size':64})
    for status in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'):
        store.transition(rid,status)
    store.set_run_result(rid, {'dev_synthetic/pixel_auroc':0.75})
    store.transition(rid,'COMPLETED')
    enqueue_run_tracking(store,rid)
    calls=[]
    class Client:
        def __init__(self,tracking_uri): calls.append(('uri',tracking_uri))
        def get_experiment_by_name(self,name): return None
        def create_experiment(self,name): return '123'
        def create_run(self,experiment_id,tags):
            calls.append(('create',tags))
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id='fake-run'))
        def log_param(self,*x): calls.append(('param',x))
        def log_metric(self,*x): calls.append(('metric',x))
        def set_tag(self,*x): calls.append(('tag',x))
        def set_terminated(self,*x,**kw): calls.append(('term',x))
    ml=types.ModuleType('mlflow'); ml.set_tracking_uri=lambda uri:None
    tracking=types.ModuleType('mlflow.tracking'); tracking.MlflowClient=Client
    monkeypatch.setitem(sys.modules,'mlflow',ml)
    monkeypatch.setitem(sys.modules,'mlflow.tracking',tracking)
    response=sync_outbox(store,tracking_uri='http://127.0.0.1:5000')
    assert response['delivered']==1
    assert store.get_run(rid)['mlflow_run_id']=='fake-run'
    assert store.pending_tracking()==[]
    assert any(c[0]=='metric' for c in calls)
