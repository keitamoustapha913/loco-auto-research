import sys
import pytest
import types
import json
from orchestrator.state_store import StateStore
from orchestrator.mlflow_tracker import enqueue_run_tracking, sync_outbox


@pytest.mark.parametrize('archived',[False,True])
def test_mlflow_outbox_sync_with_fake_client(tmp_path, monkeypatch, archived):
    store=StateStore(tmp_path/'ledger.sqlite')
    rid=store.create_run('h','toy_normal_ae','fixture',config={'size':64,'training_completion_policy':'elapsed-training-loop-v1'})
    for status in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'):
        store.transition(rid,status)
    store.set_run_result(rid, {'dev_synthetic/pixel_auroc':0.75,'training/loop_elapsed_seconds':900,
        'training/duration_seconds':900,'training/learning_seconds':600,'training/loop_overhead_seconds':300})
    store.transition(rid,'COMPLETED')
    enqueue_run_tracking(store,rid)
    with store.connect() as con:
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,0,'MODEL_RETAINED',?)",
            (rid,json.dumps({'retention_id':'fixture-retained','cohort_id':'fixture-cohort',
                'metrics':{'dev_real/logical_au_spro_0_05_compatible':.8,'dev_real/logical_pixel_ap_pooled':.7,
                    'dev_real/normal_image_fpr':.2,'dev_real/inference_e2e_ms_p95':10,
                    'dev_real/parameters':100,'dev_real/peak_cpu_rss_mib':50},
                'reference':{'checkpoint_sha256':'fixture-checkpoint'},'policy_sha256':'fixture-policy'})))
    if archived:
        with store.connect() as con:
            con.execute('INSERT INTO settings(name,value) VALUES(?,?)',('artifact_retention:'+rid,json.dumps({'status':'ARCHIVED_METRICS_ONLY','checkpoint_available':False})))
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
        def log_dict(self,*x): calls.append(('dictionary',x))
        def set_terminated(self,*x,**kw): calls.append(('term',x))
    ml=types.ModuleType('mlflow'); ml.set_tracking_uri=lambda uri:None
    tracking=types.ModuleType('mlflow.tracking'); tracking.MlflowClient=Client
    monkeypatch.setitem(sys.modules,'mlflow',ml)
    monkeypatch.setitem(sys.modules,'mlflow.tracking',tracking)
    response=sync_outbox(store,tracking_uri='http://127.0.0.1:5000')
    assert response['delivered']==1
    assert store.get_run(rid)['mlflow_run_id']=='fake-run'
    assert store.pending_tracking()==[]
    assert any(c[0]=='tag' and c[1][1]=='research.artifact_availability' and c[1][2]==('ARCHIVED_METRICS_ONLY' if archived else 'FULL') for c in calls)
    assert any(c[0]=='metric' for c in calls)
    assert any(c[0]=='tag' and c[1][1]=='training.duration_semantics' and 'elapsed_training_loop' in c[1][2] for c in calls)
    assert {c[1][1] for c in calls if c[0]=='metric'}>={'training/loop_elapsed_seconds','training/learning_seconds','training/loop_overhead_seconds'}
    assert any(c[0]=='tag' and c[1][1]=='research.retention_id' for c in calls)
    assert any(c[0]=='tag' and c[1][1]=='research.localization_weight' and c[1][2]=='0.8' for c in calls)
    assert any(c[0]=='tag' and c[1][1]=='research.active_research_incumbent' and c[1][2]=='false' for c in calls)
    assert any(c[0]=='dictionary' and c[1][2]=='retention/model_record.json' for c in calls)
    assert sync_outbox(store)['delivered']==0
    assert sum(c[0]=='create' for c in calls)==1
