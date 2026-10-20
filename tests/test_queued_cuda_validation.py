"""CUDA stage supervision with injected processes; never launches GPU work."""
import json
from pathlib import Path
import sys
import pytest

from benchmark.data_guard import build_manifest
from scripts.demo_run import make_fixture
from orchestrator import controller, experiment_runner, sandbox
from orchestrator.gpu_queue import gpu_lock, GpuBusy
from orchestrator.state_store import StateStore

REPO=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('smoke_outcome', ['pass','failure','timeout','oom','bad_result','missing_result'])
def test_smoke_uses_queued_lock_deadline_and_charged_time(tmp_path,monkeypatch,smoke_outcome):
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    monkeypatch.setenv('LOCO_FORCE_SANDBOX_TOY','1')
    make_fixture(tmp_path/'data')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(build_manifest(tmp_path/'data')))
    store=StateStore(tmp_path/'ledger.sqlite')
    rid=store.create_run('fixture','toy_normal_ae','juice_bottle',config={
        'device':'cuda','timeout_seconds':2,'cuda_smoke_required':True})
    for status in ('IMPLEMENTED','VALIDATED','QUEUED'):store.transition(rid,status)
    monkeypatch.setattr(sandbox,'command',lambda *a,**kw:[sys.executable,*a[4]])
    clock=[0.0]
    def monotonic():
        clock[0]+=.05
        return clock[0]
    monkeypatch.setattr(experiment_runner.time,'monotonic',monotonic)
    monkeypatch.setattr(experiment_runner.time,'sleep',lambda *a:None)
    commands=[]
    class Process:
        pid=99999999
        def __init__(self,command,**kw):
            with pytest.raises(GpuBusy):
                with gpu_lock(tmp_path):pass
            commands.append(command)
            self.smoke='scripts.candidate_smoke' in command
            self.returncode=0
            if self.smoke and smoke_outcome in ('failure','oom'):
                self.returncode=1
                kw['stderr'].write(b'CUDA out of memory' if smoke_outcome=='oom' else b'EXACT smoke traceback')
                kw['stderr'].flush()
            if not self.smoke and smoke_outcome!='missing_result':
                metrics=[] if smoke_outcome=='bad_result' else {'training/duration_seconds':.1}
                (tmp_path/'jobs'/rid/'measurement.json').write_text(json.dumps({'metrics':metrics}))
        def poll(self):
            return None if self.smoke and smoke_outcome=='timeout' else self.returncode
        def wait(self,timeout=None):self.returncode=-15
    monkeypatch.setattr(experiment_runner.subprocess,'Popen',Process)
    monkeypatch.setattr(experiment_runner.os,'killpg',lambda *a:None)
    result=controller.run_worker_once(store,mp,tmp_path,expected_run_id=rid)
    run=store.get_run(rid)
    assert run['gpu_seconds']>0
    assert len(store.pending_tracking())==1
    if smoke_outcome=='pass':
        assert result['status']=='COMPLETED' and len(commands)==2
        assert 'scripts.candidate_smoke' in commands[0] and 'scripts.run_candidate' in commands[1]
        assert json.loads(run['metrics_json'])['validation/cuda_smoke_passed']==1
        assert run['gpu_seconds']>=json.loads(run['metrics_json'])['validation/cuda_smoke_seconds']
    elif smoke_outcome in ('bad_result','missing_result'):
        assert len(commands)==2 and result['status']=='FAILED'
        assert 'Candidate result validation' in run['error']
    else:
        assert len(commands)==1  # Pilot never starts after failed validation.
        assert result['status']=={'failure':'FAILED','timeout':'TIMED_OUT','oom':'OOM'}[smoke_outcome]
        assert run['error'].startswith('CUDA smoke validation')
        if smoke_outcome=='failure':assert 'EXACT smoke traceback' in run['error']
    events=[e for e in store.events() if e['run_id']==rid and e['kind']=='CUDA_SMOKE_VALIDATION']
    assert len(events)==1
    assert json.loads(events[0]['payload_json'])['timeout_seconds']==2


@pytest.mark.parametrize('mutate', [False, True])
def test_smoke_worker_only_uses_two_normal_learning_images_and_frozen_inference(tmp_path,monkeypatch,mutate):
    import numpy as np
    import torch
    from scripts import candidate_smoke
    make_fixture(tmp_path/'data')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(build_manifest(tmp_path/'data')))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:1024)
    seen=[]
    class Predictor:
        def __init__(self):self.state=torch.zeros(1)
        def predict(self,x):
            assert torch.is_inference_mode_enabled()
            if mutate:
                self.state.add_(1)
            return {'anomaly_maps':np.zeros((len(x),1,*x.shape[-2:]),np.float32),
                    'image_scores':np.zeros(len(x),np.float32)}
    class Candidate:
        def train(self,config,manifest,output):
            m=json.loads(manifest.read_text())
            assert len(m['images'])==2 and all(e['split']=='train/good' for e in m['images'])
            assert config['steps']==2 and config['epochs']==config['batch_size']==1
            assert config['size']==128 and not config['synthetic_evaluation']
            seen.append('train')
            output.mkdir()
            (output/'model.pt').write_bytes(b'fixture checkpoint')
            (output/'checkpoint_manifest.json').write_text('{}')
        def load(self,path,device):
            assert device=='cuda'
            return Predictor()
    monkeypatch.setattr(candidate_smoke,'load_candidate',lambda *a:Candidate())
    cfg={'device':'cuda','category':'juice_bottle','size':128,'steps':9000}
    if mutate:
        with pytest.raises(ValueError,match='mutated'):
            candidate_smoke.run('fixture',mp,cfg,tmp_path/'smoke')
    else:
        summary=candidate_smoke.run('fixture',mp,cfg,tmp_path/'smoke')
        assert summary['status']=='PASSED' and summary['checkpoint_unchanged']
        assert not summary['pilot_reuses_smoke_weights']
    assert seen==['train']
