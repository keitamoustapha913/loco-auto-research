import json
import os
from pathlib import Path
from PIL import Image
from scripts.demo_run import make_fixture
from benchmark.data_guard import build_manifest
from orchestrator.state_store import StateStore
from orchestrator.controller import run_worker_once


def test_toy_end_to_end_cpu(tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    monkeypatch.setenv('LOCO_LEDGER',str(tmp_path/'controller'/'experiments.sqlite'))
    make_fixture(tmp_path/'good_data')
    manifest=build_manifest(tmp_path/'good_data')
    manifest_path=tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    store=StateStore()
    rid=store.create_run('toy-test','toy_normal_ae','fixture',config={'device':'cpu','epochs':1,'size':64,'seed':7,'timeout_seconds':120})
    for state in ('IMPLEMENTED','VALIDATED','QUEUED'):
        store.transition(rid,state)
    outcome=run_worker_once(store,manifest_path,tmp_path)
    assert outcome['status']=='COMPLETED'
    run=store.get_run(rid)
    assert 'dev_synthetic/pixel_auroc' in json.loads(run['metrics_json'])
    assert (Path(run['artifact_dir'])/'synthetic_overlay.png').is_file()
    assert len(store.pending_tracking())==1
