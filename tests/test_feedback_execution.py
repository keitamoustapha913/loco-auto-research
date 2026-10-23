"""Scheduling/cache changes preserve scientific predictions and original scorer."""
import copy,json,time
from pathlib import Path
import numpy as np
import pytest
from test_model_registry import lab
from benchmark.data_guard import sha256_file
from benchmark.feedback_metrics import detection_metrics,saturated_region_curve,VERSION
from orchestrator.development_feedback import (run_isolated_cases,feedback_identity,verify_feedback_cache,cpu_worker_count)


def test_serial_parallel_predictions_detection_and_region_scores_identical():
    # Analytic labels/regions, no real data. Same case identities/order and native shape.
    def case(i):
        image=np.arange(24,dtype=np.float32).reshape(6,4)/(24+i)
        mask=np.zeros_like(image,dtype=bool);mask[2:4,1:3]=True
        return {'category':['good','logical_anomalies','structural_anomalies'][i%3],
                'image_score':float(image.max()),'map':image,'mask':mask}
    serial=run_isolated_cases(range(9),case,workers=1)
    parallel=run_isolated_cases(range(9),case,workers=4)
    assert detection_metrics(serial,.9)==detection_metrics(parallel,.9)
    for a,b in zip(serial,parallel):assert np.array_equal(a['map'],b['map']) and a['map'].shape==(6,4)
    def metric(rows):return saturated_region_curve(np.concatenate([r['map'][~r['mask']] for r in rows]),[(r['map'][r['mask']],2) for r in rows])
    assert metric(serial)==metric(parallel)


def test_cache_identity_covers_every_fitting_and_inference_input():
    manifest={'split_id':'fixture','images':[{'sha256':'good-only'}]};run={'manifest_sha256':'train-only'}
    config={'inference':{'mode':'frozen'}};before={'checkpoint':'a','metadata':'b','source':'c'}
    def key(**kw):return feedback_identity(kw.get('manifest',manifest),kw.get('run',run),kw.get('config',config),kw.get('before',before),kw.get('size',128),kw.get('threshold',.3),workers=kw.get('workers',4))
    original=key()
    for args in [dict(size=256),dict(threshold=.4),dict(workers=1),dict(manifest={'split_id':'other'}),dict(run={'manifest_sha256':'other'}),dict(config={'inference':{'mode':'changed'}})]:assert key(**args)!=original
    for k in before:assert key(before=dict(before,**{k:'changed'}))!=original
    assert VERSION=='real-development-v2.0-compatible'


def cached(tmp_path):
    output=tmp_path/'jobs/run/feedback/eval';case=output/'case_0000';case.mkdir(parents=True)
    np.save(case/'map.npy',np.arange(6,dtype=np.float32).reshape(2,3))
    (output/'case_scores.json').write_text(json.dumps([{'case_index':0,'map_sha256':sha256_file(case/'map.npy')}]))
    (output/'localization.json').write_text('{}')
    summary={'cache_identity':'exact','checkpoint_unchanged':True,'artifact_dir':str(output),'counts':{'images':1},
             'cache_artifact_hashes':{n:sha256_file(output/n) for n in ['case_scores.json','localization.json']}}
    return output,summary


def test_verified_cache_reuses_only_exact_intact_outputs(tmp_path):
    root,s=cached(tmp_path)
    assert verify_feedback_cache(s,'exact',tmp_path/'jobs')==root
    with pytest.raises(ValueError,match='identity'):verify_feedback_cache(s,'changed',tmp_path/'jobs')
    np.save(root/'case_0000/map.npy',np.zeros((2,3),dtype=np.float32))
    with pytest.raises(ValueError,match='heatmap'):verify_feedback_cache(s,'exact',tmp_path/'jobs')


def test_cache_rejects_annotations_or_final_holdout_path_redirect(tmp_path):
    root,s=cached(tmp_path);outside=tmp_path/'private';outside.mkdir();(outside/'map.npy').write_bytes(b'fake fixture')
    import shutil
    shutil.rmtree(root/'case_0000');(root/'case_0000').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='heatmap'):verify_feedback_cache(s,'exact',tmp_path/'jobs')


def test_worker_count_has_internal_resource_bounds(monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1024**3))
    assert cpu_worker_count()==1


def test_fast_state_hash_preserves_metadata_and_tensor_mutation_detection():
    import torch
    from scripts.feedback_infer import state_digest
    from types import SimpleNamespace
    predictor=SimpleNamespace(net=torch.nn.Linear(2,1),metadata={'indexed_stream':[{'index':i,'hash':'a'*64} for i in range(100)]})
    before=state_digest(predictor)
    predictor.metadata['indexed_stream'][99]['hash']='b'*64
    assert state_digest(predictor)!=before
    before=state_digest(predictor)
    with torch.no_grad():predictor.net.weight.add_(1)
    assert state_digest(predictor)!=before
    predictor.bad=object()
    with pytest.raises(ValueError,match='uninspectable'):state_digest(predictor)


def test_frozen_readout_reuse_copies_verified_maps_without_model_execution(tmp_path,monkeypatch):
    from orchestrator import development_feedback as feedback
    root,s=cached(tmp_path)
    s.update(evaluation_id='first',metrics={'dev_real/logical_image_auroc':.75},plan_id='old',seed=7)
    target=tmp_path/'jobs/new/feedback/eval';target.parent.mkdir(parents=True)
    monkeypatch.setattr(feedback,'run_bounded',lambda *a,**kw:pytest.fail('must not execute inference or training for exact reuse'))
    result=feedback.reuse_feedback_artifacts(s,'exact',tmp_path/'jobs',target,{}, {'gpu_seconds':0}, 'second')
    assert result['metrics']['dev_real/logical_image_auroc']==.75
    assert result['reused_from_evaluation_id']=='first' and result['evaluation_id']=='second'
    assert 'plan_id' not in result and 'seed' not in result
    assert (target/'case_0000/map.npy').read_bytes()==(root/'case_0000/map.npy').read_bytes()
    assert s['metrics']=={'dev_real/logical_image_auroc':.75}
    feedback.verify_feedback_cache(result,'exact',tmp_path/'jobs')


def test_managed_evaluation_cache_and_cross_run_readout_are_idempotent(tmp_path,monkeypatch,lab):
    # Constructed end-to-end controller fixture; fake measurements are test data.
    # Real-checkpoint CPU worker parity is measured separately in technical evidence.
    from orchestrator import development_feedback as feedback
    from orchestrator import experiment_plan
    store,_,_,repo,add=lab
    monkeypatch.setattr(feedback,'verify_import',lambda root:{'split_id':'fixture','images':[
        {'path':f'case-{i}.png','width':4,'height':6,'category':kind,'sha256':str(i),'masks':[]} for i,kind in enumerate(['good','logical_anomalies','structural_anomalies'])],
        'summary':{'images':3,'categories':{'good':1,'logical_anomalies':1,'structural_anomalies':1}}})
    monkeypatch.setattr(feedback,'load_manifest',lambda path:({'images':[{'split':'train/good','category':'juice_bottle'}]*335},[]))
    monkeypatch.setattr(feedback,'manifest_fingerprint',lambda manifest:'normal-only-fixture')
    monkeypatch.setattr(experiment_plan,'execution_source',lambda *a:repo/'candidates/baseline/model.py')
    monkeypatch.setattr(feedback,'sandbox_command',lambda repo,module,data,folder,*a,**kw:['fixture',str(folder)])
    calls=[]
    def bounded(command,log,seconds,cancel=None):
        calls.append(command)
        if command[0]=='fixture':
            folder=Path(command[1]);np.save(folder/'map.npy',np.arange(24,dtype=np.float32).reshape(6,4))
            (folder/'inference.json').write_text(json.dumps({'image_score':.2,'e2e_ms':1.,'parameters':10,
                'trainable_parameters_before_freeze':0,'process_peak_rss_mib':20.,'measurement_protocol':'fixture'}))
        else:Path(command[-1]).write_text(json.dumps({'dev_real/logical_pixel_ap_pooled':.4}))
    monkeypatch.setattr(feedback,'run_bounded',bounded)
    def run():
        rid=add();row=store.get_run(rid);folder=Path(row['artifact_dir'])
        provenance=json.loads((folder/'run_manifest.json').read_text());provenance.update(model_module_sha256=sha256_file(repo/'candidates/baseline/model.py'),size=16)
        (folder/'run_manifest.json').write_text(json.dumps(provenance))
        m=json.loads(row['metrics_json']);m['normal/threshold']=.3
        with store.connect() as con:con.execute('UPDATE runs SET metrics_json=? WHERE run_id=?',(json.dumps(m),rid))
        return rid
    rid=run();original=store.get_run(rid)
    first=feedback._evaluate(tmp_path,store,rid,30,None)
    assert len(calls)==4 and first['metrics']['dev_real/logical_pixel_ap_pooled']==.4
    assert store.get_run(rid)==original
    assert feedback._evaluate(tmp_path,store,rid,30,None)==first and len(calls)==4
    second_id=run();second=feedback._evaluate(tmp_path,store,second_id,30,None)
    assert second['reused_from_evaluation_id']==first['evaluation_id'] and len(calls)==4
    assert second['evaluation_id']!=first['evaluation_id']
    assert 'CPU concurrency=' in first['measurement_protocol']
    verify=feedback.verify_feedback_cache
    def tamper_after_verification(*args,**kwargs):
        root=verify(*args,**kwargs)
        (tmp_path/'jobs'/rid/'checkpoint/model.pt').write_bytes(b'concurrent fixture mutation')
        return root
    monkeypatch.setattr(feedback,'verify_feedback_cache',tamper_after_verification)
    with pytest.raises(ValueError,match='immutable candidate/checkpoint'):
        feedback._evaluate(tmp_path,store,rid,30,None)


def test_admission_uses_verified_current_pipeline_cost_not_obsolete_serial_cost(lab):
    from orchestrator.campaign_config import training_execution_estimate
    from orchestrator.development_feedback import feedback_pipeline_identity
    store,c,cid,_,_=lab
    cfg=json.loads(c.get(cid)['config_json']);cfg['limits']['training_minutes_per_experiment']=15
    with store.connect() as con:
        con.execute("INSERT INTO events(time,kind,payload_json) VALUES(0,'REAL_DEVELOPMENT_FEEDBACK',?)",
                    (json.dumps({'metrics':{'dev_real/evaluation_wall_seconds':1500}}),))
        con.execute("INSERT INTO events(time,kind,payload_json) VALUES(0,'FEEDBACK_EXECUTION_PROFILE',?)",
                    (json.dumps({'category':'juice_bottle','evaluation_wall_seconds':200,'pipeline_identity':feedback_pipeline_identity()}),))
    estimate=training_execution_estimate(store,cfg,jobs=6,newly_trained=6,loop_timed=True)
    assert estimate['training_seconds_per_model']==900
    assert estimate['feedback_estimate_seconds']==250
    assert estimate['required_seconds']<4*3600
