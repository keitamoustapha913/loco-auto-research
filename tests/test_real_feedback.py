"""Real-development boundary tests use constructed fixtures, never holdout data."""
import copy
import hashlib
import io
import json
import tarfile
import types
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from benchmark import feedback_data as fd
from benchmark.feedback_metrics import detection_metrics, VERSION
from orchestrator.state_store import StateStore
from orchestrator.development_feedback import record_summary, latest_summary, evaluate_run


def package(tmp_path):
    blobs = {}
    def item(path, raw, **extra):
        blobs[path] = raw
        return dict(path=path, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), **extra)
    def png(mode, value):
        b=io.BytesIO(); Image.new(mode,(800,1600),value).save(b,format='PNG'); return b.getvalue()
    images=[]
    for cat in ('good','logical_anomalies','structural_anomalies'):
        masks=[]
        if cat!='good':
            for i in range(2 if cat=='logical_anomalies' else 1):
                masks.append(item(f'juice_bottle/ground_truth/{cat}/000/{i:03d}.png',png('L',255),
                    width=800,height=1600,mode='L',pixel_values=[255],defect_types=['fixture']))
        images.append(item(f'juice_bottle/test/{cat}/000.png',png('RGB',(40,60,80)),
            category=cat,width=800,height=1600,mode='RGB',masks=masks,defect_types=['fixture'] if masks else []))
    meta=[item('juice_bottle/defects_config.json',json.dumps([dict(pixel_value=255,defect_name='fixture',
                relative_saturation=True,saturation_threshold=1)]).encode()),
          item('juice_bottle/license.txt',b'fixture'),item('juice_bottle/readme.txt',b'fixture')]
    archive=tmp_path/'dev-feedback.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path,raw in blobs.items():
            info=tarfile.TarInfo(path); info.size=len(raw); tar.addfile(info,io.BytesIO(raw))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest=dict(schema_version=1,dataset='MVTec LOCO AD juice_bottle',defect_encoding='fixture',
        hash_algorithm='sha256',images=images,mask_association='fixture',metadata=meta,
        package=dict(sha256=sha,size_bytes=archive.stat().st_size),partition='development',
        path_base='fixture',split_id='fixture-dev',summary=dict(images=3,masks=3,multiple_mask_images=1,
            categories=dict(Counter(r['category'] for r in images)),defect_mask_counts={'fixture':3},defect_image_counts={'fixture':2}))
    (tmp_path/'dev-manifest.json').write_text(json.dumps(manifest))
    return manifest, dict(split_id='fixture-dev',archive_sha256=sha,
        manifest_canonical_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def test_archive_integrity_and_multiple_masks(tmp_path):
    manifest,approval=package(tmp_path)
    assert fd.validate_archive(tmp_path,approval)==manifest
    assert len(manifest['images'][1]['masks'])==2
    with (tmp_path/'dev-feedback.tar.gz').open('ab') as out: out.write(b'changed')
    with pytest.raises(ValueError,match='integrity'): fd.validate_archive(tmp_path,approval)


@pytest.mark.parametrize('change', ['holdout','split','association','geometry','path','private_metadata'])
def test_reject_unapproved_manifest(tmp_path,change):
    m,a=package(tmp_path)
    if change=='holdout': m['partition']='final_holdout'
    if change=='split': m['split_id']='other'
    if change=='association': m['images'][1]['masks'][0]['path']='juice_bottle/ground_truth/logical_anomalies/999/000.png'
    if change=='geometry': m['images'][1]['masks'][0]['width']=799
    if change=='path': m['images'][0]['path']='../private.png'
    if change=='private_metadata': m['holdout_manifest']={}
    # Re-pin the constructed fixture to exercise association/geometry checks themselves.
    if change in {'association','geometry','path'}:
        a['manifest_canonical_sha256']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError): fd.schema(m,a)


def test_missing_transfer_stops(tmp_path):
    with pytest.raises(FileNotFoundError,match='Required transfer missing'): fd.validate_archive(tmp_path)


def test_import_idempotence_and_symlink_rejection(tmp_path,monkeypatch):
    incoming=tmp_path/'incoming/juice_bottle_dev_v1'; incoming.mkdir(parents=True)
    m,a=package(incoming); monkeypatch.setattr(fd,'approval',lambda:a)
    fd.import_development(tmp_path); fd.import_development(tmp_path)
    root=fd.imported_root(tmp_path)
    p=root/m['images'][0]['path']; p.unlink(); p.symlink_to(incoming/'dev-manifest.json')
    with pytest.raises(ValueError,match='changed or escaped'): fd.verify_import(root)


def test_detection_metrics_ties_counts_and_no_localization_substitute():
    rows=[dict(category=c,image_score=s) for c,s in [('good',0),('good',1),('logical_anomalies',1),('structural_anomalies',2)]]
    m=detection_metrics(rows,1)
    assert m['dev_real/logical_image_auroc']==.75
    assert m['dev_real/structural_image_auroc']==1
    assert m['dev_real/normal_false_positives']==0 and m['dev_real/normal_images']==2
    assert not any('pro' in k or 'pixel_ap' in k for k in m)
    assert detection_metrics(rows,0)['dev_real/normal_image_fpr']==.5
    for bad in (float('nan'),float('inf')):
        with pytest.raises(ValueError): detection_metrics(rows,bad)
        with pytest.raises(ValueError): detection_metrics(rows+[dict(category='good',image_score=bad)],1)


def finished(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite')
    rid=store.create_run('hypothesis','masked_context_ablation','juice_bottle',config={'seed':7})
    job=tmp_path/'jobs'/rid; job.mkdir(parents=True)
    for state in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'): store.transition(rid,state)
    store.set_run_result(rid,{'dev_synthetic/pixel_ap':.4},artifact_dir=job)
    store.transition(rid,'COMPLETED')
    output=job/'feedback'/'fixture';output.mkdir(parents=True)
    summary=dict(evaluation_id='fixture',evaluator_version=VERSION,label=fd.LABEL,split_id='fixture-dev',
        checkpoint_sha256='a'*64,localization_status='unavailable',localization_blocker='reference unavailable',
        artifact_dir=str(output),metrics={'dev_real/logical_image_auroc':.75,'dev_real/structural_image_auroc':.6,
                                        'dev_real/normal_false_positives':1,'dev_real/normal_images':47})
    (output/'summary.json').write_text(json.dumps(summary))
    return store,rid,summary


def test_durable_feedback_preserves_history_and_dashboard(tmp_path,monkeypatch):
    store,rid,summary=finished(tmp_path); original=store.get_run(rid)['metrics_json']
    record_summary(store,rid,summary); record_summary(store,rid,summary)
    assert latest_summary(store,rid)==summary
    assert len(store.pending_tracking())==2
    assert store.get_run(rid)['metrics_json']==original
    monkeypatch.setenv('LOCO_LEDGER',str(tmp_path/'ledger.sqlite'))
    from dashboard.data_access import real_feedback_summaries
    assert real_feedback_summaries()[0]['metrics']==summary['metrics']
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(Path(__file__).parents[1]/'dashboard/pages/03_Results.py').run()
    assert not app.exception
    assert any('Real-anomaly' in x.value for x in app.subheader)


def test_feedback_reaches_critic_not_candidate_inputs(tmp_path,monkeypatch):
    store,rid,summary=finished(tmp_path);record_summary(store,rid,summary)
    from orchestrator import controller
    calls=[]
    monkeypatch.setattr(controller,'invoke',lambda *a,**kw:calls.append(a) or {'response':{}})
    controller.budgeted_invoke(store,'critic',Path('.'),Path('schema'),'Critique')
    assert '0.75' in calls[0][3] and 'never use anomaly images/masks to fit' in calls[0][3]
    from benchmark.data_guard import validate_manifest,DataAccessError
    with pytest.raises(DataAccessError): validate_manifest({'schema_version':1,'partition':'development'})
    with pytest.raises(PermissionError): evaluate_run(tmp_path,store,rid)


def test_inference_state_and_alignment(tmp_path,monkeypatch):
    import torch
    from scripts import feedback_infer as fi
    class Predictor:
        def __init__(self):
            self.net=torch.nn.Conv2d(3,1,1); self.scale=torch.tensor(1.0)
        def predict(self,x):
            assert torch.is_inference_mode_enabled()
            y=self.net(torch.tensor(x)).detach().numpy()
            return {'anomaly_maps':y,'image_scores':y.max((1,2,3))}
    predictor=Predictor()
    # Contain the deliberate optimizer prohibition to this test.
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer): monkeypatch.setattr(cls,'step',cls.step)
    cp=tmp_path/'checkpoint_manifest.json';cp.write_text('{}');(tmp_path/'model.pt').write_bytes(b'frozen')
    image=tmp_path/'image.png';Image.new('RGB',(80,160),(100,50,30)).save(image)
    monkeypatch.setattr(fi,'load_candidate',lambda p:types.SimpleNamespace(load=lambda *a,**kw:predictor))
    fi.infer_one('fixture',cp,image,16,tmp_path)
    assert np.load(tmp_path/'map.npy').shape==(160,80)
    meta=json.loads((tmp_path/'inference.json').read_text())
    assert meta['state_before_sha256']==meta['state_after_sha256']
    assert (tmp_path/'model.pt').read_bytes()==b'frozen'
    original=predictor.predict
    def mutate(x):
        predictor.scale.add_(1);return original(x)
    predictor.predict=mutate
    with pytest.raises(ValueError,match='state changed'): fi.infer_one('fixture',cp,image,16,tmp_path)


def test_feedback_mlflow_retry_preserves_original_values(tmp_path,monkeypatch):
    import sys
    from orchestrator.mlflow_tracker import sync_outbox
    store,rid,summary=finished(tmp_path);record_summary(store,rid,summary)
    calls=[]; tags={}; down=[True]
    class Client:
        def __init__(self,tracking_uri): pass
        def get_experiment_by_name(self,name):
            if down[0]: raise ConnectionError('offline')
            return types.SimpleNamespace(experiment_id='1')
        def search_runs(self,*a,**kw): return []
        def create_run(self,*a,**kw): return types.SimpleNamespace(info=types.SimpleNamespace(run_id='same-remote'))
        def log_param(self,*a): pass
        def log_metric(self,r,k,v): calls.append((k,v))
        def set_tag(self,r,k,v): tags[k]=v
        def log_artifact(self,*a,**kw): pass
        def set_terminated(self,*a,**kw): pass
    ml=types.ModuleType('mlflow');ml.set_tracking_uri=lambda x:None
    tracking=types.ModuleType('mlflow.tracking');tracking.MlflowClient=Client
    monkeypatch.setitem(sys.modules,'mlflow',ml);monkeypatch.setitem(sys.modules,'mlflow.tracking',tracking)
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    assert sync_outbox(store,'http://fixture')['failed']==1
    down[0]=False
    assert sync_outbox(store,'http://fixture')['delivered']==2
    assert sync_outbox(store,'http://fixture')['delivered']==0
    assert ('dev_real/logical_image_auroc',.75) in calls
    assert tags['learning.real_anomaly_fitting']=='forbidden'
    assert json.loads(store.get_run(rid)['metrics_json'])=={'dev_synthetic/pixel_ap':.4}


def test_training_interface_receives_train_good_only(tmp_path,monkeypatch):
    from scripts.demo_run import make_fixture
    from benchmark.data_guard import build_manifest
    from scripts import run_candidate as runner
    make_fixture(tmp_path/'good')
    manifest=build_manifest(tmp_path/'good')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    observed=[]
    class StopAfterBoundary(Exception): pass
    def train(config,path,output):
        observed.extend(json.loads(path.read_text())['images'])
        raise StopAfterBoundary()
    monkeypatch.setattr(runner,'load_candidate',lambda p:types.SimpleNamespace(train=train))
    with pytest.raises(StopAfterBoundary): runner.run('fixture',mp,{'size':16},tmp_path/'output')
    assert observed and all(e['split']=='train/good' for e in observed)
    assert len(observed)<len(manifest['images'])


def test_checkpoint_mutation_rejected(tmp_path,monkeypatch):
    import torch
    from scripts import feedback_infer as fi
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer): monkeypatch.setattr(cls,'step',cls.step)
    class Predictor:
        def __init__(self): self.net=torch.nn.Conv2d(3,1,1)
        def predict(self,x):
            (tmp_path/'model.pt').write_bytes(b'mutated')
            return {'anomaly_maps':np.zeros((1,1,16,16)), 'image_scores':np.zeros(1)}
    cp=tmp_path/'checkpoint_manifest.json';cp.write_text('{}');(tmp_path/'model.pt').write_bytes(b'original')
    im=tmp_path/'image.png';Image.new('RGB',(80,160)).save(im)
    monkeypatch.setattr(fi,'load_candidate',lambda p:types.SimpleNamespace(load=lambda *a,**kw:Predictor()))
    with pytest.raises(ValueError,match='checkpoint changed'): fi.infer_one('fixture',cp,im,16,tmp_path)


def test_evaluation_mounts_only_image_not_masks(tmp_path):
    from orchestrator.sandbox import command
    repo=Path(__file__).parents[1]
    empty=tmp_path/'empty';empty.mkdir()
    image=tmp_path/'image.png';image.write_bytes(b'fixture')
    cmd=command(repo,repo/'candidates/toy_normal_ae/model.py',empty,tmp_path,
                ['-m','scripts.feedback_infer','--size','16'],input_files={'image.png':image})
    assert '--unshare-all' in cmd and '/inputs/image.png' in cmd
    assert '/inputs/mask.png' not in cmd and '--dev-bind' not in cmd
    with pytest.raises(ValueError,match='unapproved sandbox input'):
        command(repo,repo/'candidates/toy_normal_ae/model.py',empty,tmp_path,['-c','pass'],input_files={'mask.png':image})


def test_feedback_campaign_stays_unauthorized():
    from orchestrator.autonomy import load_campaign_config,preflight
    config=load_campaign_config(Path(__file__).parents[1]/'configs/autonomy_real_feedback.yaml')
    assert config['research']['real_development_feedback'] is True
    assert config['limits']['max_experiments']==2
    assert config['security']['operator_accepts_shared_uid_risk'] is False
    with pytest.raises(PermissionError): preflight(config,Path('/nonexistent'))


def test_candidate_output_symlink_is_not_read(tmp_path):
    from orchestrator.development_feedback import validate_worker_outputs
    folder=tmp_path/'output';folder.mkdir()
    sentinel=tmp_path/'private';sentinel.write_text('dummy')
    (folder/'map.npy').symlink_to(sentinel)
    (folder/'inference.json').write_text('{}')
    with pytest.raises(ValueError,match='unsafe'): validate_worker_outputs(folder,800,1600)


def test_campaign_automatically_evaluates_before_critique(tmp_path,monkeypatch):
    from orchestrator import autonomy,controller,experiment_runner,codex_runner,development_feedback
    from orchestrator.campaign_state import CampaignState
    cfg=autonomy.load_campaign_config(Path(__file__).parents[1]/'configs/autonomy_real_feedback.yaml')
    cfg['research']['fresh_hypotheses']=False
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store);cid=campaigns.create(cfg)
    store.add_proposal({'hypothesis_id':'fixture_h','priority_rank':1,'new_dependencies':[]})
    source=tmp_path/'model.py';source.write_text('# fixture')
    order=[]
    monkeypatch.setenv('LOCO_GOOD_ROOT',str(tmp_path))
    monkeypatch.setattr(controller,'implement_proposal',lambda *a,**kw:{'candidate_module':str(source)})
    monkeypatch.setattr(experiment_runner,'candidate_file',lambda *a:source)
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',lambda *a,**kw:None)
    monkeypatch.setattr(experiment_runner,'run_candidate_unit_tests',lambda *a,**kw:{})
    monkeypatch.setattr(codex_runner,'commit_candidate_snapshot',lambda *a:'fixture-sha')
    def queue(*a,**kw):
        rid=store.create_run('fixture_h','fixture_h','juice_bottle',config={'campaign_id':cid})
        for state in ('IMPLEMENTED','VALIDATED','QUEUED'): store.transition(rid,state)
        return rid
    monkeypatch.setattr(controller,'queue_candidate',queue)
    def worker(*a,**kw):
        rid=kw['expected_run_id'];store.transition(rid,'RUNNING');store.set_run_result(rid,{'dev_synthetic/pixel_ap':.3})
        store.transition(rid,'COMPLETED');order.append('training');return {'run_id':rid,'status':'COMPLETED'}
    monkeypatch.setattr(controller,'run_worker_once',worker)
    def evaluate(runtime,target,rid,**kw):
        assert kw['timeout']<=600 and kw['personal_mode_authorized']
        order.append('feedback');summary={'evaluation_id':'fixture','metrics':{'dev_real/logical_image_auroc':.51}}
        record_summary(store,rid,summary);return summary
    monkeypatch.setattr(development_feedback,'evaluate_run',evaluate)
    def critic(*a,**kw):
        assert order==['training','feedback'];order.append('critic')
        return {'response':{'recommended_next_action':'investigate','objective_measured_results':'near chance development detection'}}
    monkeypatch.setattr(controller,'budgeted_invoke',critic)
    autonomy.run_cycle(store,campaigns,cid,tmp_path)
    assert order==['training','feedback','critic']
    assert campaigns.get(cid)['next_hypothesis']=='evidence-followup:fixture_h:investigate'


def test_result_card_uses_real_values_and_explicit_missing_measurements():
    from dashboard.presentation import development_card_metrics
    summary={'inference_device':'cpu','metrics':{'dev_real/logical_image_auroc':.50644,
             'dev_real/normal_false_positives':1,'dev_real/normal_images':47,
             'dev_real/inference_e2e_ms_p95':90.955,'dev_real/parameters':3420,
             'dev_synthetic/pixel_ap':.99,'inference/peak_vram_mib':123}}
    values=dict(development_card_metrics(summary))
    assert values['Logical AUROC']=='0.51'
    assert values['Logical pixel AP']=='Unavailable'
    assert values['Normal false positives']=='1 / 47'
    assert values['Inference p95 · CPU']=='91.0 ms'
    assert values['Peak inference VRAM']=='Unavailable · CPU'


def test_real_preview_ground_truth_is_verified_display_only(tmp_path,monkeypatch):
    from benchmark.data_guard import sha256_file
    from dashboard.data_access import real_heatmap_sample
    incoming=tmp_path/'incoming/juice_bottle_dev_v1';incoming.mkdir(parents=True)
    manifest,approval=package(incoming);monkeypatch.setattr(fd,'approval',lambda:approval)
    fd.import_development(tmp_path)
    output=tmp_path/'jobs/fixture/feedback/eval';(output/'case_0001').mkdir(parents=True)
    path=output/'case_0001/map.npy';np.save(path,np.zeros((1600,800),dtype=np.float32))
    (output/'case_scores.json').write_text(json.dumps([{},dict(map_sha256=sha256_file(path),image_score=0.)]))
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    sample=real_heatmap_sample({'run_id':'fixture','evaluation_id':'eval'},1)
    assert sample['mask_count']==2
    assert sample['original'].size==sample['heatmap'].size==sample['ground_truth'].size==(800,1600)
    assert np.asarray(sample['ground_truth']).min()==255
    target=fd.imported_root(tmp_path)/manifest['images'][1]['masks'][0]['path']
    target.unlink();target.symlink_to(incoming/'dev-manifest.json')
    with pytest.raises(ValueError,match='ground truth'): real_heatmap_sample({'run_id':'fixture','evaluation_id':'eval'},1)
