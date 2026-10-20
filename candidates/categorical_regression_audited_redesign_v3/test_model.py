"""Required CPU checks on artificial good-only fixtures; no CUDA or real feedback."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import model
from benchmark.data_guard import build_manifest
from scripts.feedback_infer import state_digest


@pytest.fixture(scope='module', autouse=True)
def cpu_only():
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), 'Candidate unit tests must use the no-GPU sandbox'


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('audited-normal-fixture')
    for split,count,offset in [('train/good',4,0),('validation/good',2,10)]:
        folder=root/'juice_bottle'/split;folder.mkdir(parents=True)
        for i in range(count):
            rng=np.random.default_rng(i+offset)
            Image.fromarray(rng.integers(0,256,(37,59,3),dtype=np.uint8)).save(folder/f'{i}.png')
    manifest=build_manifest(root)
    manifest['images']=[e for e in manifest['images'] if e['split']=='train/good']
    path=root/'training.json';path.write_text(json.dumps(manifest))
    return root,path,manifest


def settings(arm,seed=7,smoke=True):
    return {'device':'cpu','seed':seed,'size':128,'completion_grid':24,'completion_width':32,
            'vocabulary_bins':4,'objective':arm,'steps':2 if smoke else 120,
            'batch_size':1 if smoke else 8,'expected_optimizer_updates':2 if smoke else 120,
            'expected_source_exposures':2 if smoke else 960,'validation_smoke':smoke}


@pytest.fixture(scope='module')
def pairs(fixture):
    root,path,manifest=fixture
    result={}
    for seed in (7,19,31):
        for arm in ('categorical','deterministic'):
            folder=root/f'{arm}_{seed}'
            meta=model.train(settings(arm,seed),path,folder)
            result[arm,seed]=(folder,meta)
    return result


def test_all_three_seed_pairs_share_actual_source_masks_losses_and_weights(pairs):
    for seed in (7,19,31):
        cat,cat_meta=pairs['categorical',seed];det,det_meta=pairs['deterministic',seed]
        assert cat_meta['training_audit']==det_meta['training_audit']
        assert cat_meta['training_audit']['optimizer_updates']==2
        assert cat_meta['training_audit']['source_exposures']==2
        assert cat_meta['losses_ce_mse']==det_meta['losses_ce_mse']
        a=torch.load(cat/'model.pt',map_location='cpu',weights_only=True)
        b=torch.load(det/'model.pt',map_location='cpu',weights_only=True)
        for trunk in a:
            for key in a[trunk]:torch.testing.assert_close(a[trunk][key],b[trunk][key],rtol=0,atol=0)
        assert not torch.equal(a['categorical']['context.0.weight'],a['deterministic']['context.0.weight'])
        assert cat_meta['training_audit']['ordered_training_sources_sha256']
    assert pairs['categorical',7][1]['training_audit']['sampled_source_order_sha256']!=pairs['categorical',19][1]['training_audit']['sampled_source_order_sha256']


def test_original_trunk_and_readout_selected_without_inference_overrides(pairs):
    image=np.random.default_rng(999).random((2,3,37,59),dtype=np.float32)
    for arm in ('categorical','deterministic'):
        folder,meta=pairs[arm,7];predictor=model.load(folder/'checkpoint_manifest.json')
        assert predictor.objective==arm and predictor.device.type=='cpu'
        expected=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')[arm]
        for key,value in predictor.net.state_dict().items():torch.testing.assert_close(value,expected[key])
        result=predictor.predict(image)
        assert result['anomaly_maps'].shape==(2,1,37,59)
        assert result['image_scores'].shape==(2,)
        assert np.isfinite(result['anomaly_maps']).all()
        np.testing.assert_array_equal(result['image_scores'],result['anomaly_maps'].reshape(2,-1).max(1))
        np.testing.assert_array_equal(result['anomaly_maps'],predictor.predict(image)['anomaly_maps'])
        assert meta['source_sha256']==hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
        assert meta['origin_source_sha256']==model.ORIGIN_SOURCE_SHA256
        assert meta['model_sha256']==hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()


def test_full_declared_schedule_observes_120_updates_and_960_presentations(fixture):
    root,path,_=fixture
    from torch.optim.optimizer import register_optimizer_step_post_hook
    observed=[]
    hook=register_optimizer_step_post_hook(lambda optimizer,args,kwargs:observed.append(type(optimizer).__name__))
    try:meta=model.train(settings('categorical',smoke=False),path,root/'full-cpu-fixture')
    finally:hook.remove()
    assert len(observed)==120 and set(observed)=={'Adam'}
    assert meta['training_audit']['optimizer_updates']==120
    assert meta['training_audit']['source_exposures']==960
    assert len(meta['losses_ce_mse'])==120 and np.isfinite(meta['losses_ce_mse']).all()


@pytest.mark.parametrize('defect',['validation','anomaly','absolute_path','wrong_type'])
def test_training_split_and_provenance_rejection_before_any_optimizer_step(fixture,defect,monkeypatch):
    root,path,manifest=fixture;bad=copy.deepcopy(manifest)
    if defect=='validation':bad['images'][0]['split']='validation/good'
    elif defect=='anomaly':bad['images'][0]['split']='development/anomalies'
    elif defect=='absolute_path':bad['images'][0]['path']=str(root/bad['images'][0]['path'])
    else:bad['type']='feedback-evaluation-only'
    invalid=root/f'{defect}.json';invalid.write_text(json.dumps(bad))
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:pytest.fail('invalid inputs reached optimizer'))
    with pytest.raises(ValueError):model.train(settings('categorical'),invalid,root/f'bad-{defect}')


@pytest.mark.parametrize('value',[np.zeros((0,3,32,32),np.float32),np.zeros((1,1,32,32),np.float32),
                                  np.full((1,3,32,32),np.nan,np.float32)])
def test_invalid_prediction_inputs_are_rejected(pairs,value):
    folder,_=pairs['categorical',7]
    with pytest.raises(ValueError):model.load(folder/'checkpoint_manifest.json').predict(value)


def test_schedule_changes_are_rejected_not_silently_defaulted(fixture):
    root,path,_=fixture
    config=settings('categorical',smoke=False);config['steps']=119
    with pytest.raises(ValueError,match='explicit schedule'):model.train(config,path,root/'bad-schedule')
    config=settings('categorical');config['objective']='unknown'
    with pytest.raises(ValueError):model.train(config,path,root/'bad-objective')


def test_interior_boundary_masks_retain_background_negatives_and_alignment():
    mask=np.zeros((17,19),np.uint8);mask[3:12,4:13]=1
    interior,boundary=model._interior_and_boundary(mask,2)
    assert interior.sum()==25 and boundary.sum()==56
    assert not (interior & boundary).any()
    assert np.array_equal(interior|boundary,mask.astype(bool))
    interior_ap,boundary_ap,support=model._localization_parts(mask.astype(float),mask,2)
    assert interior_ap==1 and boundary_ap==1
    assert support=={'footprint_pixels':81,'interior_pixels':25,'boundary_pixels':56}
    a,b,_=model._localization_parts(np.zeros_like(mask,dtype=float),mask,2)
    assert 0<a<1 and 0<b<1 # Background negatives cannot be dropped to give trivial AP=1.


def test_diagnostics_fixed_generators_seeds_threshold_and_frozen_state(pairs,monkeypatch):
    from scripts import run_candidate
    folder,_=pairs['categorical',7];predictor=model.load(folder/'checkpoint_manifest.json')
    images=np.random.default_rng(97).random((2,3,32,32),dtype=np.float32)
    ids=['independent-fixture-source-A','independent-fixture-source-B']
    config=dict(model.DIAGNOSTIC_SETTINGS,diagnostic_id='within_source_proxy',
                assessment_source_ids=ids,frozen_normal_threshold=-1.)
    seeds={'primary':[],'alternate':[]}
    primary,alternate=run_candidate.synthetic_corruption,run_candidate.alternate_corruption
    def first(x,seed):seeds['primary'].append(seed);return primary(x,seed)
    def second(x,seed):seeds['alternate'].append(seed);return alternate(x,seed)
    monkeypatch.setattr(run_candidate,'synthetic_corruption',first)
    monkeypatch.setattr(run_candidate,'alternate_corruption',second)
    before=state_digest(predictor);files={p.name:model._hash(p) for p in folder.iterdir()}
    with torch.inference_mode():result=model.diagnose(predictor,images,config)
    assert seeds=={'primary':[5000,5001],'alternate':[7000,7001]}
    assert set(result['metrics'])=={'primary_interior_ap','alternate_interior_ap','primary_boundary_ap',
        'alternate_boundary_ap','photometric_alarm','photometric_false_positives','assessment_sources'}
    assert result['metrics']['photometric_false_positives']==2
    assert result['metrics']['photometric_alarm']==1
    assert result['metrics']['assessment_sources']==2
    assert [r['source_id'] for r in result['per_image']]==ids
    assert all(np.isfinite(list(r['metrics'].values())).all() for r in result['per_image'])
    assert state_digest(predictor)==before and {p.name:model._hash(p) for p in folder.iterdir()}==files
    with torch.inference_mode():
        config['frozen_normal_threshold']=1e9
        assert model.diagnose(predictor,images,config)['metrics']['photometric_false_positives']==0
        config['photometric_offset']=0.2
        with pytest.raises(ValueError,match='diagnostic settings'):model.diagnose(predictor,images,config)
