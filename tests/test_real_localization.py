import numpy as np
import pytest
from benchmark.feedback_metrics import saturated_region_curve


def brute(background, regions, limit=.05):
    # Independent direct set-overlap fixture oracle (published Eq. 1).
    thresholds=np.r_[np.inf, np.unique(np.concatenate([background]+[r[0] for r in regions]))[::-1]]
    x=np.array([np.mean(np.asarray(background)>=t) for t in thresholds])
    y=np.array([np.mean([min(np.count_nonzero(np.asarray(v)>=t)/s,1) for v,s in regions]) for t in thresholds])
    end=np.searchsorted(x,limit,side='right')
    if end<len(x):
        y=np.r_[y[:end],np.interp(limit,x[end-1:end+1],y[end-1:end+1])];x=np.r_[x[:end],limit]
    return np.trapezoid(y,x)/limit


def test_perfect_and_reversed_spro():
    assert saturated_region_curve([0]*100,[([1,1],1)])==1
    assert saturated_region_curve([1]*100,[([0,0],2)])==0


def test_saturation_multiple_regions_and_ties():
    # One region saturates at one pixel, the other needs both; equal region weights.
    regions=[([3,0],1),([2,0],2)]
    bg=[1]*100
    assert saturated_region_curve(bg,regions)==pytest.approx(.75)
    for seed in range(12):
        rng=np.random.default_rng(seed)
        bg=rng.integers(0,8,100)
        rs=[(rng.integers(0,8,10),2.5),(rng.integers(0,8,7),7)]
        assert saturated_region_curve(bg,rs)==pytest.approx(brute(bg,rs),abs=1e-12)


@pytest.mark.parametrize('regions',[[],[([1,2],0)],[([1,2],3)],[([np.nan],1)]])
def test_invalid_regions(regions):
    with pytest.raises(ValueError): saturated_region_curve([0,1],regions)


def test_mask_instances_separate_from_union_pixel_diagnostics(tmp_path):
    import json
    from PIL import Image
    from benchmark.feedback_metrics import localization_metrics
    root=tmp_path/'data';(root/'juice_bottle').mkdir(parents=True)
    (root/'juice_bottle/defects_config.json').write_text(json.dumps([
        {'pixel_value':1,'relative_saturation':True,'saturation_threshold':1},
        {'pixel_value':2,'relative_saturation':False,'saturation_threshold':1}]))
    entries=[]
    for i,cat in enumerate(('good','logical_anomalies','structural_anomalies')):
        folder=tmp_path/f'case_{i:04d}';folder.mkdir()
        score=np.zeros((2,2),np.float32);mask_records=[]
        if cat!='good':
            score[0,:]=2
            for j in range(2 if cat=='logical_anomalies' else 1):
                mask=np.zeros((2,2),np.uint8);mask[0,j]=j+1
                path=f'mask_{i}_{j}.png';Image.fromarray(mask).save(root/path)
                mask_records.append({'path':path,'pixel_values':[j+1]})
        np.save(folder/'map.npy',score)
        entries.append({'category':cat,'width':2,'height':2,'masks':mask_records})
    metrics=localization_metrics({'images':entries},root,tmp_path)
    assert metrics['dev_real/logical_annotation_regions']==2
    assert metrics['dev_real/structural_annotation_regions']==1
    assert metrics['dev_real/logical_au_spro_0_05_compatible']==1
    assert metrics['dev_real/logical_pixel_ap_pooled']==1
    assert metrics['dev_real/logical_pixel_auroc_pooled']==1
    np.save(tmp_path/'case_0001/map.npy',np.zeros((3,2)))
    with pytest.raises(ValueError,match='unaligned'): localization_metrics({'images':entries},root,tmp_path)


def test_normal_only_runner_does_not_require_synthetic_scores(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from scripts.demo_run import make_fixture
    from benchmark.data_guard import build_manifest
    from scripts import run_candidate
    make_fixture(tmp_path/'good')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(build_manifest(tmp_path/'good')))
    def forbidden(*a,**kw): raise AssertionError('synthetic generator should not run')
    monkeypatch.setattr(run_candidate,'synthetic_corruption',forbidden)
    result=run_candidate.run(Path(__file__).parents[1]/'candidates/toy_normal_ae/model.py',mp,
        {'device':'cpu','size':16,'epochs':1,'synthetic_evaluation':False},tmp_path/'output')
    assert 'normal/threshold' in result['metrics']
    assert not any(k.startswith('dev_synthetic/') for k in result['metrics'])
    assert result['provenance']['metric_status']=='NORMAL_ONLY_CALIBRATION_AND_EFFICIENCY'


def test_cuda_timing_rejects_unavailable_device(monkeypatch):
    import torch
    from benchmark.inference_benchmark import benchmark_predict
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA timing'):
        benchmark_predict(None,np.zeros((1,3,4,4)),device='cuda')


def test_real_evidence_does_not_require_synthetic_promotion_scores():
    from orchestrator.decision_engine import eligible
    m={f'dev_real/{kind}_{metric}':.4 for kind in ('logical','structural') for metric in ('image_auroc','au_spro_0_05_compatible')}
    assert eligible({'status':'COMPLETED','metrics_json':'{}','real_feedback':{'metrics':m}})[0]


def test_trusted_scoring_obeys_timeout_and_cancellation(tmp_path):
    import sys
    from orchestrator.development_feedback import run_bounded
    with pytest.raises(TimeoutError):
        run_bounded([sys.executable,'-c','import time; time.sleep(5)'],tmp_path/'timeout.log',.1)
    with pytest.raises(RuntimeError,match='cancelled'):
        run_bounded([sys.executable,'-c','import time; time.sleep(5)'],tmp_path/'cancel.log',5,lambda:True)
