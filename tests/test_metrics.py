import numpy as np
import pytest
from benchmark.train_adapter import validate_prediction, CandidateContractError
from benchmark.metrics import binary_auroc, normal_quantile_threshold, false_positive_rate, synthetic_region_pro_proxy
from benchmark.validation import evaluate_dev


def test_auroc_perfect():
    assert binary_auroc([0,1,0,1], [0.1,0.9,0.2,0.8]) == 1.0


def test_auroc_ties():
    assert binary_auroc([0,1], [0.5,0.5]) == 0.5


def test_auroc_no_pos_nan():
    assert np.isnan(binary_auroc([0,0], [1,2]))


def test_calibration_fpr():
    val = normal_quantile_threshold(np.arange(101), 0.05)
    assert 0 <= false_positive_rate(np.arange(101), val) <= 0.05 + 1/101


def test_prediction_contract_valid():
    m, s = validate_prediction({'anomaly_maps':np.ones((2,1,4,5),dtype=np.float32), 'image_scores':np.ones(2,dtype=np.float32)}, (2,3,4,5))
    assert m.shape == (2,1,4,5) and s.shape == (2,)

@pytest.mark.parametrize('failure', ['shape', 'nan', 'integer', 'empty'])
def test_prediction_contract_rejects(failure):
    maps = np.ones((2,1,4,5),dtype=np.float32)
    scores = np.ones((2,),dtype=np.float32)
    shape = (2,3,4,5)
    if failure == 'shape': maps = np.ones((2,1,4,4),dtype=np.float32)
    if failure == 'nan': scores[0] = np.nan
    if failure == 'integer': maps = maps.astype(np.int32)
    if failure == 'empty': shape = (0,3,4,5)
    with pytest.raises(CandidateContractError):
        validate_prediction({'anomaly_maps':maps, 'image_scores':scores}, shape)


def test_dev_only_prefixes():
    good = np.zeros((2,8,8)); bad = np.zeros((2,8,8)); bad[:,2:4,2:4] = 1
    mask = bad.copy()
    score = evaluate_dev(good,bad,mask)
    assert all(k.startswith(('normal/','dev_synthetic/')) for k in score)
    assert score['dev_synthetic/pixel_auroc'] == 1.0


def test_proxy_invalid_spatial_shape():
    with pytest.raises(ValueError):
        synthetic_region_pro_proxy(np.zeros((2,4,4)),np.zeros((2,5,5)))
