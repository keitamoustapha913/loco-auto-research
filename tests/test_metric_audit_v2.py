import json

import numpy as np
import pytest

from benchmark.metrics_v2 import (EVALUATOR_VERSION, aligned_binary_maps,
    binary_average_precision, calibrate_image_threshold, image_false_positive_report,
    paired_image_difference, pixel_ap_report, pixel_auroc)
from benchmark.metrics_v2 import wilson_interval
from benchmark.validation import evaluate_dev_v2
from scripts.run_candidate import (alternate_corruption, paired_patch_edits,
    restore_anomaly_map, synthetic_corruption)


def test_ap_known_ranking_ties_and_reference():
    reference = pytest.importorskip('sklearn.metrics')
    for y, s, expected in [([1, 0, 1, 0], [.9, .8, .4, .1], 5 / 6),
                            ([1, 0], [.5, .5], .5),
                            ([0, 1, 0, 1], [.1, .9, .2, .8], 1)]:
        assert binary_average_precision(y, s) == pytest.approx(expected)
        assert binary_average_precision(y, s) == pytest.approx(reference.average_precision_score(y, s))


def test_pooled_and_macro_have_different_estimands():
    maps = np.array([[[.9, .1]], [[.4, .8]]])
    masks = np.array([[[1, 0]], [[1, 0]]])
    report = pixel_ap_report(maps, masks)
    assert report['pooled'] == pytest.approx(5 / 6)
    assert report['macro_positive_images'] == pytest.approx(.75)
    assert report['images_total'] == report['images_with_positive_pixels'] == 2
    assert report['positive_pixels'] == 2 and report['pixels_total'] == 4


def test_ap_empty_positive_and_all_positive_are_explicit():
    assert np.isnan(binary_average_precision([0, 0], [1, 2]))
    assert binary_average_precision([1, 1], [1, 2]) == 1
    maps = np.array([[[.9, .1]], [[.4, .8]]])
    masks = np.array([[[1, 0]], [[0, 0]]])
    result = pixel_ap_report(maps, masks)
    assert np.isnan(result['per_image'][1])
    assert result['images_without_positive_pixels'] == 1
    assert result['macro_positive_images'] == 1


@pytest.mark.parametrize('bad', [2, -1, np.nan, np.inf, .4])
def test_invalid_masks_cannot_be_cast_to_binary(bad):
    with pytest.raises(ValueError):
        aligned_binary_maps(np.zeros((1, 2, 2)), np.array([[[0, 1], [bad, 0]]]))


def test_invalid_scores_shapes_and_direction():
    with pytest.raises(ValueError):
        pixel_ap_report(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)))
    with pytest.raises(ValueError):
        binary_average_precision([0, 1], [np.nan, 1])
    with pytest.raises(ValueError):
        binary_average_precision([0, 1], [0, np.inf])
    assert pixel_auroc(np.array([[[.1, .9]]]), np.array([[[0, 1]]])) == 1


def test_threshold_counts_and_small_calibration_boundary():
    calibration = calibrate_image_threshold(np.arange(18), .05)
    assert calibration['threshold'] == 17
    assert calibration['calibration_images'] == 18
    assert calibration['calibration_false_positives'] == 0
    assessment = image_false_positive_report([16, 17, 18, 19], calibration['threshold'])
    assert assessment == {'false_positives': 2, 'images_assessed': 4,
                          'image_fpr': .5, 'alarms': [False, False, True, True]}
    assert calibrate_image_threshold(np.arange(101), .05)['calibration_false_positives'] == 5
    lower, upper = wilson_interval(5, 36)
    assert lower < 5 / 36 < upper
    assert 0 <= lower < upper <= 1


def test_v2_requires_separate_calibration_and_records_version():
    normal = np.zeros((2, 2, 2)); cal = np.ones((2, 2, 2)) * .1
    changed = np.array([[[1., 0.], [0., 0.]]])
    masks = np.array([[[1, 0], [0, 0]]])
    metrics, provenance = evaluate_dev_v2(normal, changed, masks, calibration_maps=cal)
    assert provenance['metric_evaluator_version'] == EVALUATOR_VERSION
    assert metrics['normal/image_false_positives'] == 0
    assert metrics['normal/image_assessed'] == 2
    assert metrics['dev_synthetic/pixel_ap'] == 1
    with pytest.raises(ValueError, match='separate'):
        evaluate_dev_v2(normal, changed, masks, calibration_maps=normal)


def test_generators_reveal_shared_geometry_and_controls_are_unlabelled():
    rng = np.random.default_rng(9)
    good = rng.random((3, 64, 64), dtype=np.float32)
    donor = rng.random((3, 64, 64), dtype=np.float32)
    primary, primary_mask = synthetic_corruption(good, 1)
    alternate, alternate_mask = alternate_corruption(good, 1)
    copy, donor_control, photo_control, copy_mask = paired_patch_edits(good, donor, 1)
    assert not np.array_equal(primary_mask, alternate_mask)
    # Same seed/size gives identical target geometry: these are different edit
    # appearances, not independent logical anomaly families.
    assert np.array_equal(alternate_mask, copy_mask)
    assert not np.array_equal(alternate, copy)
    for edited, mask in ((primary, primary_mask), (alternate, alternate_mask), (copy, copy_mask),
                         (donor_control, copy_mask), (photo_control, copy_mask)):
        assert edited.shape == good.shape and mask.shape == good.shape[1:]
        assert np.array_equal(edited[:, mask == 0], good[:, mask == 0])
        assert np.any(edited[:, mask == 1] != good[:, mask == 1])
    # The two controls use an edit footprint but receive no anomaly label.
    assert np.any(donor_control != good) and np.any(photo_control != good)


def test_original_grid_restoration_and_nearest_mask_labels():
    from PIL import Image
    map_small = np.zeros((128, 128), dtype=np.float32)
    map_small[32:64, 48:80] = 1
    restored = restore_anomaly_map(map_small, (800, 1600))
    assert restored.shape == (1600, 800)
    assert restored[600, 400] > restored[100, 100]
    mask = Image.fromarray((map_small * 255).astype(np.uint8)).resize((800, 1600), Image.Resampling.NEAREST)
    assert set(np.unique(np.asarray(mask)).tolist()) == {0, 255}


def test_paired_bootstrap_resamples_images_not_pixels():
    result = paired_image_difference([.4, .5, .3, .6], [.3, .4, .2, .5],
                                     clusters=[0, 0, 1, 1], repeats=100)
    assert result['mean_difference'] == pytest.approx(.1)
    assert result['ci95'] == pytest.approx([.1, .1])
    assert result['images'] == 4 and result['clusters'] == 2


def test_historical_ledger_is_not_migrated_by_v2_evaluation(tmp_path):
    from orchestrator.state_store import StateStore
    store = StateStore(tmp_path / 'ledger.sqlite')
    rid = store.create_run('historical', 'candidate', 'fixture', config={'metric_evaluator_version': 'legacy-v1'})
    before = json.dumps(store.get_run(rid), sort_keys=True)
    pixel_ap_report(np.array([[[.9, .1]]]), np.array([[[1, 0]]]))
    assert json.dumps(store.get_run(rid), sort_keys=True) == before
