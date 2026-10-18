"""Immutable development evaluation: good-only and held-out synthetic proxies."""
from __future__ import annotations
import numpy as np
from benchmark.metrics import binary_auroc, false_positive_rate, normal_quantile_threshold, synthetic_pixel_auroc


def evaluate_dev(good_maps, synthetic_maps, synthetic_masks, target_normal_fpr=0.05, calibration_maps=None):
    good = np.asarray(good_maps, dtype=np.float64)
    bad = np.asarray(synthetic_maps, dtype=np.float64)
    masks = np.asarray(synthetic_masks, dtype=np.uint8)
    if good.ndim != 3 or bad.ndim != 3 or masks.shape != bad.shape or good.shape[1:] != bad.shape[1:]:
        raise ValueError("expected aligned arrays [N,H,W]")
    if min(len(good), len(bad)) == 0 or not np.isfinite(good).all() or not np.isfinite(bad).all():
        raise ValueError("empty/nonfinite maps")
    good_scores = good.reshape(len(good), -1).max(axis=1)
    bad_scores = bad.reshape(len(bad), -1).max(axis=1)
    if calibration_maps is not None:
        calibrated = np.asarray(calibration_maps, dtype=np.float64)
        if calibrated.ndim != 3 or calibrated.shape[1:] != good.shape[1:] or not len(calibrated) or not np.isfinite(calibrated).all():
            raise ValueError("calibration maps must be separate aligned normal set")
        cal_scores = calibrated.reshape(len(calibrated), -1).max(axis=1)
    else:
        cal_scores = good_scores  # tests only; production adapter passes disjoint maps
    threshold = normal_quantile_threshold(cal_scores, target_normal_fpr)
    results = {
        "normal/image_fpr_at_threshold": false_positive_rate(good_scores, threshold),
        "normal/threshold": threshold,
        "dev_synthetic/image_auroc": binary_auroc(np.r_[np.zeros(len(good)), np.ones(len(bad))], np.r_[good_scores, bad_scores]),
        "dev_synthetic/pixel_auroc": synthetic_pixel_auroc(bad, masks),
    }
    return results
