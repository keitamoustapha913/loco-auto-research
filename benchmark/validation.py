"""Immutable development evaluation: good-only and held-out synthetic proxies."""
from __future__ import annotations
import numpy as np
from benchmark.metrics import binary_auroc, false_positive_rate, normal_quantile_threshold, synthetic_pixel_auroc
from benchmark.metrics_v2 import (EVALUATOR_VERSION, aligned_binary_maps,
    calibrate_image_threshold, image_false_positive_report, pixel_ap_report, pixel_auroc)


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


def evaluate_dev_v2(good_maps, synthetic_maps, synthetic_masks, *, calibration_maps,
                    target_normal_fpr=0.05):
    """Versioned good-only threshold and synthetic development evaluation.

    Calibration is mandatory and must be a distinct image set. The caller owns
    path-level disjointness; the evaluator rejects same array objects and records
    sample counts. Historical v1 ``evaluate_dev`` remains unchanged.
    """
    good = np.asarray(good_maps, dtype=np.float64)
    bad = np.asarray(synthetic_maps, dtype=np.float64)
    cal = np.asarray(calibration_maps, dtype=np.float64)
    bad, labels = aligned_binary_maps(bad, synthetic_masks)
    if good.ndim != 3 or cal.ndim != 3 or not len(good) or not len(cal) or not all(good.shape[1:]) or good.shape[1:] != bad.shape[1:] or cal.shape[1:] != bad.shape[1:]:
        raise ValueError("expected aligned nonempty normal, calibration and synthetic [N,H,W] maps")
    if good_maps is calibration_maps or not np.isfinite(good).all() or not np.isfinite(cal).all():
        raise ValueError("calibration must be separate finite normal maps")
    cal_scores = cal.reshape(len(cal), -1).max(axis=1)
    good_scores = good.reshape(len(good), -1).max(axis=1)
    bad_scores = bad.reshape(len(bad), -1).max(axis=1)
    calibration = calibrate_image_threshold(cal_scores, target_normal_fpr)
    assessment = image_false_positive_report(good_scores, calibration["threshold"])
    ap = pixel_ap_report(bad, labels)
    metrics = {
        "normal/threshold": calibration["threshold"],
        "normal/image_fpr_at_threshold": assessment["image_fpr"],
        "normal/image_false_positives": assessment["false_positives"],
        "normal/image_assessed": assessment["images_assessed"],
        "normal/calibration_images": calibration["calibration_images"],
        "normal/calibration_false_positives": calibration["calibration_false_positives"],
        "dev_synthetic/image_auroc": binary_auroc(
            np.r_[np.zeros(len(good), dtype=np.uint8), np.ones(len(bad), dtype=np.uint8)],
            np.r_[good_scores, bad_scores]),
        "dev_synthetic/pixel_auroc": pixel_auroc(bad, labels),
        "dev_synthetic/pixel_ap": ap["pooled"],
        "dev_synthetic/pixel_ap_macro_positive_images": ap["macro_positive_images"],
        "dev_synthetic/ap_positive_images": ap["images_with_positive_pixels"],
        "dev_synthetic/ap_images_total": ap["images_total"],
    }
    metadata = {"metric_evaluator_version": EVALUATOR_VERSION,
                "ap_aggregation": "pooled pixels and equal-weight positive-image macro",
                "no_positive_macro_policy": "omit and count", "positive_score_direction": "higher_is_more_anomalous",
                "threshold_rule": "upper empirical order statistic, strict greater-than alarm",
                "threshold_calibration": calibration, "normal_assessment": assessment}
    return metrics, metadata
