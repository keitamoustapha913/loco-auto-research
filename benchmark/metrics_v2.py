"""Version 2 development metrics. Synthetic masks are proxy annotations, never LOCO GT.

Version 1 remains in :mod:`benchmark.metrics` for historical reproducibility.
Higher scores always mean more anomalous. No pixel is treated as an independent
sample when computing an uncertainty interval across images.
"""
from __future__ import annotations

import math
import numpy as np

EVALUATOR_VERSION = "development-v2.0"


def aligned_binary_maps(maps, masks):
    scores = np.asarray(maps, dtype=np.float64)
    labels = np.asarray(masks)
    if scores.ndim != 3 or not all(scores.shape) or scores.shape != labels.shape:
        raise ValueError("expected aligned nonempty [N,H,W] score maps and masks")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("scores must be finite and masks must contain only 0/1")
    return scores, labels.astype(np.uint8)


def binary_average_precision(labels, scores):
    """Non-interpolated AP with tied scores grouped, matching sklearn's AP.

    An empty positive class is undefined (NaN). An all-positive sample is 1.
    These boundary policies are explicit; callers must report support counts.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or not y.size or not np.isfinite(s).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("expected finite scores and aligned binary labels")
    y, s = y.ravel().astype(np.uint8), s.ravel()
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    if positives == y.size:
        return 1.0
    order = np.argsort(-s, kind="stable")
    ranked_y, ranked_s = y[order], s[order]
    cumulative = np.cumsum(ranked_y, dtype=np.int64)
    ends = np.r_[np.flatnonzero(ranked_s[:-1] != ranked_s[1:]), y.size - 1]
    tp = cumulative[ends]
    return float(np.sum((tp / (ends + 1)) * (np.diff(np.r_[0, tp]) / positives)))


def pixel_ap_report(maps, masks):
    """Report micro (all pixels pooled) and macro (positive images equally weighted).

    Images without positives are omitted from macro AP and counted explicitly.
    Their normal-image false alarms belong in a separate good-only assessment.
    """
    scores, labels = aligned_binary_maps(maps, masks)
    counts = labels.reshape(len(labels), -1).sum(axis=1)
    per_image = [binary_average_precision(y, s) for y, s in zip(labels, scores)]
    positive = [value for value in per_image if math.isfinite(value)]
    return {
        "pooled": binary_average_precision(labels, scores),
        "macro_positive_images": float(np.mean(positive)) if positive else float("nan"),
        "per_image": per_image,
        "images_total": len(labels),
        "images_with_positive_pixels": len(positive),
        "images_without_positive_pixels": int(np.sum(counts == 0)),
        "positive_pixels": int(np.sum(counts)),
        "pixels_total": int(labels.size),
    }


def pixel_auroc(maps, masks):
    from benchmark.metrics import binary_auroc
    scores, labels = aligned_binary_maps(maps, masks)
    return binary_auroc(labels, scores)


def calibrate_image_threshold(calibration_scores, false_alarm_budget=0.05):
    """Deterministic upper empirical quantile with strict ``score > threshold``.

    For 18 calibration images and a 5% nominal budget this selects their
    maximum: calibration support is too small for a finer nonzero tail rate.
    This is not a guarantee of 5% population FPR.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("expected finite one-dimensional calibration image scores")
    if not 0 < false_alarm_budget < 1:
        raise ValueError("false alarm budget must lie strictly between zero and one")
    rank = min(len(scores), max(1, math.ceil((1 - false_alarm_budget) * len(scores))))
    threshold = float(np.sort(scores)[rank - 1])
    return {"threshold": threshold, "calibration_images": len(scores),
            "calibration_false_positives": int(np.count_nonzero(scores > threshold)),
            "nominal_false_alarm_budget": float(false_alarm_budget), "order_statistic_rank": rank}


def image_false_positive_report(assessment_scores, threshold):
    scores = np.asarray(assessment_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or not math.isfinite(threshold):
        raise ValueError("expected finite one-dimensional assessment scores and threshold")
    alarms = scores > threshold
    return {"false_positives": int(np.count_nonzero(alarms)), "images_assessed": len(scores),
            "image_fpr": float(np.mean(alarms)), "alarms": alarms.tolist()}


def wilson_interval(successes, trials, z=1.959963984540054):
    """Descriptive Bernoulli interval; independent-image assumption is limited."""
    if not isinstance(successes, int) or not isinstance(trials, int) or trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("valid integer event count and positive denominator required")
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def paired_image_difference(reference, alternative, *, seed=991, repeats=2000, clusters=None):
    """Fixed-seed paired bootstrap; resample images or declared shared-donor clusters.

    Training-seed variation must be reported separately. A one-seed interval
    does not establish a population effect or justify model promotion.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(alternative, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired finite per-image measurements required")
    if not isinstance(repeats, int) or repeats < 100:
        raise ValueError("at least 100 bootstrap repeats required")
    group = np.arange(a.size) if clusters is None else np.asarray(clusters)
    if group.shape != a.shape:
        raise ValueError("one cluster label per image required")
    labels = list(dict.fromkeys(group.tolist()))
    members = [np.flatnonzero(group == label) for label in labels]
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(members), size=(repeats, len(members)))
    values = np.asarray([np.mean(delta[np.concatenate([members[i] for i in draw])]) for draw in draws])
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"mean_difference": float(np.mean(delta)), "ci95": [float(lo), float(hi)],
            "images": int(a.size), "clusters": len(members), "bootstrap_repeats": repeats,
            "seed": seed, "unit": "image" if clusters is None else "declared_cluster"}
