"""Development-only metrics. NOT the official MVTec LOCO AD scorer."""
from __future__ import annotations
import numpy as np


def binary_auroc(labels, scores):
    """Binary AUROC with proper treatment of ties; NaN for undefined class mix."""
    y = np.asarray(labels, dtype=np.int8).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size == 0 or y.size != s.size or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("invalid labels or scores")
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    v = s[order]
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[j] == v[i]:
            j += 1
        ranks[order[i:j]] = ((i+1) + j) / 2
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normal_quantile_threshold(normal_scores, false_alarm_budget=0.05):
    arr = np.asarray(normal_scores, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).all() or not 0 < false_alarm_budget < 1:
        raise ValueError("invalid normal calibration scores or FPR budget")
    return float(np.quantile(arr, 1 - false_alarm_budget))


def false_positive_rate(normal_scores, threshold):
    s = np.asarray(normal_scores, dtype=float)
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("invalid normal scores")
    return float(np.mean(s > threshold))


def synthetic_pixel_auroc(maps, masks):
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks)
    if maps.shape != masks.shape or maps.ndim < 2:
        raise ValueError("map/mask spatial mismatch")
    return binary_auroc(masks.astype(np.int8), maps)


def synthetic_region_pro_proxy(maps, masks, max_fpr=0.05, steps=80):
    """Approximate per-region overlap integrated over <=max_fpr of normal pixels.

    This is intentionally NOT official LOCO AU-sPRO; the official evaluator uses
    special logical anomaly saturation rules and belongs in a sealed process.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("Install scipy to calculate synthetic PRO proxy") from exc
    m = np.asarray(maps, dtype=np.float64)
    g = np.asarray(masks, dtype=bool)
    if m.shape != g.shape or m.ndim != 3 or not np.isfinite(m).all():
        raise ValueError("expected aligned [N,H,W] predictions and binary masks")
    normal = m[~g]
    if normal.size == 0:
        return float("nan")
    regions = []
    for i in range(m.shape[0]):
        components, count = ndimage.label(g[i])
        for k in range(1, count+1):
            regions.append((i, components == k))
    if not regions:
        return float("nan")
    levels = np.linspace(0, max_fpr, steps)
    overlaps = []
    for fpr in levels:
        threshold = np.quantile(normal, 1 - fpr) if fpr > 0 else np.max(normal)
        overlaps.append(np.mean([np.mean(m[i][mask] > threshold) for i, mask in regions]))
    return float(np.trapezoid(overlaps, levels) / max_fpr) if hasattr(np, "trapezoid") else float(np.trapz(overlaps, levels) / max_fpr)
