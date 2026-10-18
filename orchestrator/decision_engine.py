"""Deterministic promotion guard; never treats synthetic results as official LOCO evidence."""
from __future__ import annotations
import json

REQUIRED = ("normal/image_fpr_at_threshold", "dev_synthetic/pixel_auroc", "inference/e2e_ms_p95")


def eligible(run, max_latency_ms=20.0, max_normal_fpr=0.10):
    if run.get("status") not in ("COMPLETED", "REVIEWED", "PROMOTED"):
        return False, "run not completed"
    metrics = json.loads(run.get("metrics_json", "{}"))
    if any(k not in metrics for k in REQUIRED):
        return False, "missing required development proxies"
    if metrics["normal/image_fpr_at_threshold"] > max_normal_fpr:
        return False, "normal-image false positive rate too high"
    if metrics["inference/e2e_ms_p95"] > max_latency_ms:
        return False, "latency exceeds operator target"
    return True, "eligible for further investigation (NOT official LOCO performance)"


def pareto_frontier(runs):
    """Max synthetic pixel-AUROC proxy, min latency; never promote on proxy alone."""
    valid = []
    for row in runs:
        ok, _ = eligible(row, max_latency_ms=float("inf"), max_normal_fpr=1.0)
        if ok:
            metrics = json.loads(row["metrics_json"])
            valid.append((row["run_id"], metrics["dev_synthetic/pixel_auroc"], metrics["inference/e2e_ms_p95"]))
    return [id_ for id_, quality, latency in valid if not any((q2 >= quality and l2 <= latency and (q2 > quality or l2 < latency)) for other, q2, l2 in valid if other != id_)]
