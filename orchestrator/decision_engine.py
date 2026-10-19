"""Deterministic promotion guard; never treats synthetic results as official LOCO evidence."""
from __future__ import annotations
import json

REQUIRED = ("normal/image_fpr_at_threshold", "dev_synthetic/pixel_auroc", "inference/e2e_ms_p95")


def eligible(run, max_latency_ms=20.0, max_normal_fpr=0.10):
    if run.get("status") not in ("COMPLETED", "REVIEWED", "PROMOTED"):
        return False, "run not completed"
    if run.get('real_feedback'):
        feedback = run['real_feedback']
        required = ('dev_real/logical_image_auroc', 'dev_real/structural_image_auroc',
                    'dev_real/logical_au_spro_0_05_compatible', 'dev_real/structural_au_spro_0_05_compatible')
        import math
        if not all(k in feedback['metrics'] and math.isfinite(feedback['metrics'][k]) for k in required):
            return False, 'incomplete real-development evidence'
        return True, 'measured real-development evidence available for adaptive research; not final evaluation'
    metrics = json.loads(run.get("metrics_json", "{}"))
    if any(k not in metrics for k in REQUIRED):
        return False, "missing required development proxies"
    if metrics["normal/image_fpr_at_threshold"] > max_normal_fpr:
        return False, "normal-image false positive rate too high"
    if metrics["inference/e2e_ms_p95"] > max_latency_ms:
        return False, "latency exceeds operator target"
    return True, "eligible for further investigation (NOT official LOCO performance)"


def pareto_frontier(runs, include_resources=False):
    """Development frontier; full mode also minimizes size and peak VRAM."""
    valid = []
    for row in runs:
        ok, _ = eligible(row, max_latency_ms=float("inf"), max_normal_fpr=1.0)
        if ok:
            metrics = json.loads(row["metrics_json"])
            if include_resources and not all(k in metrics for k in ('inference/parameters', 'inference/peak_vram_mib')):
                continue
            vector = (-metrics['dev_synthetic/pixel_auroc'], metrics['inference/e2e_ms_p95'])
            if include_resources:
                vector += (metrics['inference/parameters'], metrics['inference/peak_vram_mib'])
            valid.append((row['run_id'], vector))
    return [rid for rid, vector in valid if not any(
        other != rid and all(b <= a for a, b in zip(vector, competitor)) and
        any(b < a for a, b in zip(vector, competitor))
        for other, competitor in valid)]
