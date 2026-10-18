"""Plotting helpers; the dashboard never invents absent metric values."""
from __future__ import annotations


def metric_text(metrics, key, digits=3):
    value = metrics.get(key)
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


def run_table(runs):
    output = []
    for row in runs:
        m = row.get("metrics", {})
        output.append({
            "run_id": row["run_id"], "status": row["status"], "stage": row["stage"],
            "candidate": row["candidate_id"], "hypothesis": row["hypothesis_id"],
            "category": row["category"], "synthetic_pixel_AUROC_PROXY": m.get("dev_synthetic/pixel_auroc"),
            "started_at": row.get('started_at'), "finished_at": row.get('finished_at'),
            "training_duration_s": (row['finished_at'] - row['started_at']) if row.get('finished_at') and row.get('started_at') else None,
            "normal_image_FPR": m.get("normal/image_fpr_at_threshold"),
            "e2e_latency_ms_p95": m.get("inference/e2e_ms_p95"),
            "peak_vram_MiB": m.get("inference/peak_vram_mib"),
            "model_params": m.get("inference/parameters"),
            "gpu_seconds": row.get("gpu_seconds"),
            "checkpoint_MiB": m.get('inference/checkpoint_mib'),
            "failure": row.get('error') or None,
            "mlflow_run_id": row.get('mlflow_run_id') or None,
        })
    return output
