"""MLflow tracking with persistent SQLite outbox and idempotent link recovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
from orchestrator.state_store import StateStore


class TrackingUnavailable(RuntimeError):
    pass


def enqueue_run_tracking(store: StateStore, run_id: str, artifact_dir=None):
    row = store.get_run(run_id)
    if not row:
        raise KeyError(run_id)
    payload = {
        "run_id": run_id, "status": row["status"], "category": row["category"],
        "stage": row["stage"], "candidate_id": row["candidate_id"],
        "hypothesis_id": row["hypothesis_id"], "git_sha": row["git_sha"],
        "manifest_sha256": row["manifest_sha256"], "metrics": json.loads(row["metrics_json"]),
        "params": json.loads(row["config_json"]), "artifact_dir": str(artifact_dir or row["artifact_dir"] or ""),
    }
    store.enqueue_tracking(f"{run_id}:{row['status']}", run_id, payload)


def sync_outbox(store: StateStore, tracking_uri=None, max_events=100):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise TrackingUnavailable("Install optional MLflow dependencies (pip install -e '.[ui]')") from exc
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    delivered, failed = 0, 0
    for item in store.pending_tracking(max_events):
        data = json.loads(item["payload_json"])
        run_id = item["run_id"]
        try:
            record = store.get_run(run_id)
            mlflow_run_id = record["mlflow_run_id"]
            if not mlflow_run_id:
                experiment_name = f"LOCO-Autoresearch-{os.getenv('LOCO_CAMPAIGN', 'initial-research')}"
                exp = client.get_experiment_by_name(experiment_name)
                if exp is None:
                    experiment_id = client.create_experiment(experiment_name)
                else:
                    experiment_id = exp.experiment_id
                run = client.create_run(experiment_id, tags={
                    "research.run_id": run_id, "research.hypothesis_id": data["hypothesis_id"],
                    "research.candidate_id": data["candidate_id"], "run.stage": data["stage"],
                    "category": data["category"], "git.commit": data["git_sha"],
                    "dataset.manifest_sha256": data["manifest_sha256"],
                    "protocol.version": "1.0", "metric.kind": "synthetic_proxy_not_official_loco",
                })
                mlflow_run_id = run.info.run_id
                store.set_mlflow_id(run_id, mlflow_run_id)
            meta_path = Path(data.get("artifact_dir") or "/nonexistent") / "run_manifest.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                for field, tag in (("model_module_sha256", "candidate.code_sha256"), ("config_sha256", "config.sha256"), ("dependencies_pyproject_sha256", "dependencies.sha256")):
                    if meta.get(field):
                        client.set_tag(mlflow_run_id, tag, meta[field])
            for k, v in data.get("params", {}).items():
                # Parameters are immutable per MLflow run; same key/value is idempotent.
                if isinstance(v, (str, int, float, bool)):
                    client.log_param(mlflow_run_id, k, str(v))
            for k, v in data["metrics"].items():
                client.log_metric(mlflow_run_id, k, float(v))
            client.set_tag(mlflow_run_id, "research.status", data["status"])
            artifact_dir = Path(data.get("artifact_dir") or "/nonexistent")
            if artifact_dir.is_dir():
                # Allowlisted output types only. Never upload raw source images or test masks.
                for filename in ("measurement.json", "run_manifest.json", "normal_preview.png", "synthetic_heatmap.png", "synthetic_overlay.png"):
                    if (artifact_dir / filename).is_file():
                        client.log_artifact(mlflow_run_id, str(artifact_dir / filename))
            # Do not terminate MLflow record before all outbox events are processed.
            if data["status"] in {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED", "OOM", "INTERRUPTED"}:
                client.set_terminated(mlflow_run_id, status="FINISHED" if data["status"] == "COMPLETED" else "FAILED")
            store.tracking_delivered(item["id"])
            delivered += 1
        except Exception as exc:
            store.tracking_failed(item["id"], str(exc))
            failed += 1
            # Stop on first remote outage; subsequent events will be retried in order.
            break
    return {"delivered": delivered, "failed": failed, "remaining": len(store.pending_tracking(10000))}
