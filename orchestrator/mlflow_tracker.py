"""MLflow tracking with persistent SQLite outbox and idempotent link recovery."""
from __future__ import annotations
import json
import os
import fcntl
from pathlib import Path
from orchestrator.state_store import StateStore


class TrackingUnavailable(RuntimeError):
    pass


def enqueue_run_tracking(store: StateStore, run_id: str, artifact_dir=None, event_key_suffix=None):
    row = store.get_run(run_id)
    if not row:
        raise KeyError(run_id)
    payload = {
        "run_id": run_id, "status": row["status"], "category": row["category"],
        "stage": row["stage"], "candidate_id": row["candidate_id"],
        "hypothesis_id": row["hypothesis_id"], "git_sha": row["git_sha"],
        "manifest_sha256": row["manifest_sha256"], "metrics": json.loads(row["metrics_json"]),
        "params": json.loads(row["config_json"]), "artifact_dir": str(artifact_dir or row["artifact_dir"] or ""),
        "created_at": row["created_at"], "started_at": row["started_at"],
        "finished_at": row["finished_at"], "gpu_seconds": row["gpu_seconds"], "error": row["error"],
    }
    store.enqueue_tracking(f"{run_id}:{event_key_suffix or row['status']}", run_id, payload)


def sync_outbox(store: StateStore, tracking_uri=None, max_events=100):
    """Serialize consumers across processes, including manual sync invocations."""
    lock = store.path.parent / 'tracking.lock'
    with lock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return _sync_locked(store, tracking_uri, max_events)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _sync_locked(store: StateStore, tracking_uri=None, max_events=100):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise TrackingUnavailable("Install optional MLflow dependencies (pip install -e '.[ui]')") from exc
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:56000")
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
                tags = {
                    "research.run_id": run_id, "research.hypothesis_id": data["hypothesis_id"],
                    "research.candidate_id": data["candidate_id"], "run.stage": data["stage"],
                    "category": data["category"], "git.commit": data["git_sha"],
                    "dataset.manifest_sha256": data["manifest_sha256"],
                    "protocol.version": "1.0", "metric.kind": "synthetic_proxy_not_official_loco",
                }
                if data.get('params', {}).get('campaign_id'):
                    tags['research.campaign_id'] = str(data['params']['campaign_id'])
                if data.get('params', {}).get('campaign_name'):
                    tags['research.campaign_name'] = str(data['params']['campaign_name'])
                # Recover a remote run created just before a local crash.
                existing = client.search_runs([experiment_id], filter_string=f"tags.`research.run_id` = '{run_id}'", max_results=2) if hasattr(client, 'search_runs') else []
                if len(existing) > 1:
                    raise RuntimeError(f'Multiple MLflow runs already exist for {run_id}; manual review required')
                if existing:
                    mlflow_run_id = existing[0].info.run_id
                else:
                    try:
                        run = client.create_run(experiment_id, tags=tags, start_time=int(data.get('started_at') or data.get('created_at') or 0) * 1000 or None)
                    except TypeError:
                        run = client.create_run(experiment_id, tags=tags)
                    mlflow_run_id = run.info.run_id
                store.set_mlflow_id(run_id, mlflow_run_id)
            # Reapply mutable campaign tags on retries and later review events;
            # older MLflow runs created before these tags were added are repaired.
            if data.get('params', {}).get('campaign_id'):
                client.set_tag(mlflow_run_id, 'research.campaign_id', str(data['params']['campaign_id']))
            if data.get('params', {}).get('campaign_name'):
                client.set_tag(mlflow_run_id, 'research.campaign_name', str(data['params']['campaign_name']))
            if data.get('git_sha'):
                client.set_tag(mlflow_run_id, 'git.commit', str(data['git_sha']))
            client.set_tag(mlflow_run_id, 'execution.gpu_seconds_semantics', 'exclusive_gpu_slot_wall_seconds')
            artifact_dir = Path(data.get("artifact_dir") or "/nonexistent").resolve()
            allowed_root = (Path(os.getenv('LOCO_RUNTIME', '.runtime')).resolve() / 'jobs' / run_id).resolve()
            if artifact_dir != allowed_root:
                artifact_dir = Path('/nonexistent')
            meta_path = artifact_dir / "run_manifest.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                if meta.get('protocol_version'):
                    client.set_tag(mlflow_run_id, 'protocol.version', str(meta['protocol_version']))
                for field, tag in (("model_module_sha256", "candidate.code_sha256"), ("config_sha256", "config.sha256"), ("dependencies_pyproject_sha256", "dependencies.sha256")):
                    if meta.get(field):
                        client.set_tag(mlflow_run_id, tag, meta[field])
                for field, tag in (("metric_evaluator_version", "evaluation.metric_version"),
                                   ("synthetic_generator_source_sha256", "evaluation.generator_source_sha256"),
                                   ("synthetic_mask_semantics", "evaluation.synthetic_mask_semantics")):
                    if meta.get(field):
                        client.set_tag(mlflow_run_id, tag, str(meta[field])[:1000])
            for k, v in data.get("params", {}).items():
                # Parameters are immutable per MLflow run; same key/value is idempotent.
                if isinstance(v, (str, int, float, bool)):
                    client.log_param(mlflow_run_id, k, str(v))
            for k, v in data["metrics"].items():
                client.log_metric(mlflow_run_id, k, float(v))
            if data.get('gpu_seconds') is not None:
                client.log_metric(mlflow_run_id, 'execution/gpu_seconds', float(data['gpu_seconds']))
            if data.get('started_at') and data.get('finished_at'):
                client.log_metric(mlflow_run_id, 'execution/wall_seconds', max(0.0, data['finished_at'] - data['started_at']))
            client.set_tag(mlflow_run_id, "research.status", data["status"])
            if data.get('error'):
                client.set_tag(mlflow_run_id, 'research.failure_reason', str(data['error'])[:500])
            with store.connect() as con:
                hypothesis = con.execute('SELECT proposal_json FROM hypotheses WHERE hypothesis_id=?',
                                         (data['hypothesis_id'],)).fetchone()
                critique = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' ORDER BY id DESC LIMIT 1",
                                       (run_id,)).fetchone()
            if hypothesis:
                proposal = json.loads(hypothesis['proposal_json'])
                client.set_tag(mlflow_run_id, 'research.source_references', json.dumps(proposal.get('references', []))[:4000])
                if proposal.get('priority_rank') is not None:
                    client.set_tag(mlflow_run_id, 'research.hypothesis_rank', str(proposal['priority_rank']))
            if critique:
                client.set_tag(mlflow_run_id, 'research.critic_assessment', critique['payload_json'][:4000])
            with store.connect() as con:
                cumulative=con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CUMULATIVE_REVIEW' ORDER BY id DESC LIMIT 1",(run_id,)).fetchone()
                retention=con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='MODEL_RETAINED' ORDER BY id DESC LIMIT 1",(run_id,)).fetchone()
            if cumulative:
                client.set_tag(mlflow_run_id,'research.cumulative_decisions',cumulative[0][:4000])
            if retention:
                model=json.loads(retention[0])
                client.set_tag(mlflow_run_id,'research.retention_id',model['retention_id'])
                client.set_tag(mlflow_run_id,'research.comparison_cohort',model['cohort_id'])
                client.set_tag(mlflow_run_id,'research.retained_checkpoint_sha256',model['reference']['checkpoint_sha256'])
                client.set_tag(mlflow_run_id,'research.promotion_policy_sha256',model['policy_sha256'])
                if hasattr(client,'log_dict'):
                    client.log_dict(mlflow_run_id,model,'retention/model_record.json')
            if artifact_dir.is_dir():
                # Allowlisted output types only. Never upload raw source images or test masks.
                for filename in ("measurement.json", "run_manifest.json", "training_audit.json", "plan_comparison.json", "development_case_metrics.json", "stage6_comparison.json", "normal_preview.png", "synthetic_input.png", "synthetic_mask.png", "synthetic_heatmap.png", "synthetic_overlay.png", "cuda_smoke_stdout.log", "cuda_smoke_stderr.log", "cuda_smoke/summary.json"):
                    item_path = artifact_dir / filename
                    if (item_path.is_file() and not item_path.is_symlink()
                            and item_path.resolve().is_relative_to(artifact_dir.resolve())):
                        client.log_artifact(mlflow_run_id, str(item_path))
                comparison = artifact_dir/'plan_comparison.json'
                if comparison.is_file() and not comparison.is_symlink():
                    record = json.loads(comparison.read_text())
                    client.set_tag(mlflow_run_id,'research.plan_id',record['plan']['plan_id'])
                    client.set_tag(mlflow_run_id,'research.plan_sha256',record['result']['plan_sha256'])
                    client.set_tag(mlflow_run_id,'research.comparison_status',record['result']['status'])
                for request in data.get('params',{}).get('diagnostics',[]):
                    name = request.get('diagnostic_id','')
                    if not __import__('re').fullmatch(r'[A-Za-z0-9_-]{1,64}',name): continue
                    path = artifact_dir/f'diagnostic_{name}.json'
                    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(artifact_dir):
                        client.log_artifact(mlflow_run_id,str(path))
                checkpoint = artifact_dir / 'checkpoint' / 'model.pt'
                if checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.resolve().is_relative_to(allowed_root):
                    client.log_artifact(mlflow_run_id, str(checkpoint), artifact_path='checkpoint')
            feedback = data.get('feedback')
            if feedback:
                client.set_tag(mlflow_run_id, 'evaluation.real_development_label', feedback['label'])
                client.set_tag(mlflow_run_id, 'evaluation.real_split_id', feedback['split_id'])
                client.set_tag(mlflow_run_id, 'evaluation.real_evaluator_version', feedback['evaluator_version'])
                client.set_tag(mlflow_run_id, 'evaluation.real_checkpoint_sha256', feedback['checkpoint_sha256'])
                client.set_tag(mlflow_run_id, 'evaluation.real_localization_status', feedback['localization_status'])
                client.set_tag(mlflow_run_id, 'learning.real_anomaly_fitting', 'forbidden')
                for key, value in feedback['metrics'].items():
                    client.log_metric(mlflow_run_id, key, float(value))
                    client.log_metric(mlflow_run_id, 'evaluation_versions/' + feedback['evaluator_version'] + '/' + key, float(value))
                feedback_dir = Path(feedback['artifact_dir'])
                expected = allowed_root / 'feedback' / feedback['evaluation_id']
                if feedback_dir.is_symlink() or feedback_dir.resolve() != expected.resolve() or not expected.resolve().is_relative_to(allowed_root):
                    raise ValueError('feedback artifact path escapes originating job')
                for filename in ('summary.json', 'case_scores.json', 'heatmap_0.png', 'heatmap_47.png', 'heatmap_118.png'):
                    path = feedback_dir / filename
                    if path.is_file() and not path.is_symlink():
                        client.log_artifact(mlflow_run_id, str(path), artifact_path='real_development/' + feedback['evaluation_id'])
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
