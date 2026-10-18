"""Read-only data access. Does not write controller ledger or inspect sealed dataset."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
from urllib.request import urlopen
from urllib.parse import quote


def runtime_root():
    return Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()


def db_path():
    return Path(os.getenv("LOCO_LEDGER", str(runtime_root() / "controller" / "experiments.sqlite"))).resolve()


def get_connection():
    db = db_path()
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=4)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def rows(query, params=()):
    conn = get_connection()
    if conn is None:
        return []
    try:
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def list_runs(limit=500):
    results = rows("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 1000)),))
    for row in results:
        try:
            row["metrics"] = json.loads(row["metrics_json"])
            row["config"] = json.loads(row["config_json"])
        except ValueError:
            row["metrics"], row["config"] = {}, {}
    return results


def get_events(limit=200):
    return rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 1000)),))


def get_proposals():
    result = rows("SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT 200")
    for item in result:
        try:
            item["proposal"] = json.loads(item["proposal_json"])
        except ValueError:
            item["proposal"] = {}
    return result


def pending_outbox():
    row = rows("SELECT COUNT(*) AS count FROM tracking_outbox WHERE delivered_at IS NULL")
    return row[0]["count"] if row else 0


def tracking_errors(limit=10):
    return rows("SELECT run_id,attempts,last_error,created_at FROM tracking_outbox WHERE delivered_at IS NULL AND attempts>0 ORDER BY id DESC LIMIT ?", (limit,))


def ledger_summary():
    counts = rows("SELECT status,COUNT(*) AS count FROM runs GROUP BY status")
    usage = rows("SELECT COALESCE(SUM(gpu_seconds),0) AS gpu_seconds FROM runs")
    return {'counts': {r['status']: r['count'] for r in counts}, 'gpu_seconds': usage[0]['gpu_seconds'] if usage else 0}


def gpu_status():
    if not shutil.which('nvidia-smi'):
        return {'available': False}
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if proc.returncode:
            return {'available': False, 'error': proc.stderr.strip()}
        name, used, total, utilization = [v.strip() for v in proc.stdout.splitlines()[0].split(',')]
        return {'available': True, 'name': name, 'memory_used_mib': int(used), 'memory_total_mib': int(total), 'utilization_percent': int(utilization)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc)}


def service_status():
    from orchestrator.lab_cli import owned_process, service_healthy, SERVICES
    runtime = runtime_root()
    return {name: bool(owned_process(runtime, name) and (name == 'tracking' or service_healthy(name))) for name in SERVICES}


def log_tail(name, lines=50):
    if name not in {'mlflow', 'dashboard', 'tracking'}:
        return ''
    path = runtime_root() / 'logs' / f'{name}.jsonl'
    return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:]) if path.is_file() else ''


def allowed_preview(run, filename):
    if filename not in {"normal_preview.png", "synthetic_input.png", "synthetic_mask.png", "synthetic_heatmap.png", "synthetic_overlay.png"}:
        return None
    job_root = (runtime_root() / "jobs").resolve()
    artifact = Path(run.get("artifact_dir") or "").resolve()
    if not artifact.is_relative_to(job_root) or artifact == job_root or artifact.name != run.get("run_id"):
        return None
    path = (artifact / filename).resolve()
    if not path.is_relative_to(artifact) or not path.is_file():
        return None
    return path


def get_mlflow_url(run):
    rid = run.get("mlflow_run_id")
    if not rid:
        return None
    base = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:56000").rstrip("/")
    if not base.startswith(("http://127.0.0.1:", "http://localhost:")):
        return None
    try:
        with urlopen(base + '/health', timeout=1) as response:
            if response.status != 200:
                return None
        from mlflow.tracking import MlflowClient
        experiment_id = MlflowClient(tracking_uri=base).get_run(rid).info.experiment_id
        return f"{base}/#/experiments/{quote(str(experiment_id))}/runs/{quote(str(rid))}"
    except Exception:
        return None


def tracking_health():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:56000")
    try:
        with urlopen(uri.rstrip('/') + '/health', timeout=2) as response:
            return 'Connected' if response.status == 200 else 'Unhealthy'
    except Exception:
        return "Offline"
