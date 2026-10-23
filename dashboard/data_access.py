"""Read-only data access. Does not write controller ledger or inspect sealed dataset."""
from __future__ import annotations
import json
import os
from collections import deque
from pathlib import Path
import sqlite3
import shutil
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import quote
from PIL import Image, UnidentifiedImageError


def runtime_root():
    return Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()


def db_path():
    return Path(os.getenv("LOCO_LEDGER", str(runtime_root() / "controller" / "experiments.sqlite"))).resolve()


def cumulative_progress():
    """Ledger-only read; never initialize, promote or inspect model/data files."""
    if not db_path().is_file():return None
    from orchestrator.state_store import StateStore
    from orchestrator.model_registry import progress, localization_history
    try:
        store=StateStore(db_path(),read_only=True)
        result=progress(store)
        result['localization_history']=localization_history(store)
        return result
    except sqlite3.OperationalError:return None


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


def campaign_summary():
    """Latest durable campaign, with no inferred or simulated agent activity."""
    records = rows("SELECT * FROM campaigns ORDER BY created_at DESC LIMIT 1")
    if not records:
        return None
    campaign = records[0]
    try:
        campaign["config"] = json.loads(campaign["config_json"])
    except ValueError:
        campaign["config"] = {}
    counts = rows("SELECT COUNT(*) AS experiments, COALESCE(SUM(r.gpu_seconds),0) AS gpu_seconds, "
                  "COALESCE(SUM(r.status IN ('COMPLETED','REVIEWED','PROMOTED')),0) AS successful "
                  "FROM campaign_runs cr JOIN runs r ON r.run_id=cr.run_id WHERE cr.campaign_id=?", (campaign["id"],))
    campaign["usage"] = counts[0] if counts else {"experiments": 0, "gpu_seconds": 0}
    campaign['run_ids'] = [r['run_id'] for r in rows(
        'SELECT r.run_id FROM campaign_runs cr JOIN runs r ON r.run_id=cr.run_id WHERE cr.campaign_id=? ORDER BY r.created_at, r.rowid', (campaign['id'],))]
    failures = rows("SELECT COUNT(*) AS n FROM campaign_events WHERE campaign_id=? AND kind IN ('AGENT_FAILED','ATTEMPT_FAILED','CONTROLLER_ERROR')", (campaign['id'],))
    campaign['failure_events'] = failures[0]['n'] if failures else 0
    campaign['usage']['gpu_seconds_live'] = campaign['usage']['gpu_seconds']
    if campaign.get('active_run_id'):
        active = rows('SELECT status,started_at FROM runs WHERE run_id=?', (campaign['active_run_id'],))
        if active and active[0]['status'] == 'RUNNING' and active[0]['started_at']:
            campaign['usage']['gpu_seconds_live'] += max(0, time.time() - active[0]['started_at'])
    campaign["events"] = rows("SELECT * FROM campaign_events WHERE campaign_id=? ORDER BY id DESC LIMIT 12", (campaign["id"],))
    from orchestrator.lab_cli import owned_process
    campaign["worker_running"] = bool(owned_process(runtime_root(), 'research'))
    return campaign


def campaign_timeline(limit=200):
    return rows("SELECT * FROM campaign_events ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 500)),))


def latest_multiseed_comparison():
    """Read only completed versioned comparisons; no inference from single runs."""
    records = rows("SELECT payload_json FROM campaign_events WHERE kind='STAGE6_COMPARISON' ORDER BY id DESC LIMIT 1")
    if not records:
        return None
    try:
        comparison = json.loads(records[0]['payload_json'])['comparison']
        if (comparison.get('metric_evaluator_version') != 'development-v2.0' or
            len(comparison.get('per_seed', [])) != 15 or
            len(comparison.get('seeds', [])) != 3):
            return None
        return comparison
    except (KeyError, TypeError, ValueError):
        return None


def pending_outbox():
    row = rows("SELECT COUNT(*) AS count FROM tracking_outbox WHERE delivered_at IS NULL")
    return row[0]["count"] if row else 0


def queue_paused():
    result = rows("SELECT value FROM settings WHERE name='paused'")
    return bool(result and result[0]["value"] == "1")


def dataset_summary():
    """Read only good-only manifest metadata; never load image data for the header."""
    try:
        from benchmark.data_guard import load_manifest
        manifest, _ = load_manifest(runtime_root() / "good_manifest.json", verify_hashes=False)
        categories = sorted({item["category"] for item in manifest["images"]})
        return {"categories": categories, "images": len(manifest["images"])}
    except (OSError, ValueError, KeyError):
        return {"categories": [], "images": 0}


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
    if name not in {'mlflow', 'dashboard', 'tracking', 'research'}:
        return ''
    path = runtime_root() / 'logs' / f'{name}.jsonl'
    if not path.is_file():
        return ''
    with path.open(errors='replace') as handle:
        return '\n'.join(line.rstrip('\n') for line in deque(handle, maxlen=max(1, min(lines, 200))))


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


def load_preview(run, filename):
    """Load only named, size-bounded development previews from this run's job dir."""
    path = allowed_preview(run, filename)
    if path is None:
        return None
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return None
        with Image.open(path) as image:
            if image.width * image.height > 24_000_000:
                return None
            return image.convert("RGB").copy()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return None


def heatmap_sample(run, opacity=0.38):
    """Compose aligned previews without resizing or changing their coordinates."""
    original = load_preview(run, "normal_preview.png")
    synthetic = load_preview(run, "synthetic_input.png")
    heatmap = load_preview(run, "synthetic_heatmap.png")
    mask = load_preview(run, "synthetic_mask.png")
    if original is None or synthetic is None or heatmap is None:
        return None
    if original.size != synthetic.size or original.size != heatmap.size or (mask and original.size != mask.size):
        return None
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")
    return {"original": original, "synthetic": synthetic, "heatmap": heatmap,
            "overlay": Image.blend(synthetic, heatmap, opacity), "mask": mask,
            "size": original.size}


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


def real_feedback_summaries():
    # Select the latest feedback per run, then present candidates in execution
    # order, independent of when tracking/evaluation events were delivered.
    result = rows("SELECT e.run_id,e.payload_json,r.candidate_id FROM events e JOIN runs r ON r.run_id=e.run_id WHERE e.kind='REAL_DEVELOPMENT_FEEDBACK' AND e.id=(SELECT MAX(latest.id) FROM events latest WHERE latest.run_id=e.run_id AND latest.kind='REAL_DEVELOPMENT_FEEDBACK') ORDER BY r.created_at, r.rowid LIMIT 100")
    summaries = []
    seen = set()
    for row in result:
        if row['run_id'] in seen: continue
        seen.add(row['run_id'])
        try:
            summary = json.loads(row['payload_json'])
            summary.update(run_id=row['run_id'],candidate_id=row['candidate_id'])
            comparison = rows("SELECT payload_json FROM events WHERE run_id=? AND kind='SCIENTIFIC_COMPARISON' ORDER BY id DESC LIMIT 1", (row['run_id'],))
            if comparison:
                summary['scientific_comparison'] = json.loads(comparison[0]['payload_json'])
            summaries.append(summary)
        except (ValueError,TypeError):
            continue
    return summaries


def real_heatmap_sample(summary, index, opacity=.4):
    """Read-only development viewer: pinned image plus validated full-size map."""
    import numpy as np
    from benchmark.feedback_data import imported_root, schema
    from benchmark.data_guard import sha256_file
    root = imported_root(runtime_root())
    manifest = json.loads((root/'dev-manifest.json').read_text()); schema(manifest)
    if not isinstance(index,int) or not 0 <= index < len(manifest['images']): raise ValueError('invalid sample index')
    entry = manifest['images'][index]
    image_path = root/entry['path']
    output = runtime_root()/'jobs'/summary['run_id']/'feedback'/summary['evaluation_id']
    from orchestrator.development_feedback import feedback_map_root
    # Legacy summaries use local maps. New exact readout reuse keeps the maps
    # under their verified original job, without copying or linking them.
    map_output = feedback_map_root(dict(summary, artifact_dir=str(output)), runtime_root()/'jobs')
    map_path = map_output/f'case_{index:04d}'/'map.npy'
    if (not output.resolve().is_relative_to(runtime_root()/'jobs') or map_path.is_symlink() or
        not map_path.resolve().is_relative_to(map_output.resolve()) or
        not image_path.resolve().is_relative_to(root.resolve()) or sha256_file(image_path)!=entry['sha256']):
        raise ValueError('unsafe development artifact')
    cases = json.loads((output/'case_scores.json').read_text())
    if sha256_file(map_path) != cases[index]['map_sha256']: raise ValueError('heatmap hash changed')
    score = np.load(map_path,allow_pickle=False)
    original = Image.open(image_path).convert('RGB')
    if score.shape != (original.height,original.width) or not np.isfinite(score).all(): raise ValueError('unaligned map')
    scaled = np.clip((score-score.min())/max(float(np.ptp(score)),1e-12),0,1)
    color = np.stack([scaled*255,scaled*60,(1-scaled)*100],axis=-1)
    overlay = Image.fromarray(np.uint8(np.asarray(original)*(1-opacity)+color*opacity))
    # Display-only union; independent mask/defect instances remain untouched for scoring.
    masks = []
    for record in entry['masks']:
        path = root/record['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256_file(path)!=record['sha256']:
            raise ValueError('unsafe or changed ground truth')
        mask = np.asarray(Image.open(path))
        if mask.shape != score.shape: raise ValueError('unaligned ground truth')
        masks.append(mask)
    union = np.logical_or.reduce([m > 0 for m in masks]) if masks else np.zeros(score.shape,dtype=bool)
    return {'original':original,'overlay':overlay,'heatmap':Image.fromarray(np.uint8(color)),
            'ground_truth':Image.fromarray(np.uint8(union)*255),'mask_count':len(masks),
            'category':entry['category'],'score':cases[index]['image_score'],'size':original.size}


def candidate_research_details(run_id):
    """Join the selected run to its own proposal, implementation and critique."""
    found = rows('SELECT * FROM runs WHERE run_id=?', (run_id,))
    if not found:
        return None
    run = found[0]
    def decode(value):
        try:
            item = json.loads(value)
            return item if isinstance(item, dict) else {}
        except (TypeError, ValueError):
            return {}
    run['config'] = decode(run['config_json'])
    run['metrics'] = decode(run['metrics_json'])
    proposals = rows('SELECT proposal_json FROM hypotheses WHERE hypothesis_id=?', (run['hypothesis_id'],))
    implementations = rows("SELECT payload_json FROM events WHERE kind='IMPLEMENTER_RESPONSE' AND json_valid(payload_json) AND json_extract(payload_json,'$.hypothesis_id')=? AND time<=? ORDER BY id DESC LIMIT 1", (run['hypothesis_id'], run['created_at']))
    reviews = rows("SELECT payload_json FROM events WHERE kind='CRITIC_REVIEW' AND run_id=? ORDER BY id DESC LIMIT 1", (run_id,))
    membership = rows('SELECT campaign_id FROM campaign_runs WHERE run_id=?', (run_id,))
    history = rows('SELECT kind,time,payload_json FROM campaign_events WHERE campaign_id=? ORDER BY id', (membership[0]['campaign_id'],)) if membership else []
    linked = []
    for event in history:
        payload = decode(event['payload_json'])
        if payload.get('run_id') == run_id or payload.get('evidence_run') == run_id or (
            payload.get('hypothesis_id') == run['hypothesis_id'] and event['time'] <= run['created_at']):
            linked.append(dict(event, payload=payload))
    return {'run': run, 'proposal': decode(proposals[0]['proposal_json']) if proposals else {},
            'implementation': decode(implementations[0]['payload_json']).get('response', {}) if implementations else {},
            'review': decode(reviews[0]['payload_json']) if reviews else {}, 'history': linked}
