"""Trusted, persistent campaign state machine. Agent JSON never becomes shell code."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import yaml

from benchmark.data_guard import load_manifest, manifest_fingerprint
from orchestrator.campaign_state import CampaignState, TERMINAL
from orchestrator.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]
REQUIRED_LIMITS = ("max_gpu_hours", "max_wall_hours", "max_experiments", "max_codex_calls",
                   "max_job_seconds", "max_consecutive_failures", "max_parallel_gpu_jobs")


class CampaignHalted(RuntimeError):
    pass


def await_authorized_stage(campaigns, campaign_id, finishing=False):
    while True:
        row = campaigns.get(campaign_id)
        if row['status'] == 'RUNNING':
            if finishing:
                limits = json.loads(row['config_json'])['limits']
                state, reason = (("LIMIT_REACHED", "wall-time budget reached")
                                 if time.time() - row['created_at'] >= limits['max_wall_hours'] * 3600
                                 else (None, ""))
            else:
                state, reason = stop_reason(campaigns, campaign_id)
            if state:
                campaigns.set_status(campaign_id, state, reason)
                raise CampaignHalted(reason)
            return
        if row['status'] == 'PAUSED':
            time.sleep(1)
            continue
        raise CampaignHalted(row['reason'] or row['status'])


def load_campaign_config(path=None):
    source = Path(path or REPO / "configs" / "autonomy.yaml").resolve(strict=True)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or not all(k in config for k in ("campaign", "limits", "research", "tracking", "security")):
        raise ValueError("invalid autonomy configuration")
    campaign, limits = config["campaign"], config["limits"]
    if campaign.get("mode") != "autonomous" or campaign.get("dataset") != "juice_bottle" or campaign.get("gpu_device") != "cuda":
        raise ValueError("campaign must use the approved juice_bottle CUDA development setup")
    if not all(isinstance(limits.get(k), (int, float)) and not isinstance(limits.get(k), bool) and limits[k] > 0 for k in REQUIRED_LIMITS):
        raise ValueError("campaign limits must be positive numbers")
    if limits["max_parallel_gpu_jobs"] != 1:
        raise ValueError("one RTX 4090 requires exclusive GPU scheduling")
    if limits["max_gpu_hours"] > float(os.getenv("LOCO_GPU_BUDGET_HOURS", "4")) or limits["max_job_seconds"] > int(os.getenv("LOCO_JOB_TIMEOUT_SECONDS", "1800")) or limits["max_codex_calls"] > int(os.getenv("LOCO_CODEX_MAX_CALLS", "30")):
        raise ValueError("campaign limits exceed existing operator resource ceilings")
    if config["research"].get("official_test_access") is not False or config["research"].get("synthetic_anomalies_allowed") is not True:
        raise ValueError("official test data must remain excluded")
    if config["security"].get("execution_mode") != "personal_bwrap":
        raise ValueError("unknown execution boundary")
    return config


def preflight(config, runtime, *, require_authorization=True, verify_hashes=True):
    from orchestrator.sandbox import available
    import torch
    if require_authorization and config["security"].get("operator_accepts_shared_uid_risk") is not True:
        raise PermissionError("Personal-server shared-UID risk is not authorized in autonomy.yaml; unattended campaign is disabled")
    if not available():
        raise RuntimeError("Bubblewrap is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    for name in ("RESEARCH_CHARTER.md", "research_program.md"):
        if not (REPO / name).is_file():
            raise FileNotFoundError(name)
    manifest_path = Path(runtime) / "good_manifest.json"
    manifest, _ = load_manifest(manifest_path, verify_hashes=verify_hashes)
    if Path(manifest["root"]).resolve(strict=True) != Path(os.environ["LOCO_GOOD_ROOT"]).resolve(strict=True):
        raise ValueError("manifest root differs from authorized good-image root")
    if {entry["category"] for entry in manifest["images"]} != {"juice_bottle"}:
        raise ValueError("campaign manifest contains an unauthorized category")
    if not any(entry["split"] == "train/good" for entry in manifest["images"]) or not any(entry["split"] == "validation/good" for entry in manifest["images"]):
        raise ValueError("normal training and validation splits required")
    return {"manifest_path": manifest_path, "fingerprint": manifest_fingerprint(manifest), "images": len(manifest["images"])}


def stop_reason(campaigns, campaign_id):
    row = campaigns.get(campaign_id)
    limits = json.loads(row["config_json"])["limits"]
    used = campaigns.usage(campaign_id)
    if row["status"] != "RUNNING":
        return row["status"], row["reason"]
    if used["wall_seconds"] >= limits["max_wall_hours"] * 3600:
        return "LIMIT_REACHED", "wall-time budget reached"
    if used["gpu_seconds"] >= limits["max_gpu_hours"] * 3600:
        return "LIMIT_REACHED", "GPU-time budget reached"
    if used["experiments"] >= limits["max_experiments"]:
        return "LIMIT_REACHED", "experiment count reached"
    if used["codex_calls"] >= limits["max_codex_calls"]:
        return "LIMIT_REACHED", "Codex-call budget reached"
    if row["consecutive_failures"] >= limits["max_consecutive_failures"]:
        return "LIMIT_REACHED", "consecutive failure limit reached"
    return None, ""


def remaining_job_seconds(campaigns, campaign_id):
    row = campaigns.get(campaign_id)
    limits = json.loads(row["config_json"])["limits"]
    used = campaigns.usage(campaign_id)
    return int(min(limits["max_job_seconds"], limits["max_gpu_hours"] * 3600 - used["gpu_seconds"],
                   limits["max_wall_hours"] * 3600 - used["wall_seconds"]))


def reconcile_unfinished_runs(store, campaigns, campaign_id, reason):
    """Never replay queued/running work after a worker crash or forced stop."""
    for run in store.list_runs(5000):
        try:
            belongs = json.loads(run['config_json']).get('campaign_id') == campaign_id
        except ValueError:
            belongs = False
        if not belongs or run['status'] not in {'QUEUED', 'RUNNING'}:
            continue
        if run['status'] == 'RUNNING' and run.get('started_at'):
            store.account_gpu_usage(run['run_id'], max(float(run['gpu_seconds'] or 0), time.time() - run['started_at']))
        store.transition(run['run_id'], 'INTERRUPTED' if run['status'] == 'RUNNING' else 'CANCELLED', reason=reason)
        campaigns.event(campaign_id, 'RUN_INTERRUPTED', run_id=run['run_id'], reason=reason)
    campaigns.update(campaign_id, active_run_id='')


def correct_review_disposition(store, run_id):
    """Revoke a stale promotion when the recorded critic asked for investigation.

    This is a narrow audited repair for campaigns created before the stricter
    promotion rule. It changes no measured result or source artifact.
    """
    run = store.get_run(run_id)
    if not run or run['status'] != 'PROMOTED':
        return False
    with store.connect() as con:
        review = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' "
                             "ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
    if not review:
        return False
    action = json.loads(review['payload_json']).get('recommended_next_action')
    if action == 'promote_for_review':
        return False
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        current = con.execute('SELECT status FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if not current or current['status'] != 'PROMOTED':
            return False
        con.execute("UPDATE runs SET status='REVIEWED',error='',updated_at=? WHERE run_id=?",
                    (time.time(), run_id))
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                    (run_id, time.time(), 'REVIEW_CORRECTION', json.dumps({
                        'from': 'PROMOTED', 'to': 'REVIEWED', 'critic_action': action,
                        'reason': 'critic did not recommend promotion; source and metrics unchanged'})))
    from orchestrator.mlflow_tracker import enqueue_run_tracking
    enqueue_run_tracking(store, run_id)
    return True


def correct_candidate_git_commit(store, run_id, repo_root=REPO):
    """Audit a late candidate-source snapshot if its model hash matches the run."""
    import hashlib
    import subprocess
    run = store.get_run(run_id)
    if not run or run['status'] not in {'COMPLETED', 'REVIEWED', 'PROMOTED', 'REJECTED', 'FAILED'}:
        raise ValueError('only finalized runs may receive a provenance correction')
    candidate_id = run['candidate_id']
    tree = Path(repo_root) / 'candidates' / 'worktrees' / candidate_id
    relative = f'candidates/{candidate_id}/model.py'
    source = subprocess.run(['git', '-C', str(tree), 'show', f'HEAD:{relative}'],
                            check=True, capture_output=True).stdout
    committed_hash = hashlib.sha256(source).hexdigest()
    expected_hash = json.loads(run['config_json']).get('model_sha256')
    if not expected_hash or committed_hash != expected_hash:
        raise ValueError('candidate commit model hash differs from measured source')
    commit = subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit == run['git_sha']:
        return commit
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        con.execute('UPDATE runs SET git_sha=?,updated_at=? WHERE run_id=?', (commit, time.time(), run_id))
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                    (run_id, time.time(), 'PROVENANCE_CORRECTION', json.dumps({
                        'from_git': run['git_sha'], 'to_git': commit,
                        'verified_model_sha256': committed_hash,
                        'reason': 'candidate source snapshot committed after bounded run; measured model bytes identical'})))
    from orchestrator.mlflow_tracker import enqueue_run_tracking
    enqueue_run_tracking(store, run_id, event_key_suffix='candidate-git-provenance')
    return commit


def choose_proposal(store, excluded=()):
    """Select evidence-ranked proposals; Codex provides the research ranking."""
    from orchestrator.controller import list_proposals
    available = [p for p in list_proposals(store) if p["status"] == "PROPOSED" and p["hypothesis_id"] not in excluded]
    available.sort(key=lambda p: (json.loads(p["proposal_json"]).get("priority_rank", 999), p["created_at"]))
    return available[0] if available else None


def run_cycle(store: StateStore, campaigns: CampaignState, campaign_id: str, runtime: Path):
    """Perform at most one measured attempt; caller persists and repeats."""
    from orchestrator.controller import (add_research_proposals, budgeted_invoke,
        implement_proposal, list_proposals, queue_candidate, run_worker_once, scout_literature)
    config = json.loads(campaigns.get(campaign_id)["config_json"])
    context = (campaigns, campaign_id)
    prior = campaigns.runs(campaign_id)
    directive = campaigns.get(campaign_id)['next_hypothesis']
    proposal = None
    if prior and directive.startswith('evidence-followup:'):
        previous = prior[-1]
        with store.connect() as con:
            critique = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' "
                                   "ORDER BY id DESC LIMIT 1", (previous['run_id'],)).fetchone()
        history = {'hypothesis': previous['hypothesis_id'], 'status': previous['status'],
                   'metrics': json.loads(previous['metrics_json']), 'failure': previous['error'],
                   'critic': json.loads(critique['payload_json']) if critique else {}}
        prompt = ("Propose THREE scientifically distinct next hypotheses based specifically on this prior measured "
                  "run and critic assessment. Include a controlled falsification of the previous mechanism where "
                  "appropriate; do not repeat rejected ideas. Synthetic metrics are development proxies only. "
                  "Use installed dependencies; new_dependencies must be [] when no install is needed. " +
                  json.dumps(history))
        generated = add_research_proposals(store, prompt, campaign=context, personal_mode=True)
        ids = {item['hypothesis_id'] for item in generated}
        fresh = [p for p in list_proposals(store)
                 if p['hypothesis_id'] in ids]
        fresh.sort(key=lambda p: json.loads(p['proposal_json']).get('priority_rank', 999))
        proposal = fresh[0] if fresh else None
        campaigns.event(campaign_id, 'EVIDENCE_ADAPTED_HYPOTHESES', source_run=previous['run_id'],
                        hypotheses=[p['hypothesis_id'] for p in fresh])
        campaigns.update(campaign_id, next_hypothesis='')
    if proposal is None:
        proposal = choose_proposal(store)
    if proposal is None:
        if prior:
            history = [{"hypothesis": r["hypothesis_id"], "status": r["status"],
                        "metrics": json.loads(r["metrics_json"]), "failure": r["error"]} for r in prior[-5:]]
            prompt = ("Propose THREE distinct falsifiable NEXT hypotheses using this measured development history and critic concerns. "
                      "Do not repeat failed approaches. Each runnable proposal must use existing installed Python dependencies "
                      "and no external pretrained weights. The new_dependencies field lists ONLY actual new package names; "
                      "use [] when nothing must be installed. Put future approval caveats in risk_and_shortcuts. "
                      "These are synthetic proxies, never official LOCO scores: " + json.dumps(history))
        else:
            scout_file = REPO / 'literature' / 'scout_notes.json'
            if scout_file.is_file():
                leads = json.loads(scout_file.read_text()).get('result', {})
                campaigns.event(campaign_id, 'LITERATURE_REUSED', source=str(scout_file))
            else:
                leads = scout_literature(store, campaign=context, personal_mode=True)
                campaigns.event(campaign_id, "LITERATURE", leads=leads)
            prior_runs = [r for r in store.list_runs(10) if r['candidate_id'] != 'toy_normal_ae']
            previous = [{'hypothesis': r['hypothesis_id'], 'status': r['status'],
                         'failure': r['error'], 'metrics': json.loads(r['metrics_json'])} for r in prior_runs]
            prompt = ("Propose THREE distinct falsifiable good-only, memory-bank-free localization hypotheses. "
                      "Make them runnable with installed PyTorch/NumPy and no external pretrained weights; "
                      "the new_dependencies field lists ONLY actual new package names and must be [] for runnable ideas; "
                      "put prose and future approval caveats in risk_and_shortcuts. "
                      "Use these previous measured outcomes and implementation failures to avoid repetition. "
                      "Treat unverified leads as unverified. Leads: " + json.dumps(leads) +
                      " Previous attempts: " + json.dumps(previous))
        add_research_proposals(store, prompt, campaign=context, personal_mode=True)
        campaigns.event(campaign_id, "HYPOTHESES_GENERATED")
        proposal = choose_proposal(store)
    if proposal is None:
        raise RuntimeError("no selectable hypothesis")
    hypothesis_id = proposal["hypothesis_id"]
    detail = json.loads(proposal["proposal_json"])
    campaigns.update(campaign_id, current_hypothesis=hypothesis_id, next_hypothesis="")
    campaigns.event(campaign_id, "HYPOTHESIS_SELECTED", hypothesis_id=hypothesis_id,
                    rationale=detail.get("priority_rationale", ""))
    if detail.get("new_dependencies"):
        store.set_hypothesis_status(hypothesis_id, "INVALID")
        campaigns.event(campaign_id, "CANDIDATE_REJECTED", hypothesis_id=hypothesis_id, reason="unapproved dependencies")
        return
    try:
        await_authorized_stage(campaigns, campaign_id)
        implemented = implement_proposal(store, hypothesis_id, campaign=context, personal_mode=True)
        from orchestrator.experiment_runner import candidate_file, probe_candidate_interface, run_candidate_unit_tests
        if candidate_file(REPO, hypothesis_id).resolve() != Path(implemented['candidate_module']).resolve():
            raise RuntimeError('candidate source resolution differs from the reviewed implementation')
        probe_candidate_interface(REPO, Path(implemented['candidate_module']),
                                  Path(os.environ['LOCO_GOOD_ROOT']), runtime, hypothesis_id)
        unit_result = run_candidate_unit_tests(REPO, Path(implemented['candidate_module']),
                                               Path(os.environ['LOCO_GOOD_ROOT']), runtime, hypothesis_id)
        campaigns.event(campaign_id, 'CANDIDATE_TESTS', hypothesis_id=hypothesis_id, **unit_result)
        from orchestrator.codex_runner import commit_candidate_snapshot
        candidate_commit = commit_candidate_snapshot(REPO, hypothesis_id)
        campaigns.event(campaign_id, 'CANDIDATE_SNAPSHOT', hypothesis_id=hypothesis_id,
                        git_commit=candidate_commit)
        campaigns.event(campaign_id, "IMPLEMENTED", hypothesis_id=hypothesis_id, module=implemented["candidate_module"])
        await_authorized_stage(campaigns, campaign_id)
        seconds = remaining_job_seconds(campaigns, campaign_id)
        if seconds < 15:
            campaigns.set_status(campaign_id, "LIMIT_REACHED", "insufficient remaining GPU/wall budget")
            return
        manifest_path = Path(runtime) / "good_manifest.json"
        rid = queue_candidate(store, hypothesis_id, hypothesis_id, "juice_bottle", "smoke", "cuda",
                              manifest_path, seconds=min(300, seconds), personal_mode_authorized=True,
                              campaign_id=campaign_id, campaign_name=config['campaign']['name'])
        campaigns.attach_run(campaign_id, rid)
        campaigns.event(campaign_id, "EXPERIMENT_QUEUED", run_id=rid)
        outcome = run_worker_once(store, manifest_path, runtime, personal_mode_authorized=True,
                                  cancel=lambda: campaigns.get(campaign_id)["status"] == "STOPPING",
                                  expected_run_id=rid)
        if outcome.get("run_id") != rid:
            raise RuntimeError("GPU scheduler did not claim the campaign's queued run")
        run = store.get_run(rid)
        campaigns.update(campaign_id, active_run_id="")
        campaigns.event(campaign_id, "EXPERIMENT_FINISHED", run_id=rid, status=run["status"])
        if run["status"] != "COMPLETED":
            raise RuntimeError(f"experiment {rid} ended {run['status']}: {run['error']}")
        await_authorized_stage(campaigns, campaign_id, finishing=True)
        provenance_file = Path(runtime) / 'jobs' / rid / 'run_manifest.json'
        provenance = json.loads(provenance_file.read_text()) if provenance_file.is_file() and provenance_file.stat().st_size < 100_000 else {}
        prompt = ("Critique this immutable measured synthetic development run. Evaluate shortcuts, "
                  "normal false positives, generator gaps, latency, memory, reproducibility and whether "
                  "the hypothesis was supported. Never claim official LOCO performance. "
                  "The trusted runner enforces finite aligned [B,1,H,W] maps and disjoint normal "
                  "calibration/assessment images. gpu_seconds measures exclusive GPU slot wall time, "
                  "not CUDA kernel time; zero peak VRAM gives no evidence of GPU computation. "
                  "This does not establish real anomaly generalization. "
                  f"Run: {json.dumps(run, default=str)} Provenance: {json.dumps(provenance)} "
                  f"Hypothesis: {json.dumps(detail)}")
        review = budgeted_invoke(store, "critic", REPO, REPO / "schemas" / "review.schema.json",
                                 prompt, timeout=180, campaign=context, personal_mode=True)["response"]
        with store.connect() as con:
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                        (rid, time.time(), "CRITIC_REVIEW", json.dumps(review)))
        campaigns.event(campaign_id, "CRITIQUE", run_id=rid, review=review)
        store.transition(rid, "REVIEWED")
        action = review["recommended_next_action"]
        if action == 'reject':
            store.transition(rid, 'REJECTED', reason='Critic rejected development proxy evidence')
        elif action == 'promote_for_review':
            from orchestrator.decision_engine import eligible
            eligible_for_review, eligibility_reason = eligible(store.get_run(rid))
            if eligible_for_review:
                store.transition(rid, 'PROMOTED', reason='Human review requested; no official benchmark claim')
            else:
                campaigns.event(campaign_id, 'PROMOTION_WITHHELD', run_id=rid, reason=eligibility_reason)
        from orchestrator.mlflow_tracker import enqueue_run_tracking
        enqueue_run_tracking(store, rid)
        store.set_hypothesis_status(hypothesis_id, "COMPLETED")
        campaigns.update(campaign_id, consecutive_failures=0)
        direction = f'evidence-followup:{hypothesis_id}:{action}'
        campaigns.update(campaign_id, next_hypothesis=direction)
        campaigns.event(campaign_id, "NEXT_DECISION", action=action,
                        next_hypothesis=direction,
                        evidence_run=rid, reason=review.get("objective_measured_results", ""))
    except CampaignHalted:
        return
    except Exception as exc:
        if store.get_run(campaigns.get(campaign_id)["active_run_id"] or "") is None:
            store.set_hypothesis_status(hypothesis_id, "FAILED")
        failures = campaigns.get(campaign_id)["consecutive_failures"] + 1
        campaigns.update(campaign_id, consecutive_failures=failures, active_run_id="")
        campaigns.event(campaign_id, "ATTEMPT_FAILED", hypothesis_id=hypothesis_id, error=str(exc))
        raise


def run_forever(campaign_id: str, runtime: Path, ledger: Path, idle_seconds=2):
    store = StateStore(ledger)
    campaigns = CampaignState(store)
    row = campaigns.get(campaign_id)
    if row is None:
        raise KeyError(campaign_id)
    config = json.loads(row["config_json"])
    preflight(config, runtime)
    # A prior worker crash must never restart an expensive RUNNING job.
    reconcile_unfinished_runs(store, campaigns, campaign_id, 'campaign worker restarted; job not replayed')
    while True:
        row = campaigns.get(campaign_id)
        if row["status"] in TERMINAL:
            return
        if row["status"] == "STOPPING":
            campaigns.set_status(campaign_id, "STOPPED", "operator stop")
            return
        if row["status"] == "PAUSED":
            time.sleep(idle_seconds)
            continue
        state, reason = stop_reason(campaigns, campaign_id)
        if state:
            campaigns.set_status(campaign_id, state, reason)
            return
        failures_before = campaigns.get(campaign_id)["consecutive_failures"]
        try:
            run_cycle(store, campaigns, campaign_id, runtime)
        except Exception as exc:
            if campaigns.get(campaign_id)["consecutive_failures"] == failures_before:
                campaigns.update(campaign_id, consecutive_failures=failures_before + 1)
            campaigns.event(campaign_id, "CONTROLLER_ERROR", error=str(exc))
            state, reason = stop_reason(campaigns, campaign_id)
            if state:
                campaigns.set_status(campaign_id, state, reason)
                return
            time.sleep(idle_seconds)
