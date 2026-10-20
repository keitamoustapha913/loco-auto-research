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
from orchestrator.budgets import CodexBudgetExhausted

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
            from orchestrator.campaign_config import remaining_wall
            if remaining_wall(row,time.time()) <= 0:
                campaigns.set_status(campaign_id,'LIMIT_REACHED','wall-time budget reached')
                raise CampaignHalted('wall-time budget reached')
            time.sleep(1)
            continue
        raise CampaignHalted(row['reason'] or row['status'])


def load_campaign_config(path=None):
    source = Path(path or REPO / "configs" / "autonomy.yaml").resolve(strict=True)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or not all(k in config for k in ('campaign','limits')):
        raise ValueError('invalid autonomy configuration')
    defaults=yaml.safe_load((REPO/'configs/autonomy.yaml').read_text())
    for section in ('research','tracking','security'):
        config[section]=dict(defaults[section], **config.get(section,{}))
    if 'research_hours' in config['limits'] and set(config['limits'])-{'research_hours','training_minutes_per_experiment'}:
        raise ValueError('do not mix time settings with legacy campaign limits')
    from orchestrator.campaign_config import normalize, time_based
    config = normalize(config)
    campaign, limits = config["campaign"], config["limits"]
    if campaign.get("mode") != "autonomous" or campaign.get("dataset") != "juice_bottle" or campaign.get("gpu_device") != "cuda":
        raise ValueError("campaign must use the approved juice_bottle CUDA development setup")
    if limits["max_parallel_gpu_jobs"] != 1:
        raise ValueError("one RTX 4090 requires exclusive GPU scheduling")
    if not time_based(config) and (limits["max_gpu_hours"] > float(os.getenv("LOCO_GPU_BUDGET_HOURS", "4")) or limits["max_job_seconds"] > int(os.getenv("LOCO_JOB_TIMEOUT_SECONDS", "1800")) or limits["max_codex_calls"] > int(os.getenv("LOCO_CODEX_MAX_CALLS", "30"))):
        raise ValueError("campaign limits exceed existing operator resource ceilings")
    if config["research"].get("official_test_access") is not False or config["research"].get("synthetic_anomalies_allowed") is not True:
        raise ValueError("official test data must remain excluded")
    if config["security"].get("execution_mode") != "personal_bwrap":
        raise ValueError("unknown execution boundary")
    if config['research'].get('study') == 'stage6_multiseed':
        from orchestrator.multiseed_study import prescribed_plan
        prescribed_plan(config)
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
    if config['research'].get('real_development_feedback'):
        from benchmark.feedback_data import verify_import, imported_root
        verify_import(imported_root(runtime))
        seconds = config['research'].get('feedback_cpu_timeout_seconds', 600)
        if not isinstance(seconds, int) or not 1 <= seconds <= 900:
            raise ValueError('feedback CPU limit must be within 1..900 seconds')
    return {"manifest_path": manifest_path, "fingerprint": manifest_fingerprint(manifest), "images": len(manifest["images"])}


def stop_reason(campaigns, campaign_id):
    row = campaigns.get(campaign_id)
    limits = json.loads(row["config_json"])["limits"]
    used = campaigns.usage(campaign_id)
    if row["status"] != "RUNNING":
        return row["status"], row["reason"]
    if used["wall_seconds"] >= limits["max_wall_hours"] * 3600:
        return "LIMIT_REACHED", "wall-time budget reached"
    from orchestrator.campaign_config import time_based
    if not time_based(json.loads(row["config_json"])) and used["gpu_seconds"] >= limits["max_gpu_hours"] * 3600:
        return "LIMIT_REACHED", "GPU-time budget reached"
    if limits["max_experiments"] is not None and used["experiments"] >= limits["max_experiments"]:
        return "LIMIT_REACHED", "experiment count reached"
    if limits["max_codex_calls"] is not None and used["codex_calls"] >= limits["max_codex_calls"]:
        return "BUDGET_EXHAUSTED", "campaign Codex-call budget exhausted"
    global_budget = campaigns.call_budget(campaign_id)['global']
    if global_budget['remaining'] == 0:
        return "BUDGET_EXHAUSTED", f"global Codex call budget exhausted ({global_budget['used']}/{global_budget['limit']}); LOCO_CODEX_MAX_CALLS"
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


def choose_proposal(store, excluded=(), allowed_ids=None):
    """Select evidence-ranked proposals; Codex provides the research ranking."""
    from orchestrator.controller import list_proposals
    available = [p for p in list_proposals(store) if p["status"] == "PROPOSED" and p["hypothesis_id"] not in excluded
                 and (allowed_ids is None or p["hypothesis_id"] in allowed_ids)]
    available.sort(key=lambda p: (json.loads(p["proposal_json"]).get("priority_rank", 999), p["created_at"]))
    return available[0] if available else None


def run_cycle(store: StateStore, campaigns: CampaignState, campaign_id: str, runtime: Path):
    """Perform at most one measured attempt; caller persists and repeats."""
    study = json.loads(campaigns.get(campaign_id)['config_json'])['research'].get('study')
    if study == 'stage6_multiseed':
        from orchestrator.multiseed_study import run_multiseed_cycle
        return run_multiseed_cycle(store, campaigns, campaign_id, runtime)
    if study == 'masked_context_ablation':
        from orchestrator.ablation_study import run_ablation_cycle
        return run_ablation_cycle(store, campaigns, campaign_id, runtime)
    from orchestrator.controller import (add_research_proposals, budgeted_invoke,
        implement_proposal, list_proposals, queue_candidate, run_worker_once, scout_literature)
    config = json.loads(campaigns.get(campaign_id)["config_json"])
    context = (campaigns, campaign_id)
    prior = campaigns.runs(campaign_id)
    directive = campaigns.get(campaign_id)['next_hypothesis']
    proposal = None
    research = config['research']
    from orchestrator.experiment_plan import plan_history, execute_plan, review_plan, validate_plan, register_plan, record_result
    # Resume the immutable comparison, never regenerate or replay a recorded cell.
    with store.connect() as con:
        reviewed_plans = {json.loads(r[0]).get('plan_id') for r in con.execute(
            "SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='CRITIQUE'", (campaign_id,))}
    for registered in plan_history(campaigns, campaign_id):
        pending = registered['plan']
        if pending['plan_id'] not in reviewed_plans:
            with store.connect() as con:
                previous_result = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_RESULT' ORDER BY id DESC", (campaign_id,)).fetchall()
            result = next((json.loads(r[0])['result'] for r in previous_result
                           if json.loads(r[0])['result']['plan_id']==pending['plan_id']), None)
            if result is None:
                result = execute_plan(store, campaigns, campaign_id, runtime, pending)
            store.set_hypothesis_status(pending['hypothesis_id'], 'INCOMPLETE' if result['status']=='INCOMPLETE' else 'COMPLETED')
            await_authorized_stage(campaigns, campaign_id, finishing=True)
            review_plan(store, campaigns, campaign_id, pending, result)
            return
    fresh_events = [event for event in campaigns.events(campaign_id, 100)
                    if event['kind'] == 'EVIDENCE_ADAPTED_HYPOTHESES']
    if not prior and research.get('fresh_hypotheses') and fresh_events:
        ids = set(json.loads(fresh_events[0]['payload_json']).get('hypotheses', []))
        fresh = [p for p in list_proposals(store)
                 if p['hypothesis_id'] in ids and p['status'] == 'PROPOSED']
        fresh.sort(key=lambda p: json.loads(p['proposal_json']).get('priority_rank', 999))
        if not fresh:
            raise RuntimeError('all fresh campaign hypotheses are exhausted; refusing stale proposals')
        proposal = fresh[0]
    if not prior and research.get('fresh_hypotheses') and not fresh_events:
        scout_file = REPO / 'literature' / 'scout_notes.json'
        if research.get('fresh_literature') or not scout_file.is_file():
            leads = scout_literature(store, campaign=context, personal_mode=True)
            campaigns.event(campaign_id, 'LITERATURE', leads=leads)
        else:
            leads = json.loads(scout_file.read_text()).get('result', {})
            campaigns.event(campaign_id, 'LITERATURE_REUSED', source=str(scout_file))
        previous = [r for r in store.list_runs(20)
                    if r['candidate_id'] != 'toy_normal_ae' and r['status'] in
                    {'COMPLETED', 'REVIEWED', 'PROMOTED', 'REJECTED', 'FAILED'}]
        evidence = []
        for run in previous[:5]:
            with store.connect() as con:
                critique = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' "
                                       "ORDER BY id DESC LIMIT 1", (run['run_id'],)).fetchone()
            evidence.append({'run_id': run['run_id'], 'hypothesis': run['hypothesis_id'], 'status': run['status'],
                             'metrics': json.loads(run['metrics_json']), 'failure': run['error'],
                             'critic': json.loads(critique['payload_json']) if critique else {}})
        prompt = ("Propose THREE distinct, ranked, falsifiable next hypotheses for good-only, "
                  "memory-bank-free logical anomaly heatmaps. Ground your choice in these literature "
                  "leads and measured prior results, including generator sensitivity and actual GPU use "
                  "where measured. THIS CAMPAIGN IS AN RTX 4090 CUDA PILOT: each proposed learning strategy "
                  "must actually train a PyTorch model on CUDA and infer on CUDA, with estimated_gpu_minutes "
                  "strictly greater than zero and within authorized campaign limits. No CPU-only screening proposals. Do not "
                  "prescribe one architecture; propose scientifically distinct mechanisms. Only normal "
                  "juice_bottle images are authorized for model learning; a one-category pilot cannot establish "
                  "claims across all five LOCO categories. Use only installed "
                  "Use the installed environment; external weights or dependencies require approval. "
                  "Select methods from the research charter, verified literature and measured evidence. "
                  "Do not assume a preferred architecture. "
                  "new_dependencies must be []. Distinguish real-development feedback from synthetic proxies, not official LOCO "
                  "results. Literature: " + json.dumps(leads) + " Prior evidence: " + json.dumps(evidence))
        generated = add_research_proposals(store, prompt, campaign=context, personal_mode=True,
                                           require_gpu_pilot=bool(research.get('require_torch_cuda')))
        ids = {item['hypothesis_id'] for item in generated}
        fresh = [p for p in list_proposals(store) if p['hypothesis_id'] in ids]
        fresh.sort(key=lambda p: json.loads(p['proposal_json']).get('priority_rank', 999))
        proposal = fresh[0] if fresh else None
        campaigns.event(campaign_id, 'EVIDENCE_ADAPTED_HYPOTHESES',
                        source_run=evidence[0]['run_id'] if evidence else '',
                        hypotheses=[p['hypothesis_id'] for p in fresh])
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
        if research.get('require_torch_cuda'):
            prompt += (' This is a CUDA pilot: all three proposals must train PyTorch on CUDA, '
                       'estimated_gpu_minutes must be greater than zero and within campaign limits; no CPU-only screen.')
        generated = add_research_proposals(store, prompt, campaign=context, personal_mode=True,
                                           require_gpu_pilot=bool(research.get('require_torch_cuda')))
        ids = {item['hypothesis_id'] for item in generated}
        fresh = [p for p in list_proposals(store)
                 if p['hypothesis_id'] in ids]
        fresh.sort(key=lambda p: json.loads(p['proposal_json']).get('priority_rank', 999))
        proposal = fresh[0] if fresh else None
        campaigns.event(campaign_id, 'EVIDENCE_ADAPTED_HYPOTHESES', source_run=previous['run_id'],
                        hypotheses=[p['hypothesis_id'] for p in fresh])
        campaigns.update(campaign_id, next_hypothesis='')
    if proposal is None:
        with store.connect() as con:
            scoped = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EVIDENCE_ADAPTED_HYPOTHESES' ORDER BY id DESC LIMIT 1", (campaign_id,)).fetchone()
        allowed = set(json.loads(scoped[0]).get('hypotheses', [])) if scoped else set()
        proposal = choose_proposal(store, allowed_ids=allowed)
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
        if research.get('require_torch_cuda'):
            prompt += (' This is a CUDA pilot: all three proposals must train PyTorch on CUDA, '
                       'estimated_gpu_minutes must be greater than zero and within campaign limits; no CPU-only screen.')
        generated = add_research_proposals(store, prompt, campaign=context, personal_mode=True,
                               require_gpu_pilot=bool(research.get('require_torch_cuda')))
        ids = {p['hypothesis_id'] for p in generated}
        campaigns.event(campaign_id, 'EVIDENCE_ADAPTED_HYPOTHESES', hypotheses=sorted(ids))
        proposal = choose_proposal(store, allowed_ids=ids)
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
    result = None
    try:
        plan = detail.get('experiment_plan')
        if not plan:
            store.set_hypothesis_status(hypothesis_id, 'INCOMPLETE')
            campaigns.event(campaign_id, 'EXPERIMENT_PLAN_REJECTED', hypothesis_id=hypothesis_id,
                            status='INCOMPLETE', reason='structured experiment_plan required; historical prose is not executable')
            raise ValueError('structured experiment_plan required; refusing default-arm execution')
        validate_plan(plan, hypothesis_id)
        register_plan(store, campaigns, campaign_id, plan)
        await_authorized_stage(campaigns, campaign_id)
        if any(a['execution']=='train' for a in plan['arms']) or plan['diagnostics']:
            implemented = implement_proposal(store, hypothesis_id, campaign=context, personal_mode=True,
                require_torch_cuda=bool(research.get('require_torch_cuda')) and any(a['execution']=='train' for a in plan['arms']), validation_runtime=runtime)
            from orchestrator.codex_runner import commit_candidate_snapshot
            candidate_commit = commit_candidate_snapshot(REPO, hypothesis_id)
            campaigns.event(campaign_id, 'CANDIDATE_SNAPSHOT', hypothesis_id=hypothesis_id, git_commit=candidate_commit)
            campaigns.event(campaign_id, 'IMPLEMENTED', hypothesis_id=hypothesis_id, module=implemented['candidate_module'])
        result = execute_plan(store, campaigns, campaign_id, runtime, plan)
        store.set_hypothesis_status(hypothesis_id, 'INCOMPLETE' if result['status']=='INCOMPLETE' else 'COMPLETED')
        await_authorized_stage(campaigns, campaign_id, finishing=True)
        review_plan(store, campaigns, campaign_id, plan, result)
    except CampaignHalted as exc:
        if detail.get('experiment_plan') and result is None:
            record_result(store,campaigns,campaign_id,detail['experiment_plan'],campaigns.runs(campaign_id),runtime,str(exc))
            store.set_hypothesis_status(hypothesis_id,'INCOMPLETE')
        return
    except CodexBudgetExhausted:
        raise
    except Exception as exc:
        if detail.get('experiment_plan') and result is None:
            record_result(store, campaigns, campaign_id, detail['experiment_plan'], campaigns.runs(campaign_id), runtime,
                          f'{type(exc).__name__}: {exc}')
        elif result is not None:
            campaigns.event(campaign_id,'PLAN_REVIEW_FAILED',plan_id=result['plan_id'],error=str(exc))
        store.set_hypothesis_status(hypothesis_id, 'INCOMPLETE')
        failures = campaigns.get(campaign_id)['consecutive_failures'] + 1
        campaigns.update(campaign_id, consecutive_failures=failures, active_run_id='')
        campaigns.event(campaign_id, 'ATTEMPT_FAILED', hypothesis_id=hypothesis_id, error=str(exc))
        raise


def close_unfinished_comparisons(store, campaigns, campaign_id, runtime, reason):
    from orchestrator.experiment_plan import plan_history,record_result
    with store.connect() as con:
        published={json.loads(r[0])['result']['plan_sha256'] for r in con.execute(
            "SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_RESULT'",(campaign_id,))}
    for entry in plan_history(campaigns,campaign_id):
        if entry['plan_sha256'] not in published:
            record_result(store,campaigns,campaign_id,entry['plan'],campaigns.runs(campaign_id),runtime,reason)
            store.set_hypothesis_status(entry['plan']['hypothesis_id'],'INCOMPLETE')


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
            from orchestrator.campaign_config import remaining_wall
            if remaining_wall(row,time.time()) <= 0:
                campaigns.set_status(campaign_id,"LIMIT_REACHED","wall-time budget reached")
                return
            time.sleep(idle_seconds)
            continue
        state, reason = stop_reason(campaigns, campaign_id)
        if state:
            close_unfinished_comparisons(store,campaigns,campaign_id,runtime,reason)
            campaigns.set_status(campaign_id, state, reason)
            return
        failures_before = campaigns.get(campaign_id)["consecutive_failures"]
        try:
            run_cycle(store, campaigns, campaign_id, runtime)
        except CampaignHalted as exc:
            close_unfinished_comparisons(store,campaigns,campaign_id,runtime,str(exc))
            if campaigns.get(campaign_id)['status'] not in TERMINAL:
                campaigns.set_status(campaign_id,'LIMIT_REACHED',str(exc))
            return
        except CodexBudgetExhausted as exc:
            campaigns.set_status(campaign_id, 'BUDGET_EXHAUSTED', str(exc))
            return
        except Exception as exc:
            if campaigns.get(campaign_id)["consecutive_failures"] == failures_before:
                campaigns.update(campaign_id, consecutive_failures=failures_before + 1)
            campaigns.event(campaign_id, "CONTROLLER_ERROR", error=str(exc))
            state, reason = stop_reason(campaigns, campaign_id)
            if state:
                campaigns.set_status(campaign_id, state, reason)
                return
            time.sleep(idle_seconds)
