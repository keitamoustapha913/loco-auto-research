"""Operator CLI: deterministic state gates around agent-driven research.

This CLI intentionally separates propose / implement / queue / worker from final
benchmarking; no hidden auto-approvals or sealed evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
import sys
from orchestrator.budgets import BudgetExceeded, max_job_seconds, enforce
from orchestrator.codex_runner import invoke, create_candidate_worktree
from orchestrator.gpu_queue import claim_next, gpu_lock
from orchestrator.experiment_runner import execute_run
from orchestrator.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]


def run_worker_once(store, manifest_path, runtime, personal_mode_authorized=False, cancel=None,
                    expected_run_id=None):
    with gpu_lock(runtime):
        duration = max_job_seconds()
        if expected_run_id:
            expected = store.get_run(expected_run_id)
            if not expected or expected['status'] != 'QUEUED':
                raise ValueError('expected campaign job is not queued')
            duration = min(duration, int(json.loads(expected['config_json']).get('timeout_seconds', duration)))
        job = claim_next(store, max_duration=duration, expected_run_id=expected_run_id)
        if not job:
            return {"status": "IDLE"}
        try:
            if personal_mode_authorized or cancel:
                outcome = execute_run(job, store, REPO, Path(manifest_path), Path(runtime),
                                      personal_mode_authorized=personal_mode_authorized, cancel=cancel)
            else:
                outcome = execute_run(job, store, REPO, Path(manifest_path), Path(runtime))
        except Exception as exc:
            current = store.get_run(job["run_id"])
            if current["status"] == "RUNNING":
                store.transition(job["run_id"], "FAILED", reason=str(exc))
            raise
        return {"run_id": job["run_id"], "status": outcome}


def budgeted_invoke(store, role, repo_root, schema, prompt, campaign=None, **kwargs):
    if campaign:
        campaigns, campaign_id = campaign
        row = campaigns.get(campaign_id)
        limits = json.loads(row['config_json'])['limits']
        remaining = int(row['created_at'] + limits['max_wall_hours'] * 3600 - __import__('time').time())
        if remaining < 10:
            raise RuntimeError('campaign wall-time budget exhausted before Codex call')
        kwargs['timeout'] = min(int(kwargs.get('timeout', 180)), remaining)
        campaigns.reserve_codex_call(campaign_id)
        campaigns.event(campaign_id, 'AGENT_STARTED', role=role)
    store.reserve_codex_call(role)
    try:
        result = invoke(role, repo_root, schema, prompt, **kwargs)
    except Exception as exc:
        if campaign:
            campaigns.event(campaign_id, 'AGENT_FAILED', role=role, error=str(exc))
        raise
    if campaign:
        campaigns.event(campaign_id, 'AGENT_COMPLETED', role=role)
    return result


def add_research_proposals(store, prompt, count=3, campaign=None, personal_mode=False,
                           require_gpu_pilot=False):
    schema = REPO / "schemas" / "proposals.schema.json"
    output = budgeted_invoke(store, "researcher", REPO, schema, prompt, timeout=240,
                             campaign=campaign, personal_mode=personal_mode)["response"]
    proposals = output["proposals"]
    if len(proposals) < count:
        raise ValueError(f"researcher returned {len(proposals)}; need at least {count}")
    families = {p["candidate_family"] for p in proposals}
    if len(families) < count:
        raise ValueError("expected materially different candidate families")
    ranks = [p.get('priority_rank') for p in proposals]
    if len(set(ranks)) != len(proposals) or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError('hypothesis priority ranks must be distinct positive integers')
    prior = [json.loads(row['proposal_json']) for row in list_proposals(store)]
    def signature(item):
        return ' '.join((item.get('claim', '') + ' ' + item.get('controlled_change', '')).lower().split())
    for proposal in proposals:
        minutes = proposal.get('estimated_gpu_minutes')
        if require_gpu_pilot and (isinstance(minutes, bool) or not isinstance(minutes, (int, float))
                                  or not 0 < minutes <= 5
                                  or 'cpu-only' in proposal.get('evaluation_plan', '').lower()):
            raise ValueError('CUDA pilot requires GPU-trained hypotheses within five minutes; CPU-only proposals remain research leads')
        if any(not isinstance(dep, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:[<>=!~]+[0-9][A-Za-z0-9_.-]*)?', dep)
               for dep in proposal.get('new_dependencies', [])):
            raise ValueError('new_dependencies must contain package names only; use [] for none')
        if any(proposal['hypothesis_id'] == old['hypothesis_id'] or
               SequenceMatcher(None, signature(proposal), signature(old)).ratio() > 0.92
               for old in prior):
            raise ValueError(f"duplicate research proposal: {proposal['hypothesis_id']}")
        prior.append(proposal)
    for proposal in proposals:
        store.add_proposal(proposal)
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)',
                        (__import__('time').time(), 'HYPOTHESIS_RANKING', json.dumps({
                            'hypothesis_id': proposal['hypothesis_id'], 'priority_rank': proposal['priority_rank'],
                            'priority_rationale': proposal['priority_rationale']})))
    return proposals


def scout_literature(store, campaign=None, personal_mode=False):
    """Untrusted citation leads, not independently verified paper findings."""
    response = budgeted_invoke(
        store, "scout", REPO, REPO / "schemas" / "literature.schema.json",
        "Investigate current logical anomaly localization research for MVTec LOCO AD. "
        "Look beyond CSAD, MuDeNet and EfficientAD; describe likely failure modes "
        "and inference memory requirements. Cite sources; if you cannot verify a "
        "reference, explicitly mark it unverified. Do not access any local test data. "
        "Do not call MCP tools; use a source URL only when you can verify it within the time limit.", timeout=120,
        campaign=campaign, personal_mode=personal_mode
    )["response"]
    filename = REPO / "literature" / "scout_notes.json"
    filename.write_text(json.dumps({"generated_by": "Codex literature scout; UNVERIFIED", "result": response}, indent=2))
    return response


def list_proposals(store):
    with store.connect() as con:
        rows = con.execute("SELECT * FROM hypotheses ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def implement_proposal(store, hypothesis_id, campaign=None, personal_mode=False,
                       require_torch_cuda=False):
    with store.connect() as con:
        row = con.execute("SELECT proposal_json FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)).fetchone()
    if not row:
        raise KeyError(hypothesis_id)
    proposal = json.loads(row["proposal_json"])
    tree = create_candidate_worktree(REPO, hypothesis_id)
    allowed_file = tree / "candidates" / hypothesis_id / "model.py"
    schema = REPO / "schemas" / "implementation.schema.json"
    instruction = (
        f"Write a minimal, falsifiable pilot implementation of hypothesis {hypothesis_id} "
        f"NOW as the module candidates/{hypothesis_id}/model.py "
        "inside your assigned worktree. The fixed trusted runner calls "
        "train(config: dict, allowed_train_manifest: Path, output_dir: Path), then "
        "load(output_dir / 'checkpoint_manifest.json', device=config['device']), then "
        "predict(images) where images are RGB float32 [B,3,H,W]. Return finite "
        "anomaly_maps [B,1,H,W] and image_scores [B]. Save "
        "output_dir/checkpoint_manifest.json and output_dir/model.pt; the trusted "
        "runner measures the latter artifact. Read-only scripts/run_candidate.py "
        "shows the precise protocol. Do not alter its calling convention. "
        "Actually create model.py, a short README, and optionally test_model.py and candidate_manifest.json; "
        "a proposal or plan alone fails verification. "
        "If you write tests, use the system temporary directory for fixtures because candidate source is mounted read-only during tests. "
        "Do not change other directories. " +
        ("This pilot must implement a real trainable PyTorch model. If config['device'] is cuda, "
         "move model and training tensors to CUDA, train there, and run inference there. "
         "Do not claim GPU execution if no CUDA tensors are used. " if require_torch_cuda else "") +
        f"Proposal: {json.dumps(proposal, sort_keys=True)}"
    )
    import subprocess
    from orchestrator.experiment_runner import verify_candidate_source
    feedback = ''
    for attempt in range(2):
        response = budgeted_invoke(store, "implementer", REPO, schema, instruction + feedback,
                                   timeout=600, worktree=tree, campaign=campaign, personal_mode=personal_mode)
        with store.connect() as con:
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)",
                        (__import__('time').time(), 'IMPLEMENTER_RESPONSE', json.dumps({
                            'hypothesis_id': hypothesis_id, 'attempt': attempt + 1,
                            'response': response['response']})))
        changed = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=all"],
                                 check=True, capture_output=True, text=True).stdout.splitlines()
        approved_prefix = f"candidates/{hypothesis_id}/"
        approved_names = {'model.py', 'README.md', 'test_model.py', 'candidate_manifest.json'}
        approved_files = {approved_prefix + name for name in approved_names}
        for line in changed:
            filename = line[3:].replace('"', '')
            if filename not in approved_files or ".." in Path(filename).parts or (tree / filename).is_symlink():
                raise PermissionError(f"Agent attempted to modify out-of-scope path: {filename}")
        for item in allowed_file.parent.rglob('*'):
            if item.is_symlink() or (item.is_file() and item.name not in approved_names):
                raise PermissionError(f"Agent wrote an unapproved candidate file: {item.name}")
        if response["response"].get("new_dependencies") or response['response'].get('external_weights'):
            raise PermissionError("Agent requested dependencies or external weights; provenance and operator approval required")
        try:
            if response['response'].get('candidate_id') != hypothesis_id or not all(
                    response['response'].get(k) is True for k in ('train_contract', 'load_contract', 'predict_contract')):
                raise ValueError('implementer did not attest the assigned candidate contract')
            if not allowed_file.is_file():
                raise ValueError('implementation missing model.py')
            verify_candidate_source(allowed_file)
            break
        except ValueError as exc:
            if attempt == 1:
                raise
            feedback = f"\nYour first attempt failed the trusted static check: {exc}. Repair only the assigned candidate files and return accurate contract claims."
    store.set_hypothesis_status(hypothesis_id, "IMPLEMENTED")
    return {"worktree": str(tree), "candidate_module": str(allowed_file), "agent": response["response"]}


def queue_candidate(store, hypothesis_id, candidate_id, category, stage, device, manifest_path, seconds=300,
                    personal_mode_authorized=False, campaign_id=None, campaign_name=None):
    from benchmark.data_guard import load_manifest, manifest_fingerprint
    from orchestrator.experiment_runner import candidate_file, sha256
    if candidate_id != "toy_normal_ae" and not personal_mode_authorized and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Operator must verify OS isolation before queueing untrusted candidates")
    module = candidate_file(REPO, candidate_id)
    manifest, paths = load_manifest(manifest_path)
    if not any(item["category"] == category for item in manifest["images"]):
        raise ValueError("category absent from authorized good-only manifest")
    enforce(store, next_gpu_seconds=seconds if device == "cuda" else 0)
    timeout = min(int(seconds), max_job_seconds())
    cfg = {"device": device, "size": 128, "epochs": 1, "seed": 7, "timeout_seconds": timeout, "category": category, "model_sha256": sha256(module)}
    if campaign_id:
        cfg["campaign_id"] = campaign_id
        if campaign_name:
            cfg["campaign_name"] = campaign_name
        cfg["execution_mode"] = "personal_bwrap" if personal_mode_authorized else "isolated"
    candidate_tree = REPO / "candidates" / "worktrees" / candidate_id
    if candidate_tree.is_dir():
        git_sha = subprocess.run(["git", "-C", str(candidate_tree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    else:
        git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    run_id = store.create_run(hypothesis_id, candidate_id, category, stage, cfg, git_sha=git_sha, manifest_sha256=manifest_fingerprint(manifest))
    for state in ("IMPLEMENTED", "VALIDATED", "QUEUED"):
        store.transition(run_id, state)
    return run_id


def run_research_campaign(store, manifest_path, device="cuda", max_experiments=3):
    """Bounded, operator-enabled Codex -> implementation -> worker -> critic cycle.

    This path is refused until OS isolation is independently verified. It does
    not and cannot open the sealed LOCO official test directory.
    """
    if os.getenv("LOCO_ALLOW_UNATTENDED") != "1" or os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Set LOCO_ALLOW_UNATTENDED=1 only AFTER restricted OS identities/mounts are tested")
    if not manifest_path:
        raise ValueError("campaign needs --manifest")
    from benchmark.data_guard import load_manifest, CATEGORIES
    manifest, _ = load_manifest(manifest_path)
    if not list_proposals(store):
        leads = scout_literature(store)
        add_research_proposals(store, "Propose 3 genuinely distinct, falsifiable MVTec LOCO logical heatmap methods; normal-only learning, no inference memory bank, no sealed benchmark access. Treat these scout leads as UNVERIFIED until independently checked: " + json.dumps(leads))
    completed = []
    proposals = [p for p in list_proposals(store) if p["status"] == "PROPOSED"]
    proposals.sort(key=lambda row: json.loads(row['proposal_json']).get('priority_rank', 999))
    for row in proposals[:max_experiments]:
        enforce(store, next_gpu_seconds=300 if device.startswith("cuda") else 0)
        hypothesis_id = row["hypothesis_id"]
        proposal = json.loads(row["proposal_json"])
        if proposal["new_dependencies"]:
            store.set_hypothesis_status(hypothesis_id, "INVALID")
            continue
        try:
            implement_proposal(store, hypothesis_id)
            eligible_categories = [cat for cat in CATEGORIES if any(item["category"] == cat and item["split"] == "validation/good" for item in manifest["images"])]
            if not eligible_categories:
                raise ValueError("No permitted normal validation images")
            category = eligible_categories[0]
            rid = queue_candidate(store, hypothesis_id, hypothesis_id, category, "smoke", device, manifest_path, 300)
            outcome = run_worker_once(store, manifest_path, Path(os.getenv("LOCO_RUNTIME", ".runtime")))
            data = store.get_run(rid)
            if data["status"] == "COMPLETED":
                measured = data["metrics_json"]
                critique = budgeted_invoke(store, "critic", REPO, REPO / "schemas" / "review.schema.json", f"Review immutable synthetic-only measured run {rid}: {measured}. Hypothesis: {proposal}. Never equate proxy results with official LOCO.", timeout=180)
                with store.connect() as con:
                    con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)", (rid, __import__('time').time(), "CRITIC_REVIEW", json.dumps(critique["response"])))
                from orchestrator.decision_engine import eligible
                ok, reason = eligible(data)
                store.transition(rid, "REVIEWED")
                store.transition(rid, "PROMOTED" if ok and critique["response"]["recommended_next_action"] != "reject" else "REJECTED", reason=reason)
                from orchestrator.mlflow_tracker import enqueue_run_tracking
                enqueue_run_tracking(store, rid)
            store.set_hypothesis_status(hypothesis_id, "COMPLETED")
            completed.append({"hypothesis": hypothesis_id, "run_id": rid, "outcome": outcome})
        except Exception as exc:
            store.set_hypothesis_status(hypothesis_id, "FAILED")
            completed.append({"hypothesis": hypothesis_id, "error": str(exc)})
    # Persist a factual audit report; agent interpretation cannot overwrite metrics.
    from orchestrator.reporting import write_campaign_report
    report_file = write_campaign_report(store, Path(os.getenv("LOCO_RUNTIME", ".runtime")) / "exports")
    # If the scheduled batch is exhausted, use the observed failures/scores to
    # seed the next iteration rather than reverting to a fixed search list.
    remaining = [p for p in list_proposals(store) if p["status"] == "PROPOSED"]
    followup = []
    if not remaining and completed:
        history = [{"hypothesis":x.get("hypothesis"), "run":x.get("run_id"), "status":store.get_run(x["run_id"])["status"] if x.get("run_id") else "FAILED", "metrics":json.loads(store.get_run(x["run_id"])["metrics_json"]) if x.get("run_id") else {}, "error":x.get("error", "")} for x in completed]
        try:
            followup = add_research_proposals(store, "Generate THREE NEW, distinct and falsifiable research hypotheses based on these measured outcomes. Critically consider synthetic shortcut learning; do not retest rejected identical ideas: " + json.dumps(history))
        except Exception as exc:
            followup = [{"error":str(exc)}]
    return {"completed": completed, "report_file": str(report_file), "next_proposals":followup, "disclaimer": "Development synthetic proxies ONLY, no official LOCO test scores"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guarded LOCO research controller")
    ap.add_argument("command", choices=["init", "status", "pause", "resume", "reconcile", "propose", "implement", "queue-toy", "worker-once", "review", "cycle", "queue-candidate", "campaign", "scout"])
    ap.add_argument("--manifest", help="validated good-only manifest JSON")
    ap.add_argument("--hypothesis", help="hypothesis ID for implementation or review")
    ap.add_argument("--run", help="run ID for review")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--candidate", help="candidate ID for validated queue submission")
    ap.add_argument("--category", default="breakfast_box")
    ap.add_argument("--stage", default="smoke", choices=["smoke", "pilot", "full"])
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-experiments", type=int, default=3)
    ap.add_argument("--prompt", default="Propose three distinct falsifiable methods for LOCO logical anomaly localization without memory banks; focus on good-only learning and safe development validation.")
    args = ap.parse_args(argv)
    store = StateStore()
    runtime = Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()
    if args.command == "init":
        outcome = {"db": str(store.path), "runtime": str(runtime), "status": "initialized"}
    elif args.command == "status":
        outcome = {"paused": store.is_paused(), "gpu_hours_recorded": store.total_gpu_seconds() / 3600, "runs": store.list_runs(25), "tracking_pending": len(store.pending_tracking(10000))}
    elif args.command in ("pause", "resume"):
        store.set_paused(args.command == "pause")
        outcome = {"paused": store.is_paused(), "note": "Active jobs are not killed by pause; no new jobs will start."}
    elif args.command == "reconcile":
        from orchestrator.gpu_queue import GpuBusy
        try:
            with gpu_lock(runtime):
                outcome = {"interrupted": store.reconcile_after_restart()}
        except GpuBusy:
            outcome = {"interrupted": [], "note": "GPU worker lock held; no reconciliation performed"}
    elif args.command == "scout":
        enforce(store)
        outcome = {"literature_leads": scout_literature(store), "warning": "Unverified literature summaries; human/source cross-check required"}
    elif args.command == "propose":
        enforce(store)
        outcome = {"proposals": add_research_proposals(store, args.prompt)}
    elif args.command == "implement":
        enforce(store)
        if not args.hypothesis:
            ap.error("--hypothesis is required")
        outcome = implement_proposal(store, args.hypothesis)
    elif args.command == "queue-candidate":
        if not (args.manifest and args.hypothesis and args.candidate):
            ap.error("queue-candidate needs --manifest, --hypothesis, --candidate")
        if args.stage == "full" and os.getenv("LOCO_APPROVE_FULL") != "1":
            raise PermissionError("Full experimental training requires explicit operator approval")
        outcome = {"run_id": queue_candidate(store, args.hypothesis, args.candidate, args.category, args.stage, args.device, args.manifest, args.timeout), "status": "QUEUED"}
    elif args.command == "campaign":
        outcome = run_research_campaign(store, args.manifest, args.device, max(1, min(args.max_experiments, 20)))
    elif args.command == "queue-toy":
        if not args.manifest:
            ap.error("--manifest is required")
        from benchmark.data_guard import load_manifest, manifest_fingerprint
        manifest, _ = load_manifest(args.manifest)
        run_id = store.create_run("fixture-demo", "toy_normal_ae", "fixture", "smoke", {"device": args.device, "manifest": str(Path(args.manifest).resolve()), "epochs": 1, "size": 64}, manifest_sha256=manifest_fingerprint(manifest))
        for s in ("IMPLEMENTED", "VALIDATED", "QUEUED"):
            store.transition(run_id, s)
        outcome = {"run_id": run_id, "status": "QUEUED"}
    elif args.command == "worker-once":
        if not args.manifest:
            ap.error("--manifest is required")
        outcome = run_worker_once(store, args.manifest, runtime)
    elif args.command == "review":
        from orchestrator.decision_engine import eligible
        if not args.run:
            ap.error("--run is required")
        run = store.get_run(args.run)
        if run is None:
            raise KeyError(args.run)
        ok, rationale = eligible(run)
        if run["status"] == "COMPLETED":
            store.transition(args.run, "REVIEWED")
            store.transition(args.run, "PROMOTED" if ok else "REJECTED", reason=rationale)
            from orchestrator.mlflow_tracker import enqueue_run_tracking
            enqueue_run_tracking(store, args.run)
        outcome = {"run_id": args.run, "eligible": ok, "reason": rationale, "status": store.get_run(args.run)["status"]}
    elif args.command == "cycle":
        # Operator-triggered, bounded autonomous *research* step. Never accesses sealed data.
        enforce(store)
        existing = list_proposals(store)
        if not existing:
            scout = scout_literature(store)
            outcome = {"stage": "proposals", "scout": scout, "proposals": add_research_proposals(store, args.prompt + " Unverified literature scout leads: " + json.dumps(scout))}
        else:
            outcome = {"stage": "pending_operator_validation", "proposals_count": len(existing), "next": "controller implement --hypothesis ID; validate candidate then enqueue using approved runner"}
    print(json.dumps(outcome, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
