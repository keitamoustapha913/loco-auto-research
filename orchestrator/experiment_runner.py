"""Deterministic subprocess executor with fixed command, time and output policies."""
from __future__ import annotations
import hashlib
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from orchestrator.budgets import max_job_seconds
from orchestrator.state_store import StateStore
from orchestrator.mlflow_tracker import enqueue_run_tracking


def safe_candidate_id(candidate_id):
    if not candidate_id or not all(c.isascii() and (c.isalnum() or c in "_-") for c in candidate_id):
        raise ValueError("candidate ID must be ASCII alphanumeric with _ or -")
    return candidate_id


def candidate_file(repo_root, candidate_id):
    root = (Path(repo_root) / "candidates").resolve(strict=True)
    cid = safe_candidate_id(candidate_id)
    local = root / cid / "model.py"
    worktree = root / "worktrees" / cid / "candidates" / cid / "model.py"
    existing = next((p for p in (local, worktree) if p.is_file()), None)
    if existing is None:
        raise FileNotFoundError(f"No candidate module for {cid}")
    path = existing.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("candidate path escapes candidates/ root")
    return path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_candidate_source(path):
    """Non-executing protocol preflight; OS sandbox remains the security boundary."""
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    methods = {n.name for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not {'train', 'load'}.issubset(functions) or 'predict' not in methods | functions:
        raise ValueError('candidate source lacks train/load/predict contract')
    if any(token in source.lower() for token in ('official_test', 'loco-sealed', 'ground_truth')):
        raise ValueError('candidate source references protected evaluation data')
    return {'source_sha256': sha256(path), 'protocol_symbols': sorted(functions | methods)}


def execute_run(run: dict, store: StateStore, repo_root: Path, manifest_path: Path, runtime: Path):
    """Never execute untrusted candidate in controller process.

    Host operator MUST apply OS/container containment to candidate subprocess;
    cwd or restricted CLI flags alone are not a permission boundary.
    """
    from benchmark.data_guard import load_manifest
    from orchestrator.budgets import enforce
    enforce(store)
    _, paths = load_manifest(manifest_path)
    if not paths:
        raise ValueError("empty good-image manifest")
    module_file = candidate_file(repo_root, run["candidate_id"])
    if run["candidate_id"] != "toy_normal_ae" and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Non-toy candidate execution disabled until operator verifies OS-level isolation (deploy/SECURITY.md)")
    job_dir = runtime / "jobs" / run["run_id"]
    job_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(run["config_json"])
    isolated = run['candidate_id'] != 'toy_normal_ae' or os.getenv('LOCO_FORCE_SANDBOX_TOY') == '1'
    input_dir = runtime / 'controller' / 'job_inputs' / run['run_id'] if isolated else job_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    config_path = input_dir / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2))
    result_path = job_dir / "measurement.json"
    command = [sys.executable, "-m", "scripts.run_candidate", "--module", str(module_file), "--manifest", str(manifest_path), "--config", str(config_path), "--output", str(job_dir), "--result", str(result_path)]
    environment = {key: value for key, value in os.environ.items() if key not in {"OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "MLFLOW_TRACKING_TOKEN"}}
    environment["PYTHONPATH"] = str(repo_root)
    if isolated:
        from orchestrator.sandbox import command as sandbox_command, write_sandbox_manifest
        sandbox_manifest, data_root = write_sandbox_manifest(manifest_path, input_dir)
        command = sandbox_command(repo_root, module_file, data_root, job_dir,
            ['-m', 'scripts.run_candidate', '--module', '/candidate/model.py',
             '--manifest', '/inputs/good_manifest.json', '--config', '/inputs/config.json',
             '--output', '/output', '--result', '/output/measurement.json'],
            use_gpu=config.get('device', 'cpu').startswith('cuda'),
            input_files={'config.json': config_path, 'good_manifest.json': sandbox_manifest})
        # Bubblewrap clears its environment; no controller or Codex secrets enter.
        environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
    timeout = min(max_job_seconds(), int(config.get("timeout_seconds", max_job_seconds())))
    started = time.monotonic()
    with (job_dir / "stdout.log").open("wb") as stdout, (job_dir / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=repo_root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        status = "COMPLETED"
        try:
            returncode = process.wait(timeout=timeout)
            if returncode != 0:
                status = "FAILED"
        except subprocess.TimeoutExpired:
            status = "TIMED_OUT"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    elapsed = time.monotonic() - started
    if status == "FAILED":
        err_tail = (job_dir / "stderr.log").read_bytes()[-20000:].lower()
        if b"out of memory" in err_tail or b"cuda error: out of memory" in err_tail:
            status = "OOM"
    try:
        if status == "COMPLETED":
            measured = json.loads(result_path.read_text())
            metrics = measured["metrics"]
            store.set_run_result(run["run_id"], metrics, gpu_seconds=elapsed if config.get("device", "cpu").startswith("cuda") else 0, artifact_dir=str(job_dir))
            store.transition(run["run_id"], "COMPLETED", expect="RUNNING")
        else:
            store.account_gpu_usage(run["run_id"], elapsed if config.get("device", "cpu").startswith("cuda") else 0)
            store.transition(run["run_id"], status, reason=f"Candidate process {status}; inspect stderr.log", expect="RUNNING")
        enqueue_run_tracking(store, run["run_id"], artifact_dir=job_dir)
        return status
    except Exception as exc:
        if store.get_run(run["run_id"])["status"] == "RUNNING":
            store.transition(run["run_id"], "FAILED", reason=f"runner post-processing: {exc}", expect="RUNNING")
        raise
