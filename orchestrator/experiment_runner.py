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
    if Path(path).stat().st_size > 1_000_000:
        raise ValueError('candidate source exceeds 1 MB review limit')
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    methods = {n.name for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not {'train', 'load'}.issubset(functions) or 'predict' not in methods | functions:
        raise ValueError('candidate source lacks train/load/predict contract')
    if any(token in source.lower() for token in ('official_test', 'loco-sealed', 'ground_truth')):
        raise ValueError('candidate source references protected evaluation data')
    train_node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'train')
    positional = [n.arg for n in train_node.args.posonlyargs + train_node.args.args]
    if len(positional) < 3 or positional[0] != 'config' or 'manifest' not in positional[1] or not any(t in positional[2] for t in ('output', 'checkpoint')):
        raise ValueError('candidate train signature must be train(config, allowed_manifest, output_dir)')
    return {'source_sha256': sha256(path), 'protocol_symbols': sorted(functions | methods)}


def probe_candidate_interface(repo_root, module_file, good_root, runtime, candidate_id):
    """Import untrusted code only inside the no-network CPU sandbox before GPU queueing."""
    from orchestrator.sandbox import command as sandbox_command
    safe_candidate_id(candidate_id)
    probe_root = Path(runtime).resolve() / 'probes'
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_root / safe_candidate_id(candidate_id)
    probe_dir.mkdir(parents=True, exist_ok=True)
    if probe_dir.is_symlink() or not probe_dir.resolve(strict=True).is_relative_to(probe_root):
        raise PermissionError('candidate probe output escapes runtime')
    script = ("import importlib.util; "
              "s=importlib.util.spec_from_file_location('candidate','/candidate/model.py'); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "assert callable(getattr(m,'train',None)); "
              "assert callable(getattr(m,'load',None)); "
              "print('interface import OK')")
    cmd = sandbox_command(repo_root, module_file, good_root, probe_dir, ['-c', script], use_gpu=False)
    try:
        result = subprocess.run(cmd, cwd=repo_root, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                                capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('candidate CPU interface probe timed out') from exc
    if result.returncode or 'interface import OK' not in result.stdout:
        raise ValueError(f'candidate CPU interface probe failed: {result.stderr[-500:]}')
    return True


def run_candidate_unit_tests(repo_root, module_file, good_root, runtime, candidate_id):
    """Execute optional candidate tests only in the same no-network CPU boundary."""
    from orchestrator.sandbox import command as sandbox_command
    module = Path(module_file).resolve(strict=True)
    if not (module.parent / 'test_model.py').is_file():
        return {'tests': 'none provided'}
    probe_root = Path(runtime).resolve() / 'probes'
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_root / safe_candidate_id(candidate_id)
    probe_dir.mkdir(parents=True, exist_ok=True)
    if probe_dir.is_symlink() or not probe_dir.resolve(strict=True).is_relative_to(probe_root):
        raise PermissionError('candidate test output escapes runtime')
    cmd = sandbox_command(repo_root, module, good_root, probe_dir,
                          ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', '/candidate/test_model.py'],
                          use_gpu=False)
    try:
        result = subprocess.run(cmd, cwd=repo_root, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                                capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('candidate CPU unit tests timed out') from exc
    if result.returncode:
        raise ValueError(f'candidate CPU unit tests failed: {(result.stdout + result.stderr)[-1000:]}')
    return {'tests': 'passed', 'output': result.stdout[-300:]}


def execute_run(run: dict, store: StateStore, repo_root: Path, manifest_path: Path, runtime: Path,
                personal_mode_authorized=False, cancel=None):
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
    if run["candidate_id"] != "toy_normal_ae" and not personal_mode_authorized and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
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
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancel and cancel():
                    status = "CANCELLED"
                    break
                if time.monotonic() >= deadline:
                    status = "TIMED_OUT"
                    break
                time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))
            if status == "COMPLETED" and process.returncode != 0:
                status = "FAILED"
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
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
