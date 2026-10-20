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


def read_training_timing(path):
    """Bound reads from candidate-writable audit artifacts."""
    import math
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size>4096: return {}
    try:
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or value.get('status') not in {'RUNNING','COMPLETED','FAILED'}: return {}
        for field in ('started_at','duration_seconds'):
            number=value.get(field)
            if field=='duration_seconds' and number is None:continue
            if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or number<0:return {}
        return value
    except (OSError,ValueError):return {}


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
    """Require executable candidate CPU tests in the no-network boundary."""
    from orchestrator.sandbox import command as sandbox_command
    module = Path(module_file).resolve(strict=True)
    if not (module.parent / 'test_model.py').is_file():
        raise ValueError('candidate CPU unit tests failed: mandatory test_model.py is missing')
    probe_root = Path(runtime).resolve() / 'probes'
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_root / safe_candidate_id(candidate_id)
    probe_dir.mkdir(parents=True, exist_ok=True)
    if probe_dir.is_symlink() or not probe_dir.resolve(strict=True).is_relative_to(probe_root):
        raise PermissionError('candidate test output escapes runtime')
    # pytest exits zero for an entirely skipped suite. Count successful test
    # bodies, not collection/setup, and fail closed when none actually run.
    test_script = '''import sys
import pytest
class Results:
    passed = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' and report.passed and not hasattr(report, 'wasxfail'):
            self.passed += 1
results = Results()
code = pytest.main(['-q', '-ra', '-p', 'no:cacheprovider', '/candidate/test_model.py'], plugins=[results])
if code == 0 and results.passed == 0:
    print('candidate validation rejected: no tests executed successfully (all skipped or xfailed)')
    code = 5
sys.exit(code)
'''
    cmd = sandbox_command(repo_root, module, good_root, probe_dir,
                          ['-c', test_script],
                          use_gpu=False)
    try:
        result = subprocess.run(cmd, cwd=repo_root, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                                capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('candidate CPU unit tests timed out') from exc
    if result.returncode:
        raise ValueError(f'candidate CPU unit tests failed: {(result.stdout + result.stderr)[-8000:]}')
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
    from orchestrator.experiment_plan import execution_source
    module_file = execution_source(run, store, runtime, repo_root)
    if run["candidate_id"] != "toy_normal_ae" and not personal_mode_authorized and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Non-toy candidate execution disabled until operator verifies OS-level isolation (deploy/SECURITY.md)")
    job_dir = runtime / "jobs" / run["run_id"]
    job_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(run["config_json"])
    if config.get('execution_kind') == 'frozen':
        import shutil
        from orchestrator.experiment_plan import verify_baseline
        _, origin = verify_baseline(store, config['frozen_baseline'], repo_root)
        if origin['manifest_sha256'] != run['manifest_sha256']:
            raise ValueError('baseline training dataset differs from the planned dataset')
        checkpoint_dir = job_dir / 'checkpoint'
        checkpoint_dir.mkdir()
        for name in ('model.pt', 'checkpoint_manifest.json'):
            shutil.copyfile(Path(origin['artifact_dir'])/'checkpoint'/name, checkpoint_dir/name)
    isolated = run['candidate_id'] != 'toy_normal_ae' or os.getenv('LOCO_FORCE_SANDBOX_TOY') == '1'
    input_dir = runtime / 'controller' / 'job_inputs' / run['run_id'] if isolated else job_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    config_path = input_dir / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2))
    result_path = job_dir / "measurement.json"
    command = [sys.executable, "-m", "scripts.run_candidate", "--module", str(module_file), "--manifest", str(manifest_path), "--config", str(config_path), "--output", str(job_dir), "--result", str(result_path)]
    environment = {key: value for key, value in os.environ.items() if key not in {"OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "MLFLOW_TRACKING_TOKEN"}}
    environment["PYTHONPATH"] = str(repo_root)
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if isolated:
        from orchestrator.sandbox import command as sandbox_command, write_sandbox_manifest
        sandbox_manifest, data_root = write_sandbox_manifest(manifest_path, input_dir)
        diagnostic_file = None
        if config.get('diagnostic_provider'):
            reference = config['diagnostic_provider']
            diagnostic_file = candidate_file(repo_root, reference['candidate_id'])
            if sha256(diagnostic_file) != reference['source_sha256']:
                raise ValueError('shared diagnostic implementation changed')
        command = sandbox_command(repo_root, module_file, data_root, job_dir,
            ['-m', 'scripts.run_candidate', '--module', '/candidate/model.py',
             '--manifest', '/inputs/good_manifest.json', '--config', '/inputs/config.json',
             '--output', '/output', '--result', '/output/measurement.json'],
            use_gpu=config.get('device', 'cpu').startswith('cuda'),
            input_files={'config.json': config_path, 'good_manifest.json': sandbox_manifest},
            pinned_source=bool(config.get('frozen_baseline')), diagnostic_file=diagnostic_file)
        # Bubblewrap clears its environment; no controller or Codex secrets enter.
        environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
    timeout = min(max_job_seconds(config), int(config.get("timeout_seconds", max_job_seconds(config))))
    stages = []
    if config.get('cuda_smoke_required'):
        if not config.get('device', '').startswith('cuda'):
            raise ValueError('CUDA smoke cannot run on a CPU job')
        smoke_command = list(command)
        index = smoke_command.index('scripts.run_candidate')
        smoke_command[index] = 'scripts.candidate_smoke'
        smoke_command[smoke_command.index('--output', index)+1] = '/output/cuda_smoke' if isolated else str(job_dir / 'cuda_smoke')
        result_index = smoke_command.index('--result', index)
        del smoke_command[result_index:result_index+2]
        stages.append(('cuda_smoke', smoke_command, min(30, timeout)))
    stages.append(('pilot', command, timeout))
    started = time.monotonic()
    deadline = started + timeout  # Both stages share the original job allowance.
    smoke_seconds = 0.0
    status = 'COMPLETED'
    failed_stage = ''
    for stage, stage_command, stage_limit in stages:
        if cancel and cancel():
            status = 'CANCELLED'
            break
        if time.monotonic() >= deadline:
            status = 'TIMED_OUT'
            break
        prefix = 'cuda_smoke_' if stage == 'cuda_smoke' else ''
        stage_started = time.monotonic()
        with (job_dir / (prefix + "stdout.log")).open("wb") as stdout, (job_dir / (prefix + "stderr.log")).open("wb") as stderr:
            process = subprocess.Popen(stage_command, cwd=repo_root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                stage_deadline = min(deadline, stage_started + stage_limit)
                while process.poll() is None:
                    if cancel and cancel():
                        status = "CANCELLED"
                        break
                    timing_file = job_dir / 'training_timing.json'
                    if stage == 'pilot' and config.get('training_budget_seconds') and timing_file.is_file() and not timing_file.is_symlink():
                        timing = read_training_timing(timing_file)
                        if timing.get('status') == 'RUNNING' and time.time()-timing['started_at'] >= config['training_budget_seconds']:
                            status = 'TIMED_OUT'
                            break
                    if time.monotonic() >= stage_deadline:
                        status = "TIMED_OUT"
                        break
                    time.sleep(min(0.5, max(0.01, stage_deadline - time.monotonic())))
                if status == "COMPLETED" and process.returncode != 0:
                    status = "FAILED"
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if stage == 'cuda_smoke':
            smoke_seconds = time.monotonic() - stage_started
            with store.connect() as con:
                con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                            (run['run_id'], time.time(), 'CUDA_SMOKE_VALIDATION',
                             json.dumps({'status': status, 'seconds': smoke_seconds, 'timeout_seconds': stage_limit})))
        if status != 'COMPLETED':
            failed_stage = stage
            break
    elapsed = time.monotonic() - started
    if status == "FAILED":
        error_log = job_dir / ('cuda_smoke_stderr.log' if failed_stage == 'cuda_smoke' else 'stderr.log')
        err_tail = error_log.read_bytes()[-20000:].lower()
        if b"training-time budget exceeded" in err_tail:
            status = "TIMED_OUT"
        if b"out of memory" in err_tail or b"cuda error: out of memory" in err_tail:
            status = "OOM"
    try:
        if status == "COMPLETED":
            if result_path.is_symlink() or not result_path.resolve().is_relative_to(job_dir.resolve()):
                raise PermissionError('candidate result artifact escapes its assigned job')
            try:
                measured = json.loads(result_path.read_text())
                metrics = measured['metrics']
                if not isinstance(metrics, dict):
                    raise ValueError('measurement metrics must be an object')
                if config.get('plan_sha256') and measured.get('provenance',{}).get('effective_configuration') != config:
                    raise ValueError('effective configuration differs from the registered planned job')
            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                import traceback
                reason = f'Candidate result validation {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)[-8000:]}'
                store.set_run_result(run['run_id'], {}, gpu_seconds=elapsed if config.get('device', 'cpu').startswith('cuda') else 0,
                                     artifact_dir=str(job_dir))
                store.transition(run['run_id'], 'FAILED', reason=reason, expect='RUNNING')
                enqueue_run_tracking(store, run['run_id'], artifact_dir=job_dir)
                return 'FAILED'
            if config.get('frozen_baseline') and sha256(job_dir/'checkpoint/model.pt') != config['frozen_baseline']['checkpoint_sha256']:
                raise ValueError('frozen baseline checkpoint changed during job execution')
            if config.get('frozen_baseline') and sha256(job_dir/'checkpoint/checkpoint_manifest.json') != config['frozen_baseline']['checkpoint_manifest_sha256']:
                raise ValueError('frozen baseline metadata changed during job execution')
            if config.get('cuda_smoke_required'):
                metrics.update({'validation/cuda_smoke_passed': 1, 'validation/cuda_smoke_seconds': smoke_seconds})
            store.set_run_result(run["run_id"], metrics, gpu_seconds=elapsed if config.get("device", "cpu").startswith("cuda") else 0, artifact_dir=str(job_dir))
            store.transition(run["run_id"], "COMPLETED", expect="RUNNING")
        else:
            charged = elapsed if config.get("device", "cpu").startswith("cuda") else 0
            timing_file = job_dir/'training_timing.json'
            timing=read_training_timing(timing_file)
            if timing:
                duration=timing.get('duration_seconds',max(0,time.time()-timing['started_at']))
                store.set_run_result(run['run_id'], {'training/duration_seconds':duration}, gpu_seconds=charged, artifact_dir=str(job_dir))
            if failed_stage == 'cuda_smoke':
                store.set_run_result(run['run_id'], {'validation/cuda_smoke_passed': 0,
                    'validation/cuda_smoke_seconds': smoke_seconds}, gpu_seconds=charged, artifact_dir=str(job_dir))
            else:
                store.account_gpu_usage(run['run_id'], charged)
            reason = f"Candidate process {status}; inspect stderr.log"
            if failed_stage == 'cuda_smoke':
                log = job_dir / 'cuda_smoke_stderr.log'
                reason = f"CUDA smoke validation {status}: {log.read_text(errors='replace')[-8000:]}"
            elif (job_dir / 'stderr.log').is_file():
                reason += '\n' + (job_dir / 'stderr.log').read_text(errors='replace')[-8000:]
            store.transition(run["run_id"], status, reason=reason, expect="RUNNING")
        enqueue_run_tracking(store, run["run_id"], artifact_dir=job_dir)
        return status
    except Exception as exc:
        if store.get_run(run["run_id"])["status"] == "RUNNING":
            store.account_gpu_usage(run["run_id"], elapsed if config.get("device", "cpu").startswith("cuda") else 0)
            store.transition(run["run_id"], "FAILED", reason=f"runner post-processing: {exc}", expect="RUNNING")
        raise
