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
from orchestrator.controller_errors import CandidateValidationError
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


def verify_candidate_source(path, *, diagnostic_only=False):
    """Non-executing protocol preflight; OS sandbox remains the security boundary."""
    if Path(path).stat().st_size > 1_000_000:
        raise CandidateValidationError('candidate source exceeds 1 MB review limit')
    source = Path(path).read_text()
    try:tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:raise CandidateValidationError('candidate source syntax: '+str(exc)) from exc
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    methods = {n.name for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if diagnostic_only:
        if 'diagnose' not in functions:raise CandidateValidationError('frozen diagnostic provider requires diagnose; training interface is not required')
    elif not {'train', 'load'}.issubset(functions) or 'predict' not in methods | functions:
        raise CandidateValidationError('candidate source lacks train/load/predict contract')
    if any(token in source.lower() for token in ('official_test', 'loco-sealed', 'ground_truth')):
        raise CandidateValidationError('candidate source references protected evaluation data')
    if diagnostic_only:return {'source_sha256': sha256(path), 'protocol_symbols': sorted(functions | methods)}
    train_node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'train')
    positional = [n.arg for n in train_node.args.posonlyargs + train_node.args.args]
    if len(positional) < 3 or positional[0] != 'config' or 'manifest' not in positional[1] or not any(t in positional[2] for t in ('output', 'checkpoint')):
        raise CandidateValidationError('candidate train signature must be train(config, allowed_manifest, output_dir)')
    return {'source_sha256': sha256(path), 'protocol_symbols': sorted(functions | methods)}


def read_training_timing(path):
    """Bound reads from candidate-writable audit artifacts; never treat setup as training."""
    import math
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size>4096:return {}
    try:
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or value.get('status') not in {'RUNNING','TRAINING','PREPARING','COMPLETED','FAILED','TIMED_OUT','UNVERIFIABLE'}:return {}
        for field in ('started_at','duration_seconds','phase_started_monotonic'):
            number=value.get(field)
            if number is None:
                if field=='started_at' and value.get('protocol') is None:return {}
                continue
            if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or number<0:return {}
        return value
    except (OSError,ValueError):return {}


def read_training_throughput(job_dir):
    folder=Path(job_dir);path=folder/'cuda_smoke/summary.json'
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or path.stat().st_size>16384:
        raise ValueError('unsafe or oversized throughput preflight summary')
    return json.loads(path.read_text())['training_throughput']


def require_training_phases(path, target=False, loop=False):
    """Static fail-closed admission; candidate code is never imported on the host."""
    import ast
    tree=ast.parse(Path(path).read_text())
    if not any(isinstance(node,ast.FunctionDef) and node.name=='training_steps' for node in tree.body):
        raise CandidateValidationError('training-only budget cannot be verified: candidate lacks explicit training_steps interface; frozen source will not be rewritten')
    if target and not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_PROTOCOL' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value==('elapsed-training-loop-v1' if loop else 'required-training-target-v1') for n in tree.body):
        raise CandidateValidationError('candidate must explicitly implement '+('elapsed-training-loop-v1 with TrainingLoopStart' if loop else 'progress-aware required-training-target-v1')+'; frozen schedules cannot be silently changed')


def probe_candidate_interface(repo_root, module_file, good_root, runtime, candidate_id, timeout=20, cancel=None, *, diagnostic_only=False):
    """Import untrusted code only inside the no-network CPU sandbox before GPU queueing."""
    from orchestrator.sandbox import command as sandbox_command
    safe_candidate_id(candidate_id)
    probe_root = Path(runtime).resolve() / 'probes'
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_root / safe_candidate_id(candidate_id)
    probe_dir.mkdir(parents=True, exist_ok=True)
    if probe_dir.is_symlink() or not probe_dir.resolve(strict=True).is_relative_to(probe_root):
        raise PermissionError('candidate probe output escapes runtime')
    required = ('diagnose',) if diagnostic_only else ('train','load')
    script = f"""import importlib.util, traceback, json, sys
from scripts.feedback_infer import verify_state_integrity
verify_state_integrity()
try:
    s=importlib.util.spec_from_file_location('candidate','/candidate/model.py')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert all(callable(getattr(m,n,None)) for n in {required!r})
except Exception:
    traceback.print_exc()
    print('CANDIDATE_IMPORT_RESULT=FAILED')
    sys.exit(1)
print('interface import OK')
"""
    cmd = sandbox_command(repo_root, module_file, good_root, probe_dir, ['-c', script], use_gpu=False)
    try:
        from orchestrator.codex_runner import run_cancellable
        result = run_cancellable(cmd,cwd=repo_root,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
                                 text=True,timeout=timeout,cancel=cancel) if cancel else subprocess.run(
            cmd,cwd=repo_root,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},capture_output=True,text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('candidate CPU interface probe timed out') from exc
    (probe_dir/'interface_stdout.log').write_text(result.stdout)
    (probe_dir/'interface_stderr.log').write_text(result.stderr)
    if result.returncode or 'interface import OK' not in result.stdout:
        detail=(result.stdout+result.stderr)[-12000:]
        if 'CANDIDATE_IMPORT_RESULT=FAILED' in result.stdout:
            raise CandidateValidationError('candidate CPU interface probe failed: '+detail)
        raise RuntimeError('CPU interface worker failed before reporting a candidate result: '+detail)
    return True


def run_candidate_unit_tests(repo_root, module_file, good_root, runtime, candidate_id, timeout=30, cancel=None):
    """Require executable candidate CPU tests in the no-network boundary."""
    from orchestrator.sandbox import command as sandbox_command
    module = Path(module_file).resolve(strict=True)
    if not (module.parent / 'test_model.py').is_file():
        raise CandidateValidationError('candidate CPU unit tests failed: mandatory test_model.py is missing')
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
print('CANDIDATE_TEST_RESULT=' + str(code))
sys.exit(code)
'''
    cmd = sandbox_command(repo_root, module, good_root, probe_dir,
                          ['-c', test_script],
                          use_gpu=False)
    try:
        from orchestrator.codex_runner import run_cancellable
        result = run_cancellable(cmd,cwd=repo_root,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
                                 text=True,timeout=timeout,cancel=cancel) if cancel else subprocess.run(
            cmd,cwd=repo_root,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},capture_output=True,text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('candidate CPU unit tests timed out') from exc
    (probe_dir/'tests_stdout.log').write_text(result.stdout)
    (probe_dir/'tests_stderr.log').write_text(result.stderr)
    if 'CANDIDATE_TEST_RESULT=' not in result.stdout:
        raise RuntimeError('CPU test worker did not report a validation result: '+(result.stdout+result.stderr)[-8000:])
    if result.returncode:
        detail=(result.stdout+result.stderr)[-8000:]
        if 'CANDIDATE_TEST_RESULT=' in result.stdout:
            raise CandidateValidationError('candidate CPU unit tests failed: '+detail)
        raise RuntimeError('CPU test worker failed before reporting candidate tests: '+detail)
    return {'tests': 'passed', 'output': result.stdout[-300:]}


def verify_completed_learning(run, job_dir):
    """No candidate execution: verify a completed learning receipt and frozen bytes."""
    folder=Path(job_dir);config=json.loads(run['config_json'])
    expected=folder.parent/run['run_id']
    if folder.is_symlink() or folder.resolve()!=expected.resolve():raise PermissionError('unsafe learning receipt directory')
    for name in ('training_receipt.json','training_timing.json','checkpoint/model.pt','checkpoint/checkpoint_manifest.json'):
        path=folder/name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('missing or unsafe completed-learning evidence: '+name)
    if (folder/'training_receipt.json').stat().st_size>131072 or (folder/'training_timing.json').stat().st_size>16384:
        raise ValueError('oversized completed-learning evidence')
    receipt=json.loads((folder/'training_receipt.json').read_text())
    timing=json.loads((folder/'training_timing.json').read_text())
    import hashlib
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (receipt.get('version')!='completed-learning-v1' or receipt.get('phase')!='diagnostics_ready' or
        receipt.get('config_sha256')!=cfg_hash or receipt.get('model_module_sha256')!=config.get('model_sha256') or
        receipt.get('good_manifest_sha256')!=run['manifest_sha256'] or receipt.get('training_timing')!=timing):
        raise ValueError('learning receipt provenance mismatch')
    if (timing.get('status')!='COMPLETED' or timing.get('verified') is not True or
        timing.get('completion_policy')!=config.get('training_completion_policy','required-training-target-v1') or
        timing.get('duration_seconds',-1)<config.get('training_budget_seconds',float('inf'))):
        raise ValueError('learning target was not successfully completed')
    audit=receipt.get('training_audit',{})
    if (audit.get('seed')!=config['seed'] or audit.get('optimizer_updates')!=timing['completed_steps'] or
        audit.get('observed_optimizer_step_calls')!=timing['completed_steps'] or audit.get('source_exposures')!=timing['source_exposures']):
        raise ValueError('learning counters differ from verified phase audit')
    if receipt['checkpoint_sha256']!=sha256(folder/'checkpoint/model.pt') or receipt['checkpoint_manifest_sha256']!=sha256(folder/'checkpoint/checkpoint_manifest.json'):
        raise ValueError('checkpoint changed after verified learning')
    return receipt


def certify_completed_learning(store, run, folder):
    """Host-only certification after child exit; never change failed run status."""
    receipt=verify_completed_learning(run,folder)
    from orchestrator.experiment_plan import digest
    proof=Path(folder)/'host_verified_training.json'
    if proof.is_symlink():raise PermissionError('unsafe host training certification path')
    if proof.exists():
        if json.loads(proof.read_text())!=receipt:raise ValueError('host training certification changed')
    else:
        with proof.open('x') as handle:json.dump(receipt,handle,sort_keys=True)
    value={'receipt_sha256':digest(receipt),'checkpoint_sha256':receipt['checkpoint_sha256']}
    with store.connect() as con:
        prior=con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='TRAINING_CHECKPOINT_VERIFIED'",(run['run_id'],)).fetchall()
        if prior:
            if any(json.loads(p[0])!=value for p in prior):raise ValueError('durable training certification changed')
        else:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                (run['run_id'],time.time(),'TRAINING_CHECKPOINT_VERIFIED',json.dumps(value)))
    return receipt


def execute_run(run: dict, store: StateStore, repo_root: Path, manifest_path: Path, runtime: Path,
                personal_mode_authorized=False, cancel=None):
    """Never execute untrusted candidate in controller process.

    Host operator MUST apply OS/container containment to candidate subprocess;
    cwd or restricted CLI flags alone are not a permission boundary.
    """
    from benchmark.data_guard import load_manifest
    from orchestrator.budgets import enforce
    config = json.loads(run['config_json'])
    enforce(store,campaign_id=config.get('campaign_id'),runtime=runtime)
    if config.get('training_completion_policy') in {'required-training-target-v1','elapsed-training-loop-v1'}:
        from orchestrator.campaign_state import CampaignState
        from orchestrator.campaign_config import training_execution_estimate,remaining_wall
        campaign=CampaignState(store).get(config['campaign_id'])
        estimate=training_execution_estimate(store,json.loads(campaign['config_json']),source_sha256=config.get('model_sha256'),loop_timed=config.get('training_completion_policy')=='elapsed-training-loop-v1')
        if remaining_wall(campaign,time.time())<estimate['required_seconds']:
            raise ValueError('insufficient original campaign time for required training, preparation and evaluation')
    _, paths = load_manifest(manifest_path)
    if not paths:
        raise ValueError("empty good-image manifest")
    from orchestrator.experiment_plan import execution_source
    module_file = execution_source(run, store, runtime, repo_root)
    if run["candidate_id"] != "toy_normal_ae" and not personal_mode_authorized and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Non-toy candidate execution disabled until operator verifies OS-level isolation (deploy/SECURITY.md)")
    job_dir = runtime / "jobs" / run["run_id"]
    job_dir.mkdir(parents=True, exist_ok=False)
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
    timeout = (max_job_seconds(config) if config.get('plan_contract_version') in {'experiment-plan-v2.0','experiment-plan-v3.0','experiment-plan-v4.0'} else
               min(max_job_seconds(config), int(config.get("timeout_seconds", max_job_seconds(config)))))
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
        stages.append(('cuda_smoke', smoke_command, timeout if config.get('campaign_configuration_version')=='campaign-time-v1' else min(30, timeout)))
    stages.append(('pilot', command, timeout))
    started = time.monotonic()
    deadline = started + timeout  # v2 shares the original campaign deadline; legacy shares its job allowance.
    smoke_seconds = 0.0
    status = 'COMPLETED'
    failed_stage = ''
    admission_error = ''
    for stage, stage_command, stage_limit in stages:
        if cancel and cancel():
            status = 'CANCELLED'
            break
        if time.monotonic() >= deadline:
            status = 'TIMED_OUT'
            break
        prefix = 'cuda_smoke_' if stage == 'cuda_smoke' else ''
        stage_started = time.monotonic()
        observed_phase=None
        phase_deadline=None
        with (job_dir / (prefix + "stdout.log")).open("wb") as stdout, (job_dir / (prefix + "stderr.log")).open("wb") as stderr:
            process = subprocess.Popen(stage_command, cwd=repo_root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                stage_deadline = min(deadline, stage_started + stage_limit)
                while process.poll() is None:
                    if cancel and cancel():
                        status = "CANCELLED"
                        break
                    timing_file = job_dir / 'training_timing.json'
                    if stage == 'pilot' and config.get('training_completion_policy') not in {'required-training-target-v1','elapsed-training-loop-v1'} and config.get('training_budget_seconds') and timing_file.is_file() and not timing_file.is_symlink():
                        timing = read_training_timing(timing_file)
                        if timing.get('protocol')=='prepared-training-step-v1' and timing.get('status')=='TRAINING':
                            phase=timing.get('completed_steps')
                            if observed_phase!=phase:
                                # Receipt time cannot charge earlier setup/preparation.
                                observed_phase=phase
                                phase_deadline=time.monotonic()+max(0,config['training_budget_seconds']-timing.get('duration_seconds',0))
                            if time.monotonic()>=phase_deadline:
                                status='TIMED_OUT'
                                break
                        else:
                            observed_phase=None
                            phase_deadline=None
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
            if status=='COMPLETED' and config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'}:
                # Disposable smoke is measured, charged and never a scientific model.
                # Admit the entire remaining comparison before its first full target.
                from orchestrator.campaign_state import CampaignState
                from orchestrator.campaign_config import training_execution_estimate,remaining_wall,remaining_plan_work
                campaign=CampaignState(store).get(config['campaign_id'])
                try:
                    profile=read_training_throughput(job_dir)
                    estimate=training_execution_estimate(store,json.loads(campaign['config_json']),
                        source_sha256=config['model_sha256'],throughput=profile,critique=True,training_settings=config,loop_timed=config.get('training_completion_policy')=='elapsed-training-loop-v1',
                        **remaining_plan_work(store,config))
                    with store.connect() as con:
                        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                            (run['run_id'],time.time(),'TRAINING_THROUGHPUT_PREFLIGHT',json.dumps({'profile':profile,'estimate':estimate})))
                    if remaining_wall(campaign,time.time())<estimate['required_seconds']:
                        raise ValueError('complete remaining comparison cannot fit original research deadline at measured end-to-end throughput; revise the plan or preparation, never shorten required learning')
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    status='FAILED';admission_error='Recoverable planning/throughput admission: '+str(exc)
        if status != 'COMPLETED':
            failed_stage = 'throughput_admission' if admission_error else stage
            break
    elapsed = time.monotonic() - started
    if status == "FAILED":
        error_log = job_dir / ('cuda_smoke_stderr.log' if failed_stage == 'cuda_smoke' else 'stderr.log')
        err_tail = error_log.read_bytes()[-20000:].lower() if error_log.is_file() else b''
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
                if config.get('training_budget_seconds') is not None and config.get('execution_kind')!='frozen':
                    timing=read_training_timing(job_dir/'training_timing.json')
                    if timing.get('protocol')!='prepared-training-step-v1' or timing.get('verified') is not True or timing.get('status')!='COMPLETED':
                        raise ValueError('training-only budget cannot be verified from completed phase audit')
                    if metrics.get('training/duration_seconds')!=timing['duration_seconds'] or ((timing['duration_seconds']<config['training_budget_seconds'] or timing.get('completion_policy')!=config.get('training_completion_policy','required-training-target-v1') or timing['duration_seconds']-config['training_budget_seconds']>timing.get('last_iteration_seconds' if config.get('training_completion_policy')=='elapsed-training-loop-v1' else 'last_step_seconds',0)+1e-6) if config.get('training_completion_policy') in {'required-training-target-v1','elapsed-training-loop-v1'} else timing['duration_seconds']>config['training_budget_seconds']):
                        raise ValueError('training-only timing metric or deadline mismatch')
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
                duration=timing.get('duration_seconds')
                if duration is not None:
                    partial={'training/duration_seconds':duration,'training/timing_verified':float(timing.get('verified',False))}
                    if config.get('training_completion_policy')=='elapsed-training-loop-v1':
                        partial.update({'training/loop_elapsed_seconds':timing.get('loop_elapsed_seconds',duration),
                            'training/learning_seconds':timing.get('learning_seconds',0),
                            'training/loop_overhead_seconds':max(0,duration-timing.get('learning_seconds',0))})
                    store.set_run_result(run['run_id'], partial, gpu_seconds=charged, artifact_dir=str(job_dir))
            if failed_stage == 'throughput_admission':
                store.set_run_result(run['run_id'], {'validation/cuda_smoke_passed':1,
                    'validation/cuda_smoke_seconds':smoke_seconds,'validation/throughput_admitted':0},
                    gpu_seconds=charged,artifact_dir=str(job_dir))
            elif failed_stage == 'cuda_smoke':
                store.set_run_result(run['run_id'], {'validation/cuda_smoke_passed': 0,
                    'validation/cuda_smoke_seconds': smoke_seconds}, gpu_seconds=charged, artifact_dir=str(job_dir))
            else:
                store.account_gpu_usage(run['run_id'], charged)
            reason = f"Candidate process {status}; inspect stderr.log"
            if failed_stage == 'cuda_smoke':
                log = job_dir / 'cuda_smoke_stderr.log'
                reason = f"CUDA smoke validation {status}:\n{log.read_text(errors='replace')[-8000:]}"
            elif (job_dir / 'stderr.log').is_file():
                reason += '\n' + (job_dir / 'stderr.log').read_text(errors='replace')[-8000:]
            if admission_error:reason=admission_error
            if config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'} and (job_dir/'training_receipt.json').is_file():
                try:
                    certify_completed_learning(store,run,job_dir)
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    reason+='\nFrozen-checkpoint reuse rejected: '+str(exc)
            store.transition(run["run_id"], status, reason=reason, expect="RUNNING")
        enqueue_run_tracking(store, run["run_id"], artifact_dir=job_dir)
        return status
    except Exception as exc:
        if store.get_run(run["run_id"])["status"] == "RUNNING":
            store.account_gpu_usage(run["run_id"], elapsed if config.get("device", "cpu").startswith("cuda") else 0)
            store.transition(run["run_id"], "FAILED", reason=f"runner post-processing: {exc}", expect="RUNNING")
        raise
