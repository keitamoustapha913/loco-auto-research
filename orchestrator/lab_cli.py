"""Operator entry point for local-only laboratory services and bounded smoke runs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil

from orchestrator.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]
KEYS = {'LOCO_RUNTIME', 'LOCO_LEDGER', 'LOCO_GOOD_ROOT', 'LOCO_SEALED_ROOT',
        'MLFLOW_TRACKING_URI', 'LOCO_CAMPAIGN', 'LOCO_GPU_BUDGET_HOURS',
        'LOCO_JOB_TIMEOUT_SECONDS', 'LOCO_CODEX_MAX_CALLS', 'LOCO_SANDBOX_VERIFIED',
        'LOCO_ALLOW_UNATTENDED'}
EXPANSION = re.compile(r'\$(?:\{([A-Z][A-Z0-9_]*)\}|([A-Z][A-Z0-9_]*))')
SERVICES = ('mlflow', 'dashboard', 'tracking')


def load_settings(env_file=None):
    """Parse literal exports and simple variable references; never source shell code."""
    path = Path(env_file or os.getenv('LOCO_ENV_FILE', REPO.parent / 'loco-research.env')).expanduser()
    values = {'HOME': str(Path.home())}
    if path.is_file():
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            match = re.fullmatch(r'(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)', line)
            if not match or match[1] not in KEYS:
                raise ValueError(f'Unsafe or unknown configuration at {path}:{number}')
            key, value = match.groups()
            value = value.strip()
            if len(value) > 1 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            def replace(ref):
                name = ref[1] or ref[2]
                if name not in values and name not in os.environ:
                    raise ValueError(f'Undefined environment variable {name} at {path}:{number}')
                return os.environ.get(name, values.get(name, ''))
            value = EXPANSION.sub(replace, value)
            if '$' in value or '`' in value or '\n' in value:
                raise ValueError(f'Unsafe configuration value at {path}:{number}')
            values[key] = value
            os.environ.setdefault(key, value)
    os.environ.setdefault('LOCO_RUNTIME', str(REPO / '.runtime'))
    os.environ.setdefault('LOCO_LEDGER', str(Path(os.environ['LOCO_RUNTIME']) / 'controller' / 'experiments.sqlite'))
    os.environ.setdefault('LOCO_GOOD_ROOT', str(Path(os.environ['LOCO_RUNTIME']) / 'good_data'))
    os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://127.0.0.1:56000')
    if os.environ['MLFLOW_TRACKING_URI'].rstrip('/') != 'http://127.0.0.1:56000':
        raise ValueError('MLFLOW_TRACKING_URI must use the local MLflow service at http://127.0.0.1:56000')
    runtime = Path(os.environ['LOCO_RUNTIME']).expanduser().resolve()
    ledger = Path(os.environ['LOCO_LEDGER']).expanduser().resolve()
    if not ledger.is_relative_to(runtime):
        raise ValueError('LOCO_LEDGER must reside within LOCO_RUNTIME')
    return runtime, ledger


@contextmanager
def lifecycle_lock(runtime):
    root = runtime / 'controller'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'lab.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def state_path(runtime, name):
    return runtime / 'controller' / f'lab-{name}.json'


def audit(runtime, event, **fields):
    path = runtime / 'logs' / 'lifecycle.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps({'time': time.time(), 'event': event, **fields}, sort_keys=True) + '\n')


def owned_process(runtime, name):
    path = state_path(runtime, name)
    try:
        record = json.loads(path.read_text())
        process = psutil.Process(int(record['pid']))
        if abs(process.create_time() - float(record['created'])) > 1 or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        command = process.cmdline()
        if not command or record['signature'] not in ' '.join(command):
            return None
        return process
    except (OSError, ValueError, KeyError, json.JSONDecodeError, psutil.Error):
        return None


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.35)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def http_healthy(port, path):
    try:
        with urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def service_healthy(name):
    if name == 'mlflow':
        return http_healthy(56000, '/health')
    if name == 'dashboard':
        return http_healthy(8501, '/_stcore/health')
    return False


def command_for(name, runtime):
    if name == 'mlflow':
        return [sys.executable, '-m', 'mlflow', 'server', '--backend-store-uri',
                f'sqlite:///{runtime / "mlflow" / "mlflow.db"}', '--artifacts-destination',
                str(runtime / 'mlflow' / 'artifacts'), '--host', '127.0.0.1', '--port', '56000', '--workers', '1']
    if name == 'dashboard':
        return [sys.executable, '-m', 'streamlit', 'run', str(REPO / 'dashboard' / 'app.py'),
                '--server.address', '127.0.0.1', '--server.port', '8501',
                '--server.headless', 'true', '--browser.gatherUsageStats', 'false']
    return [sys.executable, '-m', 'orchestrator.tracking_worker']


def spawn_service(runtime, name, timeout=35):
    existing = owned_process(runtime, name)
    if existing:
        if name == 'tracking' or service_healthy(name):
            return False
        stop_service(runtime, name)
    if name in ('mlflow', 'dashboard') and port_open(56000 if name == 'mlflow' else 8501):
        raise RuntimeError(f'{name} port is occupied by an unmanaged process; refusing to replace it')
    cmd = command_for(name, runtime)
    log_path = runtime / 'logs' / f'{name}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        process = subprocess.Popen(cmd, cwd=REPO, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'created': psutil.Process(process.pid).create_time(),
              'signature': 'orchestrator.tracking_worker' if name == 'tracking' else
                           ('mlflow' if name == 'mlflow' else str(REPO / 'dashboard' / 'app.py')),
              'command': cmd, 'started_at': time.time()}
    state_path(runtime, name).write_text(json.dumps(record))
    audit(runtime, 'service_spawn', service=name, pid=process.pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            audit(runtime, 'service_start_failed', service=name, reason='exited')
            state_path(runtime, name).unlink(missing_ok=True)
            raise RuntimeError(f'{name} exited during startup; inspect {log_path}')
        if name == 'tracking' or service_healthy(name):
            return True
        time.sleep(0.3)
    stop_service(runtime, name)
    audit(runtime, 'service_start_failed', service=name, reason='health_timeout')
    raise TimeoutError(f'{name} health check timed out; inspect {log_path}')


def stop_service(runtime, name):
    process = owned_process(runtime, name)
    if process:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8)
        except psutil.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        audit(runtime, 'service_stop', service=name, pid=process.pid)
    state_path(runtime, name).unlink(missing_ok=True)


def start(runtime, ledger):
    for module in ('mlflow', 'streamlit', 'torch'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f'Missing dependency {module}; run uv sync --extra all')
    good_root = Path(os.environ['LOCO_GOOD_ROOT']).expanduser().resolve()
    if not good_root.is_dir():
        raise FileNotFoundError(f'Authorized good-image directory missing: {good_root}')
    manifest = runtime / 'good_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Authorized good-only manifest missing: {manifest}')
    from benchmark.data_guard import load_manifest
    data, _ = load_manifest(manifest, verify_hashes=False)
    if Path(data['root']).resolve() != good_root:
        raise ValueError('Manifest root differs from LOCO_GOOD_ROOT')
    for directory in (runtime / 'jobs', runtime / 'logs', runtime / 'mlflow' / 'artifacts'):
        directory.mkdir(parents=True, exist_ok=True)
    store = StateStore(ledger)
    from orchestrator.gpu_queue import gpu_lock, GpuBusy
    try:
        with gpu_lock(runtime):
            interrupted = store.reconcile_after_restart()
            if interrupted:
                audit(runtime, 'orphan_runs_reconciled', run_ids=interrupted)
    except GpuBusy:
        audit(runtime, 'gpu_worker_active_during_start')
    started = []
    try:
        for name in SERVICES:
            if spawn_service(runtime, name):
                started.append(name)
    except Exception:
        for name in reversed(started):
            stop_service(runtime, name)
        raise
    # Resume only a previously authorized RUNNING campaign after a controller crash.
    # Explicitly STOPPED and PAUSED campaigns never start work here.
    try:
        from orchestrator.campaign_state import CampaignState
        active_campaigns = CampaignState(store)
        row = active_campaigns.latest()
        if row and row['status'] == 'RUNNING' and not owned_process(runtime, 'research'):
            from orchestrator.autonomy import preflight
            preflight(json.loads(row['config_json']), runtime)
            spawn_research(runtime, active_campaigns, row['id'])
    except Exception as exc:
        audit(runtime, 'research_recovery_failed', error=str(exc))
    return status(runtime, ledger)


def status(runtime, ledger):
    services = {}
    for name in SERVICES:
        owned = owned_process(runtime, name)
        services[name] = {'pid': owned.pid if owned else None,
                          'healthy': bool(owned and (name == 'tracking' or service_healthy(name)))}
    result = {'repository': str(REPO), 'runtime': str(runtime), 'ledger': str(ledger),
              'ledger_connected': False, 'services': services,
              'gpu_worker': 'Idle', 'autonomy': os.getenv('LOCO_ALLOW_UNATTENDED') == '1' and os.getenv('LOCO_SANDBOX_VERIFIED') == '1'}
    if ledger.is_file():
        try:
            store = StateStore(ledger, read_only=True)
            result.update(ledger_connected=True, campaign=os.getenv('LOCO_CAMPAIGN', ''),
                          runs=len(store.list_runs(5000)), pending_tracking=len(store.pending_tracking(10000)),
                          gpu_hours=store.total_gpu_seconds()/3600)
            if any(r['status'] == 'RUNNING' for r in store.list_runs(5000)):
                result['gpu_worker'] = 'Running or interrupted; inspect ledger'
            with store.connect() as con:
                try:
                    campaign_row = con.execute("SELECT id,name,status,reason,active_run_id FROM campaigns ORDER BY created_at DESC LIMIT 1").fetchone()
                except Exception:
                    campaign_row = None
            if campaign_row:
                result['research_campaign'] = dict(campaign_row)
                result['research_worker_running'] = bool(owned_process(runtime, 'research'))
                result['autonomy'] = campaign_row['status'] == 'RUNNING' and result['research_worker_running']
        except Exception as exc:
            result['ledger_error'] = str(exc)
    try:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        result['gpu_name'] = torch.cuda.get_device_name(0) if result['cuda_available'] else None
    except Exception as exc:
        result['gpu_error'] = str(exc)
    manifest = runtime / 'good_manifest.json'
    if manifest.is_file():
        try:
            from benchmark.data_guard import load_manifest
            data, _ = load_manifest(manifest, verify_hashes=False)
            result['dataset'] = {'categories': sorted({i['category'] for i in data['images']}), 'normal_images': len(data['images'])}
        except Exception as exc:
            result['dataset_error'] = str(exc)
    return result


def print_status(result):
    print('LOCO AUTONOMOUS RESEARCH LAB\n')
    print(f"Repository: {result['repository']}")
    print(f"Runtime: {result['runtime']}")
    print(f"GPU: {result.get('gpu_name') or 'Unavailable'}")
    print(f"Dataset: {result.get('dataset', 'Unavailable')}")
    print(f"Experiment ledger: {'Connected' if result['ledger_connected'] else 'Unavailable'}")
    for name, label in (('mlflow', 'MLflow'), ('dashboard', 'Dashboard'), ('tracking', 'Tracking synchronization')):
        item = result['services'][name]
        print(f"{label}: {'Running' if item['healthy'] else 'Stopped or unhealthy'}" + (f" (PID {item['pid']})" if item['pid'] else ''))
    print(f"GPU worker: {result['gpu_worker']}")
    print(f"Research autonomy: {'Enabled' if result['autonomy'] else 'Disabled'}")
    if result.get('research_campaign'):
        research = result['research_campaign']
        print(f"Campaign: {research['name']} ({research['status']}); worker {'running' if result.get('research_worker_running') else 'stopped'}")
        if result['autonomy']:
            print('Isolation mode: personal Bubblewrap; Codex and candidate host processes share the operator UID')
    if not result['autonomy']:
        print('Reason: ' + (result.get('research_campaign', {}).get('reason') or 'No authorized campaign worker is running'))
    print('Dashboard: http://127.0.0.1:8501')
    print('MLflow: http://127.0.0.1:56000')
    print(f"Pending tracking events: {result.get('pending_tracking', 'unknown')}")


def smoke(runtime, ledger, dataset, device):
    if dataset != 'juice_bottle':
        raise ValueError('Only the authorized juice_bottle smoke category is supported')
    if device != 'cuda':
        raise ValueError('This command is a CUDA infrastructure smoke test')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    from benchmark.data_guard import load_manifest
    manifest_path = runtime / 'good_manifest.json'
    manifest, _ = load_manifest(manifest_path)
    if Path(manifest['root']).resolve() != Path(os.environ['LOCO_GOOD_ROOT']).resolve():
        raise ValueError('Manifest root differs from authorized good root')
    if not any(e['category'] == dataset for e in manifest['images']):
        raise ValueError('Category absent from authorized manifest')
    store = StateStore(ledger)
    from orchestrator.budgets import enforce
    from orchestrator.controller import queue_candidate, run_worker_once
    seconds = min(300, int(os.getenv('LOCO_JOB_TIMEOUT_SECONDS', '1800')))
    enforce(store, next_gpu_seconds=seconds, runtime=runtime)
    # The existing worker lock is the single-host concurrency boundary.
    from orchestrator.gpu_queue import gpu_lock, claim_next
    from orchestrator.experiment_runner import execute_run
    with gpu_lock(runtime):
        if any(r['status'] in ('RUNNING', 'QUEUED') for r in store.list_runs(5000)):
            raise RuntimeError('Existing running or queued work; smoke will not execute another job')
        run_id = queue_candidate(store, 'infrastructure-smoke', 'toy_normal_ae', dataset,
                                 'smoke', device, manifest_path, seconds)
        run = claim_next(store, max_duration=seconds)
        try:
            outcome = execute_run(run, store, REPO, manifest_path, runtime)
        except Exception as exc:
            if store.get_run(run_id)['status'] == 'RUNNING':
                store.transition(run_id, 'FAILED', reason=str(exc), expect='RUNNING')
                from orchestrator.mlflow_tracker import enqueue_run_tracking
                enqueue_run_tracking(store, run_id)
            raise
    from orchestrator.mlflow_tracker import sync_outbox
    try:
        tracking = sync_outbox(store)
    except Exception as exc:
        tracking = {'queued': True, 'error': str(exc)}
    return {'run_id': run_id, 'status': outcome, 'metrics_kind': 'synthetic development proxy; infrastructure smoke only',
            'gpu_seconds': store.get_run(run_id)['gpu_seconds'], 'tracking': tracking}


def research_command(action, runtime, ledger, config_path=None):
    from orchestrator.campaign_state import CampaignState, TERMINAL
    from orchestrator.autonomy import load_campaign_config, preflight
    if action in {'status', 'report'} and not ledger.is_file():
        return {'campaign': None, 'worker_running': False, 'message': 'No experiment ledger initialized'}
    if action == 'start':
        config = load_campaign_config(config_path)
        preflight(config, runtime)
    store = StateStore(ledger, read_only=action in {'status', 'report'} and ledger.is_file())
    campaigns = CampaignState(store)
    if action in {'status', 'report'}:
        return research_summary(campaigns, campaigns.latest(), owned_process(runtime, 'research'),
                                report=action == 'report')
    with lifecycle_lock(runtime):
        row = campaigns.latest()
        process = owned_process(runtime, 'research')
        if action == 'start':
            if row and row['status'] in {'RUNNING', 'PAUSED', 'STOPPING'}:
                if process:
                    return research_summary(campaigns, row, process)
                if row['status'] == 'STOPPING':
                    campaigns.set_status(row['id'], 'STOPPED', 'worker absent after stop')
                else:
                    # Resume the same ledger campaign; the worker reconciles any interrupted run.
                    return spawn_research(runtime, campaigns, row['id'])
            campaign_id = campaigns.create(config)
            return spawn_research(runtime, campaigns, campaign_id)
        if not row or row['status'] in TERMINAL:
            raise RuntimeError('no active research campaign')
        if action == 'pause':
            if row['status'] == 'RUNNING':
                campaigns.set_status(row['id'], 'PAUSED', 'operator pause')
        elif action == 'resume':
            if row['status'] == 'PAUSED':
                campaigns.set_status(row['id'], 'RUNNING', 'operator resume')
                if not process:
                    return spawn_research(runtime, campaigns, row['id'])
        elif action == 'stop':
            campaigns.set_status(row['id'], 'STOPPING', 'operator stop requested')
            if not process:
                campaigns.set_status(row['id'], 'STOPPED', 'worker absent; queued work will not run')
            else:
                deadline = time.monotonic() + 15
                while owned_process(runtime, 'research') and time.monotonic() < deadline:
                    time.sleep(0.2)
                if owned_process(runtime, 'research'):
                    stop_service(runtime, 'research')
                    campaigns.set_status(row['id'], 'STOPPED', 'operator cancelled active worker')
            from orchestrator.autonomy import reconcile_unfinished_runs
            reconcile_unfinished_runs(store, campaigns, row['id'], 'operator stopped campaign')
        return research_summary(campaigns, campaigns.get(row['id']), owned_process(runtime, 'research'))


def spawn_research(runtime, campaigns, campaign_id):
    if owned_process(runtime, 'research'):
        raise RuntimeError('research worker already active')
    command = [sys.executable, '-m', 'orchestrator.autonomy_worker', '--campaign-id', campaign_id]
    log_path = runtime / 'logs' / 'research.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state_path(runtime, 'research').write_text(json.dumps({'pid': process.pid,
        'created': psutil.Process(process.pid).create_time(),
        'signature': 'orchestrator.autonomy_worker', 'command': command,
        'started_at': time.time()}))
    audit(runtime, 'research_spawn', campaign_id=campaign_id, pid=process.pid)
    time.sleep(0.5)
    if process.poll() is not None:
        campaigns.set_status(campaign_id, 'FAILED', 'campaign worker exited during startup')
        raise RuntimeError(f'research worker exited; inspect {log_path}')
    return research_summary(campaigns, campaigns.get(campaign_id), owned_process(runtime, 'research'))


def research_summary(campaigns, row, process, report=False):
    if not row:
        return {'campaign': None, 'worker_running': False, 'message': 'No autonomous campaign recorded'}
    usage = campaigns.usage(row['id'])
    result = {'campaign_id': row['id'], 'name': row['name'], 'status': row['status'],
              'worker_running': bool(process), 'usage': usage,
              'current_hypothesis': row['current_hypothesis'],
              'next_hypothesis': row['next_hypothesis'], 'active_run_id': row['active_run_id'],
              'reason': row['reason'], 'consecutive_failures': row['consecutive_failures']}
    result['codex_budget'] = campaigns.call_budget(row['id'])
    with campaigns.store.connect() as con:
        plan_result = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_RESULT' ORDER BY id DESC LIMIT 1", (row['id'],)).fetchone()
    if plan_result:
        plan = json.loads(plan_result[0])['result']
        result['experiment_plan'] = plan if report else {key:plan[key] for key in
            ('plan_id','status','expected_jobs','recorded_jobs','problems')}
    if report:
        result['limits'] = json.loads(row['config_json'])['limits']
        result['runs'] = [{**{k: r[k] for k in ('run_id', 'hypothesis_id', 'candidate_id', 'status',
                                               'gpu_seconds', 'error', 'mlflow_run_id')},
                           'development_metrics': json.loads(r['metrics_json'])}
                          for r in campaigns.runs(row['id'])]
        from orchestrator.development_feedback import latest_summary
        for run in result['runs']:
            run['real_development_feedback'] = latest_summary(campaigns.store, run['run_id'])
        result['events'] = campaigns.events(row['id'], 30)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loco-lab')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'stop', 'restart', 'status', 'doctor', 'logs'):
        commands.add_parser(name)
    smoke_parser = commands.add_parser('smoke')
    smoke_parser.add_argument('--dataset', required=True)
    smoke_parser.add_argument('--device', choices=['cuda'], required=True)
    research_parser = commands.add_parser('research')
    research_sub = research_parser.add_subparsers(dest='research_action', required=True)
    for name in ('start', 'status', 'pause', 'resume', 'stop', 'report'):
        sub = research_sub.add_parser(name)
        if name == 'start':
            sub.add_argument('--config', help='operator-owned bounded autonomy YAML')
    feedback_parser = commands.add_parser('feedback')
    feedback_sub = feedback_parser.add_subparsers(dest='feedback_action', required=True)
    feedback_sub.add_parser('import')
    feedback_sub.add_parser('status')
    sub = feedback_sub.add_parser('evaluate')
    sub.add_argument('--run-id', required=True)
    sub.add_argument('--timeout-seconds', type=int, default=600)
    sub.add_argument('--accept-shared-uid-risk', action='store_true')
    args = parser.parse_args(argv)
    try:
        runtime, ledger = load_settings()
        if args.command == 'doctor':
            from scripts.doctor import diagnose
            print(json.dumps(diagnose(), indent=2, default=str))
        elif args.command == 'status':
            print_status(status(runtime, ledger))
        elif args.command == 'logs':
            lifecycle = runtime / 'logs' / 'lifecycle.jsonl'
            if lifecycle.is_file():
                print(f'[lifecycle] {lifecycle}')
                print('\n'.join(lifecycle.read_text(errors='replace').splitlines()[-30:]))
            for name in SERVICES:
                path = runtime / 'logs' / f'{name}.jsonl'
                print(f'[{name}] {path}')
                if path.is_file():
                    print('\n'.join(path.read_text(errors='replace').splitlines()[-30:]))
        elif args.command == 'smoke':
            outcome = smoke(runtime, ledger, args.dataset, args.device)
            print(json.dumps(outcome, indent=2))
            return 0 if outcome['status'] == 'COMPLETED' else 1
        elif args.command == 'feedback':
            from benchmark.feedback_data import import_development, verify_import, imported_root
            if args.feedback_action == 'import':
                report = import_development(runtime)
                print(json.dumps({'split_id': report['split_id'], 'counts': report['summary']}, indent=2))
            elif args.feedback_action == 'status':
                report = verify_import(imported_root(runtime))
                print(json.dumps({'split_id': report['split_id'], 'counts': report['summary']}, indent=2))
            else:
                from orchestrator.development_feedback import evaluate_run
                print(json.dumps(evaluate_run(runtime, StateStore(ledger), args.run_id,
                    timeout=args.timeout_seconds, personal_mode_authorized=args.accept_shared_uid_risk), indent=2))
        elif args.command == 'research':
            print(json.dumps(research_command(args.research_action, runtime, ledger,
                                              getattr(args, 'config', None)), indent=2, default=str))
        else:
            if args.command == 'stop' and ledger.is_file():
                from orchestrator.campaign_state import CampaignState
                latest = CampaignState(StateStore(ledger)).latest()
                if latest and latest['status'] in {'RUNNING', 'PAUSED', 'STOPPING'}:
                    research_command('stop', runtime, ledger)
            with lifecycle_lock(runtime):
                if args.command in ('stop', 'restart'):
                    for name in reversed(SERVICES):
                        stop_service(runtime, name)
                if args.command in ('start', 'restart'):
                    result = start(runtime, ledger)
                    print_status(result)
                    print('\nLaboratory started successfully.')
                else:
                    print('Laboratory stopped.')
        return 0
    except Exception as exc:
        print(f'loco-lab: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
