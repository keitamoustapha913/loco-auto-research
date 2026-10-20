"""Immutable scientific plans in the existing ledger; no agent-authored commands.

COMPLETE means the requested work was measured, never that a hypothesis is true.
Historical runs and their metrics are never reclassified or rewritten here.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path

from jsonschema import validate

REPO = Path(__file__).resolve().parents[1]
VERSION = 'experiment-plan-v1.0'
RESERVED = {'device', 'seed', 'category', 'timeout_seconds', 'model_sha256', 'campaign_id',
            'campaign_name', 'execution_mode', 'metric_evaluator_version', 'synthetic_evaluation',
            'cuda_smoke_required', 'inference', 'diagnostics', 'frozen_baseline', 'execution_kind', 'diagnostic_provider',
            'trusted_source_sha256','data_pin','baseline_effective_configuration','training_budget_seconds','campaign_configuration_version'}
REQUIRED_TRAINING_POSITIVE = ('steps', 'batch_size', 'size', 'expected_source_exposures')
MATCHED_TRAINING_CONTROLS = REQUIRED_TRAINING_POSITIVE + ('expected_optimizer_updates',)


def canonicalize_diagnostic_measurements(plan):
    """Only normalize declared diagnostic aliases; preserve science and input bytes."""
    from benchmark.train_adapter import (DIAGNOSTIC_PREFIX,diagnostic_measurement_catalog,
                                          canonical_diagnostic_measurement)
    plan=copy.deepcopy(plan)
    catalog=diagnostic_measurement_catalog(plan['diagnostics'])
    seen=set()
    for request in plan['required_measurements']:
        source,key=request['source'],request['key']
        if source=='diagnostic':
            key=canonical_diagnostic_measurement(catalog,key)
            request['key']=key
            storage=catalog[key]
        elif source=='run' and key.startswith(DIAGNOSTIC_PREFIX):
            # Existing explicit ledger references are valid only when declared.
            canonical=canonical_diagnostic_measurement(catalog,key)
            storage=catalog[canonical]
            if storage!=key:raise ValueError('ambiguous run diagnostic storage alias: '+key)
        else:
            if source!='run' and key.startswith(DIAGNOSTIC_PREFIX):
                raise ValueError('diagnostic storage key conflicts with measurement source: '+source)
            storage=key
        identity=('real' if source=='real' else 'run',storage)
        if identity in seen:raise ValueError('duplicate/conflicting measurement after diagnostic canonicalization: '+storage)
        seen.add(identity)
    for comparison in plan['paired_comparisons']:
        canonical=f"{comparison['diagnostic_id']}/{comparison['metric']}"
        if canonical not in catalog:raise ValueError('paired comparison requests an undeclared diagnostic metric: '+canonical)
    return plan


def decode_agent_plan(plan):
    """Closed output-schema settings become ordinary canonical JSON dictionaries.

    Key/value_json arrays permit arbitrary scientific parameters without open
    object schemas in the CLI's constrained structured response.
    """
    plan = copy.deepcopy(plan)
    def settings(value):
        if isinstance(value,dict): return value
        if not isinstance(value,list): raise ValueError('settings must be JSON dictionaries or key/value_json arrays')
        result = {}
        for pair in value:
            if pair['name'] in result: raise ValueError('duplicate scientific setting')
            result[pair['name']] = json.loads(pair['value_json'])
        return result
    plan['training'] = settings(plan['training'])
    for arm in plan['arms']:
        if arm.get('fixed_checkpoint_seed') is None:arm.pop('fixed_checkpoint_seed',None)
        for key in ('training_overrides','inference'): arm[key] = settings(arm[key])
        if isinstance(arm['baseline_by_seed'],list):
            values = arm['baseline_by_seed']
            arm['baseline_by_seed'] = {str(b['seed']):b['baseline_id'] for b in values}
            if len(arm['baseline_by_seed'])!=len(values):raise ValueError('duplicate frozen baseline seed')
    for diagnostic in plan['diagnostics']: diagnostic['config'] = settings(diagnostic['config'])
    return canonicalize_diagnostic_measurements(plan)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_settings(plan, arm):
    return dict(plan['training'], **arm['training_overrides'])


def training_seed(arm, pair_seed):
    return pair_seed if arm.get('fixed_checkpoint_seed') is None else arm['fixed_checkpoint_seed']


def validate_plan(plan, hypothesis_id=None):
    validate(plan, json.loads((REPO / 'schemas/experiment_plan.schema.json').read_text()))
    digest(plan)  # Reject NaN/infinity even inside extensible settings.
    plan=canonicalize_diagnostic_measurements(plan)
    from orchestrator.planning_contract import local_issues, PlanningValidationError
    issues=local_issues(plan)
    if issues:raise PlanningValidationError(issues)
    if hypothesis_id is not None and plan['hypothesis_id'] != hypothesis_id:
        raise ValueError('plan hypothesis identity differs from its proposal')
    ids = [a['arm_id'] for a in plan['arms']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate experiment arm')
    if plan['baseline_arm_id'] not in ids:
        raise ValueError('comparison needs an explicit existing baseline arm identity')
    if plan['kind'] == 'controlled_comparison' and len(ids) < 2:
        raise ValueError('controlled comparison needs at least two arms')
    if plan['kind'] == 'controlled_comparison' and not plan['controlled_variables']:
        raise ValueError('controlled comparison must declare controlled variables')
    if len(plan['arms']) * len(plan['seeds']) > plan['resource_budget']['max_jobs']:
        raise ValueError('arm/seed matrix exceeds plan job allowance')
    baselines = {b['baseline_id']: b for b in plan['baselines']}
    if len(baselines) != len(plan['baselines']):
        raise ValueError('duplicate frozen baseline identity')
    seen = set()
    controlled = {}
    for arm in plan['arms']:
        settings = training_settings(plan, arm)
        if set(settings) & RESERVED or any(k.startswith('plan_') for k in settings):
            raise ValueError('training settings cannot override controller or security fields')
        if arm['execution'] == 'train':
            if arm.get('fixed_checkpoint_seed') is not None:raise ValueError('training arm cannot pretend to be a fixed checkpoint')
            if arm['source_arm_id'] or arm['baseline_by_seed']:
                raise ValueError('training arm cannot silently load a baseline')
            for name in REQUIRED_TRAINING_POSITIVE:
                if isinstance(settings.get(name), bool) or not isinstance(settings.get(name), int) or settings[name] < 1:
                    raise ValueError(f'explicit positive training {name} required')
            updates = settings.get('expected_optimizer_updates')
            if isinstance(updates,bool) or not isinstance(updates,int) or updates < 0:
                raise ValueError('explicit nonnegative expected_optimizer_updates required')
            for key in plan['controlled_variables']:
                if key not in settings:
                    raise ValueError(f'controlled variable {key} is missing')
                if key in controlled and controlled[key] != settings[key]:
                    raise ValueError(f'controlled variable {key} differs between arms')
                controlled[key] = settings[key]
        else:
            if arm.get('fixed_checkpoint_seed') is not None and not arm['baseline_by_seed']:
                raise ValueError('fixed incumbent requires exact historical baseline references')
            if arm['training_overrides']:
                raise ValueError('frozen arm cannot change training settings')
            if bool(arm['source_arm_id']) == bool(arm['baseline_by_seed']):
                raise ValueError('frozen arm needs exactly one exact source')
            if arm['source_arm_id'] and arm['source_arm_id'] not in seen:
                raise ValueError('frozen source arm must precede its dependent arm')
            if arm['baseline_by_seed']:
                if set(arm['baseline_by_seed']) != {str(s) for s in plan['seeds']}:
                    raise ValueError('each requested seed needs its own pinned baseline')
                for seed, bid in arm['baseline_by_seed'].items():
                    if bid not in baselines or baselines[bid]['seed'] != training_seed(arm,int(seed)):
                        raise ValueError('frozen baseline seed mismatch')
        seen.add(arm['arm_id'])
    diagnostic_ids = [d['diagnostic_id'] for d in plan['diagnostics']]
    if len(diagnostic_ids) != len(set(diagnostic_ids)):
        raise ValueError('duplicate diagnostic identity')
    comparisons = plan['paired_comparisons']
    if len({c['comparison_id'] for c in comparisons}) != len(comparisons):
        raise ValueError('duplicate paired comparison identity')
    for comparison in comparisons:
        if (comparison['left_arm'] not in ids or comparison['right_arm'] not in ids or
            comparison['left_arm'] == comparison['right_arm'] or comparison['diagnostic_id'] not in diagnostic_ids):
            raise ValueError('paired comparison references an absent arm or diagnostic')
    for measurement in plan['required_measurements']:
        if measurement['source'] == 'real' and not plan['evaluation']['real_development_feedback']:
            raise ValueError('required real measurement has evaluation disabled')
        if measurement['source'] == 'run' and measurement['key'].startswith('dev_synthetic/') and not plan['evaluation']['synthetic_diagnostics']:
            raise ValueError('required synthetic measurement has evaluation disabled')
        if measurement['source'] == 'diagnostic':
            parts = measurement['key'].split('/', 1)
            if len(parts) != 2 or parts[0] not in diagnostic_ids:
                raise ValueError('required diagnostic has no executable declaration')
    return plan


def cells(plan):
    return [(arm, seed) for arm in plan['arms'] for seed in plan['seeds']]


def cell_run_id(campaign_id, plan, arm_id, seed):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'loco:{campaign_id}:{digest(plan)}:{arm_id}:{seed}'))


def register_plan(store, campaigns, campaign_id, plan):
    plan=canonicalize_diagnostic_measurements(plan)
    def existing(con):
        rows = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_REGISTERED'",
                           (campaign_id,)).fetchall()
        for row in rows:
            prior = json.loads(row['payload_json'])
            if prior['plan']['plan_id'] == plan['plan_id']:
                if prior['plan_sha256'] != digest(plan):
                    raise ValueError('registered scientific protocol is immutable')
                return prior
    # Historical publications remain readable under their original policy.
    # execute_plan revalidates before any new work, after checking publication.
    with store.connect() as con:
        prior=existing(con)
    if prior:return prior
    from orchestrator.planning_contract import admit_plan
    from orchestrator import controller
    admit_plan(plan,store,controller.REPO)
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        prior=existing(con)
        if prior:return prior
        from orchestrator.model_registry import plan_context
        record = {'plan': plan, 'plan_sha256': digest(plan), 'registered_at': time.time(), 'version': VERSION,
                  'promotion_context': plan_context(store)}
        con.execute('INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (campaign_id, record['registered_at'], 'EXPERIMENT_PLAN_REGISTERED', json.dumps(record, sort_keys=True)))
    return record


def baseline_reference(store, run_id, repo_root=REPO):
    """Resolve a completed run by immutable commit and bytes, never by its name."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise PermissionError('invalid frozen baseline run identifier')
    row = store.get_run(run_id)
    if not row or row['status'] not in {'COMPLETED', 'REVIEWED', 'PROMOTED', 'REJECTED'}:
        raise ValueError('baseline must identify an existing completed frozen run')
    cfg = json.loads(row['config_json'])
    checkpoint = Path(row['artifact_dir']) / 'checkpoint/model.pt'
    metadata = checkpoint.parent/'checkpoint_manifest.json'
    runtime = store.path.parent.parent if store.path.parent.name=='controller' else store.path.parent
    expected = runtime/'jobs'/run_id
    if (Path(row['artifact_dir']).is_symlink() or Path(row['artifact_dir']).resolve()!=expected.resolve() or
        checkpoint.is_symlink() or metadata.is_symlink() or
        not checkpoint.resolve().is_relative_to(expected.resolve()) or not metadata.resolve().is_relative_to(expected.resolve())):
        raise PermissionError('frozen baseline artifacts escape the originating job')
    provenance = json.loads((Path(row['artifact_dir']) / 'run_manifest.json').read_text())
    source = subprocess.run(['git', '-C', str(repo_root), 'show',
                             f"{row['git_sha']}:candidates/{row['candidate_id']}/model.py"],
                            check=True, capture_output=True).stdout
    source_hash = hashlib.sha256(source).hexdigest()
    cp_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if source_hash != cfg['model_sha256'] or cp_hash != provenance['checkpoint_sha256']:
        raise ValueError('frozen baseline bytes differ from their recorded provenance')
    return {'baseline_id': run_id, 'run_id': run_id, 'seed': cfg['seed'],
            'source_sha256': source_hash, 'checkpoint_sha256': cp_hash,
            'checkpoint_manifest_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),
            'configuration_sha256': digest(cfg)}, source, row


def verify_baseline(store, expected, repo_root=REPO):
    actual, source, row = baseline_reference(store, expected['run_id'], repo_root)
    if any(actual[k] != expected[k] for k in actual if k != 'baseline_id'):
        raise ValueError('pinned baseline source/checkpoint/configuration/seed changed')
    return source, row


def research_evidence(store, repo_root=REPO, requested=()):
    """Bounded durable context: requested baselines, recent failures, best and recent evidence.

    Only committed candidate source, configuration and aggregate evidence are exposed.
    No image, mask, checkpoint, credential, or live ledger enters the agent mount.
    """
    rows = store.list_runs(5000)
    for rid in requested:
        if not any(r['run_id']==rid for r in rows):
            row=store.get_run(rid)
            if row: rows.append(row)
    with store.connect() as con:
        records = con.execute("SELECT run_id,payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' AND id IN (SELECT MAX(id) FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' GROUP BY run_id)").fetchall()
    summaries = {r['run_id']:None for r in rows}
    summaries.update({r['run_id']:json.loads(r['payload_json']) for r in records if r['run_id'] in summaries})
    def quality(row):
        return (summaries[row['run_id']] or {}).get('metrics', {}).get('dev_real/logical_au_spro_0_05_compatible', -1)
    from orchestrator.model_registry import states, retained, frontier
    retained_models=retained(store)
    persistent_ids=[retained_models[s['retention_id']]['run_id'] for s in states(store).values()]
    for cid in {m['cohort_id'] for m in retained_models.values()}:
        group=[m for m in retained_models.values() if m['cohort_id']==cid]
        persistent_ids += [retained_models[rid]['run_id'] for rid in frontier(group)]
    wanted = list(requested) + persistent_ids + [r['run_id'] for r in sorted(rows,key=quality,reverse=True)[:3]]
    for rid in persistent_ids:
        if not any(r['run_id']==rid for r in rows):
            row=store.get_run(rid)
            if row:rows.append(row)
    wanted += [r['run_id'] for r in rows if r['status'] in {'FAILED','OOM','TIMED_OUT','INTERRUPTED'}][:2]
    wanted += [r['run_id'] for r in rows[:3]]
    chosen = list(dict.fromkeys(wanted))
    context, files = [], {}
    for rid in chosen:
        row = next((r for r in rows if r['run_id']==rid), None)
        if not row or row['candidate_id']=='toy_normal_ae': continue
        item = {k:row[k] for k in ('run_id','hypothesis_id','candidate_id','status','error','git_sha','config_json')}
        item['real_feedback'] = summaries[rid]
        with store.connect() as con:
            review = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' ORDER BY id DESC LIMIT 1",(rid,)).fetchone()
            comparison = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='SCIENTIFIC_COMPARISON' ORDER BY id DESC LIMIT 1",(rid,)).fetchone()
        item['critique'] = json.loads(review[0]) if review else {}
        item['comparison'] = json.loads(comparison[0]) if comparison else {'status':'HISTORICAL_UNDECLARED_PROTOCOL'}
        try:
            reference, _, _ = baseline_reference(store, rid, repo_root)
            item['frozen_reference'] = reference
        except (ValueError, FileNotFoundError, KeyError, subprocess.CalledProcessError, PermissionError):
            pass
        context.append(item)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', rid) or not row['git_sha']: continue
        try:
            source = subprocess.run(['git','-C',str(repo_root),'show',f"{row['git_sha']}:candidates/{row['candidate_id']}/model.py"],
                                    check=True,capture_output=True).stdout
            if hashlib.sha256(source).hexdigest()!=json.loads(row['config_json']).get('model_sha256'): continue
        except subprocess.CalledProcessError:
            continue
        root = store.path.parent/'research_evidence'
        folder = root/rid/digest(item)[:16]
        if root.is_symlink() or any(p.is_symlink() for p in (root/rid,folder)):
            raise PermissionError('research context directory must not use symlinks')
        folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        for name, data in {'model.py':source, 'configuration.json':row['config_json'].encode(),
                           'evidence.json':json.dumps(item,sort_keys=True).encode()}.items():
            target = folder/name
            if target.exists():
                if target.is_symlink() or target.read_bytes()!=data: raise PermissionError('immutable evidence snapshot changed')
            else:
                target.write_bytes(data); target.chmod(0o444)
            files[f'{rid}/{name}'] = target
    return context, files


def execution_source(run, store, runtime, repo_root=REPO):
    """Materialize only verified committed source into controller-owned job inputs."""
    reference = json.loads(run['config_json']).get('frozen_baseline')
    if not reference:
        from orchestrator.experiment_runner import candidate_file
        return candidate_file(repo_root, run['candidate_id'])
    source, _ = verify_baseline(store, reference, repo_root)
    folder = Path(runtime) / 'controller/job_inputs' / run['run_id'] / 'source'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / 'model.py'
    if target.exists():
        if target.is_symlink() or target.read_bytes() != source:
            raise PermissionError('pinned baseline source was changed')
    else:
        target.write_bytes(source); target.chmod(0o444)
    return target


def comparison_result(plan, runs, feedback=None):
    """Check actual cell coverage/configuration and finite measured endpoints."""
    from benchmark.train_adapter import diagnostic_measurement_catalog,canonical_diagnostic_measurement
    catalog=diagnostic_measurement_catalog(plan['diagnostics'])
    feedback = feedback or {}
    expected = {(a['arm_id'], s) for a, s in cells(plan)}
    found = {}
    problems = []
    for row in runs:
        cfg = json.loads(row['config_json'])
        if cfg.get('plan_sha256') != digest(plan):
            continue
        key = (cfg.get('plan_arm'), cfg.get('plan_pair_seed',cfg.get('seed')))
        if key not in expected or key in found:
            problems.append(f'unexpected or repeated cell {key}')
        found[key] = row
    results = []
    for arm, seed in cells(plan):
        row = found.get((arm['arm_id'], seed))
        missing = []
        if row is None:
            problems.append(f"missing arm {arm['arm_id']} seed {seed}")
            results.append({'arm': arm['arm_id'], 'seed': seed, 'status': 'MISSING'})
            continue
        cfg = json.loads(row['config_json'])
        if cfg.get('execution_kind') != arm['execution']:
            missing.append('execution kind')
        for k, v in training_settings(plan, arm).items():
            if arm['execution'] == 'train' and cfg.get(k) != v:
                missing.append(f'effective setting {k}')
        if cfg.get('inference') != arm['inference'] or cfg.get('diagnostics') != plan['diagnostics']:
            missing.append('inference/diagnostic configuration')
        if row['status'] not in {'COMPLETED', 'REVIEWED', 'PROMOTED', 'REJECTED'}:
            missing.append(f"job status {row['status']}: {row.get('error', '')}")
        run_metrics = json.loads(row['metrics_json'])
        real = feedback.get(row['run_id'], {}).get('metrics', {})
        measurements = {}
        for request in plan['required_measurements']:
            source, key = request['source'], request['key']
            storage=catalog[canonical_diagnostic_measurement(catalog,key)] if source=='diagnostic' else key
            value = real.get(key) if source == 'real' else run_metrics.get(storage)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                missing.append(f'{source}:{key}')
            else:
                measurements[f'{source}:{key}'] = value
        for d in plan['diagnostics']:
            for key in d['required_metrics']:
                if catalog[f"{d['diagnostic_id']}/{key}"] not in run_metrics:
                    missing.append(f"diagnostic:{d['diagnostic_id']}/{key}")
        for key in ('protocol/optimizer_updates', 'protocol/source_exposures'):
            if key not in run_metrics:
                missing.append(key)
        for key, requested in [('optimizer_updates','expected_optimizer_updates'), ('source_exposures','expected_source_exposures')]:
            expected_count = 0 if arm['execution']=='frozen' else cfg.get(requested)
            if run_metrics.get('protocol/'+key) != expected_count:
                missing.append('training count '+key)
        actual_seed=training_seed(arm,seed)
        if cfg.get('seed') != actual_seed or run_metrics.get('protocol/seed') != actual_seed:
            missing.append('measured seed')
        if missing:
            problems.append(f"{arm['arm_id']} seed {seed}: {', '.join(missing)}")
        results.append({'arm': arm['arm_id'], 'seed': seed, 'run_id': row['run_id'],
                        'training_seed':actual_seed,
                        'status': row['status'], 'missing': missing, 'measurements': measurements,
                        'effective_configuration_sha256':digest(cfg), 'source_sha256':cfg.get('model_sha256'),
                        'git_commit':row.get('git_sha'), 'optimizer_updates':run_metrics.get('protocol/optimizer_updates'),
                        'source_exposures':run_metrics.get('protocol/source_exposures')})
    differences = []
    reference_arm = plan['baseline_arm_id']
    by_cell = {(r['arm'], r['seed']): r for r in results}
    for arm in (a for a in plan['arms'] if a['arm_id'] != reference_arm):
        for seed in plan['seeds']:
            left, right = by_cell[(arm['arm_id'], seed)], by_cell[(reference_arm, seed)]
            for key, value in left.get('measurements', {}).items():
                if key in right.get('measurements', {}):
                    differences.append({'arm': arm['arm_id'], 'reference_arm': reference_arm,
                                        'seed': seed, 'measurement': key,
                                        'difference': value - right['measurements'][key]})
    intervals = []
    for comparison in plan['paired_comparisons']:
        for seed in plan['seeds']:
            try:
                from benchmark.metrics_v2 import paired_image_difference
                measurements = []
                for arm_id in (comparison['left_arm'],comparison['right_arm']):
                    row = found[(arm_id,seed)]
                    folder = Path(row['artifact_dir'])
                    path = folder/f"diagnostic_{comparison['diagnostic_id']}.json"
                    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                        raise ValueError('unsafe paired diagnostic path')
                    items = json.loads(path.read_text()).get('per_image',[])
                    values = {item['source_id']:float(item['metrics'][comparison['metric']]) for item in items}
                    if not values or len(values)!=len(items): raise ValueError('missing/duplicate image units')
                    measurements.append(values)
                left,right=measurements
                if set(left)!=set(right):raise ValueError('paired assessment source identities differ')
                ids=sorted(left)
                interval=paired_image_difference([left[i] for i in ids],[right[i] for i in ids],
                    seed=comparison['bootstrap_seed'],repeats=comparison['repeats'])
                intervals.append(dict(interval,comparison_id=comparison['comparison_id'],training_seed=seed))
            except (KeyError,ValueError,TypeError,OSError) as exc:
                problems.append(f"paired comparison {comparison['comparison_id']} seed {seed}: {exc}")
    from orchestrator.planning_contract import evidence_classification
    eligibility=evidence_classification(plan, runs)
    if problems:
        eligibility.update(promotion_eligible=False,evidence_level='INCOMPLETE')
        eligibility['reasons'].append('Requested execution or measured evidence is incomplete.')
    fixed=any(a.get('fixed_checkpoint_seed') is not None for a in plan['arms'])
    return {'scientific_eligibility':eligibility, 'plan_id': plan['plan_id'], 'hypothesis_id': plan['hypothesis_id'],
            'plan_sha256': digest(plan), 'version': VERSION, 'kind': plan['kind'],
            'status': 'INCOMPLETE' if problems else ('COMPLETE_CONTROLLED' if plan['kind'] == 'controlled_comparison' and not fixed else 'COMPLETE_EXPLORATORY'),
            'expected_jobs': len(expected), 'recorded_jobs': len(found), 'problems': problems,
            'cells': results, 'paired_seed_differences': differences, 'paired_image_intervals': intervals,
            'interpretation': 'Execution completeness only; hypothesis support and uncertainty require critique.'}


def plan_history(campaigns, campaign_id):
    with campaigns.store.connect() as con:
        return [json.loads(r[0]) for r in con.execute(
            "SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_REGISTERED' ORDER BY id",
            (campaign_id,))]


def record_result(store, campaigns, campaign_id, plan, runs, runtime, error=''):
    from orchestrator.development_feedback import latest_summary
    from orchestrator.mlflow_tracker import enqueue_run_tracking
    result = comparison_result(plan, runs, {r['run_id']: latest_summary(store, r['run_id']) or {} for r in runs})
    if error:
        result['status'] = 'INCOMPLETE'; result['problems'].append(error)
    campaigns.event(campaign_id, 'EXPERIMENT_PLAN_RESULT', result=result)
    for row in runs:
        if json.loads(row['config_json']).get('plan_sha256') != digest(plan):
            continue
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (row['run_id'], time.time(), 'SCIENTIFIC_COMPARISON', json.dumps(result, sort_keys=True)))
        folder = Path(row['artifact_dir']) if row['artifact_dir'] else None
        if folder and folder.resolve() == (Path(runtime)/'jobs'/row['run_id']).resolve():
            (folder/'plan_comparison.json').write_text(json.dumps({'plan': plan, 'result': result}, sort_keys=True, indent=2))
            enqueue_run_tracking(store, row['run_id'], event_key_suffix='plan:' + digest(result))
    return result


def execute_plan(store, campaigns, campaign_id, runtime, plan):
    """Execute exact arm/seed cells through the existing guarded queue. Never replay."""
    from orchestrator import controller
    from orchestrator.autonomy import await_authorized_stage, CampaignHalted
    from orchestrator.budgets import BudgetExceeded, enforce
    from orchestrator.experiment_runner import candidate_file, sha256

    runtime = Path(runtime)
    plan=canonicalize_diagnostic_measurements(plan)
    record = register_plan(store, campaigns, campaign_id, plan)
    with store.connect() as con:
        finished = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_RESULT' ORDER BY id DESC",(campaign_id,)).fetchall()
    published = next((json.loads(r[0])['result'] for r in finished
                      if json.loads(r[0])['result']['plan_sha256']==digest(plan)),None)
    if published is not None:
        return published  # Published complete/incomplete comparisons are historical evidence.
    from orchestrator.planning_contract import admit_plan
    admit_plan(plan,store,controller.REPO)
    cfg = json.loads(campaigns.get(campaign_id)['config_json'])
    limits = cfg['limits']; budget = plan['resource_budget']
    error = ''
    try:
        if plan['evaluation']['real_development_feedback'] and not cfg['research'].get('real_development_feedback'):
            raise PermissionError('plan requested real feedback outside campaign authorization')
        usage = campaigns.usage(campaign_id)
        existing = [r for r in campaigns.runs(campaign_id) if json.loads(r['config_json']).get('plan_sha256') == digest(plan)]
        remaining_cells = len(cells(plan)) - len(existing)
        if ((limits['max_experiments'] is not None and remaining_cells + usage['experiments'] > limits['max_experiments']) or
            budget['max_job_seconds'] > limits['max_job_seconds'] or
            budget['max_gpu_seconds'] > limits['max_gpu_hours']*3600 or
            budget['max_wall_seconds'] > limits['max_wall_hours']*3600):
            raise ValueError('scientific plan exceeds authorized campaign ceilings; no defaults substituted')
        from orchestrator.campaign_config import time_based, plan_seconds
        if time_based(cfg) and plan_seconds(plan,cfg) > limits['max_wall_hours']*3600-usage['wall_seconds']:
            raise BudgetExceeded('complete frozen comparison plus evaluation and critique cannot fit remaining research duration')
        train_arms = [a for a in plan['arms'] if a['execution'] == 'train']
        module = candidate_file(controller.REPO, plan['hypothesis_id']) if train_arms or plan['diagnostics'] else None
        if train_arms and time_based(cfg):
            from orchestrator.experiment_runner import require_training_phases
            require_training_phases(module)
        source_hash = sha256(module) if module else ''
        trusted_files = ['orchestrator/autonomy.py','orchestrator/controller.py','orchestrator/experiment_plan.py',
                         'orchestrator/planning_contract.py','orchestrator/experiment_runner.py','orchestrator/sandbox.py','orchestrator/development_feedback.py',
                         'scripts/run_candidate.py','scripts/training_phase.py','scripts/feedback_infer.py','scripts/candidate_smoke.py',
                         'benchmark/train_adapter.py','benchmark/feedback_metrics.py','benchmark/feedback_data.py',
                         'schemas/experiment_plan.schema.json','configs/development_feedback.json']
        trusted_files += ['orchestrator/model_registry.py','configs/champion_policy.json',
                          'orchestrator/campaign_config.py','orchestrator/state_store.py','orchestrator/campaign_state.py','orchestrator/budgets.py',
                          'orchestrator/gpu_queue.py','orchestrator/codex_runner.py','orchestrator/codex_sandbox.py',
                          'benchmark/data_guard.py','benchmark/validation.py','benchmark/metrics_v2.py',
                          'benchmark/metrics.py','benchmark/inference_benchmark.py','pyproject.toml',
                          'RESEARCH_CHARTER.md','research_program.md']
        trusted_hashes = {f:sha256(REPO/f) for f in trusted_files}
        from benchmark.data_guard import load_manifest, manifest_fingerprint
        manifest, _ = load_manifest(runtime/'good_manifest.json')
        data_pin = {'fingerprint':manifest_fingerprint(manifest),'ordered_entries_sha256':digest(manifest['images'])}
        with store.connect() as con:
            pinned = con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='PLAN_SOURCE_PINNED' ORDER BY id", (campaign_id,)).fetchall()
        prior_pin = next((json.loads(r[0]) for r in pinned if json.loads(r[0])['plan_sha256'] == digest(plan)), None)
        if prior_pin and prior_pin['source_sha256'] != source_hash:
            raise ValueError('candidate source changed during a registered comparison')
        if prior_pin and (prior_pin['trusted_source_sha256']!=trusted_hashes or prior_pin['data_pin']!=data_pin):
            raise ValueError('trusted evaluator/protocol or ordered evaluation sources changed during comparison')
        if not prior_pin:
            campaigns.event(campaign_id, 'PLAN_SOURCE_PINNED', plan_sha256=digest(plan), source_sha256=source_hash,
                            trusted_source_sha256=trusted_hashes,data_pin=data_pin)
        def verify_inputs():
            current_manifest=json.loads((runtime/'good_manifest.json').read_text())
            if {'fingerprint':manifest_fingerprint(current_manifest),'ordered_entries_sha256':digest(current_manifest['images'])} != data_pin:
                raise ValueError('training/calibration/assessment source order changed during comparison')
            if any(sha256(REPO/f)!=value for f,value in trusted_hashes.items()):
                raise ValueError('trusted evaluation code changed during comparison')
            if module and sha256(module)!=source_hash:
                raise ValueError('candidate or shared diagnostic source changed during comparison')
        for arm, seed in cells(plan):
            await_authorized_stage(campaigns, campaign_id)
            rows = campaigns.runs(campaign_id)
            plan_runs = [r for r in rows if json.loads(r['config_json']).get('plan_sha256') == digest(plan)]
            rid = cell_run_id(campaign_id, plan, arm['arm_id'], seed)
            if store.get_run(rid):
                # Any recorded attempt is evidence. Interrupted/failed cells are not replayed.
                if store.get_run(rid)['status'] not in {'COMPLETED','REVIEWED','PROMOTED','REJECTED'}:
                    if plan['failure_criteria']['on_job_failure'] == 'stop':
                        break
                continue
            failed = sum(r['status'] in {'FAILED','OOM','TIMED_OUT','CANCELLED','INTERRUPTED'} for r in plan_runs)
            if failed >= plan['failure_criteria']['max_failed_jobs']:
                break
            remaining_gpu = min(limits['max_gpu_hours']*3600-campaigns.usage(campaign_id)['gpu_seconds'],
                                budget['max_gpu_seconds']-sum(r['gpu_seconds'] for r in plan_runs))
            remaining_wall = min(limits['max_wall_hours']*3600-campaigns.usage(campaign_id)['wall_seconds'],
                                 budget['max_wall_seconds']-(time.time()-record['registered_at']))
            if min(remaining_gpu, remaining_wall) < budget['max_job_seconds']:
                raise BudgetExceeded('insufficient authorized time for the exact planned job allowance')
            enforce(store, next_gpu_seconds=budget['max_job_seconds'], runtime=runtime, campaign_id=campaign_id)
            effective = training_settings(plan, arm)
            verify_inputs()
            effective.update(seed=seed, inference=arm['inference'], diagnostics=plan['diagnostics'],
                             plan_id=plan['plan_id'], plan_sha256=digest(plan), plan_arm=arm['arm_id'],
                             plan_contract_version=VERSION, execution_kind=arm['execution'], model_sha256=source_hash,
                             metric_evaluator_version='development-v2.0')
            if arm.get('fixed_checkpoint_seed') is not None:
                effective.update(seed=arm['fixed_checkpoint_seed'],plan_pair_seed=seed)
            candidate_id = plan['hypothesis_id']
            if arm['execution'] == 'frozen':
                if arm['source_arm_id']:
                    source_id = cell_run_id(campaign_id, plan, arm['source_arm_id'], seed)
                    reference, _, origin = baseline_reference(store, source_id, controller.REPO)
                else:
                    reference = next(b for b in plan['baselines'] if b['baseline_id'] == arm['baseline_by_seed'][str(seed)])
                    _, origin = verify_baseline(store, reference, controller.REPO)
                origin_cfg = json.loads(origin['config_json'])
                if origin_cfg.get('seed') != effective['seed']:
                    raise ValueError('frozen checkpoint does not belong to requested seed')
                for key in set(plan['controlled_variables']) | {'size'}:
                    if origin_cfg.get(key) != effective.get(key):
                        raise ValueError(f'frozen baseline does not match controlled setting {key}')
                effective.update(frozen_baseline=reference, baseline_effective_configuration=origin_cfg,
                                 model_sha256=reference['source_sha256'])
                if plan['diagnostics']:
                    effective['diagnostic_provider'] = {'candidate_id':plan['hypothesis_id'],'source_sha256':source_hash}
                candidate_id = origin['candidate_id']
            effective['trusted_source_sha256'] = trusted_hashes
            effective['data_pin'] = data_pin
            rid = controller.queue_candidate(store, plan['hypothesis_id'], candidate_id, 'juice_bottle', 'pilot', 'cuda',
                runtime/'good_manifest.json', seconds=budget['max_job_seconds'], personal_mode_authorized=True,
                campaign_id=campaign_id, campaign_name=cfg['campaign']['name'],
                synthetic_evaluation=plan['evaluation']['synthetic_diagnostics'],
                cuda_smoke_required=arm['execution']=='train', effective_config=effective, run_id=rid)
            campaigns.attach_run(campaign_id, rid)
            campaigns.event(campaign_id, 'EXPERIMENT_QUEUED', run_id=rid, plan_id=plan['plan_id'], arm=arm['arm_id'], seed=seed)
            outcome = controller.run_worker_once(store, runtime/'good_manifest.json', runtime,
                personal_mode_authorized=True, cancel=lambda: campaigns.get(campaign_id)['status']=='STOPPING', expected_run_id=rid)
            if outcome.get('run_id') != rid:
                raise RuntimeError('guarded queue returned a different planned job')
            row = store.get_run(rid)
            campaigns.update(campaign_id, active_run_id='')
            campaigns.event(campaign_id, 'EXPERIMENT_FINISHED', run_id=rid, status=row['status'])
            if row['status'] == 'COMPLETED':
                if plan['evaluation']['real_development_feedback']:
                    from orchestrator.development_feedback import evaluate_run
                    await_authorized_stage(campaigns, campaign_id, finishing=True)
                    remaining = min(limits['max_wall_hours']*3600-campaigns.usage(campaign_id)['wall_seconds'],
                                    budget['max_wall_seconds']-(time.time()-record['registered_at']))
                    if remaining < 1: raise BudgetExceeded('wall budget reached before required feedback')
                    summary = evaluate_run(runtime, store, rid, timeout=min(900, cfg['research'].get('feedback_cpu_timeout_seconds',600), int(remaining)),
                        personal_mode_authorized=True, cancel=lambda: campaigns.get(campaign_id)['status']=='STOPPING')
                    campaigns.event(campaign_id,'REAL_DEVELOPMENT_FEEDBACK',run_id=rid,summary=summary)
            elif plan['failure_criteria']['on_job_failure'] == 'stop':
                break
            verify_inputs()
    except (CampaignHalted, BudgetExceeded) as exc:
        error = str(exc)
        from orchestrator.campaign_config import time_based
        if time_based(cfg) and campaigns.get(campaign_id)['status']=='RUNNING':
            campaigns.set_status(campaign_id,'LIMIT_REACHED',error)
    except PermissionError as exc:
        record_result(store, campaigns, campaign_id, plan, campaigns.runs(campaign_id), runtime, str(exc))
        raise
    except Exception as exc:
        # Preserve completed cells and exact failure; never call an incomplete plan valid.
        error = f'{type(exc).__name__}: {exc}'
    active = campaigns.get(campaign_id)['active_run_id']
    if active and (store.get_run(active) or {}).get('status') not in {'RUNNING','QUEUED'}:
        campaigns.update(campaign_id,active_run_id='')
    result = record_result(store, campaigns, campaign_id, plan, campaigns.runs(campaign_id), runtime, error)
    return result


def review_plan(store, campaigns, campaign_id, plan, result):
    from orchestrator import controller
    from orchestrator.mlflow_tracker import enqueue_run_tracking
    rows = [r for r in campaigns.runs(campaign_id) if json.loads(r['config_json']).get('plan_sha256')==digest(plan)]
    prompt = ('Assess this scientific comparison using only measured evidence. Completion is execution completeness, '
              'not hypothesis support. If INCOMPLETE, do not assert a causal result, supported hypothesis, or promotion. '
              'Exploratory screening does not establish causality. Include exact comparison_status and plan_id. '
              'Separate single-seed results, paired differences and uncertainty; never invent missing diagnostics. '
              + json.dumps({'plan':plan,'completion':result,'runs':rows}))
    def request(feedback, attempt):
        return controller._budgeted_invoke_once(store, 'critic', controller.REPO, REPO/'schemas/plan_review.schema.json',
            prompt+feedback, campaign=(campaigns,campaign_id), timeout=180, personal_mode=True)
    def check(response):
        review = response['response']
        if review['plan_id'] != plan['plan_id'] or review['comparison_status'] != result['status']:
            raise ValueError('critic comparison identity/status contradicts measured completion')
        if ((result['status']=='INCOMPLETE' and (review['hypothesis_outcome']=='supported' or review['recommended_next_action']=='promote_for_review')) or
            (review['causal_claim'] and (result['status']!='COMPLETE_CONTROLLED' or not plan['paired_comparisons'] or
                any(a.get('fixed_checkpoint_seed') is not None for a in plan['arms'])))):
            raise ValueError('critic attempted a scientific claim unsupported by completed controlled work')
        return review
    _, review, _ = controller._with_agent_correction(store, (campaigns,campaign_id), 'critic', request, check)
    for row in rows:
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (row['run_id'], time.time(), 'CRITIC_REVIEW', json.dumps(review)))
        if row['status']=='COMPLETED': store.transition(row['run_id'],'REVIEWED')
        enqueue_run_tracking(store,row['run_id'],event_key_suffix='plan-review:'+digest(plan))
    campaigns.event(campaign_id,'CRITIQUE',plan_id=plan['plan_id'],review=review,
                    run_id=rows[-1]['run_id'] if rows else '')
    campaigns.event(campaign_id,'NEXT_DECISION',action=review['recommended_next_action'],
                    evidence_plan=plan['plan_id'],evidence_run=rows[-1]['run_id'] if rows else '',
                    reason=review['objective_measured_results'],comparison_status=result['status'],
                    next_hypothesis=f"evidence-followup:{plan['hypothesis_id']}:{review['recommended_next_action']}")
    campaigns.update(campaign_id,next_hypothesis=f"evidence-followup:{plan['hypothesis_id']}:{review['recommended_next_action']}",
                     consecutive_failures=0 if not any(r['status'] in {'FAILED','OOM','TIMED_OUT'} for r in rows) else campaigns.get(campaign_id)['consecutive_failures']+1)
    store.set_hypothesis_status(plan['hypothesis_id'],'INCOMPLETE' if result['status']=='INCOMPLETE' else 'COMPLETED')
    from orchestrator.model_registry import consider_comparison
    decisions=consider_comparison(store,campaigns,campaign_id,plan,result,controller.REPO)
    for row in rows:
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (row['run_id'],time.time(),'CUMULATIVE_REVIEW',json.dumps(decisions,sort_keys=True)))
        enqueue_run_tracking(store,row['run_id'],event_key_suffix='cumulative:'+digest(plan))
    return review
