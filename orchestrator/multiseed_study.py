"""Predeclared Stage 6 study; never runs without separate campaign authorization.

The trusted controller fixes all arms, seeds, budgets and evaluation inputs.
Candidate code still runs only through the existing guarded GPU queue.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmark.metrics_v2 import EVALUATOR_VERSION, paired_image_difference, wilson_interval
from orchestrator.ablation_study import _candidate_snapshot, _case_metrics, _training_record
from orchestrator.autonomy import REPO, await_authorized_stage, remaining_job_seconds

ARMS = ('local', 'global', 'combined', 'neither', 'spatial')
SEEDS = (7, 17, 29)
METRICS = ('primary_pixel_ap', 'alternate_pixel_ap', 'relocated_pixel_ap')


def prescribed_plan(config):
    research, limits = config['research'], config['limits']
    if (research.get('study') != 'stage6_multiseed' or tuple(research.get('arms', ())) != ARMS or
        tuple(research.get('seeds', ())) != SEEDS or research.get('evaluator_version') != EVALUATOR_VERSION or
        research.get('steps_per_job') != 100 or research.get('batch_size') != 4 or
        research.get('input_size') != 128 or limits.get('max_experiments') != 15 or
        limits.get('max_codex_calls') != 1 or limits.get('max_parallel_gpu_jobs') != 1 or
        limits.get('max_job_seconds') != 120 or limits.get('max_gpu_hours') != .5 or
        limits.get('max_wall_hours') != 4 or limits.get('max_consecutive_failures') != 1 or
        research.get('confirmatory_min_gain') != .03):
        raise ValueError('Stage 6 requires the reviewed fixed 5-arm, 3-seed, version-2 protocol')
    return [(seed, arm) for seed in SEEDS for arm in ARMS]


def compare_multiseed(runtime, runs):
    """Separate image-pair uncertainty from variation over three training seeds."""
    import numpy as np
    if len(runs) != 15:
        raise ValueError('all 15 measured runs are required for confirmatory comparison')
    by_key = {}
    dataset_fingerprints = set()
    generator_fingerprints = set()
    calibration_splits = set()
    assessment_splits = set()
    for run in runs:
        cfg = json.loads(run['config_json'])
        key = (cfg.get('seed'), cfg.get('context_arm'))
        if key in by_key or run['status'] not in {'COMPLETED', 'REVIEWED'}:
            raise ValueError('duplicated or incomplete Stage 6 run')
        meta = json.loads((Path(runtime) / 'jobs' / run['run_id'] / 'run_manifest.json').read_text())
        if meta.get('metric_evaluator_version') != EVALUATOR_VERSION:
            raise ValueError('mixed evaluator versions cannot be compared')
        if any(cfg.get(field) != value for field, value in
               (('size', 128), ('steps', 100), ('batch_size', 4))):
            raise ValueError('unmatched training or preprocessing configuration')
        generator_fingerprints.add(meta.get('synthetic_generator_source_sha256'))
        calibration_splits.add(meta.get('calibration_path_sha256'))
        assessment_splits.add(meta.get('assessment_path_sha256'))
        dataset_fingerprints.add(run['manifest_sha256'])
        by_key[key] = (run, _case_metrics(runtime, run['run_id']))
    if set(by_key) != set((seed, arm) for seed in SEEDS for arm in ARMS):
        raise ValueError('incomplete prescribed seed-by-arm grid')
    if (len(dataset_fingerprints) != 1 or len(generator_fingerprints) != 1 or
        len(calibration_splits) != 1 or len(assessment_splits) != 1 or
        None in generator_fingerprints | calibration_splits | assessment_splits):
        raise ValueError('dataset, generator, calibration or assessment fingerprint differs across arms')
    summary = {'metric_evaluator_version': EVALUATOR_VERSION,
               'scope': 'juice_bottle synthetic development, three training seeds, no official anomalies',
               'seeds': list(SEEDS), 'arms': list(ARMS),
               'dataset_fingerprint': dataset_fingerprints.pop(),
               'generator_source_sha256': generator_fingerprints.pop(),
               'calibration_path_sha256': calibration_splits.pop(),
               'assessment_path_sha256': assessment_splits.pop(),
               'per_seed': [], 'across_seed': {}, 'paired_differences': {}}
    for seed in SEEDS:
        anchor_cases = by_key[(seed, 'local')][1]
        keys = [(row['image_index'], row['donor_pair_cluster']) for row in anchor_cases]
        for arm in ARMS:
            run, cases = by_key[(seed, arm)]
            if [(row['image_index'], row['donor_pair_cluster']) for row in cases] != keys:
                raise ValueError('assessment image or donor pairing differs across arms')
            values = json.loads(run['metrics_json'])
            summary['per_seed'].append({'seed': seed, 'arm': arm, 'run_id': run['run_id'],
                'macro_ap': {metric: float(np.mean([row[metric] for row in cases])) for metric in METRICS},
                'pooled_ap': {metric: values[key] for metric, key in (
                    ('primary_pixel_ap', 'dev_synthetic/pixel_ap'),
                    ('alternate_pixel_ap', 'dev_synthetic/alternate_pixel_ap'),
                    ('relocated_pixel_ap', 'dev_synthetic/relocated_pixel_ap'))},
                'normal_false_positives': int(values['normal/image_false_positives']),
                'normal_images_assessed': int(values['normal/image_assessed']),
                'normal_fpr_wilson_ci95': wilson_interval(
                    int(values['normal/image_false_positives']), int(values['normal/image_assessed'])),
                'latency_p95_ms': values.get('inference/e2e_ms_p95'),
                'parameters': values.get('inference/parameters'),
                'peak_vram_mib': values.get('inference/peak_vram_mib'),
                'gpu_slot_seconds': float(run['gpu_seconds'])})
    for arm in ARMS:
        records = [row for row in summary['per_seed'] if row['arm'] == arm]
        summary['across_seed'][arm] = {}
        for metric in METRICS:
            values = np.asarray([row['macro_ap'][metric] for row in records])
            summary['across_seed'][arm][metric] = {'mean': float(values.mean()),
                'sample_sd': float(values.std(ddof=1)), 'seeds': len(values)}
    for arm in ARMS:
        if arm == 'local':
            continue
        comparisons = {}
        for metric in METRICS:
            seed_results = []
            for seed in SEEDS:
                reference = by_key[(seed, arm)][1]
                alternative = by_key[(seed, 'local')][1]
                seed_results.append(paired_image_difference(
                    [row[metric] for row in reference], [row[metric] for row in alternative],
                    clusters=[row['donor_pair_cluster'] for row in reference]))
            differences = np.asarray([row['mean_difference'] for row in seed_results])
            comparisons[metric] = {'arm_minus_local_mean': float(differences.mean()),
                'between_seed_sample_sd': float(differences.std(ddof=1)),
                'per_seed_image_cluster_intervals': seed_results}
        summary['paired_differences'][arm] = comparisons
    return summary


def run_multiseed_cycle(store, campaigns, campaign_id, runtime):
    """Execute one prescribed job, then critique only after all 15 are measured."""
    from orchestrator.controller import budgeted_invoke, queue_candidate, run_worker_once
    from orchestrator.experiment_runner import probe_candidate_interface, run_candidate_unit_tests
    from orchestrator.mlflow_tracker import enqueue_run_tracking

    campaign = campaigns.get(campaign_id)
    config = json.loads(campaign['config_json'])
    plan = prescribed_plan(config)
    previous = campaigns.runs(campaign_id)
    if len(previous) >= len(plan):
        raise RuntimeError('Stage 6 grid has already been attempted')
    if any(run['status'] != 'REVIEWED' for run in previous):
        raise RuntimeError('prior Stage 6 job was not reviewed; expensive work will not be replayed')
    for old, expected in zip(previous, plan):
        cfg = json.loads(old['config_json'])
        if (cfg.get('seed'), cfg.get('context_arm')) != expected:
            raise RuntimeError('stored Stage 6 run sequence differs from predeclared protocol')
    module, source_hash = _candidate_snapshot()
    if previous and any(json.loads(run['config_json']).get('model_sha256') != source_hash for run in previous):
        raise PermissionError('candidate source changed during confirmatory study')
    if not previous:
        import os
        probe_candidate_interface(REPO, module, Path(os.environ['LOCO_GOOD_ROOT']), runtime, 'masked_context_ablation')
        tests = run_candidate_unit_tests(REPO, module, Path(os.environ['LOCO_GOOD_ROOT']), runtime, 'masked_context_ablation')
        campaigns.event(campaign_id, 'CANDIDATE_TESTS', **tests)
    await_authorized_stage(campaigns, campaign_id)
    seed, arm = plan[len(previous)]
    hypothesis_id = f'stage6_{arm}_seed{seed}'
    with store.connect() as con:
        exists = con.execute('SELECT 1 FROM hypotheses WHERE hypothesis_id=?', (hypothesis_id,)).fetchone()
    if not exists:
        store.add_proposal({'schema_version': '1.0', 'hypothesis_id': hypothesis_id,
            'research_question': 'Does broad context improve synthetic localization across fixed seeds and controls?',
            'claim': f'Prescribed {arm} arm, seed {seed}; no adaptive selection inside confirmatory grid.',
            'expected_observation': 'Combined-minus-local macro AP >=0.03 across primary and alternate generators without more normal/control alarms.',
            'evaluation_plan': 'development-v2.0; 18 good-only calibration and 36 distinct good assessment images; fixed masks and paired bootstrap',
            'estimated_gpu_minutes': 2, 'new_dependencies': [],
            'risk_and_shortcuts': 'Synthetic geometry, editing artifacts, one category and uncertified donor controls limit scientific interpretation.',
            'priority_rank': len(previous) + 1,
            'references': ['Stage 5 campaign 9b5e808f-48e9-4d72-a5ee-2b95df84e18a']})
    campaigns.update(campaign_id, current_hypothesis=hypothesis_id)
    campaigns.event(campaign_id, 'STAGE6_ARM_SELECTED', seed=seed, arm=arm, plan_index=len(previous))
    seconds = min(120, remaining_job_seconds(campaigns, campaign_id))
    if seconds < 15:
        raise RuntimeError('insufficient remaining bounded GPU time')
    manifest = Path(runtime) / 'good_manifest.json'
    run_id = queue_candidate(store, hypothesis_id, 'masked_context_ablation', 'juice_bottle',
        'pilot', 'cuda', manifest, seconds=seconds, personal_mode_authorized=True,
        campaign_id=campaign_id, campaign_name=config['campaign']['name'], context_arm=arm, seed=seed)
    campaigns.attach_run(campaign_id, run_id)
    result = run_worker_once(store, manifest, runtime, personal_mode_authorized=True,
        cancel=lambda: campaigns.get(campaign_id)['status'] == 'STOPPING', expected_run_id=run_id)
    run = store.get_run(run_id)
    campaigns.update(campaign_id, active_run_id='')
    campaigns.event(campaign_id, 'STAGE6_RUN_FINISHED', run_id=run_id, seed=seed, arm=arm,
                    status=run['status'])
    if result.get('run_id') != run_id or run['status'] != 'COMPLETED':
        raise RuntimeError(f'Stage 6 run {run_id} ended {run["status"]}: {run["error"]}')
    await_authorized_stage(campaigns, campaign_id, finishing=True)
    record = _training_record(runtime, run_id)
    if (record['updates'], record['image_exposures'], record['training_tensor_device'],
        record['parameter_updated']) != (100, 400, 'cuda:0', True):
        raise RuntimeError('Stage 6 CUDA training exposure did not match prescribed protocol')
    meta = json.loads((Path(runtime) / 'jobs' / run_id / 'run_manifest.json').read_text())
    if meta.get('metric_evaluator_version') != EVALUATOR_VERSION:
        raise RuntimeError('Stage 6 run used an unapproved evaluator version')
    campaigns.event(campaign_id, 'STAGE6_TRAINING_VERIFIED', run_id=run_id, **record)
    store.transition(run_id, 'REVIEWED')
    enqueue_run_tracking(store, run_id)
    if len(previous) + 1 < len(plan):
        return
    comparison = compare_multiseed(runtime, previous + [store.get_run(run_id)])
    comparison_path = Path(runtime) / 'jobs' / run_id / 'stage6_comparison.json'
    comparison_path.write_text(json.dumps(comparison, sort_keys=True, indent=2))
    campaigns.event(campaign_id, 'STAGE6_COMPARISON', comparison=comparison)
    review = budgeted_invoke(store, 'critic', REPO, REPO / 'schemas' / 'review.schema.json',
        'Critique the completed predeclared Stage 6 synthetic development study. '
        'Distinguish paired image uncertainty from between-seed variation, inspect normal and '
        'artifact-control alarms, and reject official LOCO claims. Recommend a falsifiable next '
        'experiment; no new job is authorized. Final run ID: ' + run_id + '. Comparison: '
        + json.dumps(comparison), timeout=180, campaign=(campaigns, campaign_id),
        personal_mode=True)['response']
    if review['run_id'] != run_id:
        raise ValueError('critic review references a different final run')
    import time
    with store.connect() as con:
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (run_id, time.time(), 'CRITIC_REVIEW', json.dumps(review)))
    campaigns.event(campaign_id, 'CRITIQUE', run_id=run_id, review=review)
    campaigns.event(campaign_id, 'NEXT_DECISION', action=review['recommended_next_action'],
                    evidence_run=run_id, reason=review['suggested_falsification'])
    enqueue_run_tracking(store, run_id, event_key_suffix='stage6-final-comparison')
