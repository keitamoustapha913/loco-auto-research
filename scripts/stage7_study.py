"""Trusted, read-only Stage 7 study launcher for previously trained checkpoints.

It never imports candidate code in the host process. Candidate inference runs in
the existing no-network Bubblewrap mount boundary on CPU. The protocol file is
fixed in the repository before measurements are inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from benchmark.data_guard import load_manifest, manifest_fingerprint
from benchmark.metrics_v2 import paired_image_difference
from benchmark.stage7_controls import PROTOCOL_VERSION
from orchestrator.multiseed_study import ARMS, SEEDS
from orchestrator.sandbox import command as sandbox_command, write_sandbox_manifest
from orchestrator.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs' / 'stage7_frozen_eval.json'
MODEL = REPO / 'candidates' / 'masked_context_ablation' / 'model.py'
CAMPAIGN = '5aae9e6c-5162-46aa-b455-6c503fe16fb1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_protocol():
    cfg = json.loads(CONFIG.read_text())
    if (cfg != {'protocol': PROTOCOL_VERSION, 'evaluator': 'development-v2.0', 'device': 'cpu',
                'assessment_images': 12, 'assessment_stride': 3, 'edit_seed': 13007,
                'arms': ['local', 'global', 'combined', 'neither'], 'seeds': [7, 17, 29],
                'timeout_seconds_per_checkpoint': 120, 'bootstrap_repeats': 2000,
                'no_training': True, 'official_test_access': False}):
        raise ValueError('Stage 7 fixed, CPU-only protocol was modified')
    return cfg


def frozen_runs(store, runtime):
    by_key = {}
    for run in store.list_runs(limit=5000):
        cfg = json.loads(run['config_json'])
        if not run['hypothesis_id'].startswith('stage6_') or cfg.get('campaign_id') != CAMPAIGN:
            continue
        key = (cfg.get('seed'), cfg.get('context_arm'))
        if key in by_key or run['status'] != 'REVIEWED':
            raise ValueError('duplicate or incomplete Stage 6B checkpoint')
        by_key[key] = run
    if set(by_key) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError('Stage 6B run grid incomplete')
    source_hash = sha(MODEL)
    for run in by_key.values():
        cfg = json.loads(run['config_json'])
        job = runtime / 'jobs' / run['run_id']
        meta = json.loads((job / 'run_manifest.json').read_text())
        checkpoint = job / 'checkpoint' / 'checkpoint_manifest.json'
        ckpt = json.loads(checkpoint.read_text())
        if (meta.get('metric_evaluator_version') != 'development-v2.0' or
            meta.get('model_module_sha256') != source_hash or
            ckpt.get('code_sha256') != source_hash or
            ckpt.get('seed') != cfg.get('seed') or
            ckpt.get('config', {}).get('context_arm') != cfg.get('context_arm') or
            ckpt.get('config', {}).get('size') != 128 or
            run['manifest_sha256'] != meta.get('good_manifest_sha256') or
            sha(job / 'checkpoint' / 'model.pt') != ckpt['checkpoint_sha256'] or
            float(json.loads(run['metrics_json'])['inference/parameters']) != 3420):
            raise ValueError('frozen source, checkpoint or evaluator provenance mismatch')
    # The 1x1 predictor evaluates visibility only where it is zero. Check the
    # saved outcomes as well as the effective trained tensor state/scale.
    import torch
    for seed in SEEDS:
        a, b = by_key[(seed, 'neither')], by_key[(seed, 'spatial')]
        left = torch.load(runtime / 'jobs' / a['run_id'] / 'checkpoint/model.pt', map_location='cpu', weights_only=True)
        right = torch.load(runtime / 'jobs' / b['run_id'] / 'checkpoint/model.pt', map_location='cpu', weights_only=True)
        if (left['size'] != right['size'] or
            any(not torch.equal(left['state'][k], right['state'][k]) for k in left['state']) or
            not torch.equal(left['scale'], right['scale'])):
            raise ValueError('neither and spatial are no longer equivalent trained baselines')
        old_a = json.loads((runtime / 'jobs' / a['run_id'] / 'development_case_metrics.json').read_text())['cases']
        old_b = json.loads((runtime / 'jobs' / b['run_id'] / 'development_case_metrics.json').read_text())['cases']
        if old_a != old_b:
            raise ValueError('neither and spatial predictions differed on Stage 6B cases')
    return by_key


def assessment_subset(source_manifest, output_dir, count=12, stride=3):
    manifest, paths = load_manifest(source_manifest)
    val = [e for e in manifest['images'] if e['category'] == 'juice_bottle' and e['split'] == 'validation/good']
    if len(val) != 54 or len(paths) != 389:
        raise ValueError('authorized 335/54 good-image manifest changed')
    assessment = val[18:]
    selected = assessment[::stride]
    if len(selected) != count:
        raise ValueError('predeclared assessment sample count changed')
    subset = dict(manifest, images=selected)
    path = output_dir / 'selected_good_manifest.json'
    path.write_text(json.dumps(subset, sort_keys=True, indent=2))
    sandbox_manifest, good_root = write_sandbox_manifest(path, output_dir)
    return sandbox_manifest, good_root, manifest_fingerprint(subset)


def summarize(reports, cfg):
    by_key = {(r['seed'], r['arm']): r for r in reports}
    if set(by_key) != {(seed, arm) for seed in cfg['seeds'] for arm in cfg['arms']}:
        raise ValueError('incomplete frozen comparison')
    keys = [r['image_id_sha256'] for r in by_key[(7, 'local')]['cases']]
    if any([c['image_id_sha256'] for c in r['cases']] != keys for r in reports):
        raise ValueError('different assessment images across checkpoints')
    summary = {'protocol': PROTOCOL_VERSION, 'evaluator': 'development-v2.0',
               'scope': 'diagnostic synthetic development; same Stage 6B assessment images; no official anomalies',
               'arms': cfg['arms'], 'seeds': cfg['seeds'], 'images_per_run': len(keys),
               'baseline_deduplication': 'spatial == neither in weights, scale and saved predictions; count once',
               'per_run': [], 'paired': {}}
    for r in reports:
        scores = {key: [c['scores'][key] for c in r['cases']] for key in r['cases'][0]['scores']}
        summary['per_run'].append({'run_id': r['run_id'], 'seed': r['seed'], 'arm': r['arm'],
            'macro_ap': r['macro_ap'], 'pooled_ap': r['pooled_ap'],
            'normal_false_positives': r['normal_false_positives']['false_positives'],
            'normal_images': len(r['cases']), 'normal_threshold': r['normal_threshold'],
            'control_alarms': r['control_alarms'],
            'mean_score_change': {k: float(np.mean(np.asarray(v) - scores['normal'])) for k, v in scores.items() if k != 'normal'},
            'median_cpu_latency_ms': r['median_cpu_latency_ms']})
    for arm in ('global', 'combined', 'neither'):
        summary['paired'][arm + '_minus_local'] = {}
        for measure in ('insertion_target', 'relocation_target', 'removal_source', 'relocation_source'):
            comparisons = []
            for seed in SEEDS:
                a = by_key[(seed, arm)]['cases']
                b = by_key[(seed, 'local')]['cases']
                comparisons.append(paired_image_difference(
                    [c['ap'][measure] for c in a], [c['ap'][measure] for c in b],
                    repeats=cfg['bootstrap_repeats'], seed=13007 + seed))
            summary['paired'][arm + '_minus_local'][measure] = {
                'across_seed_mean': float(np.mean([x['mean_difference'] for x in comparisons])),
                'between_seed_sd': float(np.std([x['mean_difference'] for x in comparisons], ddof=1)),
                'per_seed_image_intervals': comparisons}
    return summary


def execute(runtime: Path, ledger: Path, source_manifest: Path, output: Path):
    cfg = fixed_protocol()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise PermissionError('output symlink forbidden')
    runs = frozen_runs(StateStore(ledger, read_only=True), runtime)
    sandbox_manifest, good_root, subset_fp = assessment_subset(source_manifest, output)
    reports = []
    for seed in SEEDS:
        for arm in cfg['arms']:
            run = runs[(seed, arm)]
            destination = output / f'{arm}_seed{seed}'
            destination.mkdir(exist_ok=True)
            checkpoint_dir = runtime / 'jobs' / run['run_id'] / 'checkpoint'
            meta = json.loads((checkpoint_dir / 'checkpoint_manifest.json').read_text())
            threshold = float(json.loads(run['metrics_json'])['normal/threshold'])
            spec = {**cfg, 'run_id': run['run_id'], 'seed': seed, 'arm': arm,
                    'image_count': cfg['assessment_images'], 'subset_fingerprint': subset_fp,
                    'candidate_source_sha256': sha(MODEL),
                    'checkpoint_sha256': meta['checkpoint_sha256'], 'input_size': 128,
                    'normal_threshold': threshold, 'save_preview': seed == 7}
            spec_file = output / f'spec_{arm}_seed{seed}.json'
            spec_file.write_text(json.dumps(spec, sort_keys=True))
            cmd = sandbox_command(REPO, MODEL, good_root, destination,
                ['-m', 'scripts.stage7_eval', '--module', '/candidate/model.py',
                 '--checkpoint', '/inputs/checkpoint_manifest.json',
                 '--manifest', '/inputs/good_manifest.json', '--spec', '/inputs/config.json',
                 '--output', '/output'], use_gpu=False,
                input_files={'good_manifest.json': sandbox_manifest, 'config.json': spec_file,
                             'checkpoint_manifest.json': checkpoint_dir / 'checkpoint_manifest.json',
                             'model.pt': checkpoint_dir / 'model.pt'})
            try:
                done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                      timeout=cfg['timeout_seconds_per_checkpoint'],
                                      env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, check=False)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'frozen evaluation timed out: {arm}, seed {seed}') from exc
            if done.returncode:
                raise RuntimeError(f'frozen evaluation failed: {arm}, seed {seed}: {done.stderr[-1000:]}')
            reports.append(json.loads((destination / 'stage7_cases.json').read_text()))
    summary = summarize(reports, cfg)
    for row in summary['per_run']:
        historical = runs[(row['seed'], row['arm'])]
        recorded = json.loads(historical['metrics_json'])
        row['frozen_stage6_resources'] = {
            'parameters': int(recorded['inference/parameters']),
            'cuda_e2e_p95_ms': recorded['inference/e2e_ms_p95'],
            'peak_vram_mib': recorded['inference/peak_vram_mib'],
            'gpu_slot_seconds_original_training': historical['gpu_seconds'],
        }
    summary['protocol_config_sha256'] = sha(CONFIG)
    summary['subset_fingerprint'] = subset_fp
    summary['source_manifest_fingerprint'] = manifest_fingerprint(load_manifest(source_manifest)[0])
    summary['candidate_source_sha256'] = sha(MODEL)
    (output / 'stage7_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, default=REPO / 'results' / 'metrics' / 'stage7')
    args = p.parse_args()
    result = execute(args.runtime, args.ledger, args.manifest, args.output)
    print(json.dumps({'protocol': result['protocol'], 'runs': len(result['per_run']),
                      'images_per_run': result['images_per_run'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
