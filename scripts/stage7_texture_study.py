"""Separate, exploratory Stage 7 texture and shift study; no GPU or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmark.metrics_v2 import paired_image_difference
from orchestrator.sandbox import command as sandbox_command
from orchestrator.state_store import StateStore
from scripts.stage7_study import REPO, MODEL, SEEDS, assessment_subset, frozen_runs, sha

CONFIG = REPO / 'configs' / 'stage7_texture_eval.json'


def fixed_texture_protocol():
    cfg = json.loads(CONFIG.read_text())
    if cfg != {'protocol': 'stage7-texture-equivariance-v2',
               'parent_protocol': 'stage7-frozen-diagnostics-v1',
               'evaluator': 'development-v2.0', 'device': 'cpu',
               'assessment_images': 12, 'texture_seed': 22013,
               'texture_sigma': .02, 'shift_pixels': 8,
               'no_training': True, 'official_test_access': False}:
        raise ValueError('fixed texture protocol changed')
    return cfg


def summarize_texture(reports):
    by_key = {(r['seed'], r['arm']): r for r in reports}
    arms = ('local', 'global', 'combined', 'neither')
    if set(by_key) != {(seed, arm) for seed in SEEDS for arm in arms}:
        raise ValueError('incomplete texture study')
    ids = [c['image_id_sha256'] for c in by_key[(7, 'local')]['cases']]
    if len(ids) != 12 or any([c['image_id_sha256'] for c in r['cases']] != ids for r in reports):
        raise ValueError('unmatched texture assessment images')
    summary = {'protocol': 'stage7-texture-equivariance-v2', 'evaluator': 'development-v2.0',
               'scope': 'exploratory normal-image rendering and aligned-map sensitivity; reused Stage 6B assessment images',
               'images_per_run': 12, 'arms': list(arms), 'seeds': list(SEEDS), 'per_run': [], 'paired': {}}
    for r in reports:
        row = {'run_id': r['run_id'], 'arm': r['arm'], 'seed': r['seed']}
        for metric in ('texture_minus_normal', 'shift_minus_normal', 'texture_map_corr', 'shift_aligned_map_corr'):
            vals = [c[metric] for c in r['cases'] if np.isfinite(c[metric])]
            row['mean_' + metric] = float(np.mean(vals)) if vals else None
        row['texture_alarms'] = sum(c['texture_alarm'] for c in r['cases'])
        row['shift_alarms'] = sum(c['shift_alarm'] for c in r['cases'])
        summary['per_run'].append(row)
    for arm in ('global', 'combined', 'neither'):
        comparison = []
        for seed in SEEDS:
            a = by_key[(seed, arm)]['cases']; b = by_key[(seed, 'local')]['cases']
            comparison.append(paired_image_difference(
                [c['texture_minus_normal'] for c in a],
                [c['texture_minus_normal'] for c in b], seed=22013 + seed))
        summary['paired'][arm + '_minus_local_texture_response'] = {
            'across_seed_mean': float(np.mean([c['mean_difference'] for c in comparison])),
            'per_seed_image_intervals': comparison}
    return summary


def execute(runtime, ledger, manifest, output):
    cfg = fixed_texture_protocol()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise PermissionError('output symlink forbidden')
    prior = json.loads((REPO / 'results/metrics/stage7/stage7_summary.json').read_text())
    runs = frozen_runs(StateStore(ledger, read_only=True), runtime)
    selected, good_root, fingerprint = assessment_subset(manifest, output)
    if prior['subset_fingerprint'] != fingerprint:
        raise ValueError('texture extension has different assessment images')
    reports = []
    for seed in SEEDS:
        for arm in ('local', 'global', 'combined', 'neither'):
            run = runs[(seed, arm)]
            folder = output / f'{arm}_seed{seed}'
            folder.mkdir(exist_ok=True)
            ckpt_dir = runtime / 'jobs' / run['run_id'] / 'checkpoint'
            ckpt = json.loads((ckpt_dir / 'checkpoint_manifest.json').read_text())
            spec = {**cfg, 'seed': seed, 'arm': arm, 'run_id': run['run_id'],
                    'image_count': 12, 'subset_fingerprint': fingerprint,
                    'candidate_source_sha256': sha(MODEL),
                    'checkpoint_sha256': ckpt['checkpoint_sha256'],
                    'normal_threshold': float(json.loads(run['metrics_json'])['normal/threshold'])}
            spec_file = output / f'spec_{arm}_seed{seed}.json'
            spec_file.write_text(json.dumps(spec, sort_keys=True))
            cmd = sandbox_command(REPO, MODEL, good_root, folder,
                ['-m', 'scripts.stage7_texture_eval', '--module', '/candidate/model.py',
                 '--checkpoint', '/inputs/checkpoint_manifest.json',
                 '--manifest', '/inputs/good_manifest.json', '--spec', '/inputs/config.json',
                 '--output', '/output'], use_gpu=False,
                input_files={'good_manifest.json': selected, 'config.json': spec_file,
                             'checkpoint_manifest.json': ckpt_dir / 'checkpoint_manifest.json',
                             'model.pt': ckpt_dir / 'model.pt'})
            try:
                done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=90,
                                      env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, check=False)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'texture evaluation timed out for {arm}, seed {seed}') from exc
            if done.returncode:
                raise RuntimeError(f'texture evaluation failed for {arm}, seed {seed}: {done.stderr[-1000:]}')
            reports.append(json.loads((folder / 'texture_cases.json').read_text()))
    summary = summarize_texture(reports)
    summary['subset_fingerprint'] = fingerprint
    summary['protocol_config_sha256'] = sha(CONFIG)
    (output / 'texture_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--ledger', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', default=REPO / 'results/metrics/stage7_texture', type=Path)
    args = p.parse_args()
    result = execute(args.runtime, args.ledger, args.manifest, args.output)
    print(json.dumps({'protocol': result['protocol'], 'runs': len(result['per_run']),
                      'images_per_run': result['images_per_run']}))


if __name__ == '__main__':
    main()
