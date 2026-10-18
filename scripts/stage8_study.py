"""Trusted Stage 8 CPU comparison of frozen checkpoints on provisional cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmark.data_guard import load_manifest, manifest_fingerprint, sha256_file
from benchmark.metrics_v2 import paired_image_difference
from benchmark.semantic_validation import VERSION
from orchestrator.sandbox import command as sandbox_command, write_sandbox_manifest
from orchestrator.state_store import StateStore
from scripts.stage7_study import REPO, MODEL, SEEDS, frozen_runs, sha

CONFIG = REPO / 'configs' / 'stage8_semantic_dev.json'
ARMS = ('local', 'global', 'combined', 'neither')


def checked_cases(case_dir, good_manifest):
    case_dir = Path(case_dir).resolve(strict=True)
    if not case_dir.is_relative_to(REPO / 'results' / 'metrics'):
        raise PermissionError('case bundle must be in results/metrics')
    config = json.loads(CONFIG.read_text())
    good, _ = load_manifest(good_manifest)
    bundle = json.loads((case_dir / 'case_manifest.json').read_text())
    if (bundle.get('protocol') != VERSION or bundle.get('evaluator') != 'development-v2.0' or
        bundle.get('dataset_fingerprint') != manifest_fingerprint(good) or
        bundle.get('config_sha256') != sha(CONFIG) or
        bundle.get('generator_sha256') != sha(REPO / 'benchmark' / 'semantic_validation.py')):
        raise ValueError('semantic protocol, dataset or generator changed')
    if len(bundle['cases']) != 3 * len(config['source_images']):
        raise ValueError('incomplete semantic case set')
    by_source = {}
    for case in bundle['cases']:
        if case['source_path'] not in config['source_images'] or case['kind'] not in (
            'normal', 'putative_valid_edit', 'putative_identity_mismatch'):
            raise ValueError('unapproved source or case type')
        if case['review_status'] != ('manifest_good_normal' if case['kind'] == 'normal' else 'provisional'):
            raise ValueError('review status must be derived from separate reviews')
        by_source.setdefault(case['source_path'], set()).add(case['kind'])
        for key in ('image_path', 'edit_footprint_path'):
            path = (case_dir / case[key]).resolve(strict=True)
            if not path.is_relative_to(case_dir) or sha256_file(path) != case[key.replace('_path', '_sha256')]:
                raise ValueError('case artifact hash or path changed')
    if any(kinds != {'normal', 'putative_valid_edit', 'putative_identity_mismatch'} for kinds in by_source.values()):
        raise ValueError('unmatched source triplets')
    return bundle, config


def summarize(reports):
    by_key = {(r['seed'], r['arm']): r for r in reports}
    if set(by_key) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError('incomplete frozen arm grid')
    source_order = None
    summary = {'protocol': VERSION, 'evaluator': 'development-v2.0',
               'scope': 'three reused good sources; all edited examples provisional; no semantic ground truth',
               'seeds': list(SEEDS), 'arms': list(ARMS), 'per_run': [], 'paired': {},
               'verified_semantic_violations': 0, 'provisional_violations': 3,
               'provisional_valid_edits': 3, 'manifest_good_normals': 3}
    contrasts = {}
    for seed in SEEDS:
        for arm in ARMS:
            report = by_key[(seed, arm)]
            grouped = {}
            for case in report['cases']:
                grouped.setdefault(case['source_path'], {})[case['kind']] = case
            if any(set(kinds) != {'normal', 'putative_valid_edit', 'putative_identity_mismatch'} for kinds in grouped.values()):
                raise ValueError('unmatched case triplet')
            if source_order is None:
                source_order = sorted(grouped)
            if sorted(grouped) != source_order:
                raise ValueError('unmatched image identities across arms')
            delta = [grouped[p]['putative_identity_mismatch']['image_score'] -
                     grouped[p]['putative_valid_edit']['image_score'] for p in source_order]
            contrasts[(seed, arm)] = delta
            entry = {'seed': seed, 'arm': arm, 'run_id': report['run_id'],
                     'normal_alarms': sum(grouped[p]['normal']['alarm_at_frozen_good_threshold'] for p in source_order),
                     'provisional_valid_alarms': sum(grouped[p]['putative_valid_edit']['alarm_at_frozen_good_threshold'] for p in source_order),
                     'provisional_violation_alarms': sum(grouped[p]['putative_identity_mismatch']['alarm_at_frozen_good_threshold'] for p in source_order),
                     'violation_minus_valid_scores': delta,
                     'mean_violation_minus_valid_score': float(np.mean(delta)),
                     'provisional_violation_footprint_ap': float(np.mean([
                         grouped[p]['putative_identity_mismatch']['edit_footprint_ap_proxy'] for p in source_order])),
                     'verified_semantic_ap': None}
            summary['per_run'].append(entry)
    for arm in ('global', 'combined', 'neither'):
        comparisons = [paired_image_difference(contrasts[(seed, arm)], contrasts[(seed, 'local')],
                     seed=8100 + seed, repeats=2000) for seed in SEEDS]
        summary['paired'][arm + '_minus_local_violation_valid_contrast'] = {
            'across_seed_mean': float(np.mean([x['mean_difference'] for x in comparisons])),
            'per_seed_image_intervals': comparisons}
    return summary


def execute(runtime, ledger, good_manifest, case_dir, output):
    case_dir = Path(case_dir).resolve(strict=True)
    bundle, config = checked_cases(case_dir, good_manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise PermissionError('output symlink forbidden')
    runs = frozen_runs(StateStore(ledger, read_only=True), runtime)
    good, _ = load_manifest(good_manifest)
    subset = dict(good, images=[e for e in good['images'] if e['path'] in config['source_images']])
    if len(subset['images']) != 3:
        raise ValueError('missing selected good sources')
    subset_file = output / 'selected_good_manifest.json'
    subset_file.write_text(json.dumps(subset, sort_keys=True))
    sandbox_manifest, good_root = write_sandbox_manifest(subset_file, output)
    reports = []
    for seed in SEEDS:
        for arm in ARMS:
            run = runs[(seed, arm)]
            folder = output / f'{arm}_seed{seed}'
            folder.mkdir(exist_ok=True)
            checkpoint = runtime / 'jobs' / run['run_id'] / 'checkpoint'
            meta = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
            spec = {'protocol': VERSION, 'evaluator': 'development-v2.0', 'device': 'cpu',
                    'dataset_fingerprint': config['dataset_fingerprint'],
                    'run_id': run['run_id'], 'seed': seed, 'arm': arm,
                    'candidate_source_sha256': sha(MODEL),
                    'checkpoint_sha256': meta['checkpoint_sha256'],
                    'normal_threshold': float(json.loads(run['metrics_json'])['normal/threshold'])}
            spec_file = output / f'spec_{arm}_seed{seed}.json'
            spec_file.write_text(json.dumps(spec, sort_keys=True))
            cmd = sandbox_command(REPO, MODEL, good_root, folder,
                ['-m', 'scripts.stage8_eval', '--module', '/candidate/model.py',
                 '--checkpoint', '/inputs/checkpoint_manifest.json', '--manifest', '/inputs/good_manifest.json',
                 '--spec', '/inputs/config.json', '--cases', '/cases', '--output', '/output'],
                use_gpu=False, case_dir=case_dir,
                input_files={'good_manifest.json': sandbox_manifest, 'config.json': spec_file,
                             'checkpoint_manifest.json': checkpoint / 'checkpoint_manifest.json',
                             'model.pt': checkpoint / 'model.pt'})
            try:
                done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=75,
                                      env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, check=False)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'Stage 8 CPU inference timed out: {arm}, seed {seed}') from exc
            if done.returncode:
                raise RuntimeError(f'Stage 8 CPU inference failed: {arm}, seed {seed}: {done.stderr[-1000:]}')
            reports.append(json.loads((folder / 'stage8_case_scores.json').read_text()))
    summary = summarize(reports)
    summary['dataset_fingerprint'] = config['dataset_fingerprint']
    summary['config_sha256'] = sha(CONFIG)
    summary['case_bundle_sha256'] = sha(case_dir / 'case_manifest.json')
    (output / 'stage8_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=REPO / 'results/metrics/stage8_semantic')
    p.add_argument('--output', type=Path, default=REPO / 'results/metrics/stage8_eval')
    args = p.parse_args()
    result = execute(args.runtime, args.ledger, args.manifest, args.cases, args.output)
    print(json.dumps({'protocol': VERSION, 'frozen_runs': len(result['per_run']),
                      'verified_semantic_violations': result['verified_semantic_violations']}))


if __name__ == '__main__':
    main()
