"""Versioned exploratory texture/shift diagnostic for frozen checkpoints only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.data_guard import load_manifest, manifest_fingerprint
from benchmark.stage7_controls import texture_noise_control, whole_image_controls
from benchmark.train_adapter import load_candidate, validate_prediction
from scripts.run_candidate import load_image

VERSION = 'stage7-texture-equivariance-v2'


def correlation(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float('nan')


def evaluate(module_path, checkpoint_path, manifest_path, spec_path, output):
    spec = json.loads(Path(spec_path).read_text())
    if (spec.get('protocol') != VERSION or spec.get('evaluator') != 'development-v2.0' or
        spec.get('device') != 'cpu' or spec.get('texture_sigma') != .02 or
        spec.get('shift_pixels') != 8 or spec.get('image_count') != 12):
        raise ValueError('texture/shift protocol changed')
    manifest, paths = load_manifest(manifest_path)
    if len(paths) != 12 or any(e['split'] != 'validation/good' or e['category'] != 'juice_bottle' for e in manifest['images']):
        raise ValueError('only selected good validation images permitted')
    if manifest_fingerprint(manifest) != spec['subset_fingerprint']:
        raise ValueError('image subset changed')
    if hashlib.sha256(Path(module_path).read_bytes()).hexdigest() != spec['candidate_source_sha256']:
        raise ValueError('frozen source changed')
    if json.loads(Path(checkpoint_path).read_text())['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    model = load_candidate(module_path).load(checkpoint_path, device='cpu')

    def predict(x):
        maps, scores = validate_prediction(model.predict(x[None]), (1, 3, 128, 128))
        return maps[0, 0], float(scores[0])

    rows = []
    for i, path in enumerate(paths):
        x = load_image(path, 128)
        normal_map, normal_score = predict(x)
        texture_map, texture_score = predict(texture_noise_control(x, spec['texture_seed'] + i, .02))
        shift_map, shift_score = predict(whole_image_controls(x, shift=8)['position_shift'])
        # Content moves down 8 and left 8. Compare corresponding interior
        # pixels on BOTH axes, excluding reflected borders.
        equivariance = correlation(normal_map[:-8, 8:], shift_map[8:, :-8])
        rows.append({'image_id_sha256': hashlib.sha256(manifest['images'][i]['path'].encode()).hexdigest(),
                     'normal_score': normal_score, 'texture_score': texture_score,
                     'shift_score': shift_score, 'texture_minus_normal': texture_score - normal_score,
                     'shift_minus_normal': shift_score - normal_score,
                     'texture_map_corr': correlation(normal_map, texture_map),
                     'shift_aligned_map_corr': equivariance,
                     'texture_alarm': bool(texture_score > spec['normal_threshold']),
                     'shift_alarm': bool(shift_score > spec['normal_threshold'])})
    result = {'protocol': VERSION, 'evaluator': 'development-v2.0', 'run_id': spec['run_id'],
              'seed': spec['seed'], 'arm': spec['arm'], 'subset_fingerprint': spec['subset_fingerprint'],
              'normal_threshold': spec['normal_threshold'], 'cases': rows}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'texture_cases.json').write_text(json.dumps(result, sort_keys=True, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('module', 'checkpoint', 'manifest', 'spec', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    evaluate(Path(args.module), Path(args.checkpoint), Path(args.manifest), Path(args.spec), Path(args.output))


if __name__ == '__main__':
    main()
