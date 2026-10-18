"""Frozen, no-network candidate inference on versioned semantic-development cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.data_guard import load_manifest, sha256_file
from benchmark.metrics_v2 import EVALUATOR_VERSION, pixel_ap_report
from benchmark.semantic_validation import VERSION, nearest_mask, validate_localization_target, independent_review_status
from benchmark.train_adapter import load_candidate, validate_prediction
from scripts.run_candidate import load_image


def evaluate(module, checkpoint, manifest_path, spec_path, case_dir, output):
    spec = json.loads(Path(spec_path).read_text())
    if spec.get('protocol') != VERSION or spec.get('evaluator') != EVALUATOR_VERSION or spec.get('device') != 'cpu':
        raise ValueError('Stage 8 evaluator is frozen CPU-only development-v2.0')
    good, paths = load_manifest(manifest_path)
    authorized = {e['path']: e['sha256'] for e in good['images'] if e['category'] == 'juice_bottle' and e['split'] == 'validation/good'}
    case_dir = Path(case_dir).resolve(strict=True)
    bundle = json.loads((case_dir / 'case_manifest.json').read_text())
    if bundle.get('protocol') != VERSION or bundle.get('dataset_fingerprint') != spec['dataset_fingerprint']:
        raise ValueError('case bundle differs from frozen protocol')
    if hashlib.sha256(Path(module).read_bytes()).hexdigest() != spec['candidate_source_sha256']:
        raise ValueError('candidate source changed')
    if json.loads(Path(checkpoint).read_text())['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    model = load_candidate(module).load(checkpoint, device='cpu')
    review_file = case_dir / 'reviews.json'
    reviews = json.loads(review_file.read_text()) if review_file.is_file() else []
    if not isinstance(reviews, list):
        raise ValueError('reviews must be a list')
    rows = []
    for case in bundle['cases']:
        if case.get('review_status') != ('manifest_good_normal' if case['kind'] == 'normal' else 'provisional'):
            raise ValueError('generated case status cannot be edited manually')
        review_status = independent_review_status(case, reviews)
        if case['source_path'] not in authorized or case['source_sha256'] != authorized[case['source_path']]:
            raise PermissionError('case source not in authorized good validation manifest')
        image_path = (case_dir / case['image_path']).resolve(strict=True)
        footprint_path = (case_dir / case['edit_footprint_path']).resolve(strict=True)
        if (not image_path.is_relative_to(case_dir / 'images') or
            not footprint_path.is_relative_to(case_dir / 'masks') or
            sha256_file(image_path) != case['image_sha256'] or
            sha256_file(footprint_path) != case['edit_footprint_sha256']):
            raise ValueError('case image/footprint path or hash changed')
        with Image.open(image_path) as im:
            if im.size != (800, 1600):
                raise ValueError('source-coordinate case geometry changed')
        x = load_image(image_path, 128)
        maps, scores = validate_prediction(model.predict(x[None]), (1, 3, 128, 128))
        score_map = maps[0, 0]
        footprint = np.asarray(Image.open(footprint_path).convert('L'))
        if footprint.shape != (1600, 800) or not np.isin(footprint, (0, 255)).all():
            raise ValueError('case footprint must be binary and source-aligned')
        footprint_ap = None
        if case['kind'] != 'normal':
            small_footprint = nearest_mask(footprint == 255, (128, 128))
            footprint_ap = pixel_ap_report(score_map[None], small_footprint[None])['per_image'][0]
        semantic_ap = None
        if review_status == 'reviewed_violation_localized':
            semantic_mask = validate_localization_target(case, case_dir)
            semantic_ap = pixel_ap_report(score_map[None], nearest_mask(semantic_mask, (128, 128))[None])['per_image'][0]
        rows.append({'sample_id': case['sample_id'], 'source_path': case['source_path'],
                     'kind': case['kind'], 'review_status': review_status,
                     'image_score': float(scores[0]),
                     'alarm_at_frozen_good_threshold': bool(scores[0] > spec['normal_threshold']),
                     'edit_footprint_ap_proxy': footprint_ap,
                     'semantic_region_ap_verified_only': semantic_ap})
    result = {'protocol': VERSION, 'evaluator': EVALUATOR_VERSION,
              'run_id': spec['run_id'], 'seed': spec['seed'], 'arm': spec['arm'],
              'normal_threshold': spec['normal_threshold'], 'cases': rows}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'stage8_case_scores.json').write_text(json.dumps(result, sort_keys=True, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('module', 'checkpoint', 'manifest', 'spec', 'cases', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    evaluate(Path(a.module), Path(a.checkpoint), Path(a.manifest), Path(a.spec), Path(a.cases), Path(a.output))


if __name__ == '__main__':
    main()
