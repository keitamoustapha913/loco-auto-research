"""Frozen checkpoint evaluation entry point; execute only inside candidate Bubblewrap.

No training, CUDA initialization or new threshold calibration occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmark.data_guard import load_manifest, manifest_fingerprint
from benchmark.metrics_v2 import EVALUATOR_VERSION, image_false_positive_report, pixel_ap_report
from benchmark.stage7_controls import PROTOCOL_VERSION, patch_controls, whole_image_controls
from benchmark.train_adapter import load_candidate, validate_prediction
from scripts.run_candidate import load_image, restore_anomaly_map


def evaluate(module_path: Path, checkpoint_path: Path, manifest_path: Path, spec_path: Path, output: Path):
    spec = json.loads(spec_path.read_text())
    if spec.get('protocol') != PROTOCOL_VERSION or spec.get('evaluator') != EVALUATOR_VERSION:
        raise ValueError('evaluation protocol/version mismatch')
    if spec.get('device') != 'cpu' or not 1 <= int(spec['image_count']) <= 36:
        raise ValueError('Stage 7 is bounded CPU inference only')
    manifest, paths = load_manifest(manifest_path)
    if len(paths) != spec['image_count'] or any(e['category'] != 'juice_bottle' or e['split'] != 'validation/good' for e in manifest['images']):
        raise ValueError('only preselected good validation images may be evaluated')
    if manifest_fingerprint(manifest) != spec['subset_fingerprint']:
        raise ValueError('assessment subset changed')
    if hashlib.sha256(module_path.read_bytes()).hexdigest() != spec['candidate_source_sha256']:
        raise ValueError('frozen candidate source changed')
    checkpoint_meta = json.loads(checkpoint_path.read_text())
    if checkpoint_meta['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('frozen checkpoint mismatch')
    module = load_candidate(module_path)
    model = module.load(checkpoint_path, device='cpu')
    size = int(spec['input_size'])
    if size != 128:
        raise ValueError('Stage 6B preprocessing must be preserved')

    def predict(image):
        start = time.perf_counter()
        response = model.predict(image[None])
        maps, scores = validate_prediction(response, (1, 3, size, size))
        return maps[0, 0], float(scores[0]), (time.perf_counter() - start) * 1000

    cases = []
    pooled = {name: ([], []) for name in ('insertion_target', 'relocation_target', 'removal_source', 'relocation_source')}
    previews = []
    for i, path in enumerate(paths):
        x = load_image(path, size)
        donor = load_image(paths[(i + 1) % len(paths)], size)
        whole = whole_image_controls(x)
        edits = patch_controls(x, donor, seed=int(spec['edit_seed']) + i)
        variants = {'normal': x, **whole, **{k: v for k, v in edits.items() if k not in ('source_mask', 'target_mask')}}
        maps, scores, latencies = {}, {}, {}
        for name, image in variants.items():
            maps[name], scores[name], latencies[name] = predict(image)
        target = edits['target_mask']
        source = edits['source_mask']
        ap = {
            'insertion_target': pixel_ap_report(maps['insertion'][None], target[None])['per_image'][0],
            'relocation_target': pixel_ap_report(maps['relocation_proxy'][None], target[None])['per_image'][0],
            'removal_source': pixel_ap_report(maps['removal_proxy'][None], source[None])['per_image'][0],
            'relocation_source': pixel_ap_report(maps['relocation_proxy'][None], source[None])['per_image'][0],
        }
        for key, score_name, mask in (
            ('insertion_target', 'insertion', target),
            ('relocation_target', 'relocation_proxy', target),
            ('removal_source', 'removal_proxy', source),
            ('relocation_source', 'relocation_proxy', source)):
            pooled[key][0].append(maps[score_name])
            pooled[key][1].append(mask)
        cases.append({'image_id_sha256': hashlib.sha256(manifest['images'][i]['path'].encode()).hexdigest(),
                      'image_index': i, 'scores': scores, 'latency_ms': latencies, 'ap': ap,
                      'target_pixels': int(target.sum()), 'source_pixels': int(source.sum()),
                      'normal_alarm': bool(scores['normal'] > spec['normal_threshold'])})
        if i == 0:
            with Image.open(path) as original:
                original_size = original.size
                original_rgb = original.convert('RGB').copy()
            previews.append((original_rgb, edits['insertion'], maps['insertion'], target, original_size))
    report = {'protocol': PROTOCOL_VERSION, 'evaluator': EVALUATOR_VERSION,
              'mask_semantics': 'proxy edit footprints; source removal and target insertion reported separately',
              'candidate_source_sha256': spec['candidate_source_sha256'],
              'checkpoint_sha256': spec['checkpoint_sha256'], 'run_id': spec['run_id'],
              'seed': spec['seed'], 'arm': spec['arm'], 'subset_fingerprint': spec['subset_fingerprint'],
              'normal_threshold': spec['normal_threshold'], 'cases': cases}
    for generator, (maps, masks) in pooled.items():
        ap_report = pixel_ap_report(np.stack(maps), np.stack(masks))
        report.setdefault('macro_ap', {})[generator] = ap_report['macro_positive_images']
        report.setdefault('pooled_ap', {})[generator] = ap_report['pooled']
    report['normal_false_positives'] = image_false_positive_report(
        [c['scores']['normal'] for c in cases], spec['normal_threshold'])
    report['control_alarms'] = {name: image_false_positive_report(
        [c['scores'][name] for c in cases], spec['normal_threshold'])['false_positives']
        for name in ('position_shift', 'illumination', 'donor_control', 'appearance_control')}
    report['median_cpu_latency_ms'] = float(np.median([v for c in cases for v in c['latency_ms'].values()]))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'stage7_cases.json').write_text(json.dumps(report, sort_keys=True, indent=2))
    if spec.get('save_preview') and previews:
        original, edited, amap, mask, dimensions = previews[0]
        original.save(output / 'original.png')
        restored = restore_anomaly_map(amap, dimensions)
        lo, hi = np.percentile(restored, [2, 98])
        norm = np.clip((restored - lo) / max(float(hi - lo), 1e-8), 0, 1)
        edited_rgb = Image.fromarray((edited.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)).resize(dimensions, Image.Resampling.BILINEAR)
        color = np.stack((norm * 255, np.zeros_like(norm), (1 - norm) * 80), -1)
        overlay = .62 * np.asarray(edited_rgb, np.float32) + .38 * color
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(output / 'insertion_overlay.png')
        Image.fromarray((np.asarray(Image.fromarray(mask * 255).resize(dimensions, Image.Resampling.NEAREST)) > 0).astype(np.uint8) * 255).save(output / 'insertion_mask.png')
        report['preview_original_size'] = list(dimensions)
        (output / 'stage7_cases.json').write_text(json.dumps(report, sort_keys=True, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ('module', 'checkpoint', 'manifest', 'spec', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    evaluate(Path(args.module), Path(args.checkpoint), Path(args.manifest), Path(args.spec), Path(args.output))


if __name__ == '__main__':
    main()
