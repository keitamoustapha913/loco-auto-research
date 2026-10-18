"""Versioned, good-only semantic-development examples and review gate.

Draft color edits are *provisional*. Pixel edits or a rule description cannot
establish product validity. Two independent human reviews are required before
any semantic localization metric may be reported as verified development data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark.data_guard import load_manifest, manifest_fingerprint, sha256_file

VERSION = 'semantic-development-v1.3'
KINDS = ('normal', 'putative_valid_edit', 'putative_identity_mismatch')
VERDICTS = ('valid', 'violation', 'indeterminate')
TARGET_POLICIES = {
    'missing_component': 'Independently annotated expected-component region from good-reference geometry; never the erase footprint.',
    'extra_component': 'Visible extra-component instance pixels; insertion seam is scored separately.',
    'incorrect_identity': 'Independently annotated incorrect component or predeclared pair of implicated regions.',
    'incorrect_position': 'Displaced component pixels and expected slot reported separately; neither may be silently substituted.',
    'invalid_relationship': 'Independently identified participating component regions; attribution ambiguity is explicit.',
}


def mask_from_polygon(size, vertices):
    width, height = size
    if (width <= 0 or height <= 0 or len(vertices) < 3 or
        any(len(p) != 2 or not 0 <= p[0] < width or not 0 <= p[1] < height for p in vertices)):
        raise ValueError('polygon must be inside the original image')
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in vertices], fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def box_region(size, xyxy):
    x0, y0, x1, y1 = xyxy
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError('region outside original image')
    return mask_from_polygon(size, [(x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)])


def draft_liquid_mask(rgb, polygon, red_ratio, excluded_regions=(), component_region=None):
    arr = np.asarray(rgb, dtype=np.float32) / 255
    if arr.ndim != 3 or arr.shape[2] != 3 or not 1 < red_ratio < 2:
        raise ValueError('expected RGB original and bounded red ratio')
    roi = mask_from_polygon((arr.shape[1], arr.shape[0]), polygon)
    # This is an image-derived edit region, not a semantic segmentation label.
    red = (arr[:, :, 0] > red_ratio * arr[:, :, 1]) & (arr[:, :, 0] > red_ratio * arr[:, :, 2])
    mask = roi & red & (arr[:, :, 0] > .08)
    for region in excluded_regions:
        sticker_color = (arr[:, :, 0] > .6) & (arr[:, :, 1] > .45)
        mask &= ~(box_region((arr.shape[1], arr.shape[0]), region) & sticker_color)
    if component_region is not None:
        mask &= ~box_region((arr.shape[1], arr.shape[0]), component_region)
    if not 1000 <= int(mask.sum()) <= .45 * mask.size:
        raise ValueError('draft liquid selection implausibly small or large')
    return mask


def recolor_red_liquid(rgb, binary_mask, strength):
    if not 0 <= strength <= .8:
        raise ValueError('unapproved edit strength')
    arr = np.asarray(rgb, dtype=np.float32) / 255
    if binary_mask.shape != arr.shape[:2] or binary_mask.dtype != bool:
        raise ValueError('unaligned binary region mask')
    # Same smooth renderer for both proposed valid and violation edits. Their
    # strengths differ, so artifact matching is incomplete and must be stated.
    alpha = np.asarray(Image.fromarray(binary_mask.astype(np.uint8) * 255).filter(
        ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) / 255
    alpha *= binary_mask
    alpha *= strength
    altered = arr.copy()
    altered[:, :, 1] += alpha * (arr[:, :, 0] - arr[:, :, 1])
    altered[:, :, 2] += alpha * .12 * (arr[:, :, 0] - arr[:, :, 2])
    return Image.fromarray(np.round(np.clip(altered, 0, 1) * 255).astype(np.uint8)), alpha > .01


def nearest_mask(mask, size):
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('binary source-coordinate mask required')
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST))
    return resized > 0


def restore_map(score_map, size):
    array = np.asarray(score_map, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('finite model-space map required')
    return np.asarray(Image.fromarray(array, mode='F').resize(size, Image.Resampling.BILINEAR), np.float32)


def case_digest(case):
    immutable = {k: v for k, v in case.items() if k not in ('review_status', 'reviews')}
    return hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def independent_review_status(case, reviews):
    """Certify intent only with two distinct reviewers of this exact case hash."""
    if case['kind'] == 'normal':
        return 'manifest_good_normal'
    expected = 'valid' if case['kind'] == 'putative_valid_edit' else 'violation'
    relevant = [r for r in reviews if r.get('sample_id') == case['sample_id']]
    if not relevant:
        return 'provisional'
    if any(r.get('case_sha256') != case_digest(case) for r in relevant):
        raise ValueError('invalid or stale review')
    if any(r.get('source_inspected') is not True for r in relevant):
        return 'provisional'
    if any(r.get('verdict') not in VERDICTS or
           not isinstance(r.get('reviewer_id'), str) or not r['reviewer_id'].strip() or
           not isinstance(r.get('reason'), str) or len(r['reason'].strip()) < 10 or
           r.get('blind_to_scores') is not True
           for r in relevant):
        raise ValueError('invalid or stale review')
    if len({r['reviewer_id'] for r in relevant}) != len(relevant):
        raise ValueError('reviews are not independent identities')
    if len(relevant) < 2 or any(r['verdict'] == 'indeterminate' for r in relevant):
        return 'provisional'
    if any(r['verdict'] != expected or r.get('target_region_accepted') is not True for r in relevant):
        return 'rejected'
    if expected == 'violation':
        return 'reviewed_violation_localized' if case.get('localization_target') else 'reviewed_violation_logic_only'
    return 'reviewed_valid'


def validate_localization_target(case, case_dir):
    """Separate independently drawn semantic target from generator footprint."""
    target = case.get('localization_target')
    if not target:
        raise ValueError('independent semantic target absent')
    if target.get('violation_type') not in TARGET_POLICIES or target.get('provenance') != 'independent_human_annotation':
        raise ValueError('unsupported target semantics or provenance')
    if target['violation_type'] == 'missing_component' and not target.get('expected_region_evidence'):
        raise ValueError('missing component requires independently justified expected region')
    relative = Path(target.get('mask_path', ''))
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'semantic_targets':
        raise ValueError('semantic mask path must stay in semantic_targets/')
    root = Path(case_dir).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha256_file(path) != target.get('mask_sha256'):
        raise ValueError('semantic target hash or path changed')
    mask = np.asarray(Image.open(path).convert('L'))
    if mask.shape != (1600, 800) or not np.isin(mask, (0, 255)).all() or not np.any(mask == 255):
        raise ValueError('expected nonempty binary original-coordinate target')
    return mask == 255


def build_cases(good_manifest, config_path, output):
    config = json.loads(Path(config_path).read_text())
    if (config.get('protocol') != VERSION or config.get('official_test_access') is not False or
        config.get('independent_validation_required') is not True or config.get('reviewers_required') != 2 or
        config.get('input_size') != [800, 1600] or config.get('evaluator') != 'development-v2.0'):
        raise ValueError('semantic protocol changed')
    manifest, paths = load_manifest(good_manifest)
    if manifest_fingerprint(manifest) != config['dataset_fingerprint']:
        raise ValueError('authorized good-only dataset fingerprint changed')
    by_path = {entry['path']: (entry, path) for entry, path in zip(manifest['images'], paths)}
    output = Path(output)
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise FileExistsError('semantic output must be a new empty real directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    (output / 'masks').mkdir()
    cases = []
    for index, relative in enumerate(config['source_images']):
        if relative not in by_path or by_path[relative][0]['split'] != 'validation/good' or by_path[relative][0]['category'] != 'juice_bottle':
            raise ValueError('only explicitly authorized good validation sources')
        entry, source = by_path[relative]
        original = Image.open(source).convert('RGB')
        if list(original.size) != config['input_size']:
            raise ValueError('unexpected original geometry')
        if not config.get('excluded_label_regions_xyxy'):
            raise ValueError('printed-label exclusion regions required')
        icon = box_region(original.size, config['pictogram_region_xyxy'])
        region = draft_liquid_mask(original, config['liquid_roi_polygon_xy'], config['red_min_ratio'],
                                   config['excluded_label_regions_xyxy'], config['pictogram_region_xyxy'])
        mask_files = {}
        for key, array in (('liquid_draft', region), ('pictogram_draft', icon)):
            mask_name = f'{index}_{key}.png'
            Image.fromarray(array.astype(np.uint8) * 255).save(output / 'masks' / mask_name)
            mask_files[key] = 'masks/' + mask_name
        variants = [('normal', original, np.zeros_like(region)),
                    ('putative_valid_edit', *recolor_red_liquid(original, region, config['valid_edit_strength'])),
                    ('putative_identity_mismatch', *recolor_red_liquid(original, region, config['violation_edit_strength']))]
        for kind, image, footprint in variants:
            sample_id = f'juice_bottle_{index:02d}_{kind}'
            image_rel = f'images/{sample_id}.png'
            image.save(output / image_rel)
            footprint_rel = f'masks/{sample_id}_footprint.png'
            Image.fromarray(footprint.astype(np.uint8) * 255).save(output / footprint_rel)
            case = {'sample_id': sample_id, 'kind': kind, 'source_path': relative,
                    'source_sha256': entry['sha256'], 'image_path': image_rel,
                    'image_sha256': sha256_file(output / image_rel),
                    'edit_footprint_path': footprint_rel,
                    'edit_footprint_sha256': sha256_file(output / footprint_rel),
                    'candidate_rule_id': config['draft_rule_id'],
                    'intended_validity': 'normal' if kind == 'normal' else ('valid' if kind == 'putative_valid_edit' else 'violation'),
                    'expected_violation': None if kind != 'putative_identity_mismatch' else
                    'Proposed pictogram/liquid identity mismatch after red liquid is recolored; requires independent review.',
                    'transform': {'name': 'identity' if kind == 'normal' else 'recolor_red_liquid_v1',
                                  'strength': 0 if kind == 'normal' else config['valid_edit_strength'] if kind == 'putative_valid_edit' else config['violation_edit_strength']},
                    'semantic_region_paths': mask_files,
                    'localization_semantics': 'ambiguous_pair_regions; pictogram and liquid are candidate implicated regions, neither is a certified pixel GT',
                    'review_status': 'manifest_good_normal' if kind == 'normal' else 'provisional'}
            cases.append(case)
    result = {'protocol': VERSION, 'evaluator': config['evaluator'],
              'dataset_fingerprint': config['dataset_fingerprint'],
              'config_sha256': sha256_file(Path(config_path)),
              'generator_sha256': sha256_file(Path(__file__)),
              'source_split': 'validation/good', 'independent_holdout': False,
              'cases': cases}
    (output / 'case_manifest.json').write_text(json.dumps(result, sort_keys=True, indent=2))
    return result
