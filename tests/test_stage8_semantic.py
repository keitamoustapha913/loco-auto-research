import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from benchmark.data_guard import build_manifest, manifest_fingerprint, sha256_file
from benchmark.semantic_validation import (
    VERSION, TARGET_POLICIES, box_region, build_cases, case_digest,
    draft_liquid_mask, independent_review_status, mask_from_polygon,
    nearest_mask, recolor_red_liquid, restore_map, validate_localization_target,
)
from scripts.stage8_study import summarize
from scripts.stage8_review import prepare, merge
from orchestrator import sandbox


def test_source_coordinate_masks_and_map_alignment():
    mask = mask_from_polygon((800, 1600), [(100, 200), (300, 200), (300, 400), (100, 400)])
    assert mask.shape == (1600, 800) and mask[250, 150] and not mask[150, 250]
    small = nearest_mask(mask, (128, 128))
    assert small.shape == (128, 128) and small.dtype == bool
    response = restore_map(np.ones((128, 128), np.float32), (800, 1600))
    assert response.shape == mask.shape and np.isfinite(response).all()
    with pytest.raises(ValueError):
        mask_from_polygon((800, 1600), [(0, 0), (801, 1), (1, 2)])
    with pytest.raises(ValueError):
        restore_map(np.full((2, 2), np.nan), (800, 1600))


def test_draft_recolor_does_not_change_pictogram_pixels():
    rgb = np.zeros((160, 80, 3), np.uint8)
    rgb[:] = (5, 5, 5)
    rgb[30:130, 15:60] = (70, 20, 15)
    rgb[70:100, 25:55] = (230, 170, 60)
    rgb[77:92, 34:48] = (150, 35, 22)
    image = Image.fromarray(rgb)
    icon = (33, 76, 49, 94)
    mask = draft_liquid_mask(image, [(10, 10), (70, 10), (70, 150), (10, 150)],
                             1.12, [(20, 65, 60, 105)], icon)
    assert not np.any(mask & box_region(image.size, icon))
    valid, footprint_a = recolor_red_liquid(image, mask, .08)
    violation, footprint_b = recolor_red_liquid(image, mask, .78)
    assert np.array_equal(footprint_a, footprint_b)
    assert np.array_equal(np.asarray(valid)[76:94, 33:49], rgb[76:94, 33:49])
    assert np.array_equal(np.asarray(violation)[76:94, 33:49], rgb[76:94, 33:49])
    assert not np.array_equal(np.asarray(valid), np.asarray(violation))


def test_independent_review_requires_distinct_people_exact_case_and_target_acceptance():
    case = {'sample_id': 's1', 'kind': 'putative_identity_mismatch', 'review_status': 'provisional',
            'source_sha256': 'abc', 'image_sha256': 'def'}
    digest = case_digest(case)
    a = {'sample_id': 's1', 'reviewer_id': 'alice', 'case_sha256': digest,
         'verdict': 'violation', 'target_region_accepted': True,
         'reason': 'fruit and liquid mismatch', 'source_inspected': True,
         'blind_to_scores': True}
    b = dict(a, reviewer_id='bob')
    assert independent_review_status(case, []) == 'provisional'
    assert independent_review_status(case, [a]) == 'provisional'
    assert independent_review_status(case, [a, b]) == 'reviewed_violation_logic_only'
    with pytest.raises(ValueError, match='independent'):
        independent_review_status(case, [a, a])
    with pytest.raises(ValueError, match='stale'):
        independent_review_status(case, [a, dict(b, case_sha256='old')])
    assert independent_review_status(case, [a, dict(b, verdict='valid')]) == 'rejected'
    assert independent_review_status(case, [a, dict(b, target_region_accepted=False)]) == 'rejected'
    case['localization_target'] = {'violation_type': 'incorrect_identity'}
    assert case_digest(case) != digest


def test_target_semantics_require_independent_original_coordinate_mask(tmp_path):
    assert set(TARGET_POLICIES) == {'missing_component', 'extra_component',
        'incorrect_identity', 'incorrect_position', 'invalid_relationship'}
    root = tmp_path / 'cases'; (root / 'semantic_targets').mkdir(parents=True)
    path = root / 'semantic_targets' / 'target.png'
    array = np.zeros((1600, 800), np.uint8); array[100:130, 200:220] = 255
    Image.fromarray(array).save(path)
    target = {'violation_type': 'missing_component', 'provenance': 'independent_human_annotation',
              'mask_path': 'semantic_targets/target.png', 'mask_sha256': sha256_file(path)}
    case = {'localization_target': target}
    with pytest.raises(ValueError, match='expected region'):
        validate_localization_target(case, root)
    target['expected_region_evidence'] = 'operator-approved slot diagram'
    assert validate_localization_target(case, root).sum() == 600
    target['mask_path'] = '../target.png'
    with pytest.raises(ValueError, match='semantic mask path'):
        validate_localization_target(case, root)


def test_build_cases_only_uses_good_manifest_and_keeps_edits_provisional(tmp_path):
    good = tmp_path / 'good'
    folder = good / 'juice_bottle' / 'validation' / 'good'; folder.mkdir(parents=True)
    image = np.zeros((1600, 800, 3), np.uint8)
    image[350:1450, 220:610] = (75, 20, 15)
    image[720:1020, 275:585] = (240, 190, 80)
    image[1215:1420, 275:590] = (240, 190, 80)
    image[780:920, 350:510] = (155, 35, 25)
    Image.fromarray(image).save(folder / '020.png')
    manifest = build_manifest(good)
    manifest_file = tmp_path / 'manifest.json'; manifest_file.write_text(json.dumps(manifest))
    source = 'juice_bottle/validation/good/020.png'
    config = {'protocol': VERSION, 'category': 'juice_bottle',
              'dataset_fingerprint': manifest_fingerprint(manifest), 'source_images': [source],
              'draft_rule_id': 'fixture_relation', 'liquid_roi_polygon_xy': [[200, 300], [620, 300], [620, 1490], [200, 1490]],
              'excluded_label_regions_xyxy': [[270, 710, 585, 1030], [270, 1215, 590, 1420]],
              'pictogram_region_xyxy': [350, 760, 510, 980], 'red_min_ratio': 1.12,
              'valid_edit_strength': .08, 'violation_edit_strength': .78,
              'input_size': [800, 1600], 'evaluator': 'development-v2.0',
              'reviewers_required': 2, 'independent_validation_required': True,
              'official_test_access': False}
    cfg = tmp_path / 'cfg.json'; cfg.write_text(json.dumps(config))
    bundle = build_cases(manifest_file, cfg, tmp_path / 'output')
    assert [c['review_status'] for c in bundle['cases']] == [
        'manifest_good_normal', 'provisional', 'provisional']
    assert all(c['source_path'] == source for c in bundle['cases'])
    assert all('semantic_region_paths' in c and c['localization_semantics'].startswith('ambiguous') for c in bundle['cases'])
    packet = tmp_path / 'review_packet'
    forms = prepare(tmp_path / 'output', 'reviewer-a', packet)
    assert len(forms) == 2 and all('sample_id' not in form for form in forms)
    assert len(list(packet.glob('*.png'))) == 8
    assert all('valid' not in p.name and 'mismatch' not in p.name for p in packet.iterdir())
    with pytest.raises(FileExistsError):
        build_cases(manifest_file, cfg, tmp_path / 'output')
    config['source_images'] = ['juice_bottle/test/good/020.png']
    cfg.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='authorized'):
        build_cases(manifest_file, cfg, tmp_path / 'other')


def test_summary_does_not_promote_provisional_footprints_to_semantic_ap():
    reports = []
    for seed in (7, 17, 29):
        for arm in ('local', 'global', 'combined', 'neither'):
            cases = []
            for i in range(3):
                for kind, score in [('normal', 1), ('putative_valid_edit', 1.1),
                                    ('putative_identity_mismatch', 1.3)]:
                    cases.append({'source_path': f'p{i}', 'kind': kind, 'image_score': score,
                                  'alarm_at_frozen_good_threshold': False,
                                  'edit_footprint_ap_proxy': None if kind == 'normal' else .2,
                                  'semantic_region_ap_verified_only': None})
            reports.append({'seed': seed, 'arm': arm, 'run_id': f'{seed}-{arm}', 'cases': cases})
    result = summarize(reports)
    assert result['verified_semantic_violations'] == 0
    assert len(result['per_run']) == 12
    assert all(x['verified_semantic_ap'] is None for x in result['per_run'])
    assert all(x['provisional_violation_footprint_ap'] == pytest.approx(.2) for x in result['per_run'])


def test_review_templates_hide_model_scores_and_merge_keeps_unanswered_provisional(tmp_path):
    case_dir = tmp_path / 'cases'; case_dir.mkdir()
    case = {'sample_id': 'edited-1', 'kind': 'putative_valid_edit', 'review_status': 'provisional',
            'source_path': 'juice_bottle/validation/good/020.png'}
    (case_dir / 'case_manifest.json').write_text(json.dumps({'cases': [case]}))
    a, b = prepare(case_dir, 'alice'), prepare(case_dir, 'bob')
    packet_target = tmp_path / 'packet_target'; packet_target.mkdir()
    packet_link = tmp_path / 'packet_link'; packet_link.symlink_to(packet_target, target_is_directory=True)
    with pytest.raises(PermissionError, match='symlink'):
        prepare(case_dir, 'alice', packet_link)
    assert 'image_score' not in str(a) and 'sample_id' not in str(a)
    assert 'putative_valid' not in str(a)
    pa, pb = tmp_path / 'alice.json', tmp_path / 'bob.json'
    pa.write_text(json.dumps(a)); pb.write_text(json.dumps(b))
    _, statuses = merge(case_dir, [pa, pb])
    assert statuses['edited-1'] == 'provisional'
    for part in (a, b):
        part[0].update(verdict='valid', source_inspected=True,
                       target_region_accepted=True,
                       reason='The edited bottle retains the same fruit relation.')
    pa.write_text(json.dumps(a)); pb.write_text(json.dumps(b))
    _, statuses = merge(case_dir, [pa, pb])
    assert statuses['edited-1'] == 'reviewed_valid'


def test_case_artifacts_mount_read_only_only_from_approved_results(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    model = repo / 'candidates' / 'trial' / 'model.py'
    model.parent.mkdir(parents=True)
    model.write_text('')
    data = tmp_path / 'good'; data.mkdir()
    output = tmp_path / 'output'; output.mkdir()
    cases = repo / 'results' / 'metrics' / 'cases'; cases.mkdir(parents=True)
    monkeypatch.setattr(sandbox, 'available', lambda: True)
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/usr/bin/bwrap')
    args = sandbox.command(repo, model, data, output, ['-m', 'scripts.stage8_eval'], case_dir=cases)
    assert ['--ro-bind', str(cases), '/cases'] == args[args.index(str(cases)) - 1:args.index(str(cases)) + 2]
    assert ['--bind', str(cases), '/cases'] != args[args.index(str(cases)) - 1:args.index(str(cases)) + 2]
    outside = tmp_path / 'outside'; outside.mkdir()
    with pytest.raises(PermissionError, match='results/metrics'):
        sandbox.command(repo, model, data, output, ['-m', 'scripts.stage8_eval'], case_dir=outside)
