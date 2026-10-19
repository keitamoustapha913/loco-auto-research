import json

import numpy as np
from PIL import Image
import pytest

from benchmark.data_guard import build_manifest, manifest_fingerprint
from scripts.logical_review_set import PROTOCOL, adjudicate, build, sticker_edit, template, verify_artifacts


def _fixture(tmp_path):
    root = tmp_path / 'good'
    folder = root / 'juice_bottle' / 'validation' / 'good'
    folder.mkdir(parents=True)
    for name, fruit_color in [('000', (180, 140, 40)), ('010', (170, 135, 40)),
                              ('020', (150, 45, 35)), ('030', (155, 48, 36)),
                              ('040', (152, 46, 35))]:
        pixels = np.zeros((1600, 800, 3), np.uint8)
        pixels[700:1000, 280:560] = (230, 190, 100)
        pixels[790:910, 370:470] = fruit_color
        Image.fromarray(pixels).save(folder / f'{name}.png')
    manifest = build_manifest(root)
    manifest_file = tmp_path / 'manifest.json'
    manifest_file.write_text(json.dumps(manifest))
    def rel(name):
        return f'juice_bottle/validation/good/{name}.png'
    config = {'protocol': PROTOCOL, 'category': 'juice_bottle',
              'dataset_fingerprint': manifest_fingerprint(manifest),
              'rule_id': 'fruit_liquid_relation', 'rule_question': 'Does the fruit match the liquid?',
              'original_size': [800, 1600], 'reviewers_required': 2, 'official_test_access': False,
              'eligible_source_partition': 'stage6_assessment_validation_images_18_to_53',
              'cases': [
                  {'target': rel('020'), 'target_variant': 'cherry',
                   'target_sticker_xyxy': [280, 700, 560, 1000],
                   'valid_donor': rel('030'), 'valid_donor_variant': 'cherry',
                   'valid_donor_sticker_xyxy': [280, 700, 560, 1000],
                   'cross_donor': rel('010'), 'cross_donor_variant': 'banana',
                   'cross_donor_sticker_xyxy': [280, 700, 560, 1000]},
                  {'target': rel('030'), 'target_variant': 'cherry',
                   'target_sticker_xyxy': [280, 700, 560, 1000],
                   'valid_donor': rel('040'), 'valid_donor_variant': 'cherry',
                   'valid_donor_sticker_xyxy': [280, 700, 560, 1000],
                   'cross_donor': rel('010'), 'cross_donor_variant': 'banana',
                   'cross_donor_sticker_xyxy': [280, 700, 560, 1000]}]}
    config_file = tmp_path / 'config.json'; config_file.write_text(json.dumps(config))
    return manifest_file, config_file, config


def test_paired_edits_have_original_geometry_separate_draft_and_footprint(tmp_path):
    manifest, config, _ = _fixture(tmp_path)
    output = tmp_path / 'review'
    bundle = build(manifest, config, output)
    assert len(bundle['cases']) == 4
    assert bundle['independent_confirmation'] is False
    assert len(bundle['normal_source_groups']) == 2
    assert verify_artifacts(bundle, output)
    assert (output / 'review_form_template.json').exists()
    for case in bundle['cases']:
        assert case['review_status'] == 'provisional'
        image = np.asarray(Image.open(output / case['image_path']))
        source = np.asarray(Image.open(output / case['source_image_path']))
        region = np.asarray(Image.open(output / case['draft_region_path']))
        footprint = np.asarray(Image.open(output / case['edit_footprint_path']))
        assert image.shape == source.shape == (1600, 800, 3)
        assert region.shape == footprint.shape == (1600, 800)
        assert (region > 0).sum() < (footprint > 0).sum()
        assert np.array_equal(image[:600], source[:600])
        assert np.any(image != source)
    page = (output / 'review.html').read_text()
    assert 'putative_violation' not in page and 'cross_variant' not in page
    assert 'not ground truth' in page
    with pytest.raises(FileExistsError):
        build(manifest, config, output)
    first = output / bundle['cases'][0]['image_path']
    first.write_bytes(first.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='hash'):
        verify_artifacts(bundle, output)


def test_only_authorized_good_validation_sources(tmp_path):
    manifest, config_file, config = _fixture(tmp_path)
    config['cases'][0]['target'] = 'juice_bottle/test/good/020.png'
    config_file.write_text(json.dumps(config))
    with pytest.raises(PermissionError, match='outside prior calibration'):
        build(manifest, config_file, tmp_path / 'bad')
    config['cases'][0]['target'] = 'juice_bottle/validation/good/020.png'
    config['cases'][0]['cross_donor'] = 'juice_bottle/validation/good/000.png'
    config_file.write_text(json.dumps(config))
    with pytest.raises(PermissionError, match='outside prior calibration'):
        build(manifest, config_file, tmp_path / 'calibration')


def test_review_decisions_cannot_self_certify_or_create_semantic_metric(tmp_path):
    manifest, config, _ = _fixture(tmp_path)
    bundle = build(manifest, config, tmp_path / 'review')
    a = template(bundle, 'alice'); b = template(bundle, 'bob')
    p1, p2 = tmp_path / 'alice.json', tmp_path / 'bob.json'
    p1.write_text(json.dumps(a)); p2.write_text(json.dumps(b))
    result = adjudicate(bundle, [p1, p2])
    assert {x['status'] for x in result['outcomes'].values()} == {'uncertain'}
    for form in (a, b):
        for row, case in zip(form['cases'], bundle['cases']):
            row.update(logical_decision='valid' if case['condition'] == 'same_variant' else 'violation',
                       rule_supported='yes', artifact_dominates='no', region_decision='accept',
                       reason='Fruit label and liquid appear logically consistent or inconsistent.')
    p1.write_text(json.dumps(a)); p2.write_text(json.dumps(b))
    result = adjudicate(bundle, [p1, p2])
    assert sorted(x['status'] for x in result['outcomes'].values()) == [
        'reviewed_valid', 'reviewed_valid', 'reviewed_violation_logic_only',
        'reviewed_violation_logic_only']
    assert result['metric_ready_violations'] == 0
    b['cases'][0]['artifact_dominates'] = 'yes'
    p2.write_text(json.dumps(b))
    assert list(adjudicate(bundle, [p1, p2])['outcomes'].values())[0]['status'] == 'rejected'
    b['cases'][0]['case_sha256'] = 'stale'
    p2.write_text(json.dumps(b))
    with pytest.raises(ValueError, match='hash changed'):
        adjudicate(bundle, [p1, p2])
    p2.write_text(json.dumps(a))
    with pytest.raises(ValueError, match='identities'):
        adjudicate(bundle, [p1, p2])


def test_sticker_edit_rejects_unaligned_geometry():
    image = Image.new('RGB', (800, 1600))
    with pytest.raises(ValueError, match='outside'):
        sticker_edit(image, image, (0, 0, 801, 50), (0, 0, 40, 50))
