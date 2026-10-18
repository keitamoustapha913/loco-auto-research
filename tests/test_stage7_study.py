import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.stage7_controls import PROTOCOL_VERSION, patch_controls, whole_image_controls, texture_noise_control
from scripts.stage7_study import fixed_protocol, summarize


def test_fixed_protocol_is_cpu_only_and_deduplicates_baseline():
    cfg = fixed_protocol()
    assert cfg['device'] == 'cpu' and cfg['no_training'] and not cfg['official_test_access']
    assert cfg['arms'] == ['local', 'global', 'combined', 'neither']
    assert cfg['assessment_images'] == 12 and cfg['timeout_seconds_per_checkpoint'] == 120


def test_control_masks_are_exactly_the_edited_regions_and_separate_source_target():
    x = np.full((3, 96, 96), .3, np.float32)
    x[:, 10:26, 10:26] = .9
    donor = np.full_like(x, .7)
    a = patch_controls(x, donor, seed=13)
    b = patch_controls(x, donor, seed=13)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    source, target = a['source_mask'], a['target_mask']
    assert source.sum() == target.sum() == 16 * 16
    assert not np.any(source & target)
    assert np.array_equal(a['insertion'][:, target == 0], x[:, target == 0])
    assert np.array_equal(a['removal_proxy'][:, source == 0], x[:, source == 0])
    assert np.array_equal(a['relocation_proxy'][:, (source | target) == 0], x[:, (source | target) == 0])
    assert np.array_equal(a['donor_control'][:, target == 0], x[:, target == 0])
    assert not np.array_equal(a['insertion'], a['donor_control'])


def test_whole_frame_controls_preserve_shape_and_position_shift_is_deterministic():
    x = np.arange(3 * 64 * 64, dtype=np.float32).reshape(3, 64, 64) / (3 * 64 * 64)
    controls = whole_image_controls(x)
    assert controls['position_shift'].shape == x.shape
    assert controls['illumination'].shape == x.shape
    assert np.array_equal(controls['position_shift'], whole_image_controls(x)['position_shift'])
    assert not np.array_equal(controls['position_shift'], x)
    with pytest.raises(ValueError):
        whole_image_controls(np.zeros((3, 8, 8), np.float32))


def test_texture_control_is_fixed_and_does_not_move_pixels():
    x = np.full((3, 32, 32), .5, np.float32)
    a = texture_noise_control(x, seed=17)
    assert a.shape == x.shape and a.dtype == np.float32
    assert np.array_equal(a, texture_noise_control(x, seed=17))
    assert not np.array_equal(a, texture_noise_control(x, seed=18))
    assert 0 <= a.min() <= a.max() <= 1
    with pytest.raises(ValueError):
        texture_noise_control(x, seed=1, sigma=1)


def test_texture_protocol_is_versioned_separately_from_first_diagnostic():
    from scripts.stage7_texture_study import fixed_texture_protocol
    cfg = fixed_texture_protocol()
    assert cfg['parent_protocol'] == PROTOCOL_VERSION
    assert cfg['protocol'] == 'stage7-texture-equivariance-v2'
    assert cfg['device'] == 'cpu' and cfg['texture_sigma'] == .02


def test_shift_alignment_accounts_for_both_axes():
    from scripts.stage7_texture_eval import correlation
    x = np.random.default_rng(42).random((3, 64, 64), dtype=np.float32)
    shifted = whole_image_controls(x)['position_shift']
    assert correlation(x[0, :-8, 8:], shifted[0, 8:, :-8]) == pytest.approx(1.0)
    assert correlation(x[0, :, 8:], shifted[0, :, :-8]) < .1


def test_paired_summary_requires_identical_image_ids_and_excludes_duplicate_arm():
    cfg = fixed_protocol()
    reports = []
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            offset = {'local': 0, 'global': .1, 'combined': .02, 'neither': -.1}[arm]
            cases = []
            for i in range(3):
                cases.append({'image_id_sha256': f'id{i}',
                    'scores': {name: float(i + j) for j, name in enumerate(
                        ['normal', 'position_shift', 'illumination', 'insertion', 'donor_control',
                         'appearance_control', 'removal_proxy', 'relocation_proxy'])},
                    'ap': {m: .3 + offset for m in ('insertion_target', 'relocation_target',
                                                   'removal_source', 'relocation_source')}})
            reports.append({'seed': seed, 'arm': arm, 'run_id': f'{seed}-{arm}',
                            'cases': cases, 'macro_ap': {'insertion_target': .3 + offset},
                            'pooled_ap': {'insertion_target': .3 + offset},
                            'normal_false_positives': {'false_positives': 1},
                            'normal_threshold': 1.0, 'control_alarms': {'illumination': 1},
                            'median_cpu_latency_ms': 2.0})
    result = summarize(reports, cfg)
    assert len(result['per_run']) == 12
    assert 'spatial' not in result['arms']
    assert result['paired']['global_minus_local']['insertion_target']['across_seed_mean'] == pytest.approx(.1)
    reports[1]['cases'][0]['image_id_sha256'] = 'different'
    with pytest.raises(ValueError, match='different assessment'):
        summarize(reports, cfg)


def test_sandbox_only_accepts_named_frozen_checkpoint_inputs(tmp_path, monkeypatch):
    from orchestrator import sandbox
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sandbox, 'available', lambda: True)
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/usr/bin/bwrap')
    data = tmp_path / 'good'; data.mkdir()
    output = tmp_path / 'output'; output.mkdir()
    safe = tmp_path / 'model.pt'; safe.write_bytes(b'fixture')
    args = sandbox.command(repo, repo / 'candidates/masked_context_ablation/model.py',
        data, output, ['-c', 'pass'], input_files={'model.pt': safe})
    assert '/inputs/model.pt' in args and '--unshare-all' in args
    with pytest.raises(ValueError, match='unapproved'):
        sandbox.command(repo, repo / 'candidates/masked_context_ablation/model.py',
            data, output, ['-c', 'pass'], input_files={'official_test.json': safe})
