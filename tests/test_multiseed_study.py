import json
from pathlib import Path

import pytest

from benchmark.metrics_v2 import EVALUATOR_VERSION
from orchestrator.autonomy import load_campaign_config, preflight
from orchestrator.multiseed_study import ARMS, SEEDS, compare_multiseed, prescribed_plan


def test_prepared_campaign_is_disabled_and_exactly_fifteen_serial_jobs(tmp_path):
    cfg = load_campaign_config(Path(__file__).resolve().parents[1] / 'configs/stage6_multiseed.yaml')
    assert cfg['security']['operator_accepts_shared_uid_risk'] is False
    assert prescribed_plan(cfg) == [(seed, arm) for seed in SEEDS for arm in ARMS]
    assert cfg['limits'] == {'max_gpu_hours': .5, 'max_wall_hours': 4,
                             'max_experiments': 15, 'max_codex_calls': 1,
                             'max_job_seconds': 120, 'max_consecutive_failures': 1,
                             'max_parallel_gpu_jobs': 1}
    with pytest.raises(PermissionError, match='not authorized'):
        preflight(cfg, tmp_path)


def test_edited_stage6_grid_is_rejected():
    cfg = load_campaign_config(Path(__file__).resolve().parents[1] / 'configs/stage6_multiseed.yaml')
    cfg['research']['seeds'] = [7, 17, 29, 31]
    with pytest.raises(ValueError, match='fixed'):
        prescribed_plan(cfg)
    cfg['research']['seeds'] = [7, 17, 29]
    cfg['limits']['max_codex_calls'] = 30
    with pytest.raises(ValueError, match='fixed'):
        prescribed_plan(cfg)


def test_multiseed_summary_keeps_seed_and_image_uncertainty_separate(tmp_path):
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            run_id = f'{seed}-{arm}'
            folder = tmp_path / 'jobs' / run_id
            folder.mkdir(parents=True)
            (folder / 'run_manifest.json').write_text(json.dumps({
                'metric_evaluator_version': EVALUATOR_VERSION,
                'synthetic_generator_source_sha256': 'generator-hash',
                'calibration_path_sha256': 'calibration-hash',
                'assessment_path_sha256': 'assessment-hash'}))
            offset = .02 if arm == 'combined' else 0
            cases = [{'image_index': i, 'donor_pair_cluster': i // 2,
                      'primary_pixel_ap': .1 + i * .01 + offset + seed / 10000,
                      'alternate_pixel_ap': .05 + i * .01 + offset,
                      'relocated_pixel_ap': .03 + i * .01 + offset} for i in range(4)]
            (folder / 'development_case_metrics.json').write_text(json.dumps({
                'metric_status': 'SYNTHETIC_DEVELOPMENT_PROXY_ONLY', 'cases': cases}))
            runs.append({'run_id': run_id, 'status': 'REVIEWED',
                'manifest_sha256': 'good-only-hash', 'gpu_seconds': 10,
                'config_json': json.dumps({'seed': seed, 'context_arm': arm,
                                           'size': 128, 'steps': 100, 'batch_size': 4}),
                'metrics_json': json.dumps({'dev_synthetic/pixel_ap': .3 + offset,
                    'dev_synthetic/alternate_pixel_ap': .2 + offset,
                    'dev_synthetic/relocated_pixel_ap': .1 + offset,
                    'normal/image_false_positives': 1, 'normal/image_assessed': 36})})
    summary = compare_multiseed(tmp_path, runs)
    assert len(summary['per_seed']) == 15
    assert summary['paired_differences']['combined']['primary_pixel_ap']['arm_minus_local_mean'] == pytest.approx(.02)
    assert summary['paired_differences']['combined']['primary_pixel_ap']['between_seed_sample_sd'] == pytest.approx(0)
    assert summary['across_seed']['combined']['primary_pixel_ap']['sample_sd'] > 0
    assert summary['metric_evaluator_version'] == EVALUATOR_VERSION


def test_multiseed_rejects_evaluator_mixing(tmp_path):
    with pytest.raises(ValueError, match='15'):
        compare_multiseed(tmp_path, [])


def test_results_page_shows_only_completed_versioned_multiseed_comparison(tmp_path, monkeypatch):
    from orchestrator.campaign_state import CampaignState
    from orchestrator.state_store import StateStore
    from dashboard.data_access import latest_multiseed_comparison
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LOCO_RUNTIME', str(tmp_path))
    monkeypatch.setenv('LOCO_LEDGER', str(tmp_path / 'controller' / 'experiments.sqlite'))
    store = StateStore(tmp_path / 'controller' / 'experiments.sqlite')
    campaigns = CampaignState(store)
    cfg = load_campaign_config(Path(__file__).resolve().parents[1] / 'configs/stage6_multiseed.yaml')
    cid = campaigns.create(cfg)
    assert latest_multiseed_comparison() is None
    across = {arm: {metric: {'mean': .2, 'sample_sd': .01} for metric in
                    ('primary_pixel_ap', 'alternate_pixel_ap', 'relocated_pixel_ap')} for arm in ARMS}
    paired = {arm: {metric: {'arm_minus_local_mean': .01, 'between_seed_sample_sd': .02}
                    for metric in ('primary_pixel_ap', 'alternate_pixel_ap', 'relocated_pixel_ap')}
              for arm in ARMS if arm != 'local'}
    comparison = {'metric_evaluator_version': EVALUATOR_VERSION, 'per_seed': [{}] * 15,
                  'seeds': list(SEEDS), 'arms': list(ARMS), 'across_seed': across,
                  'paired_differences': paired}
    campaigns.event(cid, 'STAGE6_COMPARISON', comparison=comparison)
    assert latest_multiseed_comparison() == comparison
    page = Path(__file__).resolve().parents[1] / 'dashboard/pages/03_Results.py'
    rendered = AppTest.from_file(page).run()
    assert not rendered.exception
    assert any('Three-seed comparison' in item.value for item in rendered.subheader)
