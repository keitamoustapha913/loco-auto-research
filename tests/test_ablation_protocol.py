import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

from benchmark.metrics import synthetic_pixel_average_precision
from orchestrator.ablation_study import _next_arm
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from scripts.run_candidate import paired_patch_edits


def test_synthetic_average_precision_has_clear_support_and_tie_behavior():
    mask = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
    perfect = np.array([[[0.9, 0.2], [0.1, 0.8]]])
    assert synthetic_pixel_average_precision(perfect, mask) == 1.0
    assert synthetic_pixel_average_precision(np.ones_like(perfect), mask) == 0.5
    with pytest.raises(ValueError, match='positive and negative'):
        synthetic_pixel_average_precision(perfect, np.zeros_like(mask))


def test_paired_edit_is_deterministic_and_uses_same_artifact_support():
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[:, 8:25, 8:25] = 1
    donor = np.full_like(image, 0.35)
    changed, control, photometric, mask = paired_patch_edits(image, donor, seed=13)
    again = paired_patch_edits(image, donor, seed=13)
    assert all(np.array_equal(a, b) for a, b in zip((changed, control, photometric, mask), again))
    assert changed.shape == control.shape == photometric.shape == image.shape
    assert mask.shape == image.shape[1:]
    assert np.any(changed != control)
    assert np.all(changed[:, mask == 0] == image[:, mask == 0])
    assert np.all(control[:, mask == 0] == image[:, mask == 0])
    assert np.all(photometric[:, mask == 0] == image[:, mask == 0])
    assert 0 < mask.sum() < mask.size


def test_second_arm_comes_from_saved_measured_decision(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    from orchestrator.autonomy import load_campaign_config
    from pathlib import Path
    cfg = load_campaign_config(Path(__file__).resolve().parents[1] / 'configs' / 'autonomy.yaml')
    cfg['campaign']['start_from']='new'
    cfg['research']['study'] = 'masked_context_ablation'
    cid = campaigns.create(cfg)
    prior = {'run_id': 'measured-combined', 'status': 'REVIEWED',
             'config_json': json.dumps({'context_arm': 'combined'})}
    campaigns.event(cid, 'ADAPTIVE_ABLATION_DECISION', source_run_id=prior['run_id'],
                    next_arm='global', evidence='measured gap',
                    falsifiable_prediction='global context will isolate shortcut',
                    shortcut_risk='position dependence')
    assert _next_arm(store, campaigns, cid, [prior]) == 'global'
    second = {'run_id': 'measured-global', 'status': 'REVIEWED',
              'config_json': json.dumps({'context_arm': 'global'})}
    assert _next_arm(store, campaigns, cid, [prior, second]) == 'local'


def test_codex_ablation_choice_must_cite_the_measured_first_run(tmp_path, monkeypatch):
    from orchestrator import ablation_study
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    from orchestrator.autonomy import load_campaign_config
    cfg = load_campaign_config(Path(__file__).resolve().parents[1] / 'configs' / 'masked_context_ablation.yaml')
    cid = campaigns.create(cfg)
    prior = {'run_id': 'measured-combined', 'status': 'REVIEWED',
             'metrics_json': json.dumps({'dev_synthetic/pixel_ap': 0.41,
                                          'robustness/artifact_matched_good_fpr': 0.33}),
             'config_json': json.dumps({'context_arm': 'combined'})}
    with store.connect() as con:
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (prior['run_id'], 1.0, 'CRITIC_REVIEW', json.dumps({
                        'critic_concerns': ['artifact shortcut'], 'recommended_next_action': 'ablate'})))
    prompts = []
    def agent(*args, **kwargs):
        prompts.append(args[4])
        return {'response': {'schema_version': '1.0', 'source_run_id': prior['run_id'],
                             'next_arm': 'global', 'evidence': 'artifact controls failed at 33 percent',
                             'falsifiable_prediction': 'global only should lower false alarms',
                             'shortcut_risk': 'position bias could persist'}}
    monkeypatch.setattr(ablation_study, 'budgeted_invoke', agent)
    assert _next_arm(store, campaigns, cid, [prior]) == 'global'
    assert '0.41' in prompts[0] and 'artifact shortcut' in prompts[0]
    assert campaigns.get(cid)['next_hypothesis'] == 'masked_context_ablation_global'
    assert _next_arm(store, campaigns, cid, [prior]) == 'global'
    assert len(prompts) == 1


def test_paired_cluster_comparison_uses_same_sources_and_reports_uncertainty(tmp_path):
    from orchestrator.ablation_study import compare_arms
    def write(run_id, arm, offset):
        folder = tmp_path / 'jobs' / run_id
        folder.mkdir(parents=True)
        cases = []
        for index in range(4):
            cases.append({'image_index': index, 'donor_pair_cluster': index // 2,
                          'primary_pixel_ap': 0.3 + offset,
                          'alternate_pixel_ap': 0.2 + offset,
                          'relocated_pixel_ap': 0.1 + offset,
                          'normal_alarm': False, 'matched_control_alarm': False,
                          'photometric_control_alarm': False})
        (folder / 'development_case_metrics.json').write_text(json.dumps({
            'metric_status': 'SYNTHETIC_DEVELOPMENT_PROXY_ONLY', 'cases': cases}))
        return {'run_id': run_id, 'config_json': json.dumps({'context_arm': arm})}
    combined = write('combined', 'combined', 0.1)
    local = write('local', 'local', 0)
    result = compare_arms(tmp_path, [combined, local], bootstrap_repeats=100)
    assert result['n_images'] == 4 and result['n_donor_clusters'] == 2
    assert result['local']['primary_pixel_ap']['combined_minus_local'] == pytest.approx(0.1)
    assert result['local']['primary_pixel_ap']['ci95'] == pytest.approx([0.1, 0.1])


def test_trusted_queue_freezes_arm_seed_and_training_budget(tmp_path):
    from benchmark.data_guard import build_manifest
    from orchestrator.controller import queue_candidate
    from scripts.demo_run import make_fixture
    root = tmp_path / 'good'
    make_fixture(root)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(build_manifest(root)))
    store = StateStore(tmp_path / 'ledger.sqlite')
    run_id = queue_candidate(store, 'masked_context_ablation_local', 'masked_context_ablation',
                             'juice_bottle', 'pilot', 'cuda', manifest, seconds=120,
                             personal_mode_authorized=True, context_arm='local')
    config = json.loads(store.get_run(run_id)['config_json'])
    assert {key: config[key] for key in ('context_arm', 'seed', 'size', 'steps', 'batch_size', 'timeout_seconds')} == {
        'context_arm': 'local', 'seed': 7, 'size': 128, 'steps': 100,
        'batch_size': 4, 'timeout_seconds': 120}
    with pytest.raises(ValueError, match='context_arm'):
        queue_candidate(store, 'invalid', 'masked_context_ablation', 'juice_bottle', 'pilot',
                        'cuda', manifest, seconds=120, personal_mode_authorized=True,
                        context_arm='other')


@pytest.mark.skipif(os.getenv('LOCO_TEST_BWRAP') != '1', reason='requires Linux user namespaces')
def test_bounded_cpu_fixture_runs_full_ablation_evaluator_inside_bubblewrap(tmp_path):
    from benchmark.data_guard import build_manifest
    from orchestrator.sandbox import command, write_sandbox_manifest
    from scripts.demo_run import make_fixture
    repo = Path(__file__).resolve().parents[1]
    model = repo / 'candidates' / 'masked_context_ablation' / 'model.py'
    good = tmp_path / 'good'
    make_fixture(good)
    source_manifest = tmp_path / 'manifest.json'
    source_manifest.write_text(json.dumps(build_manifest(good)))
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'output'
    inputs.mkdir(); output.mkdir()
    sandbox_manifest, data_root = write_sandbox_manifest(source_manifest, inputs)
    config = inputs / 'config.json'
    config.write_text(json.dumps({'category': 'juice_bottle', 'device': 'cpu', 'size': 32,
                                  'steps': 2, 'batch_size': 1, 'seed': 7,
                                  'context_arm': 'combined'}))
    process = subprocess.run(command(repo, model, data_root, output,
        ['-m', 'scripts.run_candidate', '--module', '/candidate/model.py',
         '--manifest', '/inputs/good_manifest.json', '--config', '/inputs/config.json',
         '--output', '/output', '--result', '/output/measurement.json'],
        input_files={'good_manifest.json': sandbox_manifest, 'config.json': config}),
        capture_output=True, text=True, timeout=90, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
    assert process.returncode == 0, process.stderr[-2000:]
    measurement = json.loads((output / 'measurement.json').read_text())
    metrics = measurement['metrics']
    assert measurement['provenance']['metric_evaluator_version'] == 'development-v2.0'
    assert measurement['provenance']['synthetic_mask_semantics'].startswith('edited target footprint only')
    assert metrics['normal/image_assessed'] == 2
    assert 'normal/image_false_positives' in metrics
    assert all(key in metrics for key in (
        'dev_synthetic/pixel_ap', 'dev_synthetic/alternate_pixel_ap',
        'dev_synthetic/relocated_pixel_ap', 'robustness/artifact_matched_good_fpr',
        'robustness/photometric_good_fpr'))
    assert measurement['provenance']['development_calibration_count'] == 1
    assert measurement['provenance']['normal_assessment_count'] == 2
    cases = json.loads((output / 'development_case_metrics.json').read_text())['cases']
    assert len(cases) == 2 and {case['donor_pair_cluster'] for case in cases} == {0}
