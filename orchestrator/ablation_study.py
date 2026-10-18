"""One bounded, evidence-adaptive comparison of the existing masked-context candidate.

The trusted controller chooses only a reviewed configuration value. It never
executes commands supplied by an agent and delegates each run to the existing
GPU queue and Bubblewrap runner.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from orchestrator.autonomy import REPO, await_authorized_stage, remaining_job_seconds
from orchestrator.controller import budgeted_invoke

CANDIDATE = 'masked_context_ablation'
ARMS = ('combined', 'local', 'global')
PRIOR_RUN = 'b0817b1f-d185-4827-8779-a69252e7b7e9'
SOURCE_REFERENCES = [
    'Prior measured run b0817b1f-d185-4827-8779-a69252e7b7e9 and its Research Critic review',
    'https://arxiv.org/abs/2303.14535',
    'https://arxiv.org/abs/2408.15628',
]


def _review_for_run(store, run_id):
    with store.connect() as con:
        row = con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CRITIC_REVIEW' "
                          "ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
    return json.loads(row['payload_json']) if row else {}


def _arm(run):
    return json.loads(run['config_json']).get('context_arm')


def _training_record(runtime, run_id):
    checkpoint = Path(runtime) / 'jobs' / run_id / 'checkpoint' / 'checkpoint_manifest.json'
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size > 100_000:
        raise ValueError('missing or unsafe ablation training record')
    record = json.loads(checkpoint.read_text())
    return {'updates': record.get('updates'), 'image_exposures': record.get('image_exposures'),
            'training_tensor_device': record.get('training_tensor_device'),
            'parameter_updated': record.get('parameter_updated')}


def _case_metrics(runtime, run_id):
    path = Path(runtime) / 'jobs' / run_id / 'development_case_metrics.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError('missing or unsafe per-image development metrics')
    data = json.loads(path.read_text())
    cases = data.get('cases')
    if data.get('metric_status') != 'SYNTHETIC_DEVELOPMENT_PROXY_ONLY' or not isinstance(cases, list):
        raise ValueError('invalid per-image development metric provenance')
    return cases


def compare_arms(runtime, runs, bootstrap_repeats=2000):
    """Paired donor-cluster bootstrap; uncertainty applies only to synthetic edits."""
    import numpy as np
    cases = {_arm(run): _case_metrics(runtime, run['run_id']) for run in runs}
    if 'combined' not in cases:
        return {}
    reference = cases['combined']
    keys = [(case['image_index'], case['donor_pair_cluster']) for case in reference]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('invalid paired assessment case index')
    result = {'scope': 'juice_bottle synthetic development only', 'n_images': len(keys),
              'n_donor_clusters': len({cluster for _, cluster in keys}),
              'method': 'paired donor-cluster bootstrap, 2000 fixed-seed resamples',
              'limitations': 'No official anomalies; donor controls are plausibly valid, not certified; one category and seed.'}
    for arm, arm_cases in cases.items():
        if arm == 'combined':
            continue
        if [(case['image_index'], case['donor_pair_cluster']) for case in arm_cases] != keys:
            raise ValueError('ablation arms evaluated different ordered assessment sources')
        comparisons = {}
        clusters = sorted({cluster for _, cluster in keys})
        for metric in ('primary_pixel_ap', 'alternate_pixel_ap', 'relocated_pixel_ap',
                       'normal_alarm', 'matched_control_alarm', 'photometric_control_alarm'):
            differences = np.asarray([float(a[metric]) - float(b[metric])
                                      for a, b in zip(reference, arm_cases)], dtype=float)
            if not np.isfinite(differences).all():
                raise ValueError('nonfinite paired ablation measurements')
            cluster_diffs = np.asarray([differences[[i for i, key in enumerate(keys) if key[1] == cluster]].mean()
                                        for cluster in clusters])
            rng = np.random.default_rng(991)
            draws = rng.integers(0, len(clusters), size=(bootstrap_repeats, len(clusters)))
            ci = np.quantile(cluster_diffs[draws].mean(axis=1), [0.025, 0.975])
            comparisons[metric] = {'combined_minus_' + arm: float(differences.mean()),
                                   'ci95': [float(ci[0]), float(ci[1])]}
        result[arm] = comparisons
    return result


def _candidate_snapshot():
    from orchestrator.experiment_runner import candidate_file, sha256, verify_candidate_source
    module = candidate_file(REPO, CANDIDATE)
    expected = (REPO / 'candidates' / CANDIDATE / 'model.py').resolve(strict=True)
    if module != expected or module.is_symlink():
        raise PermissionError('ablation candidate must be the reviewed source in candidates/')
    relative = str(module.relative_to(REPO))
    status = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', relative],
                            check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise PermissionError('ablation candidate source is not committed and clean')
    committed = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{relative}'],
                               check=True, capture_output=True).stdout
    source_hash = sha256(module)
    if hashlib.sha256(committed).hexdigest() != source_hash:
        raise PermissionError('candidate source differs from reviewed Git snapshot')
    verify_candidate_source(module)
    return module, source_hash


def _ensure_hypothesis(store, arm):
    hypothesis_id = f'masked_context_ablation_{arm}'
    with store.connect() as con:
        exists = con.execute('SELECT 1 FROM hypotheses WHERE hypothesis_id=?', (hypothesis_id,)).fetchone()
    if not exists:
        store.add_proposal({
            'schema_version': '1.0', 'hypothesis_id': hypothesis_id,
            'candidate_family': 'controlled_masked_context_ablation',
            'research_question': 'Does global context improve localization over matched local context on permitted development edits?',
            'claim': f'The {arm} arm provides an interpretable paired comparison under fixed data, exposure, and evaluator settings.',
            'references': SOURCE_REFERENCES,
            'baseline': 'Prior combined-context CUDA pilot and the other matched arms in this campaign',
            'controlled_change': 'Only the masked context inputs are switched among local, global and combined; network weights are retrained from the same initialization seed.',
            'expected_observation': 'A genuine contextual gain should improve several synthetic generators without raising normal or matched-control false alarms.',
            'evaluation_plan': 'Use 335 normal train images, disjoint normal calibration and assessment images, three synthetic generators, same-location donor controls, and fixed 100-update CUDA training.',
            'estimated_gpu_minutes': 2, 'new_dependencies': [],
            'risk_and_shortcuts': 'No certified logical labels; same-location donor controls may not always be valid. Generator artifacts and alignment can dominate proxy scores.',
            'priority_rank': ARMS.index(arm) + 1,
            'priority_rationale': 'The combined arm anchors the comparison; subsequent arm order follows measured evidence and a bounded Codex decision.',
        })
    return hypothesis_id


def choose_next_arm(store, campaigns, campaign_id, prior):
    """Codex chooses the second arm from the first measured run and critique."""
    prior_id = prior['run_id']
    review = _review_for_run(store, prior_id)
    if not review:
        raise RuntimeError('measured first run has no Research Critic assessment')
    prompt = (
        'Choose exactly one next ablation arm: local or global. The first run was '
        'combined local+global context. Select the arm that best falsifies the '
        'first measured outcome and critic concerns. This is a juice_bottle-only '
        'normal-good and synthetic-development study. Do not claim official LOCO '
        'performance. The network, masking, targets, optimizer, training exposure '
        'and evaluation protocol are identical across arms; only context inputs '
        'change. The third remaining arm may run afterward for completeness. '
        f'Source run: {prior_id}. Metrics: {prior["metrics_json"]}. '
        f'Critic: {json.dumps(review)}.'
    )
    decision = budgeted_invoke(store, 'researcher', REPO,
                               REPO / 'schemas' / 'ablation_decision.schema.json',
                               prompt, timeout=180, campaign=(campaigns, campaign_id),
                               personal_mode=True)['response']
    if decision['source_run_id'] != prior_id or decision['next_arm'] not in {'local', 'global'}:
        raise ValueError('research decision did not identify the measured source run or a permitted arm')
    campaigns.event(campaign_id, 'ADAPTIVE_ABLATION_DECISION', **decision)
    campaigns.update(campaign_id, next_hypothesis=f'masked_context_ablation_{decision["next_arm"]}')
    return decision['next_arm']


def _next_arm(store, campaigns, campaign_id, previous):
    if not previous:
        return 'combined'
    if any(run['status'] not in {'REVIEWED', 'REJECTED', 'PROMOTED'} for run in previous):
        raise RuntimeError('prior ablation run was not reviewed; expensive work will not be replayed')
    used = [_arm(run) for run in previous]
    if len(used) == 1:
        decisions = [event for event in campaigns.events(campaign_id, 100)
                     if event['kind'] == 'ADAPTIVE_ABLATION_DECISION']
        if decisions:
            decision = json.loads(decisions[0]['payload_json'])
            if decision.get('source_run_id') != previous[0]['run_id']:
                raise RuntimeError('saved adaptive decision refers to a different source run')
            return decision['next_arm']
        return choose_next_arm(store, campaigns, campaign_id, previous[0])
    remaining = [arm for arm in ARMS if arm not in used]
    if len(remaining) != 1:
        raise RuntimeError('ablation arm sequence is incomplete or duplicated')
    return remaining[0]


def run_ablation_cycle(store, campaigns, campaign_id, runtime):
    """Run one arm, critique it, then persist evidence for the next iteration."""
    from orchestrator.controller import queue_candidate, run_worker_once
    from orchestrator.experiment_runner import probe_candidate_interface, run_candidate_unit_tests
    from orchestrator.mlflow_tracker import enqueue_run_tracking

    campaign = campaigns.get(campaign_id)
    config = json.loads(campaign['config_json'])
    limits = config['limits']
    if limits['max_experiments'] != 3 or limits['max_parallel_gpu_jobs'] != 1:
        raise ValueError('this controlled study requires exactly three serialized arm runs')
    if config['research'].get('previous_run_id') != PRIOR_RUN:
        raise ValueError('ablation campaign must cite the reviewed prior CUDA pilot')
    prior_pilot = store.get_run(PRIOR_RUN)
    if not prior_pilot or prior_pilot['status'] not in {'REVIEWED', 'REJECTED', 'PROMOTED'} or not _review_for_run(store, PRIOR_RUN):
        raise ValueError('reviewed prior CUDA pilot and critique are required')
    previous = campaigns.runs(campaign_id)
    if len(previous) >= 3:
        raise RuntimeError('all three ablation arms have already been attempted')
    module, source_hash = _candidate_snapshot()
    if previous and any(json.loads(run['config_json']).get('model_sha256') != source_hash for run in previous):
        raise PermissionError('candidate source changed during the ablation campaign')
    if not previous:
        probe_candidate_interface(REPO, module, Path(__import__('os').environ['LOCO_GOOD_ROOT']),
                                  runtime, CANDIDATE)
        tests = run_candidate_unit_tests(REPO, module,
                                         Path(__import__('os').environ['LOCO_GOOD_ROOT']), runtime, CANDIDATE)
        campaigns.event(campaign_id, 'CANDIDATE_TESTS', candidate_id=CANDIDATE, **tests)
    await_authorized_stage(campaigns, campaign_id)
    arm = _next_arm(store, campaigns, campaign_id, previous)
    if arm in {_arm(run) for run in previous}:
        raise RuntimeError('duplicate ablation arm refused')
    hypothesis_id = _ensure_hypothesis(store, arm)
    campaigns.update(campaign_id, current_hypothesis=hypothesis_id, next_hypothesis='')
    campaigns.event(campaign_id, 'HYPOTHESIS_SELECTED', hypothesis_id=hypothesis_id,
                    rationale=('combined baseline' if arm == 'combined' else 'measured adaptive decision'))
    await_authorized_stage(campaigns, campaign_id)
    seconds = min(120, remaining_job_seconds(campaigns, campaign_id))
    if seconds < 15:
        raise RuntimeError('insufficient remaining bounded GPU time')
    manifest = Path(runtime) / 'good_manifest.json'
    run_id = queue_candidate(store, hypothesis_id, CANDIDATE, 'juice_bottle', 'pilot', 'cuda',
                             manifest, seconds=seconds, personal_mode_authorized=True,
                             campaign_id=campaign_id, campaign_name=config['campaign']['name'],
                             context_arm=arm)
    campaigns.attach_run(campaign_id, run_id)
    campaigns.event(campaign_id, 'EXPERIMENT_QUEUED', run_id=run_id, arm=arm)
    result = run_worker_once(store, manifest, runtime, personal_mode_authorized=True,
                             cancel=lambda: campaigns.get(campaign_id)['status'] == 'STOPPING',
                             expected_run_id=run_id)
    run = store.get_run(run_id)
    campaigns.update(campaign_id, active_run_id='')
    campaigns.event(campaign_id, 'EXPERIMENT_FINISHED', run_id=run_id, arm=arm, status=run['status'])
    if result.get('run_id') != run_id or run['status'] != 'COMPLETED':
        raise RuntimeError(f'ablation run {run_id} ended {run["status"]}: {run["error"]}')
    await_authorized_stage(campaigns, campaign_id, finishing=True)
    training = _training_record(runtime, run_id)
    matched_exposure = (training['updates'] == 100 and training['image_exposures'] == 400 and
                        training['training_tensor_device'] == 'cuda:0' and
                        training['parameter_updated'] is True)
    campaigns.event(campaign_id, 'TRAINING_EXPOSURE_CHECK', run_id=run_id,
                    matched=matched_exposure, record=training)
    history = [{'run_id': old['run_id'], 'arm': _arm(old), 'metrics': json.loads(old['metrics_json']),
                'training': _training_record(runtime, old['run_id']),
                'critique': _review_for_run(store, old['run_id'])} for old in previous]
    comparison = compare_arms(runtime, previous + [run]) if previous else {}
    if comparison:
        campaigns.event(campaign_id, 'ABLATION_COMPARISON', source_run=run_id, comparison=comparison)
    prompt = (
        'Critique this measured juice_bottle-only ablation run and compare it with prior arms when '
        'available. Synthetic pixel AUROC/AP and region overlap are development proxies only. '
        'The same-location donor control is plausibly layout-preserving, not certified valid. '
        'Examine generator disagreement, normal and artifact-control false alarms, paired arms, '
        'training exposures, latency, memory and missing five-category evidence. Do not promote '
        'an arm based on one proxy metric. State a falsifiable next experiment. '
        f'Current run: {json.dumps(run, default=str)}. Actual training: {json.dumps(training)}. '
        f'Prior arms: {json.dumps(history)}. Exposure matched: {matched_exposure}. '
        f'Paired development comparison: {json.dumps(comparison)}.'
    )
    review = budgeted_invoke(store, 'critic', REPO, REPO / 'schemas' / 'review.schema.json',
                             prompt, timeout=180, campaign=(campaigns, campaign_id),
                             personal_mode=True)['response']
    if review['run_id'] != run_id:
        raise ValueError('critic response references a different run')
    with store.connect() as con:
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (run_id, time.time(), 'CRITIC_REVIEW', json.dumps(review)))
    store.transition(run_id, 'REVIEWED')
    enqueue_run_tracking(store, run_id)
    campaigns.event(campaign_id, 'CRITIQUE', run_id=run_id, review=review)
    campaigns.event(campaign_id, 'NEXT_DECISION', action=review['recommended_next_action'],
                    evidence_run=run_id, reason=review['suggested_falsification'])
    if not matched_exposure:
        raise RuntimeError('ablation training exposure or CUDA evidence did not match the fixed protocol')
