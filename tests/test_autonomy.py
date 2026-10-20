import json
import os
import subprocess
from pathlib import Path

import pytest

from orchestrator.autonomy import load_campaign_config, stop_reason, choose_proposal, reconcile_unfinished_runs
from orchestrator.campaign_state import CampaignState
from orchestrator.codex_sandbox import command
from orchestrator.state_store import StateStore


REPO = Path(__file__).resolve().parents[1]


def config():
    from legacy_campaign import legacy_config
    return legacy_config()


def test_campaign_persistence_and_call_budget(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    cfg = config()
    cfg['limits']['max_codex_calls'] = 1
    cid = campaigns.create(cfg)
    assert CampaignState(StateStore(store.path)).get(cid)['status'] == 'RUNNING'
    with pytest.raises(RuntimeError, match='already active'):
        campaigns.create(cfg)
    campaigns.reserve_codex_call(cid)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        campaigns.reserve_codex_call(cid)
    assert stop_reason(campaigns, cid)[0] == 'BUDGET_EXHAUSTED'
    campaigns.set_status(cid, 'PAUSED')
    campaigns.set_status(cid, 'RUNNING')
    campaigns.set_status(cid, 'STOPPED')
    with pytest.raises(ValueError, match='cannot be resumed'):
        campaigns.set_status(cid, 'RUNNING')


def test_campaign_gpu_budget_charges_failure(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    cfg = config()
    cfg['limits']['max_gpu_hours'] = 1 / 3600
    cid = campaigns.create(cfg)
    rid = store.create_run('h', 'toy_normal_ae', 'juice_bottle', config={'device': 'cuda'})
    campaigns.attach_run(cid, rid)
    for status in ('IMPLEMENTED', 'VALIDATED', 'QUEUED', 'RUNNING'):
        store.transition(rid, status)
    store.account_gpu_usage(rid, 2)
    store.transition(rid, 'FAILED', reason='CUDA OOM')
    assert campaigns.usage(cid)['gpu_seconds'] == 2
    assert stop_reason(campaigns, cid) == ('LIMIT_REACHED', 'GPU-time budget reached')


def test_experiment_cap_and_crash_recovery_never_replay(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    cfg = config()
    cfg['limits']['max_experiments'] = 1
    cid = campaigns.create(cfg)
    rid = store.create_run('h', 'toy_normal_ae', 'juice_bottle', config={'campaign_id': cid, 'device': 'cuda'})
    for status in ('IMPLEMENTED', 'VALIDATED', 'QUEUED', 'RUNNING'):
        store.transition(rid, status)
    campaigns.attach_run(cid, rid)
    reconcile_unfinished_runs(store, campaigns, cid, 'fixture worker crash')
    assert store.get_run(rid)['status'] == 'INTERRUPTED'
    assert campaigns.get(cid)['active_run_id'] == ''
    assert stop_reason(campaigns, cid) == ('LIMIT_REACHED', 'experiment count reached')
    reconcile_unfinished_runs(store, campaigns, cid, 'second recovery')
    assert store.get_run(rid)['status'] == 'INTERRUPTED'


def test_failure_limit_and_pause_gate(tmp_path):
    from orchestrator.autonomy import await_authorized_stage
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    cfg = config()
    cfg['limits']['max_consecutive_failures'] = 2
    cid = campaigns.create(cfg)
    campaigns.update(cid, consecutive_failures=2)
    assert stop_reason(campaigns, cid) == ('LIMIT_REACHED', 'consecutive failure limit reached')
    with pytest.raises(Exception, match='consecutive failure'):
        await_authorized_stage(campaigns, cid)


def test_proposal_selection_uses_agent_rank_and_skips_rejected(tmp_path):
    store = StateStore(tmp_path / 'ledger.sqlite')
    for hid, rank in [('slow', 3), ('fast', 1)]:
        store.add_proposal({'hypothesis_id': hid, 'priority_rank': rank})
    assert choose_proposal(store)['hypothesis_id'] == 'fast'
    store.set_hypothesis_status('fast', 'FAILED')
    assert choose_proposal(store)['hypothesis_id'] == 'slow'


def test_candidate_signature_mismatch_rejected_before_gpu(tmp_path):
    from orchestrator.experiment_runner import verify_candidate_source
    module = tmp_path / 'model.py'
    module.write_text('def train(good_images, output_dir, config=None): pass\n'
                      'def load(checkpoint_manifest, device=None): pass\n'
                      'def predict(images): pass\n')
    with pytest.raises(ValueError, match='train signature'):
        verify_candidate_source(module)


def test_personal_mode_is_disabled_by_default():
    assert config()['security']['operator_accepts_shared_uid_risk'] is False
    from orchestrator.autonomy import preflight
    with pytest.raises(PermissionError, match='not authorized'):
        preflight(config(), Path('/nonexistent'))


def test_official_access_config_is_rejected(tmp_path):
    import yaml
    cfg = config()
    cfg['research']['official_test_access'] = True
    path = tmp_path / 'unsafe.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='official test'):
        load_campaign_config(path)


def test_codex_workspace_mounts_are_narrow(tmp_path):
    output = tmp_path / 'response.json'
    candidate = REPO / 'candidates' / 'worktrees' / 'sandboxfixture' / 'candidates' / 'sandboxfixture'
    candidate.mkdir(parents=True, exist_ok=True)
    try:
        cmd = command(REPO, 'implementer', REPO / 'schemas' / 'implementation.schema.json', output,
                      'fixture task', candidate)
        joined = ' '.join(cmd)
        assert '--unshare-all' in cmd and '--clearenv' in cmd
        assert str(REPO / 'orchestrator') not in joined
        assert str(REPO / '.git') not in joined
        assert str(Path.home() / '.ssh') not in joined
        assert str(candidate) in cmd
        assert '--bind' in cmd
        with pytest.raises(PermissionError, match='assigned candidate'):
            command(REPO, 'implementer', REPO / 'schemas' / 'implementation.schema.json', output,
                    'fixture task', tmp_path)
    finally:
        # Remove only this deliberately created fixture path.
        candidate.rmdir()
        candidate.parent.rmdir()
        candidate.parent.parent.rmdir()


def test_dashboard_campaign_empty_and_recorded(tmp_path, monkeypatch):
    from dashboard.data_access import campaign_summary
    from dashboard.presentation import direction_label
    monkeypatch.setenv('LOCO_RUNTIME', str(tmp_path))
    monkeypatch.setenv('LOCO_LEDGER', str(tmp_path / 'ledger.sqlite'))
    store = StateStore(tmp_path / 'ledger.sqlite')
    assert campaign_summary() is None
    cid = CampaignState(store).create(config())
    assert campaign_summary()['id'] == cid
    assert campaign_summary()['usage']['experiments'] == 0
    assert direction_label('evidence-followup:persistent_topology_localization:investigate').startswith('Investigate persistent topology localization')


def test_cli_campaign_start_pause_resume_stop(tmp_path, monkeypatch):
    import yaml
    from orchestrator import autonomy, lab_cli
    cfg = config()
    cfg['security']['operator_accepts_shared_uid_risk'] = True
    source = tmp_path / 'approved.yaml'
    source.write_text(yaml.safe_dump(cfg))
    ledger = tmp_path / 'controller' / 'ledger.sqlite'
    monkeypatch.setattr(autonomy, 'preflight', lambda *a, **k: {'images': 389})
    monkeypatch.setattr(lab_cli, 'spawn_research',
                        lambda runtime, campaigns, cid: lab_cli.research_summary(campaigns, campaigns.get(cid), None))
    started = lab_cli.research_command('start', tmp_path, ledger, source)
    assert started['status'] == 'RUNNING'
    assert lab_cli.research_command('pause', tmp_path, ledger)['status'] == 'PAUSED'
    assert lab_cli.research_command('resume', tmp_path, ledger)['status'] == 'RUNNING'
    assert lab_cli.research_command('stop', tmp_path, ledger)['status'] == 'STOPPED'
    with pytest.raises(RuntimeError, match='no active'):
        lab_cli.research_command('resume', tmp_path, ledger)


def test_next_hypothesis_generation_uses_measured_critique(tmp_path, monkeypatch):
    from orchestrator import autonomy, controller
    store = StateStore(tmp_path / 'ledger.sqlite')
    campaigns = CampaignState(store)
    cid = campaigns.create(config())
    rid = store.create_run('prior_h', 'toy_normal_ae', 'juice_bottle', config={'campaign_id': cid})
    for status in ('IMPLEMENTED', 'VALIDATED', 'QUEUED', 'RUNNING'):
        store.transition(rid, status)
    store.set_run_result(rid, {'dev_synthetic/pixel_auroc': 0.42})
    store.transition(rid, 'COMPLETED')
    campaigns.attach_run(cid, rid)
    campaigns.update(cid, active_run_id='', next_hypothesis='evidence-followup:prior_h:investigate')
    with store.connect() as con:
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                    (rid, 1.0, 'CRITIC_REVIEW', json.dumps({'recommended_next_action': 'investigate',
                                                              'critic_concerns': ['shortcut risk']})))
    prompts = []
    def proposals(target_store, prompt, **kwargs):
        prompts.append(prompt)
        item = {'hypothesis_id': 'new_h', 'priority_rank': 1, 'priority_rationale': 'Measured failure',
                'new_dependencies': [],'experiment_plan': __import__('plan_fixtures').experiment_plan('new_h')}
        target_store.add_proposal(item)
        return [item]
    monkeypatch.setattr(controller, 'add_research_proposals', proposals)
    monkeypatch.setattr(controller, 'implement_proposal', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fixture stop')))
    with pytest.raises(RuntimeError, match='fixture stop'):
        autonomy.run_cycle(store, campaigns, cid, tmp_path)
    assert '0.42' in prompts[0]
    assert 'shortcut risk' in prompts[0]
    assert campaigns.get(cid)['current_hypothesis'] == 'new_h'


def test_fresh_gpu_cycle_uses_prior_evidence_and_cuda_contract(tmp_path, monkeypatch):
    from orchestrator import autonomy, controller
    store = StateStore(tmp_path / 'ledger.sqlite')
    prior = store.create_run('prior_research', 'prior_candidate', 'juice_bottle', config={})
    for state in ('IMPLEMENTED', 'VALIDATED', 'QUEUED', 'RUNNING'):
        store.transition(prior, state)
    store.set_run_result(prior, {'dev_synthetic/generator_gap': -0.14})
    store.transition(prior, 'COMPLETED')
    with store.connect() as con:
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",
                    (prior, 1.0, 'CRITIC_REVIEW', json.dumps({'critic_concerns': ['generator shortcut']})))
    cfg = config()
    cfg['research'].update(fresh_literature=True, fresh_hypotheses=True, require_torch_cuda=True)
    campaigns = CampaignState(store)
    cid = campaigns.create(cfg)
    prompts = []
    monkeypatch.setattr(controller, 'scout_literature', lambda *a, **k: {'sources': []})
    def propose(target, prompt, **kwargs):
        prompts.append(prompt)
        item = {'hypothesis_id': 'fresh_cuda_h', 'priority_rank': 1,
                'priority_rationale': 'independent generator evidence', 'new_dependencies': [],
                'experiment_plan': __import__('plan_fixtures').experiment_plan('fresh_cuda_h')}
        target.add_proposal(item)
        return [item]
    monkeypatch.setattr(controller, 'add_research_proposals', propose)
    def implement(*args, **kwargs):
        assert kwargs['require_torch_cuda'] is True
        raise RuntimeError('fixture stop')
    monkeypatch.setattr(controller, 'implement_proposal', implement)
    with pytest.raises(RuntimeError, match='fixture stop'):
        autonomy.run_cycle(store, campaigns, cid, tmp_path)
    assert 'generator shortcut' in prompts[0]
    assert '-0.14' in prompts[0]
    assert campaigns.get(cid)['current_hypothesis'] == 'fresh_cuda_h'


@pytest.mark.skipif(os.getenv('LOCO_TEST_BWRAP') != '1', reason='requires Linux user namespaces')
def test_codex_personal_mount_adversarial_fixture(tmp_path):
    output = tmp_path / 'response.json'
    candidate = REPO / 'candidates' / 'worktrees' / 'codexboundaryfixture' / 'candidates' / 'codexboundaryfixture'
    candidate.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / 'fake-secret'
    sentinel.write_text('dummy')
    try:
        cmd = command(REPO, 'implementer', REPO / 'schemas' / 'implementation.schema.json', output,
                      'fixture task', candidate)
        script = ("import json; from pathlib import Path; "
                  f"print(json.dumps({{'outside':Path('{sentinel}').exists(),"
                  "'controller':Path('/workspace/orchestrator/controller.py').exists(),"
                  "'ssh':Path('/home/agent/.ssh/id_rsa').exists(),"
                  "'candidate_write':(Path('/workspace/candidates/codexboundaryfixture/model.py').write_text('ok') == 2)}))")
        cmd = cmd[:cmd.index('--')] + ['--', '/usr/bin/python3', '-c', script]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == {'outside': False, 'controller': False,
                                           'ssh': False, 'candidate_write': True}
    finally:
        (candidate / 'model.py').unlink(missing_ok=True)
        candidate.rmdir()
        candidate.parent.rmdir()
        candidate.parent.parent.rmdir()
