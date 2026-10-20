import json
from pathlib import Path
import pytest
from jsonschema import validate
from orchestrator import controller
from orchestrator.codex_runner import AgentFailure, invoke
from orchestrator.experiment_runner import verify_candidate_source
from orchestrator.state_store import StateStore
from scripts.demo_run import make_fixture
from benchmark.data_guard import build_manifest


def proposal(index):
    mechanisms = {1: 'predict missing object placement from neighboring context',
                  2: 'distill spatial teacher residuals into a compact student',
                  3: 'model component relationships with a parametric graph'}
    return {'schema_version':'1.0','followup_of':'','followup_kind':'new', 'hypothesis_id':f'hypothesis_{index}',
            'research_question':f'Can independent idea {index} improve localization?',
            'claim':f'The method can {mechanisms[index]} and change proxy localization.',
            'references':['https://example.org/primary-paper'], 'baseline':'toy reference',
            'expected_observation':'Higher proxy localization at similar latency.',
            'controlled_change':f'Change only the mechanism that {mechanisms[index]}.',
            'evaluation_plan':'Use held-out normal images and two synthetic generators.',
            'estimated_gpu_minutes':10, 'risk_and_shortcuts':'May learn edit artifacts.',
            'new_dependencies':[], 'candidate_family':f'family_{index}',
            'priority_rank':index, 'priority_rationale':'Ranked by expected information gain and bounded cost.',
            'experiment_plan': __import__('plan_fixtures').agent_plan(__import__('plan_fixtures').experiment_plan(f'hypothesis_{index}'))}


def test_ranked_distinct_proposals_and_duplicate_guard(tmp_path, monkeypatch):
    store = StateStore(tmp_path / 'db.sqlite')
    batch = [proposal(i) for i in (1,2,3)]
    schema = json.loads((Path(__file__).parents[1] / 'schemas/proposals.schema.json').read_text())
    validate({'schema_version':'1.0','proposals':batch}, schema)
    monkeypatch.setattr(controller, 'budgeted_invoke', lambda *args, **kwargs: {'response':{'proposals':batch}})
    result = controller.add_research_proposals(store, 'test')
    assert [p['priority_rank'] for p in result] == [1,2,3]
    assert len([e for e in store.events() if e['kind']=='HYPOTHESIS_RANKING']) == 3
    with pytest.raises(ValueError, match='duplicate'):
        controller.add_research_proposals(store, 'repeat')
    assert len(controller.list_proposals(store)) == 3


def test_dependency_field_rejects_narrative_instead_of_package_names(tmp_path, monkeypatch):
    store = StateStore(tmp_path / 'db.sqlite')
    batch = [proposal(i) for i in (1, 2, 3)]
    batch[0]['new_dependencies'] = ['None requested; use existing PyTorch.']
    monkeypatch.setattr(controller, 'budgeted_invoke', lambda *args, **kwargs: {'response': {'proposals': batch}})
    with pytest.raises(ValueError, match='package names'):
        controller.add_research_proposals(store, 'test')
    assert controller.list_proposals(store) == []


def test_cuda_pilot_rejects_cpu_only_hypotheses(tmp_path, monkeypatch):
    store = StateStore(tmp_path / 'db.sqlite')
    batch = [proposal(i) for i in (1, 2, 3)]
    batch[0]['estimated_gpu_minutes'] = 0
    batch[0]['evaluation_plan'] = 'CPU-only screening on good images.'
    monkeypatch.setattr(controller, 'budgeted_invoke',
                        lambda *args, **kwargs: {'response': {'proposals': batch}})
    with pytest.raises(ValueError, match='CUDA pilot'):
        controller.add_research_proposals(store, 'test', require_gpu_pilot=True)
    assert controller.list_proposals(store) == []


def test_nonexecuting_candidate_protocol_check(tmp_path):
    toy = Path(__file__).parents[1] / 'candidates/toy_normal_ae/model.py'
    assert 'source_sha256' in verify_candidate_source(toy)
    suspicious = tmp_path / 'model.py'
    suspicious.write_text('def train(*a): pass\ndef load(*a): pass\ndef predict(*a): pass\n# official_test\n')
    with pytest.raises(ValueError, match='protected'):
        verify_candidate_source(suspicious)


def test_implementer_rejected_under_operator_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_CODEX_ISOLATION_VERIFIED','1')
    repo = Path(__file__).parents[1]
    with pytest.raises(AgentFailure, match='separate verified OS identity'):
        invoke('implementer',repo,repo/'schemas/implementation.schema.json','test',worktree=repo/'candidates')


def test_supervised_cycle_interfaces_with_mocked_agents(tmp_path, monkeypatch):
    """Exercise state flow without invoking Codex or executing candidate code."""
    monkeypatch.setenv('LOCO_RUNTIME', str(tmp_path))
    monkeypatch.setenv('LOCO_ALLOW_UNATTENDED', '1')
    monkeypatch.setenv('LOCO_SANDBOX_VERIFIED', '1')
    make_fixture(tmp_path / 'good_data')
    manifest_path = tmp_path / 'good_manifest.json'
    manifest_path.write_text(json.dumps(build_manifest(tmp_path / 'good_data')))
    store = StateStore(tmp_path / 'ledger.sqlite')
    legacy = proposal(1); legacy.pop('experiment_plan')
    store.add_proposal(legacy)  # Explicit historical contract fixture, not a structured-plan bypass.
    monkeypatch.setattr(controller, 'implement_proposal', lambda store, hypothesis: {'candidate': hypothesis})
    def queue(store, hypothesis, candidate, category, stage, device, manifest, seconds):
        rid = store.create_run(hypothesis, candidate, category, stage, {'device': device})
        for state in ('IMPLEMENTED', 'VALIDATED', 'QUEUED'):
            store.transition(rid, state)
        return rid
    monkeypatch.setattr(controller, 'queue_candidate', queue)
    def worker(store, manifest, runtime):
        run = next(r for r in store.list_runs() if r['status']=='QUEUED')
        store.transition(run['run_id'], 'RUNNING')
        store.set_run_result(run['run_id'], {'normal/image_fpr_at_threshold':0.02,
            'dev_synthetic/pixel_auroc':0.6, 'inference/e2e_ms_p95':10.0})
        store.transition(run['run_id'], 'COMPLETED')
        return {'status':'COMPLETED'}
    monkeypatch.setattr(controller, 'run_worker_once', worker)
    monkeypatch.setattr(controller, 'budgeted_invoke', lambda *args, **kwargs: {
        'response': {'recommended_next_action':'investigate', 'critic_concerns':['synthetic shortcut']}})
    monkeypatch.setattr(controller, 'add_research_proposals', lambda *args, **kwargs: [proposal(2), proposal(3), proposal(1)])
    result = controller.run_research_campaign(store, manifest_path, device='cpu', max_experiments=1)
    assert len(result['completed']) == 1
    rid = result['completed'][0]['run_id']
    assert store.get_run(rid)['status'] in {'PROMOTED','REJECTED'}
    assert any(e['kind']=='CRITIC_REVIEW' for e in store.events())
    assert result['next_proposals']
