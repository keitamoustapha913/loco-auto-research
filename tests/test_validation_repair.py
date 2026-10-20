"""CPU rejection and bounded repair execute before any CUDA job is queued."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest

from orchestrator import controller, experiment_runner
from orchestrator.state_store import StateStore


def prepare(tmp_path, monkeypatch, failures):
    store = StateStore(tmp_path/'ledger.sqlite')
    store.add_proposal({'hypothesis_id':'fixture'})
    tree = tmp_path/'candidates/worktrees/fixture'
    folder = tree/'candidates/fixture'; folder.mkdir(parents=True)
    (folder/'model.py').write_text('def train(config, manifest, output): pass\ndef load(*a): pass\ndef predict(*a): pass')
    (folder/'test_model.py').write_text('def test_cpu(): assert True')
    monkeypatch.setattr(controller, 'REPO', tmp_path)
    monkeypatch.setattr(controller, 'create_candidate_worktree', lambda *a: tree)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a,**k: SimpleNamespace(stdout='',returncode=0))
    monkeypatch.setenv('LOCO_GOOD_ROOT',str(tmp_path))
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',lambda *a: True)
    checks=[]
    def check(*a):
        checks.append(1)
        if len(checks)<=failures:
            raise ValueError('EXACT CPU ERROR: all 3 tests skipped: CUDA required')
        return {'tests':'passed'}
    monkeypatch.setattr(experiment_runner,'run_candidate_unit_tests',check)
    prompts=[]
    def invoke(*a,**kw):
        prompts.append(a[4])
        return {'response':{'candidate_id':'fixture','train_contract':True,'load_contract':True,
                            'predict_contract':True,'new_dependencies':[],'external_weights':[]}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    return store,tree,prompts,checks


def test_failed_cpu_validation_is_repaired_in_same_worktree(tmp_path,monkeypatch):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,1)
    result=controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert len(prompts)==len(checks)==2
    assert 'EXACT CPU ERROR' not in prompts[0] and 'EXACT CPU ERROR' in prompts[1]
    assert 'CPU-only sandbox' in prompts[0]
    assert result['worktree']==str(tree) and result['repair_used']
    assert result['cpu_validation']['tests']=='passed'


def test_cpu_validation_repair_is_not_unbounded(tmp_path,monkeypatch):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,10)
    with pytest.raises(ValueError,match='EXACT CPU ERROR'):
        controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert len(prompts)==len(checks)==2
    assert store.list_runs()==[]


def test_cuda_repair_reuses_workspace_and_has_no_second_repair(tmp_path,monkeypatch):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,1)
    monkeypatch.setattr(controller,'create_candidate_worktree',lambda *a: pytest.fail('repair recreated worktree'))
    with pytest.raises(ValueError,match='EXACT CPU ERROR'):
        controller.implement_proposal(store,'fixture',validation_runtime=tmp_path,repair_feedback='CUDA traceback sentinel')
    assert len(prompts)==1 and 'CUDA traceback sentinel' in prompts[0]


@pytest.mark.parametrize('failure_prefix', ['CUDA smoke validation FAILED: ', 'Candidate process FAILED: '])
@pytest.mark.parametrize('experiment_cap', [1,8])
def test_failed_planned_job_is_charged_and_never_replayed_as_a_repaired_comparison(tmp_path,monkeypatch,experiment_cap,failure_prefix):
    from orchestrator import autonomy,codex_runner
    from orchestrator.campaign_state import CampaignState
    from plan_fixtures import experiment_plan
    cfg=autonomy.load_campaign_config(Path(__file__).parents[1]/'configs/autonomy_real_feedback.yaml')
    cfg['research']['fresh_hypotheses']=False
    from scripts.demo_run import make_fixture
    from benchmark.data_guard import build_manifest
    make_fixture(tmp_path/'good_data')
    (tmp_path/'good_manifest.json').write_text(json.dumps(build_manifest(tmp_path/'good_data')))
    cfg['limits']['max_experiments']=experiment_cap
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store);cid=campaigns.create(cfg)
    plan=experiment_plan('fixture',seeds=(17,),arms=('control',))
    store.add_proposal({'hypothesis_id':'fixture','priority_rank':1,'new_dependencies':[],'experiment_plan':plan})
    campaigns.event(cid,'EVIDENCE_ADAPTED_HYPOTHESES',hypotheses=['fixture'])
    source=tmp_path/'model.py';source.write_text('# fixture');calls=[]
    def implement(*a,**kw):
        calls.append(kw)
        return {'candidate_module':str(source),'cpu_validation':{'tests':'passed'},'repair_used':False}
    monkeypatch.setattr(controller,'implement_proposal',implement)
    monkeypatch.setattr(experiment_runner,'candidate_file',lambda *a:source)
    monkeypatch.setattr(codex_runner,'commit_candidate_snapshot',lambda *a:'fixture-sha')
    def queue(*a,**kw):
        assert kw['cuda_smoke_required']
        config=dict(kw['effective_config'],campaign_id=cid)
        rid=store.create_run('fixture','fixture','juice_bottle',config=config,run_id=kw['run_id'])
        for state in ('IMPLEMENTED','VALIDATED','QUEUED'):store.transition(rid,state)
        return rid
    monkeypatch.setattr(controller,'queue_candidate',queue)
    jobs=[]
    def worker(*a,**kw):
        rid=kw['expected_run_id'];jobs.append(rid);store.transition(rid,'RUNNING')
        store.account_gpu_usage(rid,2);store.transition(rid,'FAILED',reason=failure_prefix+'EXACT CUDA TRACEBACK')
        return {'run_id':rid,'status':'FAILED'}
    monkeypatch.setattr(controller,'run_worker_once',worker)
    def critic(*a,**kw):
        assert 'EXACT CUDA TRACEBACK' in a[4]
        return {'response':{'plan_id':plan['plan_id'],'comparison_status':'INCOMPLETE','hypothesis_outcome':'unassessed',
            'causal_claim':False,'recommended_next_action':'investigate','objective_measured_results':'implementation failure'}}
    monkeypatch.setattr(controller,'_budgeted_invoke_once',critic)
    autonomy.run_cycle(store,campaigns,cid,tmp_path)
    assert len(jobs)==len(calls)==1 and campaigns.usage(cid)['gpu_seconds']==2
    assert store.get_run(jobs[0])['status']=='FAILED'
    assert campaigns.get(cid)['consecutive_failures']==1
    with store.connect() as con:
        assert con.execute('SELECT status FROM hypotheses WHERE hypothesis_id=?',('fixture',)).fetchone()[0]=='INCOMPLETE'


def test_repaired_snapshot_keeps_both_git_versions(tmp_path):
    from orchestrator.codex_runner import commit_candidate_snapshot
    repo=tmp_path/'repo';repo.mkdir()
    def git(*args):
        return subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','user.name=Fixture',
            '-c','user.email=fixture@example.invalid','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init')
    (repo/'README.md').write_text('fixture')
    git('add','README.md');git('commit','-m','initial')
    tree=repo/'candidates/worktrees/fixture'
    git('worktree','add','--detach',str(tree),'HEAD')
    folder=tree/'candidates/fixture';folder.mkdir(parents=True)
    model=folder/'model.py';model.write_text('original source')
    first=commit_candidate_snapshot(repo,'fixture')
    model.write_text('repaired source')
    second=commit_candidate_snapshot(repo,'fixture')
    assert first!=second
    assert git('show',first+':candidates/fixture/model.py')=='original source'
    assert git('show',second+':candidates/fixture/model.py')=='repaired source'
    assert commit_candidate_snapshot(repo,'fixture')==second


def test_implementer_removes_extra_license_on_bounded_repair(tmp_path,monkeypatch):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,0)
    folder=tree/'candidates/fixture'
    original=controller.budgeted_invoke
    def invoke(*a,**kw):
        if not prompts:
            (folder/'LICENSE').write_text('generated licensing note')
        else:
            assert 'Unapproved candidate files' in a[4] and 'LICENSE' in a[4]
            assert 'README.md' in a[4]
            (folder/'README.md').write_text('License and provenance recorded here')
            (folder/'LICENSE').unlink()
        return original(*a,**kw)
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    result=controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert len(prompts)==2 and len(checks)==1 and result['repair_used']
    assert not (folder/'LICENSE').exists()
    assert len([e for e in store.events() if e['kind']=='REPAIR_REQUESTED'])==1


def test_unremoved_extra_file_never_passes_validation(tmp_path,monkeypatch):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,0)
    (tree/'candidates/fixture/LICENSE').write_text('extra')
    with pytest.raises(ValueError,match='Unapproved candidate files'):
        controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert len(prompts)==2 and checks==[]
    assert (tree/'candidates/fixture/LICENSE').is_file()  # Controller never deletes it.


@pytest.mark.parametrize('unsafe',['outside','symlink','dependency'])
def test_real_permission_violations_do_not_get_automatic_repair(tmp_path,monkeypatch,unsafe):
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,0)
    if unsafe=='outside':
        monkeypatch.setattr(controller.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=' M orchestrator/controller.py\n',returncode=0))
    elif unsafe=='symlink':
        outside=tmp_path/'sentinel';outside.write_text('protected')
        (tree/'candidates/fixture/LICENSE').symlink_to(outside)
    else:
        original=controller.budgeted_invoke
        def invoke(*a,**kw):
            result=original(*a,**kw)
            result['response']['new_dependencies']=['new_package']
            return result
        monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    with pytest.raises(PermissionError):
        controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert len(prompts)==1 and checks==[]
