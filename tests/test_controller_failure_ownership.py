"""Offline controller regressions: real active-campaign state, no research calls."""
import json
import subprocess
from pathlib import Path

import pytest

from orchestrator import controller, experiment_runner
from orchestrator.campaign_state import CampaignState
from orchestrator.state_store import StateStore
from test_time_campaigns import simple
from plan_fixtures import experiment_plan


@pytest.fixture
def active_implementation(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store)
    cid=campaigns.create(simple('new'))
    plan=experiment_plan('fixture',seeds=(607,))
    plan.update(schema_version='4.0',confirmation=False,kind='exploratory',training={'batch_size':8,'size':128},controlled_variables=['batch_size','size'])
    plan.pop('resource_budget')
    for arm in plan['arms']:arm.update(execution='frozen',training_overrides={},inference={})
    store.add_proposal({'hypothesis_id':'fixture','experiment_plan':plan})
    tree=tmp_path/'candidates/worktrees/fixture';folder=tree/'candidates/fixture';folder.mkdir(parents=True)
    (folder/'model.py').write_text('def diagnose(predictor, images, config): return {}\n')
    (folder/'test_model.py').write_text('def test_cpu(): assert True\n')
    subprocess.run(['git','init',str(tree)],check=True,capture_output=True)
    monkeypatch.setattr(controller,'REPO',tmp_path)
    monkeypatch.setenv('LOCO_GOOD_ROOT',str(tmp_path))
    calls=[];probes=[]
    def invoke(*args,**kwargs):
        calls.append((args,kwargs))
        return {'response':{'candidate_id':'fixture','implemented_files':['model.py'],'train_contract':False,'load_contract':False,
                            'predict_contract':False,'new_dependencies':[],'external_weights':[]}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',lambda *a,**k:probes.append(k) or True)
    monkeypatch.setattr(experiment_runner,'run_candidate_unit_tests',lambda *a,**k:{'tests':'passed'})
    return store,campaigns,cid,tmp_path,folder,calls,probes


def test_actual_active_campaign_frozen_validation_does_not_require_training(active_implementation,monkeypatch):
    store,c,cid,runtime,folder,calls,probes=active_implementation
    monkeypatch.setattr(experiment_runner,'require_training_phases',lambda *a,**k:pytest.fail('frozen provider must not train'))
    created=c.get(cid)['created_at']
    result=controller.implement_proposal(store,'fixture',campaign=(c,cid),personal_mode=True,validation_runtime=runtime)
    assert result['cpu_validation']=={'tests':'passed'}
    assert len(calls)==1 and probes[0]['diagnostic_only']
    assert probes[0]['timeout']>14000
    assert c.get(cid)['created_at']==created
    assert store.list_runs()==[]


def test_active_campaign_new_training_still_checks_training_interface(active_implementation,monkeypatch):
    store,c,cid,runtime,folder,calls,probes=active_implementation
    with store.connect() as con:
        p=json.loads(con.execute('select proposal_json from hypotheses where hypothesis_id=?',('fixture',)).fetchone()[0])
        for a in p['experiment_plan']['arms']:a['execution']='train'
        con.execute('update hypotheses set proposal_json=? where hypothesis_id=?',(json.dumps(p),'fixture'))
    (folder/'model.py').write_text("TRAINING_PROTOCOL='elapsed-training-loop-v1'\ndef training_steps(*a): yield\ndef train(config, manifest, output): pass\ndef load(*a): pass\ndef predict(*a): pass\ndef diagnose(*a): pass\n")
    def invoke(*args,**kwargs):
        calls.append(1)
        return {'response':{'candidate_id':'fixture','implemented_files':['model.py'],'train_contract':True,'load_contract':True,
            'predict_contract':True,'new_dependencies':[],'external_weights':[]}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    result=controller.implement_proposal(store,'fixture',campaign=(c,cid),personal_mode=True,validation_runtime=runtime)
    assert result['cpu_validation']['tests']=='passed' and len(calls)==1
    assert not probes[0].get('diagnostic_only',False)


def test_candidate_contract_error_is_repaired_once(active_implementation,monkeypatch):
    store,c,cid,runtime,folder,calls,probes=active_implementation
    count=[0]
    def tests(*a,**k):
        count[0]+=1
        if count[0]==1:raise controller.CandidateValidationError('candidate test assertion failed: wrong output shape')
        return {'tests':'passed'}
    monkeypatch.setattr(experiment_runner,'run_candidate_unit_tests',tests)
    result=controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert result['repair_used'] and len(calls)==2
    assert 'wrong output shape' in calls[-1][0][4]


@pytest.mark.parametrize('error',[UnboundLocalError('a local binding defect'),NameError('missing import'),
                                 TypeError('trusted helper bug'),ValueError('trusted state transition defect')])
def test_raw_controller_fault_never_becomes_candidate_repair(active_implementation,monkeypatch,error):
    from orchestrator.controller_errors import TrustedControllerError
    store,c,cid,runtime,folder,calls,probes=active_implementation
    # Failure in trusted stage commit, not generated candidate work.
    def commit(*a,**k):raise error
    monkeypatch.setattr(c,'validated_stage',commit)
    with pytest.raises(TrustedControllerError) as caught:
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert len(calls)==1 and caught.value.fatal
    events=c.events(cid,100)
    fault=next(json.loads(e['payload_json']) for e in events if e['kind']=='TRUSTED_INFRASTRUCTURE_ERROR')
    assert type(error).__name__ in fault['traceback'] and fault['fingerprint']
    assert not any(e['kind']=='REPAIR_REQUESTED' for e in events)


def test_frozen_programming_bug_quarantines_branch_across_restart_and_hypotheses(active_implementation,monkeypatch):
    from orchestrator.controller_errors import TrustedControllerError
    store,c,cid,runtime,folder,calls,probes=active_implementation
    def broken(*a,**kw):raise UnboundLocalError('fault in protected probe infrastructure')
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',broken)
    with pytest.raises(TrustedControllerError) as first:
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert not first.value.fatal and len(calls)==1
    restarted=CampaignState(StateStore(store.path))
    with pytest.raises(TrustedControllerError) as second:
        controller.implement_proposal(store,'fixture',campaign=(restarted,cid),validation_runtime=runtime)
    assert len(calls)==1 and first.value.detail['fingerprint']==second.value.detail['fingerprint']
    store.add_proposal({'hypothesis_id':'different','experiment_plan':dict(experiment_plan(),arms=[{'execution':'frozen'}])})
    with pytest.raises(TrustedControllerError):
        controller.implement_proposal(store,'different',campaign=(restarted,cid),validation_runtime=runtime)
    assert len(calls)==1  # no invocation or new worktree to rediscover the same fault
    assert len([e for e in c.events(cid,100) if e['kind']=='TRUSTED_INFRASTRUCTURE_ERROR'])==1


def test_nonfatal_trusted_failure_restores_evidence_and_continues_without_resetting_clock(tmp_path,monkeypatch):
    from orchestrator import autonomy
    from orchestrator.controller_errors import trusted_failure
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple('new'));created=c.get(cid)['created_at']
    monkeypatch.setattr(autonomy,'preflight',lambda *a:None)
    counts=[]
    def cycle(target,campaigns,ident,runtime):
        counts.append(1)
        if len(counts)==1:
            try:raise NameError('isolated diagnostic infrastructure fault')
            except NameError as exc:raise trusted_failure(exc,'implementer.frozen_diagnostics',fatal=False)
        assert campaigns.get(ident)['created_at']==created
        assert any(e['kind']=='RECOVERY_RESTORED' for e in campaigns.events(ident,100))
        assert campaigns.get(ident)['consecutive_failures']==0
        campaigns.set_status(ident,'COMPLETED','safe independent fixture complete')
    monkeypatch.setattr(autonomy,'run_cycle',cycle)
    autonomy.run_forever(cid,tmp_path,store.path,idle_seconds=0)
    assert len(counts)==2 and c.get(cid)['status']=='COMPLETED'


def test_unclassified_internal_execution_error_stops_without_ten_retries(tmp_path,monkeypatch):
    from orchestrator import autonomy
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple('new'));calls=[]
    monkeypatch.setattr(autonomy,'preflight',lambda *a:None)
    def cycle(*a):calls.append(1);raise RuntimeError('internal controller execution defect')
    monkeypatch.setattr(autonomy,'run_cycle',cycle)
    autonomy.run_forever(cid,tmp_path,store.path,idle_seconds=0)
    assert calls==[1] and c.get(cid)['status']=='FAILED'
    assert any(e['kind']=='TRUSTED_INFRASTRUCTURE_ERROR' for e in c.events(cid,100))


def test_missing_or_empty_implementation_cannot_pass_validation(active_implementation):
    from orchestrator.controller_errors import CandidateValidationError
    store,c,cid,runtime,folder,calls,probes=active_implementation
    (folder/'model.py').unlink()
    with pytest.raises(CandidateValidationError,match='/workspace/candidates/fixture/model.py'):
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert len(calls)==2 and not probes and not store.list_runs()


def test_readout_only_plan_never_calls_implementer(tmp_path,monkeypatch):
    from orchestrator import autonomy
    monkeypatch.setattr(controller,'implement_proposal',lambda *a,**kw:pytest.fail('readout-only must reuse source'))
    p=experiment_plan();p['diagnostics']=[]
    for a in p['arms']:a['execution']='frozen'
    assert autonomy.prepare_plan_candidate(None,None,None,tmp_path,p,{}) is None


def test_raw_valueerror_in_trusted_probe_is_not_misclassified_by_its_type(active_implementation,monkeypatch):
    from orchestrator.controller_errors import TrustedControllerError
    store,c,cid,runtime,folder,calls,probes=active_implementation
    def broken(*a,**k):raise ValueError('trusted helper internally converted an invalid value')
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',broken)
    with pytest.raises(TrustedControllerError):
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert len(calls)==1 and not any(e['kind']=='REPAIR_REQUESTED' for e in c.events(cid,100))


def test_empty_response_cannot_reuse_an_unvalidated_file_as_success(active_implementation,monkeypatch):
    store,c,cid,runtime,folder,calls,probes=active_implementation
    def invoke(*a,**k):
        calls.append(1)
        return {'response':{'candidate_id':'fixture','implemented_files':[],'train_contract':False,
                            'load_contract':False,'predict_contract':False,'new_dependencies':[],'external_weights':[]}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    with pytest.raises(controller.CandidateValidationError,match='empty or mismatched implemented_files'):
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert len(calls)==2 and not probes and not store.list_runs()


def test_quarantine_marks_unfinished_plan_incomplete_and_keeps_frozen_plan_bytes(tmp_path):
    from orchestrator import autonomy
    from orchestrator.controller_errors import trusted_failure
    from orchestrator.experiment_plan import register_plan,digest
    store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple('new'))
    plan=experiment_plan('fixture');before=digest(plan)
    store.add_proposal({'hypothesis_id':'fixture','experiment_plan':plan});register_plan(store,c,cid,plan)
    c.update(cid,current_hypothesis='fixture');created=c.get(cid)['created_at']
    try:raise NameError('protected validation dependency missing')
    except NameError as exc:failure=trusted_failure(exc,'implementer.frozen_diagnostics',fatal=False)
    autonomy.handle_trusted_failure(store,c,cid,tmp_path,failure)
    assert c.get(cid)['status']=='RUNNING' and c.get(cid)['created_at']==created
    result=next(json.loads(e['payload_json'])['result'] for e in c.events(cid,100) if e['kind']=='EXPERIMENT_PLAN_RESULT')
    assert result['status']=='INCOMPLETE' and result['recorded_jobs']==0
    assert not result['scientific_eligibility']['promotion_eligible']
    assert digest(plan)==before and not store.list_runs()
    assert any(e['kind']=='PLAN_ABANDONED' for e in c.events(cid,100))


from test_model_registry import lab as retained_lab


def test_trusted_recovery_preserves_retained_checkpoint_and_selection(retained_lab,monkeypatch):
    from orchestrator import autonomy,model_registry as registry
    from orchestrator.controller_errors import trusted_failure
    store,c,cid,repo,add=retained_lab
    rid=add();registry.initialize(store,repo);before=registry.states(store)
    c.set_status(cid,'COMPLETED');cid=c.create(simple('previous_best'));created=c.get(cid)['created_at']
    monkeypatch.setattr(autonomy,'REPO',repo)
    try:raise UnboundLocalError('isolated validation binding')
    except UnboundLocalError as exc:failure=trusted_failure(exc,'implementer.frozen_diagnostics',fatal=False)
    autonomy.handle_trusted_failure(store,c,cid,store.path.parent,failure)
    assert registry.states(store)==before and c.get(cid)['created_at']==created
    recovery=next(json.loads(e['payload_json']) for e in c.events(cid,100) if e['kind']=='RECOVERY_RESTORED')
    assert recovery['verified_state']['models'][0]['run_id']==rid
    for model in registry.retained(store).values():registry.verify_archive(model)


@pytest.mark.skipif(__import__('os').getenv('LOCO_TEST_BWRAP')!='1',reason='CPU Bubblewrap integration')
def test_implementer_mount_has_real_editing_interface_without_calling_codex(tmp_path,monkeypatch):
    import tempfile
    from orchestrator.codex_sandbox import command
    repo=Path(__file__).parents[1]
    home=tmp_path/'fake_home';(home/'.codex').mkdir(parents=True)
    (home/'.codex/auth.json').write_text('dummy-auth-not-a-real-credential')
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:home))
    parent=repo/'candidates/worktrees';parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent,prefix='ownership-fixture-') as folder:
        hid=Path(folder).name;candidate=Path(folder)/'candidates'/hid;candidate.mkdir(parents=True)
        cmd=command(repo,'implementer',repo/'schemas/implementation.schema.json',tmp_path/'response.json','fixture',candidate)
        script=f'''from pathlib import Path
for name in ('AGENTS.md','RESEARCH_CHARTER.md','research_program.md'):
    assert Path('/workspace',name).read_text()
p=Path('/workspace/candidates/{hid}/model.py')
p.write_text('def diagnose(*args): return {{}}')
assert p.read_text().startswith('def diagnose')
assert not Path('/workspace/orchestrator/controller.py').exists()
assert not Path('/dev/nvidia0').exists()
print('workspace files and CPU command interface work')
'''
        # Exercise the real mounts with a CPU command, never a Codex agent call.
        result=subprocess.run(cmd[:cmd.index('--')]+['--','/usr/bin/python3','-c',script],
                              capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        assert (candidate/'model.py').is_file()
        assert 'CPU command interface work' in result.stdout


def test_corrupt_recovery_reference_halts_instead_of_retrying(retained_lab,monkeypatch):
    from orchestrator import autonomy
    from orchestrator.experiment_plan import baseline_reference
    from orchestrator.controller_errors import trusted_failure
    store,c,cid,repo,add=retained_lab
    rid=add();reference,_,_=baseline_reference(store,rid,repo)
    c.set_status(cid,'COMPLETED');cid=c.create(simple('new'))
    c.event(cid,'VERIFIED_RESEARCH_STATE',references=[reference])
    monkeypatch.setattr(autonomy,'REPO',repo)
    with store.connect() as con:con.execute("UPDATE runs SET git_sha='invalid-fixture' WHERE run_id=?",(rid,))
    try:raise NameError('isolated preflight failure')
    except NameError as exc:failure=trusted_failure(exc,'implementer.frozen_diagnostics',fatal=False)
    autonomy.handle_trusted_failure(store,c,cid,store.path.parent,failure)
    assert c.get(cid)['status']=='FAILED'
    assert 'verified recovery unavailable' in c.get(cid)['reason']
    faults=[json.loads(e['payload_json']) for e in c.events(cid,100) if e['kind']=='TRUSTED_INFRASTRUCTURE_ERROR']
    assert any(f['operation']=='campaign.trusted_recovery' and 'PermissionError' in f['traceback'] for f in faults)


@pytest.mark.parametrize('failure',[NameError('validator missing import'),ValueError('internal validator state fault')])
def test_planning_interface_does_not_turn_unknown_internal_faults_into_bad_agent_plans(monkeypatch,failure):
    from orchestrator import experiment_plan as execution,planning_contract
    def broken(*a,**k):raise failure
    monkeypatch.setattr(execution,'validate_plan',broken)
    with pytest.raises(type(failure)):
        planning_contract.inspect_plan(experiment_plan('fixture'))



def test_cpu_worker_cannot_claim_success_without_reporting_validation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from orchestrator import sandbox
    module=tmp_path/'model.py';module.write_text('')
    (tmp_path/'test_model.py').write_text('def test_fixture(): pass')
    monkeypatch.setattr(sandbox,'command',lambda *a,**kw:['fixture-only'])
    monkeypatch.setattr(experiment_runner.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    with pytest.raises(RuntimeError,match='did not report a validation result'):
        experiment_runner.run_candidate_unit_tests(tmp_path,module,tmp_path,tmp_path,'fixture')


def test_personal_implementer_receives_exact_workspace_and_real_interface(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from orchestrator import codex_runner,codex_sandbox
    repo=Path(__file__).parents[1]
    # No worktree or model is created: the invocation is intercepted before any
    # process or actual personal-sandbox preparation runs.
    # Use a real temporary directory under the permitted candidate root.
    import tempfile
    with tempfile.TemporaryDirectory(dir=repo/'candidates/worktrees',prefix='interface-fixture-') as folder:
        tree=Path(folder);captured={};output={}
        schema=tmp_path/'schema.json';schema.write_text('{"type":"object","properties":{},"required":[],"additionalProperties":false}')
        def cmd(root,role,schema_file,response,task,candidate,**kwargs):
            captured.update(task=task,candidate=candidate);output['path']=response
            return ['mocked-codex-never-executed']
        def run(*a,**kw):
            output['path'].write_text('{}')
            return SimpleNamespace(returncode=0,stderr='')
        monkeypatch.setattr(codex_sandbox,'command',cmd)
        monkeypatch.setattr(codex_runner.subprocess,'run',run)
        codex_runner.invoke('implementer',repo,schema,'offline contract',worktree=tree,personal_mode=True)
        assert f'/workspace/candidates/{tree.name}/model.py' in captured['task']
        assert 'No separate trusted-runner command tool exists' in captured['task']
        assert captured['candidate']==tree/'candidates'/tree.name


@pytest.mark.skipif(__import__('os').getenv('LOCO_TEST_BWRAP')!='1',reason='active-campaign CPU Bubblewrap integration')
def test_active_frozen_controller_runs_actual_import_and_cpu_tests(tmp_path,monkeypatch):
    import tempfile
    repo=Path(__file__).parents[1]
    with tempfile.TemporaryDirectory(dir=repo/'candidates/worktrees',prefix='active-frozen-fixture-') as folder:
        tree=Path(folder);hid=tree.name;candidate=tree/'candidates'/hid;candidate.mkdir(parents=True)
        subprocess.run(['git','init',str(tree)],check=True,capture_output=True)
        (candidate/'model.py').write_text('''def diagnose(predictor, normal, config):
    assert len(config['calibration_good_images'])==18
    assert len(normal)==36
    return {'metrics':{'mean_score':float(normal.mean())},
            'per_image':[{'source_id':key,'metrics':{'mean_score':float(im.mean())}}
                         for key,im in zip(config['assessment_source_ids'],normal)]}
''')
        (candidate/'test_model.py').write_text('''import importlib.util
import numpy as np
from benchmark.train_adapter import diagnostic_inputs,validate_diagnostic_output

def test_normal_calibration_contract():
    spec=importlib.util.spec_from_file_location('fixture','/candidate/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    declaration={'diagnostic_id':'fixture_check','config':{'min_calibration_images':18},'required_metrics':['mean_score']}
    cal=np.zeros((18,3,16,16),np.float32);normal=np.ones((36,3,16,16),np.float32)
    ids=['a'+str(i) for i in range(36)]
    config=diagnostic_inputs(declaration,cal,normal,['c'+str(i) for i in range(18)],ids,.5)
    validate_diagnostic_output(declaration,module.diagnose(None,normal,config),ids,['mean_score'])
''')
        store=StateStore(tmp_path/'ledger.sqlite');c=CampaignState(store);cid=c.create(simple('new'))
        plan=experiment_plan(hid,seeds=(607,));plan.update(schema_version='4.0',confirmation=False,kind='exploratory',
            training={'batch_size':8,'size':16},controlled_variables=['batch_size','size'])
        plan.pop('resource_budget')
        for arm in plan['arms']:arm.update(execution='frozen',training_overrides={},inference={})
        store.add_proposal({'hypothesis_id':hid,'experiment_plan':plan})
        data=tmp_path/'data';data.mkdir();monkeypatch.setenv('LOCO_GOOD_ROOT',str(data))
        def response(*a,**k):return {'response':{'candidate_id':hid,'implemented_files':['model.py','test_model.py'],
            'train_contract':False,'load_contract':False,'predict_contract':False,'new_dependencies':[],'external_weights':[]}}
        monkeypatch.setattr(controller,'budgeted_invoke',response)
        result=controller.implement_proposal(store,hid,campaign=(c,cid),personal_mode=True,validation_runtime=tmp_path)
        assert result['cpu_validation']['tests']=='passed' and not result['repair_used']
        assert '1 passed' in result['cpu_validation']['output']
        assert not store.list_runs() and c.get(cid)['consecutive_failures']==0


def test_campaign_deadline_during_cpu_probe_is_not_a_trusted_defect(active_implementation,monkeypatch):
    import time
    from orchestrator.autonomy import CampaignHalted
    store,c,cid,runtime,folder,calls,probes=active_implementation
    created=c.get(cid)['created_at']
    def timeout(*a,**k):
        monkeypatch.setattr(time,'time',lambda:created+14401)
        raise TimeoutError('CPU probe reached remaining research deadline')
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',timeout)
    with pytest.raises(CampaignHalted,match='wall-time budget reached'):
        controller.implement_proposal(store,'fixture',campaign=(c,cid),validation_runtime=runtime)
    assert c.get(cid)['status']=='LIMIT_REACHED' and c.get(cid)['created_at']==created
    assert len(calls)==1
    assert not any(e['kind'] in {'REPAIR_REQUESTED','TRUSTED_INFRASTRUCTURE_ERROR'} for e in c.events(cid,100))
