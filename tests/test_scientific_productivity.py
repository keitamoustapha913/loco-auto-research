"""Offline replays of campaign interface failures; no research calls or GPU work."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.train_adapter import (CandidateContractError, diagnostic_inputs,
    validate_diagnostic_resources, validate_diagnostic_output, load_candidate)
from orchestrator.experiment_runner import verify_candidate_source
from orchestrator.planning_contract import comparison_controls, inspect_plan
from plan_fixtures import experiment_plan


def declaration(n=18):
    return {'diagnostic_id':'operating_point', 'config':{'min_calibration_images':n},
            'required_metrics':['realized_good_pixel_fpr']}


def test_replay_missing_18_calibration_sources_with_production_inputs():
    cal=np.full((18,3,16,16),.2,np.float32)
    assess=np.full((36,3,16,16),.3,np.float32)
    ids=[f'c{i}' for i in range(18)]; assessment_ids=[f'a{i}' for i in range(36)]
    cfg=diagnostic_inputs(declaration(),cal,assess,ids,assessment_ids,.4)
    # Exact dependency of the previously failed operating-point implementation.
    assert len(cfg['calibration_good_images'])==18
    assert len(cfg['calibration_source_ids'])==18
    assert cfg['frozen_normal_threshold']==.4
    assert cfg['assessment_source_ids']==assessment_ids
    cfg['calibration_good_images'][0]=0
    assert np.all(cal==np.float32(.2))  # isolated working copy, not shared assessment
    assert np.all(assess==np.float32(.3))


@pytest.mark.parametrize('problem',['overlap','shape','duplicate','threshold'])
def test_diagnostic_inputs_fail_closed(problem):
    c=np.zeros((18,3,16,16));a=np.zeros((36,3,16,16))
    ci=[f'c{i}' for i in range(18)]; ai=[f'a{i}' for i in range(36)];threshold=.5
    if problem=='overlap':ai[0]=ci[0]
    if problem=='shape':a=a[:,:,:8,:]
    if problem=='duplicate':ci[1]=ci[0]
    if problem=='threshold':threshold=float('nan')
    with pytest.raises(CandidateContractError):diagnostic_inputs(declaration(),c,a,ci,ai,threshold)


def test_unavailable_calibration_rejected_before_candidate_import(tmp_path,monkeypatch):
    from scripts import run_candidate
    manifest={'images':[{'split':'train/good','category':'fixture'}]+[
        {'split':'validation/good','category':'fixture'} for _ in range(54)]}
    monkeypatch.setattr(run_candidate,'load_manifest',lambda _: (manifest,list(range(55))))
    monkeypatch.setattr(run_candidate,'load_candidate',lambda *a,**k:pytest.fail('must reject before candidate import or training'))
    with pytest.raises(CandidateContractError,match='requests 19, only 18'):
        run_candidate.run(tmp_path/'model.py',tmp_path/'manifest.json',
                          {'category':'fixture','size':16,'diagnostics':[declaration(19)]},tmp_path/'out')


def test_frozen_provider_requires_diagnose_not_training(tmp_path):
    module=tmp_path/'model.py'
    module.write_text("def diagnose(predictor, assessment_good_images, diagnostic_config):\n    return {'metrics': {'realized_good_pixel_fpr': 0.0}}\n")
    assert 'diagnose' in verify_candidate_source(module,diagnostic_only=True)['protocol_symbols']
    assert callable(load_candidate(module,diagnostic_only=True).diagnose)
    with pytest.raises(ValueError,match='train/load/predict'):verify_candidate_source(module)
    module.write_text('def train(*args): pass\n')
    with pytest.raises(ValueError,match='requires diagnose'):verify_candidate_source(module,diagnostic_only=True)


def test_replay_frozen_implementer_false_attestations(tmp_path,monkeypatch):
    from orchestrator import controller,experiment_runner
    from orchestrator.state_store import StateStore
    store=StateStore(tmp_path/'ledger.sqlite')
    plan=experiment_plan('fixture',seeds=(503,))
    for arm in plan['arms']:
        arm.update(execution='frozen',training_overrides={})
    store.add_proposal({'hypothesis_id':'fixture','experiment_plan':plan})
    tree=tmp_path/'candidates/worktrees/fixture'; folder=tree/'candidates/fixture';folder.mkdir(parents=True)
    (folder/'model.py').write_text('def diagnose(predictor, images, config): return {}')
    (folder/'test_model.py').write_text('def test_cpu(): assert True')
    import subprocess
    subprocess.run(['git','init',str(tree)],check=True,capture_output=True)
    monkeypatch.setattr(controller,'REPO',tmp_path)
    monkeypatch.setattr(controller,'create_candidate_worktree',lambda *a:tree)
    monkeypatch.setenv('LOCO_GOOD_ROOT',str(tmp_path))
    seen=[]
    def invoke(*a,**k):
        seen.append(a[4])
        return {'response':{'candidate_id':'fixture','implemented_files':['model.py'],'train_contract':False,'load_contract':False,
                            'predict_contract':False,'new_dependencies':[],'external_weights':[]}}
    monkeypatch.setattr(controller,'budgeted_invoke',invoke)
    monkeypatch.setattr(experiment_runner,'require_training_phases',lambda *a,**k:pytest.fail('frozen diagnostics must not train'))
    monkeypatch.setattr(experiment_runner,'probe_candidate_interface',lambda *a,**kw:kw['diagnostic_only'])
    monkeypatch.setattr(experiment_runner,'run_candidate_unit_tests',lambda *a,**k:{'tests':'passed'})
    result=controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert not result['repair_used']
    assert len(seen)==1 and 'do not implement train' in seen[0]


def test_diagnostic_pair_contract_and_frozen_threshold():
    d=declaration();result={'metrics':{'realized_good_pixel_fpr':0.},
                         'per_image':[{'source_id':'a','metrics':{'recall':.5}}]}
    assert validate_diagnostic_output(d,result,['a'],['recall']) is result
    bad=copy.deepcopy(result);bad['per_image'][0]['metrics']={}
    with pytest.raises(CandidateContractError,match='omitted'):validate_diagnostic_output(d,bad,['a'],['recall'])


def test_unmatched_historical_seed_is_explicitly_noncausal():
    p=experiment_plan(seeds=(503,));p['arms'][0]['fixed_checkpoint_seed']=461
    report=comparison_controls(p)
    assert report['actual_seeds_by_arm']['control']==[461]
    assert report['actual_seeds_by_arm']['intervention']==[503]
    assert not report['causal_claim_eligible']
    assert any('unmatched actual training seeds' in s for s in report['confounds'])
    assert p['seeds']==[503]  # never repair by relabeling historical seeds


def test_matched_controls_allow_causal_scope_without_claiming_reproducibility():
    p=experiment_plan(seeds=(7,19,31))
    p['training'].update(sampling_protocol='normal_seeded_stream_v1',sampling_seed=73)
    p['controlled_variables']=list(p['training'])
    report=comparison_controls(p)
    assert report['causal_claim_eligible'] and not report['confounds']
    assert report['sampling_stream_status']=='DECLARED_NOT_INDEPENDENTLY_VERIFIED'
    assert report['reproducibility_status']=='UNCONFIRMED'
    p['arms'][1]['training_overrides']['sampling_seed']=74
    assert not inspect_plan(p)['valid']


def test_frozen_training_override_and_size_controls_stay_strict():
    p=experiment_plan();p['arms'][0]['execution']='frozen'
    p['arms'][0]['training_overrides']={'batch_size':4}
    assert not inspect_plan(p)['valid']
    p=experiment_plan();p['training'].pop('size')
    assert not inspect_plan(p)['valid']


def test_provider_schema_still_supports_plain_names_and_semantic_controls():
    from pathlib import Path
    from orchestrator.planning_contract import constrain_output_schema
    from orchestrator.codex_runner import provider_output_schema
    root=Path(__file__).parents[1]
    schema=constrain_output_schema(json.loads((root/'schemas/experiment_plan_agent.schema.json').read_text()))
    assert provider_output_schema(schema)


@pytest.mark.skipif(__import__('os').getenv('LOCO_TEST_BWRAP')!='1',reason='requires Linux user namespaces; CPU only')
def test_real_bubblewrap_diagnostic_only_probe_and_calibration_inputs(tmp_path):
    import tempfile
    from pathlib import Path
    from orchestrator.experiment_runner import probe_candidate_interface
    from orchestrator.sandbox import command
    import subprocess
    repo=Path(__file__).parents[1]
    data=tmp_path/'data';data.mkdir()
    with tempfile.TemporaryDirectory(dir=repo/'candidates',prefix='diagnostic-contract-') as folder:
        module=Path(folder)/'model.py'
        module.write_text("def diagnose(predictor, normal, config):\n    assert len(config['calibration_good_images'])==18\n    assert len(normal)==36\n    return {'metrics':{'alarm':0.0}}\n")
        assert probe_candidate_interface(repo,module,data,tmp_path,'diagnostic_fixture',diagnostic_only=True)
        output=tmp_path/'out';output.mkdir()
        script='''import numpy as np
from benchmark.train_adapter import load_candidate,diagnostic_inputs,validate_diagnostic_output
m=load_candidate('/candidate/model.py',diagnostic_only=True)
d={'diagnostic_id':'operating_point','config':{'min_calibration_images':18},'required_metrics':['alarm']}
a=np.zeros((36,3,16,16),np.float32);c=np.ones((18,3,16,16),np.float32)
ids=[str(i) for i in range(36)]
config=diagnostic_inputs(d,c,a,['c'+str(i) for i in range(18)],ids,.2)
validate_diagnostic_output(d,m.diagnose(None,a,config),ids)
print('diagnostic input contract OK')
'''
        result=subprocess.run(command(repo,module,data,output,['-c',script]),capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        assert 'diagnostic input contract OK' in result.stdout


def test_probe_preserves_full_logs_and_decisive_exception_tail(tmp_path,monkeypatch):
    (tmp_path/'model.py').write_text('def diagnose(*args): return {}\n')
    from orchestrator import experiment_runner,sandbox
    monkeypatch.setattr(sandbox,'command',lambda *a,**kw:['fixture'])
    stderr='old context\n'*3000+'Traceback: diagnostic.py:143\nValueError: DECISIVE calibration contract\n'
    monkeypatch.setattr(experiment_runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout='preflight output\nCANDIDATE_IMPORT_RESULT=FAILED\n',stderr=stderr,returncode=1))
    with pytest.raises(ValueError,match='DECISIVE calibration contract'):
        experiment_runner.probe_candidate_interface(tmp_path,tmp_path/'model.py',tmp_path,tmp_path,'fixture',diagnostic_only=True)
    assert (tmp_path/'probes/fixture/interface_stderr.log').read_text()==stderr


def test_control_report_retains_observed_unequal_exposures_and_updates():
    from orchestrator.experiment_plan import digest
    p=experiment_plan(seeds=(7,))
    runs=[{'config_json':json.dumps({'plan_sha256':digest(p),'plan_arm':a['arm_id'],'seed':7}),
           'metrics_json':json.dumps({'protocol/optimizer_updates':count,'protocol/source_exposures':count*8})}
          for a,count in zip(p['arms'],[100,90])]
    report=comparison_controls(p,runs)
    assert [r['optimizer_updates'] for r in report['observed_training_counts']]==[100,90]
    assert [r['source_exposures'] for r in report['observed_training_counts']]==[800,720]
    assert 'does not imply equal' in report['interpretation']


def test_new_training_attestation_still_fails_with_actionable_fields(tmp_path,monkeypatch):
    from test_validation_repair import prepare
    from orchestrator import controller
    store,tree,prompts,checks=prepare(tmp_path,monkeypatch,0)
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**k:{'response':{
        'candidate_id':'fixture','implemented_files':['model.py'],'train_contract':False,'load_contract':True,'predict_contract':True,
        'new_dependencies':[],'external_weights':[]}})
    with pytest.raises(ValueError,match="required true fields missing: .*train_contract"):
        controller.implement_proposal(store,'fixture',validation_runtime=tmp_path)
    assert checks==[] and store.list_runs()==[]


def test_researcher_receives_actionable_followup_requirements(tmp_path,monkeypatch):
    from test_internal_research import batch
    from orchestrator import controller
    from orchestrator.state_store import StateStore
    prompts=[]
    monkeypatch.setattr(controller,'budgeted_invoke',lambda *a,**k:prompts.append(a[4]) or {
        'response':{'literature_request':'','proposals':batch()}})
    monkeypatch.setattr(controller,'scout_literature',lambda *a,**k:pytest.fail('literature is optional'))
    controller.add_research_proposals(StateStore(tmp_path/'ledger.sqlite'),'Investigate evidence',personal_mode=True)
    assert len(prompts)==1
    for term in ['competing explanations','support AND rejection','information gain per research time',
                 'sampling_protocol/sampling_seed','another narrower proxy is not automatically useful']:
        assert term in prompts[0]


def test_v4_confirmation_demands_matched_controls_but_exploration_remains_free():
    from orchestrator.model_registry import SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY
    p=experiment_plan(seeds=(7,19,31));p['schema_version']='4.0';p.pop('resource_budget')
    p['training']={'batch_size':8,'size':128,'optimizer':'SGD','learning_rate':.01}
    p['controlled_variables']=['batch_size','size'];p['confirmation']=False
    p['evaluation']['real_development_feedback']=True
    p['required_measurements']=[{'source':'real','key':k} for k in (SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY)]
    assert inspect_plan(p)['valid']  # exploratory work is not prohibited
    p['confirmation']=True
    report=inspect_plan(p)
    assert not report['valid']
    assert any(e['code']=='CONFIRMATION_PREREQUISITES' for e in report['errors'])
    p['training'].update(sampling_protocol='independent_normal_rng_v1',sampling_seed=73)
    p['controlled_variables']=list(p['training'])
    p['confirmation_requirements']={'normal_images':47,'max_additional_false_positives':1,
        'paired_controls':['initial_state','optimizer','input:0'],
        'uncertainty':'Paired training seeds and source images; marginal normal counts; repeated development selection.'}
    assert inspect_plan(p)['valid']
    assert inspect_plan(p)['scientific_eligibility']['promotion_eligible']  # ceiling, not an actual promotion


def test_planning_capability_recognizes_audited_elapsed_loop_baseline(tmp_path,monkeypatch):
    from orchestrator import experiment_plan as execution
    from orchestrator.planning_contract import baseline_capability,planning_instructions
    audit={'seed':461,'optimizer_updates':100,'source_exposures':800}
    (tmp_path/'run_manifest.json').write_text(json.dumps({'training_audit':audit}))
    config={'seed':461,'size':128,'batch_size':8,'training_budget_seconds':900,
            'training_completion_policy':'elapsed-training-loop-v1'}
    monkeypatch.setattr(execution,'verify_baseline',lambda *a:(b'def load(checkpoint, device): pass',
        {'config_json':json.dumps(config),'artifact_dir':str(tmp_path)}))
    cap=baseline_capability(None,{},tmp_path)
    assert cap['replication_audit_evidence_available']
    assert cap['actual_training_seed']==461 and cap['required_image_size']==128
    assert cap['frozen_training_overrides']=={}
    contract=json.loads(planning_instructions())
    assert {'sampling_protocol','sampling_seed'}<=set(contract['matched_training_controls_for_confirmation'])
