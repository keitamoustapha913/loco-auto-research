"""Offline readiness: source fidelity, real CPU fixtures and driver-only job doubles.

No GPU job or real-development evaluation is executed by these tests.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from orchestrator import controller,development_feedback,planning_contract
from orchestrator.experiment_plan import (digest,execute_plan,comparison_result,validate_plan,
                                          cell_run_id,training_settings)
from orchestrator.experiment_runner import (verify_candidate_source,probe_candidate_interface,
                                           run_candidate_unit_tests,candidate_file)
from test_experiment_plan import laboratory

REPO=Path(__file__).resolve().parents[1]
CID='categorical_regression_audited_redesign_v3'
FOLDER=REPO/'candidates'/CID
MANIFEST=json.loads((FOLDER/'candidate_manifest.json').read_text())
PLAN=MANIFEST['experiment_plan']


def test_frozen_plan_source_interfaces_and_original_scientific_code_are_preserved():
    assert digest(PLAN)=='50656331af3de0705eafab4cf90aa01e237db0fc8705f865ad702c288060c17b'
    validate_plan(PLAN)
    original=subprocess.check_output(['git','-C',str(REPO),'show',
        MANIFEST['origin_git_commit']+':candidates/'+MANIFEST['origin_candidate_id']+'/model.py'])
    assert hashlib.sha256(original).hexdigest()==MANIFEST['origin_source_sha256']
    old={n.name:n for n in ast.parse(original).body if hasattr(n,'name')}
    new={n.name:n for n in ast.parse((FOLDER/'model.py').read_text()).body if hasattr(n,'name')}
    for name in ('Completion','Predictor','load','_device','_hash'):
        assert ast.dump(new[name])==ast.dump(old[name]),name
    old_loop=next(n for n in old['train'].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    new_loop=copy.deepcopy(next(n for n in new['train'].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step'))
    def audit_only(n):
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='sampled':return True
        if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id in {'source_exposures','optimizer_updates'}:return True
        if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='index':return True
        return isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='mask_order'
    new_loop.body=[n for n in new_loop.body if not audit_only(n)]
    assert ast.dump(new_loop)==ast.dump(old_loop) # The complete learning computation is unchanged.
    def initialization(nodes):
        return [n for n in nodes if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'net','baseline','optimizer','rng','data'}]
    assert [ast.dump(n) for n in initialization(new['train'].body)]==[ast.dump(n) for n in initialization(old['train'].body)]
    assert verify_candidate_source(FOLDER/'model.py')['source_sha256']==hashlib.sha256((FOLDER/'model.py').read_bytes()).hexdigest()


@pytest.mark.skipif(os.getenv('LOCO_TEST_BWRAP')!='1',reason='explicit no-GPU Bubblewrap fixture test')
def test_actual_candidate_import_cpu_suite_and_full_runner_contract_in_no_gpu_boundary(tmp_path):
    module=FOLDER/'model.py';empty=tmp_path/'empty';empty.mkdir()
    assert probe_candidate_interface(REPO,module,empty,tmp_path,CID)
    result=run_candidate_unit_tests(REPO,module,empty,tmp_path,CID)
    assert '13 passed' in result['output']
    from orchestrator.sandbox import command
    from scripts.demo_run import make_fixture
    data=tmp_path/'artificial_good';make_fixture(data)
    # Candidate evaluation needs enough mask pixels, but only a tiny permitted
    # artificial source set. No real normal/development images are read.
    output=tmp_path/'cpu_runner';output.mkdir()
    script='''import hashlib,json,sys,time
from pathlib import Path
import torch
from benchmark.data_guard import build_manifest
from scripts.run_candidate import run
sys.path.insert(0,'/candidate')
import model
assert not torch.cuda.is_available()
torch.set_num_threads(2)
plan=json.loads(Path('/candidate/candidate_manifest.json').read_text())['experiment_plan']
manifest=Path('/output/fixture_manifest.json');manifest.write_text(json.dumps(build_manifest('/data')))
checks=[]
for arm in plan['arms']:
    # Disposable CPU smoke configuration, never a claimed execution of the plan.
    cfg=dict(plan['training'],**arm['training_overrides'],device='cpu',category='juice_bottle',seed=7,
             plan_id='CPU-CONTRACT-FIXTURE',plan_sha256='f'*64,plan_arm=arm['arm_id'],execution_kind='train',
             steps=2,batch_size=1,expected_optimizer_updates=2,expected_source_exposures=2,
             validation_smoke=True,synthetic_evaluation=True,diagnostics=plan['diagnostics'],inference={},trusted_source_sha256={})
    started=time.monotonic()
    result=run(Path('/candidate/model.py'),manifest,cfg,Path('/output')/arm['arm_id'])
    required={m['key'] for m in plan['required_measurements'] if m['source']=='run'}
    assert required<=result['metrics'].keys()
    for key in plan['diagnostics'][0]['required_metrics']:
        assert 'diagnostic/within_source_proxy/'+key in result['metrics']
    assert not any(k.startswith('diagnostic/diagnostic/') for k in result['metrics'])
    assert result['metrics']['protocol/optimizer_updates']==2
    assert result['metrics']['protocol/source_exposures']==2
    assert result['provenance']['checkpoint_sha256']==hashlib.sha256((Path('/output')/arm['arm_id']/'checkpoint/model.pt').read_bytes()).hexdigest()
    checks.append({'arm':arm['arm_id'],'device':'cpu','fixture_only':True,'wall_seconds':time.monotonic()-started,
                   'metrics':sorted(result['metrics'])})
Path('/output/readiness.json').write_text(json.dumps(checks))
print('CPU_RUNNER_CONTRACT_PASSED')
'''
    cmd=command(REPO,module,data,output,['-c',script],use_gpu=False)
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,(proc.stdout+proc.stderr)[-6000:]
    assert 'CPU_RUNNER_CONTRACT_PASSED' in proc.stdout
    checks=json.loads((output/'readiness.json').read_text())
    assert [r['arm'] for r in checks]==['categorical','deterministic']
    assert all(r['device']=='cpu' and r['fixture_only'] for r in checks)


def test_exact_frozen_plan_six_cells_dispatched_and_missing_evidence_is_incomplete(laboratory,monkeypatch):
    """GPU worker/real feedback are explicit test doubles: verifies dispatch only."""
    store,c,cid,runtime,repo,calls,_=laboratory;p=copy.deepcopy(PLAN)
    store.add_proposal({'hypothesis_id':CID})
    folder=repo/'candidates'/CID;folder.mkdir();(folder/'model.py').write_bytes((FOLDER/'model.py').read_bytes())
    with store.connect() as con:
        cfg=json.loads(c.get(cid)['config_json']);cfg['research']['real_development_feedback']=True
        con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    # Historical reference integrity is independently checked against the real
    # read-only ledger during readiness inspection; this fixture has no live DB.
    def capability(s,ref,root):
        assert ref==p['baselines'][0]
        return {'reference':ref,'configuration':{'size':128},'recorded_inference':{},
                'named_inference_config_parameter':False}
    monkeypatch.setattr(planning_contract,'baseline_capability',capability)
    def worker(*args,**kwargs):
        rid=kwargs['expected_run_id'];row=store.get_run(rid);cfg=json.loads(row['config_json']);calls.append(cfg)
        store.transition(rid,'RUNNING');job=runtime/'jobs'/rid;job.mkdir(parents=True)
        # Artificial endpoint values are not model measurements or research evidence.
        metrics={m['key']:1. for m in p['required_measurements'] if m['source']=='run'}
        metrics.update({'protocol/optimizer_updates':120,'protocol/source_exposures':960,'protocol/seed':cfg['seed']})
        diagnostic=p['diagnostics'][0]
        values={name:.5 for name in diagnostic['required_metrics']}
        metrics.update({'diagnostic/within_source_proxy/'+k:v for k,v in values.items()})
        (job/'diagnostic_within_source_proxy.json').write_text(json.dumps({'metrics':values,
            'per_image':[{'source_id':f'artificial-source-{i}','metrics':values} for i in range(2)]}))
        store.set_run_result(rid,metrics,gpu_seconds=0.,artifact_dir=str(job));store.transition(rid,'COMPLETED')
        summary={'evaluation_id':'FIXTURE-'+rid,'metrics':{m['key']:.5 for m in p['required_measurements'] if m['source']=='real'},
                 'label':'ARTIFICIAL DISPATCH FIXTURE ONLY'}
        development_feedback.record_summary(store,rid,summary)
        return {'run_id':rid,'status':'COMPLETED'}
    monkeypatch.setattr(controller,'run_worker_once',worker)
    monkeypatch.setattr(development_feedback,'evaluate_run',lambda runtime,s,rid,**kw:development_feedback.latest_summary(s,rid))
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_EXPLORATORY',result['problems']
    assert not result['scientific_eligibility']['promotion_eligible']
    assert [(cfg['plan_arm'],cfg['seed'],cfg['objective']) for cfg in calls]==[
        (arm,seed,arm) for arm in ('categorical','deterministic') for seed in (7,19,31)]
    for cfg in calls:
        assert all(cfg[key]==value for key,value in p['training'].items())
        assert cfg['diagnostics']==p['diagnostics'] and cfg['inference']=={}
        assert cfg['model_sha256']==hashlib.sha256((FOLDER/'model.py').read_bytes()).hexdigest()
    rows=c.runs(cid);feedback={r['run_id']:development_feedback.latest_summary(store,r['run_id']) for r in rows}
    assert len(rows)==6 and result['recorded_jobs']==6
    assert all(not cell['missing'] for cell in result['cells'])
    assert len(result['paired_image_intervals'])==9
    assert comparison_result(p,rows[:-1],feedback)['status']=='INCOMPLETE'
    broken=copy.deepcopy(rows);broken[-1]['status']='FAILED'
    assert comparison_result(p,broken,feedback)['status']=='INCOMPLETE'
    assert execute_plan(store,c,cid,runtime,p)==result and len(calls)==6
    assert p==PLAN
