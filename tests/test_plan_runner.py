"""Small artificial CPU fixtures verify observable execution, not scientific performance."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.data_guard import build_manifest
from benchmark.train_adapter import load_candidate, load_predictor, CandidateContractError
from scripts.demo_run import make_fixture
from scripts.run_candidate import run
from plan_fixtures import experiment_plan

SOURCE = '''
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def train(config, allowed_manifest, output_dir):
    manifest=json.loads(Path(allowed_manifest).read_text())
    assert all(e['split']=='train/good' for e in manifest['images'])
    net=torch.nn.Conv2d(3,1,1)
    opt=torch.optim.SGD(net.parameters(),lr=.01)
    count=0
    for step in range(config['steps']):
        batch=[]
        for j in range(config['batch_size']):
            e=manifest['images'][(count+j)%len(manifest['images'])]
            im=Image.open(Path(manifest['root'])/e['path']).convert('RGB').resize((config['size'],config['size']))
            batch.append(np.asarray(im,dtype=np.float32).transpose(2,0,1)/255)
        x=torch.from_numpy(np.stack(batch))
        opt.zero_grad();net(x).square().mean().backward();opt.step()
        count+=len(batch)
    output_dir.mkdir(parents=True)
    torch.save(net.state_dict(),output_dir/'model.pt')
    meta={'training_audit':{'seed':config['seed'],'optimizer_updates':config['steps'],'source_exposures':count}}
    (output_dir/'checkpoint_manifest.json').write_text(json.dumps(meta))
    return meta

class Predictor:
    def __init__(self,net,inference_config):
        self.net=net.eval();self.inference_config=inference_config
    @torch.inference_mode()
    def predict(self,images):
        value=self.net(torch.as_tensor(images)).sigmoid()
        if self.inference_config.get('mode')=='intervention':value=value*.5
        return {'anomaly_maps':value.numpy(),'image_scores':value.flatten(1).amax(1).numpy()}

def load(checkpoint_manifest,device='cpu',inference_config=None):
    assert device=='cpu'
    net=torch.nn.Conv2d(3,1,1)
    net.load_state_dict(torch.load(Path(checkpoint_manifest).parent/'model.pt',weights_only=True,map_location='cpu'))
    return Predictor(net,inference_config or {})

def diagnose(predictor,assessment_good_images,diagnostic_config):
    assert torch.is_inference_mode_enabled()
    return {'metrics':{'mean_score':float(predictor.predict(assessment_good_images)['anomaly_maps'].mean())},
            'assessment_images':len(assessment_good_images)}
'''


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    make_fixture(tmp_path/'data')
    manifest=tmp_path/'good_manifest.json';manifest.write_text(json.dumps(build_manifest(tmp_path/'data')))
    source=tmp_path/'model.py';source.write_text(SOURCE)
    p=experiment_plan(seeds=(17,),arms=('control',))
    config=dict(p['training'],device='cpu',category='juice_bottle',seed=17,synthetic_evaluation=False,
        plan_id=p['plan_id'],plan_sha256='a'*64,plan_arm='control',execution_kind='train',
        inference={'mode':'control'},diagnostics=p['diagnostics'],trusted_source_sha256={})
    return source,manifest,config


def test_cpu_runner_executes_settings_diagnostics_and_frozen_prediction(tmp_path,monkeypatch,inputs):
    source,manifest,config=inputs
    output=tmp_path/'job';result=run(source,manifest,config,output)
    assert result['metrics']['protocol/optimizer_updates']==2
    assert result['metrics']['protocol/source_exposures']==4
    assert result['metrics']['protocol/seed']==17
    assert 'diagnostic/fixture_check/mean_score' in result['metrics']
    assert result['provenance']['effective_configuration']==config
    assert result['provenance']['training_audit']['observed_optimizer_step_calls']==2
    from scripts import feedback_infer
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer):monkeypatch.setattr(cls,'step',cls.step)
    image=tmp_path/'data/juice_bottle/validation/good/0000.png'
    checkpoint=output/'checkpoint/model.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for mode in ('control','intervention'):
        folder=tmp_path/mode;folder.mkdir()
        feedback_infer.infer_one(source,output/'checkpoint/checkpoint_manifest.json',image,16,folder,{'mode':mode})
    a=json.loads((tmp_path/'control/inference.json').read_text())
    b=json.loads((tmp_path/'intervention/inference.json').read_text())
    assert b['image_score']==pytest.approx(a['image_score']*.5)
    assert before==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert a['state_before_sha256']==a['state_after_sha256'] and a['device']=='cpu'


@pytest.mark.parametrize('defect',['counts','missing_diagnostic','state_mutation'])
def test_runner_rejects_unfulfilled_training_or_diagnostic_contract(tmp_path,inputs,defect):
    source,manifest,config=inputs
    if defect=='counts':config['expected_optimizer_updates']=3
    elif defect=='missing_diagnostic':config['diagnostics'][0]['required_metrics'].append('not_implemented')
    else:source.write_text(SOURCE.replace("assert torch.is_inference_mode_enabled()", "predictor.net.weight.add_(1)"))
    with pytest.raises(ValueError,match={'counts':'expected_optimizer_updates','missing_diagnostic':'omitted','state_mutation':'changed frozen'}[defect]):
        run(source,manifest,config,tmp_path/'job')


def test_frozen_job_never_calls_train(tmp_path,monkeypatch,inputs):
    source,manifest,config=inputs;run(source,manifest,config,tmp_path/'trained')
    module=load_candidate(source)
    module.train=lambda *a:pytest.fail('frozen baseline was retrained')
    monkeypatch.setattr('scripts.run_candidate.load_candidate',lambda *a:module)
    output=tmp_path/'frozen';shutil.copytree(tmp_path/'trained/checkpoint',output/'checkpoint')
    config.update(execution_kind='frozen',inference={'mode':'intervention'})
    result=run(source,manifest,config,output)
    assert result['metrics']['protocol/optimizer_updates']==result['metrics']['protocol/source_exposures']==0
    assert (output/'checkpoint/model.pt').read_bytes()==(tmp_path/'trained/checkpoint/model.pt').read_bytes()


def test_shared_diagnostic_can_assess_unchanged_baseline_source(tmp_path,monkeypatch,inputs):
    source,manifest,config=inputs;run(source,manifest,config,tmp_path/'trained')
    baseline=load_candidate(source);provider=SimpleNamespace(diagnose=baseline.diagnose)
    del baseline.diagnose
    baseline.train=lambda *a:pytest.fail('frozen baseline was trained')
    monkeypatch.setattr('scripts.run_candidate.load_candidate',lambda p:provider if str(p)=='/diagnostics/model.py' else baseline)
    output=tmp_path/'frozen';shutil.copytree(tmp_path/'trained/checkpoint',output/'checkpoint')
    config.update(execution_kind='frozen',diagnostic_provider={'candidate_id':'fixture','source_sha256':'a'*64})
    result=run(source,manifest,config,output)
    assert 'diagnostic/fixture_check/mean_score' in result['metrics']
    assert result['metrics']['protocol/optimizer_updates']==0 and source.read_text()==SOURCE


@pytest.mark.skipif(__import__('os').getenv('LOCO_TEST_BWRAP')!='1',reason='requires Linux user namespaces')
def test_pinned_baseline_and_diagnostic_mounts_are_read_only_and_narrow(tmp_path,monkeypatch):
    import tempfile
    import subprocess
    from orchestrator.sandbox import command
    repo=Path(__file__).resolve().parents[1]
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    pinned=tmp_path/'controller/job_inputs/fixture/source';pinned.mkdir(parents=True)
    model=pinned/'model.py';model.write_text('pinned model')
    data=tmp_path/'data';data.mkdir();output=tmp_path/'output';output.mkdir()
    secret=tmp_path/'fake-codex-credential';secret.write_text('fixture only')
    with tempfile.TemporaryDirectory(prefix='plan-boundary-',dir=repo/'candidates') as folder:
        diagnostic=Path(folder)/'model.py';diagnostic.write_text('pinned diagnostic')
        script=("import json,errno;from pathlib import Path\n"
                "denied=[]\n"
                "for name in ['/candidate/model.py','/diagnostics/model.py']:\n"
                " try:Path(name).write_text('forbidden')\n"
                " except OSError as e:\n"
                "  if e.errno in (errno.EROFS,errno.EACCES,errno.EPERM):denied.append(name)\n"
                "  else:raise\n"
                f"print(json.dumps([len(denied),Path('{secret}').exists(),Path('/workspace').exists()]))")
        proc=subprocess.run(command(repo,model,data,output,['-c',script],pinned_source=True,diagnostic_file=diagnostic),
                            capture_output=True,text=True,timeout=20)
        assert proc.returncode==0,proc.stderr
        assert json.loads(proc.stdout)==[2,False,False]
        assert model.read_text()=='pinned model' and diagnostic.read_text()=='pinned diagnostic'


def test_nonempty_inference_settings_cannot_be_silently_ignored():
    old=SimpleNamespace(load=lambda path,device:object())
    with pytest.raises(CandidateContractError,match='require load'):load_predictor(old,'unused','cpu',{'mode':'other'})
    ignored=SimpleNamespace(load=lambda path,device,inference_config:object())
    with pytest.raises(CandidateContractError,match='acknowledge'):load_predictor(ignored,'unused','cpu',{'mode':'other'})
