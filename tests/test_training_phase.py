"""Strict phase timing with CPU fixtures and simulated async CUDA; no GPU jobs."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts.run_candidate import train_measured
from scripts.training_phase import (TrainingStep,TrainingClock,TrainingBudgetExceeded,
                                    TrainingTimeUnverifiable,run_training_phases,VERSION)
from training_phase_fixtures import phase_module


def test_setup_batch_preparation_and_checkpoint_saving_are_excluded(tmp_path):
    started=time.perf_counter()
    metadata,elapsed=train_measured(phase_module(steps=2,preparation_delay=.04,training_delay=.005,finalize_delay=.08),
        {'device':'cpu','training_budget_seconds':.06},tmp_path,tmp_path/'checkpoint')
    wall=time.perf_counter()-started
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert wall>.18 and .008<elapsed<.06
    assert audit['verified'] and audit['status']=='COMPLETED' and audit['completed_steps']==2
    assert audit['duration_seconds']==elapsed and metadata=={'fixture_only':True}


def test_cumulative_deadline_is_not_reset_after_data_preparation(tmp_path):
    updates=[]
    with pytest.raises(TrainingBudgetExceeded):
        train_measured(phase_module(steps=4,preparation_delay=.01,training_delay=.025,on_update=lambda *a:updates.append(a)),
                       {'device':'cpu','training_budget_seconds':.06},tmp_path,tmp_path/'checkpoint')
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['status']=='TIMED_OUT' and audit['verified'] is False
    assert len(updates)<4 and .05<audit['duration_seconds']<.15


def test_old_candidate_is_unverifiable_and_never_called(tmp_path):
    called=[]
    with pytest.raises(TrainingTimeUnverifiable,match='training-only budget cannot be verified'):
        train_measured(SimpleNamespace(train=lambda *a:called.append(1)),
                       {'device':'cpu','training_budget_seconds':10},tmp_path,tmp_path/'checkpoint')
    assert called==[]
    assert json.loads((tmp_path/'training_timing.json').read_text())['verified'] is False


def test_legacy_call_is_explicitly_unverified_not_training_only(tmp_path):
    train_measured(SimpleNamespace(train=lambda *a:None),{'device':'cpu'},tmp_path,tmp_path/'checkpoint')
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert audit['status']=='UNVERIFIABLE' and audit['verified'] is False
    assert 'duration_seconds' not in audit and 'function_wall_seconds' in audit


def test_async_cuda_setup_is_drained_outside_and_learning_inside_timer(tmp_path):
    pending=[.08];events=[]
    def synchronize():
        events.append('sync');time.sleep(pending[0]);pending[0]=0
    clock=TrainingClock(tmp_path/'training_timing.json',.04,synchronize)
    def forward_backward_update():
        events.append('update');pending[0]=.01
    start=time.perf_counter();clock.measure(forward_backward_update)
    assert time.perf_counter()-start>.08
    assert .008<clock.elapsed<.04 and events==['sync','update','sync']


def test_pending_cuda_learning_work_is_subject_to_deadline(tmp_path):
    calls=[]
    def synchronize():
        calls.append(1)
        if len(calls)>1:time.sleep(.2)
    clock=TrainingClock(tmp_path/'training_timing.json',.02,synchronize)
    with pytest.raises(TrainingBudgetExceeded):clock.measure(lambda:None)
    assert .015<clock.elapsed<.1 and clock.active is False


@pytest.mark.parametrize('kind',['file_io','module_init','optimizer_init'])
def test_setup_operations_inside_step_are_rejected(tmp_path,kind):
    class Bad(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,x):
            if kind=='file_io':(tmp_path/'forbidden').write_text('checkpoint during forward')
            elif kind=='module_init':torch.nn.Linear(1,1)
            else:torch.optim.SGD([self.weight],lr=.1)
            return self.weight*x
    def provider(*a):
        net=Bad();yield TrainingStep(net,torch.optim.SGD(net.parameters(),lr=.1),(torch.tensor(1.),))
    with pytest.raises(TrainingTimeUnverifiable):
        run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=1)
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('kind',['optimizer','forward'])
def test_learning_outside_phases_is_not_free(tmp_path,kind):
    def provider(*a):
        net=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(net.parameters(),lr=.1)
        if kind=='optimizer':optimizer.step()
        else:net(torch.ones(1,1))
        yield TrainingStep(net,optimizer,(torch.ones(1,1),))
    with pytest.raises(TrainingTimeUnverifiable,match='outside trusted training phase'):
        run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=1)


def test_generator_without_steps_cannot_claim_verified_training(tmp_path):
    def provider(*a):
        if False:yield
        return {}
    with pytest.raises(TrainingTimeUnverifiable,match='no explicit training steps'):
        run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=1)


def test_forward_backward_and_optimizer_update_all_occur(tmp_path):
    updates=[]
    run_training_phases(phase_module(steps=3,on_update=lambda _,weight:updates.append(weight)),
                        {'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=2)
    assert len(updates)==3 and updates[2]<updates[1]<updates[0]<1


def test_closure_optimizer_is_explicitly_supported(tmp_path):
    class Loss(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def forward(self):return self.weight.square()
    def provider(*a):
        net=Loss();optimizer=torch.optim.LBFGS(net.parameters(),max_iter=2)
        yield TrainingStep(net,optimizer,use_closure=True)
        return {'weight':float(net.weight.detach())}
    result,_=run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=2)
    assert result['weight']==0


@pytest.mark.parametrize('when',['preparation','saving'])
def test_manual_weight_updates_cannot_hide_between_steps(tmp_path,when):
    class Loss(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def forward(self):return self.weight.square()
    def provider(*a):
        model=Loss();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        yield TrainingStep(model,optimizer)
        model.weight.data.add_(1)  # Bypasses optimizer hooks and version counters.
        if when=='preparation':yield TrainingStep(model,optimizer)
        return {}
    with pytest.raises(TrainingTimeUnverifiable,match='parameters/buffers changed'):
        run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=2)


def test_lazy_model_initialization_is_not_charged_as_training(tmp_path):
    def provider(*a):
        net=torch.nn.LazyLinear(1);optimizer=torch.optim.SGD(net.parameters(),lr=.1)
        yield TrainingStep(net,optimizer,(torch.ones(1,1),))
    with pytest.raises(TrainingTimeUnverifiable,match='materialize lazy model'):
        run_training_phases(SimpleNamespace(training_steps=provider),{'device':'cpu'},tmp_path,tmp_path/'checkpoint',budget=2)


def test_timer_and_hooks_are_restored_after_exhaustion(tmp_path):
    import signal
    original=signal.getsignal(signal.SIGALRM);original_init=torch.nn.Module.__init__
    with pytest.raises(TrainingBudgetExceeded):
        train_measured(phase_module(training_delay=.1),{'device':'cpu','training_budget_seconds':.01},tmp_path,tmp_path/'checkpoint')
    assert signal.getsignal(signal.SIGALRM)==original and signal.getitimer(signal.ITIMER_REAL)[0]==0
    assert torch.nn.Module.__init__ is original_init


def test_completed_overrun_cannot_claim_compliance_if_alarm_delivery_is_delayed(tmp_path):
    import signal
    clock=TrainingClock(tmp_path/'timing.json',.01)
    def delayed():
        signal.setitimer(signal.ITIMER_REAL,0)  # Simulates delayed/unavailable alarm delivery.
        time.sleep(.025)
    with pytest.raises(TrainingBudgetExceeded):clock.measure(delayed)
    assert clock.elapsed>=.02


@pytest.mark.parametrize('target',[False,True])
def test_phase_contract_through_full_cpu_runner_in_bubblewrap(tmp_path,target):
    import os,shutil,subprocess
    if os.getenv('LOCO_TEST_BWRAP')!='1':pytest.skip('full regression enables explicit Bubblewrap checks')
    from orchestrator.sandbox import BENCHMARK_FILES,command
    from scripts.demo_run import make_fixture
    repo=Path(__file__).parents[1];copy=tmp_path/'repo'
    folder=copy/'candidates/phase_fixture';folder.mkdir(parents=True)
    for package,names in [('benchmark',BENCHMARK_FILES),('scripts',('__init__.py','run_candidate.py','training_phase.py','candidate_smoke.py','stage7_eval.py','stage7_texture_eval.py','stage8_eval.py','feedback_infer.py'))]:
        (copy/package).mkdir()
        for name in names:shutil.copyfile(repo/package/name,copy/package/name)
    shutil.copyfile(repo/'pyproject.toml',copy/'pyproject.toml')
    module=folder/'model.py'
    module.write_text('''import json,time
from pathlib import Path
import numpy as np
import torch
from scripts.training_phase import TrainingStep
class Objective(torch.nn.Module):
    def __init__(self):
        super().__init__();time.sleep(.15)
        self.weight=torch.nn.Parameter(torch.tensor(.5))
    def forward(self,x):return (self.weight*x).square().mean()
def train(config,manifest,output):raise RuntimeError('legacy train must not be used')
def training_steps(config,allowed_manifest,output):
    assert all(x['split']=='train/good' for x in json.loads(Path(allowed_manifest).read_text())['images'])
    net=Objective();optimizer=torch.optim.SGD(net.parameters(),lr=.01)
    count=0
    while True:
        batch=torch.ones(1,3,16,16)
        progress=yield TrainingStep(net,optimizer,(batch,),source_exposures=1)
        count+=1
        if config.get('training_completion_policy')=='required-training-target-v1':
            if progress['target_reached']:break
        elif count==2:break
    time.sleep(.15)
    output=Path(output);output.mkdir(parents=True)
    torch.save(net.state_dict(),output/'model.pt')
    meta={'seed':config['seed'],'training_audit':{'seed':config['seed'],'optimizer_updates':count,'source_exposures':count}}
    (output/'checkpoint_manifest.json').write_text(json.dumps(meta))
    return meta
class Predictor:
    def __init__(self,net):self.net=net
    def predict(self,images):
        maps=(torch.as_tensor(images).mean(1,keepdim=True)-self.net.weight).abs().detach().numpy()
        return {'anomaly_maps':maps,'image_scores':maps.reshape(len(images),-1).max(1)}
def load(checkpoint_manifest,device='cpu'):
    net=Objective().to(device);net.load_state_dict(torch.load(Path(checkpoint_manifest).parent/'model.pt',weights_only=True))
    return Predictor(net)
''')
    data=tmp_path/'artificial_good';make_fixture(data);output=tmp_path/'output';output.mkdir()
    script='''import json
from pathlib import Path
import torch
from benchmark.data_guard import build_manifest
from scripts.run_candidate import run
assert not torch.cuda.is_available()
manifest=Path('/output/manifest.json');manifest.write_text(json.dumps(build_manifest('/data')))
config={'device':'cpu','category':'juice_bottle','size':16,'seed':7,'steps':2,
'expected_optimizer_updates':2,'expected_source_exposures':2,'execution_kind':'train',
'plan_sha256':'f'*64,'plan_id':'fixture','plan_arm':'fixture','trusted_source_sha256':{},
'synthetic_evaluation':False,'inference':{},'diagnostics':[],
'budget_fixture':True,'training_budget_seconds':.1}
result=run(Path('/candidate/model.py'),manifest,config,Path('/output/run'))
assert result['metrics']['training/timing_verified']==1
assert result['metrics']['training/phase_steps']==2
assert result['metrics']['protocol/optimizer_updates']==2
assert result['metrics']['protocol/source_exposures']==2
assert 0<result['metrics']['training/duration_seconds']<.1
assert result['provenance']['training_timing_protocol']=='prepared-training-step-v1'
print('VERIFIED_PHASE_CPU_FIXTURE')
'''
    if target:
        script=script.replace("result=run(","config.update(plan_contract_version='experiment-plan-v3.0',training_completion_policy='required-training-target-v1',training_budget_seconds=.03)\nfrom benchmark.data_guard import sha256_file\nconfig['model_sha256']=sha256_file(Path('/candidate/model.py'))\nresult=run(")
        script=script.replace("assert result['metrics']['training/phase_steps']==2","assert result['metrics']['training/phase_steps']>=1")
        script=script.replace("assert result['metrics']['protocol/optimizer_updates']==2","assert result['metrics']['protocol/optimizer_updates']==result['metrics']['training/phase_steps']")
        script=script.replace("assert result['metrics']['protocol/source_exposures']==2","assert result['metrics']['protocol/source_exposures']==result['metrics']['training/phase_steps']")
        script=script.replace("assert 0<result['metrics']['training/duration_seconds']<.1","assert result['metrics']['training/duration_seconds']>=.03\nassert result['metrics']['training/target_reached']==1\nassert Path('/output/run/training_receipt.json').is_file()")
    process=subprocess.run(command(copy,module,data,output,['-c',script],use_gpu=False),capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    assert 'VERIFIED_PHASE_CPU_FIXTURE' in process.stdout
