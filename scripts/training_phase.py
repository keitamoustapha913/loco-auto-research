"""Versioned timing: v4 elapsed loop, historical learning-only contracts unchanged.

A candidate supplies training_steps(config, train_manifest, output_dir), a generator
which yields TrainingStep descriptors and returns checkpoint metadata. The trusted
loop executes the mathematical update. This is a checked Python protocol within the
existing sandbox, not an additional OS security boundary against malicious Python.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import signal
import sys
import time

VERSION = 'prepared-training-step-v1'
TARGET_VERSION = 'required-training-target-v1'
LOOP_VERSION = 'elapsed-training-loop-v1'


@dataclass(frozen=True)
class TrainingLoopStart:
    """Explicit boundary after initialization and before the first batch fetch."""
    objective: object


class TrainingTimeUnverifiable(ValueError):
    pass


class TrainingTargetIncomplete(TrainingTimeUnverifiable):
    pass


class TrainingBudgetExceeded(TimeoutError):
    pass


@dataclass
class TrainingStep:
    objective: object  # torch.nn.Module: prepared model + differentiable scalar loss
    optimizer: object  # already initialized torch.optim.Optimizer
    args: tuple = ()   # fully prepared, device-resident inputs
    kwargs: dict = field(default_factory=dict)
    use_closure: bool = False  # explicit support for closure-based optimizers
    source_exposures: int | None = None  # required target runs: prepared source presentations
    zero_grad_set_to_none: bool = True


class TrainingClock:
    def __init__(self, path, budget, synchronize=lambda: None, *, target=False, loop=False):
        if budget is not None and (isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or budget<=0):
            raise ValueError('invalid trusted training-only deadline')
        self.path=Path(path);self.budget=budget;self.synchronize=synchronize;self.target=target
        self.exposures=0;self.last_step_seconds=0.
        self.elapsed=0.;self.steps=0;self.active=False;self.phase_started=None
        self.status='PREPARING';self.verified=False
        from collections import deque
        self.created=time.perf_counter();self.samples=deque(maxlen=64)
        self.costs=dict(preparation_seconds=0.,integrity_seconds=0.,validation_seconds=0.,
                        audit_io_seconds=0.,setup_sync_seconds=0.,learning_sync_seconds=0.)
        self.initial_preparation_seconds=0.;self.finalization_seconds=0.
        self.loop_mode=loop;self.loop_started=None;self.loop_finished=None;self.last_boundary=None
        self.last_iteration_seconds=0.
        self.publish()

    def duration(self):
        if not self.loop_mode:return self.elapsed
        if self.loop_started is None:return 0.
        return (self.loop_finished if self.loop_finished is not None else time.perf_counter())-self.loop_started

    def publish(self):
        started=time.perf_counter()
        data={'protocol':VERSION,'status':self.status,'verified':self.verified,
              'duration_seconds':self.duration(),'budget_seconds':self.budget,
              'learning_seconds':self.elapsed,'loop_elapsed_seconds':self.duration() if self.loop_mode else None,
              'timing_definition':LOOP_VERSION if self.loop_mode else 'learning-computation',
              'completion_policy':LOOP_VERSION if self.loop_mode else TARGET_VERSION if self.target else 'maximum',
              'target_seconds':self.budget if self.target else None,
              'source_exposures':self.exposures,'last_step_seconds':self.last_step_seconds,
              'overshoot_seconds':max(0,self.duration()-self.budget) if self.target else 0,
              'last_iteration_seconds':self.last_iteration_seconds,
              'completed_steps':self.steps,'phase_started_monotonic':self.phase_started}
        data['execution_profile']={'version':'training-throughput-v1','costs':dict(self.costs),
            'initial_preparation_seconds':self.initial_preparation_seconds,
            'finalization_seconds':self.finalization_seconds,'observed_wall_seconds':started-self.created,
            'sampled_updates':len(self.samples),'sampled_learning_seconds':sum(s[0] for s in self.samples),
            'sampled_step_wall_seconds':sum(s[1] for s in self.samples),
            'sampled_max_step_wall_seconds':max((s[1] for s in self.samples),default=0.),
            'semantics':'wall preparation/integrity/I-O excluded; post-learning CUDA sync included in learning; first four updates excluded from steady samples'}
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,allow_nan=False));temporary.replace(self.path)
        self.costs['audit_io_seconds']+=time.perf_counter()-started

    def measure(self, operation):
        # Drain setup/data-transfer CUDA work before starting this phase.
        synchronization=time.perf_counter();self.synchronize()
        self.costs['setup_sync_seconds']+=time.perf_counter()-synchronization
        remaining=None if self.budget is None or self.target else self.budget-self.elapsed
        if remaining is not None and remaining<=0:raise TrainingBudgetExceeded('training-time budget exceeded')
        old_handler=None;old_timer=None
        def expired(signum,frame):raise TrainingBudgetExceeded('training-time budget exceeded')
        self.status='TRAINING';self.phase_started=time.monotonic()
        self.publish()  # Audit I/O is outside the measured mathematical update.
        if remaining is not None:
            old_handler=signal.signal(signal.SIGALRM,expired)
            old_timer=signal.setitimer(signal.ITIMER_REAL,remaining)
        started=time.perf_counter();self.active=True
        learning_finished=None;learning_sync=0.
        try:
            result=operation()
            # Wait for queued forward/backward/update kernels inside the deadline.
            synchronization=time.perf_counter();self.synchronize()
            learning_finished=time.perf_counter()
            learning_sync=learning_finished-synchronization
            if self.budget is not None and not self.target and self.elapsed+learning_finished-started>self.budget:
                raise TrainingBudgetExceeded('training-time budget exceeded')
            return result
        finally:
            self.last_step_seconds=(learning_finished if learning_finished is not None else time.perf_counter())-started
            self.elapsed+=self.last_step_seconds
            self.active=False;self.phase_started=None
            self.costs['learning_sync_seconds']+=learning_sync
            if remaining is not None:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,old_handler)
                if old_timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.000001,old_timer[0]-(time.perf_counter()-started)),old_timer[1])
            self.status='PREPARING';self.publish()


def run_training_phases(module, config, manifest, output, *, budget=None, preflight_steps=None):
    output=Path(output);audit=output.parent/'training_timing.json'
    provider=getattr(module,'training_steps',None)
    if not callable(provider):
        audit.write_text(json.dumps({'protocol':VERSION,'status':'UNVERIFIABLE','verified':False,
            'reason':'training_steps interface missing; whole candidate.train timing is not training-only'}))
        raise TrainingTimeUnverifiable('training-only budget cannot be verified: candidate must expose training_steps; refusing whole candidate.train timing')
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    from torch.nn.modules.module import register_module_forward_pre_hook
    device=torch.device(config.get('device','cpu'))
    synchronize=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda:None)
    loop=config.get('training_completion_policy')==LOOP_VERSION
    target=config.get('training_completion_policy') in {TARGET_VERSION,LOOP_VERSION} and preflight_steps is None
    if target and budget is None:raise TrainingTimeUnverifiable('required training target is missing')
    clock=TrainingClock(audit,budget,synchronize,target=target,loop=loop)
    # Guard explicit phase semantics, not self-reported candidate elapsed numbers.
    def io_guard(event,args):
        if clock.active and event in {'open','os.system','subprocess.Popen','os.rename','os.remove','os.mkdir'}:
            raise TrainingTimeUnverifiable('I/O or checkpoint/data preparation inside a training step is forbidden')
    sys.addaudithook(io_guard)  # Inert after this run; audit hooks cannot be removed.
    def optimizer_guard(optimizer,args,kwargs):
        if not clock.active:raise TrainingTimeUnverifiable('optimizer update outside trusted training phase')
    def forward_guard(model,args):
        if not clock.active and torch.is_grad_enabled() and model.training and any(p.requires_grad for p in model.parameters()):
            raise TrainingTimeUnverifiable('gradient-enabled forward outside trusted training phase')
    original_sleep=time.sleep
    def guarded_sleep(seconds):
        if (target and clock.active) or (loop and clock.loop_started is not None and clock.loop_finished is None):
            raise TrainingTimeUnverifiable('sleep cannot fill required training duration')
        return original_sleep(seconds)
    time.sleep=guarded_sleep
    original_init=torch.nn.Module.__init__
    original_optimizer_init=torch.optim.Optimizer.__init__
    def module_init(model,*args,**kwargs):
        if clock.active:raise TrainingTimeUnverifiable('model initialization inside training step is forbidden')
        if loop and clock.loop_started is not None and clock.loop_finished is None:
            raise TrainingTimeUnverifiable('initialize model components before TrainingLoopStart')
        original_init(model,*args,**kwargs)
    def optimizer_init(optimizer,*args,**kwargs):
        if clock.active:raise TrainingTimeUnverifiable('optimizer initialization inside training step is forbidden')
        if loop and clock.loop_started is not None and clock.loop_finished is None:
            raise TrainingTimeUnverifiable('initialize optimizers before TrainingLoopStart')
        original_optimizer_init(optimizer,*args,**kwargs)
    optimizer_hook=register_optimizer_step_pre_hook(optimizer_guard)
    forward_hook=register_module_forward_pre_hook(forward_guard)
    torch.nn.Module.__init__=module_init
    torch.optim.Optimizer.__init__=optimizer_init
    iterator=None
    tracked=[]
    def state_fingerprint(model):
        import hashlib
        fingerprint=hashlib.sha256()
        for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
            fingerprint.update(name.encode())
            fingerprint.update(str((tensor.dtype,tuple(tensor.shape))).encode())
            try:raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            except Exception as exc:raise TrainingTimeUnverifiable('cannot verify exposed model state for this tensor layout') from exc
            fingerprint.update(raw)
        return fingerprint.hexdigest()
    def check_unchanged(expected):
        if any(state_fingerprint(model)!=value for model,value in expected):
            raise TrainingTimeUnverifiable('model parameters/buffers changed during unmeasured preparation or checkpoint saving')
    try:
        iterator=iter(provider(config,manifest,output))
        progress=None
        if target and not hasattr(iterator,'send'):raise TrainingTimeUnverifiable('target training requires a progress-aware generator')
        while True:
            iteration_started=time.perf_counter()
            if target and config.get('research_deadline') is not None and time.time()>=config['research_deadline']:
                raise TrainingBudgetExceeded('overall research deadline reached before training completion')
            # Learning-computation timer is off here. The v4 loop clock counts
            # batch work after TrainingLoopStart; initialization/finalization do not.
            section=time.perf_counter()
            expected=[(model,state_fingerprint(model)) for model in tracked]
            clock.costs['integrity_seconds']+=time.perf_counter()-section
            section=time.perf_counter()
            try:step=next(iterator) if progress is None else iterator.send(progress)
            except StopIteration as stopped:
                clock.finalization_seconds=time.perf_counter()-section
                if loop:clock.loop_finished=clock.last_boundary
                section=time.perf_counter()
                check_unchanged(expected)
                clock.costs['integrity_seconds']+=time.perf_counter()-section
                if not clock.steps:raise TrainingTimeUnverifiable('no explicit training steps were executed')
                if target and clock.duration()<budget:
                    raise TrainingTargetIncomplete(f'training ended prematurely: {clock.duration():.6f} of required {budget:.6f} '+('training-loop elapsed seconds' if loop else 'actual learning seconds'))
                metadata=stopped.value
                clock.status='COMPLETED';clock.verified=True;clock.publish()
                return metadata,clock.duration()
            preparation=time.perf_counter()-section;clock.costs['preparation_seconds']+=preparation
            if isinstance(step,TrainingLoopStart):
                if not loop or clock.loop_started is not None or not isinstance(step.objective,torch.nn.Module):
                    raise TrainingTimeUnverifiable('invalid or repeated training-loop start boundary')
                check_unchanged(expected)
                tracked.append(step.objective)
                synchronize()  # Drain one-time initialization before starting the loop.
                clock.initial_preparation_seconds=preparation
                clock.costs['preparation_seconds']-=preparation
                clock.loop_started=time.perf_counter();clock.publish()
                continue
            if loop and clock.loop_started is None:
                raise TrainingTimeUnverifiable('elapsed training loop requires TrainingLoopStart(objective) before the first batch')
            if not clock.steps and not loop:clock.initial_preparation_seconds=preparation
            section=time.perf_counter()
            check_unchanged(expected)
            clock.costs['integrity_seconds']+=time.perf_counter()-section
            section=time.perf_counter()
            if target and config.get('research_deadline') is not None and time.time()>=config['research_deadline']:
                raise TrainingBudgetExceeded('overall research deadline reached during preparation')
            if (target and ((clock.loop_finished is not None) if loop else clock.elapsed>=budget)) or (preflight_steps is not None and clock.steps>=preflight_steps):
                raise TrainingTimeUnverifiable('target reached: generator must finalize and return, not request another update')
            if target and (isinstance(getattr(step,'source_exposures',None),bool) or not isinstance(getattr(step,'source_exposures',None),int) or step.source_exposures<1):
                raise TrainingTimeUnverifiable('target training must declare positive prepared source_exposures per update')
            if not isinstance(step,TrainingStep) or not isinstance(step.objective,torch.nn.Module) or not isinstance(step.optimizer,torch.optim.Optimizer):
                raise TrainingTimeUnverifiable('training_steps must yield a prepared TrainingStep with Module objective and Optimizer')
            if not isinstance(step.args,tuple) or not isinstance(step.kwargs,dict):
                raise TrainingTimeUnverifiable('training step inputs must be materialized args/kwargs')
            if not all(isinstance(x,bool) for x in (step.use_closure,step.zero_grad_set_to_none)):
                raise TrainingTimeUnverifiable('invalid explicit optimizer controls')
            def check_input(value):
                if isinstance(value,torch.Tensor):
                    if value.device.type!=device.type:raise TrainingTimeUnverifiable('prepare and transfer batch tensors before yielding the step')
                elif isinstance(value,(tuple,list)):
                    for item in value:check_input(item)
                elif isinstance(value,dict):
                    for item in value.values():check_input(item)
                elif value is not None and not isinstance(value,(int,float,bool)):
                    raise TrainingTimeUnverifiable('training inputs must be prepared tensors/scalars, not paths, loaders or callbacks')
            check_input(step.args);check_input(step.kwargs)
            from torch.nn.parameter import UninitializedParameter,UninitializedBuffer
            if any(isinstance(value,(UninitializedParameter,UninitializedBuffer)) for value in list(step.objective.parameters())+list(step.objective.buffers())):
                raise TrainingTimeUnverifiable('materialize lazy model parameters/buffers during initialization before yielding training steps')
            if not any(model is step.objective for model in tracked):tracked.append(step.objective)
            parameters={id(p):p for p in step.objective.parameters()}
            owned=[p for group in step.optimizer.param_groups for p in group['params']]
            if not owned or any(id(p) not in parameters or p.device.type!=device.type for p in owned):
                raise TrainingTimeUnverifiable('optimizer parameters must belong to the prepared objective on the declared device')
            clock.costs['validation_seconds']+=time.perf_counter()-section
            def update():
                def closure():
                    step.optimizer.zero_grad(set_to_none=step.zero_grad_set_to_none)
                    loss=step.objective(*step.args,**step.kwargs)
                    if not isinstance(loss,torch.Tensor) or loss.numel()!=1 or not loss.requires_grad or not torch.isfinite(loss).all():
                        raise TrainingTimeUnverifiable('training objective must return a finite differentiable scalar loss')
                    loss.backward()
                    return loss
                if step.use_closure:step.optimizer.step(closure)
                else:
                    closure();step.optimizer.step()
            clock.measure(update);clock.steps+=1
            boundary=time.perf_counter();clock.last_boundary=boundary
            clock.last_iteration_seconds=boundary-iteration_started
            elapsed=boundary-clock.loop_started if loop else clock.elapsed
            reached=clock.steps>=preflight_steps if preflight_steps is not None else elapsed>=budget if budget is not None else False
            if loop and reached:clock.loop_finished=boundary
            if target or preflight_steps is not None:
                clock.exposures+=step.source_exposures or 0
                progress={'elapsed_seconds':elapsed,'target_seconds':budget,'target_reached':reached,
                          'optimizer_updates':clock.steps,'source_exposures':clock.exposures}
            clock.publish()
            if clock.steps>4:clock.samples.append((clock.last_step_seconds,time.perf_counter()-iteration_started))
    except BaseException as exc:
        clock.status=('FAILED' if isinstance(exc,TrainingTargetIncomplete) else
                      'TIMED_OUT' if isinstance(exc,TrainingBudgetExceeded) else
                      'UNVERIFIABLE' if isinstance(exc,TrainingTimeUnverifiable) else 'FAILED')
        clock.verified=False;clock.publish()
        raise
    finally:
        try:
            # Cleanup stays outside the timer but cannot perform uncharged learning.
            if iterator is not None and hasattr(iterator,'close'):iterator.close()
        finally:
            optimizer_hook.remove();forward_hook.remove();torch.nn.Module.__init__=original_init
            torch.optim.Optimizer.__init__=original_optimizer_init;time.sleep=original_sleep
