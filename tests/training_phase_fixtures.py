"""Artificial CPU tensor updates; never a research candidate or dataset experiment."""
import time
from types import SimpleNamespace
import torch
from scripts.training_phase import TrainingStep


def phase_module(*, steps=1, preparation_delay=0., training_delay=0., finalize_delay=0., on_update=None):
    class Objective(torch.nn.Module):
        def __init__(self):
            super().__init__();time.sleep(preparation_delay)
            self.weight=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,batch):
            time.sleep(training_delay)
            return (self.weight*batch).square().mean()
    def provider(config,manifest,output):
        model=Objective();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        for index in range(steps):
            time.sleep(preparation_delay)
            batch=torch.ones(2)
            yield TrainingStep(model,optimizer,(batch,))
            if on_update:on_update(index,float(model.weight.detach()))
        time.sleep(finalize_delay)
        return {'fixture_only':True}
    return SimpleNamespace(training_steps=provider)
