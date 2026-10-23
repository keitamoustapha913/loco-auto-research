"""CPU-only state probes: no scientific checkpoints or datasets."""
from types import SimpleNamespace

import pytest
import torch

from scripts.feedback_infer import state_digest


@pytest.mark.parametrize('name', ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked', 'offset'])
def test_all_parameters_and_buffers_are_frozen(name):
    net = torch.nn.BatchNorm2d(3).eval()
    net.register_buffer('offset', torch.zeros(3), persistent=False)
    predictor = SimpleNamespace(net=net)
    before = state_digest(predictor)
    with torch.no_grad():
        getattr(net, name).add_(1)
    assert state_digest(predictor) != before


def test_mode_is_inference_state_and_normal_inference_is_stable():
    net = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Dropout()).eval()
    predictor = SimpleNamespace(net=net)
    before = state_digest(predictor)
    with torch.inference_mode():
        net(torch.ones(2, 3, 4, 4))
    assert state_digest(predictor) == before
    net[1].train()
    assert state_digest(predictor) != before


def test_many_temporary_state_wrappers_do_not_hide_mutations():
    net = torch.nn.ModuleList([torch.nn.BatchNorm2d(3).eval() for _ in range(32)])
    predictor = SimpleNamespace(nets=net)
    for block in net:
        before = state_digest(predictor)
        block.running_mean.add_(1)
        assert state_digest(predictor) != before
