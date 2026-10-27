"""Verify caller-owned randomness without changing pretrained tensors."""
import random
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from rmsampler.rng import respect_external_rng


class SamplerFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def sample_fn(self):
        torch.manual_seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        return torch.randn(8) * self.weight


def test_rng_and_frozen_parameters():
    model = SamplerFixture()
    before, architecture = model.weight.clone(), repr(model)
    info = respect_external_rng(model)
    assert len(info['removed_only_literal_rng_resets']) == 4
    torch.manual_seed(123)
    first = model.sample_fn()
    torch.manual_seed(124)
    second = model.sample_fn()
    torch.manual_seed(123)
    np.testing.assert_array_equal(first.numpy(), model.sample_fn().numpy())
    assert not torch.equal(first, second)
    assert torch.equal(before, model.weight) and architecture == repr(model)
