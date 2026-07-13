"""Gradient accumulation: step counts, scheduler cadence, loss scaling,
clipping order, and the final partial accumulation group."""

import math

import pytest
import torch
from torch import nn

from mtsd_attr.train_common import _train_one_epoch


class SpySGD(torch.optim.SGD):
    """Records the parameter gradient at every optimiser step."""

    def __init__(self, params, param_ref, lr=0.0):
        super().__init__(params, lr=lr)
        self.param_ref = param_ref
        self.step_grads = []

    def step(self, closure=None):
        self.step_grads.append(float(self.param_ref.grad))
        return super().step(closure)


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _loader(values):
    # (images, targets, paths) triples; images shape (1, 1).
    return [(torch.tensor([[float(v)]]), torch.tensor([[0]]), None)
            for v in values]


def _run(values, accum, grad_clip=0.0):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    # loss = w * v, so d(loss)/dw = v: the recorded step gradient equals the
    # scaled sum of the group's batch values.
    loss_fn = lambda logits, targets: (logits.sum(), {})  # noqa: E731
    optimizer = SpySGD(model.parameters(), model.weight, lr=0.0)
    scheduler = CountingScheduler()
    device = torch.device("cpu")
    mean_loss, _, optim_steps = _train_one_epoch(
        model, _loader(values), loss_fn, optimizer, scheduler, device,
        amp=False, grad_clip=grad_clip, accum_steps=accum)
    return mean_loss, optim_steps, optimizer.step_grads, scheduler.steps


def test_step_counts_match_ceil_of_batches_over_accum():
    for n, accum in ((7, 3), (6, 3), (5, 1), (1, 4), (8, 2)):
        _, optim_steps, step_grads, sched_steps = _run(list(range(1, n + 1)),
                                                       accum)
        expected = math.ceil(n / accum)
        assert optim_steps == expected
        assert len(step_grads) == expected
        assert sched_steps == expected


def test_loss_scaling_including_final_partial_group():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    _, _, step_grads, _ = _run(values, accum=3)
    # Full groups average over 3 batches; the final partial group over 1.
    assert step_grads[0] == pytest.approx((1 + 2 + 3) / 3, rel=1e-6)
    assert step_grads[1] == pytest.approx((4 + 5 + 6) / 3, rel=1e-6)
    assert step_grads[2] == pytest.approx(7 / 1, rel=1e-6)


def test_no_accumulation_is_per_batch():
    values = [2.0, 4.0]
    mean_loss, optim_steps, step_grads, _ = _run(values, accum=1)
    assert optim_steps == 2
    assert step_grads == pytest.approx([2.0, 4.0])
    # Reported loss is the unscaled per-batch mean (w=1, lr=0).
    assert mean_loss == pytest.approx((2.0 + 4.0) / 2)


def test_clipping_applies_immediately_before_each_step(monkeypatch):
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm):
        calls.append(max_norm)
        return original(params, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    _, optim_steps, step_grads, _ = _run([10.0, 10.0, 10.0], accum=2,
                                         grad_clip=0.5)
    assert len(calls) == optim_steps == 2
    # Gradients recorded at step time are already clipped to max_norm.
    assert all(abs(g) <= 0.5 + 1e-6 for g in step_grads)
