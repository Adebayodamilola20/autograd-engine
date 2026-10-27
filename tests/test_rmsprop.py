"""RMSprop (nabla/optim/rmsprop.py): the update rule, convergence, both engines, checkpoints."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nabla.losses.tensor_losses import tensor_cross_entropy
from nabla.nn import MLP, Parameter
from nabla.nn.tensor_mlp import TensorMLP
from nabla.optim import SGD, RMSprop


def bowl(params, centre=(3.0, -2.0), curvature=(1.0, 1.0)):
    total = None
    for p, c, k in zip(params, centre, curvature):
        term = k * (p - c) ** 2
        total = term if total is None else total + term
    return total


def run(opt, params, steps, **kwargs):
    for _ in range(steps):
        opt.zero_grad()
        bowl(params, **kwargs).backward()
        opt.step()
    return [p.data for p in params]


def test_first_step_by_hand():
    # s1 = (1-0.99) * g^2 = 0.04 ; step = lr * g / sqrt(s1) = 0.01 * 2 / 0.2 = 0.1
    p = Parameter(0.0)
    p.grad = 2.0
    opt = RMSprop([p], lr=0.01, eps=1e-12)
    opt.step()
    assert opt.square_avg[0] == pytest.approx(0.04)
    assert p.data == pytest.approx(-0.1)


def test_step_size_does_not_depend_on_gradient_scale():
    # dividing by sqrt(s) cancels the scale of g on the first step
    moves = []
    for g in (0.01, 100.0):
        p = Parameter(0.0)
        p.grad = g
        RMSprop([p], lr=0.01, eps=1e-12).step()
        moves.append(p.data)
    assert moves[0] == pytest.approx(moves[1])


def test_converges_to_a_known_minimum():
    params = [Parameter(0.0), Parameter(0.0)]
    result = run(RMSprop(params, lr=0.01), params, steps=2000)
    assert result == pytest.approx([3.0, -2.0], abs=2e-2)


def test_handles_a_ravine_better_than_sgd_on_a_short_budget():
    # One direction 100x steeper than the other. SGD's lr is capped by the steep
    # direction (stable only below 0.01), so it crawls along the flat one;
    # RMSprop rescales each direction and covers both in the same 100 steps.
    curv = (100.0, 1.0)
    rms = [Parameter(0.0), Parameter(0.0)]
    run(RMSprop(rms, lr=0.05), rms, steps=100, curvature=curv)
    sgd = [Parameter(0.0), Parameter(0.0)]
    run(SGD(sgd, lr=0.009), sgd, steps=100, curvature=curv)
    err = lambda ps: math.hypot(ps[0].data - 3.0, ps[1].data + 2.0)  # noqa: E731
    assert err(rms) < err(sgd)


def test_momentum_and_weight_decay():
    params = [Parameter(0.0), Parameter(0.0)]
    result = run(RMSprop(params, lr=0.005, momentum=0.5), params, steps=1500)
    assert result == pytest.approx([3.0, -2.0], abs=5e-2)
    decayed = [Parameter(0.0), Parameter(0.0)]
    plain = [Parameter(0.0), Parameter(0.0)]
    run(RMSprop(decayed, lr=0.01, weight_decay=1.0), decayed, steps=2000)
    run(RMSprop(plain, lr=0.01), plain, steps=2000)
    assert abs(decayed[0].data) < abs(plain[0].data)      # pulled towards zero


@pytest.mark.parametrize("kwargs, message", [
    ({"alpha": 1.0}, "alpha"), ({"eps": 0.0}, "eps"), ({"momentum": -0.1}, "momentum"),
])
def test_rejects_bad_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RMSprop([Parameter(0.0)], **kwargs)


def test_trains_a_tensor_network():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    model = TensorMLP(2, [8], 2, activation="relu", seed=0)
    opt = RMSprop(model.parameters(), lr=0.01, momentum=0.9)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = tensor_cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.2
    assert (np.argmax(model(x).data, axis=1) == y).mean() == 1.0


def test_state_dict_round_trip_resumes_identically():
    model = MLP(2, [3], 1, seed=0)
    opt = RMSprop(model.parameters(), lr=0.01, momentum=0.9)
    for p in model.parameters():
        p.grad = 0.5
    opt.step()
    saved = opt.state_dict()

    clone = MLP(2, [3], 1, seed=0)
    for a, b in zip(clone.parameters(), model.parameters()):
        a.data = b.data
    opt2 = RMSprop(clone.parameters(), lr=0.5)
    opt2.load_state_dict(saved)
    assert opt2.lr == 0.01 and opt2.step_count == 1
    for o in (opt, opt2):
        for p in o.params:
            p.grad = 0.25
        o.step()
    assert [p.data for p in clone.parameters()] == pytest.approx([p.data for p in model.parameters()])


def test_state_dict_rejects_a_mismatched_buffer():
    opt = RMSprop([Parameter(0.0), Parameter(1.0)])
    with pytest.raises(ValueError, match="buffer"):
        opt.load_state_dict({"square_avg": [0.0]})
