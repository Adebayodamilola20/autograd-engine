"""Learning-rate schedules (nabla/optim/schedules.py) and their use in the Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from nabla.data import DataLoader, Dataset
from nabla.losses import softmax_cross_entropy
from nabla.nn import MLP
from nabla.optim import SGD
from nabla.optim.schedules import constant, cosine, exponential, step_decay, warmup
from nabla.training import Trainer


def rates(schedule, n):
    return [schedule(e, 123.0) for e in range(1, n + 1)]   # current lr is ignored


def test_constant():
    assert rates(constant(0.3), 3) == [0.3, 0.3, 0.3]


def test_step_decay_drops_every_k_epochs():
    assert rates(step_decay(1.0, every=2, gamma=0.5), 6) == pytest.approx([1, 1, 0.5, 0.5, 0.25, 0.25])


def test_exponential():
    assert rates(exponential(1.0, 0.9), 3) == pytest.approx([1.0, 0.9, 0.81])


def test_cosine_endpoints_midpoint_and_tail():
    s = cosine(0.1, total_epochs=5, min_lr=0.01)
    r = rates(s, 7)
    assert r[0] == pytest.approx(0.1) and r[4] == pytest.approx(0.01)
    assert r[2] == pytest.approx(0.055)                 # halfway down
    assert r[5] == r[6] == pytest.approx(0.01)          # flat after the end
    assert all(a >= b for a, b in zip(r, r[1:]))        # never goes back up
    assert rates(cosine(0.1, total_epochs=1), 2) == [0.1, 0.1]


def test_warmup_ramps_then_hands_over_with_renumbered_epochs():
    s = warmup(0.1, epochs=4, after=step_decay(0.1, every=1, gamma=0.5))
    assert rates(s, 6) == pytest.approx([0.025, 0.05, 0.075, 0.1, 0.1, 0.05])
    assert rates(warmup(0.2, epochs=2), 4) == pytest.approx([0.1, 0.2, 0.2, 0.2])


def test_schedules_depend_only_on_the_epoch():
    # resuming at epoch k must give what a straight run had at epoch k
    s = cosine(0.1, total_epochs=10)
    assert s(7, 0.5) == s(7, 0.0001)


@pytest.mark.parametrize("make, message", [
    (lambda: step_decay(0.1, every=0), "every"),
    (lambda: step_decay(0.1, every=1, gamma=0.0), "gamma"),
    (lambda: exponential(0.1, gamma=1.5), "gamma"),
    (lambda: cosine(0.1, total_epochs=0), "total_epochs"),
    (lambda: cosine(0.1, total_epochs=5, min_lr=0.2), "min_lr"),
    (lambda: warmup(0.1, epochs=0), "warmup"),
])
def test_bad_arguments(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_epoch_zero_is_rejected():
    with pytest.raises(ValueError, match="count from 1"):
        cosine(0.1, 5)(0, 0.1)


def test_trainer_applies_the_schedule_each_epoch():
    rng = np.random.default_rng(0)
    data = Dataset(rng.normal(size=(20, 2)), np.arange(20) % 2)
    model = MLP(2, [3], 2, seed=0)
    trainer = Trainer(model, SGD(model.parameters(), lr=1.0), softmax_cross_entropy, verbose=False)
    seen = []
    trainer.fit(DataLoader(data, batch_size=10, seed=0), epochs=4,
                lr_schedule=warmup(0.2, epochs=2, after=step_decay(0.2, every=1, gamma=0.5)),
                callbacks=[lambda m: seen.append(m.learning_rate)])
    assert seen == pytest.approx([0.1, 0.2, 0.2, 0.1])
