r"""Phase 10 -- stochastic gradient descent, with momentum and weight decay.

Plain SGD
---------
.. math:: \theta_{t+1} = \theta_t - \eta\,g_t, \qquad g_t = \nabla_\theta \mathcal{L}

One line. Everything else in this file is a fix for one of its weaknesses.

Why "stochastic"
----------------
True gradient descent computes :math:`\nabla\mathcal{L}` over the *entire*
dataset before taking a single step. On 60,000 MNIST images that is one update
per full pass -- accurate, and hopelessly slow. **Stochastic** gradient descent
estimates the gradient from a small batch instead. The estimate is noisy, but
it is unbiased, and you get hundreds of updates per epoch rather than one.

The noise turns out to be useful, not merely tolerable: it lets the optimiser
escape sharp local minima and saddle points that a deterministic method would
settle into. Small batches generalise better partly for this reason.

The weakness: ill-conditioned valleys
-------------------------------------
Consider a loss shaped like a long, narrow ravine -- curvature 100 across it,
curvature 1 along it. Stability requires :math:`\eta < 2/100`, but progress
along the valley then advances at :math:`\eta \cdot 1 = 0.02` per step. The
result is the classic zig-zag: violent oscillation across the ravine,
glacial progress towards the minimum.

Momentum
--------
.. math::
    v_{t+1} &= \mu v_t + g_t\\
    \theta_{t+1} &= \theta_t - \eta\, v_{t+1}

Instead of stepping along the current gradient, step along a running average of
recent gradients. The physical picture is a ball rolling downhill with
inertia rather than a hiker re-deciding at every footstep.

Why it fixes the ravine: the oscillating across-valley components have
alternating signs and **cancel** in the average, while the consistent
along-valley component **accumulates**. In the steady state, a constant
gradient produces :math:`v = g/(1-\mu)`, so momentum :math:`\mu = 0.9`
multiplies the effective step by 10 in directions where the gradient is
consistent -- and by nothing at all where it flips sign every step.

That :math:`1/(1-\mu)` factor is also the practical warning: switching momentum
on without lowering :math:`\eta` can amplify your effective learning rate
tenfold and diverge.

Nesterov momentum
-----------------
Nesterov's variant evaluates the gradient *after* the momentum step -- "look
where you are about to land, not where you are". It anticipates overshoot and
brakes earlier, giving slightly better convergence. We use the standard
reformulation that avoids needing a gradient at a second point.

Weight decay
------------
Adds :math:`\lambda\theta` to the gradient, which is the gradient of an
:math:`\tfrac{\lambda}{2}\|\theta\|^{2}` penalty -- L2 regularisation.
It pulls weights toward zero unless the data actively pushes back, which limits
how confidently the model can fit noise.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..nn.parameter import Parameter
from .optimizer import Optimizer

__all__ = ["SGD"]


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

    Parameters
    ----------
    params
        Usually ``model.parameters()``.
    lr
        Learning rate :math:`\\eta`.
    momentum
        :math:`\\mu` in :math:`[0, 1)`. ``0.0`` gives plain SGD; ``0.9`` is the
        standard choice and multiplies the effective step by ~10 in consistent
        directions, so reduce ``lr`` accordingly when enabling it.
    nesterov
        Use Nesterov's look-ahead variant. Requires ``momentum > 0``.
    weight_decay
        L2 penalty coefficient :math:`\\lambda`.

    Examples
    --------
    >>> from nabla.nn import MLP
    >>> model = MLP(2, [4], 1, seed=0)
    >>> opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> opt.zero_grad()
    >>> loss = model([1.0, 2.0])[0] ** 2
    >>> loss.backward()
    >>> opt.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        *,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov requires momentum > 0")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        self.momentum = float(momentum)
        self.nesterov = nesterov
        self.weight_decay = float(weight_decay)

        # One velocity buffer per parameter, indexed positionally so it stays
        # aligned with self.params and serialises cleanly.
        self.velocity: list[float] = [0.0] * len(self.params)

    def step(self) -> None:
        """Apply one update to every parameter.

        Reads ``param.grad`` -- which ``backward()`` filled in -- and writes
        ``param.data``. Note it never touches the graph: by the time we get
        here the graph has done its job and will be discarded.
        """
        self.step_count += 1
        lr = self.lr
        mu = self.momentum
        wd = self.weight_decay

        for i, param in enumerate(self.params):
            grad = param.grad

            # L2 regularisation, folded into the gradient:
            #   d/dθ [L + λ/2·θ²] = dL/dθ + λθ
            if wd:
                grad = grad + wd * param.data

            if mu:
                # v ← μv + g   (exponentially-weighted sum of past gradients)
                v = mu * self.velocity[i] + grad
                self.velocity[i] = v
                # Nesterov: step along the gradient *plus* the upcoming
                # momentum, i.e. look ahead to where momentum is taking us.
                update = grad + mu * v if self.nesterov else v
            else:
                update = grad

            # The line the whole field rests on: move against the gradient.
            param.data -= lr * update

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                "weight_decay": self.weight_decay,
                "velocity": [
                    v.tolist() if hasattr(v, "tolist") else v for v in self.velocity
                ],
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.momentum = float(state.get("momentum", self.momentum))
        self.nesterov = bool(state.get("nesterov", self.nesterov))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        velocity = state.get("velocity")
        if velocity is not None:
            if len(velocity) != len(self.params):
                raise ValueError(
                    f"velocity buffer has {len(velocity)} entries for "
                    f"{len(self.params)} parameters"
                )
            self.velocity = [
                np.asarray(v, dtype=np.float64) if isinstance(v, list) else float(v)
                for v in velocity
            ]

    def __repr__(self) -> str:
        bits = [f"lr={self.lr:g}"]
        if self.momentum:
            bits.append(f"momentum={self.momentum:g}")
        if self.nesterov:
            bits.append("nesterov=True")
        if self.weight_decay:
            bits.append(f"weight_decay={self.weight_decay:g}")
        return f"SGD({', '.join(bits)}, params={self._scalar_count():,})"
