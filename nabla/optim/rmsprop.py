r"""RMSprop -- a per-weight step size from a running average of squared gradients.

The idea Adam is built on, without the momentum term. Keep an exponentially
weighted average of each parameter's squared gradient,

.. math::

    s \leftarrow \rho\, s + (1-\rho)\, g^{2},
    \qquad
    \theta \leftarrow \theta - \eta\, \frac{g}{\sqrt{s} + \epsilon},

so a weight whose gradient is consistently large takes small steps and one
whose gradient is tiny takes large ones. Plain SGD uses one step size for
every weight, which is why it struggles when some directions of the loss are
far steeper than others.

There is no bias correction (unlike Adam): ``s`` starts at zero, so the first
few steps are *larger* than ``lr``, which is part of why RMSprop is usually run
with a smaller ``lr`` than SGD. ``momentum`` adds a velocity buffer on top of
the scaled step, as in the original Hinton lecture variant.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .optimizer import Optimizer

__all__ = ["RMSprop"]


def _restore(value: Any) -> Any:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    return float(value)


class RMSprop(Optimizer):
    """RMSprop (Tieleman & Hinton, 2012).

    Parameters
    ----------
    params
        Usually ``model.parameters()``; scalar or tensor parameters.
    lr
        Learning rate. ``1e-3`` is a common default.
    alpha
        Decay rate ``rho`` of the squared-gradient average, in ``[0, 1)``.
    eps
        Added to the denominator for numerical stability.
    momentum
        If non-zero, accumulate the scaled step in a velocity buffer.
    weight_decay
        L2 penalty folded into the gradient.

    Examples
    --------
    >>> from nabla.nn import MLP
    >>> model = MLP(2, [4], 1, seed=0)
    >>> opt = RMSprop(model.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        *,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if momentum < 0.0:
            raise ValueError(f"momentum must be non-negative, got {momentum}")

        self.alpha = float(alpha)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        n = len(self.params)
        self.square_avg: list[Any] = [0.0] * n   # running mean of g², float or array
        self.velocity: list[Any] = [0.0] * n

    def step(self) -> None:
        self.step_count += 1
        rho = self.alpha
        for i, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data

            # s ← ρs + (1-ρ)g²
            self.square_avg[i] = rho * self.square_avg[i] + (1.0 - rho) * grad * grad
            # ``** 0.5`` works on floats and arrays alike, so one optimiser
            # serves both engines (same choice as Adam).
            scaled = grad / (self.square_avg[i] ** 0.5 + self.eps)

            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + scaled
                scaled = self.velocity[i]

            param.data -= self.lr * scaled

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "alpha": self.alpha,
                "eps": self.eps,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "square_avg": [x.tolist() if hasattr(x, "tolist") else x for x in self.square_avg],
                "velocity": [x.tolist() if hasattr(x, "tolist") else x for x in self.velocity],
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.alpha = float(state.get("alpha", self.alpha))
        self.eps = float(state.get("eps", self.eps))
        self.momentum = float(state.get("momentum", self.momentum))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        for key in ("square_avg", "velocity"):
            values = state.get(key)
            if values is None:
                continue
            if len(values) != len(self.params):
                raise ValueError(
                    f"RMSprop '{key}' buffer has {len(values)} entries for "
                    f"{len(self.params)} parameters"
                )
            setattr(self, key, [_restore(v) for v in values])

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr:g}, alpha={self.alpha:g}, momentum={self.momentum:g}, "
            f"params={self._scalar_count():,})"
        )
