r"""Phase 10 -- Adam: adaptive moment estimation.

The problem Adam solves
-----------------------
SGD uses **one** learning rate for every parameter. But parameters are not
alike. A weight attached to an MNIST corner pixel -- zero in almost every image
-- receives a gradient near zero almost always, and needs large steps on the
rare occasions it gets one. A weight in the middle of the image gets a strong
gradient every batch and needs small steps. One :math:`\eta` cannot serve both.

Adam gives every parameter its own effective step size, derived from that
parameter's own gradient history.

The two moments
---------------
Adam keeps two exponentially-weighted moving averages per parameter:

.. math::
    m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t
        &&\text{the mean -- which way has it been going?}\\
    v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_t^{2}
        &&\text{the uncentred variance -- how big has it been?}

:math:`m` is momentum by another name. :math:`v` tracks the typical *magnitude*
of the gradient, ignoring sign.

The update
----------
.. math:: \theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}

Dividing by :math:`\sqrt{v}` is the adaptive part. A parameter with
consistently large gradients gets a large denominator and therefore small
steps; a parameter with tiny gradients gets small denominator and larger steps.
The ratio :math:`m/\sqrt{v}` is roughly a *signal-to-noise ratio*, and it is
dimensionless -- scale the loss by 1000 and both :math:`m` and
:math:`\sqrt{v}` scale by 1000, leaving the step unchanged. That scale
invariance is why Adam's default :math:`\eta = 0.001` works across wildly
different problems, and it is the main practical reason people reach for it.

Bias correction -- the subtle part
----------------------------------
Both averages start at zero, which biases them toward zero early on. Concretely,
after the first step with a constant gradient :math:`g`:

.. math:: m_1 = (1-\beta_1)\,g = 0.1\,g

The estimate is 10x too small, and :math:`v_1 = 0.001\,g^{2}` is 1000x too
small. Uncorrected, the first update would be
:math:`m_1/\sqrt{v_1} \approx 0.1g / 0.0316|g| \approx 3.2` -- wrong by
threefold, in a direction that depends on nothing meaningful.

Expanding the recursion for a constant gradient gives
:math:`m_t = (1-\beta_1^{t})\,g`, so dividing it out fixes the bias exactly:

.. math:: \hat{m}_t = \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}}

At :math:`t=1` this rescales :math:`m` by :math:`1/(1-0.9) = 10` -- recovering
:math:`g` exactly -- and the correction decays to 1 as :math:`t` grows. Without
it, Adam takes a badly-scaled first few hundred steps; with it, the very first
step is already sensible. It is the difference between Adam and "momentum
divided by a bad variance estimate".

Epsilon
-------
:math:`\epsilon = 10^{-8}` in the denominator prevents division by zero when a
gradient has been zero throughout. It also caps the maximum step at roughly
:math:`\eta/\epsilon` for a parameter that suddenly receives its first gradient.

Adam vs SGD
-----------
Adam converges faster in wall-clock terms and needs far less learning-rate
tuning, which is why it is the default for most work. SGD with momentum,
carefully tuned, often generalises slightly better on large vision models --
an active area of debate. Phase 22 runs both on the same problem.

Cost: Adam stores two extra floats per parameter, so optimiser state is 2x the
model size. For a 109k-parameter MLP that is nothing; for a 100B-parameter
language model it is the reason training needs so much memory.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..nn.parameter import Parameter
from .optimizer import Optimizer

__all__ = ["Adam", "AdamW"]


def _restore(value: Any) -> Any:
    """Rebuild a moment buffer entry, which may be a float or an array."""
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    return float(value)


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2014).

    Parameters
    ----------
    params
        Usually ``model.parameters()``.
    lr
        Learning rate. ``1e-3`` is the standard default and usually works.
    betas
        ``(beta1, beta2)`` decay rates for the first and second moments.
        ``beta1=0.9`` averages over roughly the last 10 gradients;
        ``beta2=0.999`` over roughly the last 1000.
    eps
        Added to the denominator for numerical stability.
    weight_decay
        L2 penalty folded into the gradient. See ``AdamW`` for why that is not
        quite the same as true weight decay.

    Examples
    --------
    >>> from nabla.nn import MLP
    >>> model = MLP(2, [4], 1, seed=0)
    >>> opt = Adam(model.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        n = len(self.params)
        self.m: list[float] = [0.0] * n  # first moment  (mean)
        self.v: list[float] = [0.0] * n  # second moment (uncentred variance)

    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        b1, b2 = self.beta1, self.beta2

        # Bias-correction factors depend only on t, so compute them once for
        # the whole parameter list rather than per parameter.
        bias1 = 1.0 - b1**t
        bias2 = 1.0 - b2**t

        for i, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data

            # m ← β₁m + (1-β₁)g        running mean of the gradient
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad
            # v ← β₂v + (1-β₂)g²       running mean of the squared gradient
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad * grad

            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2

            # θ ← θ - η·m̂/(√v̂ + ε)
            # ``** 0.5`` rather than ``math.sqrt``: it works elementwise on a
            # NumPy array as well as on a float, so one optimiser serves both
            # the scalar and the tensor engine.
            param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
                "m": [x.tolist() if hasattr(x, "tolist") else x for x in self.m],
                "v": [x.tolist() if hasattr(x, "tolist") else x for x in self.v],
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.eps = float(state.get("eps", self.eps))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        for key in ("m", "v"):
            values = state.get(key)
            if values is None:
                continue
            if len(values) != len(self.params):
                raise ValueError(
                    f"Adam '{key}' buffer has {len(values)} entries for "
                    f"{len(self.params)} parameters"
                )
            setattr(self, key, [_restore(v) for v in values])

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr:g}, betas=({self.beta1:g}, {self.beta2:g}), "
            f"params={len(self.params):,})"
        )


class AdamW(Adam):
    r"""Adam with **decoupled** weight decay (Loshchilov & Hutter, 2017).

    The distinction is easy to miss and genuinely matters. Ordinary Adam folds
    the L2 penalty into the gradient, so it then passes through the
    :math:`1/\sqrt{\hat v}` rescaling along with everything else. A parameter
    with large gradients gets a large :math:`\sqrt{\hat v}` and therefore
    *less* regularisation -- exactly backwards from what was intended, since
    the penalty was meant to be uniform.

    AdamW applies the decay directly to the parameter, outside the adaptive
    machinery:

    .. math:: \theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon} - \eta\lambda\theta

    This is why modern transformer training uses AdamW rather than Adam.
    """

    def step(self) -> None:
        decay = self.weight_decay
        lr = self.lr

        # Run the adaptive update with the penalty switched off...
        self.weight_decay = 0.0
        try:
            super().step()
        finally:
            self.weight_decay = decay

        # ...then decay the parameters directly, untouched by 1/√v̂.
        if decay:
            for param in self.params:
                param.data -= lr * decay * param.data

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr:g}, weight_decay={self.weight_decay:g}, "
            f"params={len(self.params):,})"
        )
