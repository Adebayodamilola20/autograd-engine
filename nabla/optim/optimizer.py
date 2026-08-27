r"""Phase 10 -- the optimiser base class.

What an optimiser is
--------------------
Backpropagation answers *"which direction increases the loss?"*. It does not
say what to do about it. That is the optimiser's job, and its entire contract
is one method:

.. math:: \theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}

Move each parameter a small step **against** its gradient. The minus sign is
the whole idea: the gradient points uphill, so we walk downhill.

Why a step size at all
----------------------
The gradient is a *local* linearisation -- it tells you the slope exactly at
the current point, and says nothing about how far that slope holds. Take too
big a step and you overshoot into terrain the linearisation never described;
too small and you crawl. The learning rate :math:`\eta` is the admission that
we only trust the gradient nearby, and it is the single most important
hyperparameter in deep learning. Phase 22 sweeps it and shows both failure
modes directly.

Why optimisers get more complicated than that one line
-------------------------------------------------------
Plain gradient descent struggles with **ill-conditioned** loss surfaces --
valleys that are steep in one direction and nearly flat in another. The step
size that is stable for the steep direction is far too small for the flat one,
so you oscillate across the valley while barely advancing along it. Momentum
and Adam are two different answers to that problem; each subclass explains its
own.

State
-----
An optimiser that remembers anything between steps (momentum buffers, Adam's
moment estimates) keeps it here, keyed by parameter identity. That state is
part of a checkpoint: resuming training from saved weights *without* the
optimiser state restarts momentum from zero and produces a visible bump in the
loss curve.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..nn.parameter import Parameter

__all__ = ["Optimizer"]


class Optimizer:
    """Base class: holds parameters, exposes ``step()`` and ``zero_grad()``.

    Parameters
    ----------
    params
        The parameters to update, normally ``model.parameters()``.
    lr
        Learning rate.
    """

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params: list[Parameter] = list(params)
        if not self.params:
            raise ValueError(
                "optimizer got an empty parameter list -- did you pass "
                "model.parameters() rather than the model?"
            )
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}")

        self.lr = float(lr)
        self.step_count = 0

    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """Clear every parameter's gradient.

        Must be called before each backward pass. ``Value.backward()``
        accumulates by design (decision D6), so without this, step *n* uses the
        sum of gradients from steps 1..n. The model still appears to train --
        badly, with an effective learning rate that grows without bound --
        which is what makes the bug so hard to spot. Reproducing PyTorch's
        semantics here means reproducing its most famous footgun on purpose.

        Delegates to ``param.reset_grad()`` rather than assigning ``0.0``: a
        tensor parameter must reset to a zero *array* of the right shape, and a
        bare float would silently lose that shape until the next backward pass.
        """
        for param in self.params:
            param.reset_grad()

    def step(self) -> None:
        """Apply one update. Implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------

    def gradient_norm(self) -> float:
        r"""Global :math:`L^2` norm :math:`\sqrt{\sum_i g_i^{2}}` over all
        parameters.

        The standard single-number health check on a training run. Growing
        without bound means gradients are exploding; collapsing toward zero
        means they are vanishing or the model has converged.
        """
        total = 0.0
        for p in self.params:
            g = p.grad
            if isinstance(g, np.ndarray):
                # One tensor parameter carries many gradients; the global norm
                # is over every scalar, so flatten before accumulating.
                total += float(np.dot(g.ravel(), g.ravel()))
            else:
                total += g * g
        return total**0.5

    def clip_grad_norm(self, max_norm: float) -> float:
        r"""Rescale all gradients so their global norm is at most ``max_norm``.

        .. math::
            \mathbf{g} \leftarrow \mathbf{g}\cdot\min\!\left(1,
            \frac{\text{max\_norm}}{\|\mathbf{g}\|}\right)

        Scaling the whole gradient vector by one factor preserves its
        *direction* and only shortens the step -- unlike clipping each
        component independently, which would bend the update away from the true
        descent direction.

        The standard remedy for exploding gradients. Returns the norm
        **before** clipping, which is the number worth logging.
        """
        total = self.gradient_norm()
        if total > max_norm and total > 0:
            scale = max_norm / total
            for param in self.params:
                param.grad *= scale
        return total

    def state_dict(self) -> dict[str, Any]:
        """Optimiser hyperparameters and step count, for checkpointing.

        Subclasses with per-parameter buffers extend this.
        """
        return {"lr": self.lr, "step_count": self.step_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = float(state.get("lr", self.lr))
        self.step_count = int(state.get("step_count", 0))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr:g}, params={len(self.params):,})"
