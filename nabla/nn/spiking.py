r"""Spiking neural networks: LIF neurons trained by surrogate gradients.

The problem, stated exactly
---------------------------
A spiking neuron emits a spike when its membrane potential crosses a threshold.
As a function that is the Heaviside step:

.. math:: S = \Theta(U - \theta)

which is exactly the function backpropagation cannot handle. Its derivative is
zero everywhere it is defined, and undefined at the one point that matters:

.. math::
    \frac{\partial S}{\partial U} = 0 \quad (U \neq \theta),
    \qquad \text{undefined at } U = \theta

Run that through the chain rule and every gradient in the network is
multiplied by zero. The network does not train badly; it does not train at
all. This is the *dead neuron problem*, and it is why spiking networks were
trained by biologically-motivated local rules for decades rather than by
gradient descent.

The fix, and why it is a lie that works
---------------------------------------
Surrogate gradients. Use the true step function on the forward pass, so the
network really is spiking and really is binary, but substitute a smooth,
non-zero derivative on the backward pass:

.. math::
    S = \Theta(U - \theta)
    \qquad\text{but}\qquad
    \frac{\partial S}{\partial U} := \sigma'(U - \theta)

**Forward and backward deliberately disagree.** The backward pass computes the
gradient of a function the forward pass did not compute. Every other backward
rule in this repository is the true derivative of its forward operation, and
``gradcheck`` verifies exactly that; this one is not, and cannot be. Finite
differences of the real forward function would return zero, because the real
forward function has zero gradient. That is not a bug in the check, it is the
thing being worked around.

What can still be verified is that the implementation computes the surrogate
it claims to: ``tests/test_spiking.py`` gradient-checks each spike node
against the smooth function it is standing in for, which catches an algebra
error in the surrogate without pretending the step function is
differentiable.

Justification for the substitution: :math:`\Theta` is the limit of a family of
smooth sigmoids as their width goes to zero. The surrogate is the derivative
of a member of that family with non-zero width. So this is a gradient of a
*nearby* network, which is enough to descend on, and it is the standard method
in the field (Neftci, Mostafa and Zenke, 2019).

The neuron
----------
Leaky integrate-and-fire, in discrete time:

.. math::
    U[t] = \lambda U[t-1] + I[t] - \theta S[t-1]

    S[t] = \Theta(U[t] - \theta)

Three terms, each doing one job:

* :math:`\lambda U[t-1]` -- **leak**. Membrane potential decays toward rest, so
  inputs that do not arrive close together in time fail to add up. This is what
  makes the neuron sensitive to timing rather than to a running total.
* :math:`I[t]` -- **input current**, the usual weighted sum of what arrived.
* :math:`-\theta S[t-1]` -- **reset**. After firing, drop by one threshold.

Reset by subtraction rather than to zero. Both appear in the literature;
subtraction preserves the remainder above threshold instead of discarding it,
which loses less information when the input is strong, and keeps a gradient
path through the reset that reset-to-zero severs.

Backpropagation through time
----------------------------
``U[t]`` depends on ``U[t-1]``, so unrolling ``T`` timesteps builds a graph
``T`` steps deep in the time direction as well as deep in the layer direction.
Nothing here implements BPTT: writing the recurrence in a Python loop *is* the
unrolling, and ``backward()`` walks it because it is one graph like any other.
That is worth noticing, since BPTT is usually presented as its own algorithm.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..core.tensor import Tensor
from .module import Module
from .tensor_mlp import Linear

__all__ = [
    "SURROGATES",
    "spike",
    "surrogate_derivative",
    "LIF",
    "SpikingMLP",
    "rate_encode",
]


# ======================================================================
# the surrogate derivatives
# ======================================================================


def _fast_sigmoid(x: np.ndarray, sharpness: float) -> np.ndarray:
    r"""SuperSpike: :math:`(1 + \beta|x|)^{-2}`.

    From Zenke and Ganguli (2018). Heavy tails: a neuron far from threshold
    still receives some gradient, so a unit that has stopped firing can be
    recruited back. Cheap, and the most common choice.
    """
    return 1.0 / (1.0 + sharpness * np.abs(x)) ** 2


def _atan(x: np.ndarray, sharpness: float) -> np.ndarray:
    r""":math:`\left(1 + (\tfrac{\pi}{2}\beta x)^2\right)^{-1}`.

    The derivative of a scaled arctangent. Lighter tails than SuperSpike and a
    sharper peak, so gradient concentrates on neurons near threshold.
    """
    return 1.0 / (1.0 + (0.5 * np.pi * sharpness * x) ** 2)


def _sigmoid_derivative(x: np.ndarray, sharpness: float) -> np.ndarray:
    r""":math:`4\sigma(\beta x)(1 - \sigma(\beta x))`, the logistic bump.

    The most obvious choice and the easiest to reason about, since the step
    function genuinely is the zero-width limit of the logistic. The factor of
    4 normalises the peak to 1 so that ``sharpness`` and the learning rate stay
    independent.
    """
    s = 1.0 / (1.0 + np.exp(-sharpness * np.clip(x, -60.0, 60.0)))
    return 4.0 * s * (1.0 - s)


SURROGATES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "fast_sigmoid": _fast_sigmoid,
    "atan": _atan,
    "sigmoid": _sigmoid_derivative,
}


def surrogate_derivative(
    x: np.ndarray, kind: str = "fast_sigmoid", sharpness: float = 5.0
) -> np.ndarray:
    """The surrogate :math:`\\partial S/\\partial U` at distance ``x`` from threshold.

    All three are normalised to peak at 1 when ``x == 0``, so ``sharpness``
    changes the *width* of the gradient window without also rescaling every
    gradient in the network. Without that normalisation, tuning the surrogate
    silently retunes the learning rate too.
    """
    try:
        return SURROGATES[kind](np.asarray(x, dtype=np.float64), sharpness)
    except KeyError:
        raise ValueError(
            f"unknown surrogate {kind!r}; available: "
            f"{', '.join(sorted(SURROGATES))}"
        ) from None


# ======================================================================
# the spike node
# ======================================================================


def spike(
    potential: Tensor,
    *,
    threshold: float = 1.0,
    surrogate: str = "fast_sigmoid",
    sharpness: float = 5.0,
) -> Tensor:
    r"""Emit binary spikes, and pass a smooth gradient back through them.

    Forward is exactly :math:`\Theta(U - \theta)`: the output is 0.0 or 1.0 and
    nothing else. Backward multiplies by the surrogate derivative.

    This is the only node in the repository whose backward is not the
    derivative of its forward, and the deviation is the entire technique
    rather than an approximation for speed. See the module docstring.

    Examples
    --------
    >>> import numpy as np
    >>> u = Tensor(np.array([0.0, 0.9, 1.1, 5.0]))
    >>> s = spike(u, threshold=1.0)
    >>> s.data.tolist()
    [0.0, 0.0, 1.0, 1.0]
    >>> s.sum().backward()
    >>> bool(np.all(u.grad > 0))      # every unit gets gradient, even silent ones
    True
    """
    shifted = potential.data - threshold
    out = Tensor((shifted > 0.0).astype(np.float64), (potential,), "spike")

    def _backward() -> None:
        potential.grad = potential.grad + (
            surrogate_derivative(shifted, surrogate, sharpness) * out.grad
        )

    out._backward = _backward
    return out


# ======================================================================
# layers
# ======================================================================


class LIF(Module):
    r"""A layer of leaky integrate-and-fire neurons.

    Holds the synaptic weights and the membrane state. ``step()`` advances the
    simulation by one timestep; the caller drives the loop, because the loop
    *is* the BPTT unrolling and hiding it would hide the point.

    Parameters
    ----------
    n_in, n_out
        Synapse counts. The weights are an ordinary dense layer: what makes
        this spiking is what happens to the result, not how it is computed.
    decay
        :math:`\lambda`, the membrane leak per timestep, in ``[0, 1]``. At 0
        the neuron has no memory and the network is a stack of thresholded
        dense layers; at 1 it never forgets, which is the non-leaky
        integrate-and-fire neuron. Around 0.9 gives a memory of roughly ten
        timesteps.
    threshold
        :math:`\theta`, the firing level.
    learn_decay
        Whether :math:`\lambda` is trained. Off by default: it is a single
        scalar per layer, it interacts awkwardly with the surrogate, and the
        A/B this was written for needs the two arms to differ in as few ways as
        possible.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        decay: float = 0.9,
        threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        sharpness: float = 5.0,
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        # 1.0 inclusive: that is the non-leaky integrate-and-fire neuron, which
        # is a standard model in its own right (and the one most ANN-to-SNN
        # conversion work uses), not an invalid setting.
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if surrogate not in SURROGATES:
            raise ValueError(
                f"unknown surrogate {surrogate!r}; available: "
                f"{', '.join(sorted(SURROGATES))}"
            )
        self.n_in = n_in
        self.n_out = n_out
        self.decay = decay
        self.threshold = threshold
        self.surrogate = surrogate
        self.sharpness = sharpness
        self.synapse = Linear(n_in, n_out, activation="linear", rng=rng,
                              label=f"{label}syn.")

    # ------------------------------------------------------------------

    def initial_state(self, batch: int) -> Tensor:
        """Membrane potential at ``t = 0``: every neuron at rest."""
        return Tensor(np.zeros((batch, self.n_out)))

    def step_from_current(
        self, current: Tensor, potential: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""One timestep, given the input current already computed.

        .. math::
            U[t] = \lambda U[t-1] + I[t] - \theta S[t-1]

        The reset uses the spikes computed *from the updated potential*, so a
        neuron that fires at ``t`` is reset going into ``t+1``. Writing it this
        way (rather than carrying ``S[t-1]`` separately) keeps the state to a
        single tensor and the recurrence to a single line.

        Separated from ``step`` because when the input to a layer does not
        change over time, :math:`I` is the same every timestep and computing it
        once is both ``T`` times less work and the honest thing to charge for
        in an energy model.
        """
        updated = potential * self.decay + current
        spikes = spike(
            updated,
            threshold=self.threshold,
            surrogate=self.surrogate,
            sharpness=self.sharpness,
        )
        # Detach nothing: the reset term carries gradient too, and severing it
        # is a common and quiet way to make a spiking network train worse.
        return spikes, updated - spikes * self.threshold

    def step(self, x: Tensor, potential: Tensor) -> tuple[Tensor, Tensor]:
        """One timestep from a layer input. Returns ``(spikes, new_potential)``."""
        return self.step_from_current(self.synapse(x), potential)


class SpikingMLP(Module):
    r"""A feed-forward spiking network, matched to ``TensorMLP``.

    Same layer sizes, same weight initialisation, same parameter count. The
    differences are that every hidden unit emits binary spikes rather than a
    real number, and that the whole thing runs for ``timesteps`` steps per
    input.

    The output layer does not spike
        It integrates its input current and the accumulated membrane potential
        is used as the logits. Counting output spikes also works, but throws
        away magnitude: with ``T`` timesteps a spike count can only take
        ``T + 1`` values, so ten classes separated by a handful of spikes give
        a coarse, tie-prone signal, and cross-entropy gets a much weaker
        gradient. Reading out the potential is standard practice and costs
        nothing extra.

    Parameters
    ----------
    timesteps
        How long to simulate per input. The central accuracy/energy dial: more
        steps means more chances to spike and a finer rate code, at linearly
        more compute and, on real neuromorphic hardware, linearly more energy.

    Examples
    --------
    >>> import numpy as np
    >>> net = SpikingMLP(4, [8], 3, timesteps=5, seed=0)
    >>> logits, stats = net(np.zeros((2, 4)), collect_stats=True)
    >>> logits.shape
    (2, 3)
    >>> stats["spike_count"]
    0.0
    """

    def __init__(
        self,
        n_in: int,
        hidden: Sequence[int],
        n_out: int,
        *,
        timesteps: int = 25,
        decay: float = 0.9,
        threshold: float = 1.0,
        surrogate: str = "fast_sigmoid",
        sharpness: float = 5.0,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if timesteps < 1:
            raise ValueError(f"timesteps must be at least 1, got {timesteps}")
        rng = rng or np.random.default_rng(seed)

        self.sizes = [n_in, *hidden, n_out]
        self.n_in = n_in
        self.n_out = n_out
        self.timesteps = timesteps
        self.decay = decay
        self.threshold = threshold
        self.surrogate = surrogate
        self.sharpness = sharpness

        self.layers = [
            LIF(self.sizes[i], self.sizes[i + 1], decay=decay, threshold=threshold,
                surrogate=surrogate, sharpness=sharpness, rng=rng, label=f"L{i}.")
            for i in range(len(self.sizes) - 2)
        ]
        # The readout is a plain dense layer. Its "membrane" is the running sum
        # of its input current, which needs no threshold and therefore no
        # surrogate.
        self.readout = Linear(self.sizes[-2], n_out, activation="linear",
                              rng=rng, label="out.")

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor | np.ndarray,
        *,
        collect_stats: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """``(batch, n_in)`` -> ``(batch, n_out)`` logits.

        The input is injected as a constant current at every timestep, rather
        than converted to a Poisson spike train. Both are standard; constant
        injection is deterministic, needs far fewer timesteps for the same
        accuracy, and keeps this comparison about the *network* rather than
        about how much noise the encoder added. ``rate_encode`` is provided for
        the other choice.
        """
        current = x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float64))
        batch = current.shape[0]

        potentials = [layer.initial_state(batch) for layer in self.layers]
        accumulated: Tensor | None = None
        spike_total = 0.0

        # The input is injected unchanged at every timestep, so the first
        # layer's synaptic current is identical on every one of them. Compute
        # it once. This is the dominant matmul in the network (784x128 against
        # 128x10 downstream), so hoisting it makes the forward pass nearly T
        # times cheaper, and it is what the energy model must charge for: one
        # pass of multiply-accumulates, not T of them.
        input_current = self.layers[0].synapse(current)

        for _ in range(self.timesteps):
            signal, potentials[0] = self.layers[0].step_from_current(
                input_current, potentials[0]
            )
            if collect_stats:
                # Read .data, not the graph: this is instrumentation, and must
                # not add a node or a gradient path.
                spike_total += float(signal.data.sum())
            for i, layer in enumerate(self.layers[1:], start=1):
                signal, potentials[i] = layer.step(signal, potentials[i])
                if collect_stats:
                    spike_total += float(signal.data.sum())
            step_out = self.readout(signal)
            accumulated = step_out if accumulated is None else accumulated + step_out

        # Mean rather than sum over time, so the logit scale does not depend on
        # `timesteps`. Without this, changing T silently rescales the loss and
        # the effective learning rate along with it.
        logits = accumulated * (1.0 / self.timesteps)

        if not collect_stats:
            return logits
        return logits, {
            "spike_count": spike_total,
            "spikes_per_sample": spike_total / batch,
            "spike_rate": spike_total / (batch * self.timesteps * self.n_hidden_units()),
        }

    # ------------------------------------------------------------------

    def n_hidden_units(self) -> int:
        return sum(self.sizes[1:-1])

    def synapses_per_timestep(self) -> list[int]:
        """Fan-out of each layer, for the energy model."""
        return [self.sizes[i] * self.sizes[i + 1] for i in range(len(self.sizes) - 1)]

    def config(self) -> dict:
        return {
            "sizes": self.sizes,
            "timesteps": self.timesteps,
            "decay": self.decay,
            "threshold": self.threshold,
            "surrogate": self.surrogate,
            "sharpness": self.sharpness,
        }

    def summary(self) -> str:
        lines = [
            f"SpikingMLP({' -> '.join(map(str, self.sizes))}, "
            f"T={self.timesteps}, decay={self.decay}, "
            f"surrogate={self.surrogate})",
            f"{'layer':<14} {'type':<10} {'params':>10}",
            "-" * 38,
        ]
        for i, layer in enumerate(self.layers):
            lines.append(f"{'L' + str(i):<14} {'LIF':<10} "
                         f"{layer.num_parameters():>10,}")
        lines.append(f"{'readout':<14} {'Linear':<10} "
                     f"{self.readout.num_parameters():>10,}")
        lines.append("-" * 38)
        lines.append(f"{'total':<14} {'':<10} {self.num_parameters():>10,}")
        return "\n".join(lines)


# ======================================================================
# input encoding
# ======================================================================


def rate_encode(
    x: np.ndarray, timesteps: int, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Poisson rate coding: ``(batch, features)`` -> ``(T, batch, features)``.

    Each pixel intensity in ``[0, 1]`` becomes the per-timestep probability of
    a spike, so a bright pixel fires often and a dark one rarely. This is what
    a real spiking sensor produces, and it is genuinely stochastic: the same
    image gives a different spike train every time.

    Not used by ``SpikingMLP`` by default. It needs many more timesteps to
    average the sampling noise away, and in a comparison against a dense
    network that extra noise would be charged to "spiking" when it actually
    belongs to the encoder.
    """
    rng = rng or np.random.default_rng()
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return (rng.random((timesteps, *x.shape)) < x).astype(np.float64)
