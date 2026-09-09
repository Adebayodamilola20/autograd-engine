r"""A simulated energy model for dense and spiking inference.

This is an **estimate from an operation count**, not a measurement. Nothing
here was run on neuromorphic hardware, and no wall-clock power was measured.
What it does is apply the standard published per-operation energies to the
operations each network actually performs, which is how essentially every
SNN paper reports energy. Treating the output as a measurement would be
wrong; treating it as a principled comparison of operation counts is fair.

The per-operation numbers
-------------------------
From Horowitz, "Computing's Energy Problem (and what we can do about it)",
ISSCC 2014, for 45nm CMOS at 0.9V:

===================================  ========
32-bit float multiply-accumulate     4.6 pJ
32-bit float add (accumulate only)   0.9 pJ
===================================  ========

The MAC figure is the 3.7 pJ multiplier plus the 0.9 pJ adder.

Why a spike is cheaper
----------------------
This is the whole energy argument for spiking networks, and it is a single
observation. A dense layer computes :math:`\sum_i w_i x_i`, where every
:math:`x_i` is a real number, so every synapse needs a **multiply and an add**.

When the input is a binary spike, :math:`x_i \in \{0, 1\}`, so
:math:`w_i x_i` is either :math:`w_i` or nothing. There is no multiply. A
synapse whose input did not spike does no work at all, and one whose input did
spike does a single **add**.

So a spiking layer costs ``(number of spikes) x fanout`` accumulates, against a
dense layer's ``fan_in x fan_out`` multiply-accumulates, and the ratio is
:math:`\frac{0.9}{4.6} \times (\text{spike rate}) \approx 0.2 \times r`. Below
about 20% activity the spiking layer wins on this accounting; above it, it
does not.

The three costs that get quietly dropped
----------------------------------------
Published comparisons often count only the spike-driven synapses. Three real
costs are left out, and all three are included here, because leaving them out
is how a spiking network is made to look better than it is:

1. **Membrane updates.** Every neuron, every timestep, does a decay multiply
   and an add whether or not anything spiked. That is ``T x n_neurons`` MACs
   and it is charged even for a completely silent network.
2. **A static analog input layer.** If the input is injected as constant
   current rather than encoded as spikes, the first layer is doing real
   multiplies. It is computed once rather than ``T`` times, but it is a full
   dense layer's worth of MACs, and on a small network it can be most of the
   total.
3. **Timesteps.** The spiking network runs ``T`` times to the dense network's
   once. Cost 1 scales with ``T`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PJ_MAC",
    "PJ_AC",
    "EnergyBreakdown",
    "dense_energy",
    "spiking_energy",
]

# Picojoules, 45nm CMOS, 32-bit floating point (Horowitz, ISSCC 2014).
PJ_MAC = 4.6
PJ_AC = 0.9


@dataclass(frozen=True)
class EnergyBreakdown:
    """Energy per inference, in picojoules, itemised.

    Itemised rather than a single number so that the result can be argued
    with: the interesting question is almost always which term dominates, and
    a scalar hides that.
    """

    macs: float
    accumulates: float
    synaptic_pj: float
    membrane_pj: float

    @property
    def total_pj(self) -> float:
        return self.synaptic_pj + self.membrane_pj

    @property
    def total_nj(self) -> float:
        return self.total_pj / 1000.0

    def as_dict(self) -> dict[str, float]:
        return {
            "macs": self.macs,
            "accumulates": self.accumulates,
            "synaptic_pj": self.synaptic_pj,
            "membrane_pj": self.membrane_pj,
            "total_pj": self.total_pj,
            "total_nj": self.total_nj,
        }


def dense_energy(sizes: Sequence[int]) -> EnergyBreakdown:
    """One forward pass of a dense network with layer widths ``sizes``.

    Every synapse in every layer performs exactly one multiply-accumulate,
    once. There is no state, no time dimension, and no dependence on the data:
    a dense network costs the same on a blank image as on a busy one.

    Examples
    --------
    >>> breakdown = dense_energy([784, 128, 10])
    >>> breakdown.macs
    101632.0
    >>> round(breakdown.total_nj, 2)
    467.51
    """
    macs = float(sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)))
    return EnergyBreakdown(
        macs=macs,
        accumulates=0.0,
        synaptic_pj=macs * PJ_MAC,
        membrane_pj=0.0,
    )


def spiking_energy(
    sizes: Sequence[int],
    *,
    timesteps: int,
    layer_spikes: Sequence[float],
    static_input: bool = True,
    input_spikes: float = 0.0,
) -> EnergyBreakdown:
    """One inference of a spiking network, averaged per sample.

    Parameters
    ----------
    sizes
        Layer widths, e.g. ``[784, 128, 10]``. Hidden layers are the entries
        between the first and last.
    timesteps
        How long the network was simulated.
    layer_spikes
        Mean spikes emitted **per sample, summed over all timesteps**, by each
        hidden layer. One entry per hidden layer. These drive the layer
        downstream of them.
    static_input
        ``True`` when the input was injected as constant analog current, in
        which case the first layer is a dense MAC layer evaluated once.
        ``False`` when the input arrived as spikes, making the first layer
        spike-driven like the rest.
    input_spikes
        Mean input spikes per sample summed over time. Used only when
        ``static_input`` is ``False``.

    Examples
    --------
    A silent network still pays for its membrane updates:

    >>> quiet = spiking_energy([784, 128, 10], timesteps=25, layer_spikes=[0.0])
    >>> quiet.accumulates
    0.0
    >>> quiet.membrane_pj > 0
    True
    """
    hidden = list(sizes[1:-1])
    if len(layer_spikes) != len(hidden):
        raise ValueError(
            f"expected {len(hidden)} spike counts, one per hidden layer, "
            f"got {len(layer_spikes)}"
        )

    macs = 0.0
    accumulates = 0.0

    # --- the input layer ---------------------------------------------
    if static_input:
        # Constant current: I = Wx + b is identical at every timestep, so it
        # is computed once. Full dense cost, paid a single time.
        macs += float(sizes[0] * sizes[1])
    else:
        # Spike-encoded input: only the synapses whose presynaptic neuron
        # fired do anything, and each is an add.
        accumulates += float(input_spikes * sizes[1])

    # --- spike-driven layers -----------------------------------------
    # Each hidden layer's spikes drive the layer after it. The last hidden
    # layer drives the readout.
    for index, spikes in enumerate(layer_spikes):
        fanout = sizes[index + 2]
        accumulates += float(spikes * fanout)

    # --- membrane updates --------------------------------------------
    # Every hidden neuron, every timestep: one decay multiply and one add,
    # whether or not it spiked. Charged even when the network is silent.
    membrane_ops = float(timesteps * sum(hidden))

    return EnergyBreakdown(
        macs=macs,
        accumulates=accumulates,
        synaptic_pj=macs * PJ_MAC + accumulates * PJ_AC,
        membrane_pj=membrane_ops * PJ_MAC,
    )
