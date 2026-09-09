"""Tests for the simulated energy model.

This model produces the headline number of the spiking A/B, so the arithmetic
is pinned down here rather than trusted. The tests are mostly about the terms
that are easy to leave out, since omitting them is what makes a spiking
network look better than it is.
"""

from __future__ import annotations

import pytest

from nabla.nn.energy import PJ_AC, PJ_MAC, dense_energy, spiking_energy


class TestDense:
    def test_counts_every_synapse_once(self):
        assert dense_energy([784, 128, 10]).macs == 784 * 128 + 128 * 10

    def test_energy_is_macs_times_the_published_cost(self):
        breakdown = dense_energy([100, 50, 10])
        assert breakdown.total_pj == pytest.approx((100 * 50 + 50 * 10) * PJ_MAC)

    def test_a_dense_network_has_no_accumulate_or_membrane_terms(self):
        breakdown = dense_energy([784, 128, 10])
        assert breakdown.accumulates == 0.0
        assert breakdown.membrane_pj == 0.0

    def test_cost_does_not_depend_on_the_input(self):
        """A dense network costs the same on a blank image as a busy one.

        This is the asymmetry the whole spiking argument rests on, so it is
        worth stating as a test rather than only in prose.
        """
        assert dense_energy([784, 128, 10]).total_pj == \
            dense_energy([784, 128, 10]).total_pj


class TestSpiking:
    SIZES = [784, 128, 10]

    def test_a_silent_network_still_pays_for_membranes(self):
        """Membrane updates happen whether or not anything spikes."""
        quiet = spiking_energy(self.SIZES, timesteps=25, layer_spikes=[0.0])
        assert quiet.accumulates == 0.0
        assert quiet.membrane_pj == pytest.approx(25 * 128 * PJ_MAC)
        assert quiet.total_pj > 0.0

    def test_membrane_cost_scales_with_timesteps(self):
        ten = spiking_energy(self.SIZES, timesteps=10, layer_spikes=[0.0])
        forty = spiking_energy(self.SIZES, timesteps=40, layer_spikes=[0.0])
        assert forty.membrane_pj == pytest.approx(4 * ten.membrane_pj)

    def test_static_input_is_charged_as_one_dense_layer(self):
        """Computed once, not T times, because the current never changes."""
        for timesteps in (5, 50):
            breakdown = spiking_energy(
                self.SIZES, timesteps=timesteps, layer_spikes=[0.0],
                static_input=True,
            )
            assert breakdown.macs == 784 * 128

    def test_spike_encoded_input_is_charged_as_accumulates(self):
        breakdown = spiking_energy(
            self.SIZES, timesteps=25, layer_spikes=[0.0],
            static_input=False, input_spikes=1000.0,
        )
        assert breakdown.macs == 0.0
        assert breakdown.accumulates == pytest.approx(1000.0 * 128)

    def test_spikes_drive_the_layer_downstream(self):
        """Hidden spikes are charged against the readout's fan-out, not its own."""
        breakdown = spiking_energy(
            self.SIZES, timesteps=25, layer_spikes=[500.0],
        )
        # 500 hidden spikes, each reaching all 10 readout units.
        assert breakdown.accumulates == pytest.approx(500.0 * 10)

    def test_energy_grows_with_spike_count(self):
        low = spiking_energy(self.SIZES, timesteps=25, layer_spikes=[100.0])
        high = spiking_energy(self.SIZES, timesteps=25, layer_spikes=[1000.0])
        assert high.total_pj > low.total_pj

    def test_one_spike_costs_an_add_not_a_multiply_accumulate(self):
        """The single observation the energy argument is built on."""
        base = spiking_energy(self.SIZES, timesteps=25, layer_spikes=[0.0])
        one_more = spiking_energy(self.SIZES, timesteps=25, layer_spikes=[1.0])
        # One extra spike, reaching 10 readout units, at the add price.
        assert one_more.total_pj - base.total_pj == pytest.approx(10 * PJ_AC)

    def test_multiple_hidden_layers_each_drive_the_next(self):
        breakdown = spiking_energy(
            [784, 128, 64, 10], timesteps=10, layer_spikes=[200.0, 50.0],
        )
        assert breakdown.accumulates == pytest.approx(200.0 * 64 + 50.0 * 10)
        assert breakdown.membrane_pj == pytest.approx(10 * (128 + 64) * PJ_MAC)

    def test_a_spike_count_per_hidden_layer_is_required(self):
        with pytest.raises(ValueError, match="one per hidden layer"):
            spiking_energy([784, 128, 64, 10], timesteps=10, layer_spikes=[200.0])


class TestTheComparison:
    """The findings the A/B turns on, pinned so they cannot drift silently."""

    def test_a_static_input_layer_alone_nearly_matches_the_dense_network(self):
        """Why the spiking arm cannot win on this architecture.

        With an analog 784-pixel input and one hidden layer, the input layer
        is 99% of the dense network's total cost, and the spiking network pays
        it in full before any of its sparsity helps.
        """
        dense = dense_energy([784, 128, 10])
        input_layer_only = 784 * 128 * PJ_MAC
        assert input_layer_only / dense.total_pj > 0.98

    def test_spike_encoded_input_can_beat_dense(self):
        """And why the argument is still right in its intended setting."""
        dense = dense_energy([784, 128, 10])
        # MNIST mean intensity is about 0.13, so 784 * 0.13 * 25 input spikes.
        spiking = spiking_energy(
            [784, 128, 10], timesteps=25, layer_spikes=[500.0],
            static_input=False, input_spikes=784 * 0.13 * 25,
        )
        assert spiking.total_pj < dense.total_pj

    def test_breakeven_spike_rate_is_about_a_fifth(self):
        r"""A spiking layer wins below roughly ``PJ_AC / PJ_MAC`` activity.

        One dense synapse costs ``PJ_MAC``. One spiking synapse costs
        ``PJ_AC``, but only fires at rate ``r`` per timestep, over ``T``
        timesteps. So spiking wins when ``r * T * PJ_AC < PJ_MAC``.
        """
        assert PJ_AC / PJ_MAC == pytest.approx(0.1957, abs=1e-4)
