"""Tests for the spiking layer.

Testing this honestly needs care, because the central operation is one whose
backward pass is *deliberately not* the derivative of its forward pass. The
usual finite-difference check does not apply and must not be faked into
appearing to pass.

What is verified instead:

* the forward pass really is the Heaviside step (binary, exact),
* the true gradient really is zero, so the problem being solved is real
  (``TestTheDeadNeuronProblem``),
* each surrogate really is the derivative of the smooth function it claims to
  stand for, checked by finite differences against that function
  (``TestSurrogateGradients``) -- this catches an algebra slip in the surrogate
  without pretending the step function is differentiable,
* the LIF recurrence leaks, integrates, fires and resets as specified,
* gradient flows backward across timesteps, which is what BPTT means here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nabla.core.tensor import Tensor
from nabla.losses.tensor_losses import tensor_cross_entropy
from nabla.nn.spiking import (
    LIF,
    SURROGATES,
    SpikingMLP,
    rate_encode,
    spike,
    surrogate_derivative,
)
from nabla.nn.tensor_mlp import TensorMLP

# The smooth function each surrogate is the derivative of. Verified by
# differentiating these numerically, below.
ANTIDERIVATIVES = {
    "fast_sigmoid": lambda x, b: x / (1.0 + b * np.abs(x)),
    "atan": lambda x, b: (2.0 / (np.pi * b)) * np.arctan(0.5 * np.pi * b * x),
    "sigmoid": lambda x, b: (4.0 / b) / (1.0 + np.exp(-b * x)),
}


class TestTheForwardPassIsExact:
    def test_output_is_binary(self):
        u = Tensor(np.random.default_rng(0).normal(size=200) * 3)
        values = set(np.unique(spike(u).data).tolist())
        assert values <= {0.0, 1.0}

    def test_it_is_the_heaviside_step(self):
        u = Tensor(np.array([-1.0, 0.0, 0.99, 1.0, 1.01, 50.0]))
        assert spike(u, threshold=1.0).data.tolist() == [0, 0, 0, 0, 1, 1]

    def test_threshold_is_strict(self):
        """Exactly at threshold does not fire; the comparison is ``>``."""
        assert spike(Tensor(np.array([1.0])), threshold=1.0).data.tolist() == [0.0]


class TestTheDeadNeuronProblem:
    """The problem surrogate gradients exist to solve, demonstrated."""

    def test_the_true_derivative_is_zero_everywhere(self):
        """Finite differences on the real spike function return exactly zero.

        This is why a spiking network cannot be trained by ordinary
        backpropagation, and why the surrogate is not an optimisation but a
        precondition for training at all.
        """
        eps = 1e-5
        for x in (-2.0, -0.5, 0.5, 2.0, 10.0):
            high = float((x + eps) > 0.0)
            low = float((x - eps) > 0.0)
            assert (high - low) / (2 * eps) == 0.0

    def test_the_surrogate_gradient_is_not_zero(self):
        """Every unit receives gradient, including ones nowhere near firing."""
        u = Tensor(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
        spike(u, threshold=0.0).sum().backward()
        assert np.all(u.grad > 0.0)

    def test_gradient_is_largest_at_the_threshold(self):
        """Units closest to firing are the ones most worth nudging."""
        u = Tensor(np.array([-3.0, -0.5, 0.0, 0.5, 3.0]))
        spike(u, threshold=0.0).sum().backward()
        assert u.grad.argmax() == 2
        assert u.grad[0] < u.grad[1] < u.grad[2]
        assert u.grad[4] < u.grad[3] < u.grad[2]


class TestSurrogateGradients:
    """Each surrogate must be the derivative of the function it stands for.

    The step function cannot be finite-differenced, but the *surrogate* can.
    Checking it against its own antiderivative catches an algebra error in the
    surrogate, which is the failure mode actually available here.
    """

    @pytest.mark.parametrize("kind", sorted(SURROGATES))
    def test_matches_the_derivative_of_its_antiderivative(self, kind):
        sharpness, eps = 3.0, 1e-6
        x = np.array([-2.0, -0.7, -0.1, 0.0, 0.1, 0.7, 2.0])
        smooth = ANTIDERIVATIVES[kind]

        numeric = (smooth(x + eps, sharpness) - smooth(x - eps, sharpness)) / (2 * eps)
        analytic = surrogate_derivative(x, kind, sharpness)

        assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("kind", sorted(SURROGATES))
    def test_peaks_at_one_on_the_threshold(self, kind):
        """Normalisation, so sharpness does not secretly rescale the gradient."""
        assert surrogate_derivative(np.array([0.0]), kind, 5.0)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("kind", sorted(SURROGATES))
    def test_is_positive_and_decreasing_away_from_threshold(self, kind):
        x = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
        g = surrogate_derivative(x, kind, 3.0)
        assert np.all(g > 0.0)
        assert np.all(np.diff(g) < 0.0)

    def test_sharpness_narrows_the_window(self):
        far = np.array([1.0])
        wide = surrogate_derivative(far, "fast_sigmoid", 1.0)[0]
        narrow = surrogate_derivative(far, "fast_sigmoid", 20.0)[0]
        assert narrow < wide

    def test_the_spike_node_applies_the_chosen_surrogate(self):
        """The node's backward must be exactly ``surrogate_derivative``."""
        for kind in sorted(SURROGATES):
            u = Tensor(np.array([-1.0, -0.2, 0.3, 2.0]))
            spike(u, threshold=0.5, surrogate=kind, sharpness=4.0).sum().backward()
            expected = surrogate_derivative(u.data - 0.5, kind, 4.0)
            assert np.allclose(u.grad, expected)

    def test_upstream_gradient_is_chained_not_replaced(self):
        u = Tensor(np.array([0.1, 0.1]))
        (spike(u, threshold=0.0) * Tensor(np.array([3.0, -2.0]))).sum().backward()
        base = surrogate_derivative(np.array([0.1, 0.1]), "fast_sigmoid", 5.0)
        assert np.allclose(u.grad, base * np.array([3.0, -2.0]))

    def test_an_unknown_surrogate_is_rejected(self):
        with pytest.raises(ValueError, match="unknown surrogate"):
            surrogate_derivative(np.array([0.0]), "triangular")


class TestLIFDynamics:
    def _layer(self, **kw):
        kw.setdefault("decay", 0.9)
        return LIF(2, 2, rng=np.random.default_rng(0), **kw)

    def test_potential_leaks_without_input(self):
        layer = self._layer()
        layer.synapse.W.data = np.zeros((2, 2))
        layer.synapse.b.data = np.zeros(2)

        potential = Tensor(np.array([[1.0, 0.5]]))
        _, after = layer.step(Tensor(np.zeros((1, 2))), potential)
        assert np.allclose(after.data, [[0.9, 0.45]])

    def test_subthreshold_input_integrates_over_time(self):
        """Repeated small inputs accumulate until they cross threshold."""
        layer = self._layer(decay=1.0, threshold=1.0)
        layer.synapse.W.data = np.eye(2) * 0.3
        layer.synapse.b.data = np.zeros(2)

        potential = layer.initial_state(1)
        x = Tensor(np.ones((1, 2)))
        fired = []
        for _ in range(5):
            spikes, potential = layer.step(x, potential)
            fired.append(spikes.data[0, 0])
        # 0.3 per step: crosses 1.0 on the fourth.
        assert fired == [0.0, 0.0, 0.0, 1.0, 0.0]

    def test_firing_resets_by_subtracting_the_threshold(self):
        layer = self._layer(decay=1.0, threshold=1.0)
        layer.synapse.W.data = np.zeros((2, 2))
        layer.synapse.b.data = np.zeros(2)

        spikes, after = layer.step(Tensor(np.zeros((1, 2))), Tensor(np.array([[1.5, 0.2]])))
        assert spikes.data.tolist() == [[1.0, 0.0]]
        # 1.5 fired and dropped by 1.0, keeping the 0.5 remainder. 0.2 did not.
        assert np.allclose(after.data, [[0.5, 0.2]])

    def test_decay_outside_the_unit_interval_is_rejected(self):
        for bad in (-0.1, 1.5, 2.0):
            with pytest.raises(ValueError, match="decay"):
                LIF(2, 2, decay=bad)

    def test_decay_of_one_is_allowed(self):
        """Non-leaky integrate-and-fire is a real model, not a bad setting."""
        assert LIF(2, 2, decay=1.0).decay == 1.0


class TestBackpropThroughTime:
    def test_gradient_reaches_back_across_timesteps(self):
        """A loss at the last timestep must move weights used at the first.

        This is the whole content of BPTT here: the recurrence was written as
        a Python loop, so the unrolled graph is an ordinary graph and
        ``backward()`` walks it without knowing about time.
        """
        layer = LIF(3, 3, decay=0.9, rng=np.random.default_rng(0))
        x = Tensor(np.ones((1, 3)))

        potential = layer.initial_state(1)
        for _ in range(6):
            _, potential = layer.step(x, potential)

        layer.zero_grad()
        potential.sum().backward()
        assert np.abs(layer.synapse.W.grad).max() > 0.0

    def test_more_timesteps_accumulate_more_gradient(self):
        """Evidence the earlier steps really are in the graph, not dropped."""
        def run(steps):
            layer = LIF(3, 3, decay=0.9, rng=np.random.default_rng(0))
            potential = layer.initial_state(1)
            for _ in range(steps):
                _, potential = layer.step(Tensor(np.ones((1, 3))), potential)
            layer.zero_grad()
            potential.sum().backward()
            return np.abs(layer.synapse.W.grad).sum()

        assert run(6) > run(1)


class TestSpikingMLP:
    def test_parameter_count_matches_the_dense_network(self):
        """The A/B needs the two arms to differ in behaviour, not in size."""
        snn = SpikingMLP(784, [128], 10, timesteps=5, seed=0)
        ann = TensorMLP(784, [128], 10, seed=0)
        assert snn.num_parameters() == ann.num_parameters() == 101_770

    def test_output_shape(self):
        net = SpikingMLP(8, [16], 4, timesteps=5, seed=0)
        assert net(np.zeros((3, 8))).shape == (3, 4)

    def test_stats_report_real_spikes(self):
        net = SpikingMLP(20, [30], 4, timesteps=10, seed=0)
        x = np.random.default_rng(0).random((5, 20))
        _, stats = net(x, collect_stats=True)
        assert stats["spike_count"] > 0
        assert 0.0 < stats["spike_rate"] <= 1.0
        assert stats["spikes_per_sample"] == pytest.approx(stats["spike_count"] / 5)

    def test_zero_input_produces_no_spikes(self):
        """Weights are zero-biased, so nothing fires without input."""
        net = SpikingMLP(8, [16], 4, timesteps=5, seed=0)
        _, stats = net(np.zeros((2, 8)), collect_stats=True)
        assert stats["spike_count"] == 0.0

    def test_every_parameter_receives_gradient(self):
        """The end-to-end proof that surrogate gradients reach the input layer."""
        net = SpikingMLP(12, [16], 3, timesteps=8, seed=0)
        x = np.random.default_rng(0).random((4, 12))
        loss = tensor_cross_entropy(net(x), np.array([0, 1, 2, 0]))
        net.zero_grad()
        loss.backward()
        for name, parameter in net.named_parameters():
            assert np.abs(parameter.grad).max() > 0.0, f"{name} received no gradient"

    def test_it_can_learn_a_small_problem(self):
        """Loss must fall well below chance, or the gradients are decorative."""
        from nabla.optim import Adam

        rng = np.random.default_rng(0)
        x = rng.random((24, 10))
        y = (x[:, 0] > 0.5).astype(int)

        net = SpikingMLP(10, [24], 2, timesteps=10, seed=0)
        optimiser = Adam(net.parameters(), lr=5e-3)
        first = last = None
        for step in range(60):
            loss = tensor_cross_entropy(net(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step == 0:
                first = loss.item()
            last = loss.item()

        assert first > 0.5
        assert last < first * 0.6

    def test_timesteps_must_be_positive(self):
        with pytest.raises(ValueError, match="timesteps"):
            SpikingMLP(4, [4], 2, timesteps=0)

    def test_logit_scale_is_independent_of_timesteps(self):
        """Averaging over time, so changing T does not rescale the loss."""
        short = SpikingMLP(8, [16], 4, timesteps=5, seed=0)
        long = SpikingMLP(8, [16], 4, timesteps=40, seed=0)
        x = np.random.default_rng(0).random((4, 8))
        ratio = np.abs(long(x).data).mean() / np.abs(short(x).data).mean()
        assert 0.2 < ratio < 5.0


class TestRateEncoding:
    def test_shape_and_binary_output(self):
        x = np.random.default_rng(0).random((4, 6))
        out = rate_encode(x, 20, rng=np.random.default_rng(0))
        assert out.shape == (20, 4, 6)
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}

    def test_firing_rate_tracks_intensity(self):
        x = np.array([[0.0, 0.25, 0.75, 1.0]])
        out = rate_encode(x, 4000, rng=np.random.default_rng(0))
        assert np.allclose(out.mean(axis=0)[0], [0.0, 0.25, 0.75, 1.0], atol=0.02)
