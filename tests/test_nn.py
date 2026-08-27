"""Phase 16 -- the neural-network layer.

The acceptance test for Phase 7 is at the bottom: gradient-check a complete MLP
against finite differences. If that passes, every layer, activation and
parameter is wired correctly, because a single wrong sign anywhere would show
up as a disagreement.
"""

from __future__ import annotations

import math
import random

import pytest

from nabla import Value
from nabla.core.gradcheck import check_gradients, check_parameter_gradients
from nabla.nn import (
    MLP,
    Layer,
    Module,
    Neuron,
    Parameter,
    get_activation,
    he_normal,
    leaky_relu,
    softplus,
    xavier_uniform,
)


@pytest.fixture
def rng():
    return random.Random(1234)


def pre_activations(model: MLP, xs: list[float]) -> list[float]:
    """Every layer's pre-activation values, computed independently of forward().

    Used to check that a gradient-check test point is not sitting on a ReLU
    kink -- see ``TestReLUKinkLimitation``.
    """
    zs: list[float] = []
    activations = list(xs)
    for layer in model.layers:
        z_layer = [
            n.bias.data + sum(w.data * a for w, a in zip(n.weights, activations))
            for n in layer.neurons
        ]
        zs.extend(z_layer)
        if layer.activation_name == "relu":
            activations = [max(0.0, z) for z in z_layer]
        elif layer.activation_name == "tanh":
            activations = [math.tanh(z) for z in z_layer]
        elif layer.activation_name == "sigmoid":
            activations = [1.0 / (1.0 + math.exp(-z)) for z in z_layer]
        elif layer.activation_name == "leaky_relu":
            activations = [z if z > 0 else 0.01 * z for z in z_layer]
        else:
            activations = z_layer
    return zs


def assert_away_from_relu_kink(model: MLP, xs: list[float], margin: float = 1e-3) -> None:
    """Fail loudly if a test point sits on a non-differentiable kink.

    Without this guard, a future change to initialisation could silently move a
    pre-activation onto 0.0 and turn a gradient-check test flaky for a reason
    that has nothing to do with a wrong derivative.
    """
    if model.activation_name not in ("relu", "leaky_relu"):
        return
    closest = min(abs(z) for z in pre_activations(model, xs))
    assert closest > margin, (
        f"test point sits {closest:.2e} from a ReLU kink, where finite "
        "differences are not a valid check. Choose a different seed or input."
    )


# ======================================================================
# Parameter and Module
# ======================================================================


class TestParameter:
    def test_is_a_value(self):
        p = Parameter(0.5)
        assert isinstance(p, Value)
        assert p.data == 0.5
        assert p.is_leaf

    def test_participates_in_the_graph(self):
        p = Parameter(3.0)
        (p * 2).backward()
        assert p.grad == 2.0

    def test_repr_says_parameter(self):
        assert "Parameter" in repr(Parameter(1.0, label="w"))


class TestModuleRegistration:
    def test_registers_a_direct_parameter(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(1.0)

        assert len(M().parameters()) == 1

    def test_registers_parameters_in_a_list(self):
        # The case PyTorch requires nn.ParameterList for. Layer relies on it.
        class M(Module):
            def __init__(self):
                super().__init__()
                self.ws = [Parameter(float(i)) for i in range(5)]

        assert len(M().parameters()) == 5

    def test_registers_nested_modules(self):
        class Inner(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(1.0)

        class Outer(Module):
            def __init__(self):
                super().__init__()
                self.a = Inner()
                self.b = Inner()

        assert len(Outer().parameters()) == 2

    def test_ignores_plain_attributes(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.name = "not a parameter"
                self.size = 42
                self.values = [Value(1.0), Value(2.0)]  # Values, not Parameters

        assert M().parameters() == []

    def test_names_are_qualified_and_stable(self):
        model = MLP(2, [3], 1, seed=0)
        names = [n for n, _ in model.named_parameters()]
        assert len(names) == len(set(names))
        assert names == [n for n, _ in model.named_parameters()]  # deterministic
        assert any("layers.0" in n for n in names)

    def test_reassignment_does_not_leave_a_stale_registration(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.ws = [Parameter(1.0), Parameter(2.0), Parameter(3.0)]

        m = M()
        assert len(m.parameters()) == 3
        m.ws = [Parameter(9.0)]
        assert len(m.parameters()) == 1

    def test_assignment_before_super_init_raises(self):
        class Broken(Module):
            def __init__(self):
                self.w = Parameter(1.0)  # forgot super().__init__()

        with pytest.raises(RuntimeError, match="super"):
            Broken()

    def test_zero_grad_clears_parameters(self):
        model = MLP(2, [3], 1, seed=0)
        model([1.0, 2.0])[0].backward()
        assert any(p.grad != 0 for p in model.parameters())
        model.zero_grad()
        assert all(p.grad == 0 for p in model.parameters())

    def test_train_eval_propagates(self):
        model = MLP(2, [3], 1, seed=0)
        model.eval()
        assert all(not m.training for m in model.modules())
        model.train()
        assert all(m.training for m in model.modules())

    def test_forward_must_be_implemented(self):
        class M(Module):
            pass

        with pytest.raises(NotImplementedError):
            M()()


class TestStateDict:
    def test_round_trip(self):
        a = MLP(3, [4], 2, seed=0)
        b = MLP(3, [4], 2, seed=99)
        assert a.predict([1.0, 2.0, 3.0]) != b.predict([1.0, 2.0, 3.0])

        b.load_state_dict(a.state_dict())
        assert a.predict([1.0, 2.0, 3.0]) == b.predict([1.0, 2.0, 3.0])

    def test_strict_rejects_a_mismatched_checkpoint(self):
        a = MLP(3, [4], 2, seed=0)
        b = MLP(3, [8], 2, seed=0)  # different architecture
        with pytest.raises(KeyError, match="mismatch"):
            b.load_state_dict(a.state_dict())

    def test_non_strict_loads_what_it_can(self):
        a = MLP(3, [4], 2, seed=0)
        b = MLP(3, [8], 2, seed=0)
        b.load_state_dict(a.state_dict(), strict=False)  # no exception


# ======================================================================
# Neuron
# ======================================================================


class TestNeuron:
    def test_parameter_count(self, rng):
        n = Neuron(5, rng=rng)
        assert len(n.parameters()) == 6  # 5 weights + 1 bias

    def test_bias_starts_at_zero(self, rng):
        # No symmetry problem for biases, and no reason to prefer a shift.
        assert Neuron(4, rng=rng).bias.data == 0.0

    def test_weights_are_not_all_equal(self, rng):
        # Identical weights would make every neuron in a layer compute the
        # same thing forever -- the symmetry problem.
        weights = [w.data for w in Neuron(10, rng=rng).weights]
        assert len(set(weights)) > 1

    def test_computes_the_weighted_sum_plus_bias(self, rng):
        n = Neuron(3, activation="linear", rng=rng)
        for i, w in enumerate(n.weights):
            w.data = float(i + 1)  # 1, 2, 3
        n.bias.data = 0.5
        # 1*1 + 2*2 + 3*3 + 0.5
        assert n([1.0, 2.0, 3.0]).data == pytest.approx(14.5)

    def test_applies_the_activation(self, rng):
        n = Neuron(2, activation="tanh", rng=rng)
        for w in n.weights:
            w.data = 1.0
        n.bias.data = 0.0
        assert n([0.5, 0.5]).data == pytest.approx(math.tanh(1.0))

    def test_rejects_wrong_input_length(self, rng):
        with pytest.raises(ValueError, match="expected 3 inputs"):
            Neuron(3, rng=rng)([1.0, 2.0])

    def test_rejects_zero_inputs(self, rng):
        with pytest.raises(ValueError, match=">= 1"):
            Neuron(0, rng=rng)

    def test_gradients_match_the_analytic_rule(self, rng):
        # ∂a/∂wᵢ = σ'(z)·xᵢ  and  ∂a/∂b = σ'(z)
        n = Neuron(3, activation="tanh", rng=rng)
        xs = [0.5, -1.0, 2.0]
        out = n(xs)
        out.backward()

        z = sum(w.data * x for w, x in zip(n.weights, xs)) + n.bias.data
        dsigma = 1.0 - math.tanh(z) ** 2
        assert n.bias.grad == pytest.approx(dsigma)
        for w, x in zip(n.weights, xs):
            assert w.grad == pytest.approx(dsigma * x)

    def test_a_zero_input_gives_its_weight_no_gradient(self, rng):
        # Why MNIST border-pixel weights barely move.
        n = Neuron(2, activation="linear", rng=rng)
        n([0.0, 1.0]).backward()
        assert n.weights[0].grad == 0.0
        assert n.weights[1].grad != 0.0

    def test_a_saturated_neuron_learns_nothing(self, rng):
        n = Neuron(1, activation="tanh", rng=rng)
        n.weights[0].data = 50.0
        n([1.0]).backward()
        assert abs(n.weights[0].grad) < 1e-40


# ======================================================================
# Layer
# ======================================================================


class TestLayer:
    def test_output_count_and_parameter_count(self, rng):
        layer = Layer(3, 4, rng=rng)
        assert len(layer([1.0, 2.0, 3.0])) == 4
        assert len(layer.parameters()) == 3 * 4 + 4

    def test_neurons_are_independent(self, rng):
        # Different random weights -> different outputs for the same input.
        outs = [v.data for v in Layer(4, 5, rng=rng)([1.0, 1.0, 1.0, 1.0])]
        assert len(set(outs)) > 1

    def test_rejects_wrong_input_length(self, rng):
        with pytest.raises(ValueError, match="expected 3 inputs"):
            Layer(3, 2, rng=rng)([1.0])

    def test_rejects_invalid_sizes(self, rng):
        with pytest.raises(ValueError, match=">= 1"):
            Layer(0, 3, rng=rng)

    def test_weight_matrix_shape(self, rng):
        layer = Layer(3, 4, rng=rng)
        matrix = layer.weight_matrix()
        assert len(matrix) == 4 and all(len(row) == 3 for row in matrix)
        assert len(layer.bias_vector()) == 4

    def test_he_init_scales_with_fan_in(self):
        # std ≈ sqrt(2/fan_in): a 400-input layer should have roughly half the
        # spread of a 100-input one.
        narrow = Layer(100, 50, activation="relu", rng=random.Random(0))
        wide = Layer(400, 50, activation="relu", rng=random.Random(0))

        def std(layer):
            ws = [w.data for n in layer.neurons for w in n.weights]
            mean = sum(ws) / len(ws)
            return (sum((w - mean) ** 2 for w in ws) / len(ws)) ** 0.5

        assert std(narrow) / std(wide) == pytest.approx(2.0, rel=0.15)

    def test_xavier_uses_both_fans(self):
        # limit = sqrt(6/(fan_in+fan_out)) -- depends on n_out, unlike He.
        a = Layer(100, 10, activation="tanh", rng=random.Random(0))
        b = Layer(100, 1000, activation="tanh", rng=random.Random(0))
        spread = lambda l: max(abs(w.data) for n in l.neurons for w in n.weights)  # noqa: E731
        assert spread(a) > spread(b)


# ======================================================================
# MLP
# ======================================================================


class TestMLP:
    def test_architecture_is_configurable(self):
        assert MLP(4, [8, 6], 3, seed=0).sizes == [4, 8, 6, 3]

    def test_mnist_parameter_count(self):
        # 784*128 + 128 + 128*64 + 64 + 64*10 + 10
        assert MLP(784, [128, 64], 10, seed=0).num_parameters() == 109_386

    def test_output_size(self):
        assert len(MLP(3, [5], 7, seed=0)([1.0, 2.0, 3.0])) == 7

    def test_no_hidden_layers(self):
        model = MLP(3, [], 2, seed=0)
        assert len(model.layers) == 1
        assert len(model([1.0, 2.0, 3.0])) == 2

    def test_output_activation_is_linear_by_default(self):
        # Classifiers emit logits; softmax lives in the loss.
        assert MLP(2, [4], 3, seed=0).layers[-1].activation_name == "linear"

    def test_logits_can_be_negative(self):
        model = MLP(2, [8, 8], 5, activation="tanh", seed=3)
        outs = [v.data for v in model([2.0, -3.0])]
        assert any(o < 0 for o in outs)

    def test_seed_is_reproducible(self):
        a = MLP(3, [4], 2, seed=42).predict([1.0, 2.0, 3.0])
        b = MLP(3, [4], 2, seed=42).predict([1.0, 2.0, 3.0])
        c = MLP(3, [4], 2, seed=43).predict([1.0, 2.0, 3.0])
        assert a == b
        assert a != c

    def test_summary_mentions_every_layer(self):
        text = MLP(784, [128, 64], 10, seed=0).summary()
        assert "784" in text and "128" in text and "109,386" in text

    def test_accepts_value_inputs_for_input_gradients(self):
        # Needed for saliency maps / adversarial examples.
        model = MLP(2, [4], 1, seed=0)
        x = Value(0.5)
        model([x, Value(-0.3)])[0].backward()
        assert x.grad != 0.0

    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "leaky_relu"])
    def test_every_activation_produces_gradients(self, activation):
        model = MLP(3, [5, 4], 2, activation=activation, seed=0)
        out = model([0.5, -0.5, 1.0])
        (out[0] + out[1]).backward()
        # At least some parameters must have moved; relu can legitimately
        # zero many of them.
        assert any(p.grad != 0 for p in model.parameters())


# ======================================================================
# Activations (Phase 8)
# ======================================================================


class TestActivations:
    def test_registry_resolves_names(self):
        for name in ("linear", "tanh", "relu", "sigmoid", "leaky_relu", "softplus"):
            assert callable(get_activation(name))

    def test_registry_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown activation"):
            get_activation("swish")

    def test_registry_accepts_a_callable(self):
        fn = lambda v: v * 2  # noqa: E731
        assert get_activation(fn) is fn

    @pytest.mark.parametrize("x", [-3.0, -0.5, 0.5, 3.0])
    def test_leaky_relu_forward(self, x):
        expected = x if x > 0 else 0.01 * x
        assert leaky_relu(Value(x)).data == pytest.approx(expected)

    def test_leaky_relu_keeps_negative_gradient_alive(self):
        # The whole point: a dead ReLU gets 0, a leaky one gets the slope.
        x = Value(-5.0)
        leaky_relu(x).backward()
        assert x.grad == pytest.approx(0.01)

        y = Value(-5.0)
        y.relu().backward()
        assert y.grad == 0.0

    @pytest.mark.parametrize("x", [-4.0, -0.5, 0.5, 4.0])
    def test_leaky_relu_gradient_is_correct_though_composed(self, x):
        # We wrote no backward rule for leaky_relu -- the chain rule made it.
        assert check_gradients(lambda v: leaky_relu(v), [x]).passed

    @pytest.mark.parametrize("x", [-30.0, -1.0, 0.0, 1.0, 30.0])
    def test_softplus_forward(self, x):
        assert softplus(Value(x)).data == pytest.approx(math.log1p(math.exp(-abs(x))) + max(x, 0.0))

    def test_softplus_is_stable_at_extremes(self):
        # The naive log(1+exp(x)) overflows at 800.
        assert softplus(Value(800.0)).data == pytest.approx(800.0)
        assert softplus(Value(-800.0)).data == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("x", [-5.0, -0.5, 0.5, 5.0])
    def test_softplus_derivative_is_the_sigmoid(self, x):
        v = Value(x)
        softplus(v).backward()
        assert v.grad == pytest.approx(1.0 / (1.0 + math.exp(-x)))

    def test_relu_derivative_beats_sigmoid_for_depth(self):
        """Measured, not asserted: stack 10 layers and compare attenuation.

        Sigmoid's peak derivative is 0.25, so ten layers can attenuate by
        0.25^10 ≈ 1e-6 at best. ReLU's is exactly 1, so it does not attenuate.
        """
        def chain(activation, depth=10):
            x = Value(0.5)
            v = x
            for _ in range(depth):
                v = activation(v)
            v.backward()
            return abs(x.grad)

        from nabla.nn import relu, sigmoid, tanh

        assert chain(relu) == pytest.approx(1.0)
        assert chain(sigmoid) < 1e-6
        assert chain(tanh) < chain(relu)


class TestInitializers:
    def test_he_normal_variance(self):
        rng = random.Random(0)
        samples = [he_normal(100, 50, rng) for _ in range(20000)]
        variance = sum(s * s for s in samples) / len(samples)
        assert variance == pytest.approx(2.0 / 100, rel=0.05)

    def test_xavier_uniform_range(self):
        rng = random.Random(0)
        limit = math.sqrt(6.0 / (100 + 50))
        samples = [xavier_uniform(100, 50, rng) for _ in range(5000)]
        assert max(samples) <= limit and min(samples) >= -limit
        assert max(samples) == pytest.approx(limit, rel=0.02)

    def test_auto_picks_by_activation(self):
        from nabla.nn import for_activation, he_normal as he, xavier_uniform as xu

        assert for_activation("relu") is he
        assert for_activation("tanh") is xu

    def test_unknown_initializer_raises(self):
        from nabla.nn import get_initializer

        with pytest.raises(ValueError, match="unknown initializer"):
            get_initializer("magic")


# ======================================================================
# THE ACCEPTANCE TEST FOR PHASE 7
# ======================================================================


class TestGradientsThroughWholeNetworks:
    """Finite-difference verification through complete networks.

    This is the test that matters. It composes every layer, activation and the
    loss, then checks each parameter's gradient against a method that shares no
    code with the backward rules. A single wrong sign anywhere fails it.
    """

    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "leaky_relu"])
    def test_mlp_with_mse(self, activation):
        from nabla.losses import mse_loss

        # seed 0, not 5: see TestReLUKinkLimitation below for why seed 5 puts
        # a pre-activation at exactly 0.0, where finite differences legitimately
        # disagree with the subgradient we chose.
        model = MLP(3, [4, 3], 2, activation=activation, seed=0)
        xs = [0.6, -0.4, 1.2]
        assert_away_from_relu_kink(model, xs)

        result = check_parameter_gradients(
            lambda: mse_loss(model(xs), [0.25, -0.5]),
            model.parameters(),
            order=4,  # tighter truncation error; some gradients here are ~1e-7
            label=f"MLP 3-4-3-2 {activation} + MSE",
        )
        assert result.passed, "\n" + str(result)

    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    def test_mlp_with_cross_entropy(self, activation):
        from nabla.losses import softmax_cross_entropy

        model = MLP(4, [5, 4], 3, activation=activation, seed=7)
        batch = [[0.5, -1.0, 0.3, 0.8], [-0.2, 0.9, -1.1, 0.4]]
        targets = [2, 0]

        result = check_parameter_gradients(
            lambda: softmax_cross_entropy([model(x) for x in batch], targets),
            model.parameters(),
            label=f"MLP 4-5-4-3 {activation} + cross-entropy",
        )
        assert result.passed, "\n" + str(result)

    def test_deep_network(self):
        """Six layers. Gradients must survive the whole chain."""
        from nabla.losses import mse_loss

        model = MLP(2, [3, 3, 3, 3, 3], 1, activation="tanh", seed=11)
        result = check_parameter_gradients(
            lambda: mse_loss(model([0.3, -0.7]), [0.5]),
            model.parameters(),
            tol=1e-4,  # deep composition accumulates round-off
            label="6-layer MLP",
        )
        assert result.passed, "\n" + str(result)

    def test_every_parameter_receives_a_gradient(self):
        from nabla.losses import softmax_cross_entropy

        model = MLP(4, [6, 5], 3, activation="tanh", seed=2)
        model.zero_grad()
        softmax_cross_entropy([model([0.5, -1.0, 0.3, 0.8])], [1]).backward()

        dead = [n for n, p in model.named_parameters() if p.grad == 0.0]
        assert not dead, f"{len(dead)} parameters got no gradient: {dead[:5]}"

    def test_gradient_stats_report_health(self):
        from nabla.losses import mse_loss

        model = MLP(3, [4], 2, activation="tanh", seed=0)
        model.zero_grad()
        mse_loss(model([1.0, 2.0, 3.0]), [0.0, 0.0]).backward()

        stats = model.gradient_stats()
        assert stats["count"] == model.num_parameters()
        assert stats["mean"] > 0
        assert stats["max"] >= stats["min"]


# ======================================================================
# A real limitation, documented rather than hidden
# ======================================================================


class TestReLUKinkLimitation:
    r"""Where gradient checking legitimately fails -- and why that is fine.

    ReLU has no derivative at exactly :math:`x=0`: the left slope is 0, the
    right slope is 1, and no single tangent exists. We choose the subgradient
    **0**, matching PyTorch.

    A central finite difference cannot see that choice. It evaluates at
    :math:`\pm h` either side of the kink and reports the *average* slope, 0.5.
    So the two methods disagree by 100% -- and both are defensible.

    This is a property of the function, not a bug in the engine, and it is the
    one case where a failed gradient check should be believed rather than
    acted on. It is pinned here so the behaviour is a documented decision
    instead of a mystery someone rediscovers later.
    """

    def test_finite_differences_disagree_at_the_kink(self):
        x = Value(0.0)
        x.relu().backward()
        assert x.grad == 0.0  # our subgradient choice

        result = check_gradients(lambda v: v.relu(), [0.0])
        assert not result.passed
        assert result.analytic[0] == 0.0
        assert result.numerical[0] == pytest.approx(0.5)  # the average slope

    def test_they_agree_everywhere_else(self):
        for x in (-1.0, -0.01, 0.01, 1.0):
            assert check_gradients(lambda v: v.relu(), [x]).passed

    def test_seed_5_puts_a_whole_layer_on_the_kink(self):
        """The concrete case this project hit, preserved as a regression test.

        With seed 5 every first-layer ReLU is dead for this input, so the
        layer outputs all zeros. The second layer's pre-activation is then
        ``bias + 0`` -- and biases initialise to exactly 0.0, landing its ReLU
        precisely on the kink.

        Two lessons in one: gradient checks need smooth points, and a whole
        dead layer is a real failure mode (see ``test_dying_relu`` below).
        """
        model = MLP(3, [4, 3], 2, activation="relu", seed=5)
        xs = [0.6, -0.4, 1.2]

        zs = pre_activations(model, xs)
        assert min(abs(z) for z in zs) == 0.0

        with pytest.raises(AssertionError, match="ReLU kink"):
            assert_away_from_relu_kink(model, xs)

    def test_dying_relu_is_observable(self):
        """A dead unit receives exactly zero gradient and can never recover.

        This is not a bug -- it is the documented weakness of ReLU, and the
        reason leaky ReLU exists.
        """
        from nabla.losses import mse_loss
        from nabla.nn import Neuron

        n = Neuron(2, activation="relu", rng=random.Random(0))
        n.weights[0].data = -5.0
        n.weights[1].data = -5.0
        n.bias.data = -1.0

        n.zero_grad()
        mse_loss([n([1.0, 1.0])], [1.0]).backward()
        assert all(p.grad == 0.0 for p in n.parameters())

        # Leaky ReLU keeps a path open.
        n2 = Neuron(2, activation="leaky_relu", rng=random.Random(0))
        n2.weights[0].data = -5.0
        n2.weights[1].data = -5.0
        n2.bias.data = -1.0
        n2.zero_grad()
        mse_loss([n2([1.0, 1.0])], [1.0]).backward()
        assert all(p.grad != 0.0 for p in n2.parameters())
