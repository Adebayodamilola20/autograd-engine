"""Phase 5 -- every gradient, verified against finite differences.

``test_backward.py`` checks gradients against values derived by hand. This file
checks them against a method that shares **no code and no reasoning** with the
backward rules: the definition of the derivative as a limit, evaluated with the
forward pass only.

Two independent methods agreeing to ten digits is what makes the engine
trustworthy. One of them agreeing with itself is not.

The suite also verifies the *checker*, in ``TestTheCheckerCanFail`` -- a
verification tool that cannot detect a wrong derivative proves nothing.
"""

from __future__ import annotations

import math
import random

import pytest

from nabla import Value
from nabla.core.gradcheck import (
    check_gradients,
    check_parameter_gradients,
    epsilon_sweep,
    numerical_gradient,
    relative_error,
)


def assert_grads_agree(fn, inputs, *, tol=1e-6, order=2, label=""):
    """Run a gradient check and fail with the full comparison table."""
    result = check_gradients(fn, inputs, tol=tol, order=order, label=label)
    assert result.passed, "\n" + str(result)
    return result


# ======================================================================
# Every operation, at several points
# ======================================================================


class TestArithmeticOperations:
    POINTS = [(2.0, 3.0), (-1.5, 4.0), (0.5, -0.25), (7.0, -3.5), (0.1, 0.9)]

    @pytest.mark.parametrize("a, b", POINTS)
    def test_add(self, a, b):
        assert_grads_agree(lambda x, y: x + y, [a, b], label="x + y")

    @pytest.mark.parametrize("a, b", POINTS)
    def test_mul(self, a, b):
        assert_grads_agree(lambda x, y: x * y, [a, b], label="x * y")

    @pytest.mark.parametrize("a, b", POINTS)
    def test_sub(self, a, b):
        assert_grads_agree(lambda x, y: x - y, [a, b], label="x - y")

    @pytest.mark.parametrize("a, b", POINTS)
    def test_div(self, a, b):
        assert_grads_agree(lambda x, y: x / y, [a, b], label="x / y")

    @pytest.mark.parametrize("a", [-3.0, -0.5, 0.5, 2.0, 10.0])
    def test_neg(self, a):
        assert_grads_agree(lambda x: -x, [a], label="-x")

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("n", [2, 3, -1, -2, 0.5, 1.5])
    def test_pow_constant_exponent(self, a, n):
        assert_grads_agree(lambda x: x**n, [a], label=f"x ** {n}")

    @pytest.mark.parametrize("a", [-3.0, -1.0, 2.0])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_pow_negative_base_integer_exponent(self, a, n):
        assert_grads_agree(lambda x: x**n, [a], label=f"x ** {n}")

    @pytest.mark.parametrize("a, b", [(2.0, 3.0), (0.5, 2.0), (3.0, -1.5), (1.2, 0.3)])
    def test_pow_both_values(self, a, b):
        """The two-term rule: b·a^(b-1) and a^b·ln a."""
        assert_grads_agree(lambda x, y: x**y, [a, b], label="x ** y")

    @pytest.mark.parametrize("base", [2.0, 0.5, math.e, 10.0])
    def test_rpow(self, base):
        assert_grads_agree(lambda x: base**x, [1.3], label=f"{base:g} ** x")


class TestTranscendental:
    @pytest.mark.parametrize("a", [-3.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    def test_exp(self, a):
        assert_grads_agree(lambda x: x.exp(), [a], label="exp(x)")

    @pytest.mark.parametrize("a", [0.01, 0.5, 1.0, 2.0, 50.0])
    def test_log(self, a):
        assert_grads_agree(lambda x: x.log(), [a], label="log(x)")

    def test_log_near_zero_needs_a_smaller_step(self):
        """At x=1e-3 the derivative is 1000 and the curvature is enormous.

        The default h=1e-5 is too coarse relative to the scale of the function
        here; a smaller step recovers the accuracy. This is the truncation
        term h²f'''/6 becoming visible, not an engine bug -- and it is worth
        seeing once, because the same effect appears in cross-entropy when a
        predicted probability is tiny.
        """
        loose = check_gradients(lambda x: x.log(), [1e-3], eps=1e-5)
        tight = check_gradients(lambda x: x.log(), [1e-3], eps=1e-9)
        assert tight.max_error < loose.max_error


class TestActivations:
    @pytest.mark.parametrize("a", [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])
    def test_tanh(self, a):
        assert_grads_agree(lambda x: x.tanh(), [a], label="tanh(x)")

    @pytest.mark.parametrize("a", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_sigmoid(self, a):
        assert_grads_agree(lambda x: x.sigmoid(), [a], label="sigmoid(x)")

    @pytest.mark.parametrize("a", [-5.0, -1.0, -0.01, 0.01, 1.0, 5.0])
    def test_relu_away_from_the_kink(self, a):
        assert_grads_agree(lambda x: x.relu(), [a], label="relu(x)")

    def test_saturated_tanh_still_agrees(self):
        # Gradient ~1e-7 here. Relative comparison keeps the standard honest
        # even when the absolute magnitude is tiny.
        assert_grads_agree(lambda x: x.tanh(), [4.0], tol=1e-4, order=4)


# ======================================================================
# Composite expressions -- the chain rule composing itself
# ======================================================================


class TestCompositeExpressions:
    EXPRESSIONS = [
        ("x*y + b", lambda x, y, b: x * y + b, [2.0, -3.0, 1.0]),
        ("(x*y + b)²", lambda x, y, b: (x * y + b) ** 2, [2.0, -3.0, 1.0]),
        ("tanh(x*w + b)", lambda x, w, b: (x * w + b).tanh(), [1.5, 2.0, -0.5]),
        ("sigmoid(x*w + b)", lambda x, w, b: (x * w + b).sigmoid(), [1.5, 2.0, -0.5]),
        ("relu(x*w + b)", lambda x, w, b: (x * w + b).relu(), [1.5, 2.0, -0.5]),
        ("exp(x*y)", lambda x, y: (x * y).exp(), [0.7, 1.3]),
        ("log(x² + y²)", lambda x, y: (x**2 + y**2).log(), [1.5, 2.5]),
        ("x/(1+exp(-y))", lambda x, y: x / (1 + (-y).exp()), [2.0, 0.5]),
        ("(x+y)·(x-y)", lambda x, y: (x + y) * (x - y), [3.0, 1.0]),
        ("tanh(tanh(tanh(x)))", lambda x: x.tanh().tanh().tanh(), [0.6]),
        ("x·tanh(x)", lambda x: x * x.tanh(), [1.1]),
        ("exp(-x²)", lambda x: (-(x**2)).exp(), [0.8]),
        ("log(exp(x)+exp(y))", lambda x, y: (x.exp() + y.exp()).log(), [1.0, 2.0]),
        ("(x·y·z)²", lambda x, y, z: (x * y * z) ** 2, [1.5, -2.0, 0.5]),
        ("x/y/z", lambda x, y, z: x / y / z, [8.0, 2.0, 2.0]),
    ]

    @pytest.mark.parametrize(
        "label, fn, point", EXPRESSIONS, ids=[e[0] for e in EXPRESSIONS]
    )
    def test_expression(self, label, fn, point):
        assert_grads_agree(fn, point, label=label)


class TestMultiPathExpressions:
    """The cases where accumulation is load-bearing.

    Finite differences do not care how many paths connect an input to the
    output -- they only perturb and re-evaluate. So they are the perfect check
    on whether ``+=`` collected all of them.
    """

    MULTIPATH = [
        ("x·x", lambda x: x * x, [3.0]),
        ("x·x·x", lambda x: x * x * x, [2.0]),
        ("x² + x", lambda x: x**2 + x, [2.5]),
        ("x·tanh(x) + exp(x)", lambda x: x * x.tanh() + x.exp(), [0.9]),
        ("(x+y)·(x+y)", lambda x, y: (x + y) * (x + y), [1.0, 2.0]),
        ("x/x", lambda x: x / x, [4.0]),
        ("x - x", lambda x: x - x, [4.0]),
        ("diamond", lambda x: (2 * x) + (x**3), [2.0]),
        (
            "deep reuse",
            lambda x, y: ((x * y + x) * (x - y) + x.tanh()) ** 2,
            [1.3, -0.7],
        ),
        (
            "shared intermediate",
            lambda x, y: (lambda u: u**2 + 3 * u + u.exp())(x + y),
            [0.5, 0.7],
        ),
    ]

    @pytest.mark.parametrize(
        "label, fn, point", MULTIPATH, ids=[e[0] for e in MULTIPATH]
    )
    def test_expression(self, label, fn, point):
        assert_grads_agree(fn, point, label=label)

    def test_a_value_used_ten_times(self):
        def fn(x):
            total = Value(0.0)
            for k in range(1, 11):
                total = total + x**k
            return total

        # Σ x^k for k=1..10  ->  Σ k·x^(k-1)
        result = assert_grads_agree(fn, [0.9], label="Σ x^k")
        expected = sum(k * 0.9 ** (k - 1) for k in range(1, 11))
        assert result.analytic[0] == pytest.approx(expected)


# ======================================================================
# Randomised expressions -- fuzzing the chain rule
# ======================================================================


def _build_tree(rng: random.Random, depth: int, n_inputs: int):
    """Random smooth expression tree. Deterministic given the seed.

    Only globally-smooth operations are used: a randomly-generated ``relu``
    would sometimes land on its kink, where finite differences legitimately
    disagree, and a flaky test is worse than no test.
    """
    if depth == 0:
        if rng.random() < 0.8:
            return ("var", rng.randrange(n_inputs))
        return ("const", round(rng.uniform(-2.0, 2.0), 3))

    op = rng.choice(["+", "*", "-", "tanh", "sigmoid", "square"])
    if op in ("tanh", "sigmoid", "square"):
        return (op, _build_tree(rng, depth - 1, n_inputs))
    return (op, _build_tree(rng, depth - 1, n_inputs), _build_tree(rng, depth - 1, n_inputs))


def _eval_tree(tree, values):
    kind = tree[0]
    if kind == "var":
        return values[tree[1]]
    if kind == "const":
        return Value(tree[1])
    if kind == "tanh":
        return _eval_tree(tree[1], values).tanh()
    if kind == "sigmoid":
        return _eval_tree(tree[1], values).sigmoid()
    if kind == "square":
        return _eval_tree(tree[1], values) ** 2
    left = _eval_tree(tree[1], values)
    right = _eval_tree(tree[2], values)
    return {"+": left.__add__, "*": left.__mul__, "-": left.__sub__}[kind](right)


class TestRandomExpressions:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_smooth_expression(self, seed):
        """40 randomly-generated expression trees, each gradient-checked.

        Hand-written tests probe the cases the author thought of. This probes
        the ones they did not -- deep nesting, repeated variables in odd
        positions, mixed unary and binary chains.
        """
        rng = random.Random(seed)
        n_inputs = rng.randint(1, 3)
        tree = _build_tree(rng, depth=4, n_inputs=n_inputs)
        point = [round(rng.uniform(-1.5, 1.5), 3) for _ in range(n_inputs)]

        assert_grads_agree(
            lambda *vs: _eval_tree(tree, vs), point, tol=1e-5, label=f"seed {seed}"
        )


# ======================================================================
# The checker itself
# ======================================================================


class TestTheCheckerCanFail:
    """A verification tool that always passes verifies nothing.

    These tests inject deliberately wrong backward rules and confirm the
    checker catches them -- including the subtle failure modes that a weaker
    check would miss.
    """

    @staticmethod
    def _custom_square(x: Value, wrong_factor: float) -> Value:
        """x² with a deliberately mis-derived gradient (should be 2x)."""
        out = Value(x.data**2, (x,), "sq")

        def _backward():
            x.grad += wrong_factor * x.data * out.grad

        out._backward = _backward
        return out

    def test_catches_a_wrong_coefficient(self):
        # Should be 2x; claims x. The classic "forgot the chain rule factor".
        result = check_gradients(lambda x: self._custom_square(x, 1.0), [3.0])
        assert not result.passed
        assert result.max_error > 0.4

    def test_catches_a_sign_error(self):
        result = check_gradients(lambda x: self._custom_square(x, -2.0), [3.0])
        assert not result.passed

    def test_catches_a_missing_accumulation(self):
        """Exactly the ``=`` instead of ``+=`` bug, which is invisible on
        single-use graphs and halves the gradient on reused ones."""

        def broken_double_use(x: Value) -> Value:
            out = Value(x.data * x.data, (x, x), "*")

            def _backward():
                x.grad = x.data * out.grad  # '=' not '+=' -- only one path

            out._backward = _backward
            return out

        result = check_gradients(broken_double_use, [3.0])
        assert not result.passed
        assert result.analytic[0] == pytest.approx(3.0)
        assert result.numerical[0] == pytest.approx(6.0, rel=1e-6)

    def test_passes_the_correct_version(self):
        result = check_gradients(lambda x: self._custom_square(x, 2.0), [3.0])
        assert result.passed

    def test_raise_on_failure(self):
        with pytest.raises(AssertionError, match="FAIL"):
            check_gradients(
                lambda x: self._custom_square(x, 1.0), [3.0], raise_on_failure=True
            )

    def test_result_string_contains_the_table(self):
        text = str(check_gradients(lambda x: x**3, [2.0], names=["x"]))
        assert "PASS" in text and "analytic" in text and "x" in text


class TestRelativeError:
    def test_identical_values(self):
        assert relative_error(2.0, 2.0) == 0.0

    def test_scale_invariance(self):
        # 1% disagreement is 1% at any magnitude.
        assert relative_error(100.0, 101.0) == pytest.approx(1.0 / 101.0)
        assert relative_error(1e-6, 1.01e-6) == pytest.approx(0.01 / 1.01, rel=1e-6)

    def test_both_near_zero_uses_absolute_difference(self):
        # Without the floor this would report 0.67 for two numbers that are
        # both indistinguishable from zero.
        assert relative_error(1e-18, 3e-18) == pytest.approx(2e-18)

    def test_sign_disagreement_is_large(self):
        assert relative_error(5.0, -5.0) == pytest.approx(2.0)


class TestFiniteDifferenceMachinery:
    def test_order_four_is_more_accurate_than_order_two(self):
        """The 5-point stencil has O(h⁴) truncation error against O(h²)."""
        fn = lambda x: x.exp() * x.tanh()  # noqa: E731
        e2 = check_gradients(fn, [1.3], eps=1e-4, order=2).max_error
        e4 = check_gradients(fn, [1.3], eps=1e-4, order=4).max_error
        assert e4 < e2

    def test_numerical_gradient_ignores_backward_rules_entirely(self):
        """Independence is the whole point, so prove it rather than assert it.

        Here is an operation whose backward rule is nonsense. The numerical
        gradient still comes out right, because it is derived from the forward
        pass alone and never consults ``_backward`` or ``.grad``. That is what
        makes it a valid check on the analytic path.
        """

        def square_with_nonsense_gradient(x: Value) -> Value:
            out = Value(x.data**2, (x,), "sq")

            def _backward():
                x.grad += 999.0 * out.grad  # wildly wrong

            out._backward = _backward
            return out

        numerical = numerical_gradient(square_with_nonsense_gradient, [3.0])
        assert numerical[0] == pytest.approx(6.0, rel=1e-6)  # 2x, correct

        # ...while the analytic path faithfully reports the nonsense, and the
        # checker flags the disagreement.
        assert not check_gradients(square_with_nonsense_gradient, [3.0]).passed

    def test_rejects_a_bad_order(self):
        with pytest.raises(ValueError, match="order must be"):
            numerical_gradient(lambda x: x, [1.0], order=3)

    def test_rejects_non_scalar_output(self):
        with pytest.raises(TypeError, match="must return a Value"):
            check_gradients(lambda x: 5.0, [1.0])  # type: ignore[arg-type,return-value]

    def test_inputs_are_restored_after_perturbation(self):
        point = [1.0, 2.0, 3.0]
        numerical_gradient(lambda a, b, c: a * b * c, point)
        assert point == [1.0, 2.0, 3.0]

    def test_epsilon_sweep_shows_the_u_curve(self):
        """The central claim of docs/01 §3, measured rather than asserted.

        Error falls like h² while truncation dominates, bottoms out near
        h≈1e-5, then rises again as catastrophic cancellation takes over.
        """
        rows = epsilon_sweep(lambda x: x**3, [2.0])
        errors = {h: err for h, _, err in rows}

        assert errors[1e-1] > errors[1e-5]   # truncation dominates on the left
        assert errors[1e-13] > errors[1e-5]  # cancellation dominates on the right

        # h² scaling: shrinking h by 10 should cut the error by ~100.
        assert errors[1e-2] / errors[1e-3] == pytest.approx(100.0, rel=0.2)

        # And no step size ever recovers full double precision.
        assert min(errors.values()) > 1e-14


# ======================================================================
# Checking parameters in place -- the harness Phase 7 will rely on
# ======================================================================


class TestParameterGradientChecking:
    def test_checks_a_hand_built_two_layer_network(self):
        """A miniature MLP wired by hand: 2 inputs -> 2 tanh units -> 1 output,
        squared-error loss. Nine parameters, all checked at once.

        This is the acceptance test pattern for Phase 7 -- gradients verified
        through an entire network, not one operation at a time.
        """
        w = [Value(v) for v in (0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.9, 0.7)]
        x1, x2, target = Value(0.6), Value(-0.4), 0.25

        def loss_fn() -> Value:
            h1 = (x1 * w[0] + x2 * w[1] + w[2]).tanh()
            h2 = (x1 * w[3] + x2 * w[4] + w[5]).tanh()
            out = h1 * w[6] + h2 * w[7] + w[8]
            return (out - target) ** 2

        result = check_parameter_gradients(
            loss_fn, w, names=[f"w{i}" for i in range(9)], label="2-2-1 MLP"
        )
        assert result.passed, "\n" + str(result)

    def test_restores_parameter_values_exactly(self):
        params = [Value(1.5), Value(-2.5)]
        originals = [p.data for p in params]
        check_parameter_gradients(lambda: params[0] * params[1], params)
        assert [p.data for p in params] == originals

    def test_restores_values_even_when_the_loss_raises(self):
        params = [Value(1.0)]
        calls = {"n": 0}

        def flaky() -> Value:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("boom")
            return params[0] ** 2

        with pytest.raises(RuntimeError):
            check_parameter_gradients(flaky, params)
        assert params[0].data == 1.0

    def test_detects_a_stale_gradient(self):
        """If a caller forgets to zero grads between steps, the analytic value
        is inflated. The checker must notice."""
        p = Value(2.0)
        p.grad = 100.0  # left over from a previous step

        # check_parameter_gradients zeroes first, so this passes...
        assert check_parameter_gradients(lambda: p**2, [p]).passed

        # ...but the underlying failure mode is real, so confirm the checker
        # would catch it if the zeroing were removed.
        p.grad = 0.0
        (p**2).backward()
        (p**2).backward()  # accumulated twice: 4 + 4 = 8, truth is 4
        assert relative_error(p.grad, 4.0) > 0.4
