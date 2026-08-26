"""Phase 2 -- forward correctness of every operation.

Backward rules are only meaningful if the forward pass is right, so the two are
tested separately. Every expected value here comes from the mathematical
definition, computed independently of the engine.
"""

from __future__ import annotations

import math

import pytest

from nabla import Value


class TestArithmetic:
    def test_add(self):
        assert (Value(2.0) + Value(3.0)).data == 5.0

    def test_add_scalar(self):
        assert (Value(2.0) + 3).data == 5.0

    def test_mul(self):
        assert (Value(2.0) * Value(3.0)).data == 6.0

    def test_sub(self):
        assert (Value(5.0) - Value(3.0)).data == 2.0

    def test_sub_is_not_commutative(self):
        assert (Value(3.0) - Value(5.0)).data == -2.0
        assert (3 - Value(5.0)).data == -2.0

    def test_neg(self):
        assert (-Value(2.5)).data == -2.5

    def test_div(self):
        assert (Value(6.0) / Value(3.0)).data == 2.0

    def test_rdiv(self):
        assert (6 / Value(3.0)).data == 2.0

    def test_div_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            Value(1.0) / Value(0.0)


class TestPower:
    @pytest.mark.parametrize(
        "base, exp, expected",
        [
            (2.0, 3.0, 8.0),
            (2.0, -1.0, 0.5),
            (4.0, 0.5, 2.0),
            (5.0, 0.0, 1.0),
            (-2.0, 3.0, -8.0),  # negative base, integer exponent: fine
            (-2.0, 2.0, 4.0),
        ],
    )
    def test_constant_exponent(self, base, exp, expected):
        assert (Value(base) ** exp).data == pytest.approx(expected)

    def test_value_exponent(self):
        assert (Value(2.0) ** Value(3.0)).data == pytest.approx(8.0)

    def test_negative_base_fractional_exponent_rejected(self):
        # Python would return a complex number here, silently corrupting the
        # graph. We refuse instead.
        with pytest.raises(ValueError, match="complex"):
            Value(-2.0) ** 0.5

    def test_value_exponent_requires_positive_base(self):
        with pytest.raises(ValueError, match="positive base"):
            Value(-2.0) ** Value(3.0)

    def test_rpow(self):
        assert (2 ** Value(3.0)).data == pytest.approx(8.0)

    def test_rpow_requires_positive_base(self):
        with pytest.raises(ValueError, match="positive base"):
            (-2) ** Value(3.0)


class TestTranscendental:
    @pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
    def test_exp(self, x):
        assert Value(x).exp().data == pytest.approx(math.exp(x))

    def test_exp_of_zero_is_one(self):
        assert Value(0.0).exp().data == 1.0

    @pytest.mark.parametrize("x", [0.001, 0.5, 1.0, 2.0, 1000.0])
    def test_log(self, x):
        assert Value(x).log().data == pytest.approx(math.log(x))

    def test_log_of_one_is_zero(self):
        assert Value(1.0).log().data == 0.0

    @pytest.mark.parametrize("x", [0.0, -1.0])
    def test_log_domain_error(self, x):
        with pytest.raises(ValueError, match="positive"):
            Value(x).log()

    def test_exp_log_round_trip(self):
        assert Value(3.7).log().exp().data == pytest.approx(3.7)


class TestActivations:
    @pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_tanh(self, x):
        assert Value(x).tanh().data == pytest.approx(math.tanh(x))

    def test_tanh_is_odd_and_bounded(self):
        assert Value(0.0).tanh().data == 0.0
        assert Value(-2.0).tanh().data == pytest.approx(-Value(2.0).tanh().data)
        assert -1.0 < Value(50.0).tanh().data <= 1.0

    def test_tanh_saturates_without_overflow(self):
        assert Value(1000.0).tanh().data == pytest.approx(1.0)
        assert Value(-1000.0).tanh().data == pytest.approx(-1.0)

    @pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_sigmoid(self, x):
        assert Value(x).sigmoid().data == pytest.approx(1.0 / (1.0 + math.exp(-x)))

    def test_sigmoid_at_zero_is_half(self):
        assert Value(0.0).sigmoid().data == 0.5

    def test_sigmoid_is_stable_at_extremes(self):
        # The naive 1/(1+exp(-x)) overflows for x = -800. The branched form
        # in the implementation does not.
        assert Value(-800.0).sigmoid().data == pytest.approx(0.0, abs=1e-300)
        assert Value(800.0).sigmoid().data == pytest.approx(1.0)
        assert 0.0 <= Value(-800.0).sigmoid().data <= 1.0

    def test_sigmoid_symmetry(self):
        # σ(-x) = 1 - σ(x)
        for x in (0.3, 2.0, 7.5):
            assert Value(-x).sigmoid().data == pytest.approx(
                1.0 - Value(x).sigmoid().data
            )

    @pytest.mark.parametrize(
        "x, expected", [(-3.0, 0.0), (-0.001, 0.0), (0.0, 0.0), (0.001, 0.001), (3.0, 3.0)]
    )
    def test_relu(self, x, expected):
        assert Value(x).relu().data == pytest.approx(expected)

    def test_relu_is_idempotent(self):
        for x in (-2.0, 0.0, 4.0):
            assert Value(x).relu().relu().data == Value(x).relu().data


class TestEdgeCases:
    """The cases the brief calls out explicitly."""

    def test_zero_values(self):
        z = Value(0.0)
        assert (z + z).data == 0.0
        assert (z * Value(5.0)).data == 0.0
        assert z.exp().data == 1.0
        assert z.tanh().data == 0.0
        assert z.relu().data == 0.0
        assert z.sigmoid().data == 0.5

    def test_negative_values(self):
        n = Value(-4.0)
        assert (n * Value(-2.0)).data == 8.0
        assert (n**2).data == 16.0
        assert n.relu().data == 0.0
        assert n.tanh().data < 0.0

    def test_large_and_small_magnitudes(self):
        assert (Value(1e200) * Value(1e200)).data == math.inf  # honest overflow
        assert (Value(1e-200) * Value(1e-200)).data == 0.0  # honest underflow

    def test_expression_composition(self):
        # A small nested expression, checked against direct arithmetic.
        x, y, b = Value(2.0), Value(-3.0), Value(1.0)
        out = ((x * y + b) ** 2) / (x + y).tanh().exp()
        expected = ((2.0 * -3.0 + 1.0) ** 2) / math.exp(math.tanh(2.0 - 3.0))
        assert out.data == pytest.approx(expected)
