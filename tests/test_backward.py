"""Phase 2/4 -- backward correctness, against derivatives computed by hand.

Every expected gradient in this file was derived on paper before it was typed.
Where a literal appears (``assert x.grad == 30.0``) the derivation is in the
comment above it. This is the *first* line of defence; the *second*, entirely
independent one is finite-difference checking in ``test_gradients.py``
(Phase 5). Two methods that share no reasoning must agree.
"""

from __future__ import annotations

import math

import pytest

from nabla import Value


# ======================================================================
# Single operations: one node, one rule
# ======================================================================


class TestSingleOperationRules:
    def test_add_routes_gradient_unchanged(self):
        # z = x + y  ->  dz/dx = 1, dz/dy = 1
        x, y = Value(2.0), Value(3.0)
        z = x + y
        z.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_mul_swaps_the_inputs(self):
        # z = x * y  ->  dz/dx = y = 3, dz/dy = x = 2
        x, y = Value(2.0), Value(3.0)
        z = x * y
        z.backward()
        assert x.grad == 3.0
        assert y.grad == 2.0

    def test_sub_flips_the_sign_of_the_second_operand(self):
        # z = x - y  ->  dz/dx = 1, dz/dy = -1
        x, y = Value(7.0), Value(4.0)
        z = x - y
        z.backward()
        assert x.grad == 1.0
        assert y.grad == -1.0

    def test_neg(self):
        # z = -x  ->  dz/dx = -1
        x = Value(3.0)
        (-x).backward()
        assert x.grad == -1.0

    def test_div(self):
        # z = x / y  ->  dz/dx = 1/y = 0.25,  dz/dy = -x/y² = -8/16 = -0.5
        x, y = Value(8.0), Value(4.0)
        z = x / y
        z.backward()
        assert x.grad == pytest.approx(0.25)
        assert y.grad == pytest.approx(-0.5)

    def test_pow_constant_exponent(self):
        # z = x³  ->  dz/dx = 3x² = 3*4 = 12
        x = Value(2.0)
        (x**3).backward()
        assert x.grad == pytest.approx(12.0)

    def test_pow_negative_exponent(self):
        # z = x⁻¹  ->  dz/dx = -x⁻² = -1/16
        x = Value(4.0)
        (x**-1).backward()
        assert x.grad == pytest.approx(-1.0 / 16.0)

    def test_pow_both_operands_are_values(self):
        # z = a^b, a=2, b=3, z=8
        #   dz/da = b·a^(b-1) = 3·4  = 12
        #   dz/db = a^b·ln a  = 8·ln2 ≈ 5.5452
        a, b = Value(2.0), Value(3.0)
        (a**b).backward()
        assert a.grad == pytest.approx(12.0)
        assert b.grad == pytest.approx(8.0 * math.log(2.0))

    def test_rpow(self):
        # z = 2^x, x=3, z=8  ->  dz/dx = 2^x·ln2 = 8·ln2
        x = Value(3.0)
        (2**x).backward()
        assert x.grad == pytest.approx(8.0 * math.log(2.0))

    def test_exp(self):
        # z = e^x  ->  dz/dx = e^x = z
        x = Value(1.5)
        z = x.exp()
        z.backward()
        assert x.grad == pytest.approx(math.exp(1.5))
        assert x.grad == pytest.approx(z.data)  # the rule reuses the output

    def test_log(self):
        # z = ln x  ->  dz/dx = 1/x = 0.25
        x = Value(4.0)
        x.log().backward()
        assert x.grad == pytest.approx(0.25)

    def test_tanh(self):
        # z = tanh x  ->  dz/dx = 1 - tanh²x
        x = Value(0.8)
        x.tanh().backward()
        assert x.grad == pytest.approx(1.0 - math.tanh(0.8) ** 2)

    def test_tanh_derivative_peaks_at_zero(self):
        # The maximum of 1 - tanh²x is 1, at x = 0.
        x = Value(0.0)
        x.tanh().backward()
        assert x.grad == pytest.approx(1.0)

    def test_tanh_saturates_to_zero_gradient(self):
        # This is the vanishing-gradient mechanism, measured directly.
        x = Value(10.0)
        x.tanh().backward()
        assert x.grad == pytest.approx(0.0, abs=1e-8)

    def test_sigmoid(self):
        # z = σ(x)  ->  dz/dx = σ(1-σ)
        x = Value(0.5)
        s = 1.0 / (1.0 + math.exp(-0.5))
        x.sigmoid().backward()
        assert x.grad == pytest.approx(s * (1.0 - s))

    def test_sigmoid_derivative_peaks_at_one_quarter(self):
        # σ'(0) = 0.25 -- sigmoid attenuates gradient by 4x even at its best.
        x = Value(0.0)
        x.sigmoid().backward()
        assert x.grad == pytest.approx(0.25)

    def test_relu_passes_positive_gradient(self):
        x = Value(3.0)
        x.relu().backward()
        assert x.grad == 1.0

    def test_relu_blocks_negative_gradient(self):
        x = Value(-3.0)
        x.relu().backward()
        assert x.grad == 0.0

    def test_relu_at_exactly_zero_uses_the_zero_subgradient(self):
        # Not differentiable here; 0 is a valid subgradient and matches
        # PyTorch's convention. Pinned so a future refactor cannot drift.
        x = Value(0.0)
        x.relu().backward()
        assert x.grad == 0.0


# ======================================================================
# Composite expressions: the chain rule composing itself
# ======================================================================


class TestHandComputedExpressions:
    def test_running_example_from_the_docs(self):
        r"""L = (x·w + b)², with x=2, w=-3, b=1.

        Forward:  u = xw = -6,  v = u + b = -5,  L = v² = 25
        Backward: L̄ = 1
                  v̄ = 2v         = -10
                  ū = v̄·1        = -10 ;  b̄ = v̄·1 = -10
                  x̄ = ū·w = -10·-3 = +30
                  w̄ = ū·x = -10·2  = -20

        Cross-check with pencil calculus:
            ∂L/∂w = 2(xw+b)·x = 2(-5)(2)  = -20  ✓
            ∂L/∂x = 2(xw+b)·w = 2(-5)(-3) = +30  ✓
            ∂L/∂b = 2(xw+b)·1 = -10             ✓
        """
        x, w, b = Value(2.0, label="x"), Value(-3.0, label="w"), Value(1.0, label="b")
        L = (x * w + b) ** 2
        assert L.data == 25.0

        L.backward()
        assert L.grad == 1.0
        assert x.grad == pytest.approx(30.0)
        assert w.grad == pytest.approx(-20.0)
        assert b.grad == pytest.approx(-10.0)

    def test_perturbation_matches_the_predicted_change(self):
        """The gradient's *meaning*: nudge w, watch L move by ~ grad·epsilon.

        This is the definition of a derivative, tested directly, and it is the
        idea that Phase 5's gradient checker automates.
        """
        def loss(w_val: float) -> float:
            return (2.0 * w_val + 1.0) ** 2

        w = Value(-3.0)
        L = (Value(2.0) * w + Value(1.0)) ** 2
        L.backward()

        eps = 1e-6
        measured = (loss(-3.0 + eps) - loss(-3.0)) / eps
        assert measured == pytest.approx(w.grad, rel=1e-4)

    def test_docstring_example_d_equals_ab_plus_a(self):
        # d = a·b + a  ->  ∂d/∂a = b + 1 = 4,  ∂d/∂b = a = 2
        a, b = Value(2.0), Value(3.0)
        d = a * b + a
        assert d.data == 8.0
        d.backward()
        assert a.grad == 4.0
        assert b.grad == 2.0

    def test_product_of_sum_and_product(self):
        r"""e = (a·b)·(a+b) = a²b + ab², a=2, b=3.

        ∂e/∂a = 2ab + b² = 12 + 9  = 21
        ∂e/∂b = a² + 2ab = 4  + 12 = 16
        """
        a, b = Value(2.0), Value(3.0)
        c = a * b          # 6
        d = a + b          # 5
        e = c * d          # 30
        assert e.data == 30.0

        e.backward()
        assert a.grad == pytest.approx(21.0)
        assert b.grad == pytest.approx(16.0)

    def test_neuron_forward_and_backward(self):
        r"""A single neuron: o = tanh(x·w + b), x=1.5, w=2, b=-0.5.

        n = xw + b = 2.5
        o = tanh(2.5)
        ∂o/∂n = 1 - tanh²(2.5)
        ∂o/∂w = ∂o/∂n · x
        ∂o/∂x = ∂o/∂n · w
        ∂o/∂b = ∂o/∂n
        """
        x, w, b = Value(1.5), Value(2.0), Value(-0.5)
        n = x * w + b
        o = n.tanh()
        assert n.data == pytest.approx(2.5)

        o.backward()
        dodn = 1.0 - math.tanh(2.5) ** 2
        assert n.grad == pytest.approx(dodn)
        assert w.grad == pytest.approx(dodn * 1.5)
        assert x.grad == pytest.approx(dodn * 2.0)
        assert b.grad == pytest.approx(dodn)

    def test_deeply_nested_expression(self):
        r"""f = ln( (x·y)² + e^{y} ), x=2, y=1.

        u = xy = 2 ;  p = u² = 4 ;  q = e^y = e ;  s = p + q ;  f = ln s

        ∂f/∂s = 1/s
        ∂f/∂p = 1/s        ->  ∂f/∂u = 2u/s
        ∂f/∂x = 2u·y / s   = (2·2·1)/s = 4/s
        ∂f/∂y = 2u·x / s + e^y / s = (4·2 + e)/s
        """
        x, y = Value(2.0), Value(1.0)
        f = ((x * y) ** 2 + y.exp()).log()

        s = 4.0 + math.e
        assert f.data == pytest.approx(math.log(s))

        f.backward()
        assert x.grad == pytest.approx(4.0 / s)
        assert y.grad == pytest.approx((8.0 + math.e) / s)


# ======================================================================
# Gradient accumulation: the bug that only appears when a value is reused
# ======================================================================


class TestGradientAccumulation:
    def test_x_times_x(self):
        """L = x·x. Both operands are the *same node*.

        The multiply rule fires twice -- once per edge -- contributing x each
        time. With ``+=`` the result is 2x = 6. With ``=`` it would be 3.
        This single test is what separates a correct engine from a subtly
        broken one.
        """
        x = Value(3.0)
        L = x * x
        L.backward()
        assert x.grad == pytest.approx(6.0)  # 2x, not x

    def test_x_squared_via_pow_agrees_with_x_times_x(self):
        # Two different graphs for the same function must give the same grad.
        a = Value(3.0)
        (a * a).backward()

        b = Value(3.0)
        (b**2).backward()

        assert a.grad == pytest.approx(b.grad)

    def test_x_cubed_by_repeated_multiplication(self):
        # L = x·x·x  ->  dL/dx = 3x² = 27 at x = 3.
        # Three paths from x to L, each contributing x² = 9.
        x = Value(3.0)
        (x * x * x).backward()
        assert x.grad == pytest.approx(27.0)

    def test_diamond_graph(self):
        r"""Two paths that rejoin.

              ┌──> b = 2x ──┐
          x ──┤              ├──> L = b + c
              └──> c = x³ ──┘

        dL/dx = 2 + 3x² = 2 + 12 = 14 at x = 2.
        """
        x = Value(2.0)
        b = 2 * x
        c = x**3
        L = b + c
        L.backward()
        assert x.grad == pytest.approx(14.0)

    def test_three_paths_with_different_local_derivatives(self):
        # L = x² + tanh(x) + e^x  ->  dL/dx = 2x + (1-tanh²x) + e^x
        x = Value(0.7)
        L = x**2 + x.tanh() + x.exp()
        L.backward()
        expected = 2 * 0.7 + (1 - math.tanh(0.7) ** 2) + math.exp(0.7)
        assert x.grad == pytest.approx(expected)

    def test_shared_intermediate_used_by_two_consumers(self):
        r"""u is consumed twice; its gradient is the sum of both demands.

        u = x + y ;  L = u² + 3u  ->  dL/du = 2u + 3
        dL/dx = dL/dy = 2u + 3 = 2·5 + 3 = 13
        """
        x, y = Value(2.0), Value(3.0)
        u = x + y
        L = u**2 + 3 * u
        L.backward()
        assert u.grad == pytest.approx(13.0)
        assert x.grad == pytest.approx(13.0)
        assert y.grad == pytest.approx(13.0)

    def test_backward_accumulates_across_calls(self):
        """PyTorch semantics, reproduced deliberately (decision D6).

        Calling backward() twice on the same graph sums two gradients. This is
        the mechanism behind gradient accumulation over micro-batches -- and
        the reason a training loop must call zero_grad().
        """
        x = Value(3.0)
        L = x * x
        L.backward()
        assert x.grad == pytest.approx(6.0)

        L.backward()
        assert x.grad == pytest.approx(12.0)  # accumulated, not reset

        L.zero_grad()
        L.backward()
        assert x.grad == pytest.approx(6.0)  # cleared, back to one gradient

    def test_two_independent_graphs_share_a_leaf(self):
        # A parameter used in two separate losses collects both gradients --
        # exactly what happens with weight sharing.
        w = Value(2.0)
        (w * 3).backward()
        (w * 5).backward()
        assert w.grad == pytest.approx(8.0)


# ======================================================================
# Structure: ordering and scale
# ======================================================================


class TestBackwardStructure:
    def test_seed_gradient_is_one(self):
        x = Value(2.0)
        L = x * 5
        L.backward()
        assert L.grad == 1.0

    def test_intermediate_nodes_receive_gradients_too(self):
        x = Value(2.0)
        u = x * 3      # 6
        L = u + 1      # 7
        L.backward()
        assert u.grad == 1.0
        assert x.grad == 3.0

    def test_nodes_off_the_path_are_untouched(self):
        # A value not used to compute L must keep grad == 0.
        x, unused = Value(2.0), Value(9.0)
        (x * 3).backward()
        assert unused.grad == 0.0

    def test_constant_leaves_get_gradients_that_nobody_reads(self):
        # `3` becomes a constant node; it does receive a gradient. Harmless --
        # no optimiser ever looks at it.
        x = Value(2.0)
        out = x * 3
        out.backward()
        constant = [p for p in out._prev if p is not x][0]
        assert constant.grad == pytest.approx(2.0)

    def test_deep_chain_does_not_hit_the_recursion_limit(self):
        """A 20,000-link chain. The textbook recursive topological sort would
        raise RecursionError here; the iterative one (decision D5) does not.

        y = x + 0.001 repeated N times  ->  dy/dx = 1 exactly.
        """
        x = Value(1.0)
        y = x
        for _ in range(20_000):
            y = y + 0.001
        y.backward()
        assert x.grad == pytest.approx(1.0)

    def test_deep_multiplicative_chain_scales_gradient(self):
        """Gradient explosion, demonstrated at small scale.

        y = x·1.1^n  ->  dy/dx = 1.1^n. For n=100 that is ~13780: the gradient
        is four orders of magnitude larger than the seed. Multiply by a
        thousand layers and you get overflow -- this is exactly the mechanism
        behind exploding gradients (Phase 22).
        """
        x = Value(1.0)
        y = x
        for _ in range(100):
            y = y * 1.1
        y.backward()
        assert x.grad == pytest.approx(1.1**100, rel=1e-9)

    def test_deep_shrinking_chain_vanishes_gradient(self):
        # The mirror image: 0.9^100 ≈ 2.65e-5. Gradient nearly gone.
        x = Value(1.0)
        y = x
        for _ in range(100):
            y = y * 0.9
        y.backward()
        assert x.grad == pytest.approx(0.9**100, rel=1e-9)
        assert x.grad < 1e-4


class TestEdgeCaseGradients:
    def test_gradient_at_zero(self):
        # L = x², dL/dx = 2x = 0 at x = 0. A zero gradient is a real answer,
        # not a failure -- it means we are at a stationary point.
        x = Value(0.0)
        (x**2).backward()
        assert x.grad == 0.0

    def test_gradient_with_negative_values(self):
        # L = x·y with x=-2, y=-3  ->  dL/dx = -3, dL/dy = -2
        x, y = Value(-2.0), Value(-3.0)
        (x * y).backward()
        assert x.grad == -3.0
        assert y.grad == -2.0

    def test_multiplication_by_zero_kills_the_gradient(self):
        # dL/dx = y = 0: x genuinely has no influence on L here.
        x, y = Value(5.0), Value(0.0)
        (x * y).backward()
        assert x.grad == 0.0
        assert y.grad == 5.0

    def test_gradient_through_a_zero_valued_intermediate(self):
        x = Value(2.0)
        u = x - 2.0        # exactly 0.0
        L = u * u          # 0.0
        L.backward()
        assert x.grad == 0.0  # 2u = 0

    def test_subtraction_of_a_value_from_itself(self):
        # L = x - x = 0 for all x, so dL/dx must be exactly 0 --
        # +1 from the left path and -1 from the right, cancelling.
        x = Value(4.0)
        L = x - x
        assert L.data == 0.0
        L.backward()
        assert x.grad == 0.0

    def test_division_of_a_value_by_itself(self):
        # L = x/x = 1, so dL/dx = 0: (1/x) + (-x/x²) = 0.
        x = Value(4.0)
        L = x / x
        assert L.data == pytest.approx(1.0)
        L.backward()
        assert x.grad == pytest.approx(0.0, abs=1e-12)
