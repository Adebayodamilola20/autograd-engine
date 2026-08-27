"""Phase 16 -- optimisers.

Two kinds of test here:

* **Mechanical** -- does the update formula match the paper, step by step,
  with hand-computed arithmetic?
* **Behavioural** -- does it actually minimise a function whose minimum we
  know?

Both are needed. An optimiser can implement a formula perfectly and still fail
to converge if a sign is wrong somewhere else, and it can converge on an easy
problem while implementing the wrong algorithm.
"""

from __future__ import annotations

import math

import pytest

from nabla.nn import MLP, Parameter
from nabla.optim import SGD, Adam, AdamW, Optimizer


def quadratic_bowl(params, centre=(3.0, -2.0), curvature=(1.0, 1.0)):
    r"""L = Σ cᵢ(θᵢ - centreᵢ)², whose minimum is exactly at ``centre``.

    Setting ``curvature`` unequal creates an ill-conditioned valley -- the
    situation momentum and Adam exist to handle.
    """
    total = None
    for p, c, k in zip(params, centre, curvature):
        term = k * (p - c) ** 2
        total = term if total is None else total + term
    return total


def run(optimizer, params, steps=200, **kwargs):
    for _ in range(steps):
        optimizer.zero_grad()
        quadratic_bowl(params, **kwargs).backward()
        optimizer.step()
    return [p.data for p in params]


# ======================================================================
# base class
# ======================================================================


class TestOptimizerBase:
    def test_rejects_empty_parameters(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            SGD([], lr=0.1)

    def test_rejects_non_positive_lr(self):
        with pytest.raises(ValueError, match="positive"):
            SGD([Parameter(1.0)], lr=0.0)
        with pytest.raises(ValueError, match="positive"):
            SGD([Parameter(1.0)], lr=-0.1)

    def test_step_is_abstract(self):
        with pytest.raises(NotImplementedError):
            Optimizer([Parameter(1.0)], lr=0.1).step()

    def test_zero_grad_clears(self):
        p = Parameter(1.0)
        p.grad = 5.0
        SGD([p], lr=0.1).zero_grad()
        assert p.grad == 0.0

    def test_gradient_norm(self):
        a, b = Parameter(1.0), Parameter(1.0)
        a.grad, b.grad = 3.0, 4.0
        assert SGD([a, b], lr=0.1).gradient_norm() == pytest.approx(5.0)

    def test_clip_preserves_direction(self):
        """Scaling by one factor keeps the descent direction; per-component
        clipping would bend it."""
        a, b = Parameter(1.0), Parameter(1.0)
        a.grad, b.grad = 30.0, 40.0  # norm 50
        opt = SGD([a, b], lr=0.1)

        before = opt.clip_grad_norm(5.0)
        assert before == pytest.approx(50.0)  # returns the pre-clip norm
        assert opt.gradient_norm() == pytest.approx(5.0)
        assert a.grad / b.grad == pytest.approx(30.0 / 40.0)  # direction intact

    def test_clip_leaves_small_gradients_alone(self):
        a = Parameter(1.0)
        a.grad = 0.5
        SGD([a], lr=0.1).clip_grad_norm(5.0)
        assert a.grad == 0.5

    def test_step_count_increments(self):
        p = Parameter(1.0)
        opt = SGD([p], lr=0.1)
        for i in range(3):
            assert opt.step_count == i
            opt.step()
        assert opt.step_count == 3


# ======================================================================
# SGD
# ======================================================================


class TestSGD:
    def test_the_update_rule_by_hand(self):
        # θ ← θ - η·g  =  5.0 - 0.1*2.0 = 4.8
        p = Parameter(5.0)
        p.grad = 2.0
        SGD([p], lr=0.1).step()
        assert p.data == pytest.approx(4.8)

    def test_moves_against_the_gradient(self):
        up, down = Parameter(0.0), Parameter(0.0)
        up.grad, down.grad = 1.0, -1.0
        SGD([up, down], lr=0.5).step()
        assert up.data < 0 and down.data > 0

    def test_converges_to_a_known_minimum(self):
        params = [Parameter(0.0), Parameter(0.0)]
        result = run(SGD(params, lr=0.1), params)
        assert result == pytest.approx([3.0, -2.0], abs=1e-6)

    def test_diverges_when_the_learning_rate_is_too_high(self):
        r"""The stability boundary, derived and then observed.

        For :math:`L = (\theta-c)^2` the gradient is :math:`2(\theta-c)`, so the
        error :math:`e = \theta - c` evolves as

        .. math:: e_{t+1} = e_t(1 - 2\eta)

        which converges iff :math:`|1-2\eta| < 1`, i.e. :math:`0 < \eta < 1`.
        At :math:`\eta = 1.1` the factor is :math:`-1.2`, so the error grows by
        20% per step *and alternates sign* -- the classic overshoot-and-
        oscillate divergence.

        This is not a bug. It is exactly why learning rates must be tuned, and
        Phase 22 explores it on a real network.
        """
        params = [Parameter(0.0), Parameter(0.0)]
        run(SGD(params, lr=1.1), params, steps=60)

        # error after n steps ≈ |e₀|·1.2ⁿ
        predicted = 3.0 * 1.2**60
        assert abs(params[0].data) == pytest.approx(predicted, rel=0.05)
        assert abs(params[0].data) > 1e5

    def test_stays_stable_just_below_the_boundary(self):
        # η = 0.9 gives |1-2η| = 0.8 < 1: converges, if slowly.
        params = [Parameter(0.0), Parameter(0.0)]
        run(SGD(params, lr=0.9), params, steps=300)
        assert [p.data for p in params] == pytest.approx([3.0, -2.0], abs=1e-6)

    def test_momentum_by_hand(self):
        # step 1: v = 0.9*0 + 2 = 2      θ = 5 - 0.1*2 = 4.8
        # step 2: v = 0.9*2 + 2 = 3.8    θ = 4.8 - 0.1*3.8 = 4.42
        p = Parameter(5.0)
        opt = SGD([p], lr=0.1, momentum=0.9)
        p.grad = 2.0
        opt.step()
        assert p.data == pytest.approx(4.8)
        p.grad = 2.0
        opt.step()
        assert p.data == pytest.approx(4.42)

    def test_momentum_reaches_the_steady_state_amplification(self):
        r"""A constant gradient drives the velocity to :math:`g/(1-\mu)`.

        Expanding the recursion :math:`v_{t+1} = \mu v_t + g` for constant
        :math:`g` gives a geometric series summing to :math:`g/(1-\mu)`, which
        is **10x** at :math:`\mu = 0.9`. That factor is why switching momentum
        on without lowering the learning rate can diverge.

        ``lr`` is tiny here so the parameter barely moves and the gradient we
        feed in stays effectively constant, isolating the buffer's behaviour.
        """
        opt = SGD([Parameter(0.0)], lr=1e-12, momentum=0.9)
        for _ in range(300):
            opt.params[0].grad = 1.0
            opt.step()
        assert opt.velocity[0] == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-6)

    def test_momentum_cancels_an_alternating_gradient(self):
        """The other half of why momentum helps in a ravine: components that
        flip sign every step average to nearly nothing, while consistent ones
        accumulate."""
        opt = SGD([Parameter(0.0)], lr=1e-12, momentum=0.9)
        for i in range(300):
            opt.params[0].grad = 1.0 if i % 2 == 0 else -1.0
            opt.step()
        # 1/(1+μ) ≈ 0.526 instead of 10 -- a 19x difference in effective step.
        assert abs(opt.velocity[0]) == pytest.approx(1.0 / (1.0 + 0.9), rel=1e-6)

    def test_momentum_converges_faster_in_a_ravine(self):
        """The situation momentum exists for: curvature 20 across, 1 along."""
        ravine = {"centre": (3.0, -2.0), "curvature": (20.0, 0.4)}

        plain = [Parameter(0.0), Parameter(0.0)]
        run(SGD(plain, lr=0.02), plain, steps=120, **ravine)

        fast = [Parameter(0.0), Parameter(0.0)]
        run(SGD(fast, lr=0.02, momentum=0.9), fast, steps=120, **ravine)

        def distance(ps):
            return math.dist([p.data for p in ps], [3.0, -2.0])

        assert distance(fast) < distance(plain)

    def test_nesterov_requires_momentum(self):
        with pytest.raises(ValueError, match="nesterov requires"):
            SGD([Parameter(1.0)], lr=0.1, nesterov=True)

    def test_nesterov_converges(self):
        params = [Parameter(0.0), Parameter(0.0)]
        result = run(SGD(params, lr=0.05, momentum=0.9, nesterov=True), params)
        assert result == pytest.approx([3.0, -2.0], abs=1e-5)

    def test_weight_decay_pulls_toward_zero(self):
        # With no data gradient at all, decay alone should shrink the weight.
        p = Parameter(1.0)
        opt = SGD([p], lr=0.1, weight_decay=0.5)
        for _ in range(10):
            opt.zero_grad()
            opt.step()
        assert 0.0 < p.data < 1.0

    def test_weight_decay_shifts_the_optimum(self):
        # Minimum of (θ-3)² + (λ/2)θ² is at 3/(1 + λ/2) < 3.
        params = [Parameter(0.0), Parameter(0.0)]
        run(SGD(params, lr=0.05, weight_decay=0.4), params, steps=500)
        assert params[0].data < 3.0

    def test_rejects_bad_momentum(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            SGD([Parameter(1.0)], lr=0.1, momentum=1.0)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            SGD([Parameter(1.0)], lr=0.1, momentum=-0.1)


# ======================================================================
# Adam
# ======================================================================


class TestAdam:
    def test_first_step_is_almost_exactly_lr(self):
        r"""Bias correction makes the first step ≈ ±lr, whatever the gradient
        magnitude.

        m̂₁ = g and v̂₁ = g², so m̂/√v̂ = g/|g| = sign(g). Without bias
        correction this would be 0.1g/(0.0316|g|) ≈ 3.16 -- three times too
        large. That is the whole reason the correction exists.

        The step falls a hair short of exactly ``lr`` because of the epsilon in
        the denominator: the true value is lr·g/(|g| + ε), so the shortfall is
        a relative ε/|g|. For g = 0.001 that is 1e-5, which is why the
        tolerance here is 1e-4 rather than machine precision -- and it is worth
        checking rather than papering over, since it confirms ε is where we
        think it is.
        """
        for gradient in (0.001, 1.0, 1000.0):
            p = Parameter(0.0)
            p.grad = gradient
            Adam([p], lr=0.1).step()
            assert p.data == pytest.approx(-0.1, rel=1e-4)

            # and the shortfall is exactly the epsilon term, not drift
            expected = -0.1 * gradient / (gradient + 1e-8)
            assert p.data == pytest.approx(expected, rel=1e-12)

    def test_is_scale_invariant(self):
        """Multiply the loss by 1000 and Adam takes the same steps.

        m and √v both scale linearly, so their ratio does not. This is why
        lr=1e-3 works across wildly different problems.
        """
        a = Parameter(0.0)
        opt_a = Adam([a], lr=0.01)
        b = Parameter(0.0)
        opt_b = Adam([b], lr=0.01)
        for _ in range(20):
            a.grad, b.grad = 0.5, 500.0
            opt_a.step()
            opt_b.step()
        assert a.data == pytest.approx(b.data, rel=1e-6)

    def test_sgd_is_not_scale_invariant(self):
        # The contrast that makes the previous test meaningful.
        a, b = Parameter(0.0), Parameter(0.0)
        a.grad, b.grad = 0.5, 500.0
        SGD([a], lr=0.01).step()
        SGD([b], lr=0.01).step()
        assert abs(b.data) == pytest.approx(1000 * abs(a.data))

    def test_moment_update_by_hand(self):
        # m₁ = 0.1*g = 0.2 ;  v₁ = 0.001*g² = 0.004
        p = Parameter(0.0)
        p.grad = 2.0
        opt = Adam([p], lr=0.1)
        opt.step()
        assert opt.m[0] == pytest.approx(0.1 * 2.0)
        assert opt.v[0] == pytest.approx(0.001 * 4.0)

    def test_converges_to_a_known_minimum(self):
        params = [Parameter(0.0), Parameter(0.0)]
        result = run(Adam(params, lr=0.1), params, steps=600)
        assert result == pytest.approx([3.0, -2.0], abs=1e-3)

    def test_handles_the_ravine_without_tuning(self):
        params = [Parameter(0.0), Parameter(0.0)]
        run(Adam(params, lr=0.1), params, steps=600,
            centre=(3.0, -2.0), curvature=(50.0, 0.1))
        assert math.dist([p.data for p in params], [3.0, -2.0]) < 0.05

    def test_zero_gradient_does_not_divide_by_zero(self):
        p = Parameter(1.0)
        opt = Adam([p], lr=0.1)
        opt.zero_grad()
        opt.step()
        assert math.isfinite(p.data)

    @pytest.mark.parametrize("betas", [(1.0, 0.999), (0.9, 1.0), (-0.1, 0.9)])
    def test_rejects_bad_betas(self, betas):
        with pytest.raises(ValueError, match=r"beta[12] must be"):
            Adam([Parameter(1.0)], betas=betas)

    def test_rejects_bad_eps(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            Adam([Parameter(1.0)], eps=0.0)


class TestAdamW:
    def test_decay_is_decoupled_from_the_adaptive_scaling(self):
        r"""Adam folds the penalty into the gradient, so it passes through the
        1/√v̂ rescaling. AdamW applies it directly. With no data gradient the
        two therefore differ."""
        a = Parameter(1.0)
        adam = Adam([a], lr=0.1, weight_decay=0.1)
        b = Parameter(1.0)
        adamw = AdamW([b], lr=0.1, weight_decay=0.1)

        for _ in range(5):
            adam.zero_grad()
            adamw.zero_grad()
            adam.step()
            adamw.step()

        assert a.data != pytest.approx(b.data)
        assert b.data < 1.0  # AdamW shrinks it by exactly lr*λ*θ each step

    def test_decay_amount_matches_the_formula(self):
        p = Parameter(1.0)
        opt = AdamW([p], lr=0.1, weight_decay=0.5)
        opt.zero_grad()
        opt.step()
        # No gradient, so only the decay term acts: θ ← θ - lr·λ·θ
        assert p.data == pytest.approx(1.0 - 0.1 * 0.5 * 1.0)

    def test_still_converges(self):
        params = [Parameter(0.0), Parameter(0.0)]
        run(AdamW(params, lr=0.1), params, steps=600)
        assert [p.data for p in params] == pytest.approx([3.0, -2.0], abs=1e-3)


# ======================================================================
# state
# ======================================================================


class TestOptimizerState:
    @pytest.mark.parametrize(
        "build",
        [
            lambda p: SGD(p, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01),
            lambda p: Adam(p, lr=0.02, betas=(0.8, 0.99)),
            lambda p: AdamW(p, lr=0.02, weight_decay=0.05),
        ],
    )
    def test_round_trip_preserves_the_trajectory(self, build):
        """Resuming without optimiser state restarts momentum from zero and
        produces a visible bump. This checks the state really is restored."""
        params_a = [Parameter(0.0), Parameter(0.0)]
        opt_a = build(params_a)
        run(opt_a, params_a, steps=25)
        state = opt_a.state_dict()

        params_b = [Parameter(p.data) for p in params_a]
        opt_b = build(params_b)
        opt_b.load_state_dict(state)

        run(opt_a, params_a, steps=25)
        run(opt_b, params_b, steps=25)
        assert [p.data for p in params_b] == pytest.approx(
            [p.data for p in params_a], rel=1e-12
        )

    def test_mismatched_buffer_length_raises(self):
        opt = Adam([Parameter(1.0), Parameter(2.0)])
        with pytest.raises(ValueError, match="buffer has"):
            opt.load_state_dict({"m": [0.0], "v": [0.0, 0.0]})

    def test_sgd_mismatched_velocity_raises(self):
        opt = SGD([Parameter(1.0)], lr=0.1, momentum=0.9)
        with pytest.raises(ValueError, match="velocity buffer"):
            opt.load_state_dict({"velocity": [0.0, 0.0]})


# ======================================================================
# on a real model
# ======================================================================


class TestOnARealModel:
    @pytest.mark.parametrize(
        "build",
        [
            lambda p: SGD(p, lr=0.2),
            lambda p: SGD(p, lr=0.2, momentum=0.9),
            lambda p: Adam(p, lr=0.05),
            lambda p: AdamW(p, lr=0.05),
        ],
    )
    def test_every_optimiser_reduces_a_real_loss(self, build):
        from nabla.losses import mse_loss

        model = MLP(2, [6], 1, activation="tanh", seed=0)
        opt = build(model.parameters())
        data = [([0.5, -0.3], 0.8), ([-0.2, 0.7], -0.4), ([1.0, 1.0], 0.1)]

        def epoch():
            opt.zero_grad()
            loss = mse_loss(
                [model(x)[0] for x, _ in data], [t for _, t in data]
            )
            loss.backward()
            opt.step()
            return loss.data

        first = epoch()
        for _ in range(150):
            last = epoch()
        assert last < first * 0.1

    def test_parameters_actually_change(self):
        from nabla.losses import mse_loss

        model = MLP(2, [4], 1, seed=0)
        before = [p.data for p in model.parameters()]
        opt = SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        mse_loss(model([1.0, 2.0]), [0.0]).backward()
        opt.step()
        after = [p.data for p in model.parameters()]
        assert sum(1 for a, b in zip(before, after) if a != b) > 0
