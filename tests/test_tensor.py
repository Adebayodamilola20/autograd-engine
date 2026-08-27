"""Phase 16/18 -- the tensor engine.

The headline test is ``TestBothEnginesAgree``: build the same network on
``Value`` and on ``Tensor``, give them identical weights, and confirm the
gradients match to float64 precision. That is the claim the whole tensor phase
rests on -- same algorithm, different granularity.

Everything else verifies the three things tensors add over scalars:
broadcasting, matmul, and reductions.
"""

from __future__ import annotations

import numpy as np
import pytest

from nabla.core.graph import graph_size, is_topologically_sorted, topological_sort
from nabla.core.tensor import Tensor, _unbroadcast

# ======================================================================
# helpers
# ======================================================================


def numerical_grad(fn, arrays, eps=1e-6):
    """Finite differences over ndarray inputs. Touches no backward rule."""
    grads = []
    for base in arrays:
        g = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            saved = base[idx]
            base[idx] = saved + eps
            plus = float(np.sum(fn(*[Tensor(a) for a in arrays]).data))
            base[idx] = saved - eps
            minus = float(np.sum(fn(*[Tensor(a) for a in arrays]).data))
            base[idx] = saved
            g[idx] = (plus - minus) / (2 * eps)
        grads.append(g)
    return grads


def assert_gradcheck(fn, arrays, tol=1e-6):
    tensors = [Tensor(a.copy()) for a in arrays]
    out = fn(*tensors)
    (out if out.data.size == 1 else out.sum()).backward()

    analytic = [t.grad for t in tensors]
    numerical = numerical_grad(fn, [a.copy() for a in arrays])

    for i, (a, n) in enumerate(zip(analytic, numerical)):
        assert a.shape == arrays[i].shape, (
            f"gradient shape {a.shape} does not match input {arrays[i].shape}"
        )
        scale = np.maximum(np.maximum(np.abs(a), np.abs(n)), 1e-8)
        err = float(np.max(np.abs(a - n) / scale))
        assert err < tol, f"input {i}: max relative error {err:.2e}"


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {
        "A": rng.normal(size=(3, 4)),
        "B": rng.normal(size=(4, 2)),
        "P": rng.uniform(0.5, 2.0, size=(3, 4)),  # strictly positive, for log
        "v": rng.normal(size=(4,)),
        "row": rng.normal(size=(1, 4)),
        "col": rng.normal(size=(3, 1)),
    }


# ======================================================================
# construction
# ======================================================================


class TestConstruction:
    def test_from_nested_lists(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.data.dtype == np.float64

    def test_grad_starts_as_zeros_of_the_right_shape(self):
        t = Tensor(np.ones((3, 5)))
        assert t.grad.shape == (3, 5)
        assert not t.grad.any()

    def test_helpers(self):
        assert Tensor.zeros(2, 3).shape == (2, 3)
        assert Tensor.ones(4).data.tolist() == [1.0, 1.0, 1.0, 1.0]
        assert Tensor.randn(5, 5, rng=np.random.default_rng(0)).shape == (5, 5)

    def test_item_requires_a_scalar(self):
        assert Tensor(3.0).item() == 3.0
        with pytest.raises(ValueError):
            Tensor([1.0, 2.0]).item()

    def test_reset_grad_preserves_shape(self):
        t = Tensor(np.ones((2, 3)))
        t.grad = np.ones((2, 3))
        t.reset_grad()
        assert t.grad.shape == (2, 3) and not t.grad.any()

    def test_rejects_bad_operands(self):
        with pytest.raises(TypeError):
            Tensor([1.0]) + "two"


# ======================================================================
# broadcasting -- the part most likely to be wrong
# ======================================================================


class TestUnbroadcast:
    """Direct tests of the reduction that undoes forward broadcasting."""

    def test_removes_prepended_axes(self):
        grad = np.ones((3, 4))
        assert _unbroadcast(grad, (4,)).tolist() == [3.0, 3.0, 3.0, 3.0]

    def test_sums_stretched_axes_keeping_dims(self):
        grad = np.ones((3, 4))
        assert _unbroadcast(grad, (1, 4)).shape == (1, 4)
        assert _unbroadcast(grad, (1, 4)).tolist() == [[3.0, 3.0, 3.0, 3.0]]
        assert _unbroadcast(grad, (3, 1)).tolist() == [[4.0], [4.0], [4.0]]

    def test_identity_when_shapes_match(self):
        grad = np.arange(12.0).reshape(3, 4)
        assert np.array_equal(_unbroadcast(grad, (3, 4)), grad)

    def test_full_reduction_to_scalar(self):
        assert _unbroadcast(np.ones((3, 4)), ()).item() == 12.0


class TestBroadcastingGradients:
    def test_bias_gradient_is_the_batch_sum(self):
        r"""The case that matters: a ``(n_out,)`` bias added to a
        ``(batch, n_out)`` activation is *used* ``batch`` times, so its
        gradient is the sum over the batch axis -- not one row of it.

        Getting this wrong is the classic tensor-autodiff bug, and it produces
        gradients that are too small by exactly the batch size.
        """
        x = Tensor(np.zeros((8, 3)))
        b = Tensor(np.zeros(3))
        (x + b).sum().backward()
        assert b.grad.tolist() == [8.0, 8.0, 8.0]

    @pytest.mark.parametrize("shape", [(4,), (1, 4), (3, 1), (3, 4)])
    def test_add_broadcast_gradcheck(self, arrays, shape):
        rng = np.random.default_rng(1)
        other = rng.normal(size=shape)
        assert_gradcheck(lambda a, b: a + b, [arrays["A"], other])

    @pytest.mark.parametrize("shape", [(4,), (1, 4), (3, 1)])
    def test_mul_broadcast_gradcheck(self, arrays, shape):
        rng = np.random.default_rng(2)
        other = rng.normal(size=shape)
        assert_gradcheck(lambda a, b: a * b, [arrays["A"], other])

    def test_scalar_broadcast(self, arrays):
        assert_gradcheck(lambda a: a * 3.0 + 1.0, [arrays["A"]])

    def test_gradient_shape_always_matches_input_shape(self, arrays):
        x, b = Tensor(arrays["A"]), Tensor(arrays["v"])
        (x * b).sum().backward()
        assert x.grad.shape == arrays["A"].shape
        assert b.grad.shape == arrays["v"].shape


# ======================================================================
# elementwise and matmul
# ======================================================================


class TestElementwise:
    @pytest.mark.parametrize(
        "name, fn, key",
        [
            ("add", lambda a, b: a + b, "A"),
            ("sub", lambda a, b: a - b, "A"),
            ("mul", lambda a, b: a * b, "A"),
        ],
    )
    def test_binary(self, arrays, name, fn, key):
        assert_gradcheck(fn, [arrays[key], arrays[key].copy()])

    def test_div(self, arrays):
        assert_gradcheck(lambda a, b: a / b, [arrays["A"], arrays["P"]])

    def test_neg(self, arrays):
        assert_gradcheck(lambda a: -a, [arrays["A"]])

    @pytest.mark.parametrize("n", [2, 3, -1])
    def test_pow(self, arrays, n):
        assert_gradcheck(lambda a: a**n, [arrays["P"]])

    def test_pow_rejects_tensor_exponent(self, arrays):
        with pytest.raises(TypeError, match="not implemented"):
            Tensor(arrays["P"]) ** Tensor(arrays["P"])

    def test_reflected_operators(self, arrays):
        assert_gradcheck(lambda a: 2.0 + a, [arrays["A"]])
        assert_gradcheck(lambda a: 2.0 * a, [arrays["A"]])
        assert_gradcheck(lambda a: 2.0 - a, [arrays["A"]])
        assert_gradcheck(lambda a: 2.0 / a, [arrays["P"]])

    @pytest.mark.parametrize(
        "name, fn, key",
        [
            ("exp", lambda a: a.exp(), "A"),
            ("log", lambda a: a.log(), "P"),
            ("tanh", lambda a: a.tanh(), "A"),
            ("sigmoid", lambda a: a.sigmoid(), "A"),
            ("relu", lambda a: a.relu(), "A"),
        ],
    )
    def test_unary(self, arrays, name, fn, key):
        assert_gradcheck(fn, [arrays[key]])

    def test_log_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            Tensor([[1.0, -1.0]]).log()

    def test_sigmoid_is_stable_at_extremes(self):
        s = Tensor([[-800.0, 0.0, 800.0]]).sigmoid()
        assert np.all(np.isfinite(s.data))
        assert s.data.tolist()[0] == pytest.approx([0.0, 0.5, 1.0])


class TestMatmul:
    def test_forward_matches_numpy(self, arrays):
        got = (Tensor(arrays["A"]) @ Tensor(arrays["B"])).data
        assert np.allclose(got, arrays["A"] @ arrays["B"])

    def test_gradcheck(self, arrays):
        assert_gradcheck(lambda a, b: a @ b, [arrays["A"], arrays["B"]])

    def test_gradient_formula_by_hand(self):
        r"""Ā = C̄Bᵀ and B̄ = AᵀC̄, checked with a seed of all ones.

        With C̄ = 1, Ā = 1·Bᵀ means each Ā entry is the row-sum of B, and
        B̄ = Aᵀ·1 means each B̄ entry is the column-sum of A.
        """
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        ta, tb = Tensor(A), Tensor(B)
        (ta @ tb).sum().backward()

        assert np.allclose(ta.grad, np.ones((2, 2)) @ B.T)
        assert np.allclose(tb.grad, A.T @ np.ones((2, 2)))
        assert ta.grad.tolist() == [[11.0, 15.0], [11.0, 15.0]]  # row sums of B
        assert tb.grad.tolist() == [[4.0, 4.0], [6.0, 6.0]]  # column sums of A

    def test_reduces_to_the_scalar_rule(self):
        """For 1x1 matrices, matmul's rule *is* Value.__mul__'s rule."""
        a, b = Tensor([[3.0]]), Tensor([[5.0]])
        (a @ b).backward()
        assert a.grad.item() == 5.0  # the other operand
        assert b.grad.item() == 3.0

    def test_shape_mismatch_raises(self, arrays):
        with pytest.raises(ValueError):
            Tensor(arrays["A"]) @ Tensor(arrays["A"])

    def test_rmatmul(self, arrays):
        out = arrays["A"] @ Tensor(arrays["B"])
        assert out.shape == (3, 2)


# ======================================================================
# reductions and shapes
# ======================================================================


class TestReductions:
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_sum_gradcheck(self, arrays, axis, keepdims):
        assert_gradcheck(lambda a: a.sum(axis=axis, keepdims=keepdims), [arrays["A"]])

    def test_sum_backward_broadcasts_ones(self, arrays):
        t = Tensor(arrays["A"])
        t.sum().backward()
        assert np.array_equal(t.grad, np.ones_like(arrays["A"]))

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_mean_gradcheck(self, arrays, axis):
        assert_gradcheck(lambda a: a.mean(axis=axis), [arrays["A"]])

    def test_mean_backward_is_one_over_n(self, arrays):
        t = Tensor(arrays["A"])
        t.mean().backward()
        assert np.allclose(t.grad, 1.0 / arrays["A"].size)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_max_gradcheck(self, arrays, axis):
        assert_gradcheck(lambda a: a.max(axis=axis), [arrays["A"]])

    def test_max_routes_gradient_only_to_the_winner(self):
        t = Tensor([[1.0, 5.0, 3.0]])
        t.max().backward()
        assert t.grad.tolist() == [[0.0, 1.0, 0.0]]

    def test_max_splits_ties(self):
        """Not differentiable at a tie; splitting evenly keeps the total right
        and is a valid subgradient."""
        t = Tensor([[5.0, 5.0, 1.0]])
        t.max().backward()
        assert t.grad.tolist() == [[0.5, 0.5, 0.0]]
        assert t.grad.sum() == pytest.approx(1.0)


class TestShapeOps:
    def test_reshape_gradcheck(self, arrays):
        assert_gradcheck(lambda a: a.reshape(4, 3), [arrays["A"]])

    def test_reshape_backward_restores_shape(self, arrays):
        t = Tensor(arrays["A"])
        t.reshape(12).sum().backward()
        assert t.grad.shape == (3, 4)

    def test_transpose_gradcheck(self, arrays):
        assert_gradcheck(lambda a: a.transpose(), [arrays["A"]])

    def test_transpose_property(self, arrays):
        assert Tensor(arrays["A"]).T.shape == (4, 3)

    def test_transpose_inverse_permutation(self):
        t = Tensor(np.arange(24.0).reshape(2, 3, 4))
        out = t.transpose(2, 0, 1)
        assert out.shape == (4, 2, 3)
        (out * 2.0).sum().backward()
        assert t.grad.shape == (2, 3, 4)

    def test_getitem_gradcheck(self, arrays):
        assert_gradcheck(lambda a: a[1:], [arrays["A"]])

    def test_getitem_scatter_add_accumulates_repeats(self):
        """Fancy indexing with repeats must accumulate, not overwrite -- the
        same rule as everywhere else, in NumPy's clothing."""
        t = Tensor(np.arange(6.0).reshape(3, 2))
        t[[0, 0, 2]].sum().backward()
        assert t.grad.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]


# ======================================================================
# softmax
# ======================================================================


class TestSoftmaxFamily:
    def test_probabilities_sum_to_one_per_row(self, arrays):
        p = Tensor(arrays["A"]).softmax(axis=-1).data
        assert np.allclose(p.sum(axis=-1), 1.0)

    def test_log_softmax_is_stable(self):
        out = Tensor([[1000.0, 999.0, 1.0]]).log_softmax().data
        assert np.all(np.isfinite(out))
        assert out[0, 0] == pytest.approx(-0.31326168751822286)

    def test_softmax_jacobian_matches_the_closed_form(self):
        r"""∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ), obtained here as a vector-Jacobian
        product with a one-hot seed."""
        z = Tensor([[2.0, 1.0, 0.1]])
        p = z.softmax()
        probs = p.data.ravel()

        seed = np.zeros((1, 3))
        seed[0, 0] = 1.0
        p.backward(seed)

        expected = probs[0] * (np.eye(3)[0] - probs)
        assert np.allclose(z.grad.ravel(), expected)

    def test_softmax_of_a_constant_function_has_zero_gradient(self):
        # softmax(x).sum() is identically 1, so its gradient is exactly 0.
        # A gradient check here compares two piles of round-off; this is the
        # documented reason such a test would 'fail' for no real reason.
        t = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        t.softmax().sum().backward()
        assert np.max(np.abs(t.grad)) < 1e-15

    def test_non_degenerate_softmax_gradcheck(self, arrays):
        weights = np.random.default_rng(3).normal(size=(3, 4))
        assert_gradcheck(
            lambda a: a.softmax() * Tensor(weights), [arrays["A"]], tol=1e-5
        )


# ======================================================================
# backward mechanics
# ======================================================================


class TestBackward:
    def test_non_scalar_needs_an_explicit_seed(self):
        with pytest.raises(RuntimeError, match="non-scalar"):
            Tensor([[1.0, 2.0]]).backward()

    def test_explicit_seed_computes_a_vector_jacobian_product(self):
        x = Tensor([[2.0, 3.0]])
        y = x * x  # yᵢ = xᵢ², so J = diag(2x)
        y.backward(np.array([[1.0, 10.0]]))
        assert x.grad.tolist() == [[4.0, 60.0]]  # [1*2*2, 10*2*3]

    def test_accumulates_like_the_scalar_engine(self):
        x = Tensor([[3.0]])
        loss = (x * x).sum()
        loss.backward()
        assert x.grad.item() == pytest.approx(6.0)
        loss.backward()
        assert x.grad.item() == pytest.approx(12.0)  # accumulated

    def test_zero_grad_walks_the_graph(self):
        x = Tensor([[3.0]])
        loss = (x * x).sum()
        loss.backward()
        loss.zero_grad()
        assert not x.grad.any()

    def test_reuses_the_scalar_topological_sort(self):
        """``graph.py`` was written for ``Value`` and works on ``Tensor``
        unmodified -- because it only ever needed ``_prev``. That reuse is the
        proof the two engines run the same algorithm."""
        a, b = Tensor([[1.0, 2.0]]), Tensor([[3.0, 4.0]])
        loss = ((a * b).tanh() + a).sum()
        order = topological_sort(loss)
        assert is_topologically_sorted(order)
        assert order[-1] is loss
        assert graph_size(loss)["nodes"] == len(order)

    def test_diamond_accumulates(self):
        x = Tensor([[2.0]])
        ((x * 3.0) + (x * 5.0)).sum().backward()
        assert x.grad.item() == pytest.approx(8.0)


# ======================================================================
# THE HEADLINE TEST
# ======================================================================


class TestBothEnginesAgree:
    """Same network, same weights, two engines -- identical gradients.

    If this passes, the tensor engine is not an approximation or a rewrite. It
    is the same reverse-mode algorithm applied at a coarser granularity, and
    every derivation from the scalar phase carries over unchanged.
    """

    @staticmethod
    def _build_matched_pair(n_in=6, hidden=(5, 4), n_out=3, activation="tanh", seed=7):
        from nabla.nn import MLP
        from nabla.nn.tensor_mlp import TensorMLP

        scalar = MLP(n_in, list(hidden), n_out, activation=activation, seed=seed)
        tensor = TensorMLP(n_in, list(hidden), n_out, activation=activation, seed=seed)

        # Copy the scalar model's weights into the tensor model.
        for sl, tl in zip(scalar.layers, tensor.layers):
            tl.W.data = np.array(
                [[w.data for w in n.weights] for n in sl.neurons]
            ).T.copy()
            tl.b.data = np.array([n.bias.data for n in sl.neurons]).copy()
        return scalar, tensor

    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
    def test_forward_and_backward_match(self, activation):
        from nabla.losses import softmax_cross_entropy
        from nabla.losses.tensor_losses import tensor_cross_entropy

        scalar, tensor = self._build_matched_pair(activation=activation)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 6))
        Y = rng.integers(0, 3, size=8).tolist()

        s_loss = softmax_cross_entropy([scalar(x.tolist()) for x in X], Y)
        t_loss = tensor_cross_entropy(tensor(X), Y)
        assert s_loss.data == pytest.approx(t_loss.item(), abs=1e-12)

        scalar.zero_grad()
        tensor.zero_grad()
        s_loss.backward()
        t_loss.backward()

        for sl, tl in zip(scalar.layers, tensor.layers):
            sW = np.array([[w.grad for w in n.weights] for n in sl.neurons]).T
            sb = np.array([n.bias.grad for n in sl.neurons])
            assert np.max(np.abs(sW - tl.W.grad)) < 1e-14
            assert np.max(np.abs(sb - tl.b.grad)) < 1e-14

    def test_tensor_graph_is_orders_of_magnitude_smaller(self):
        from nabla.losses import softmax_cross_entropy
        from nabla.losses.tensor_losses import tensor_cross_entropy

        scalar, tensor = self._build_matched_pair()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 6))
        Y = rng.integers(0, 3, size=8).tolist()

        s_nodes = graph_size(
            softmax_cross_entropy([scalar(x.tolist()) for x in X], Y)
        )["nodes"]
        t_nodes = graph_size(tensor_cross_entropy(tensor(X), Y))["nodes"]

        assert t_nodes < 50
        assert s_nodes > 20 * t_nodes

    def test_mse_also_matches(self):
        from nabla.losses import mse_loss
        from nabla.losses.tensor_losses import tensor_mse

        scalar, tensor = self._build_matched_pair(n_out=3, activation="tanh")
        x = [0.4, -0.2, 0.9, 1.1, -0.6, 0.3]
        targets = [0.5, -0.25, 0.75]

        s_loss = mse_loss(scalar(x), targets)
        t_loss = tensor_mse(tensor(np.array([x])), np.array([targets]))
        assert s_loss.data == pytest.approx(t_loss.item(), abs=1e-12)


class TestTensorModel:
    def test_parameter_count_matches_the_scalar_model(self):
        from nabla.nn import MLP
        from nabla.nn.tensor_mlp import TensorMLP

        assert (
            TensorMLP(784, [128, 64], 10, seed=0).num_parameters()
            == MLP(784, [128, 64], 10, seed=0).num_parameters()
            == 109_386
        )

    def test_forward_shapes(self):
        from nabla.nn.tensor_mlp import TensorMLP

        model = TensorMLP(4, [6], 3, seed=0)
        assert model(np.zeros((16, 4))).shape == (16, 3)
        assert model(np.zeros(4)).shape == (1, 3)  # single sample promoted

    def test_predict_and_proba(self):
        from nabla.nn.tensor_mlp import TensorMLP

        model = TensorMLP(4, [6], 3, seed=0)
        x = np.random.default_rng(0).normal(size=(5, 4))
        assert model.predict(x).shape == (5,)
        probs = model.predict_proba(x)
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_state_dict_round_trip(self):
        from nabla.nn.tensor_mlp import TensorMLP

        a = TensorMLP(4, [6], 3, seed=0)
        b = TensorMLP(4, [6], 3, seed=99)
        x = np.random.default_rng(0).normal(size=(3, 4))
        assert not np.allclose(a.predict_proba(x), b.predict_proba(x))

        b.load_state_dict(a.state_dict())
        assert np.allclose(a.predict_proba(x), b.predict_proba(x))

    def test_state_dict_is_json_serialisable(self):
        import json

        from nabla.nn.tensor_mlp import TensorMLP

        state = TensorMLP(4, [6], 3, seed=0).state_dict()
        assert json.loads(json.dumps(state)).keys() == state.keys()

    def test_shape_mismatch_on_load_raises(self):
        from nabla.nn.tensor_mlp import TensorMLP

        a = TensorMLP(4, [6], 3, seed=0)
        b = TensorMLP(4, [8], 3, seed=0)
        with pytest.raises((ValueError, KeyError)):
            b.load_state_dict(a.state_dict())

    def test_optimisers_work_on_tensor_parameters(self):
        from nabla.losses.tensor_losses import tensor_cross_entropy
        from nabla.nn.tensor_mlp import TensorMLP
        from nabla.optim import SGD, Adam

        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 4))
        Y = (X[:, 0] > 0).astype(int).tolist()

        for build in (lambda p: SGD(p, lr=0.1, momentum=0.9), lambda p: Adam(p, lr=0.05)):
            model = TensorMLP(4, [8], 2, activation="relu", seed=0)
            opt = build(model.parameters())
            first = last = None
            for _ in range(60):
                opt.zero_grad()
                loss = tensor_cross_entropy(model(X), Y)
                loss.backward()
                opt.step()
                last = loss.item()
                first = first if first is not None else last
            assert last < first * 0.5

    def test_gradient_norm_counts_every_scalar(self):
        from nabla.nn.tensor_mlp import TensorMLP
        from nabla.optim import SGD

        model = TensorMLP(4, [6], 3, seed=0)
        opt = SGD(model.parameters(), lr=0.1)
        assert opt._scalar_count() == model.num_parameters()
