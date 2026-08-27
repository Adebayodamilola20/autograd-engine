"""Phase 2 -- the ``Value`` container itself.

Not the maths (that is ``test_operations`` / ``test_backward``), but the object:
construction, coercion, graph metadata, and the conveniences.
"""

from __future__ import annotations

import pytest

from nabla import Value


class TestConstruction:
    def test_stores_data_as_float(self):
        v = Value(3)
        assert v.data == 3.0
        assert isinstance(v.data, float)

    def test_grad_starts_at_zero(self):
        # A gradient is only meaningful relative to some output. Before
        # backward() has named one, 0.0 is the only honest answer.
        assert Value(3.0).grad == 0.0

    def test_leaf_has_no_parents_or_op(self):
        v = Value(3.0)
        assert v._prev == ()
        assert v._op == ""
        assert v.is_leaf

    def test_result_of_an_op_is_not_a_leaf(self):
        out = Value(1.0) + Value(2.0)
        assert not out.is_leaf
        assert out._op == "+"
        assert len(out._prev) == 2

    def test_label_is_preserved(self):
        assert Value(1.0, label="x").label == "x"

    def test_ids_are_unique_and_increasing(self):
        a, b = Value(1.0), Value(2.0)
        assert b._id > a._id

    def test_repr_shows_data_and_grad(self):
        v = Value(2.5, label="a")
        text = repr(v)
        assert "a=" in text and "2.5" in text

    def test_slots_prevent_stray_attributes(self):
        # __slots__ is a deliberate memory optimisation (see D7 in
        # docs/00-architecture.md); this pins the behaviour it implies.
        with pytest.raises(AttributeError):
            Value(1.0).some_new_attribute = 1  # type: ignore[attr-defined]


class TestCoercion:
    """Plain numbers are wrapped into constant nodes so ops stay uniform."""

    @pytest.mark.parametrize("scalar", [2, 2.0, -1, 0])
    def test_scalars_are_accepted(self, scalar):
        assert (Value(1.0) + scalar).data == 1.0 + scalar

    def test_reflected_operators_work(self):
        x = Value(4.0)
        assert (2 + x).data == 6.0
        assert (2 * x).data == 8.0
        assert (2 - x).data == -2.0
        assert (2 / x).data == 0.5

    def test_rejects_unsupported_types(self):
        with pytest.raises(TypeError):
            Value(1.0) + "two"  # type: ignore[operator]
        with pytest.raises(TypeError):
            Value(1.0) * None  # type: ignore[operator]

    def test_rejects_bool(self):
        # bool is a subclass of int in Python; silently treating True as 1.0
        # hides real bugs, so it is rejected explicitly.
        with pytest.raises(TypeError):
            Value(1.0) + True  # type: ignore[operator]


class TestConveniences:
    def test_comparisons_use_data(self):
        a, b = Value(1.0), Value(2.0)
        assert a < b and b > a and a <= b and b >= a
        assert a < 2.0 and a >= 1.0

    def test_equality_stays_identity_based(self):
        # Defining __eq__ would break hashing, and graph traversal relies on
        # identity (two nodes holding 2.0 are different nodes).
        a, b = Value(2.0), Value(2.0)
        assert a != b
        assert len({a, b}) == 2

    def test_float_and_item(self):
        v = Value(1.5)
        assert float(v) == 1.5
        assert v.item() == 1.5

    def test_sum_helper(self):
        vals = [Value(float(i)) for i in range(5)]
        total = Value.sum(vals)
        assert total.data == 10.0
        total.backward()
        # Addition routes gradient unchanged, so every term gets exactly 1.
        assert all(v.grad == 1.0 for v in vals)

    def test_sum_of_empty(self):
        assert Value.sum([]).data == 0.0

    def test_builtin_sum_also_works(self):
        # builtin sum() starts at int 0, which __radd__ handles.
        total = sum([Value(1.0), Value(2.0)])
        assert total.data == 3.0


class TestGraphMetadata:
    def test_op_labels_are_recorded(self):
        x = Value(2.0)
        assert (x + x)._op == "+"
        assert (x * x)._op == "*"
        assert (x**2)._op == "**2"
        assert x.exp()._op == "exp"
        assert x.log()._op == "log"
        assert x.tanh()._op == "tanh"
        assert x.sigmoid()._op == "sigmoid"
        assert x.relu()._op == "relu"

    def test_parents_are_the_actual_operands(self):
        a, b = Value(1.0), Value(2.0)
        out = a * b
        assert out._prev[0] is a
        assert out._prev[1] is b

    def test_reused_input_appears_twice_in_prev(self):
        # This is the structural fact that makes gradient accumulation
        # necessary -- one node, two incoming edges.
        x = Value(3.0)
        out = x * x
        assert out._prev == (x, x)

    def test_zero_grad_clears_whole_graph(self):
        x = Value(2.0)
        y = (x * x + x).tanh()
        y.backward()
        assert x.grad != 0.0
        y.zero_grad()
        assert x.grad == 0.0
        assert all(n.grad == 0.0 for n in [x, y])
