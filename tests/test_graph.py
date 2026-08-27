"""Phase 3 -- the computational graph and its traversal.

``test_backward.py`` checks that gradients come out right. This file checks the
*structure* underneath them: that the graph is built correctly, that the
topological ordering actually has the property backpropagation relies on, and
that traversal scales.

The most important test here is ``TestWhyOrderingMatters``, which builds a
deliberately broken backward pass to prove that the ordering is load-bearing --
a test that passes for the wrong reason is worse than no test.
"""

from __future__ import annotations

import pytest

from nabla import Value
from nabla.core.graph import (
    build_edges,
    format_graph,
    graph_size,
    is_topologically_sorted,
    leaves,
    topological_sort,
)


@pytest.fixture
def running_example():
    """L = (x·w + b)² with x=2, w=-3, b=1 -- the example from the docs.

    Returns (L, x, w, b). Six nodes, five edges, depth 3.
    """
    x = Value(2.0, label="x")
    w = Value(-3.0, label="w")
    b = Value(1.0, label="b")
    u = x * w
    u.label = "u"
    v = u + b
    v.label = "v"
    L = v**2
    L.label = "L"
    return L, x, w, b


# ======================================================================
# Graph construction
# ======================================================================


class TestGraphConstruction:
    def test_every_operation_creates_an_edge(self, running_example):
        L, x, w, b = running_example
        nodes, edges = build_edges(L)
        assert len(nodes) == 6
        assert len(edges) == 5

    def test_edges_point_from_parent_to_child(self):
        a, b = Value(1.0), Value(2.0)
        out = a + b
        _, edges = build_edges(out)
        assert (a, out) in edges
        assert (b, out) in edges
        assert (out, a) not in edges

    def test_leaves_are_the_inputs(self, running_example):
        L, x, w, b = running_example
        found = leaves(L)
        assert len(found) == 3
        assert set(map(id, found)) == {id(x), id(w), id(b)}

    def test_graph_statistics(self, running_example):
        L, *_ = running_example
        stats = graph_size(L)
        assert stats == {"nodes": 6, "edges": 5, "leaves": 3, "depth": 3}

    def test_depth_of_a_chain_equals_its_length(self):
        x = Value(1.0)
        y = x
        for _ in range(10):
            y = y.tanh()
        assert graph_size(y)["depth"] == 10

    def test_a_reused_node_is_counted_once(self):
        """x·x has three nodes, not four -- one x, shared."""
        x = Value(3.0)
        out = x * x
        assert graph_size(out)["nodes"] == 2  # x and out
        assert graph_size(out)["edges"] == 2  # two edges from the same source

    def test_shared_subexpression_is_not_duplicated(self):
        r"""u is built once and consumed twice; the graph must reflect that.

            x ──> u = x² ──┬──> a = u + 1 ──┐
                           └──> b = u * 2 ──┴──> L = a + b
        """
        x = Value(2.0)
        u = x**2
        a = u + 1
        b = u * 2
        L = a + b
        nodes = topological_sort(L)
        # u must appear exactly once even though two nodes consume it.
        assert sum(1 for n in nodes if n is u) == 1

    def test_constants_become_real_leaf_nodes(self):
        x = Value(2.0)
        out = x * 3
        assert graph_size(out)["leaves"] == 2  # x, and the constant 3


# ======================================================================
# Topological ordering
# ======================================================================


class TestTopologicalSort:
    def test_returns_every_reachable_node_exactly_once(self, running_example):
        L, *_ = running_example
        order = topological_sort(L)
        ids = [id(n) for n in order]
        assert len(ids) == len(set(ids)) == 6

    def test_root_is_last(self, running_example):
        L, *_ = running_example
        assert topological_sort(L)[-1] is L

    def test_leaves_come_before_the_nodes_that_use_them(self, running_example):
        L, x, w, b = running_example
        order = topological_sort(L)
        position = {id(n): i for i, n in enumerate(order)}
        for node in order:
            for parent in node._prev:
                assert position[id(parent)] < position[id(node)]

    def test_property_holds_on_a_complicated_graph(self):
        # A tangle with reuse, branching and several depths.
        a, b = Value(1.5), Value(-0.5)
        c = a * b
        d = c.tanh()
        e = (d + a) * (c - b)
        f = (e**2 + d.exp()) / (a + 1.0)
        assert is_topologically_sorted(topological_sort(f))

    def test_detects_a_bad_ordering(self):
        """The validator must be able to fail, or it proves nothing."""
        x = Value(2.0)
        out = x * 3
        good = topological_sort(out)
        assert is_topologically_sorted(good)
        assert not is_topologically_sorted(list(reversed(good)))

    def test_diamond_is_visited_once_per_node(self):
        r"""Without a visited set, a chain of k diamonds costs 2^k visits.

              ┌─> b ─┐
          x ──┤      ├──> L
              └─> c ─┘
        """
        x = Value(1.0)
        node = x
        for _ in range(20):  # 2^20 = 1,048,576 paths if traversal were naive
            node = (node * 2) + (node * 3)
        order = topological_sort(node)
        # Linear in the number of nodes, not exponential in paths.
        assert len(order) < 200
        assert len(order) == len({id(n) for n in order})

    def test_disconnected_branch_is_excluded(self):
        # A value that does not contribute to the root is not in its graph.
        x, unused = Value(1.0), Value(9.0)
        out = x * 2
        assert all(n is not unused for n in topological_sort(out))

    def test_single_node_graph(self):
        x = Value(5.0)
        assert topological_sort(x) == [x]


# ======================================================================
# Why the ordering matters -- proved by breaking it
# ======================================================================


class TestWhyOrderingMatters:
    r"""Backpropagation is only correct because of the traversal order.

    The correctness condition is: *process a node only after every node that
    consumes it has been processed*. These tests violate it on purpose and show
    the gradient come out wrong -- which is the evidence that the ordering in
    ``Value.backward()`` is doing real work.
    """

    @staticmethod
    def _backward_in_given_order(root, order):
        """A stripped-down backward pass that uses whatever order it is handed."""
        for node in topological_sort(root):
            node.grad = 0.0
        root.grad = 1.0
        for node in order:
            node._backward()

    def test_correct_order_gives_the_right_answer(self):
        r"""L = (x+1)·(x+2) at x=3  ->  dL/dx = 2x + 3 = 9."""
        x = Value(3.0)
        b = x + 1.0
        c = x + 2.0
        L = b * c
        self._backward_in_given_order(L, list(reversed(topological_sort(L))))
        assert x.grad == pytest.approx(9.0)

    def test_processing_a_node_before_its_consumer_loses_gradient(self):
        r"""Same graph, but ``b`` is processed before ``L``.

        At that moment ``b.grad`` is still 0, so ``b`` pushes nothing to ``x``.
        Only ``c``'s path survives, and ``dL/dx`` comes out as ``c``'s
        contribution alone (= b = 4) instead of 9.
        """
        x = Value(3.0)
        b = x + 1.0   # 4
        c = x + 2.0   # 5
        L = b * c     # 20

        order = topological_sort(L)  # parents first -- exactly backwards
        broken = [b] + [n for n in reversed(order) if n is not b]
        self._backward_in_given_order(L, broken)

        assert x.grad != pytest.approx(9.0)
        assert x.grad == pytest.approx(4.0)  # one path silently missing

    def test_forward_order_produces_zero_gradient(self):
        """Walking the graph forwards propagates nothing at all: every node's
        gradient is still 0 when it is asked to push."""
        x = Value(3.0)
        L = (x + 1.0) * (x + 2.0)
        self._backward_in_given_order(L, topological_sort(L))
        assert x.grad == 0.0

    def test_the_engine_uses_the_correct_order(self):
        """And the real ``backward()`` gets it right."""
        x = Value(3.0)
        L = (x + 1.0) * (x + 2.0)
        L.backward()
        assert x.grad == pytest.approx(9.0)


# ======================================================================
# Scale
# ======================================================================


class TestTraversalScale:
    def test_fifty_thousand_node_chain(self):
        """Decision D5: iterative traversal, so depth is bounded by heap
        memory rather than CPython's ~1000-frame recursion limit.

        A recursive topological sort raises RecursionError well before here.
        """
        x = Value(1.0)
        y = x
        for _ in range(50_000):
            y = y + 1.0

        order = topological_sort(y)
        assert len(order) == 100_001  # x, plus a constant and a sum per link
        assert is_topologically_sorted(order)

        y.backward()
        assert x.grad == pytest.approx(1.0)

    def test_wide_graph(self):
        # 5000 leaves feeding one sum: wide rather than deep.
        vals = [Value(1.0) for _ in range(5000)]
        total = Value.sum(vals)
        total.backward()
        assert total.data == 5000.0
        assert all(v.grad == 1.0 for v in vals)


# ======================================================================
# Debug output
# ======================================================================


class TestFormatGraph:
    def test_includes_labels_ops_and_values(self, running_example):
        L, *_ = running_example
        L.backward()
        text = format_graph(L)
        assert "6 nodes" in text
        assert "depth 3" in text
        for token in ("x", "w", "b", "L", "*", "+", "**2"):
            assert token in text

    def test_truncates_long_graphs(self):
        x = Value(1.0)
        y = x
        for _ in range(100):
            y = y + 1.0
        text = format_graph(y, max_nodes=10)
        assert "more nodes" in text
        assert len(text.splitlines()) < 20
