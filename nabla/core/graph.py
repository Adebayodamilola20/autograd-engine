"""Computational-graph traversal.

The autodiff *node* (``Value``) knows how to compute one local derivative.
This module knows how to walk the DAG those nodes form. Keeping the two apart
means the traversal code is written once and reused unchanged by the scalar
``Value`` engine (Phase 2) and the ``Tensor`` engine (Phase 18) -- both expose a
``_prev`` tuple of parents, and that is the only thing anything here requires.

Why the ordering matters
------------------------
A node may only push gradient to its parents once *its own* gradient is final,
and its gradient is final only after every node that consumed it has reported
in. So the backward pass must satisfy:

    process a node only after all of its children (consumers) are processed.

A topological sort of a DAG orders every node after all of its *parents*.
Reversed, it orders every node after all of its *children* -- exactly the
condition above. Hence::

    backward pass == reversed(topological_sort(root))

See docs/01-what-is-automatic-differentiation.md section 11.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .value import Value

__all__ = [
    "topological_sort",
    "build_edges",
    "iter_nodes",
    "leaves",
    "graph_size",
    "is_topologically_sorted",
    "format_graph",
]


def topological_sort(root: Any) -> list[Any]:
    """Return every node reachable from ``root``, parents before children.

    Guarantee
    ---------
    For every edge ``u -> v`` in the graph (i.e. ``u in v._prev``), ``u``
    appears strictly before ``v`` in the returned list. ``root`` is always last.

    Algorithm
    ---------
    Depth-first search emitted in *post-order*: a node is appended only after
    its entire ancestry has been appended. A ``visited`` set makes each node
    appear exactly once -- without it a diamond-shaped graph would be walked
    exponentially many times.

    Why this is written iteratively
    -------------------------------
    The textbook version recurses. CPython's default recursion limit is ~1000
    frames, and a scalar engine produces graphs *far* deeper than that (a
    single 128-neuron layer already chains ~256 nodes for one neuron's dot
    product). We use an explicit stack instead, so depth is bounded by heap
    memory rather than the C stack. ``tests/test_graph.py`` differentiates a
    50,000-node chain to prove it.

    The stack holds ``(node, expanded)`` pairs. ``expanded=False`` means
    "descend into this node's parents"; ``expanded=True`` is the marker pushed
    underneath them, popped only once they are all done, at which point the
    node is safe to emit.

    Complexity
    ----------
    O(V + E) time, O(V) space.

    Notes
    -----
    No cycle detection is performed, and none is needed: an edge is created
    only when a node is *constructed* from parents that already exist, so an
    edge can never point backwards in time. The DAG property is guaranteed by
    construction.
    """
    order: list[Any] = []
    visited: set[int] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()

        if expanded:
            # All parents have been emitted; this node is now safe to emit.
            order.append(node)
            continue

        if id(node) in visited:
            # Reached again through another path -- already scheduled.
            continue
        visited.add(id(node))

        # Marker first, so it is popped *after* everything pushed below it.
        stack.append((node, True))
        for parent in node._prev:
            if id(parent) not in visited:
                stack.append((parent, False))

    return order


def iter_nodes(root: Any) -> Iterator[Any]:
    """Yield every node reachable from ``root``, in topological order."""
    yield from topological_sort(root)


def build_edges(root: Any) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Return ``(nodes, edges)`` for the graph rooted at ``root``.

    ``edges`` contains ``(parent, child)`` pairs, oriented in the direction data
    flowed on the forward pass. Used by the visualisation layer (Phase 6).
    """
    nodes = topological_sort(root)
    edges: list[tuple[Any, Any]] = []
    for node in nodes:
        for parent in node._prev:
            edges.append((parent, node))
    return nodes, edges


def leaves(root: Any) -> list[Any]:
    """Return the leaf nodes -- those with no parents.

    Leaves are the inputs and the learnable parameters: the nodes whose
    gradients the optimiser actually consumes. Every other node's gradient is a
    stepping stone that is discarded with the graph.
    """
    return [node for node in topological_sort(root) if not node._prev]


def graph_size(root: Any) -> dict[str, int]:
    """Summary statistics for the graph rooted at ``root``.

    Returns counts of nodes, edges, leaves and the longest path length (the
    depth the backward pass must traverse). Useful for the benchmarking and
    performance phases, where "how big is one forward pass?" is the question
    that explains the runtime.
    """
    nodes, edges = build_edges(root)

    # Longest path to each node, computed in topological order so every
    # parent's depth is known before it is needed.
    depth: dict[int, int] = {}
    max_depth = 0
    for node in nodes:
        d = 0
        for parent in node._prev:
            d = max(d, depth[id(parent)] + 1)
        depth[id(node)] = d
        max_depth = max(max_depth, d)

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "leaves": sum(1 for n in nodes if not n._prev),
        "depth": max_depth,
    }


def is_topologically_sorted(order: Sequence[Any]) -> bool:
    """True if every node in ``order`` appears after all of its parents.

    This is the property the backward pass depends on, so it is asserted
    directly in the test suite rather than inferred from the results.
    """
    seen: set[int] = set()
    for node in order:
        for parent in node._prev:
            if id(parent) not in seen:
                return False
        seen.add(id(node))
    return True


def format_graph(root: Any, *, max_nodes: int = 50) -> str:
    """A plain-text dump of the graph, in topological order.

    The zero-dependency debugging tool: readable in a terminal, in a test
    failure message, or in a notebook, long before the SVG renderer exists.
    """
    nodes = topological_sort(root)
    lines = [
        f"computational graph: {len(nodes)} nodes, "
        f"depth {graph_size(root)['depth']}",
        f"{'idx':>4}  {'label':<10} {'op':<8} {'data':>14} {'grad':>14}  inputs",
        "-" * 76,
    ]
    index = {id(n): i for i, n in enumerate(nodes)}
    for i, node in enumerate(nodes):
        if i >= max_nodes:
            lines.append(f"... {len(nodes) - max_nodes} more nodes")
            break
        parents = ", ".join(f"#{index[id(p)]}" for p in node._prev) or "-"
        lines.append(
            f"{i:>4}  {node.label or '':<10} {node._op or 'leaf':<8} "
            f"{node.data:>14.6g} {node.grad:>14.6g}  {parents}"
        )
    return "\n".join(lines)
