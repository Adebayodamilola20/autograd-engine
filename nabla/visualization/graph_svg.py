"""Phase 6 -- rendering a computational graph as a self-contained SVG.

Why write a renderer instead of shelling out to Graphviz
--------------------------------------------------------
Three reasons, in order of importance:

1. **Layered layout is the lesson.** Assigning each node to a layer by its
   longest path from the leaves *is* the depth computation the backward pass
   depends on. Implementing it makes the graph's structure concrete rather
   than delegating it to a black box.
2. **No system dependency.** Graphviz needs a native ``dot`` binary. A reader
   who clones this repo can render graphs with nothing but Python.
3. **The output is reusable.** Pure SVG with no external references embeds
   directly in the docs (Phase 15) and in the React demo (Phase 14).

``graph_dot.py`` still exports Graphviz DOT for anyone who prefers it.

The layout algorithm
--------------------
A Sugiyama-style layered drawing, simplified for DAGs that are already
topologically sorted:

1. **Layer assignment** -- ``layer(v) = 1 + max(layer(p) for p in parents)``,
   computed in topological order so every parent is known. Leaves land in
   layer 0. This is exactly the "depth" from ``graph_size()``.
2. **Ordering within a layer** -- place each node near the average position of
   its parents (the barycentre heuristic), then break ties by insertion order.
   One pass is enough for the small graphs we draw and keeps edge crossings
   low without an iterative optimiser.
3. **Coordinates** -- layers become columns (left to right, matching the
   direction of the forward pass), nodes become rows within a column.
4. **Edges** -- cubic Béziers from the right edge of the parent to the left
   edge of the child, so crossings stay readable.

Each node shows its label, operation, ``data`` and ``grad`` -- the four things
you need to follow a backward pass by eye.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.graph import topological_sort

__all__ = ["render_svg", "graph_to_svg"]

# ---------------------------------------------------------------- geometry
NODE_W = 132
NODE_H = 54
OP_R = 15  # radius of the little operation circle
H_GAP = 74  # horizontal space between a node column and the next op column
V_GAP = 18
MARGIN = 28

# ---------------------------------------------------------------- palette
# Chosen to stay legible in both light and dark viewers: the SVG carries its
# own background, and a prefers-color-scheme block swaps the whole scheme.
THEME = {
    "light": {
        "bg": "#fbfbfd",
        "grid": "#eceef3",
        "node": "#ffffff",
        "node_stroke": "#c9cedb",
        "leaf": "#eef4ff",
        "leaf_stroke": "#8fb0e8",
        "root": "#fff3e0",
        "root_stroke": "#e8a33d",
        "op": "#f3f4f8",
        "op_stroke": "#9aa2b5",
        "edge": "#aab1c2",
        "text": "#1a1d24",
        "muted": "#6b7385",
        "data": "#1c4f9c",
        "grad": "#b3401f",
    },
    "dark": {
        "bg": "#14161c",
        "grid": "#1e212a",
        "node": "#1c1f28",
        "node_stroke": "#333949",
        "leaf": "#152232",
        "leaf_stroke": "#3f6ba8",
        "root": "#2b2113",
        "root_stroke": "#a8762b",
        "op": "#232733",
        "op_stroke": "#4a5266",
        "edge": "#404757",
        "text": "#e8eaf0",
        "muted": "#8b93a6",
        "data": "#7fb0ff",
        "grad": "#ff9a76",
    },
}


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _fmt(x: float) -> str:
    """Format a number compactly but without lying about its magnitude."""
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4g}"


# ====================================================================
# layout
# ====================================================================


def _assign_layers(nodes: Sequence[Any]) -> dict[int, int]:
    """``layer(v) = 1 + max(layer(parents))``, leaves at 0.

    Valid in one pass precisely because ``nodes`` is topologically sorted --
    the same guarantee the backward pass relies on, used here for drawing.
    """
    layer: dict[int, int] = {}
    for node in nodes:
        layer[id(node)] = (
            0 if not node._prev else 1 + max(layer[id(p)] for p in node._prev)
        )
    return layer


def _order_within_layers(
    nodes: Sequence[Any], layer: dict[int, int]
) -> dict[int, list[Any]]:
    """Group by layer, then sort each layer by its parents' mean row.

    The barycentre heuristic: a node drawn near the average height of the nodes
    feeding it produces far fewer edge crossings than insertion order alone.
    """
    columns: dict[int, list[Any]] = {}
    for node in nodes:
        columns.setdefault(layer[id(node)], []).append(node)

    row: dict[int, float] = {}
    for depth in sorted(columns):
        column = columns[depth]
        if depth > 0:
            column.sort(
                key=lambda n: (
                    sum(row.get(id(p), 0.0) for p in n._prev) / max(len(n._prev), 1)
                    if n._prev
                    else 0.0
                )
            )
        for i, node in enumerate(column):
            row[id(node)] = float(i)
    return columns


# ====================================================================
# rendering
# ====================================================================


def graph_to_svg(
    root: Any,
    *,
    title: str = "",
    show_grad: bool = True,
    max_nodes: int = 300,
) -> str:
    """Render the graph rooted at ``root`` as a standalone SVG string.

    Parameters
    ----------
    root
        Any node exposing ``_prev``, ``_op``, ``label``, ``data``, ``grad`` --
        so this works for ``Value`` and, unchanged, for ``Tensor``.
    show_grad
        Include the gradient row. Turn it off to draw a graph before
        ``backward()`` has run, when every gradient is still 0.
    max_nodes
        Refuse to draw graphs beyond this size. A 784-input neuron has 3,138
        nodes; the resulting picture would be unreadable, and the useful
        response is a clear error rather than a 40 MB file.

    Returns
    -------
    str
        A complete ``<svg>`` document: no external fonts, no scripts, no CSS
        links. Safe to inline in Markdown, HTML or a React component.
    """
    nodes = topological_sort(root)
    if len(nodes) > max_nodes:
        raise ValueError(
            f"graph has {len(nodes)} nodes, above max_nodes={max_nodes}. "
            "Rendering it would be unreadable -- visualise a smaller "
            "expression, or raise max_nodes deliberately."
        )

    layer = _assign_layers(nodes)
    columns = _order_within_layers(nodes, layer)
    depth = max(columns) if columns else 0

    # ---- coordinates -------------------------------------------------
    col_pitch = NODE_W + H_GAP
    row_pitch = NODE_H + V_GAP
    tallest = max((len(c) for c in columns.values()), default=1)

    pos: dict[int, tuple[float, float]] = {}
    for d, column in columns.items():
        # Centre each column vertically against the tallest one.
        offset = (tallest - len(column)) * row_pitch / 2.0
        for i, node in enumerate(column):
            pos[id(node)] = (
                MARGIN + d * col_pitch,
                MARGIN + offset + i * row_pitch,
            )

    width = MARGIN * 2 + (depth + 1) * NODE_W + depth * H_GAP
    height = MARGIN * 2 + tallest * row_pitch - V_GAP
    if title:
        height += 34

    top = 34 if title else 0
    parts: list[str] = []

    # ---- document + theme -------------------------------------------
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} '
        f'{height:.0f}" width="{width:.0f}" height="{height:.0f}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
        f'role="img" aria-label="computational graph">'
    )
    light, dark = THEME["light"], THEME["dark"]
    css_vars = "".join(f"--{k}:{v};" for k, v in light.items())
    css_dark = "".join(f"--{k}:{v};" for k, v in dark.items())
    parts.append(
        "<style>"
        f":root{{{css_vars}}}"
        f"@media (prefers-color-scheme: dark){{:root{{{css_dark}}}}}"
        ".bg{fill:var(--bg)}"
        ".card{fill:var(--node);stroke:var(--node_stroke);stroke-width:1.25}"
        ".card.leaf{fill:var(--leaf);stroke:var(--leaf_stroke)}"
        ".card.root{fill:var(--root);stroke:var(--root_stroke);stroke-width:2}"
        ".op{fill:var(--op);stroke:var(--op_stroke);stroke-width:1.25}"
        ".edge{stroke:var(--edge);stroke-width:1.5;fill:none}"
        ".lbl{fill:var(--text);font-size:13px;font-weight:600}"
        ".sub{fill:var(--muted);font-size:10px;letter-spacing:.06em}"
        ".data{fill:var(--data);font-size:11.5px}"
        ".grad{fill:var(--grad);font-size:11.5px}"
        ".opt{fill:var(--text);font-size:12px;font-weight:700}"
        ".title{fill:var(--text);font-size:14px;font-weight:700}"
        "</style>"
    )
    parts.append(f'<rect class="bg" width="{width:.0f}" height="{height:.0f}"/>')
    if title:
        parts.append(f'<text class="title" x="{MARGIN}" y="22">{_esc(title)}</text>')

    # ---- edges (drawn first, so nodes sit on top) --------------------
    parts.append('<g class="edges">')
    for node in nodes:
        if not node._prev:
            continue
        cx, cy = pos[id(node)]
        cy += top
        for parent in node._prev:
            px, py = pos[id(parent)]
            py += top
            x1, y1 = px + NODE_W, py + NODE_H / 2
            x2, y2 = cx - OP_R * 2 - 4, cy + NODE_H / 2
            mid = (x1 + x2) / 2
            parts.append(
                f'<path class="edge" d="M{x1:.1f},{y1:.1f} '
                f'C{mid:.1f},{y1:.1f} {mid:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
            )
    parts.append("</g>")

    # ---- operation bubbles -------------------------------------------
    parts.append('<g class="ops">')
    for node in nodes:
        if not node._op:
            continue
        x, y = pos[id(node)]
        y += top
        ox, oy = x - OP_R - 4, y + NODE_H / 2
        parts.append(f'<circle class="op" cx="{ox:.1f}" cy="{oy:.1f}" r="{OP_R}"/>')
        symbol = node._op if len(node._op) <= 4 else node._op[:4]
        parts.append(
            f'<text class="opt" x="{ox:.1f}" y="{oy + 4:.1f}" '
            f'text-anchor="middle">{_esc(symbol)}</text>'
        )
    parts.append("</g>")

    # ---- node cards ---------------------------------------------------
    parts.append('<g class="nodes">')
    for node in nodes:
        x, y = pos[id(node)]
        y += top
        kind = "leaf" if not node._prev else ""
        if node is root:
            kind = "root"
        parts.append(
            f'<rect class="card {kind}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{NODE_W}" height="{NODE_H}" rx="8"/>'
        )

        name = node.label or (node._op or "const")
        parts.append(f'<text class="lbl" x="{x + 10:.1f}" y="{y + 17:.1f}">{_esc(name)}</text>')
        if node._op and node.label:
            parts.append(
                f'<text class="sub" x="{x + NODE_W - 10:.1f}" y="{y + 17:.1f}" '
                f'text-anchor="end">{_esc(node._op)}</text>'
            )

        parts.append(
            f'<text class="data" x="{x + 10:.1f}" y="{y + 33:.1f}">'
            f'data {_esc(_fmt(node.data))}</text>'
        )
        if show_grad:
            parts.append(
                f'<text class="grad" x="{x + 10:.1f}" y="{y + 47:.1f}">'
                f'grad {_esc(_fmt(node.grad))}</text>'
            )
    parts.append("</g>")

    parts.append("</svg>")
    return "".join(parts)


def render_svg(
    root: Any,
    path: str | Path | None = None,
    *,
    title: str = "",
    show_grad: bool = True,
    max_nodes: int = 300,
) -> str:
    """Render the graph and optionally write it to ``path``.

    >>> from nabla import Value
    >>> x, w = Value(2.0, label='x'), Value(-3.0, label='w')
    >>> L = (x * w + Value(1.0, label='b')).tanh()
    >>> L.backward()
    >>> svg = render_svg(L, 'artifacts/neuron.svg', title='L = tanh(xw + b)')
    """
    svg = graph_to_svg(root, title=title, show_grad=show_grad, max_nodes=max_nodes)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")
    return svg
