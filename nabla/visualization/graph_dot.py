"""Phase 6 -- optional Graphviz DOT export.

``graph_svg.py`` renders without any system dependency, which is what the
project uses by default. This module exists for readers who already have
Graphviz and want its layout engine, which handles large or awkward graphs
better than our simple layered algorithm.

Produces DOT text; rendering it is up to the caller::

    dot -Tpng graph.dot -o graph.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.graph import build_edges

__all__ = ["to_dot", "render_dot"]


def _escape_quoted(text: str) -> str:
    r"""Escape for an ordinary DOT quoted string.

    Only ``\`` and ``"`` are special, and the backslash must go first or the
    backslashes added for the quotes get escaped in turn.

    Not ``html.escape``. That is a different language: it produces ``&quot;``
    and ``&lt;``, which Graphviz has no reason to interpret and draws
    literally, so a label reading ``a<b`` would render as ``a&lt;b``.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _escape_record(text: str) -> str:
    r"""Escape for a DOT **record** label, which has extra syntax.

    Inside ``shape=record`` these all mean something structural:

    ======  ====================================================
    ``|``   field separator
    ``{}``  nest a group, flipping the layout direction
    ``<>``  a port name, for edges that attach to one field
    ``"``   ends the quoted string
    ``\``   escape character
    ======  ====================================================

    Only ``|`` was escaped before, so a label containing a quote closed the
    string early and produced a file Graphviz refuses to parse, while braces
    and angle brackets silently restructured the box. Labels come from
    ``Value.label``, which is user-supplied, so none of these are exotic.
    """
    out = text.replace("\\", "\\\\")
    for char in '"|{}<>':
        out = out.replace(char, "\\" + char)
    return out


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4g}"


def to_dot(root: Any, *, title: str = "", show_grad: bool = True) -> str:
    """Return Graphviz DOT source for the graph rooted at ``root``.

    Values become record-shaped boxes carrying label / data / grad; operations
    become small ellipses on the edge into their output, which is the
    convention micrograd-style diagrams use and which keeps the picture
    readable.
    """
    nodes, edges = build_edges(root)
    uid = {id(n): f"n{i}" for i, n in enumerate(nodes)}

    lines = ["digraph computational_graph {", "  rankdir=LR;"]
    if title:
        lines += [
            f'  label="{_escape_quoted(title)}";',
            "  labelloc=t;",
            "  fontsize=16;",
        ]
    lines += [
        '  node [fontname="monospace", fontsize=10];',
        '  edge [color="#888888"];',
    ]

    for node in nodes:
        name = node.label or (node._op or "const")
        fields = [name, f"data {_fmt(node.data)}"]
        if show_grad:
            fields.append(f"grad {_fmt(node.grad)}")
        record = " | ".join(_escape_record(f) for f in fields)
        fill = "#eef4ff" if not node._prev else ("#fff3e0" if node is root else "#ffffff")
        lines.append(
            f'  {uid[id(node)]} [shape=record, style=filled, fillcolor="{fill}", '
            f'label="{{{record}}}"];'
        )

        if node._op:
            op_id = f"{uid[id(node)]}op"
            lines.append(
                f'  {op_id} [shape=ellipse, style=filled, fillcolor="#f3f4f8", '
                f'label="{_escape_quoted(node._op)}"];'
            )
            lines.append(f"  {op_id} -> {uid[id(node)]};")

    for parent, child in edges:
        lines.append(f"  {uid[id(parent)]} -> {uid[id(child)]}op;")

    lines.append("}")
    return "\n".join(lines)


def render_dot(root: Any, path: str | Path | None = None, **kwargs: Any) -> str:
    """Build DOT source and optionally write it to ``path``."""
    dot = to_dot(root, **kwargs)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dot, encoding="utf-8")
    return dot
