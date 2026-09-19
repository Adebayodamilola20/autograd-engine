"""Tests for the SVG and DOT renderers.

This module had no test file, which is how the DOT escaping bug survived: the
output is only ever looked at as a picture, and a picture that fails to render
gets blamed on Graphviz rather than on the generator.

Both renderers take text straight from ``Value.label``, which is supplied by
the caller, so the escaping is the part worth testing. The two targets need
*different* escaping schemes, and confusing them was the original bug:

* SVG is XML. ``<``, ``>``, ``&`` and ``"`` become entities.
* DOT is not XML. ``\\`` and ``"`` take backslashes, and a ``record`` label
  additionally gives ``|``, ``{``, ``}``, ``<`` and ``>`` structural meaning.

Graphviz is not installed in CI, so validity is checked structurally rather
than by running ``dot``. The checks below (balanced quotes, correct field
count) are the ones that actually catch the bug that existed.
"""

from __future__ import annotations

import pytest

from nabla.core.value import Value
from nabla.visualization import graph_to_svg, render_dot, render_svg, to_dot


@pytest.fixture
def graph():
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    out = (x * y + x).tanh()
    out.label = "out"
    out.backward()
    return out


# ======================================================================
# SVG
# ======================================================================


class TestSvg:
    def test_produces_an_svg_document(self, graph):
        svg = graph_to_svg(graph)
        assert svg.lstrip().startswith("<svg")
        assert svg.rstrip().endswith("</svg>")

    def test_renders_a_single_leaf(self):
        """A graph with no edges is still a graph."""
        assert graph_to_svg(Value(5.0)).lstrip().startswith("<svg")

    def test_writes_a_file(self, graph, tmp_path):
        path = tmp_path / "g.svg"
        render_svg(graph, path)
        assert path.read_text().lstrip().startswith("<svg")

    @pytest.mark.parametrize(
        "label", ["<script>alert(1)</script>", 'a" onload="x', "a & b", "a<b>c"]
    )
    def test_labels_are_xml_escaped(self, label):
        """Labels are caller-supplied, so raw markup must not survive."""
        svg = graph_to_svg(Value(1.0, label=label) * 2.0)
        assert "<script>" not in svg
        assert 'onload="x' not in svg

    def test_ampersand_becomes_an_entity(self):
        svg = graph_to_svg(Value(1.0, label="a&b") * 2.0)
        assert "&amp;" in svg


# ======================================================================
# DOT
# ======================================================================


def _label_bodies(dot: str) -> list[str]:
    """The text inside each ``label="..."``, with escapes left intact."""
    out = []
    for line in dot.splitlines():
        if 'label="' not in line:
            continue
        body = line.split('label="', 1)[1]
        # Walk to the closing quote, skipping backslash-escaped ones.
        result, i = [], 0
        while i < len(body):
            if body[i] == "\\" and i + 1 < len(body):
                result.append(body[i:i + 2])
                i += 2
                continue
            if body[i] == '"':
                break
            result.append(body[i])
            i += 1
        out.append("".join(result))
    return out


class TestDot:
    def test_produces_a_digraph(self, graph):
        dot = to_dot(graph)
        assert dot.startswith("digraph")
        assert dot.rstrip().endswith("}")

    def test_writes_a_file(self, graph, tmp_path):
        path = tmp_path / "g.dot"
        render_dot(graph, path)
        assert path.read_text().startswith("digraph")

    @pytest.mark.parametrize("char", ['"', "|", "{", "}", "<", ">", "\\"])
    def test_every_record_metacharacter_is_escaped(self, char):
        r"""``|`` was escaped; the other six were not.

        A quote closed the DOT string early and produced a file Graphviz
        refuses to parse. Braces and angle brackets are structural inside a
        ``record`` label, so they silently restructured the box instead.
        """
        dot = to_dot(Value(1.0, label=f"a{char}b") * 2.0)
        body = next(b for b in _label_bodies(dot) if "data 1" in b)
        assert f"\\{char}" in body, f"{char!r} was not escaped"

    def test_a_quote_in_a_label_keeps_the_line_parseable(self):
        """The specific failure: an unescaped quote ends the string early.

        Counting unescaped quotes on the line is the check that would have
        caught it. Before the fix there were four, not two.
        """
        dot = to_dot(Value(1.0, label='a"b') * 2.0)
        line = next(ln for ln in dot.splitlines() if "data 1" in ln)

        unescaped, i = 0, 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == '"':
                unescaped += 1
            i += 1
        # Exactly one quoted string on the line: fillcolor="..." and
        # label="..." give two pairs.
        assert unescaped == 4, f"unbalanced quoting: {line}"

    def test_a_pipe_does_not_add_a_record_field(self):
        """An unescaped ``|`` would split the box into an extra field."""
        plain = to_dot(Value(1.0, label="ab") * 2.0)
        piped = to_dot(Value(1.0, label="a|b") * 2.0)

        def fields(dot):
            body = next(b for b in _label_bodies(dot) if "data 1" in b)
            # Split on separators that are not backslash-escaped.
            return len(body.replace("\\|", "").split("|"))

        assert fields(plain) == fields(piped)

    def test_the_title_is_escaped_but_not_html_escaped(self):
        """The title is a plain quoted string, not a record and not XML.

        ``html.escape`` was used here, which is a different language: it
        produces ``&quot;``, and Graphviz draws that literally rather than as
        a quote.
        """
        dot = to_dot(Value(1.0) * 2.0, title='my "graph"')
        title = next(ln for ln in dot.splitlines() if ln.strip().startswith("label="))
        assert '\\"graph\\"' in title
        assert "&quot;" not in title

    def test_backslash_is_escaped_first(self):
        r"""``a\b`` must become ``a\\b``, not be read as an escape sequence.

        Order matters: escaping quotes before backslashes would turn the
        backslashes added for the quotes into escaped backslashes.
        """
        body = next(
            b for b in _label_bodies(to_dot(Value(1.0, label="a\\b") * 2.0))
            if "data 1" in b
        )
        assert "a\\\\b" in body

    def test_ordinary_labels_are_untouched(self, graph):
        """Escaping must not add backslashes to labels that need none."""
        bodies = _label_bodies(to_dot(graph))
        plain = next(b for b in bodies if b.startswith("{x |"))
        assert "\\" not in plain
        assert plain.startswith("{x | data 2 | grad ")
