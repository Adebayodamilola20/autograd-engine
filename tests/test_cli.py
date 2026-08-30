"""Tests for the ``nabla`` command line interface.

The weight here is deliberately lopsided. ``TestRejects`` is the largest class
in the file because the expression parser is the only part of this repository
that turns a *string* into executable behaviour, and the shortest implementation
of it -- ``eval`` with an emptied ``__builtins__`` -- is a remote code execution
hole. Every other test here checks that a subcommand works; those check that a
category of input never works, which is the harder property and the one that
silently regresses.
"""

from __future__ import annotations

import json

import pytest

from nabla.cli.expr import (
    ExpressionError,
    build,
    evaluate,
    free_variables,
    parse_assignments,
)
from nabla.cli.main import main
from nabla.core.gradcheck import check_gradients
from nabla.core.value import Value

# ======================================================================
# the parser: what it must accept
# ======================================================================


class TestEvaluates:
    def test_matches_hand_computed_derivatives(self):
        """``d = x*y + x`` -- the example from the README's first code block."""
        out, env = evaluate("x*y + x", {"x": 2, "y": 3})
        out.backward()
        assert out.data == 8.0
        assert env["x"].grad == 4.0  # y + 1
        assert env["y"].grad == 2.0  # x

    def test_squared_affine(self):
        out, env = evaluate("(x*w + b)**2", {"x": 2, "w": -3, "b": 1})
        out.backward()
        assert out.data == 25.0
        assert env["x"].grad == pytest.approx(30.0)   # 2(xw+b)w
        assert env["w"].grad == pytest.approx(-20.0)  # 2(xw+b)x
        assert env["b"].grad == pytest.approx(-10.0)  # 2(xw+b)

    def test_negative_base_with_literal_exponent(self):
        """``(-5)**2`` must work.

        A literal exponent has to reach ``Value.__pow__`` as a plain number.
        Wrapped in a ``Value`` it selects the general ``a**b`` rule, whose
        derivative involves ``log(a)`` and therefore rejects a negative base --
        breaking ordinary arithmetic that every other tool handles.
        """
        out, env = evaluate("x**2", {"x": -5})
        out.backward()
        assert out.data == 25.0
        assert env["x"].grad == pytest.approx(-10.0)

    @pytest.mark.parametrize("fn", ["exp", "log", "tanh", "sigmoid", "relu"])
    def test_every_exposed_function_is_differentiable(self, fn):
        """Whatever we advertise in ``--help`` must actually work."""
        out, env = evaluate(f"{fn}(x)", {"x": 0.6})
        out.backward()
        assert env["x"].grad != 0.0

    @pytest.mark.parametrize(
        "expression,point",
        [
            ("x*y + x", {"x": 2.0, "y": 3.0}),
            ("(x*w + b)**2", {"x": 2.0, "w": -3.0, "b": 1.0}),
            ("tanh(x*w + b)", {"x": 0.5, "w": 1.5, "b": -0.2}),
            ("exp(x)/(exp(x)+1)", {"x": 0.7}),
            ("log(x*x + 1)", {"x": 1.3}),
        ],
    )
    def test_agrees_with_finite_differences(self, expression, point):
        """The parser must not quietly build the *wrong* graph.

        Every test above asserts a derivative the author also derived, which is
        circular. This one compares against finite differences, which share no
        reasoning with the analytic rules.
        """
        names = list(point)
        result = check_gradients(
            lambda *vals: build(expression, dict(zip(names, vals))),
            [point[n] for n in names],
        )
        assert result.passed, f"{expression}: max rel err {result.max_error:.2e}"

    def test_unary_minus_and_precedence(self):
        out, _ = evaluate("-x + 2*y", {"x": 1, "y": 3})
        assert out.data == 5.0

    def test_build_uses_the_callers_values(self):
        """``build`` must not substitute fresh leaves.

        ``analytic_gradient`` creates the leaves itself and reads ``.grad`` back
        off *those* objects. A version that rebuilt them would report every
        analytic gradient as zero and every gradient check as a failure.
        """
        x = Value(2.0, label="x")
        out = build("x*x", {"x": x})
        out.backward()
        assert x.grad == pytest.approx(4.0)


# ======================================================================
# the parser: what it must refuse
# ======================================================================


class TestRejects:
    @pytest.mark.parametrize(
        "expression",
        [
            "().__class__.__bases__",          # the classic sandbox escape
            "x.__class__",
            "x.__class__.__mro__[1]",
            "(1).__class__",
            "x[0]",                            # subscripting
            "[i for i in range(3)]",           # comprehension
            "{1: 2}",
            "lambda: 1",
            "x if x else 1",                   # conditional expression
            "x and x",                         # boolean ops are not arithmetic
            "not x",
            "x < 1",                           # comparisons produce bools
            "f'{x}'",
            "(y := 2)",                        # walrus
            "x @ x",                           # matmul is not defined on Value
            "x | x",
            "x % 2",
            "...",
        ],
    )
    def test_non_arithmetic_syntax_is_refused(self, expression):
        """The walker raises on any node it does not recognise.

        Rejecting by *omission* rather than by blocklist is the point: a new
        Python grammar node cannot silently become reachable here.
        """
        with pytest.raises(ExpressionError):
            evaluate(expression, {"x": 1.0})

    @pytest.mark.parametrize(
        "expression",
        [
            '__import__("os").system("echo pwned")',
            'open("/etc/passwd")',
            "exec('x=1')",
            "eval('1')",
            "globals()",
            "print(x)",
        ],
    )
    def test_dangerous_calls_are_refused(self, expression):
        with pytest.raises(ExpressionError):
            evaluate(expression, {"x": 1.0})

    def test_unknown_function_names_the_alternatives(self):
        """An error a person can act on beats a correct-but-silent refusal."""
        with pytest.raises(ExpressionError, match="unknown function 'foo'"):
            evaluate("foo(x)", {"x": 1.0})

    def test_a_called_name_is_not_reported_as_a_missing_variable(self):
        """``open(x)`` is an unknown *function*, not an unbound variable.

        Reporting it as the latter suggests ``--at open=<number>``, which is
        both useless and actively misleading.
        """
        with pytest.raises(ExpressionError, match="unknown function"):
            evaluate("open(x)", {"x": 1.0})

    def test_missing_variable_says_which_one(self):
        with pytest.raises(ExpressionError, match="missing value.*'y'"):
            evaluate("x + y", {"x": 1.0})

    def test_function_arity_is_enforced(self):
        with pytest.raises(ExpressionError, match="exactly one"):
            evaluate("tanh(x, x)", {"x": 1.0})

    def test_booleans_are_not_numbers(self):
        with pytest.raises(ExpressionError):
            evaluate("True + x", {"x": 1.0})

    def test_empty_expression(self):
        with pytest.raises(ExpressionError, match="empty"):
            evaluate("   ", {})

    def test_syntax_error_is_reported_as_our_error_type(self):
        with pytest.raises(ExpressionError, match="could not parse"):
            evaluate("x +", {"x": 1.0})


class TestFreeVariables:
    def test_finds_names_in_order_of_appearance(self):
        assert free_variables("w*x + b") == ["w", "x", "b"]

    def test_excludes_known_functions(self):
        assert free_variables("tanh(x)") == ["x"]

    def test_excludes_called_names(self):
        assert free_variables("foo(x)") == ["x"]


class TestParseAssignments:
    def test_basic(self):
        assert parse_assignments(["x=2", "w=-3.5"]) == {"x": 2.0, "w": -3.5}

    def test_tolerates_trailing_commas(self):
        assert parse_assignments(["x=2,", "y=3"]) == {"x": 2.0, "y": 3.0}

    @pytest.mark.parametrize("bad", ["x", "=2", "x=abc", "2x=1"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ExpressionError):
            parse_assignments([bad])


# ======================================================================
# the command line itself
# ======================================================================


class TestCommandLine:
    def test_bare_invocation_prints_the_banner(self, capsys):
        assert main([]) == 0
        assert "nabla" in capsys.readouterr().out

    def test_grad_prints_value_and_gradients(self, capsys):
        assert main(["grad", "x*y + x", "--at", "x=2", "y=3"]) == 0
        out = capsys.readouterr().out
        assert "8" in out and "+4" in out and "+2" in out

    def test_grad_check_passes_on_a_correct_expression(self, capsys):
        assert main(["grad", "tanh(x)", "--at", "x=0.4", "--check"]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_bad_expression_exits_2_without_a_traceback(self, capsys):
        assert main(["grad", "x.__class__", "--at", "x=1"]) == 2
        assert "not allowed" in capsys.readouterr().err

    def test_graph_writes_svg(self, tmp_path, capsys):
        destination = tmp_path / "g.svg"
        assert main(["graph", "tanh(x*w+b)", "--at", "x=2", "w=-3", "b=1",
                     "-o", str(destination)]) == 0
        assert destination.read_text().startswith("<svg")
        assert "nodes" in capsys.readouterr().out

    def test_graph_writes_dot_when_asked(self, tmp_path):
        destination = tmp_path / "g.dot"
        assert main(["graph", "x*y", "--at", "x=2", "y=3", "-o", str(destination)]) == 0
        assert destination.read_text().startswith("digraph")

    def test_predict_rejects_a_column_count_mismatch(self, tmp_path):
        """A wrong-width CSV must fail loudly rather than predict nonsense."""
        checkpoint = tmp_path / "ckpt.json"
        checkpoint.write_text(json.dumps({
            "format": "nabla-checkpoint-v1",
            "model": {},
            "metadata": {"architecture": {"sizes": [4, 3, 2], "activation": "relu"}},
        }))
        csv = tmp_path / "rows.csv"
        csv.write_text("1,2\n3,4\n")          # 2 columns, model wants 4
        with pytest.raises(SystemExit, match="columns"):
            main(["predict", str(checkpoint), str(csv)])
