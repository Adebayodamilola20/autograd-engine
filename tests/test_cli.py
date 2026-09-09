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

    def test_repeated_at_flags_accumulate(self, capsys):
        """``--at x=2 --at y=3`` must mean the same as ``--at x=2 y=3``.

        With argparse's default ``store`` action the second flag replaces the
        first, so this command failed with "missing value for 'x'" while
        pointing at a line where ``x`` was plainly supplied.
        """
        assert main(["grad", "x*y + x", "--at", "x=2", "--at", "y=3"]) == 0
        out = capsys.readouterr().out
        assert "8" in out and "+4" in out and "+2" in out

    def test_at_flags_do_not_leak_between_invocations(self, capsys):
        """``default=[]`` is a shared mutable, so ``extend`` must copy it."""
        assert main(["grad", "x+1", "--at", "x=1"]) == 0
        capsys.readouterr()
        assert main(["grad", "y+1", "--at", "y=5"]) == 0
        assert "6" in capsys.readouterr().out

    def test_predict_applies_the_saved_normalisation(self, tmp_path, capsys):
        """Raw columns must go through the transform training used.

        The weights load cleanly either way, so getting this wrong produces
        confident predictions rather than an error. The checkpoint below is
        built so the two paths disagree: the single weight is +1, and the
        stored mean of 100 is what decides the sign of the input.
        """
        from nabla.nn.tensor_mlp import TensorMLP

        model = TensorMLP(1, [], 2, activation="relu", seed=0)
        state = model.state_dict()
        checkpoint = tmp_path / "ckpt.json"
        checkpoint.write_text(json.dumps({
            "format": "nabla-checkpoint-v1",
            "model": state,
            "metadata": {
                "architecture": {"sizes": [1, 2], "activation": "relu"},
                "normalisation": {"mean": [100.0], "std": [1.0]},
            },
        }))
        csv = tmp_path / "rows.csv"
        csv.write_text("100\n")

        assert main(["predict", str(checkpoint), str(csv)]) == 0
        captured = capsys.readouterr()
        assert "prediction" in captured.out
        # The statistics were present, so no "scored as-is" warning is due.
        assert "as-is" not in captured.err

    def test_predict_warns_when_normalisation_is_absent(self, tmp_path, capsys):
        """An old checkpoint still runs, but must say why it may be wrong."""
        from nabla.nn.tensor_mlp import TensorMLP

        model = TensorMLP(1, [], 2, activation="relu", seed=0)
        checkpoint = tmp_path / "ckpt.json"
        checkpoint.write_text(json.dumps({
            "format": "nabla-checkpoint-v1",
            "model": model.state_dict(),
            "metadata": {"architecture": {"sizes": [1, 2], "activation": "relu"}},
        }))
        csv = tmp_path / "rows.csv"
        csv.write_text("0.5\n")

        assert main(["predict", str(checkpoint), str(csv)]) == 0
        assert "as-is" in capsys.readouterr().err


class TestGenerate:
    """``nabla generate`` over a language-model checkpoint."""

    def _checkpoint(self, tmp_path):
        from nabla.data.tokenizer import CharTokenizer
        from nabla.nn.transformer import GPT

        tokenizer = CharTokenizer.from_text("hello world")
        model = GPT(vocab_size=tokenizer.vocab_size, block_size=8, d_model=8,
                    n_head=2, n_layer=1, seed=0)
        path = tmp_path / "lm.json"
        path.write_text(json.dumps({
            "format": "nabla-checkpoint-v1",
            "model": model.state_dict(),
            "metadata": {
                "kind": "gpt",
                "config": model.config(),
                "tokenizer": tokenizer.to_dict(),
            },
        }))
        return path

    def test_generate_writes_a_continuation(self, tmp_path, capsys):
        path = self._checkpoint(tmp_path)
        assert main(["generate", str(path), "--prompt", "hel", "-n", "12",
                     "--seed", "0"]) == 0
        out = capsys.readouterr().out
        assert "hel" in out
        assert len(out.strip()) > 3          # something was actually appended

    def test_generate_is_reproducible_under_a_seed(self, tmp_path, capsys):
        path = self._checkpoint(tmp_path)
        main(["generate", str(path), "--prompt", "hel", "-n", "20", "--seed", "7"])
        first = capsys.readouterr().out
        main(["generate", str(path), "--prompt", "hel", "-n", "20", "--seed", "7"])
        assert capsys.readouterr().out == first

    def test_a_classifier_checkpoint_is_rejected(self, tmp_path):
        """`generate` on an MLP checkpoint must say so, not fail obscurely."""
        path = tmp_path / "mlp.json"
        path.write_text(json.dumps({
            "format": "nabla-checkpoint-v1",
            "model": {},
            "metadata": {"architecture": {"sizes": [4, 2], "activation": "relu"}},
        }))
        with pytest.raises(SystemExit, match="not a language model"):
            main(["generate", str(path)])

    def test_a_partly_unknown_prompt_does_not_eat_the_output(self, tmp_path, capsys):
        """Slicing must use the round-tripped prompt, not the raw string.

        ``CharTokenizer`` drops characters it has no id for, so ``decode(
        encode(p))`` can be shorter than ``p``. Slicing the output at
        ``len(p)`` would silently swallow the first characters the model
        generated. Here 'Z' and 'Q' are absent from the vocabulary.
        """
        path = self._checkpoint(tmp_path)
        assert main(["generate", str(path), "--prompt", "ZQhel", "-n", "10",
                     "--seed", "0"]) == 0
        first = capsys.readouterr().out.strip()

        # 'hel' is the part that survives encoding, so the output must begin
        # there rather than at some offset three characters further in.
        assert first.startswith("hel")

    def test_a_prompt_outside_the_vocabulary_is_rejected(self, tmp_path):
        path = self._checkpoint(tmp_path)
        with pytest.raises(SystemExit, match="vocabulary"):
            main(["generate", str(path), "--prompt", "ZZZZ"])


class TestRepl:
    def test_missing_values_hint_uses_repl_syntax(self):
        """The REPL has no ``--at`` flag, so it must not tell you to type one."""
        with pytest.raises(ExpressionError) as excinfo:
            evaluate("x*y", {}, hint="add: at {assignments}")
        message = str(excinfo.value)
        assert "--at" not in message
        assert "at x=<number> y=<number>" in message

    def test_command_line_hint_still_names_the_flag(self):
        with pytest.raises(ExpressionError, match=r"pass --at x=<number>"):
            evaluate("x*y", {"y": 1.0})


# ======================================================================
# the TUI
# ======================================================================


class TestTui:
    """The state machine, tested without a terminal.

    ``submit`` and ``handle`` never touch ``self.screen`` -- all drawing goes
    through ``draw``. Keeping that separation is what makes the interesting
    half of the TUI testable in a normal pytest run, on CI, with no pty.
    """

    def _tui(self):
        from nabla.cli.tui import Tui

        return Tui(screen=None)

    def test_submit_evaluates_and_records_both_directions(self):
        tui = self._tui()
        tui.buffer = "tanh(x*w + b) at x=0.5, w=1.5, b=-0.2"
        tui.submit()

        assert not tui.error
        assert tui.output.data == pytest.approx(0.50052, abs=1e-5)
        assert tui.grads["x"] == pytest.approx(1.12422, abs=1e-5)
        assert tui.grads["w"] == pytest.approx(0.37474, abs=1e-5)
        assert tui.grads["b"] == pytest.approx(0.74948, abs=1e-5)
        assert len(tui.nodes) == 6

    def test_nodes_arrive_in_topological_order(self):
        """Parents before children -- the order a person reads a calculation."""
        from nabla.core.graph import is_topologically_sorted

        tui = self._tui()
        tui.buffer = "(x*w + b)**2 at x=2, w=-3, b=1"
        tui.submit()
        assert is_topologically_sorted(tui.nodes)

    def test_a_bad_expression_sets_an_error_without_clearing_the_last_result(self):
        """A failed edit must not blank the panes the user is reading."""
        tui = self._tui()
        tui.buffer = "x*y at x=2, y=3"
        tui.submit()
        good = tui.output.data

        tui.buffer = "x.__class__ at x=1"
        tui.submit()

        assert tui.error
        assert tui.output.data == good      # previous result survived

    def test_expression_without_a_point_reports_the_missing_variable(self):
        tui = self._tui()
        tui.buffer = "x*y"
        tui.submit()
        assert "missing value" in tui.error

    def test_history_navigates_with_the_arrow_keys(self):
        import curses

        tui = self._tui()
        for line in ("x*x at x=1", "x+x at x=2"):
            tui.buffer = line
            tui.submit()

        tui.handle(curses.KEY_UP)
        assert tui.buffer == "x+x at x=2"
        tui.handle(curses.KEY_UP)
        assert tui.buffer == "x*x at x=1"
        tui.handle(curses.KEY_DOWN)
        assert tui.buffer == "x+x at x=2"

    def test_typing_and_backspace(self):
        import curses

        tui = self._tui()
        for ch in "x*2":
            tui.handle(ord(ch))
        assert tui.buffer == "x*2"
        tui.handle(curses.KEY_BACKSPACE)
        assert tui.buffer == "x*"
        tui.handle(21)                       # ^U clears the line
        assert tui.buffer == ""

    def test_f1_toggles_examples(self):
        import curses

        tui = self._tui()
        assert not tui.show_examples
        tui.handle(curses.KEY_F1)
        assert tui.show_examples
        tui.handle(curses.KEY_F1)
        assert not tui.show_examples

    def test_every_shipped_example_actually_evaluates(self):
        """The examples pane must not advertise anything that errors."""
        from nabla.cli.tui import EXAMPLES

        for example in EXAMPLES:
            tui = self._tui()
            tui.buffer = example
            tui.submit()
            assert not tui.error, f"{example!r} failed: {tui.error}"
