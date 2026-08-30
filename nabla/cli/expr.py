r"""Parse a mathematical expression into a ``Value`` graph, without ``eval``.

Why this file exists
--------------------
``Value`` overloads Python's arithmetic operators, so the shortest way to turn
the string ``"x*w + b"`` into a graph is ``eval(expr, {}, {"x": Value(2), ...})``
-- the operators do the rest. It is three lines and it works.

It is also a remote code execution hole waiting for the day an expression
arrives from somewhere other than the user's own keyboard. ``eval`` with an
emptied ``__builtins__`` is *not* a sandbox; the classic escape

    ().__class__.__bases__[0].__subclasses__()

walks from a literal to every class the interpreter has loaded, and from there
to ``os.system``. The repository already takes its one network-facing file
seriously (see ``web/README.md``), and a parser that is safe only while nobody
pipes a file into it is not safe.

So we parse instead. ``ast.parse(mode="eval")`` gives us Python's own tokeniser
and grammar for free -- no hand-rolled precedence table, no bugs in it -- and
then we walk the tree and refuse every node type that is not arithmetic. The
whitelist is small enough to read in one sitting, which is the property that
makes it trustworthy:

* numbers, names, parentheses
* ``+ - * / ** //`` and unary ``+ -``
* calls to the named functions in ``FUNCTIONS``, and nothing else

Attribute access, subscripting, comprehensions, lambdas, walrus, f-strings,
starred arguments and keywords are all rejected by omission, because the walker
raises on any node it does not recognise rather than ignoring it. That default
is the whole design: a new Python grammar node cannot silently become reachable.
"""

from __future__ import annotations

import ast
from typing import Callable, Iterable, Mapping

from ..core.value import Value

__all__ = ["ExpressionError", "FUNCTIONS", "build", "evaluate", "free_variables"]


class ExpressionError(ValueError):
    """Raised for anything we will not evaluate, with a reason a human can act on."""


# The unary functions a Value already knows how to differentiate. Keeping this
# a plain dict -- rather than reaching into Value with getattr -- means adding a
# name here is a deliberate act, and `dir(Value)` can never become the API.
FUNCTIONS: dict[str, Callable[[Value], Value]] = {
    "exp": lambda v: v.exp(),
    "log": lambda v: v.log(),
    "tanh": lambda v: v.tanh(),
    "sigmoid": lambda v: v.sigmoid(),
    "relu": lambda v: v.relu(),
}

_BINOPS: dict[type, Callable[[Value, Value], Value]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


def _parse(expression: str) -> ast.Expression:
    """Python's grammar, our restrictions. Syntax errors become our error type."""
    if not expression.strip():
        raise ExpressionError("empty expression")
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"could not parse {expression!r}: {exc.msg}") from exc


def free_variables(expression: str) -> list[str]:
    """Every name in ``expression`` that is not a known function.

    Used by the CLI to tell you *which* variables you still owe a value for,
    before it starts evaluating and fails on the first one.
    """
    tree = _parse(expression)

    # A name in call position is being used as a function, not a variable.
    # Without this, `open(x)` is reported as "missing value for 'open'" and the
    # suggested fix is `--at open=<number>` -- advice that is both useless and
    # actively confusing. Excluding them lets _build raise "unknown function"
    # instead, which is the truth.
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    # ast.walk is breadth-first, so it yields `w*x + b` as b, w, x. That order
    # reaches the user in "pass --at ..." hints, where a scrambled list is
    # needlessly hard to follow. Sort by position in the source instead.
    found: list[tuple[tuple[int, int], str]] = [
        ((node.lineno, node.col_offset), node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in FUNCTIONS
        and node.id not in called
    ]

    seen: dict[str, None] = {}  # dict preserves insertion order; dedupes repeats
    for _, name in sorted(found):
        seen[name] = None
    return list(seen)


def _literal_number(node: ast.AST) -> float | None:
    """``2`` and ``-2`` as floats; ``None`` for anything that is not a literal.

    Written out rather than using ``ast.literal_eval`` so that a negated
    literal -- which is a ``UnaryOp`` wrapping a ``Constant``, not a constant --
    is recognised too.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal_number(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _build(node: ast.AST, env: Mapping[str, Value]) -> Value:
    """Recursively turn one AST node into a ``Value``, rejecting anything else."""

    # --- literals -----------------------------------------------------
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(
                f"only real numbers are allowed, got {node.value!r}"
            )
        return Value(float(node.value))

    # --- variables ----------------------------------------------------
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in FUNCTIONS:
            raise ExpressionError(f"{node.id} is a function; write {node.id}(x)")
        raise ExpressionError(f"no value given for variable {node.id!r}")

    # --- a + b, a * b, a ** b ... -------------------------------------
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(
                f"unsupported operator {type(node.op).__name__}; "
                f"allowed: + - * / **"
            )

        # `x ** 2` must reach Value.__pow__ with a *plain number* exponent.
        # Wrapping the 2 in a Value selects the general a**b rule instead,
        # whose derivative involves log(a) and so rejects any negative base --
        # making `(x*w+b)**2` fail at x*w+b = -5, which is ordinary arithmetic
        # every other tool handles. A literal exponent is a constant, not a
        # variable anyone differentiates with respect to, so pass it through.
        if isinstance(node.op, ast.Pow):
            exponent = _literal_number(node.right)
            if exponent is not None:
                return _build(node.left, env) ** exponent

        return op(_build(node.left, env), _build(node.right, env))

    # --- -a, +a -------------------------------------------------------
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_build(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return _build(node.operand, env)
        raise ExpressionError(
            f"unsupported unary operator {type(node.op).__name__}"
        )

    # --- tanh(x), exp(x) ... ------------------------------------------
    if isinstance(node, ast.Call):
        # `func` must be a bare Name. Rejecting Attribute here is what stops
        # `x.__class__` and every other traversal out of the arithmetic world.
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls like tanh(x) are allowed")
        fn = FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ExpressionError(
                f"unknown function {node.func.id!r}; "
                f"available: {', '.join(sorted(FUNCTIONS))}"
            )
        if node.keywords or len(node.args) != 1:
            raise ExpressionError(
                f"{node.func.id}() takes exactly one positional argument"
            )
        return fn(_build(node.args[0], env))

    raise ExpressionError(
        f"{type(node).__name__} is not allowed in an expression -- "
        "this parser accepts arithmetic only"
    )


def build(expression: str, env: Mapping[str, Value]) -> Value:
    """Build the graph for ``expression`` on top of ``Value`` objects you own.

    ``evaluate`` below creates its own leaves from plain floats, which is what
    you want at the command line. Gradient checking is not that case: it makes
    the leaves itself and reads ``.grad`` back off *those* objects afterwards,
    so a function that quietly substitutes fresh ones reports every analytic
    gradient as zero. This entry point exists so the caller's nodes are the
    ones that end up in the graph.
    """
    return _build(_parse(expression).body, env)


def evaluate(
    expression: str,
    point: Mapping[str, float],
    *,
    require_all: bool = True,
) -> tuple[Value, dict[str, Value]]:
    """Build the graph for ``expression`` at ``point``.

    Returns ``(output, variables)``. Nothing is differentiated here -- call
    ``output.backward()`` and then read ``.grad`` off each variable. Keeping
    those separate means the caller can inspect the graph first, which is what
    the ``graph`` subcommand does.

    Parameters
    ----------
    require_all
        Raise if ``expression`` mentions a variable ``point`` does not supply.
        The alternative -- quietly defaulting to zero -- would return a
        confident wrong answer, which is worse than an error.
    """
    tree = _parse(expression)

    if require_all:
        missing = [name for name in free_variables(expression) if name not in point]
        if missing:
            raise ExpressionError(
                f"missing value{'s' if len(missing) > 1 else ''} for "
                f"{', '.join(repr(m) for m in missing)}; "
                f"pass --at {' '.join(f'{m}=<number>' for m in missing)}"
            )

    env = {name: Value(float(v), label=name) for name, v in point.items()}
    return _build(tree.body, env), env


def parse_assignments(pairs: Iterable[str]) -> dict[str, float]:
    """Turn ``["x=2", "w=-3"]`` into ``{"x": 2.0, "w": -3.0}``.

    Accepts ``x=2`` and ``x 2`` styles equally, since argparse splitting on
    spaces makes both natural to type.
    """
    point: dict[str, float] = {}
    for raw in pairs:
        item = raw.strip().rstrip(",")
        if "=" not in item:
            raise ExpressionError(
                f"expected name=value, got {raw!r} (for example: x=2)"
            )
        name, _, number = item.partition("=")
        name = name.strip()
        if not name.isidentifier():
            raise ExpressionError(f"{name!r} is not a valid variable name")
        try:
            point[name] = float(number)
        except ValueError:
            raise ExpressionError(
                f"{number.strip()!r} is not a number (in {raw!r})"
            ) from None
    return point
