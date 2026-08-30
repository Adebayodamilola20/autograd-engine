r"""``nabla`` -- one command over the whole engine.

Run:  nabla                 (the banner and the list of subcommands)
      nabla grad "x*y + x" --at x=2 y=3
      nabla repl

Why a CLI at all
----------------
Everything this repository does was reachable only by running a specific script
out of ``examples/`` or ``benchmarks/``. That is fine once you know the layout
and hostile before you do. A single entry point turns "read the README and find
the right file" into ``nabla --help``.

The interaction model is ``git``-shaped, not chatbot-shaped: subcommands and
arguments, not sentences. The one exception is ``repl``, where you type a
mathematical expression and the engine answers with its derivatives -- which is
as close to a prompt as an autodiff engine has any business getting, and is
also the single clearest demonstration of what the project is.

Subcommands deliberately shell out to nothing. Each one imports the library
directly, so ``nabla train`` is the same code path as ``import nabla``, and a
failure here is a real failure rather than a subprocess quirk.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

from .. import __version__
from .expr import FUNCTIONS, ExpressionError, build, evaluate, parse_assignments

ROOT = Path(__file__).resolve().parents[2]

# ANSI, but only when we are actually attached to a terminal. Piping `nabla
# grad ... > out.txt` should produce text, not escape sequences.
_TTY = sys.stdout.isatty()


def _b(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _TTY else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _TTY else text


BANNER = f"""
  {_b("∇ nabla")} {_dim(__version__)} -- reverse-mode automatic differentiation, from scratch

  {_b("grad")}     differentiate an expression at a point
  {_b("graph")}    render the computational graph as SVG or DOT
  {_b("tui")}      full-screen gradient workspace
  {_b("repl")}     interactive shell: type maths, get derivatives
  {_b("train")}    train a network on a CSV
  {_b("predict")}  run a trained checkpoint over new rows
  {_b("demo")}     launch the draw-a-digit web demo
  {_b("bench")}    measure against PyTorch

  {_dim("nabla <command> --help  for the options of any one of them")}
"""


# ======================================================================
# grad
# ======================================================================


def _report(expression: str, point: dict[str, float], out, env) -> None:
    """Print value and every partial derivative, aligned."""
    at = ", ".join(f"{k}={v:g}" for k, v in point.items())
    print(f"\n  {_b(expression)}   at   {at}" if at else f"\n  {_b(expression)}")
    print(f"    value      = {out.data:.6g}")
    width = max((len(k) for k in env), default=1)
    for name, node in env.items():
        print(f"    ∂/∂{name:<{width}}  = {node.grad:+.6g}")


def cmd_grad(args: argparse.Namespace) -> int:
    point = parse_assignments(args.at)
    out, env = evaluate(args.expression, point)
    out.backward()
    _report(args.expression, point, out, env)

    if args.check:
        # Finite differences share no code and no reasoning with the analytic
        # rules -- the same argument Phase 5 makes, available from the CLI.
        from ..core.gradcheck import check_gradients

        names = list(env)
        result = check_gradients(
            lambda *vals: build(args.expression, dict(zip(names, vals))),
            [point[n] for n in names],
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"\n    gradient check: {verdict}  (max rel err {result.max_error:.2e})")
        return 0 if result.passed else 1
    return 0


# ======================================================================
# graph
# ======================================================================


def cmd_graph(args: argparse.Namespace) -> int:
    point = parse_assignments(args.at)
    out, env = evaluate(args.expression, point)
    out.label = args.label
    if not args.no_backward:
        out.backward()

    destination = Path(args.output)
    if destination.suffix == ".dot":
        from ..visualization import to_dot

        destination.write_text(to_dot(out, title=args.expression), encoding="utf-8")
    else:
        from ..visualization import graph_to_svg

        destination.write_text(graph_to_svg(out), encoding="utf-8")

    from ..core.graph import graph_size

    size = graph_size(out)
    print(f"  {args.expression}  ->  {destination}")
    print(
        f"  {size['nodes']} nodes, {size['edges']} edges, "
        f"{size['leaves']} leaves, depth {size['depth']}"
    )
    return 0


# ======================================================================
# repl
# ======================================================================

REPL_HELP = f"""
  Type an expression and where to evaluate it:

    x*y + x  at  x=2, y=3
    (x*w + b)**2  at x=2, w=-3, b=1
    tanh(x*w + b) at x=0.5, w=1.5, b=-0.2

  Functions: {', '.join(sorted(FUNCTIONS))}
  Operators: + - * / **

  :h  this help      :q  quit
"""


def cmd_repl(args: argparse.Namespace) -> int:
    print(f"\n  {_b('∇ nabla')} -- type an expression, get its derivatives.  "
          f"{_dim(':h help   :q quit')}")
    while True:
        try:
            line = input(f"\n{_b('∇>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in (":q", ":quit", "quit", "exit"):
            return 0
        if line in (":h", ":help", "?"):
            print(REPL_HELP)
            continue

        # "expr at x=1, y=2" -- split on the last ` at ` so an expression may
        # itself contain the letters a-t without being torn in half.
        expression, _, assignments = line.rpartition(" at ")
        if not expression:
            expression, assignments = line, ""

        try:
            point = parse_assignments(assignments.replace(",", " ").split())
            out, env = evaluate(expression.strip(), point)
            out.backward()
            _report(expression.strip(), point, out, env)
        except ExpressionError as exc:
            print(f"  {exc}")
        except (ArithmeticError, ValueError) as exc:
            print(f"  cannot evaluate: {exc}")
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    # Imported here rather than at module scope so that `nabla grad` on a
    # system without a working curses build still runs.
    from .tui import run

    return run()


# ======================================================================
# train / predict
# ======================================================================


def cmd_train(args: argparse.Namespace) -> int:
    forwarded = ["--csv", str(args.csv)] if args.csv else []
    for flag, value in (
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--lr", args.lr),
        ("--seed", args.seed),
    ):
        forwarded += [flag, str(value)]
    if args.hidden:
        forwarded += ["--hidden", *[str(h) for h in args.hidden]]

    return _run_script("examples/train_your_own_data.py", forwarded)


def cmd_predict(args: argparse.Namespace) -> int:
    import numpy as np

    from ..nn.tensor_mlp import TensorMLP
    from ..training.checkpoint import load_checkpoint

    payload = load_checkpoint(args.checkpoint)
    arch = payload.get("metadata", {}).get("architecture") or {}
    sizes = arch.get("sizes")
    if not sizes:
        raise SystemExit(
            f"{args.checkpoint} has no architecture metadata, so the network "
            "shape cannot be recovered; retrain with a current version"
        )

    # Read and validate the CSV *before* loading weights. Both orders reject a
    # bad file, but this one reports the mismatch the user can fix rather than
    # a state_dict error from deep inside the model, and it costs nothing.
    rows = np.genfromtxt(args.csv, delimiter=",", skip_header=1 if args.header else 0)
    rows = np.atleast_2d(rows)
    if rows.shape[1] == sizes[0] + 1:
        rows = rows[:, :-1]          # a label column came along; ignore it
    if rows.shape[1] != sizes[0]:
        raise SystemExit(
            f"{args.csv} has {rows.shape[1]} columns but the model expects "
            f"{sizes[0]} features"
        )

    model = TensorMLP(
        sizes[0], list(sizes[1:-1]), sizes[-1],
        activation=arch.get("activation") or "relu", seed=0,
    )
    load_checkpoint(args.checkpoint, model=model)

    probabilities = model.predict_proba(rows)
    predictions = probabilities.argmax(axis=-1)

    print("  row  prediction  confidence")
    for i, (p, probs) in enumerate(zip(predictions, probabilities)):
        if args.limit and i >= args.limit:
            print(f"  ... {len(predictions) - args.limit:,} more")
            break
        print(f"  {i:>3}  {int(p):>10}  {probs.max():>10.2%}")
    return 0


# ======================================================================
# demo / bench -- thin wrappers over the scripts that already exist
# ======================================================================


def _run_script(relative: str, argv: Sequence[str]) -> int:
    """Execute a repository script in-process, with ``argv`` as its arguments.

    ``runpy`` rather than ``subprocess`` so the script shares this interpreter
    and this virtualenv -- ``nabla demo`` must not silently pick up a different
    Python than the one the CLI was installed into.
    """
    script = ROOT / relative
    if not script.exists():
        raise SystemExit(
            f"{relative} is missing -- this looks like an installed copy "
            "rather than a clone; that script ships only in the repository"
        )
    saved = sys.argv
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_:
        return int(exit_.code or 0)
    finally:
        sys.argv = saved
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    return _run_script("web/server.py", ["--port", str(args.port), "--host", args.host])


def cmd_bench(args: argparse.Namespace) -> int:
    return _run_script("benchmarks/compare.py", ["--skip-scalar"] if args.quick else [])


# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nabla",
        description="Reverse-mode automatic differentiation, built from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"nabla {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p = subparsers.add_parser("grad", help="differentiate an expression at a point")
    p.add_argument("expression", help='for example: "x*y + x"')
    p.add_argument("--at", nargs="*", default=[], metavar="name=value",
                   help="where to evaluate, e.g. --at x=2 y=3")
    p.add_argument("--check", action="store_true",
                   help="verify against finite differences")
    p.set_defaults(func=cmd_grad)

    p = subparsers.add_parser("graph", help="render the computational graph")
    p.add_argument("expression")
    p.add_argument("--at", nargs="*", default=[], metavar="name=value")
    p.add_argument("-o", "--output", default="graph.svg",
                   help="output path; .dot writes Graphviz instead of SVG")
    p.add_argument("--label", default="L", help="name for the output node")
    p.add_argument("--no-backward", action="store_true",
                   help="draw values only, without running the backward pass")
    p.set_defaults(func=cmd_graph)

    p = subparsers.add_parser("repl", help="interactive gradient shell")
    p.set_defaults(func=cmd_repl)

    p = subparsers.add_parser("tui", help="full-screen gradient workspace")
    p.set_defaults(func=cmd_tui)

    p = subparsers.add_parser("train", help="train a network on a CSV")
    p.add_argument("csv", nargs="?", type=Path,
                   help="CSV whose last column is the label; omit for a synthetic demo")
    p.add_argument("--hidden", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = subparsers.add_parser("predict", help="run a trained checkpoint over new rows")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("csv", type=Path)
    p.add_argument("--header", action="store_true", help="skip the first row")
    p.add_argument("--limit", type=int, default=20, help="rows to print; 0 for all")
    p.set_defaults(func=cmd_predict)

    p = subparsers.add_parser("demo", help="launch the draw-a-digit web demo")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_demo)

    p = subparsers.add_parser("bench", help="measure against PyTorch")
    p.add_argument("--quick", action="store_true",
                   help="skip the scalar engine, which dominates the runtime")
    p.set_defaults(func=cmd_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        print(BANNER)
        return 0

    try:
        return int(args.func(args) or 0)
    except ExpressionError as exc:
        # A malformed expression is user error, not a crash. A traceback here
        # would bury the one line that says what to fix.
        print(f"nabla: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
