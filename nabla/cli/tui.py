r"""A full-screen terminal interface for the engine.

Run:  nabla tui

What this is for
----------------
``nabla repl`` prints a line and scrolls it away. That is fine for one
question, and wrong for the thing people actually do with an autodiff engine,
which is change one number and look at what moved. A full-screen layout can
keep the expression, the forward values, the graph structure and every gradient
on screen at once, so the comparison is spatial instead of remembered.

The three panes correspond to the three things the engine does, in order:
build a graph, evaluate it forward, sweep it backward.

Implementation notes
--------------------
``curses`` is in the standard library, so this costs the repository nothing.
The whole file is drawing and key handling; every number on screen comes from
``nabla.cli.expr`` and ``Value.backward()``, and there is no second
implementation of anything mathematical here.

Two details that are easy to get wrong and unpleasant to debug:

* ``addstr`` raises if you write to the last cell of the last line, because
  the cursor would have to advance off the screen. Everything is clipped
  through ``_put`` rather than trusting the terminal to be big enough.
* A resize arrives as ``KEY_RESIZE`` and invalidates every cached coordinate,
  so layout is recomputed on each frame instead of being stored.
"""

from __future__ import annotations

import curses
from typing import Any

from ..core.graph import topological_sort
from .expr import ExpressionError, evaluate, parse_assignments

__all__ = ["run"]

HELP = [
    ("enter", "evaluate"),
    ("↑ ↓", "history"),
    ("^u", "clear line"),
    ("f1", "examples"),
    ("^c", "quit"),
]

EXAMPLES = [
    "x*y + x  at x=2, y=3",
    "(x*w + b)**2  at x=2, w=-3, b=1",
    "tanh(x*w + b)  at x=0.5, w=1.5, b=-0.2",
    "exp(x)/(exp(x)+1)  at x=0.7",
    "log(x*x + 1)  at x=1.3",
    "relu(x*w)  at x=-2, w=3",
]

# Colour slots. Teal carries the forward pass, amber the backward pass -- the
# same pairing the project's diagrams use, so the two directions stay visually
# distinct in every medium.
C_DIM = 1
C_FWD = 2
C_BWD = 3
C_HEAD = 4
C_ERR = 5
C_ACC = 6


class Tui:
    def __init__(self, screen: Any) -> None:
        self.screen = screen
        self.buffer = ""
        self.history: list[str] = []
        self.history_index = 0
        self.status = "type an expression, then a point:  tanh(x*w+b) at x=2 w=-3 b=1"
        self.error = ""
        self.show_examples = False

        # The last successful evaluation. Held rather than reformatted on the
        # fly so a failed edit does not blank the panes you are reading.
        self.expression = ""
        self.point: dict[str, float] = {}
        self.nodes: list[Any] = []
        self.grads: dict[str, float] = {}
        self.output: Any = None

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def submit(self) -> None:
        line = self.buffer.strip()
        if not line:
            return
        self.history.append(line)
        self.history_index = len(self.history)
        self.buffer = ""
        self.error = ""

        # Split on the last " at " so an expression may itself contain those
        # letters without being torn in half.
        expression, _, assignments = line.rpartition(" at ")
        if not expression:
            expression, assignments = line, ""

        try:
            point = parse_assignments(assignments.replace(",", " ").split())
            out, env = evaluate(expression.strip(), point)
            out.backward()
        except ExpressionError as exc:
            self.error = str(exc)
            return
        except (ArithmeticError, ValueError) as exc:
            self.error = f"cannot evaluate: {exc}"
            return

        self.expression = expression.strip()
        self.point = point
        self.output = out
        self.grads = {name: node.grad for name, node in env.items()}
        # Topological order is parents-before-children, which is also the order
        # a person reads a calculation in, so the pane needs no extra sorting.
        self.nodes = topological_sort(out)
        self.status = f"{len(self.nodes)} nodes    ∂ computed in one backward sweep"

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def _put(self, y: int, x: int, text: str, attr: int = 0) -> None:
        """Write text, clipped to the window.

        curses raises on a write that would push the cursor past the last cell,
        so every string is truncated to what actually fits.
        """
        height, width = self.screen.getmaxyx()
        if not (0 <= y < height) or x >= width:
            return
        room = width - x - 1
        if room <= 0:
            return
        try:
            self.screen.addstr(y, x, text[:room], attr)
        except curses.error:
            pass

    def _rule(self, y: int, width: int) -> None:
        self._put(y, 0, "─" * (width - 1), curses.color_pair(C_DIM))

    def draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()

        # --- header ---------------------------------------------------
        self._put(0, 2, "∇ nabla", curses.color_pair(C_ACC) | curses.A_BOLD)

        # The hint is right-aligned and the subtitle left-aligned, so on a
        # narrow terminal they collide and the subtitle is rendered as
        # "gradient workenter evaluate...". Drop whichever does not fit rather
        # than overprinting: the key hints are the more useful of the two.
        subtitle, subtitle_x = "gradient workspace", 10
        hint = "  ".join(f"{k} {v}" for k, v in HELP)
        hint_x = width - len(hint) - 3

        if hint_x > subtitle_x + len(subtitle) + 2:
            self._put(0, subtitle_x, subtitle, curses.color_pair(C_DIM))
            self._put(0, hint_x, hint, curses.color_pair(C_DIM))
        elif hint_x > subtitle_x:
            self._put(0, hint_x, hint, curses.color_pair(C_DIM))
        else:
            self._put(0, subtitle_x, subtitle, curses.color_pair(C_DIM))
        self._rule(1, width)

        body_top, body_bottom = 2, height - 4
        if self.show_examples:
            self._draw_examples(body_top, body_bottom)
        elif self.output is None:
            self._draw_splash(body_top, body_bottom)
        else:
            self._draw_result(body_top, body_bottom, width)

        # --- input ----------------------------------------------------
        self._rule(height - 3, width)
        if self.error:
            self._put(height - 2, 2, self.error[: width - 4], curses.color_pair(C_ERR))
        else:
            self._put(height - 2, 2, self.status, curses.color_pair(C_DIM))

        prompt = "∇> "
        self._put(height - 1, 2, prompt, curses.color_pair(C_ACC) | curses.A_BOLD)
        self._put(height - 1, 2 + len(prompt), self.buffer)
        # Park the hardware cursor after the text so the terminal's own caret
        # is the one the user sees blinking.
        try:
            self.screen.move(height - 1, min(2 + len(prompt) + len(self.buffer), width - 1))
        except curses.error:
            pass

    def _draw_splash(self, top: int, bottom: int) -> None:
        lines = [
            ("Reverse-mode automatic differentiation, from scratch.", C_DIM),
            ("", C_DIM),
            ("Type a mathematical expression and where to evaluate it.", 0),
            ("The engine builds the graph, runs it forward, then sweeps", C_DIM),
            ("it backward and reports every partial derivative.", C_DIM),
            ("", C_DIM),
            ("    tanh(x*w + b)  at x=0.5, w=1.5, b=-0.2", C_FWD),
            ("", C_DIM),
            ("Functions: exp  log  tanh  sigmoid  relu", C_DIM),
            ("Operators: +  -  *  /  **", C_DIM),
            ("", C_DIM),
            ("f1 for more examples.", C_DIM),
        ]
        for i, (text, colour) in enumerate(lines):
            if top + i >= bottom:
                break
            self._put(top + 1 + i, 4, text, curses.color_pair(colour) if colour else 0)

    def _draw_examples(self, top: int, bottom: int) -> None:
        self._put(top + 1, 4, "EXAMPLES", curses.color_pair(C_DIM) | curses.A_BOLD)
        for i, example in enumerate(EXAMPLES):
            if top + 3 + i >= bottom:
                break
            self._put(top + 3 + i, 4, example, curses.color_pair(C_FWD))
        self._put(min(top + 4 + len(EXAMPLES), bottom - 1), 4,
                  "f1 to go back", curses.color_pair(C_DIM))

    def _draw_result(self, top: int, bottom: int, width: int) -> None:
        row = top + 1
        self._put(row, 2, self.expression, curses.A_BOLD)
        at = "   ".join(f"{k} = {v:g}" for k, v in self.point.items())
        self._put(row, 4 + len(self.expression), at, curses.color_pair(C_DIM))
        row += 2

        # Two columns when there is room, stacked when there is not.
        split = width // 2 if width >= 76 else width
        left_rows = self._draw_forward(row, bottom, 2, split - 4)
        if split == width:
            row = min(row + left_rows + 1, bottom)
            self._draw_backward(row, bottom, 2)
        else:
            self._draw_backward(row, bottom, split)

    def _draw_forward(self, top: int, bottom: int, x: int, room: int) -> int:
        self._put(top, x, "FORWARD", curses.color_pair(C_FWD) | curses.A_BOLD)
        drawn = 0
        for i, node in enumerate(self.nodes):
            y = top + 2 + i
            if y >= bottom:
                self._put(y - 1, x + 2, f"... {len(self.nodes) - i} more",
                          curses.color_pair(C_DIM))
                break
            label = node.label or (node._op or "?")
            self._put(y, x + 2, f"{label:<12}", curses.color_pair(C_DIM))
            self._put(y, x + 14, f"{node.data:>12.6g}", curses.color_pair(C_FWD))
            drawn = i + 2
        return drawn + 2

    def _draw_backward(self, top: int, bottom: int, x: int) -> None:
        self._put(top, x, "BACKWARD", curses.color_pair(C_BWD) | curses.A_BOLD)
        for i, (name, grad) in enumerate(self.grads.items()):
            y = top + 2 + i
            if y >= bottom:
                break
            self._put(y, x + 2, f"∂/∂{name:<9}", curses.color_pair(C_DIM))
            self._put(y, x + 14, f"{grad:>+12.6g}", curses.color_pair(C_BWD))

        y = top + 3 + len(self.grads)
        if y < bottom and self.output is not None:
            self._put(y, x + 2, f"{'value':<12}", curses.color_pair(C_DIM))
            self._put(y, x + 14, f"{self.output.data:>12.6g}", curses.A_BOLD)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------

    def handle(self, key: int) -> bool:
        """Process one keypress. Returns False when the app should exit."""
        if key in (curses.KEY_RESIZE,):
            return True
        if key in (curses.KEY_F1,):
            self.show_examples = not self.show_examples
            return True
        if key in (10, 13, curses.KEY_ENTER):
            self.show_examples = False
            self.submit()
            return True
        if key in (curses.KEY_BACKSPACE, 127, 8):
            self.buffer = self.buffer[:-1]
            return True
        if key == 21:  # ^U
            self.buffer = ""
            return True
        if key == curses.KEY_UP and self.history:
            self.history_index = max(0, self.history_index - 1)
            self.buffer = self.history[self.history_index]
            return True
        if key == curses.KEY_DOWN and self.history:
            self.history_index = min(len(self.history), self.history_index + 1)
            self.buffer = (
                "" if self.history_index >= len(self.history)
                else self.history[self.history_index]
            )
            return True
        if 32 <= key < 127:
            self.buffer += chr(key)
            return True
        return True

    def loop(self) -> None:
        while True:
            self.draw()
            self.screen.refresh()
            try:
                key = self.screen.getch()
            except KeyboardInterrupt:
                return
            if key == 3:  # ^C
                return
            if not self.handle(key):
                return


def _init_colours() -> None:
    """Set up colour pairs, degrading quietly on a monochrome terminal."""
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    for slot, colour in (
        (C_DIM, curses.COLOR_WHITE),
        (C_FWD, curses.COLOR_CYAN),
        (C_BWD, curses.COLOR_YELLOW),
        (C_HEAD, curses.COLOR_WHITE),
        (C_ERR, curses.COLOR_RED),
        (C_ACC, curses.COLOR_CYAN),
    ):
        curses.init_pair(slot, colour, -1)


def _main(screen: Any) -> int:
    curses.curs_set(1)
    _init_colours()
    screen.keypad(True)
    Tui(screen).loop()
    return 0


def run() -> int:
    """Entry point. ``curses.wrapper`` restores the terminal even on a crash."""
    try:
        return curses.wrapper(_main)
    except curses.error as exc:
        print(f"nabla: the terminal is too small or unsupported ({exc})")
        return 1
