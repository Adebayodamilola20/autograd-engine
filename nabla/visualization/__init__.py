"""Phase 6/13 -- seeing the graph, the data and the training run.

``graph_svg`` renders a computational graph with no system dependencies;
``graph_dot`` exports Graphviz DOT for those who prefer it; ``plots`` holds the
matplotlib figures used by the training and experiment scripts.
"""

from importlib import import_module

from .graph_dot import render_dot, to_dot
from .graph_svg import graph_to_svg, render_svg

__all__ = ["render_svg", "graph_to_svg", "to_dot", "render_dot"]


def __getattr__(name: str):
    """Lazily expose the matplotlib helpers.

    Keeps ``import nabla.visualization`` working when matplotlib is absent --
    the SVG renderer has no dependencies and should not be held hostage by an
    optional extra.

    Uses ``import_module`` rather than ``from . import plots``, which looks
    equivalent and recurses forever. ``from . import plots`` compiles to
    ``_handle_fromlist``, whose first act is ``hasattr(package, "plots")``.
    A package defining ``__getattr__`` answers that by *calling* it, so the
    lookup re-enters this function, runs ``from . import plots`` again, and
    never reaches the import. Every lazy access died with RecursionError,
    including ones for names that do not exist, so even
    ``hasattr(nabla.visualization, "anything")`` raised instead of returning
    ``False``.

    ``import_module`` addresses the submodule directly and never consults the
    parent's attributes, so the recursion cannot start.
    """
    # Asking for the submodule itself must not fall through to the attribute
    # search below, which would look for `plots.plots`.
    if name == "plots":
        return import_module(".plots", __name__)
    if name.startswith("__"):
        # Dunder lookups (copy, pickle, inspect) must fail fast rather than
        # drag matplotlib in as a side effect of introspection.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    plots = import_module(".plots", __name__)
    try:
        return getattr(plots, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
