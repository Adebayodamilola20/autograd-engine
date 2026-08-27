"""Phase 6/13 -- seeing the graph, the data and the training run.

``graph_svg`` renders a computational graph with no system dependencies;
``graph_dot`` exports Graphviz DOT for those who prefer it; ``plots`` holds the
matplotlib figures used by the training and experiment scripts.
"""

from .graph_dot import render_dot, to_dot
from .graph_svg import graph_to_svg, render_svg

__all__ = ["render_svg", "graph_to_svg", "to_dot", "render_dot"]


def __getattr__(name: str):
    """Lazily expose the matplotlib helpers.

    Keeps ``import nabla.visualization`` working when matplotlib is absent --
    the SVG renderer has no dependencies and should not be held hostage by an
    optional extra.
    """
    from . import plots

    if hasattr(plots, name):
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
