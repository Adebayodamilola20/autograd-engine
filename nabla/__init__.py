"""nabla -- a reverse-mode automatic differentiation engine, written from scratch.

Named after the gradient operator, nabla, and built to answer one question:
what actually happens when you call ``.backward()``?

The whole library rests on ``nabla.core.value.Value``, a scalar node that
records how it was computed and knows how to hand its gradient back to the
values it came from. Everything above it -- neurons, layers, losses,
optimisers, the training loop -- is a consequence of that one class.

No autodiff framework is used anywhere in the implementation. NumPy appears for
data handling and inside the tensor kernels; PyTorch appears only in
``benchmarks/``, as something to be measured against.

    >>> from nabla import Value
    >>> a, b = Value(2.0, label='a'), Value(3.0, label='b')
    >>> d = a * b + a
    >>> d.backward()
    >>> a.grad, b.grad
    (4.0, 2.0)
"""

from .core.graph import (
    build_edges,
    format_graph,
    graph_size,
    is_topologically_sorted,
    leaves,
    topological_sort,
)
from .core.value import Value

__version__ = "0.1.0"

__all__ = [
    "Value",
    "topological_sort",
    "build_edges",
    "leaves",
    "graph_size",
    "is_topologically_sorted",
    "format_graph",
    "__version__",
]
