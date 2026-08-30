"""The ``nabla`` command-line interface.

``nabla.cli.main:main`` is the entry point installed by ``pyproject.toml``.
The expression parser lives in ``expr`` and is deliberately importable on its
own -- it is the only part with a security story worth testing in isolation.
"""

from .expr import ExpressionError, evaluate, free_variables

__all__ = ["ExpressionError", "evaluate", "free_variables"]
