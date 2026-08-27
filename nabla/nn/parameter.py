"""``Parameter`` -- a ``Value`` that a model is allowed to learn.

Why a separate class at all
---------------------------
Structurally a parameter is just a leaf ``Value``: it has data, it collects a
gradient, it has no parents. So why subclass?

Because *something* has to answer the question "which of the thousands of
values in this graph should the optimiser update?" An MLP's forward pass
creates leaves for the inputs and for every numeric constant, and none of those
should move. The type is the answer: ``Module.parameters()`` walks the model
and returns exactly the ``Parameter`` instances.

PyTorch draws the same distinction for the same reason (``nn.Parameter`` vs a
plain tensor), which is a good sign the abstraction is load-bearing rather than
decorative.

Note the empty ``__slots__``: it lets ``Parameter`` inherit ``Value``'s slots
without reintroducing a per-instance ``__dict__``, preserving the memory
behaviour of decision D7 across the subclass.
"""

from __future__ import annotations

from ..core.value import Value

__all__ = ["Parameter", "Learnable"]


class Learnable:
    """Marker for anything an optimiser is allowed to update.

    ``Module`` registers by *type*, and the tensor engine (Phase 18) needs its
    own parameter class wrapping ``Tensor`` rather than ``Value``. Rather than
    teach ``Module`` about both concrete classes -- and import ``Tensor`` into
    a module that has no other need for it -- both inherit this marker.

    ``__slots__ = ()`` keeps the mix-in layout-compatible with ``Value`` and
    ``Tensor``, which both define real slots; adding any here would raise
    "multiple bases have instance lay-out conflict".
    """

    __slots__ = ()


class Parameter(Value, Learnable):
    """A learnable scalar.

    Examples
    --------
    >>> w = Parameter(0.5, label='w0')
    >>> isinstance(w, Value)
    True
    """

    __slots__ = ()

    def __init__(self, data: float, label: str = "") -> None:
        super().__init__(data, label=label)

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        return f"Parameter({name}data={self.data:.6g}, grad={self.grad:.6g})"
