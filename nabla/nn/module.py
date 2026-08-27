"""``Module`` -- the base class every network component inherits.

The problem it solves
---------------------
A trained model is a tree: an ``MLP`` holds ``Layer`` objects, each ``Layer``
holds ``Neuron`` objects, each ``Neuron`` holds weights and a bias. The
optimiser needs a flat list of every learnable scalar in that tree. Writing
that traversal by hand in every class would be repetitive and easy to get
subtly wrong -- forget one layer and it silently never trains.

``Module`` does the bookkeeping once, by intercepting attribute assignment.
When a subclass writes ``self.weights = [...]``, ``__setattr__`` notices the
values are ``Parameter`` objects and records them. ``parameters()`` then walks
the tree recursively.

This is exactly how ``torch.nn.Module`` works, and building it yourself makes
several PyTorch behaviours stop being mysterious:

* why a parameter stored in a plain Python list is sometimes invisible to the
  optimiser (PyTorch needs ``nn.ParameterList``; we handle lists directly),
* why ``model.parameters()`` and ``optimizer.zero_grad()`` are separate calls,
* why ``model.train()`` / ``model.eval()`` propagate to every submodule.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from ..core.value import Value
from .parameter import Learnable, Parameter

__all__ = ["Module"]


class Module:
    """Base class for anything holding parameters or submodules.

    Subclasses must call ``super().__init__()`` before assigning attributes,
    and implement ``forward()``.
    """

    def __init__(self) -> None:
        # object.__setattr__ bypasses our own hook, which would otherwise
        # recurse while these very dicts are being created.
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        """Record parameters and submodules as they are assigned.

        Handles four cases:

        * a ``Parameter``          -> registered directly,
        * a ``Module``             -> registered as a child,
        * a list/tuple of either   -> each item registered as ``name.i``,
        * anything else            -> stored normally, not tracked.

        The list case matters: ``Layer`` naturally holds ``self.neurons = [...]``
        and every one of them must be reachable. PyTorch famously does *not*
        do this (you need ``nn.ModuleList``), and forgetting it is a classic
        silent-failure bug -- so we handle it.
        """
        if "_params" not in self.__dict__:
            raise RuntimeError(
                f"{type(self).__name__} assigned {name!r} before calling "
                "super().__init__(); parameters would not be tracked."
            )

        # A name being reassigned must not leave a stale registration behind.
        self._params.pop(name, None)
        self._modules.pop(name, None)
        for key in [k for k in self._modules if k.startswith(f"{name}.")]:
            del self._modules[key]
        for key in [k for k in self._params if k.startswith(f"{name}.")]:
            del self._params[key]

        if isinstance(value, Learnable):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, (list, tuple)) and value:
            for i, item in enumerate(value):
                if isinstance(item, Learnable):
                    self._params[f"{name}.{i}"] = item
                elif isinstance(item, Module):
                    self._modules[f"{name}.{i}"] = item

        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # traversal
    # ------------------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield ``(qualified_name, parameter)`` for the whole tree.

        Depth-first and deterministic (insertion order), so a saved checkpoint
        always reloads into the same slots.
        """
        for name, param in self._params.items():
            yield (f"{prefix}{name}", param)
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    def parameters(self) -> list[Parameter]:
        """Every learnable scalar in this module and its children.

        This is what gets handed to the optimiser. Anything not in this list
        will never be updated -- which is exactly what we want for inputs and
        constants, and exactly the bug to look for when a layer will not learn.
        """
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield (prefix.rstrip(".") or type(self).__name__, self)
        for name, module in self._modules.items():
            yield from module.named_modules(prefix=f"{prefix}{name}.")

    def modules(self) -> list["Module"]:
        return [m for _, m in self.named_modules()]

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zero.

        Required before each training step, because ``backward()`` accumulates
        (decision D6). Skipping it means step *n* is taken using the sum of the
        gradients from steps 1..n -- the model appears to train, badly, which
        makes the bug hard to spot.

        Note this clears only *parameters*, not the whole graph: intermediate
        nodes are discarded with the graph after each forward pass anyway, so
        clearing them would be wasted work.
        """
        for param in self.parameters():
            param.reset_grad()

    def gradient_stats(self) -> dict[str, float]:
        """Summary statistics over all parameter gradients.

        The fastest health check on a training run. A mean absolute gradient of
        exactly 0 means nothing is flowing; 1e-12 means vanishing; 1e6 means
        exploding, and the next step will overshoot badly.
        """
        grads: list[float] = []
        for p in self.parameters():
            g = p.grad
            if hasattr(g, "shape"):
                # A tensor parameter holds a whole array of gradients.
                grads.extend(np.abs(np.ravel(g)).tolist())
            else:
                grads.append(abs(g))
        if not grads:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0, "zeros": 0}
        return {
            "count": len(grads),
            "mean": sum(grads) / len(grads),
            "max": max(grads),
            "min": min(grads),
            "zeros": sum(1 for g in grads if g == 0.0),
        }

    # ------------------------------------------------------------------
    # modes
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        """Set training mode on this module and all children.

        Nothing in the current library behaves differently between modes --
        we have no dropout or batch-norm. The flag exists because the *shape*
        of the API matters: it is where those layers would hook in, and it
        makes the eventual difference explicit rather than a surprise.
        """
        object.__setattr__(self, "training", mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ------------------------------------------------------------------
    # serialisation (Phase 23)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Parameter values keyed by qualified name, ready to serialise.

        Tensor parameters are converted to nested lists so the result is
        JSON-serialisable without a custom encoder.
        """
        out: dict[str, Any] = {}
        for name, param in self.named_parameters():
            data = param.data
            out[name] = data.tolist() if hasattr(data, "tolist") else float(data)
        return out

    def load_state_dict(self, state: dict[str, float], *, strict: bool = True) -> None:
        """Load parameter values produced by ``state_dict()``.

        With ``strict=True`` a mismatch in keys raises rather than silently
        loading a partially-initialised model -- the failure mode where a
        checkpoint appears to load but half the weights are still random.
        """
        current = dict(self.named_parameters())
        if strict:
            missing = set(current) - set(state)
            unexpected = set(state) - set(current)
            if missing or unexpected:
                raise KeyError(
                    f"state_dict mismatch: {len(missing)} missing "
                    f"(e.g. {sorted(missing)[:3]}), {len(unexpected)} unexpected "
                    f"(e.g. {sorted(unexpected)[:3]})"
                )
        for name, param in current.items():
            if name not in state:
                continue
            if hasattr(param.data, "shape"):
                loaded = np.asarray(state[name], dtype=np.float64)
                if loaded.shape != param.data.shape:
                    raise ValueError(
                        f"{name}: checkpoint has shape {loaded.shape}, "
                        f"model expects {param.data.shape}"
                    )
                param.data = loaded
            else:
                param.data = float(state[name])

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total number of learnable *scalars*.

        Not the number of ``Parameter`` objects: one tensor parameter of shape
        (784, 128) is a single object holding 100,352 learnable numbers. This
        returns the number people mean by "parameter count".
        """
        total = 0
        for p in self.parameters():
            total += int(np.size(p.data))
        return total

    def summary(self) -> str:
        """A per-layer parameter count, in the style of ``model.summary()``.

        Useful for the benchmarks: parameter count is the number that explains
        both memory use and why a scalar engine is slow.
        """
        lines = [
            f"{type(self).__name__}",
            f"{'layer':<28} {'type':<16} {'params':>10}",
            "-" * 56,
        ]
        total = 0
        for name, module in self._modules.items():
            count = module.num_parameters()
            total += count
            lines.append(f"{name:<28} {type(module).__name__:<16} {count:>10,}")
        own = len(self._params)
        if own:
            total += own
            lines.append(f"{'(direct)':<28} {'Parameter':<16} {own:>10,}")
        lines.append("-" * 56)
        lines.append(f"{'total':<28} {'':<16} {total:>10,}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # calling
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.num_parameters()} parameters)"


def as_values(xs: Any) -> list[Value]:
    """Coerce a sequence of numbers or ``Value`` objects into ``Value``s.

    Lets callers write ``model([0.5, -1.2])`` without wrapping by hand, while
    still accepting ``Value`` inputs when gradients with respect to the *input*
    are wanted (adversarial examples, saliency maps).
    """
    return [x if isinstance(x, Value) else Value(x) for x in xs]
