"""Phase 14 -- a small HTTP API serving our own trained network.

Run:
    python web/server.py
    python web/server.py --port 8080 --checkpoint artifacts/mnist_checkpoints/best.json

Then open http://127.0.0.1:8000.

Every prediction this server returns is computed by ``nabla`` -- the same
``TensorMLP`` forward pass, the same ``log_softmax``, the same weights that
``examples/train_mnist.py`` produced. There is no ONNX export, no PyTorch, no
second implementation to drift out of sync. The browser draws pixels, the
server runs our engine, and the numbers that come back are ours.

Why the standard library and not Flask/FastAPI
---------------------------------------------
This server has four endpoints and no auth, no database, no async. The stdlib
``http.server`` covers that completely, and it means ``git clone`` followed by
``python web/server.py`` works with **zero installation** beyond NumPy, which
the engine already required. For a project whose point is that you can read all
of it, a dependency that would triple the install surface to save forty lines
is a bad trade.

The honest caveat: ``http.server`` is explicitly not for production
(single-threaded per request, minimal hardening). This binds to 127.0.0.1 by
default and says so. Exposing it publicly would need a real WSGI/ASGI server
in front, and the ``--host`` flag warns when you change it.

Security notes, since this is the only network-facing code in the repo
---------------------------------------------------------------------
* **Binds to localhost only** unless ``--host`` is given explicitly.
* **Request bodies are capped** (``MAX_BODY``); an unbounded ``read()`` on a
  Content-Length the client controls is a trivial memory-exhaustion bug.
* **Input is fully validated** -- exactly 784 finite numbers in [0, 1]. The
  model would happily consume NaN and return NaN, which is a confusing failure
  rather than a dangerous one, but garbage in should be rejected at the door.
* **Static files are served from a fixed directory** with the resolved path
  checked against it, so ``../../etc/passwd`` cannot escape.
* **No user input ever reaches the filesystem, a shell, or ``eval``.** The only
  thing the request body does is become a NumPy array.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nabla.core.tensor import Tensor  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.training.checkpoint import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "mnist_checkpoints" / "best.json"

MAX_BODY = 256 * 1024  # a 784-float JSON payload is ~15 KB; this is generous
N_PIXELS = 784

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}


# ======================================================================
# the model
# ======================================================================


class ServedModel:
    """Our trained network, plus the introspection the UI wants.

    Holds one ``TensorMLP`` reconstructed from a checkpoint. Predictions run
    the real forward pass and additionally report each layer's activations, so
    the browser can show what the network is actually doing rather than just
    its verdict.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        payload = load_checkpoint(checkpoint_path)
        arch = payload.get("metadata", {}).get("architecture") or {}

        sizes = arch.get("sizes")
        if not sizes:
            # Older checkpoints predate architecture metadata. The layer shapes
            # fully determine the sizes, so recover them rather than refusing.
            sizes = self._infer_sizes(payload["model"])

        self.sizes = list(sizes)
        self.activation = arch.get("activation") or "relu"
        self.model = TensorMLP(
            self.sizes[0],
            self.sizes[1:-1],
            self.sizes[-1],
            activation=self.activation,
            seed=0,
        )
        self.model.load_state_dict(payload["model"])
        self.model.eval()

        self.epoch = payload.get("epoch", 0)
        self.history = payload.get("history", {})
        self.metadata = payload.get("metadata", {})
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def _infer_sizes(state: dict[str, Any]) -> list[int]:
        """Recover ``[784, 128, 64, 10]`` from the weight matrix shapes."""
        weights = sorted(k for k in state if k.endswith(".W"))
        sizes: list[int] = []
        for i, key in enumerate(weights):
            rows = np.asarray(state[key]).shape
            if i == 0:
                sizes.append(int(rows[0]))
            sizes.append(int(rows[1]))
        return sizes

    def describe(self) -> dict[str, Any]:
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                {
                    "index": i,
                    "n_in": layer.n_in,
                    "n_out": layer.n_out,
                    "activation": layer.activation_name,
                    "parameters": layer.num_parameters(),
                    "weight_shape": list(layer.W.shape),
                }
            )
        val_acc = self.history.get("val_acc") or []
        return {
            "sizes": self.sizes,
            "activation": self.activation,
            "parameters": self.model.num_parameters(),
            "layers": layers,
            "epoch": self.epoch,
            "best_val_accuracy": max(val_acc) if val_acc else None,
            "checkpoint": str(self.checkpoint_path.relative_to(ROOT)),
            "engine": "nabla (own reverse-mode autodiff)",
        }

    def predict(self, pixels: np.ndarray) -> dict[str, Any]:
        """Run one image through the network, reporting the whole journey.

        Returns the prediction, the full probability distribution, the raw
        logits, and per-layer activation summaries -- everything the UI needs
        to show *how* the answer was reached, not just what it was.
        """
        started = time.perf_counter()

        x = Tensor(pixels.reshape(1, -1))
        activations = []
        current = x
        for i, layer in enumerate(self.model.layers):
            current = layer(current)
            values = np.asarray(current.data).ravel()
            activations.append(
                {
                    "layer": i,
                    "size": int(values.size),
                    "mean": float(values.mean()),
                    "max": float(values.max()),
                    "min": float(values.min()),
                    # Fraction of units outputting exactly zero. For ReLU this
                    # is the sparsity of the representation -- typically well
                    # over half, which is a real and often surprising fact
                    # about how these networks encode things.
                    "zero_fraction": float((values == 0).mean()),
                    "preview": [float(v) for v in values[:64]],
                }
            )

        logits = np.asarray(current.data).ravel()
        probabilities = np.asarray(current.softmax(axis=-1).data).ravel()
        prediction = int(probabilities.argmax())
        ranked = np.argsort(probabilities)[::-1]

        return {
            "prediction": prediction,
            "confidence": float(probabilities[prediction]),
            "probabilities": [float(p) for p in probabilities],
            "logits": [float(v) for v in logits],
            "runner_up": int(ranked[1]),
            "runner_up_confidence": float(probabilities[ranked[1]]),
            "activations": activations,
            "milliseconds": (time.perf_counter() - started) * 1000.0,
        }


# ======================================================================
# validation
# ======================================================================


class BadRequest(Exception):
    """Raised for anything the client got wrong. Becomes a 400."""


def parse_pixels(payload: Any) -> np.ndarray:
    """Validate a request body into a ``(784,)`` float array in [0, 1].

    Deliberately strict. The model would accept NaN and return NaN, and a user
    staring at ten NaN probabilities has no way to tell whether they broke the
    request or we broke the engine. Rejecting at the boundary keeps that
    distinction clear.
    """
    if not isinstance(payload, dict):
        raise BadRequest("body must be a JSON object")

    pixels = payload.get("pixels")
    if pixels is None:
        raise BadRequest("missing 'pixels'")
    if not isinstance(pixels, list):
        raise BadRequest("'pixels' must be an array")
    if len(pixels) != N_PIXELS:
        raise BadRequest(f"'pixels' must have exactly {N_PIXELS} entries, got {len(pixels)}")

    try:
        array = np.asarray(pixels, dtype=np.float64)
    except (TypeError, ValueError):
        raise BadRequest("'pixels' must contain only numbers") from None

    if not np.all(np.isfinite(array)):
        raise BadRequest("'pixels' contains NaN or infinity")
    if array.min() < 0.0 or array.max() > 1.0:
        raise BadRequest("'pixels' values must be in [0, 1]")

    return array


# ======================================================================
# the server
# ======================================================================


class Handler(BaseHTTPRequestHandler):
    server_version = "nabla/0.1"
    model: ServedModel  # injected below

    # -- helpers -------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This API is same-origin only; no CORS header is sent on purpose.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, code: int, message: str) -> None:
        self._drain_request_body()
        self._json(code, {"error": message})

    # How much unread request body to swallow before giving up on a tidy
    # reply. Bounded, because draining is work the client asked us to do, and
    # an unbounded drain is the denial of service the size cap exists to
    # prevent. Memory is not at risk either way (the bytes are read in chunks
    # and discarded); time is.
    #
    # It has to be comfortably larger than MAX_BODY to be worth anything. Set
    # equal to it, the server drains exactly up to the limit and leaves the
    # entire overage unread, which resets the connection anyway and makes the
    # drain pure waste. 8 MB covers any plausible accidental oversend, which
    # is the case that deserves a real error message. A client deliberately
    # sending more than that is abusive, and a reset is the right answer.
    DRAIN_LIMIT = 8 * 1024 * 1024

    def _drain_request_body(self) -> None:
        """Consume the request body we are about to refuse to read.

        Rejecting an oversized POST without reading it leaves the client
        still sending. The server writes its 400 and closes, the kernel sees
        data arriving for a closed socket, and answers with RST. The client
        never gets the response: instead of "request body too large" it sees
        a connection reset, which says nothing about what went wrong.

        So the body has to be consumed before replying, even though it is
        being discarded. This is a general HTTP server obligation rather than
        anything specific to this endpoint.
        """
        if getattr(self, "_body_consumed", False):
            return
        self._body_consumed = True
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return

        # Past the limit, refuse on the header alone and let the connection
        # reset. Draining *part* of an oversized body is the worst of both:
        # the remainder still triggers the reset, so the reply is lost anyway
        # and the reading was wasted. Worse, `Content-Length` is chosen by the
        # client, so a request claiming 50 MB and sending 14 bytes would block
        # here forever. Refusing on the header is the whole point of the cap.
        if remaining > self.DRAIN_LIMIT:
            return

        # Even within the limit the declared length may exceed what is
        # actually sent, so a short timeout keeps a slow or lying client from
        # holding the handler open. Being unable to drain is not fatal: the
        # worst case is the reset we were trying to avoid.
        previous = self.connection.gettimeout()
        self.connection.settimeout(2.0)
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (TimeoutError, OSError):
            pass
        finally:
            self.connection.settimeout(previous)

    def _read_body(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise BadRequest("invalid Content-Length") from None
        if length <= 0:
            raise BadRequest("empty request body")
        if length > MAX_BODY:
            # Never trust a client-supplied length to size an allocation.
            raise BadRequest(f"request body too large (limit {MAX_BODY} bytes)")
        raw = self.rfile.read(length)
        # Read, so there is nothing left for `_error` to drain. Without this
        # the drain would block waiting for bytes the client already sent.
        self._body_consumed = True
        try:
            return json.loads(raw)
        except json.JSONDecodeError as err:
            raise BadRequest(f"invalid JSON: {err.msg}") from None

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming)
        path = self.path.split("?", 1)[0]

        if path == "/api/model":
            self._json(200, self.model.describe())
            return
        if path == "/api/health":
            self._json(200, {"status": "ok"})
            return
        if path.startswith("/api/"):
            self._error(404, f"no such endpoint: {path}")
            return

        self._serve_static("index.html" if path == "/" else path.lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/predict":
            self._error(404, f"no such endpoint: {self.path}")
            return
        try:
            pixels = parse_pixels(self._read_body())
        except BadRequest as err:
            self._error(400, str(err))
            return
        except Exception:  # pragma: no cover - defensive
            self._error(400, "could not read request")
            return

        try:
            self._json(200, self.model.predict(pixels))
        except Exception as err:  # pragma: no cover - defensive
            # Do not leak a traceback to the client; log it instead.
            self.log_error("prediction failed: %r", err)
            self._error(500, "prediction failed")

    # -- static --------------------------------------------------------

    def _serve_static(self, relative: str) -> None:
        """Serve from STATIC only. Any path that resolves outside is refused."""
        candidate = (STATIC / relative).resolve()
        try:
            candidate.relative_to(STATIC.resolve())
        except ValueError:
            # `../` traversal, an absolute path, or a symlink pointing out.
            self._error(403, "forbidden")
            return

        if not candidate.is_file():
            self._error(404, "not found")
            return

        content_type = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send(200, candidate.read_bytes(), content_type)

    def log_message(self, fmt: str, *args: Any) -> None:
        """One tidy line per request instead of the stdlib's noisy default."""
        sys.stderr.write(f"    {self.address_string()}  {fmt % args}\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Serve the nabla MNIST demo.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="listen address; anything other than localhost exposes the demo",
    )
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"\n  No checkpoint at {checkpoint}\n\n"
            "  Train one first:\n"
            "      python examples/train_mnist.py\n",
            file=sys.stderr,
        )
        return 1

    print("\n\033[1mnabla -- MNIST demo server\033[0m")
    print("─" * 62)
    model = ServedModel(checkpoint)
    info = model.describe()
    print(f"    checkpoint    {info['checkpoint']}  (epoch {info['epoch'] + 1})")
    print(f"    architecture  {' → '.join(map(str, info['sizes']))}  ({info['activation']})")
    print(f"    parameters    {info['parameters']:,}")
    if info["best_val_accuracy"]:
        print(f"    val accuracy  {info['best_val_accuracy'] * 100:.2f}%")

    Handler.model = model
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"\n    \033[1mhttp://{args.host}:{args.port}\033[0m")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "\n    \033[1mWarning:\033[0m listening on a non-local address.\n"
            "    http.server is not hardened for public exposure -- put a real\n"
            "    server in front of it if this needs to be reachable."
        )
    print("    Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n    stopped.\n")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
