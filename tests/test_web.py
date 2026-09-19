"""Phase 14 -- tests for the demo API.

The web server is the only network-facing code in this repository, so it is the
only place where a bug has a security dimension rather than just a correctness
one. Two things are tested here:

**Validation.** Every rejection path, because the interesting failures are the
inputs that *should* be refused and are not. A model handed NaN returns NaN
confidently, which is a far more confusing outcome than an error.

**Serving.** That predictions come out of our engine correctly, that path
traversal is blocked, and that oversized bodies are refused before being read.

The handler is exercised through a real socket on an ephemeral port rather
than by calling methods directly -- the traversal and body-limit defences live
in HTTP handling, so testing below that layer would test the wrong thing.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from web.server import BadRequest, Handler, ServedModel, parse_pixels

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "mnist_checkpoints" / "best.json"

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="no trained checkpoint; run examples/train_mnist.py first",
)


# ======================================================================
# validation -- no server needed
# ======================================================================


class TestParsePixels:
    def test_accepts_a_valid_image(self):
        out = parse_pixels({"pixels": [0.5] * 784})
        assert out.shape == (784,)
        assert out.dtype == np.float64

    def test_accepts_the_boundary_values(self):
        assert parse_pixels({"pixels": [0.0] * 784}).sum() == 0.0
        assert parse_pixels({"pixels": [1.0] * 784}).sum() == 784.0

    def test_rejects_a_non_object_body(self):
        with pytest.raises(BadRequest, match="JSON object"):
            parse_pixels([1, 2, 3])

    def test_rejects_a_missing_key(self):
        with pytest.raises(BadRequest, match="missing"):
            parse_pixels({})

    def test_rejects_a_non_array(self):
        with pytest.raises(BadRequest, match="must be an array"):
            parse_pixels({"pixels": "784 pixels honest"})

    @pytest.mark.parametrize("n", [0, 1, 783, 785, 1000])
    def test_rejects_the_wrong_length(self, n):
        with pytest.raises(BadRequest, match="exactly 784"):
            parse_pixels({"pixels": [0.5] * n})

    def test_rejects_non_numbers(self):
        with pytest.raises(BadRequest, match="only numbers"):
            parse_pixels({"pixels": ["a"] * 784})

    def test_rejects_nan(self):
        """The important one. NaN propagates through every matmul and produces
        ten NaN probabilities, which looks like our engine broke."""
        with pytest.raises(BadRequest, match="NaN or infinity"):
            parse_pixels({"pixels": [float("nan")] * 784})

    def test_rejects_infinity(self):
        with pytest.raises(BadRequest, match="NaN or infinity"):
            parse_pixels({"pixels": [float("inf")] * 784})

    @pytest.mark.parametrize("value", [-0.001, 1.001, 255.0, -5.0])
    def test_rejects_values_outside_the_unit_range(self, value):
        """Pixels were scaled to [0,1] for training; 0-255 input would hand the
        first layer activations 255x too large."""
        with pytest.raises(BadRequest, match=r"\[0, 1\]"):
            parse_pixels({"pixels": [value] * 784})

    def test_a_single_bad_pixel_is_enough_to_reject(self):
        pixels = [0.5] * 784
        pixels[400] = float("nan")
        with pytest.raises(BadRequest):
            parse_pixels({"pixels": pixels})


# ======================================================================
# the served model
# ======================================================================


@pytest.fixture(scope="module")
def model():
    if not CHECKPOINT.exists():
        pytest.skip("no trained checkpoint")
    return ServedModel(CHECKPOINT)


@needs_checkpoint
class TestServedModel:
    def test_describe_reports_the_architecture(self, model):
        info = model.describe()
        assert info["sizes"][0] == 784
        assert info["sizes"][-1] == 10
        assert info["parameters"] > 0
        assert len(info["layers"]) == len(info["sizes"]) - 1

    def test_predict_returns_a_valid_distribution(self, model):
        out = model.predict(np.full(784, 0.5))
        assert len(out["probabilities"]) == 10
        assert all(0.0 <= p <= 1.0 for p in out["probabilities"])
        assert sum(out["probabilities"]) == pytest.approx(1.0)

    def test_prediction_is_the_argmax_of_the_probabilities(self, model):
        out = model.predict(np.full(784, 0.3))
        assert out["prediction"] == max(
            range(10), key=lambda k: out["probabilities"][k]
        )
        assert out["confidence"] == pytest.approx(max(out["probabilities"]))

    def test_runner_up_is_second(self, model):
        out = model.predict(np.full(784, 0.3))
        assert out["runner_up"] != out["prediction"]
        assert out["runner_up_confidence"] <= out["confidence"]

    def test_activations_match_the_layer_sizes(self, model):
        out = model.predict(np.zeros(784))
        sizes = model.describe()["sizes"][1:]
        assert [a["size"] for a in out["activations"]] == sizes

    def test_relu_layers_report_some_silent_units(self, model):
        """Sanity check that we are reading real activations: a trained ReLU
        network zeroes a substantial fraction of its hidden units."""
        out = model.predict(np.full(784, 0.5))
        hidden = out["activations"][:-1]
        assert any(a["zero_fraction"] > 0.0 for a in hidden)
        assert all(0.0 <= a["zero_fraction"] <= 1.0 for a in out["activations"])

    def test_it_actually_recognises_mnist_digits(self, model):
        """End to end: the served model must reproduce training accuracy.

        Guards against loading the weights into the wrong shape, transposing a
        matrix, or serving an untrained model -- all of which would still
        return a well-formed distribution.
        """
        from nabla.data.mnist import load_mnist

        _, test = load_mnist(n_train=10, n_test=100, seed=0)
        correct = sum(
            model.predict(test.x[i])["prediction"] == int(test.y[i])
            for i in range(100)
        )
        assert correct >= 90, f"only {correct}/100 correct -- weights are wrong"

    def test_is_deterministic(self, model):
        a = model.predict(np.full(784, 0.42))
        b = model.predict(np.full(784, 0.42))
        assert a["probabilities"] == b["probabilities"]

    def test_infer_sizes_recovers_architecture_from_weight_shapes(self):
        """Older checkpoints have no architecture metadata; the shapes are
        enough to reconstruct it."""
        state = {
            "layers.0.W": np.zeros((784, 128)),
            "layers.1.W": np.zeros((128, 64)),
            "layers.2.W": np.zeros((64, 10)),
        }
        assert ServedModel._infer_sizes(state) == [784, 128, 64, 10]


# ======================================================================
# the HTTP surface
# ======================================================================


@pytest.fixture(scope="module")
def server(model):
    """A real server on an ephemeral port, torn down after the module."""
    Handler.model = model
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        body = err.read()
        try:
            return err.code, json.loads(body)
        except json.JSONDecodeError:
            return err.code, body


def post(base: str, path: str, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@needs_checkpoint
class TestEndpoints:
    def test_health(self, server):
        assert get(server, "/api/health") == (200, {"status": "ok"})

    def test_model_endpoint(self, server):
        status, body = get(server, "/api/model")
        assert status == 200
        assert body["sizes"][0] == 784

    def test_predict_endpoint(self, server):
        status, body = post(server, "/api/predict", {"pixels": [0.5] * 784})
        assert status == 200
        assert 0 <= body["prediction"] <= 9
        assert sum(body["probabilities"]) == pytest.approx(1.0)

    def test_an_oversized_body_gets_the_error_not_a_reset(self, server):
        """Refusing a body without reading it resets the connection.

        The size cap itself always worked: nothing was over-allocated. But
        the server replied and closed while the client was still sending, so
        the kernel answered the incoming data with RST and the client saw a
        connection reset instead of "request body too large". The 400 existed
        and never arrived.

        Sent over a raw socket, because urllib reports the reset as a
        transport error and the status line is the thing under test.
        """
        host, port = server.removeprefix("http://").split(":")
        payload = json.dumps({"pixels": [0.5] * 100_000}).encode()
        assert len(payload) > 256 * 1024, "payload must exceed MAX_BODY"

        with socket.create_connection((host, int(port)), timeout=15) as sock:
            sock.sendall(
                b"POST /api/predict HTTP/1.1\r\nHost: t\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + payload
            )
            data = b""
            while chunk := sock.recv(8192):
                data += chunk

        assert int(data.split(b" ")[1]) == 400
        assert b"too large" in data

    @pytest.mark.parametrize(
        "payload",
        [
            {"pixels": [0.5] * 10},
            {"pixels": "nope"},
            {},
            {"pixels": [float("inf")] * 784},
            {"pixels": [2.0] * 784},
        ],
    )
    def test_bad_input_gives_400_with_a_reason(self, server, payload):
        status, body = post(server, "/api/predict", payload)
        assert status == 400
        assert "error" in body and body["error"]

    def test_unknown_endpoints_404(self, server):
        assert get(server, "/api/nonexistent")[0] == 404

    def test_static_files_are_served(self, server):
        for path in ("/", "/style.css", "/app.js"):
            with urllib.request.urlopen(server + path) as response:
                assert response.status == 200
                assert len(response.read()) > 0

    def test_index_declares_nosniff(self, server):
        with urllib.request.urlopen(server + "/") as response:
            assert response.headers["X-Content-Type-Options"] == "nosniff"


@needs_checkpoint
class TestSecurity:
    @staticmethod
    def raw_get(base: str, path: str) -> int:
        """Send an unnormalised path. urllib and curl both collapse `..`
        client-side, so a traversal test through them proves nothing."""
        host, port = base.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            sock.sendall(
                f"GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode()
            )
            data = b""
            while chunk := sock.recv(4096):
                data += chunk
        return int(data.split(b" ")[1])

    @pytest.mark.parametrize(
        "path",
        [
            "/../server.py",
            "/../../etc/passwd",
            "/./../../nabla/core/value.py",
            "/../../../../../../etc/hosts",
        ],
    )
    def test_path_traversal_is_refused(self, server, path):
        assert self.raw_get(server, path) in (403, 404)

    def test_traversal_does_not_leak_file_contents(self, server):
        """The status code could be right while the body is still wrong."""
        host, port = server.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            sock.sendall(
                b"GET /../server.py HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"
            )
            data = b""
            while chunk := sock.recv(4096):
                data += chunk
        assert b"BaseHTTPRequestHandler" not in data
        assert b"forbidden" in data

    def test_oversized_body_is_refused_without_being_read(self, server):
        """An unbounded read() sized by a client-controlled Content-Length is a
        one-line memory exhaustion bug. We must refuse on the header alone."""
        host, port = server.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            # Claim 50 MB, then send almost nothing. A server that trusted the
            # header would block here waiting for data that never arrives.
            sock.sendall(
                b"POST /api/predict HTTP/1.1\r\nHost: t\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 52428800\r\nConnection: close\r\n\r\n"
                b'{"pixels":[0.1'
            )
            data = b""
            while chunk := sock.recv(4096):
                data += chunk
        assert b"400" in data.split(b"\r\n")[0]
        assert b"too large" in data

    def test_no_cors_header_is_advertised(self, server):
        """Same-origin only. A wildcard CORS header would let any page on the
        internet drive a locally-running model."""
        with urllib.request.urlopen(server + "/api/model") as response:
            assert "Access-Control-Allow-Origin" not in response.headers
