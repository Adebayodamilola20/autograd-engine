"""The MNIST loader, tested offline.

Real MNIST is 11 MB and needs the network, so these tests write tiny IDX
files in the exact on-disk format and fake the HTTP layer. That covers the
parts that break in practice: header parsing and validation, the cache check,
mirror fallback and retries, atomic downloads that leave no partial file, and
the normalise / standardise / flatten / subset options.
"""

from __future__ import annotations

import functools
import gzip
import io
import struct

import numpy as np
import pytest

from nabla.data import mnist
from nabla.data.mnist import MNIST_FILES, download_mnist, load_mnist, read_idx


def idx_bytes(array: np.ndarray, dtype_code: int = 0x08, zero: int = 0) -> bytes:
    header = struct.pack(">HBB", zero, dtype_code, array.ndim)
    header += struct.pack(f">{array.ndim}I", *array.shape)
    return gzip.compress(header + array.astype(np.uint8).tobytes())


def write_fake_mnist(root, n_train=12, n_test=5, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {
        "train_images": rng.integers(0, 256, (n_train, 28, 28)),
        "train_labels": np.arange(n_train) % 10,
        "test_images": rng.integers(0, 256, (n_test, 28, 28)),
        "test_labels": (np.arange(n_test) + 3) % 10,
    }
    root.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        (root / MNIST_FILES[key]).write_bytes(idx_bytes(arr))
    return arrays


# ---------------------------------------------------------------- read_idx


def test_read_idx_round_trips_shape_and_values(tmp_path):
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    path = tmp_path / "a.gz"
    path.write_bytes(idx_bytes(arr))
    out = read_idx(path)
    assert out.shape == (2, 3, 4) and out.dtype == np.uint8
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("kwargs, message", [
    ({"zero": 7}, "not an IDX file"),
    ({"dtype_code": 0x0D}, "expected unsigned bytes"),
])
def test_read_idx_rejects_bad_headers(tmp_path, kwargs, message):
    path = tmp_path / "bad.gz"
    path.write_bytes(idx_bytes(np.zeros((2, 2)), **kwargs))
    with pytest.raises(ValueError, match=message):
        read_idx(path)


def test_read_idx_catches_truncated_files(tmp_path):
    good = gzip.decompress(idx_bytes(np.zeros((4, 4))))
    path = tmp_path / "short.gz"
    path.write_bytes(gzip.compress(good[:-3]))
    with pytest.raises(ValueError, match="header says"):
        read_idx(path)


# ---------------------------------------------------------------- load_mnist


def test_load_mnist_defaults(tmp_path):
    arrays = write_fake_mnist(tmp_path)
    train, test = load_mnist(tmp_path, download=False, seed=None)
    assert train.x.shape == (12, 784) and test.x.shape == (5, 784)
    assert 0.0 <= train.x.min() and train.x.max() <= 1.0
    assert np.array_equal(train.y, arrays["train_labels"])


def test_standardize_uses_training_statistics_only(tmp_path):
    write_fake_mnist(tmp_path)
    train, test = load_mnist(tmp_path, download=False, standardize=True, seed=None)
    assert abs(train.x.mean()) < 1e-9 and abs(train.x.std() - 1.0) < 1e-9
    # the test set is shifted by the *training* mean, so it is not exactly 0
    assert abs(test.x.mean()) > 1e-6


def test_unflattened_images_and_raw_pixels(tmp_path):
    arrays = write_fake_mnist(tmp_path)
    train, _ = load_mnist(tmp_path, download=False, flatten=False, normalize=False, seed=None)
    assert train.x.shape == (12, 28, 28)
    assert np.array_equal(train.x, arrays["train_images"])   # raw 0..255 pixels


def test_subsets_are_reproducible(tmp_path):
    write_fake_mnist(tmp_path)
    a, _ = load_mnist(tmp_path, download=False, n_train=5, n_test=2, seed=1)
    b, _ = load_mnist(tmp_path, download=False, n_train=5, n_test=2, seed=1)
    assert len(a) == 5 and np.array_equal(a.y, b.y)


def test_missing_files_without_download_explain_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        load_mnist(tmp_path / "empty", download=False)


# ---------------------------------------------------------------- downloading


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_falls_back_to_the_next_mirror(tmp_path, monkeypatch):
    payload = idx_bytes(np.zeros((1, 2, 2)))
    calls = []

    def fake_urlopen(request, timeout):
        url = request.full_url
        calls.append(url)
        if url.startswith(mnist.MIRRORS[0]):
            raise OSError("first mirror down")
        return FakeResponse(payload)

    monkeypatch.setattr(mnist.urllib.request, "urlopen", fake_urlopen)
    root = download_mnist(tmp_path, quiet=False)
    for name in MNIST_FILES.values():
        assert (root / name).read_bytes() == payload
        assert not (root / f"{name}.part").exists()
    assert sum(u.startswith(mnist.MIRRORS[1]) for u in calls) == 4


def test_cached_files_are_not_downloaded_again(tmp_path, monkeypatch):
    write_fake_mnist(tmp_path)
    monkeypatch.setattr(mnist.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("should not download"))
    assert download_mnist(tmp_path, quiet=True) == tmp_path


def test_every_mirror_failing_gives_manual_instructions(tmp_path, monkeypatch):
    def always_fail(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr(mnist.urllib.request, "urlopen", always_fail)
    with pytest.raises(RuntimeError, match="download these"):
        download_mnist(tmp_path, quiet=True, attempts=2)
    assert not list(tmp_path.glob("*.part"))   # no debris left behind


def test_load_mnist_downloads_when_files_are_missing(tmp_path, monkeypatch):
    fake = {}
    rng = np.random.default_rng(0)
    fake[MNIST_FILES["train_images"]] = idx_bytes(rng.integers(0, 256, (6, 28, 28)))
    fake[MNIST_FILES["train_labels"]] = idx_bytes(np.arange(6))
    fake[MNIST_FILES["test_images"]] = idx_bytes(rng.integers(0, 256, (2, 28, 28)))
    fake[MNIST_FILES["test_labels"]] = idx_bytes(np.arange(2))
    monkeypatch.setattr(mnist.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(fake[request.full_url.rsplit("/", 1)[1]]))
    monkeypatch.setattr(mnist, "download_mnist", functools.partial(download_mnist, quiet=True))
    train, test = load_mnist(tmp_path, seed=None)
    assert len(train) == 6 and len(test) == 2
