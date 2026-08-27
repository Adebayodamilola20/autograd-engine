"""Phase 12 -- loading MNIST.

MNIST is 70,000 handwritten digits: 60,000 for training, 10,000 for testing,
each a 28x28 grayscale image with a label from 0 to 9. It has been the standard
first benchmark for image classification since LeCun et al. published it in
1998, which makes it the right target here -- the numbers are directly
comparable to a very large literature.

For a fully-connected network the images are **flattened** from 28x28 to a
784-vector. That throws away all spatial structure: the model has no idea that
pixel 100 sits next to pixel 101. It still reaches ~98%, which says something
about how much signal there is in raw pixel intensities. (A convolutional
network keeps the structure and reaches ~99.5% -- the gap is exactly the value
of the inductive bias we discarded.)

The IDBX file format
--------------------
A simple binary format, parsed here directly rather than through a library, so
nothing is hidden:

    offset 0: magic number (4 bytes, big-endian)
              0x0803 = 3-D array of unsigned bytes (images)
              0x0801 = 1-D array of unsigned bytes (labels)
    offset 4: dimension sizes, 4 bytes each, big-endian, one per dimension
    then:     the data, row-major, one unsigned byte per pixel

Big-endian is a legacy of the format's age -- worth knowing, because reading it
as little-endian yields a plausible-looking array of nonsense.

Preprocessing
-------------
Pixels arrive as integers in ``[0, 255]`` and are scaled to ``[0, 1]``.

This is not cosmetic. Section §2 of ``value.py`` explains why: the gradient of a
weight is proportional to its input, so a feature of magnitude 255 hands its
weight a gradient 255x larger than a feature of magnitude 1. With raw pixels the
first layer's gradients are enormous, the usable learning rate collapses, and
the network trains far worse. ``experiments/`` measures the difference.

Optionally the data can also be **standardised** to zero mean and unit variance
using the training set's statistics -- and only the training set's, since using
test statistics would leak information from data the model is supposed to have
never seen.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .dataset import Dataset

__all__ = ["load_mnist", "download_mnist", "read_idx", "MNIST_FILES"]

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Yann LeCun's original host now rate-limits aggressively; these are the
# standard mirrors used by PyTorch and TensorFlow.
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


def _fetch(url: str, target: Path, *, timeout: float, quiet: bool) -> None:
    """Download one file to ``target``, atomically, with progress.

    Two details that matter more than they look:

    **Chunked, not ``response.read()``.** The socket timeout applies to each
    individual read. Pulling an entire 9.9 MB body in one call means one slow
    stretch anywhere in the transfer kills it. Reading in 64 KB chunks means
    the timeout only fires if the server genuinely stalls, which is what a
    timeout should mean. It also lets us show progress rather than appearing
    hung for half an hour.

    **Written to ``.part`` and renamed on success.** ``rename`` is atomic
    within a filesystem, so the real filename never exists in a half-written
    state. Without this, an interrupted download leaves a truncated file that
    the "already cached?" check happily accepts on the next run. ``read_idx``
    would catch the corruption eventually, but "your MNIST file is corrupt" is
    a much worse error than simply downloading it again.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "nabla-autograd/0.1"})
    part = target.with_suffix(target.suffix + ".part")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(part, "wb") as handle:
                while chunk := response.read(65536):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if not quiet and total:
                        pct = downloaded / total * 100
                        print(
                            f"\r    {target.name}  {downloaded / 1e6:5.1f} /"
                            f" {total / 1e6:.1f} MB  ({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
        if not quiet:
            print()
        part.replace(target)          # atomic: no partial file under the real name
    finally:
        part.unlink(missing_ok=True)  # never leave debris behind on failure


def download_mnist(
    root: str | Path = DEFAULT_ROOT,
    *,
    quiet: bool = False,
    timeout: float = 60.0,
    attempts: int = 2,
) -> Path:
    """Download the four MNIST files if they are not already cached.

    Returns the directory holding them. Each mirror is tried ``attempts``
    times before moving on, because a transient failure on a slow connection
    is common and is not a reason to give up on an otherwise good mirror.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for filename in MNIST_FILES.values():
        target = root / filename
        if target.exists() and target.stat().st_size > 0:
            continue

        last_error: Exception | None = None
        for attempt in range(attempts):
            for mirror in MIRRORS:
                try:
                    if not quiet:
                        host = mirror.split("/")[2]
                        retry = f"  (attempt {attempt + 1})" if attempt else ""
                        print(f"  downloading {filename} from {host}{retry}")
                    _fetch(mirror + filename, target, timeout=timeout, quiet=quiet)
                    break
                except Exception as err:  # noqa: BLE001 - try the next mirror
                    last_error = err
                    if not quiet:
                        print(f"\r    failed: {type(err).__name__}: {err}")
            else:
                continue      # every mirror failed this round; try again
            break             # a mirror succeeded
        else:
            raise RuntimeError(
                f"Could not download {filename} from any mirror after "
                f"{attempts} attempts. Last error: {last_error}\n\n"
                f"To continue without a working connection, download these "
                f"four files by hand and place them in\n    {root}\n"
                + "".join(f"      {MIRRORS[0]}{name}\n" for name in MNIST_FILES.values())
            )
    return root


def read_idx(path: str | Path) -> np.ndarray:
    """Parse one gzipped IDX file into a NumPy array.

    Implemented directly against the format spec (see the module docstring)
    rather than via a helper library -- it is twenty lines, and reading them is
    more informative than trusting them.
    """
    with gzip.open(path, "rb") as handle:
        # Header: two zero bytes, a data-type code, then the dimension count.
        zero, dtype_code, n_dims = struct.unpack(">HBB", handle.read(4))
        if zero != 0:
            raise ValueError(f"{path}: not an IDX file (leading bytes {zero})")
        if dtype_code != 0x08:
            raise ValueError(
                f"{path}: expected unsigned bytes (0x08), got 0x{dtype_code:02x}"
            )
        # One big-endian 4-byte size per dimension.
        shape = struct.unpack(f">{n_dims}I", handle.read(4 * n_dims))
        data = np.frombuffer(handle.read(), dtype=np.uint8)

    if data.size != int(np.prod(shape)):
        raise ValueError(
            f"{path}: header says {shape} ({np.prod(shape)} values) "
            f"but the file holds {data.size}"
        )
    return data.reshape(shape)


def load_mnist(
    root: str | Path = DEFAULT_ROOT,
    *,
    normalize: bool = True,
    standardize: bool = False,
    flatten: bool = True,
    n_train: int | None = None,
    n_test: int | None = None,
    download: bool = True,
    seed: int | None = 0,
) -> tuple[Dataset, Dataset]:
    """Load MNIST as ``(train, test)`` datasets.

    Parameters
    ----------
    normalize
        Scale pixels from ``[0, 255]`` to ``[0, 1]``. Leave this on -- see the
        module docstring for why raw pixel magnitudes wreck the first layer's
        gradients.
    standardize
        Additionally subtract the mean and divide by the standard deviation,
        both computed on the **training set only**. Using test statistics would
        leak information about data the model must not have seen.
    flatten
        Reshape ``(n, 28, 28)`` to ``(n, 784)``. Required for an MLP.
    n_train, n_test
        Take a random subset. Essential for the scalar engine, which needs
        about 20 seconds per sample on the full architecture.
    seed
        Seeds the subset selection.

    Returns
    -------
    (train, test) : tuple[Dataset, Dataset]

    Examples
    --------
    >>> train, test = load_mnist(n_train=1000, n_test=200)   # doctest: +SKIP
    >>> train.x.shape, train.y.shape                          # doctest: +SKIP
    ((1000, 784), (1000,))
    """
    root = Path(root)
    missing = [f for f in MNIST_FILES.values() if not (root / f).exists()]
    if missing:
        if not download:
            raise FileNotFoundError(
                f"missing MNIST files in {root}: {missing}. "
                "Call with download=True or run nabla.data.mnist.download_mnist()."
            )
        download_mnist(root)

    train_x = read_idx(root / MNIST_FILES["train_images"]).astype(np.float64)
    train_y = read_idx(root / MNIST_FILES["train_labels"]).astype(np.int64)
    test_x = read_idx(root / MNIST_FILES["test_images"]).astype(np.float64)
    test_y = read_idx(root / MNIST_FILES["test_labels"]).astype(np.int64)

    if normalize:
        train_x /= 255.0
        test_x /= 255.0

    if standardize:
        # Training statistics only -- never the test set's.
        mean = train_x.mean()
        std = train_x.std() or 1.0
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

    if flatten:
        train_x = train_x.reshape(len(train_x), -1)
        test_x = test_x.reshape(len(test_x), -1)

    train = Dataset(train_x, train_y, name="mnist/train")
    test = Dataset(test_x, test_y, name="mnist/test")

    if n_train is not None:
        train = train.subset(n_train, seed=seed)
    if n_test is not None:
        test = test.subset(n_test, seed=seed)

    return train, test
