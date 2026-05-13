"""
MNIST data loader — no ML libraries required.
Downloads the four IDX binary files from a public mirror, caches them
to ./data/mnist/, and returns NumPy arrays.
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

# Google's hosted copy is reliable and doesn't require credentials
_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
_CACHE_DIR = Path(__file__).parent / "data" / "mnist"


def _download(filename: str) -> Path:
    """Download a gzipped MNIST file if not already cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / filename
    if dest.exists():
        return dest
    url = _BASE_URL + filename
    print(f"Downloading {filename} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def _read_images(path: Path) -> np.ndarray:
    """Parse IDX3-ubyte image file → float32 array (n, 784)."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    """Parse IDX1-ubyte label file → int array (n,)."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int32)


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded float32 matrix."""
    n = labels.shape[0]
    oh = np.zeros((n, num_classes), dtype=np.float32)
    oh[np.arange(n), labels] = 1.0
    return oh


def load_mnist() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Download (if needed) and return MNIST arrays.

    Returns
    -------
    X_train : float32 (60000, 784)  — pixel values in [0, 1]
    y_train : float32 (60000, 10)   — one-hot labels
    X_test  : float32 (10000, 784)
    y_test  : float32 (10000, 10)
    y_train_int : int32 (60000,)    — raw class indices
    y_test_int  : int32 (10000,)
    """
    train_img_path  = _download(_FILES["train_images"])
    train_lbl_path  = _download(_FILES["train_labels"])
    test_img_path   = _download(_FILES["test_images"])
    test_lbl_path   = _download(_FILES["test_labels"])

    X_train      = _read_images(train_img_path)
    y_train_int  = _read_labels(train_lbl_path)
    X_test       = _read_images(test_img_path)
    y_test_int   = _read_labels(test_lbl_path)

    y_train = one_hot(y_train_int)
    y_test  = one_hot(y_test_int)

    print(
        f"Loaded MNIST — train: {X_train.shape}, test: {X_test.shape}"
    )
    return X_train, y_train, X_test, y_test, y_train_int, y_test_int
