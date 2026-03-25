"""Multiclass softmax regression — parameter initialization (Story 3.1)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def init_softmax_params(
    num_classes: int,
    input_dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize weight matrix W and bias vector b for softmax regression.

    - W has shape (k, d) with entries drawn i.i.d. from N(0, 1), scaled by 0.01.
    - b has shape (k,) and is all zeros.

    Parameters
    ----------
    num_classes : int
        k, number of classes (rows of W).
    input_dim : int
        d, feature dimension (columns of W).
    rng : np.random.Generator, optional
        Random number generator for reproducible W. If None, uses ``default_rng()``.
    """
    if num_classes < 1 or input_dim < 1:
        raise ValueError("num_classes and input_dim must be positive integers.")

    k, d = num_classes, input_dim
    gen = rng if rng is not None else np.random.default_rng()
    W = 0.01 * gen.standard_normal((k, d))
    b = np.zeros(k, dtype=np.float64)
    return W, b


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Row-wise numerically stable softmax.

    For each row, computes softmax(logits) via subtracting the row maximum
    before ``exp`` to reduce overflow risk.

    Parameters
    ----------
    logits : np.ndarray
        Shape (n, k) class scores, or shape (k,) for a single sample (treated
        as one row).

    Returns
    -------
    np.ndarray
        Same shape as input; each row sums to 1 (within ``1e-6``).
    """
    x = np.asarray(logits, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    m = np.max(x, axis=1, keepdims=True)
    shifted = x - m
    exp_s = np.exp(shifted)
    denom = np.sum(exp_s, axis=1, keepdims=True)
    out = exp_s / denom

    row_sums = np.sum(out, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise AssertionError("stable_softmax: row sums not ~1 within 1e-6.")

    return out.reshape(-1) if single else out


if __name__ == "__main__":
    # Section 3.6 sanity check (Story 3.2 action item)
    example = np.array([1.2, 0.2, -0.4], dtype=np.float64)
    p = stable_softmax(example)
    expected = np.array([0.64, 0.23, 0.13], dtype=np.float64)
    assert np.allclose(p, expected, atol=0.02), (p, expected)
