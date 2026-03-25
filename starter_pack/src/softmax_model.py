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
