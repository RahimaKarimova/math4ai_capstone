"""Multiclass softmax regression: init, softmax, loss, vectorized forward (Stories 3.1–3.4)."""

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


def mean_cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean cross-entropy: average over the batch of ``-log p_y``,
    where ``p_y`` is the predicted probability of the true class.

    Parameters
    ----------
    probs : np.ndarray
        Softmax probabilities, shape (n, k) or (k,) for a single example.
    labels : np.ndarray
        True class indices in ``{0, ..., k-1}``, shape (n,) or scalar.
    """
    P = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.intp)

    if P.ndim == 1:
        P = P.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim != 1 or P.shape[0] != y.shape[0]:
        raise ValueError("labels must be 1-D with length matching number of rows in probs.")

    n, k = P.shape
    if np.any((y < 0) | (y >= k)):
        raise ValueError("labels must be in [0, k) for k classes.")

    idx = np.arange(n)
    p_y = P[idx, y]
    p_y = np.clip(p_y, 1e-15, 1.0)
    return float(np.mean(-np.log(p_y)))


def l2_weight_penalty(W: np.ndarray, l2_lambda: float) -> float:
    """
    L2 regularization term ``(λ/2) * ||W||_F^2`` (Frobenius norm squared).
    Applied to weights ``W`` only, not bias.
    """
    if l2_lambda == 0.0:
        return 0.0
    Wm = np.asarray(W, dtype=np.float64)
    return 0.5 * float(l2_lambda) * float(np.sum(Wm * Wm))


def softmax_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    l2_lambda: float = 0.0,
) -> float:
    """
    Full training loss: mean cross-entropy on softmax(logits) plus optional L2 on ``W``.
    """
    probs = stable_softmax(logits)
    return mean_cross_entropy(probs, labels) + l2_weight_penalty(W, l2_lambda)


def predict_proba(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized forward pass: class probabilities for a batch.

    Computes scores ``S = X W^T + \\mathbf{1} b^T`` with ``X`` of shape
    ``(n, d)``, ``W`` of shape ``(k, d)``, ``b`` of shape ``(k,)``, then
    applies :func:`stable_softmax` row-wise to obtain ``P`` of shape
    ``(n, k)``.

    A single sample may be passed as ``X`` with shape ``(d,)``; the return
    shape is then ``(k,)``.
    """
    Xm = np.asarray(X, dtype=np.float64)
    Wm = np.asarray(W, dtype=np.float64)
    bm = np.asarray(b, dtype=np.float64)

    single = Xm.ndim == 1
    if single:
        Xm = Xm.reshape(1, -1)

    n, d = Xm.shape
    k, d_w = Wm.shape
    if d != d_w:
        raise ValueError(f"X has {d} features but W has {d_w} columns.")
    if bm.shape != (k,):
        raise ValueError(f"Expected b of shape ({k},), got {bm.shape}.")

    S = Xm @ Wm.T + bm
    P = stable_softmax(S)
    return P.reshape(-1) if single else P


if __name__ == "__main__":
    # Section 3.6 sanity check (Story 3.2 action item)
    example = np.array([1.2, 0.2, -0.4], dtype=np.float64)
    p = stable_softmax(example)
    expected = np.array([0.64, 0.23, 0.13], dtype=np.float64)
    assert np.allclose(p, expected, atol=0.02), (p, expected)

    # Section 3.6 loss, true class 0, no L2 (Story 3.3 action item)
    W_dummy = np.zeros((3, 1), dtype=np.float64)
    loss_36 = softmax_loss(example, 0, W_dummy, l2_lambda=0.0)
    assert np.isclose(loss_36, 0.45, atol=0.02), loss_36

    # L2: (λ/2) * ||W||_F^2 only on W
    W_ones = np.ones((2, 3), dtype=np.float64)
    pen = l2_weight_penalty(W_ones, l2_lambda=2.0)
    assert np.isclose(pen, 6.0), pen

    # Story 3.4: S = X W^T + 1 b^T, then stable softmax; predict_proba API
    W_i = np.eye(3, dtype=np.float64)
    b_z = np.zeros(3, dtype=np.float64)
    X_batch = np.tile(example.reshape(1, -1), (2, 1))
    P_batch = predict_proba(X_batch, W_i, b_z)
    assert P_batch.shape == (2, 3)
    assert np.allclose(P_batch[0], P_batch[1])
    assert np.allclose(P_batch[0], p)
    P_one = predict_proba(example, W_i, b_z)
    assert P_one.shape == (3,) and np.allclose(P_one, p)
