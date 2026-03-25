"""Reusable mini-batch training loop with selectable optimizer."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .optimizers import build_optimizer
except ImportError:  # pragma: no cover - fallback when run as script/module file.
    from optimizers import build_optimizer


ArrayLike = np.ndarray


def iterate_minibatches(
    X: ArrayLike,
    y: ArrayLike,
    batch_size: int,
    *,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """Yield (X_batch, y_batch) mini-batches from arrays."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    n = len(X)
    if len(y) != n:
        raise ValueError("X and y must contain the same number of rows.")

    if rng is None:
        rng = np.random.default_rng(0)

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bidx = indices[start:end]
        yield X[bidx], y[bidx]


def train(
    params: Sequence[ArrayLike],
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    grad_fn: Callable[[Sequence[ArrayLike], ArrayLike, ArrayLike], Sequence[ArrayLike]],
    epochs: int = 200,
    batch_size: int = 64,
    optimizer: str = "sgd",
    X_val: Optional[ArrayLike] = None,
    y_val: Optional[ArrayLike] = None,
    loss_fn: Optional[Callable[[Sequence[ArrayLike], ArrayLike, ArrayLike], float]] = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Train parameters with a selectable optimizer and unified interface.

    Parameters
    ----------
    params:
        Sequence of mutable parameter arrays updated in-place.
    grad_fn:
        Callable returning gradients aligned with `params` for one mini-batch.
    optimizer:
        One of {'sgd', 'momentum', 'adam'}.

    Returns
    -------
    history : dict
        Contains `train_loss` and `val_loss` lists when `loss_fn` is provided.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    opt = build_optimizer(optimizer)
    rng = np.random.default_rng(seed)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        for X_batch, y_batch in iterate_minibatches(
            X_train,
            y_train,
            batch_size,
            shuffle=shuffle,
            rng=rng,
        ):
            grads = grad_fn(params, X_batch, y_batch)
            opt.step(params, grads)

        if loss_fn is not None:
            history["train_loss"].append(float(loss_fn(params, X_train, y_train)))
            if X_val is not None and y_val is not None:
                history["val_loss"].append(float(loss_fn(params, X_val, y_val)))

    return history


def unpack_two_layer_params(params: Sequence[ArrayLike]) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Convenience helper for one-hidden-layer models."""
    if len(params) != 4:
        raise ValueError("Expected four parameter arrays: [W1, b1, W2, b2].")
    W1, b1, W2, b2 = params
    return W1, b1, W2, b2

