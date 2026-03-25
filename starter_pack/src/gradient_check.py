"""Finite-difference gradient checks for softmax regression and the one-hidden-layer NN (Story 5.1)."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, MutableMapping, Tuple

import numpy as np

import softmax_model as sm
from neural_network import OneHiddenLayerNN


def numerical_gradients(
    params: MutableMapping[str, np.ndarray],
    loss_fn: Callable[[], float],
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """
    Central-difference gradients for every entry of each array in ``params``.

    ``loss_fn`` must read the **same** ndarray objects as in ``params`` (in-place
    perturbation). Each parameter is restored after its own scan.
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in params.items():
        g = np.zeros_like(arr, dtype=np.float64)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            f_plus = loss_fn()
            arr[idx] = orig - eps
            f_minus = loss_fn()
            arr[idx] = orig
            g[idx] = (f_plus - f_minus) / (2.0 * eps)
        out[name] = g
    return out


def max_relative_error(
    analytical: Mapping[str, np.ndarray],
    numerical: Mapping[str, np.ndarray],
) -> Tuple[float, str]:
    """Return (max relative error, parameter name with that max)."""
    worst = 0.0
    worst_key = ""
    for key in analytical:
        a = np.asarray(analytical[key], dtype=np.float64).ravel()
        n = np.asarray(numerical[key], dtype=np.float64).ravel()
        denom = np.maximum(1e-8, np.abs(a) + np.abs(n))
        rel = np.max(np.abs(a - n) / denom)
        if rel > worst:
            worst = float(rel)
            worst_key = key
    return worst, worst_key


def check_softmax_gradients(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    l2_lambda: float = 0.0,
    eps: float = 1e-5,
    rtol: float = 1e-3,
) -> Tuple[float, str]:
    """
    Compare :func:`softmax_model.compute_gradients` to finite differences on
    :func:`softmax_model.softmax_loss`.
    """
    Xm = np.asarray(X, dtype=np.float64)
    Wm = np.asarray(W, dtype=np.float64)
    bm = np.asarray(b, dtype=np.float64)

    params: MutableMapping[str, np.ndarray] = {"W": Wm, "b": bm}

    def loss() -> float:
        logits = Xm @ Wm.T + bm
        return sm.softmax_loss(logits, y, Wm, l2_lambda)

    analytical = sm.compute_gradients(Xm, y, Wm, bm, l2_lambda)
    numerical = numerical_gradients(params, loss, eps=eps)
    err, key = max_relative_error(analytical, numerical)
    if err > rtol:
        raise AssertionError(
            f"Softmax gradient check failed: max rel error {err:.3e} on '{key}' "
            f"(rtol={rtol}, eps={eps})."
        )
    return err, key


def check_neural_net_gradients(
    model: OneHiddenLayerNN,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
    rtol: float = 1e-3,
) -> Tuple[float, str]:
    """
    Compare :meth:`OneHiddenLayerNN.compute_gradients` to finite differences on
    :meth:`OneHiddenLayerNN._cross_entropy_with_l2`.
    """
    Xm = np.asarray(X, dtype=np.float64)
    params: MutableMapping[str, np.ndarray] = {
        "W1": model.W1,
        "b1": model.b1,
        "W2": model.W2,
        "b2": model.b2,
    }

    def loss() -> float:
        return model._cross_entropy_with_l2(Xm, y)

    analytical = model.compute_gradients(Xm, y)
    numerical = numerical_gradients(params, loss, eps=eps)
    err, key = max_relative_error(analytical, numerical)
    if err > rtol:
        raise AssertionError(
            f"Neural net gradient check failed: max rel error {err:.3e} on '{key}' "
            f"(rtol={rtol}, eps={eps})."
        )
    return err, key


def run_default_checks(
    eps: float = 1e-5,
    softmax_rtol: float = 1e-3,
    nn_rtol: float = 1e-3,
) -> None:
    """Tiny random batches; raises ``AssertionError`` if any check fails."""
    rng = np.random.default_rng(0)

    n, d, k = 6, 4, 3
    Xs = rng.standard_normal((n, d))
    ys = rng.integers(0, k, size=n, dtype=np.intp)
    Ws = rng.standard_normal((k, d)) * 0.05
    bs = rng.standard_normal(k) * 0.05
    check_softmax_gradients(
        Xs, ys, Ws, bs, l2_lambda=0.02, eps=eps, rtol=softmax_rtol
    )

    nn = OneHiddenLayerNN(
        input_dim=d,
        num_classes=k,
        hidden_width=5,
        seed=1,
        l2_lambda=0.01,
    )
    check_neural_net_gradients(nn, Xs, ys, eps=eps, rtol=nn_rtol)


if __name__ == "__main__":
    run_default_checks()
