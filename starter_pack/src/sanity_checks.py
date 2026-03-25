"""Training and forward-pass sanity checks for softmax and one-hidden-layer NN (Story 5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

import softmax_model as sm
from data_loader import load_synthetic
from neural_network import OneHiddenLayerNN
from optimizers import SGD


def _assert_finite_params_softmax(W: np.ndarray, b: np.ndarray) -> None:
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise AssertionError("Softmax parameters contain NaN or Inf.")


def _assert_finite_params_nn(model: OneHiddenLayerNN) -> None:
    for name in ("W1", "b1", "W2", "b2"):
        arr = getattr(model, name)
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"Neural net parameter {name} contains NaN or Inf.")


def _load_moons_xy() -> Tuple[np.ndarray, np.ndarray]:
    """All moons splits concatenated; falls back to NumPy-only toy data if files missing."""
    try:
        Xt, Xv, Xte, yt, yv, yte = load_synthetic("moons")
        X = np.vstack([Xt, Xv, Xte]).astype(np.float64, copy=False)
        y = np.concatenate([yt, yv, yte]).astype(np.intp, copy=False)
        return X, y
    except (OSError, FileNotFoundError, ValueError):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((400, 2))
        y = (X[:, 0] + 0.35 * X[:, 1] > 0).astype(np.intp)
        return X, y


def _softmax_batch_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    logits = X @ W.T + b
    return sm.softmax_loss(logits, y, W, l2_lambda=0.0)


def _linearly_separable_2d(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs (k=2) separated along x0 so softmax can fit."""
    n0 = (n + 1) // 2
    n1 = n - n0
    shift0 = np.array([-2.5, 0.0], dtype=np.float64)
    shift1 = np.array([2.5, 0.0], dtype=np.float64)
    X0 = rng.standard_normal((n0, 2)) * 0.25 + shift0
    X1 = rng.standard_normal((n1, 2)) * 0.25 + shift1
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=np.intp), np.ones(n1, dtype=np.intp)])
    return X, y


def check_softmax_tiny_overfit(
    rng: np.random.Generator,
    *,
    n_samples: int = 5,
    steps: int = 300,
    loss_threshold: float = 0.05,
    lr: float = 0.2,
) -> Tuple[str, float]:
    """Train on ``n_samples`` points for ``steps`` full-batch SGD steps; loss must drop below threshold."""
    k, d = 2, 2
    X, y = _linearly_separable_2d(rng, n_samples)
    W, b = sm.init_softmax_params(k, d, rng=rng)
    for _ in range(steps):
        g = sm.compute_gradients(X, y, W, b, l2_lambda=0.0)
        W -= lr * g["W"]
        b -= lr * g["b"]
    loss = _softmax_batch_loss(X, y, W, b)
    if loss >= loss_threshold:
        return "FAIL", float(loss)
    return "PASS", float(loss)


def check_nn_tiny_overfit(
    rng: np.random.Generator,
    *,
    n_samples: int = 5,
    steps: int = 300,
    loss_threshold: float = 0.05,
    lr: float = 0.2,
    hidden_width: int = 32,
) -> Tuple[str, float]:
    k, d = 2, 2
    X, y = _linearly_separable_2d(rng, n_samples)
    model = OneHiddenLayerNN(
        input_dim=d,
        num_classes=k,
        hidden_width=hidden_width,
        seed=int(rng.integers(0, 2**31 - 1)),
        l2_lambda=0.0,
    )
    opt = SGD(lr=lr)
    params = model.parameters()
    for _ in range(steps):
        grads = model.compute_gradients(X, y)
        grad_list = [grads["W1"], grads["b1"], grads["W2"], grads["b2"]]
        opt.step(params, grad_list)
    loss = model._cross_entropy_with_l2(X, y)
    if loss >= loss_threshold:
        return "FAIL", float(loss)
    return "PASS", float(loss)


def check_softmax_loss_decrease(
    rng: np.random.Generator,
    *,
    n_samples: int = 10,
    steps: int = 50,
    lr: float = 0.05,
) -> Tuple[str, float, float]:
    k, d = 2, 2
    X, y = _linearly_separable_2d(rng, n_samples)
    W, b = sm.init_softmax_params(k, d, rng=rng)
    loss0 = _softmax_batch_loss(X, y, W, b)
    for _ in range(steps):
        g = sm.compute_gradients(X, y, W, b, l2_lambda=0.0)
        W -= lr * g["W"]
        b -= lr * g["b"]
    loss1 = _softmax_batch_loss(X, y, W, b)
    if loss1 >= loss0:
        return "FAIL", float(loss0), float(loss1)
    return "PASS", float(loss0), float(loss1)


def check_nn_loss_decrease(
    rng: np.random.Generator,
    *,
    n_samples: int = 10,
    steps: int = 50,
    lr: float = 0.05,
    hidden_width: int = 32,
) -> Tuple[str, float, float]:
    k, d = 2, 2
    X, y = _linearly_separable_2d(rng, n_samples)
    model = OneHiddenLayerNN(
        input_dim=d,
        num_classes=k,
        hidden_width=hidden_width,
        seed=int(rng.integers(0, 2**31 - 1)),
        l2_lambda=0.0,
    )
    opt = SGD(lr=lr)
    params = model.parameters()
    loss0 = model._cross_entropy_with_l2(X, y)
    for _ in range(steps):
        grads = model.compute_gradients(X, y)
        opt.step(params, [grads["W1"], grads["b1"], grads["W2"], grads["b2"]])
    loss1 = model._cross_entropy_with_l2(X, y)
    if loss1 >= loss0:
        return "FAIL", float(loss0), float(loss1)
    return "PASS", float(loss0), float(loss1)


def check_softmax_nan_inf_training(
    rng: np.random.Generator,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.05,
) -> str:
    X, y = _load_moons_xy()
    k = int(np.max(y)) + 1
    d = X.shape[1]
    W, b = sm.init_softmax_params(k, d, rng=rng)
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            g = sm.compute_gradients(X[idx], y[idx], W, b, l2_lambda=0.0)
            W -= lr * g["W"]
            b -= lr * g["b"]
            _assert_finite_params_softmax(W, b)
    return "PASS"


def check_nn_nan_inf_training(
    rng: np.random.Generator,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_width: int = 32,
) -> str:
    X, y = _load_moons_xy()
    k = int(np.max(y)) + 1
    d = X.shape[1]
    model = OneHiddenLayerNN(
        input_dim=d,
        num_classes=k,
        hidden_width=hidden_width,
        seed=0,
        l2_lambda=0.0,
    )
    opt = SGD(lr=0.05)
    params = model.parameters()
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            grads = model.compute_gradients(X[idx], y[idx])
            opt.step(params, [grads["W1"], grads["b1"], grads["W2"], grads["b2"]])
            _assert_finite_params_nn(model)
    return "PASS"


def check_softmax_predict_proba_rowsum(
    rng: np.random.Generator,
    *,
    n_rows: int = 20,
    atol: float = 1e-6,
) -> Tuple[str, float]:
    X, y = _load_moons_xy()
    if len(X) < n_rows:
        raise RuntimeError("Not enough samples for probability-sum check.")
    Xb = X[:n_rows]
    k = int(np.max(y)) + 1
    d = X.shape[1]
    W, b = sm.init_softmax_params(k, d, rng=rng)
    P = sm.predict_proba(Xb, W, b)
    row_sums = np.sum(P, axis=1)
    err = float(np.max(np.abs(row_sums - 1.0)))
    if not np.allclose(row_sums, 1.0, atol=atol):
        return "FAIL", err
    return "PASS", err


def check_nn_predict_proba_rowsum(
    rng: np.random.Generator,
    *,
    n_rows: int = 20,
    atol: float = 1e-6,
    hidden_width: int = 32,
) -> Tuple[str, float]:
    X, y = _load_moons_xy()
    if len(X) < n_rows:
        raise RuntimeError("Not enough samples for probability-sum check.")
    Xb = X[:n_rows]
    k = int(np.max(y)) + 1
    d = X.shape[1]
    model = OneHiddenLayerNN(
        input_dim=d,
        num_classes=k,
        hidden_width=hidden_width,
        seed=0,
        l2_lambda=0.0,
    )
    P = model.predict_proba(Xb)
    row_sums = np.sum(P, axis=1)
    err = float(np.max(np.abs(row_sums - 1.0)))
    if not np.allclose(row_sums, 1.0, atol=atol):
        return "FAIL", err
    return "PASS", err


@dataclass
class SanityReport:
    softmax: Dict[str, Any] = field(default_factory=dict)
    neural_network: Dict[str, Any] = field(default_factory=dict)


def run_all_checks(seed: int = 123) -> SanityReport:
    rng = np.random.default_rng(seed)
    report = SanityReport()

    st, loss = check_softmax_tiny_overfit(rng)
    report.softmax["tiny_overfit"] = {"status": st, "final_train_loss": loss}

    nt, nloss = check_nn_tiny_overfit(rng)
    report.neural_network["tiny_overfit"] = {"status": nt, "final_train_loss": nloss}

    ss, l0, l1 = check_softmax_loss_decrease(rng)
    report.softmax["loss_decrease"] = {"status": ss, "loss_initial": l0, "loss_final": l1}

    ns, nl0, nl1 = check_nn_loss_decrease(rng)
    report.neural_network["loss_decrease"] = {"status": ns, "loss_initial": nl0, "loss_final": nl1}

    report.softmax["nan_inf_training"] = {"status": check_softmax_nan_inf_training(rng)}
    report.neural_network["nan_inf_training"] = {"status": check_nn_nan_inf_training(rng)}

    ps, pe = check_softmax_predict_proba_rowsum(rng)
    report.softmax["predict_proba_row_sums"] = {"status": ps, "max_abs_error": pe}

    pn, pne = check_nn_predict_proba_rowsum(rng)
    report.neural_network["predict_proba_row_sums"] = {"status": pn, "max_abs_error": pne}

    return report


def assert_all_pass(report: SanityReport) -> None:
    for model_name, block in (("softmax", report.softmax), ("neural_network", report.neural_network)):
        for key, val in block.items():
            if isinstance(val, dict):
                st = val.get("status")
                if st is None:
                    continue
            else:
                st = val
            if st != "PASS":
                raise AssertionError(f"{model_name}.{key}: {val}")


if __name__ == "__main__":
    rep = run_all_checks()
    assert_all_pass(rep)
    print("All Story 5.2 sanity checks passed.")
