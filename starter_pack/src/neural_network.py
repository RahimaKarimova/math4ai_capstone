"""One-hidden-layer neural network with forward and backward passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from .optimizers import build_optimizer
except ImportError:  # pragma: no cover - fallback when run as script/module file.
    from optimizers import build_optimizer


ArrayLike = np.ndarray


def stable_softmax_rows(scores: ArrayLike) -> ArrayLike:
    """Row-wise numerically stable softmax."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


@dataclass
class OneHiddenLayerNN:
    """One-hidden-layer NN with tanh activations and softmax output."""

    input_dim: int
    num_classes: int
    hidden_width: int = 32
    seed: int = 0
    l2_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive.")

        rng = np.random.default_rng(self.seed)

        # Assignment-required initialization: standard normal scaled by 0.01.
        self.W1 = 0.01 * rng.standard_normal((self.hidden_width, self.input_dim))
        self.b1 = np.zeros((self.hidden_width,), dtype=np.float64)
        self.W2 = 0.01 * rng.standard_normal((self.num_classes, self.hidden_width))
        self.b2 = np.zeros((self.num_classes,), dtype=np.float64)

    def forward(self, X: ArrayLike):
        """Compute Z1, H, and S for a batch X of shape (n, d)."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n, d).")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has d={X.shape[1]}, expected d={self.input_dim}."
            )

        Z1 = X @ self.W1.T + self.b1
        H = np.tanh(Z1)
        S = H @ self.W2.T + self.b2
        return Z1, H, S

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Return class probabilities P of shape (n, k)."""
        _, _, S = self.forward(X)
        return stable_softmax_rows(S)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Return class predictions of shape (n,)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def parameters(self):
        """Return mutable parameter arrays in a fixed order."""
        return [self.W1, self.b1, self.W2, self.b2]

    def _mean_cross_entropy(self, X: ArrayLike, y: ArrayLike) -> float:
        """Average cross-entropy only (no regularization)."""
        probs = self.predict_proba(X)
        n = len(y)
        correct_probs = probs[np.arange(n), y]
        return float(-np.mean(np.log(np.clip(correct_probs, 1e-12, None))))

    def _cross_entropy_with_l2(self, X: ArrayLike, y: ArrayLike) -> float:
        """Average cross-entropy plus (λ/2)||W||² on weights (same scaling as softmax)."""
        ce = self._mean_cross_entropy(X, y)
        l2_penalty = 0.5 * self.l2_lambda * (
            np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2)
        )
        return float(ce + l2_penalty)

    def compute_gradients(self, X: ArrayLike, y: ArrayLike):
        """Compute gradients for W1, b1, W2, b2 using vectorized backprop."""
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels.")
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of examples.")
        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("y contains labels outside [0, num_classes).")

        n = len(X)
        Z1, H, S = self.forward(X)
        P = stable_softmax_rows(S)

        Y = np.zeros((n, self.num_classes), dtype=np.float64)
        Y[np.arange(n), y] = 1.0

        # dS = (P - Y) / n
        dS = (P - Y) / n
        if dS.shape != (n, self.num_classes):
            raise RuntimeError("dS has invalid shape.")

        # dW2 = dS^T H + lambda * W2  (L2 term is (lambda/2)||W||^2)
        dW2 = dS.T @ H + (self.l2_lambda * self.W2)
        if dW2.shape != (self.num_classes, self.hidden_width):
            raise RuntimeError("dW2 has invalid shape.")

        # db2 = dS^T 1  (equivalent to row-wise sum over batch)
        db2 = np.sum(dS, axis=0)
        if db2.shape != (self.num_classes,):
            raise RuntimeError("db2 has invalid shape.")

        # dZ1 = (dS W2) ⊙ (1 - H ⊙ H)
        dZ1 = (dS @ self.W2) * (1.0 - H * H)
        if dZ1.shape != (n, self.hidden_width):
            raise RuntimeError("dZ1 has invalid shape.")

        # dW1 = dZ1^T X + lambda * W1
        dW1 = dZ1.T @ X + (self.l2_lambda * self.W1)
        if dW1.shape != (self.hidden_width, self.input_dim):
            raise RuntimeError("dW1 has invalid shape.")

        # db1 = dZ1^T 1  (equivalent to row-wise sum over batch)
        db1 = np.sum(dZ1, axis=0)
        if db1.shape != (self.hidden_width,):
            raise RuntimeError("db1 has invalid shape.")

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def train(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        *,
        optimizer: str = "sgd",
        epochs: int = 200,
        batch_size: int = 64,
        seed: int = 0,
    ):
        """Train end-to-end and return (train_loss_history, val_loss_history).

        Training loss each epoch is CE + (λ/2)||W||². Validation logging and
        best-checkpoint selection use **validation cross-entropy only**, per the
        capstone protocol.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("X_train and X_val must be 2D arrays.")
        if y_train.ndim != 1 or y_val.ndim != 1:
            raise ValueError("y_train and y_val must be 1D arrays.")
        if X_train.shape[1] != self.input_dim or X_val.shape[1] != self.input_dim:
            raise ValueError("Feature dimension mismatch with model input_dim.")
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Feature and label arrays must have matching lengths.")

        opt = build_optimizer(optimizer)
        rng = np.random.default_rng(seed)

        best_val_loss = np.inf
        best_params = [p.copy() for p in self.parameters()]
        train_loss_history = []
        val_loss_history = []

        n_train = len(X_train)
        for _ in range(epochs):
            # Shuffle each epoch as required.
            order = rng.permutation(n_train)
            Xs = X_train[order]
            ys = y_train[order]

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                Xb = Xs[start:end]
                yb = ys[start:end]

                # Forward pass and batch loss computation for training dynamics.
                _ = self._cross_entropy_with_l2(Xb, yb)
                grads = self.compute_gradients(Xb, yb)
                grad_list = [grads["W1"], grads["b1"], grads["W2"], grads["b2"]]
                opt.step(self.parameters(), grad_list)

            train_loss = self._cross_entropy_with_l2(X_train, y_train)
            val_loss = self._mean_cross_entropy(X_val, y_val)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = [p.copy() for p in self.parameters()]

        # Restore best validation-loss checkpoint after training budget ends.
        self.W1[...] = best_params[0]
        self.b1[...] = best_params[1]
        self.W2[...] = best_params[2]
        self.b2[...] = best_params[3]

        return train_loss_history, val_loss_history

