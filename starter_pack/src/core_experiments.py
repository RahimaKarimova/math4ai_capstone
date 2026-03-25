"""
Core experiment runner for comparing the softmax baseline with a
one-hidden-layer neural network across the provided datasets.

This script loads the datasets, trains both models with the same
protocol, generates decision boundary plots / learning curves,
and saves the evaluation results.

Dependencies used in this script::

* ``data_loader`` for deterministic dataset loading
* ``softmax_model`` functions for parameter initialization, forward pass,
  gradient computation and training
* ``neural_network.OneHiddenLayerNN`` for the second model
* ``metrics`` for accuracy and cross‑entropy metrics
* ``plotting`` for generating decision boundaries and learning curves

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    # Assume we're running inside the starter_pack; import relative modules
    from src.data_loader import load_synthetic, load_digits_split
    import src.softmax_model as sm
    from src.neural_network import OneHiddenLayerNN
    from src.metrics import accuracy, mean_cross_entropy
    from src.plotting import plot_decision_boundary, plot_learning_curves
except ImportError:
    # Fallback: import modules directly if starter_pack/src is on PYTHONPATH
    from data_loader import load_synthetic, load_digits_split
    import softmax_model as sm
    from neural_network import OneHiddenLayerNN
    from metrics import accuracy, mean_cross_entropy
    from plotting import plot_decision_boundary, plot_learning_curves

#
# Output directories
#
# Resolve output directories relative to the project root.
# Figures and experiment results are stored under /figures and /results.

BASE_DIR: Path = Path(__file__).resolve().parent.parent
_candidate_sp = BASE_DIR / "starter_pack"
if _candidate_sp.exists() and _candidate_sp.is_dir():
    FIG_DIR: Path = _candidate_sp / "figures"
    RESULTS_DIR: Path = _candidate_sp / "results"
else:
    FIG_DIR = BASE_DIR / "figures"
    RESULTS_DIR = BASE_DIR / "results"
# Create directories if they do not exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # Stores evaluation metrics for all experiments. 
results_records: list[tuple[str, str, float, float]] = []


def train_softmax_on_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    l2_lambda: float = 0.0,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    epochs: int = 200,
    seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list]]:
    """Train softmax regression and return final parameters and loss history.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test : arrays
        Dataset splits.
    l2_lambda, learning_rate, batch_size, epochs : float / int
        Training hyperparameters.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    (W, b), (train_hist, val_hist)
        Final learned parameters and history of train/val loss per epoch.
    """
    # Determine model dimensions
    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]
    W, b = sm.init_softmax_params(num_classes, input_dim, rng=np.random.default_rng(seed))
    train_hist, val_hist = sm.train(
        X_train,
        y_train,
        X_val,
        y_val,
        W,
        b,
        l2_lambda=l2_lambda,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        rng=np.random.default_rng(seed),
    )
    # Evaluate on test set
    probs_test = sm.predict_proba(X_test, W, b)
    y_pred = np.argmax(probs_test, axis=1)
    test_acc = accuracy(y_test, y_pred)
    test_ce = mean_cross_entropy(y_test, probs_test)
    return (W, b), (train_hist, val_hist)


def train_nn_on_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_width: int = 32,
    optimizer: str = "sgd",
    l2_lambda: float = 0.0,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    epochs: int = 200,
    seed: int = 0,
) -> Tuple[OneHiddenLayerNN, Tuple[list, list]]:
    """Train a one‑hidden‑layer neural network and return the model and history.

    The ``optimizer`` argument selects between 'sgd', 'momentum', and 'adam'.
    For Adam the default learning rate is overridden to 0.001 as specified
    in the assignment; otherwise ``learning_rate`` is used.
    """
    input_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    nn = OneHiddenLayerNN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_width=hidden_width,
        seed=seed,
        l2_lambda=l2_lambda,
    )
    # Adjust learning rate for Adam
    if optimizer.lower().strip() == "adam":
        lr = 0.001
    else:
        lr = learning_rate
    train_hist, val_hist = nn.train(
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    # Evaluate on test set
    probs_test = nn.predict_proba(X_test)
    y_pred = np.argmax(probs_test, axis=1)
    test_acc = accuracy(y_test, y_pred)
    test_ce = mean_cross_entropy(y_test, probs_test)
    return nn, (train_hist, val_hist)


def story_gaussian():
    """Run STORY 6.1: compare models on the linear Gaussian synthetic dataset."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_synthetic("linear_gaussian")
    # Softmax regression
    (W_s, b_s), _ = train_softmax_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=200,
        learning_rate=0.05,
        batch_size=64,
    )
    # Neural network (SGD, width 32)
    nn, _ = train_nn_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        hidden_width=32,
        optimizer="sgd",
        epochs=200,
        batch_size=64,
    )
    # Compute and record test metrics
    # Softmax metrics
    probs_softmax = sm.predict_proba(X_test, W_s, b_s)
    acc_softmax = accuracy(y_test, np.argmax(probs_softmax, axis=1))
    ce_softmax = mean_cross_entropy(y_test, probs_softmax)
    results_records.append(("gaussian", "softmax", float(acc_softmax), float(ce_softmax)))
    # Neural network metrics
    probs_nn = nn.predict_proba(X_test)
    acc_nn = accuracy(y_test, np.argmax(probs_nn, axis=1))
    ce_nn = mean_cross_entropy(y_test, probs_nn)
    results_records.append(("gaussian", "nn_width32_sgd", float(acc_nn), float(ce_nn)))

    # Decision boundary plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Softmax boundary
    pred_fn_softmax = lambda X: np.argmax(sm.predict_proba(X, W_s, b_s), axis=1)
    plot_decision_boundary(
        pred_fn_softmax, X_train, y_train, axes[0], title="Gaussian: softmax"
    )
    # NN boundary
    pred_fn_nn = lambda X: nn.predict(X)
    plot_decision_boundary(
        pred_fn_nn, X_train, y_train, axes[1], title="Gaussian: NN (width 32)"
    )
    fig.suptitle("Gaussian comparison (softmax vs. NN)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gaussian_comparison.png")
    plt.close(fig)
    print("Saved Gaussian comparison figure.")


def story_moons():
    """Run STORY 6.2: compare models on the nonlinear moons synthetic dataset."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_synthetic("moons")
    # Softmax regression
    (W_s, b_s), _ = train_softmax_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=200,
        learning_rate=0.05,
        batch_size=64,
    )
    # Neural network (SGD, width 32)
    nn, _ = train_nn_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        hidden_width=32,
        optimizer="sgd",
        epochs=200,
        batch_size=64,
    )
    # Compute and record test metrics
    probs_softmax = sm.predict_proba(X_test, W_s, b_s)
    acc_softmax = accuracy(y_test, np.argmax(probs_softmax, axis=1))
    ce_softmax = mean_cross_entropy(y_test, probs_softmax)
    results_records.append(("moons", "softmax", float(acc_softmax), float(ce_softmax)))
    probs_nn = nn.predict_proba(X_test)
    acc_nn = accuracy(y_test, np.argmax(probs_nn, axis=1))
    ce_nn = mean_cross_entropy(y_test, probs_nn)
    results_records.append(("moons", "nn_width32_sgd", float(acc_nn), float(ce_nn)))

    # Decision boundary plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pred_fn_softmax = lambda X: np.argmax(sm.predict_proba(X, W_s, b_s), axis=1)
    plot_decision_boundary(
        pred_fn_softmax, X_train, y_train, axes[0], title="Moons: softmax"
    )
    pred_fn_nn = lambda X: nn.predict(X)
    plot_decision_boundary(
        pred_fn_nn, X_train, y_train, axes[1], title="Moons: NN (width 32)"
    )
    fig.suptitle("Moons comparison (softmax vs. NN)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "moons_comparison.png")
    plt.close(fig)
    print("Saved Moons comparison figure.")


def story_digits():
    """Run STORY 6.3: compare models on the digits benchmark."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_digits_split()
    # Softmax regression (Section 4.2 specification)
    (W_s, b_s), (train_hist_s, val_hist_s) = train_softmax_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        l2_lambda=1e-4,
        learning_rate=0.05,
        batch_size=64,
        epochs=200,
    )
    # Neural network (SGD, width 32) with identical protocol
    nn, (train_hist_nn, val_hist_nn) = train_nn_on_dataset(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        hidden_width=32,
        optimizer="sgd",
        l2_lambda=1e-4,
        learning_rate=0.05,
        batch_size=64,
        epochs=200,
    )
    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_learning_curves(
        train_hist_s, val_hist_s, axes[0], title="Digits: softmax"
    )
    plot_learning_curves(
        train_hist_nn, val_hist_nn, axes[1], title="Digits: NN (SGD, width 32)"
    )
    # Combined plot
    epochs = np.arange(1, len(train_hist_s) + 1)
    axes[2].plot(epochs, train_hist_s, label="softmax train", linewidth=1.0)
    axes[2].plot(epochs, val_hist_s, label="softmax val", linewidth=1.0)
    axes[2].plot(epochs, train_hist_nn, label="NN train", linewidth=1.0)
    axes[2].plot(epochs, val_hist_nn, label="NN val", linewidth=1.0)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("cross‑entropy loss")
    axes[2].set_title("Digits training comparison")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "digits_training_comparison.png")
    plt.close(fig)
    print("Saved digits comparison figures.")

    # Compute and record test metrics for the digits dataset.  As with the
    # synthetic tasks, we evaluate both the softmax regression and the
    # neural network on the held‑out test set.  Recording these values
    # allows us to summarise all Epic 6 experiments in a single table at
    # the end of the script.
    # Softmax metrics
    probs_softmax = sm.predict_proba(X_test, W_s, b_s)
    acc_softmax = accuracy(y_test, np.argmax(probs_softmax, axis=1))
    ce_softmax = mean_cross_entropy(y_test, probs_softmax)
    results_records.append(("digits", "softmax", float(acc_softmax), float(ce_softmax)))
    # Neural network metrics
    probs_nn = nn.predict_proba(X_test)
    acc_nn = accuracy(y_test, np.argmax(probs_nn, axis=1))
    ce_nn = mean_cross_entropy(y_test, probs_nn)
    results_records.append(("digits", "nn_width32_sgd", float(acc_nn), float(ce_nn)))


if __name__ == "__main__":
    # Run all stories sequentially.  Comment out calls if you only need
    # individual experiments.  Note: running all three stories can take
    # several minutes depending on your hardware.
    story_gaussian()
    story_moons()
    story_digits()
    # After all experiments finish, aggregate the metrics
    # and export them to a CSV summary file.
    try:
        import pandas as pd

        df_results = pd.DataFrame(
            results_records,
            columns=["dataset", "model", "test_accuracy", "test_cross_entropy"],
        )
        # Print formatted table to the console.
        print(
            df_results.to_string(index=False, float_format="{:0.4f}".format)
        )
        
        # Persist the results to a CSV file. 
        csv_path = RESULTS_DIR / "core_experiments_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    except ImportError:
        # If pandas is not available, fall back to printing the raw list
        print("\nSummary of Epic 6 results (pandas not installed):")
        for rec in results_records:
            dataset, model, acc, ce = rec
            print(
                f"{dataset:8s} | {model:15s} | accuracy = {acc:.4f}, cross‑entropy = {ce:.4f}"
            )
            