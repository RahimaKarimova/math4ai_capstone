"""Data loading and splitting utilities for the Math4AI capstone."""

from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_digits_split():
    """Load digits data and apply the fixed split indices.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    dd = np.load(DATA_DIR / "digits_data.npz")
    X, y = dd["X"], dd["y"]

    si = np.load(DATA_DIR / "digits_split_indices.npz")
    train_idx, val_idx, test_idx = si["train_idx"], si["val_idx"], si["test_idx"]

    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]


def load_synthetic(name):
    """Load a pre-split synthetic dataset ('linear_gaussian' or 'moons').

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    data = np.load(DATA_DIR / f"{name}.npz")
    return (
        data["X_train"], data["X_val"], data["X_test"],
        data["y_train"], data["y_val"], data["y_test"],
    )
