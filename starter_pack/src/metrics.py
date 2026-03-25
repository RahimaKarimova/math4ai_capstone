"""Shared evaluation metrics for the Math4AI capstone."""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of correctly classified examples."""
    return np.mean(y_true == y_pred)


def mean_cross_entropy(y_true, probs):
    """Average negative log-probability of the true class.

    Parameters
    ----------
    y_true : array of shape (n,), integer class labels
    probs  : array of shape (n, k), predicted class probabilities
    """
    n = len(y_true)
    correct_probs = probs[np.arange(n), y_true]
    return -np.mean(np.log(np.clip(correct_probs, 1e-12, None)))


def one_hot(y, k):
    """Convert label vector of shape (n,) to one-hot matrix of shape (n, k)."""
    n = len(y)
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    return Y
