"""Shared plotting utilities for the Math4AI capstone."""

import numpy as np
import matplotlib.pyplot as plt


COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
          "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#64748b"]
MESH_STEP = 0.02


def plot_decision_boundary(predict_fn, X, y, ax, title=""):
    """Plot predicted class regions and overlay data points.

    Parameters
    ----------
    predict_fn : callable, takes array of shape (m, 2) and returns class labels of shape (m,)
    X          : array of shape (n, 2), input features
    y          : array of shape (n,), true class labels
    ax         : matplotlib Axes
    title      : str
    """
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, MESH_STEP),
        np.arange(y_min, y_max, MESH_STEP),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predict_fn(grid).reshape(xx.shape)

    classes = np.unique(y)
    k = len(classes)
    cmap_bg = plt.cm.get_cmap("RdBu" if k == 2 else "tab10", k)

    ax.contourf(xx, yy, zz, levels=np.arange(k + 1) - 0.5, cmap=cmap_bg, alpha=0.25)

    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[i % len(COLORS)],
                   edgecolors="k", linewidths=0.4, s=18, label=f"class {c}")

    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def plot_learning_curves(train_losses, val_losses, ax, title=""):
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : list or array, one value per epoch
    val_losses   : list or array, one value per epoch
    ax           : matplotlib Axes
    title        : str
    """
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", linewidth=1.2)
    ax.plot(epochs, val_losses, label="val", linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
