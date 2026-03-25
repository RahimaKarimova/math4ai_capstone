"""EPIC 7 — Required ablations (Section 7.3 of the capstone handout).

Story 7.1: Capacity ablation on moons (hidden widths {2, 8, 32}).
Story 7.2: Optimizer study on digits (SGD, Momentum, Adam).
Story 7.3: Failure-case analysis (width-2 under-capacity on moons).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.data_loader import load_synthetic, load_digits_split
    from src.neural_network import OneHiddenLayerNN
    from src.metrics import accuracy, mean_cross_entropy
    from src.plotting import plot_decision_boundary, plot_learning_curves
except ImportError:
    from data_loader import load_synthetic, load_digits_split
    from neural_network import OneHiddenLayerNN
    from metrics import accuracy, mean_cross_entropy
    from plotting import plot_decision_boundary, plot_learning_curves

# ---------------------------------------------------------------------------
# Output directories (same logic as core_experiments.py)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
_candidate_sp = BASE_DIR / "starter_pack"
if _candidate_sp.exists() and _candidate_sp.is_dir():
    FIG_DIR: Path = _candidate_sp / "figures"
    RESULTS_DIR: Path = _candidate_sp / "results"
else:
    FIG_DIR = BASE_DIR / "figures"
    RESULTS_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ablation_records: list[tuple] = []


# ===================================================================
# STORY 7.1 — Capacity ablation on moons (widths 2, 8, 32)
# ===================================================================

def story_capacity_ablation():
    """Train NN on moons with hidden widths {2, 8, 32} and produce boundary plots."""
    print("=" * 60)
    print("STORY 7.1 — Capacity ablation on moons")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_synthetic("moons")
    widths = [2, 8, 32]
    models = {}

    for w in widths:
        print(f"\n  Training NN with hidden_width={w} ...")
        nn = OneHiddenLayerNN(
            input_dim=X_train.shape[1],
            num_classes=int(np.max(y_train)) + 1,
            hidden_width=w,
            seed=0,
            l2_lambda=0.0,
        )
        train_hist, val_hist = nn.train(
            X_train, y_train,
            X_val, y_val,
            optimizer="sgd",
            epochs=200,
            batch_size=64,
            seed=0,
        )
        # Test metrics
        probs = nn.predict_proba(X_test)
        acc = accuracy(y_test, np.argmax(probs, axis=1))
        ce = mean_cross_entropy(y_test, probs)
        models[w] = nn
        ablation_records.append(("moons", f"nn_width{w}_sgd", float(acc), float(ce)))
        print(f"    width={w}: test_acc={acc:.4f}, test_ce={ce:.4f}")

        # Individual boundary figure
        fig_single, ax_single = plt.subplots(1, 1, figsize=(5, 4))
        plot_decision_boundary(
            lambda X, _nn=nn: _nn.predict(X),
            X_train, y_train, ax_single,
            title=f"Moons: NN (width {w})",
        )
        fig_single.tight_layout()
        fig_single.savefig(FIG_DIR / f"moons_nn_width{w}.png")
        plt.close(fig_single)

    # Three-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, w in enumerate(widths):
        nn = models[w]
        plot_decision_boundary(
            lambda X, _nn=nn: _nn.predict(X),
            X_train, y_train, axes[i],
            title=f"width = {w}",
        )
    fig.suptitle("Capacity ablation on moons (hidden widths 2, 8, 32)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "moons_capacity_ablation.png")
    plt.close(fig)
    print("\n  Saved moons_capacity_ablation.png")

    return models


# ===================================================================
# STORY 7.2 — Optimizer study on digits (SGD, Momentum, Adam)
# ===================================================================

def story_optimizer_study():
    """Compare SGD, Momentum, Adam on digits NN (width 32, Section 4.2 defaults)."""
    print("\n" + "=" * 60)
    print("STORY 7.2 — Optimizer study on digits")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_digits_split()

    optimizers = ["sgd", "momentum", "adam"]
    histories = {}

    for opt_name in optimizers:
        print(f"\n  Training NN with optimizer={opt_name} ...")
        nn = OneHiddenLayerNN(
            input_dim=X_train.shape[1],
            num_classes=int(np.max(y_train)) + 1,
            hidden_width=32,
            seed=0,
            l2_lambda=1e-4,
        )
        train_hist, val_hist = nn.train(
            X_train, y_train,
            X_val, y_val,
            optimizer=opt_name,
            epochs=200,
            batch_size=64,
            seed=0,
        )
        # Test metrics (using best-val-CE checkpoint already restored by nn.train)
        probs = nn.predict_proba(X_test)
        acc = accuracy(y_test, np.argmax(probs, axis=1))
        ce = mean_cross_entropy(y_test, probs)

        # Best val epoch
        best_epoch = int(np.argmin(val_hist)) + 1
        best_val_ce = float(min(val_hist))

        histories[opt_name] = (train_hist, val_hist)
        ablation_records.append(("digits", f"nn_width32_{opt_name}", float(acc), float(ce)))
        print(f"    {opt_name}: test_acc={acc:.4f}, test_ce={ce:.4f}, "
              f"best_val_ce={best_val_ce:.4f} @ epoch {best_epoch}")

        # Individual learning curve figure
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 4))
        plot_learning_curves(
            train_hist, val_hist, ax_single,
            title=f"Digits: NN ({opt_name})",
        )
        fig_single.tight_layout()
        fig_single.savefig(FIG_DIR / f"digits_nn_curves_{opt_name}.png")
        plt.close(fig_single)

    # Combined optimizer comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs_arr = np.arange(1, 201)
    for opt_name in optimizers:
        train_hist, val_hist = histories[opt_name]
        ax.plot(epochs_arr, train_hist, label=f"{opt_name} train", linewidth=1.0)
        ax.plot(epochs_arr, val_hist, label=f"{opt_name} val", linewidth=1.0, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Optimizer comparison on digits (NN, width 32)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "digits_optimizer_comparison.png")
    plt.close(fig)
    print("\n  Saved digits_optimizer_comparison.png")

    return histories


# ===================================================================
# STORY 7.3 — Failure-case analysis
# ===================================================================

def story_failure_case(capacity_models: dict | None = None):
    """Analyse the width-2 under-capacity failure on moons.

    The width-2 NN has only 2 hidden units, which means it can produce
    at most 2 piecewise-linear folds of the input space through tanh.
    The moons dataset requires a curved boundary that separates two
    interleaving half-circles.  With only 2 hidden neurons the model
    cannot represent the necessary nonlinear partition and effectively
    collapses to a near-linear decision boundary — similar to softmax
    regression.

    This is an under-capacity failure: the hypothesis class is too small
    to capture the target geometry, regardless of optimizer or training
    budget.
    """
    print("\n" + "=" * 60)
    print("STORY 7.3 — Failure-case analysis (width-2 on moons)")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_synthetic("moons")

    # If width-2 model wasn't passed in, train it
    if capacity_models and 2 in capacity_models:
        nn2 = capacity_models[2]
    else:
        nn2 = OneHiddenLayerNN(
            input_dim=X_train.shape[1],
            num_classes=int(np.max(y_train)) + 1,
            hidden_width=2,
            seed=0,
            l2_lambda=0.0,
        )
        nn2.train(
            X_train, y_train, X_val, y_val,
            optimizer="sgd", epochs=200, batch_size=64, seed=0,
        )

    probs2 = nn2.predict_proba(X_test)
    acc2 = accuracy(y_test, np.argmax(probs2, axis=1))
    ce2 = mean_cross_entropy(y_test, probs2)

    # Also get width-32 for comparison
    if capacity_models and 32 in capacity_models:
        nn32 = capacity_models[32]
    else:
        nn32 = OneHiddenLayerNN(
            input_dim=X_train.shape[1],
            num_classes=int(np.max(y_train)) + 1,
            hidden_width=32,
            seed=0,
            l2_lambda=0.0,
        )
        nn32.train(
            X_train, y_train, X_val, y_val,
            optimizer="sgd", epochs=200, batch_size=64, seed=0,
        )

    probs32 = nn32.predict_proba(X_test)
    acc32 = accuracy(y_test, np.argmax(probs32, axis=1))
    ce32 = mean_cross_entropy(y_test, probs32)

    # Failure-case figure: width-2 vs width-32 side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_decision_boundary(
        lambda X: nn2.predict(X),
        X_train, y_train, axes[0],
        title=f"width 2 — acc {acc2:.2f} (FAILURE)",
    )
    plot_decision_boundary(
        lambda X: nn32.predict(X),
        X_train, y_train, axes[1],
        title=f"width 32 — acc {acc32:.2f}",
    )
    fig.suptitle("Failure case: under-capacity (width 2 vs 32 on moons)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_case_width2_moons.png")
    plt.close(fig)

    print(f"\n  width  2: test_acc={acc2:.4f}, test_ce={ce2:.4f}")
    print(f"  width 32: test_acc={acc32:.4f}, test_ce={ce32:.4f}")
    print("\n  Mechanistic explanation:")
    print("  The width-2 network has only 2 hidden neurons. Each tanh unit")
    print("  partitions the 2D input space with a single oriented sigmoid-like")
    print("  fold. Two such folds cannot carve out the curved boundary needed")
    print("  to separate two interleaving half-circles (moons). The model is")
    print("  therefore restricted to a nearly linear decision region, matching")
    print("  softmax regression performance. This is an under-capacity failure:")
    print("  the hypothesis class is too small, not an optimization failure.")
    print("\n  Saved failure_case_width2_moons.png")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    capacity_models = story_capacity_ablation()
    story_optimizer_study()
    story_failure_case(capacity_models)

    # Save ablation results to CSV
    print("\n" + "=" * 60)
    print("Ablation results summary")
    print("=" * 60)
    try:
        import pandas as pd

        df = pd.DataFrame(
            ablation_records,
            columns=["dataset", "model", "test_accuracy", "test_cross_entropy"],
        )
        print(df.to_string(index=False, float_format="{:0.4f}".format))
        csv_path = RESULTS_DIR / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    except ImportError:
        print("\nSummary (pandas not installed):")
        for rec in ablation_records:
            ds, model, acc, ce = rec
            print(f"  {ds:8s} | {model:20s} | acc={acc:.4f}, ce={ce:.4f}")
