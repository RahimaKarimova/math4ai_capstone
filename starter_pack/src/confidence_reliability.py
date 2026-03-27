from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.data_loader import load_digits_split
    import src.softmax_model as sm
    from src.neural_network import OneHiddenLayerNN
except ImportError:
    from data_loader import load_digits_split
    import softmax_model as sm
    from neural_network import OneHiddenLayerNN


BASE_DIR: Path = Path(__file__).resolve().parent.parent
FIG_DIR: Path = BASE_DIR / "figures"
RESULTS_DIR: Path = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Return per-example entropy: -sum_c p(c|x) log p(c|x)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def reliability_bins(
    confidences: np.ndarray,
    correct_mask: np.ndarray,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean confidence, empirical accuracy, and counts per confidence bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_conf = np.zeros(n_bins, dtype=np.float64)
    mean_acc = np.zeros(n_bins, dtype=np.float64)

    # Include confidence==1.0 in the last bin.
    bin_ids = np.digitize(confidences, edges[1:-1], right=False)

    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(np.sum(mask))
        if counts[b] > 0:
            mean_conf[b] = float(np.mean(confidences[mask]))
            mean_acc[b] = float(np.mean(correct_mask[mask]))
        else:
            # Keep empty bins on the diagonal for visual continuity.
            center = 0.5 * (edges[b] + edges[b + 1])
            mean_conf[b] = center
            mean_acc[b] = center

    return mean_conf, mean_acc, counts


def save_reliability_plot(
    confidences: np.ndarray,
    correct_mask: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Create and save a 5-bin reliability plot."""
    mean_conf, mean_acc, counts = reliability_bins(confidences, correct_mask, n_bins=5)
    widths = np.full_like(mean_conf, fill_value=0.18, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        mean_conf,
        mean_acc,
        width=widths,
        color="#3b82f6",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#ef4444", linewidth=1.2, label="perfectly calibrated")

    for x, y, n in zip(mean_conf, mean_acc, counts):
        ax.text(x, min(y + 0.03, 1.0), f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_softmax_digits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train softmax model using the digits protocol in the project."""
    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]
    seed = 0
    W, b = sm.init_softmax_params(num_classes, input_dim, rng=np.random.default_rng(seed))
    sm.train(
        X_train,
        y_train,
        X_val,
        y_val,
        W,
        b,
        l2_lambda=1e-4,
        learning_rate=0.05,
        batch_size=64,
        epochs=200,
        rng=np.random.default_rng(seed),
    )
    return W, b


def train_nn_digits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> OneHiddenLayerNN:
    """Train one-hidden-layer NN using the digits protocol in the project."""
    seed = 0
    nn = OneHiddenLayerNN(
        input_dim=X_train.shape[1],
        num_classes=int(np.max(y_train)) + 1,
        hidden_width=32,
        seed=seed,
        l2_lambda=1e-4,
    )
    nn.train(
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer="sgd",
        epochs=200,
        batch_size=64,
        seed=seed,
    )
    return nn


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_digits_split()

    # Train both models on the same fixed split/protocol.
    W_s, b_s = train_softmax_digits(X_train, y_train, X_val, y_val)
    nn = train_nn_digits(X_train, y_train, X_val, y_val)

    # Full probability matrices.
    probs_softmax = sm.predict_proba(X_test, W_s, b_s)
    probs_nn = nn.predict_proba(X_test)
    np.save(RESULTS_DIR / "digits_test_proba_softmax.npy", probs_softmax)
    np.save(RESULTS_DIR / "digits_test_proba_nn.npy", probs_nn)

    # Confidence and entropy vectors.
    pred_softmax = np.argmax(probs_softmax, axis=1)
    pred_nn = np.argmax(probs_nn, axis=1)
    conf_softmax = np.max(probs_softmax, axis=1)
    conf_nn = np.max(probs_nn, axis=1)
    ent_softmax = predictive_entropy(probs_softmax)
    ent_nn = predictive_entropy(probs_nn)

    correct_softmax = pred_softmax == y_test
    correct_nn = pred_nn == y_test
    np.savez(
        RESULTS_DIR / "digits_uncertainty_softmax.npz",
        confidence=conf_softmax,
        entropy=ent_softmax,
        correct=correct_softmax,
    )
    np.savez(
        RESULTS_DIR / "digits_uncertainty_nn.npz",
        confidence=conf_nn,
        entropy=ent_nn,
        correct=correct_nn,
    )

    # Reliability plots (5 bins).
    save_reliability_plot(
        conf_softmax,
        correct_softmax,
        "Digits reliability (softmax, 5 bins)",
        FIG_DIR / "reliability_softmax.png",
    )
    save_reliability_plot(
        conf_nn,
        correct_nn,
        "Digits reliability (NN, 5 bins)",
        FIG_DIR / "reliability_nn.png",
    )

    # Entropy comparison: correct vs incorrect for both models.
    sm_ent_correct = float(np.mean(ent_softmax[correct_softmax]))
    sm_ent_incorrect = float(np.mean(ent_softmax[~correct_softmax]))
    nn_ent_correct = float(np.mean(ent_nn[correct_nn]))
    nn_ent_incorrect = float(np.mean(ent_nn[~correct_nn]))

    labels = ["Softmax-correct", "Softmax-incorrect", "NN-correct", "NN-incorrect"]
    values = [sm_ent_correct, sm_ent_incorrect, nn_ent_correct, nn_ent_incorrect]
    colors = ["#22c55e", "#ef4444", "#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean predictive entropy")
    ax.set_title("Predictive entropy by correctness (digits test set)")
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_comparison.png", dpi=150)
    plt.close(fig)

    print("Saved probability matrices:")
    print(f"- {RESULTS_DIR / 'digits_test_proba_softmax.npy'}")
    print(f"- {RESULTS_DIR / 'digits_test_proba_nn.npy'}")
    print("Saved uncertainty vectors:")
    print(f"- {RESULTS_DIR / 'digits_uncertainty_softmax.npz'}")
    print(f"- {RESULTS_DIR / 'digits_uncertainty_nn.npz'}")
    print("Saved figures:")
    print(f"- {FIG_DIR / 'reliability_softmax.png'}")
    print(f"- {FIG_DIR / 'reliability_nn.png'}")
    print(f"- {FIG_DIR / 'entropy_comparison.png'}")


if __name__ == "__main__":
    main()
