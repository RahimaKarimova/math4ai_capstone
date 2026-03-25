"""EPIC 8 — Repeated-seed evaluation on the digits benchmark (Section 7.4).

STORY 8.1: Train both models on digits with seeds {0,1,2,3,4}, report
test accuracy and test cross-entropy per seed, then compute
mean x̄ ± 2.776 · s / √5  (95 % CI, t-distribution with df=4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from src.data_loader import load_digits_split
    import src.softmax_model as sm
    from src.neural_network import OneHiddenLayerNN
    from src.metrics import accuracy, mean_cross_entropy
except ImportError:
    from data_loader import load_digits_split
    import softmax_model as sm
    from neural_network import OneHiddenLayerNN
    from metrics import accuracy, mean_cross_entropy

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
_candidate_sp = BASE_DIR / "starter_pack"
if _candidate_sp.exists() and _candidate_sp.is_dir():
    RESULTS_DIR: Path = _candidate_sp / "results"
else:
    RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration selected on validation cross-entropy only (never test)
# ---------------------------------------------------------------------------
# Softmax: lr=0.05, batch=64, epochs=200, l2_lambda=1e-4  (Section 4.2)
# NN:      hidden_width=32, optimizer="sgd", lr=0.05, batch=64,
#          epochs=200, l2_lambda=1e-4                       (Section 4.2)

SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776  # t_{0.025, df=4} for 95 % CI with 5 observations


def run_softmax_seed(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    """Train softmax on digits with a single seed; return (test_acc, test_ce)."""
    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]
    rng = np.random.default_rng(seed)
    W, b = sm.init_softmax_params(num_classes, input_dim, rng=rng)
    sm.train(
        X_train, y_train, X_val, y_val, W, b,
        l2_lambda=1e-4,
        learning_rate=0.05,
        batch_size=64,
        epochs=200,
        rng=np.random.default_rng(seed),
    )
    probs = sm.predict_proba(X_test, W, b)
    acc = accuracy(y_test, np.argmax(probs, axis=1))
    ce = mean_cross_entropy(y_test, probs)
    return float(acc), float(ce)


def run_nn_seed(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    """Train NN on digits with a single seed; return (test_acc, test_ce)."""
    nn = OneHiddenLayerNN(
        input_dim=X_train.shape[1],
        num_classes=int(np.max(y_train)) + 1,
        hidden_width=32,
        seed=seed,
        l2_lambda=1e-4,
    )
    nn.train(
        X_train, y_train, X_val, y_val,
        optimizer="sgd",
        epochs=200,
        batch_size=64,
        seed=seed,
    )
    probs = nn.predict_proba(X_test)
    acc = accuracy(y_test, np.argmax(probs, axis=1))
    ce = mean_cross_entropy(y_test, probs)
    return float(acc), float(ce)


def ci_string(values):
    """Return 'mean ± half-width' string for a 95 % CI (t, df=4)."""
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    half = T_CRIT * std / np.sqrt(len(arr))
    return mean, std, half


def main():
    print("=" * 65)
    print("EPIC 8 / STORY 8.1 — Repeated-seed evaluation on digits")
    print("=" * 65)

    X_train, X_val, X_test, y_train, y_val, y_test = load_digits_split()

    # ------------------------------------------------------------------
    # Softmax configuration (chosen on validation CE, not test)
    # ------------------------------------------------------------------
    print("\nSelected softmax config (validation-based):")
    print("  lr=0.05, batch=64, epochs=200, l2_lambda=1e-4")

    sm_accs, sm_ces = [], []
    for s in SEEDS:
        acc, ce = run_softmax_seed(
            X_train, y_train, X_val, y_val, X_test, y_test, s,
        )
        sm_accs.append(acc)
        sm_ces.append(ce)
        print(f"  seed {s}: acc={acc:.4f}, ce={ce:.4f}")

    # ------------------------------------------------------------------
    # NN configuration (chosen on validation CE, not test)
    # ------------------------------------------------------------------
    print("\nSelected NN config (validation-based):")
    print("  hidden_width=32, optimizer=sgd, lr=0.05, batch=64, "
          "epochs=200, l2_lambda=1e-4")

    nn_accs, nn_ces = [], []
    for s in SEEDS:
        acc, ce = run_nn_seed(
            X_train, y_train, X_val, y_val, X_test, y_test, s,
        )
        nn_accs.append(acc)
        nn_ces.append(ce)
        print(f"  seed {s}: acc={acc:.4f}, ce={ce:.4f}")

    # ------------------------------------------------------------------
    # Compute 95 % confidence intervals
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("95 % Confidence Intervals  (t_{0.025,4} = 2.776, n = 5)")
    print("=" * 65)

    rows = []  # for CSV

    for model_name, accs, ces in [
        ("softmax", sm_accs, sm_ces),
        ("nn_width32_sgd", nn_accs, nn_ces),
    ]:
        acc_mean, acc_std, acc_hw = ci_string(accs)
        ce_mean, ce_std, ce_hw = ci_string(ces)

        print(f"\n  {model_name}:")
        print(f"    accuracy:       {acc_mean:.4f} +/- {acc_hw:.4f}  "
              f"[{acc_mean - acc_hw:.4f}, {acc_mean + acc_hw:.4f}]")
        print(f"    cross-entropy:  {ce_mean:.4f} +/- {ce_hw:.4f}  "
              f"[{ce_mean - ce_hw:.4f}, {ce_mean + ce_hw:.4f}]")

        rows.append({
            "model": model_name,
            "metric": "accuracy",
            "mean": acc_mean,
            "std": acc_std,
            "ci_half_width": acc_hw,
            "ci_lower": acc_mean - acc_hw,
            "ci_upper": acc_mean + acc_hw,
            "seed_0": accs[0], "seed_1": accs[1], "seed_2": accs[2],
            "seed_3": accs[3], "seed_4": accs[4],
        })
        rows.append({
            "model": model_name,
            "metric": "cross_entropy",
            "mean": ce_mean,
            "std": ce_std,
            "ci_half_width": ce_hw,
            "ci_lower": ce_mean - ce_hw,
            "ci_upper": ce_mean + ce_hw,
            "seed_0": ces[0], "seed_1": ces[1], "seed_2": ces[2],
            "seed_3": ces[3], "seed_4": ces[4],
        })

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = RESULTS_DIR / "repeated_seed_table.csv"
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format="%.6f")
    except ImportError:
        # Fallback: write CSV manually
        cols = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                vals = []
                for c in cols:
                    v = r[c]
                    vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
                f.write(",".join(vals) + "\n")
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # Interpretation paragraph
    # ------------------------------------------------------------------
    sm_acc_mean, _, sm_acc_hw = ci_string(sm_accs)
    nn_acc_mean, _, nn_acc_hw = ci_string(nn_accs)
    sm_ce_mean, _, sm_ce_hw = ci_string(sm_ces)
    nn_ce_mean, _, nn_ce_hw = ci_string(nn_ces)

    acc_overlap = (sm_acc_mean - sm_acc_hw) <= (nn_acc_mean + nn_acc_hw) and \
                  (nn_acc_mean - nn_acc_hw) <= (sm_acc_mean + sm_acc_hw)
    ce_overlap = (sm_ce_mean - sm_ce_hw) <= (nn_ce_mean + nn_ce_hw) and \
                 (nn_ce_mean - nn_ce_hw) <= (sm_ce_mean + sm_ce_hw)

    print("\n" + "=" * 65)
    print("Interpretation")
    print("=" * 65)

    # Determine which model is better on accuracy
    if nn_acc_mean > sm_acc_mean:
        better_acc = "neural network"
    else:
        better_acc = "softmax regression"

    print(f"""
  Over 5 random seeds, the {better_acc} achieves the higher mean test
  accuracy ({max(nn_acc_mean, sm_acc_mean):.4f} vs {min(nn_acc_mean, sm_acc_mean):.4f}).

  Accuracy CIs {'overlap' if acc_overlap else 'do NOT overlap'}: \
softmax [{sm_acc_mean - sm_acc_hw:.4f}, {sm_acc_mean + sm_acc_hw:.4f}] vs \
NN [{nn_acc_mean - nn_acc_hw:.4f}, {nn_acc_mean + nn_acc_hw:.4f}].
  Cross-entropy CIs {'overlap' if ce_overlap else 'do NOT overlap'}: \
softmax [{sm_ce_mean - sm_ce_hw:.4f}, {sm_ce_mean + sm_ce_hw:.4f}] vs \
NN [{nn_ce_mean - nn_ce_hw:.4f}, {nn_ce_mean + nn_ce_hw:.4f}].
""")

    if acc_overlap:
        print("  Because the accuracy confidence intervals overlap, we cannot")
        print("  conclude at the 95 % level that one model is strictly superior")
        print("  to the other on this benchmark. The variability across seeds is")
        print("  comparable to the gap between models, suggesting both perform")
        print("  similarly on the digits task under this training protocol.")
    else:
        print(f"  Because the accuracy confidence intervals do NOT overlap, the")
        print(f"  {better_acc} is statistically significantly better at the 95 %")
        print(f"  level. The result is stable across seeds — seed-to-seed")
        print(f"  variability is small relative to the inter-model gap.")

    print()


if __name__ == "__main__":
    main()
