#!/usr/bin/env python3
"""Inspect all four starter-pack data files: shapes, value ranges, class distributions, and split integrity."""

from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def describe(name, X, y):
    print(f"  {name} — X shape: {X.shape}, y shape: {y.shape}")
    print(f"    X dtype: {X.dtype}, range: [{X.min():.4f}, {X.max():.4f}]")
    classes, counts = np.unique(y, return_counts=True)
    dist = ", ".join(f"class {c}: {n}" for c, n in zip(classes, counts))
    print(f"    y dtype: {y.dtype}, classes: {len(classes)}, distribution: {dist}")

# --- Linear Gaussian ---
section("Linear Gaussian (linear_gaussian.npz)")
lg = np.load(DATA / "linear_gaussian.npz")
for split in ("train", "val", "test"):
    describe(split, lg[f"X_{split}"], lg[f"y_{split}"])

# --- Moons ---
section("Moons (moons.npz)")
mo = np.load(DATA / "moons.npz")
for split in ("train", "val", "test"):
    describe(split, mo[f"X_{split}"], mo[f"y_{split}"])

# --- Digits data ---
section("Digits data (digits_data.npz)")
dd = np.load(DATA / "digits_data.npz")
X_digits, y_digits = dd["X"], dd["y"]
print(f"  X shape: {X_digits.shape}, dtype: {X_digits.dtype}, range: [{X_digits.min():.4f}, {X_digits.max():.4f}]")
classes, counts = np.unique(y_digits, return_counts=True)
dist = ", ".join(f"{c}: {n}" for c, n in zip(classes, counts))
print(f"  y shape: {y_digits.shape}, dtype: {y_digits.dtype}, classes: {len(classes)}")
print(f"  distribution: {dist}")

# --- Digits split indices ---
section("Digits split indices (digits_split_indices.npz)")
si = np.load(DATA / "digits_split_indices.npz")
train_idx, val_idx, test_idx = si["train_idx"], si["val_idx"], si["test_idx"]
print(f"  train_idx: {train_idx.shape}, val_idx: {val_idx.shape}, test_idx: {test_idx.shape}")
print(f"  total indices: {len(train_idx) + len(val_idx) + len(test_idx)}, dataset size: {len(y_digits)}")

# Disjointness check
all_idx = np.concatenate([train_idx, val_idx, test_idx])
n_unique = len(np.unique(all_idx))
print(f"  unique indices: {n_unique}, expected: {len(all_idx)}")
assert n_unique == len(all_idx), "FAIL: indices are NOT disjoint!"
print("  Disjoint check: PASSED")

# Coverage check
assert n_unique == len(y_digits), f"FAIL: indices cover {n_unique} but dataset has {len(y_digits)}"
print("  Coverage check: PASSED (indices cover entire dataset)")

# Show split class distributions
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    cls, cnt = np.unique(y_digits[idx], return_counts=True)
    dist = ", ".join(f"{c}: {n}" for c, n in zip(cls, cnt))
    print(f"  {name} ({len(idx)} samples): {dist}")
