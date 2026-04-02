# Math4AI Final Capstone

## Team

| Name | Role |
|------|------|
| Aisel Mamedova | Softmax regression, core experiments |
| Rəhimə Kərimova | Neural network, ablation experiments |
| Şəmistan Hüseynov | Experiment orchestration and plots |
| Gülnur Məmmədova | Report and presentation integration |

## Project Overview

This project implements and compares softmax regression and a one-hidden-layer neural network on three datasets: a linear Gaussian synthetic task, a nonlinear moons synthetic task, and an 8×8 digits benchmark. Advanced analysis (Track B) covers confidence calibration and reliability.

Key results on the digits benchmark:
- Softmax regression: **94.0%** test accuracy
- Neural network (width 32, SGD): **93.8%** test accuracy

## Repository Structure

```
deliverables/               # Assignment handout
starter_pack/
  data/                     # Fixed datasets and split indices
  scripts/                  # Data preparation utilities
  src/                      # Model, training, and evaluation code
  figures/                  # Experiment plots
  results/                  # Saved experiment outputs and tables
  report/                   # Final PDF report and LaTeX source
  slides/                   # Presentation materials
```

## Setup

**Requirements:** Python 3.8+, NumPy, Matplotlib, scikit-learn (for data only)

```bash
pip install numpy matplotlib scikit-learn
```

All scripts are run from the **repository root** and resolve paths automatically — no `PYTHONPATH` changes needed.

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden width (NN) | 32 |
| Optimizer | SGD |
| Learning rate | 0.05 |
| Batch size | 64 |
| Epochs | 200 |
| L2 regularization (λ) | 0.0 |

Ablation experiments sweep hidden widths `{2, 8, 32}` and optimizers `{SGD, momentum, Adam}`.

## Reproducing Results

**1. Verify data files** (~2 seconds):

```bash
python starter_pack/scripts/make_digits_split.py       # reproduces split indices
python starter_pack/scripts/generate_synthetic.py      # regenerates synthetic datasets
python starter_pack/scripts/inspect_data.py            # prints data shapes/stats
```

Expected: no errors; `data_summary.txt` printed to console and saved to `starter_pack/results/`.

**2. Run core experiments** (~1–2 minutes):

```bash
python starter_pack/src/core_experiments.py
```

Expected outputs:
- `starter_pack/results/core_experiments_results.csv` — accuracy and cross-entropy per model/dataset
- `starter_pack/figures/` — decision boundary plots and learning curves for all three datasets

**3. Run ablation experiments** (~2–3 minutes):

```bash
python starter_pack/src/ablation_experiments.py
```

Expected outputs:
- `starter_pack/results/ablation_results.csv` — capacity and optimizer sweep results
- `starter_pack/figures/moons_capacity_ablation.png`, `digits_optimizer_comparison.png`, `failure_case_width2_moons.png`

**4. Run sanity checks** (~10 seconds):

```bash
python starter_pack/src/sanity_checks.py
python starter_pack/src/gradient_check.py
```

Expected: all checks print `PASS`; gradient check reports relative error < 1e-5.

**5. Run Track B — confidence and reliability** (~1 minute):

```bash
python starter_pack/src/confidence_reliability.py
python starter_pack/src/repeated_seed_eval.py
```

Expected outputs:
- `starter_pack/results/entropy_summary.csv` and `entropy_summary.md` — predictive entropy statistics
- `starter_pack/results/repeated_seed_table.csv` — mean ± std accuracy over multiple seeds
- `starter_pack/figures/reliability_softmax.png`, `reliability_nn.png`, `entropy_comparison.png`

## Key Files

- [starter_pack/src/softmax_model.py](starter_pack/src/softmax_model.py) — softmax regression implementation
- [starter_pack/src/neural_network.py](starter_pack/src/neural_network.py) — one-hidden-layer neural network
- [starter_pack/src/optimizers.py](starter_pack/src/optimizers.py) — SGD, momentum, Adam
- [starter_pack/src/core_experiments.py](starter_pack/src/core_experiments.py) — main experiment runner
- [starter_pack/report/Math4AI_Final_Capstone_Project.pdf](starter_pack/report/Math4AI_Final_Capstone_Project.pdf) — final report
- [starter_pack/slides/Math4AI_Capstone_Presentation_final.pptx](starter_pack/slides/Math4AI_Capstone_Presentation_final.pptx) — final slides

## References

- Assignment: [deliverables/math4ai_capstone_assignment.tex](deliverables/math4ai_capstone_assignment.tex)
- Starter pack overview: [starter_pack/README.md](starter_pack/README.md)
- Starter checklist: [starter_pack/CHECKLIST.md](starter_pack/CHECKLIST.md)
