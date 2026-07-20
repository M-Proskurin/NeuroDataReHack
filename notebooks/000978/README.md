# 000978 notebooks — Single-day W-track learning (CA1–PFC)

One notebook per pipeline stage, mirroring `src/000978/`. See the
[top-level notebooks README](../README.md) for layout and the VS Code kernel
setup. This dandiset is analyzed in parallel with 000447 — never merged.

| Notebook | Mirrors | Purpose |
|----------|---------|---------|
| `00_data_sanity_checks.ipynb`  | `src/common/download.py`               | Stream one file; QC spikes, run/sleep epochs, behavior, learning curve |
| `01_extraction.ipynb`          | `src/000978/01_extraction.py`          | Binned rates on run/sleep epochs; run_session metadata |
| `02_baseline_linear.ipynb`     | `src/000978/02_baseline_linear.py`     | PCA + lap-based dPCA (space × session) + position-indexed GPFA |
| `03_nonlinear_embedding.ipynb` | `src/000978/03_nonlinear_embedding.py` | UMAP / CEBRA; track↔manifold + interactive 3-D (position- and session-coloured, session selector) |
| `06b_session_sequence.ipynb`   | `src/000978/06b_session_sequence.py`   | Session-to-final Procrustes/CCA — the learning convergence curve |
| `06c_sleep_projection.ipynb`   | `src/000978/06c_sleep_projection.py`   | Sleep epochs vs awake manifold (replay geometry — inconclusive) |
| `06_dimensionality.ipynb`      | `src/000978/06_dimensionality.py`      | Intrinsic dimensionality; per-session + **low-D-but-drifting** (`06_dim_drift.py`) |

**Result:** each run session's manifold is **low-dimensional (~3)** but **drifts
across the day** (pooled dimension ~8; session subspaces rotate) and **converges**
toward the final-session geometry as the animal learns. ZT2 (two separate days)
is excluded from the learning-trajectory analyses.
