# 000447 notebooks — Novel-familiar-novel W-track (CA1–PFC)

One notebook per pipeline stage, mirroring `src/000447/`. See the
[top-level notebooks README](../README.md) for layout and the VS Code kernel
setup.

| Notebook | Mirrors | Purpose |
|----------|---------|---------|
| `00_data_sanity_checks.ipynb`  | `src/common/download.py`         | Stream one session, QC spikes/epochs/behavior |
| `01_extraction.ipynb`          | `src/000447/01_extraction.py`    | Inspect NWB, sanity-check binned rates |
| `02_baseline_linear.ipynb`     | `src/000447/02_baseline_linear.py` | PCA / GPFA / dPCA figures |
| `03_nonlinear_embedding.ipynb` | `src/000447/03_nonlinear_embedding.py` | CEBRA / UMAP / Isomap embeddings |
| `04_cross_condition.ipynb`     | `src/000447/04_cross_condition.py` | Procrustes / CCA (novel↔familiar, CA1↔PFC) |
| `06_dimensionality.ipynb`      | `src/000447/06_dimensionality.py` | Bi-CV reconstruction-error curves |

(Stage 5, topology, is optional and not yet built as a notebook.)
