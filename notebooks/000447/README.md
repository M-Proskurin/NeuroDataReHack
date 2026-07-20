# 000447 notebooks — Novel-familiar-novel W-track (CA1–PFC)

One notebook per pipeline stage, mirroring `src/000447/`. See the
[top-level notebooks README](../README.md) for layout and the VS Code kernel
setup.

| Notebook | Mirrors | Purpose |
|----------|---------|---------|
| `00_data_sanity_checks.ipynb`  | `src/common/download.py`         | Stream one session, QC spikes/epochs/behavior |
| `01_extraction.ipynb`          | `src/000447/01_extraction.py`    | Inspect NWB, sanity-check binned rates |
| `02_baseline_linear.ipynb`     | `src/000447/02_baseline_linear.py` | PCA + lap-based dPCA (CV reg + permutation) + position-indexed GPFA |
| `03_nonlinear_embedding.ipynb` | `src/000447/03_nonlinear_embedding.py` | UMAP / CEBRA (sup + unsup CEBRA-Time); track↔manifold + interactive 3-D |
| `04_cross_condition.ipynb`     | `src/000447/04_cross_condition.py` | Procrustes / CCA novel↔familiar & CA1↔PFC; linearized arm-matched comparison |
| `06_dimensionality.ipynb`      | `src/000447/06_dimensionality.py` | Intrinsic dimensionality (TwoNN / participation ratio / Isomap / decoding); novel-vs-familiar summary |

**Result:** the novel and familiar maps are **low-dimensional (~4) and curved**;
compared in track-relative coordinates their geometry is **shared but reshaped**,
with dimensionality ~unchanged. Track↔manifold figures/movies come from
`src/000447/fig_trajectory_manifold.py` and `movie_umap_rotate.py`; stage 5
(topology, `05_topology.py`) is optional and not built as a notebook.
