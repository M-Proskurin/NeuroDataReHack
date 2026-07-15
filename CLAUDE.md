# CLAUDE.md

## Project
Manifold/dimensionality-reduction analysis of two related DANDI dandisets (same lab, same reference paper — see below):

- **[000447](https://dandiarchive.org/dandiset/000447)** — "Novel-familiar-novel WTrack (CA1-PFC)". Hippocampal (CA1) + prefrontal cortex (PFC) recordings, 5 subjects, 3 behavioral epochs/file (novel, then familiar), ~33 GiB.
- **[000978](https://dandiarchive.org/dandiset/000978)** — "Single Day W-Track Learning". CA1 + PFC recordings, 8 subjects, 8 behavioral sessions interleaved with sleep (17 total epochs) recorded across a **single day**, ~323 GB.

**Core questions:**
1. (000447) How does population-level neural geometry (the "cognitive map") transform between novel and familiar contexts, and how do CA1 and PFC manifolds relate to each other?
2. (000978) How does the manifold evolve session-by-session *during* initial learning within a single day, and do sleep epochs show compressed/distorted versions of the awake manifold (replay)?

These two dandisets are analyzed **in parallel, not merged** — different animals, different designs. No cross-dandiset comparison or alignment for now.

Reference paper: Shin & Jadhav, *Geometric transformation of cognitive maps for generalization across hippocampal-prefrontal circuits*, Cell Reports 2023 (DOI: 10.1016/j.celrep.2023.112246).

## Data
- Each NWB file (000447): 3 behavioral epochs (novel W-track first exposure, then familiar epochs)
- Each NWB file (000978): 8 behavioral sessions + interleaved sleep epochs (17 total) across a single day
  - **Exception — ZT2:** subject ZT2 has *two* files (`obj-u40err`, `obj-1dss6zi`) that are **two separate recording days** (2024-05-02 and 2024-05-03), ~4 run sessions each — NOT one split day. Do not stitch them into an 8-session sequence. ZT2 is **excluded from the 6b learning-trajectory analysis**; its data stays usable for per-file stage 2/3 and 6c. Rate matrices are keyed by a filename-derived `session_key`, so ZT2 appears as two keys.
- Variables (both): spike times (`Units`), LFP (`ElectricalSeries`), position/velocity (`SpatialSeries`), electrode metadata
- Access via `pynwb`; consider `spikeinterface` for spike-sorted unit handling
- Keep data organized by dandiset: `data/raw/000447/`, `data/raw/000978/` (gitignored) — do not commit raw data
- Keep processed outputs similarly namespaced by dandiset under `data/processed/`

## Pipeline

### Shared steps (both dandisets)
1. **Extraction** — pull spikes + position + epoch/session/condition labels from NWB into a common time-binned rate matrix (start with 50–100ms bins, sqrt-transform)
2. **Baseline linear** — PCA, GPFA, dPCA (factor out position/novelty/session/epoch variance explicitly)
3. **Nonlinear embedding** — CEBRA (behavior-aligned, primary tool), Isomap/UMAP/PHATE as comparisons
4. **Topology check (optional)** — persistent homology (`ripser`/`giotto-tda`) to confirm ring/toroidal structure
5. **Dimensionality estimation** — don't rely on a single method; triangulate:
   - **TwoNN** (Facco et al. 2017) as the primary model-free intrinsic dimensionality estimate — robust to curvature/noise, no fitting required
   - **Decoding-vs-dimension curve** — sweep CEBRA latent dimension, track decoding accuracy for position/novelty/session; take the saturation point
   - **Isomap residual variance vs. dimension** — secondary check, already computing Isomap for topology
   - **PCA participation ratio** — report as an upper bound only (linear methods overestimate dimensionality of curved manifolds); the gap vs. TwoNN is itself informative about curvature
   - Run estimators on held-out/cross-validated data, not the full dataset — noisy spike rates otherwise inflate the estimate

### 000447-specific: novel vs. familiar comparison
6a. Procrustes/CCA between novel vs. familiar embeddings, and between CA1 vs. PFC embeddings

### 000978-specific: within-day learning trajectory
6b. **Session-sequence alignment** — align each of the 8 sessions' embeddings to the final (most-familiar) session; track alignment quality (Procrustes disparity / CCA correlation) as a function of session number to get a convergence time course, rather than a single before/after contrast
6c. **Sleep epoch analysis** — embed or project sleep epochs into the awake-derived manifold to assess whether replay events sample the same geometric structure or a compressed/distorted version of it
6d. Run 6b/6c separately for CA1 and PFC, and check consistency across animals (6b: the 7 clean single-day animals, ZT2 excluded — see Data note)

No cross-dandiset merging or alignment — each dandiset's pipeline runs independently end to end.

## Conventions
- Python, one script/notebook per pipeline stage under `src/` and `notebooks/`; keep 000447 and 000978 pipelines in separate subfolders (`src/000447/`, `src/000978/`) sharing common utility code (`src/common/`)
- Separate CA1 and PFC processing — never pool units across regions in one matrix
- Always carry epoch/session label (novel/familiar/epoch index, or session index/sleep flag for 000978) and animal ID alongside neural data as metadata columns, not separate untracked arrays
- Save intermediate rate matrices and embeddings as `.npz`/`.h5` under `data/processed/<dandiset_id>/` so pipeline stages are independently rerunnable
- Do not combine animals or sessions across the two dandisets in a single analysis — keep them as parallel, independent studies

## Key libraries
`pynwb`, `spikeinterface`, `cebra`, `scikit-learn` (PCA/Isomap), `umap-learn`, `elephant` or custom GPFA, `ripser`/`giotto-tda`

## Open questions to track
- Optimal bin size / smoothing kernel for each dataset (may differ given sleep epochs in 000978)
- Whether alignment (Procrustes/CCA) should be done per-animal or pooled, within each dandiset
- How much of the "transformation" between novel/familiar (000447) or across sessions (000978) is trivial rate change vs. genuine geometric reshaping
- Whether sleep-epoch embeddings in 000978 should use the same manifold basis as awake epochs, or be fit independently and compared post hoc
