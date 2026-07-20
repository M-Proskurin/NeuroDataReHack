# NeuroDataReHack

Manifold / dimensionality-reduction analysis of two related DANDI dandisets
(same lab, same reference paper), analyzed **in parallel, never merged**:

- **[000447](https://dandiarchive.org/dandiset/000447)** — novel-familiar-novel
  W-track (CA1 + PFC), 5 subjects. How does the "cognitive map" transform
  between novel and familiar contexts, and how do CA1 and PFC relate?
- **[000978](https://dandiarchive.org/dandiset/000978)** — single-day W-track
  learning (CA1 + PFC), 8 subjects, run sessions interleaved with sleep. How
  does the manifold evolve session-by-session during learning, and do sleep
  epochs replay the awake geometry?

See [CLAUDE.md](CLAUDE.md) for the full pipeline and conventions.

## Key findings

- Both cognitive maps live on a **low-dimensional but curved** manifold — model-free
  intrinsic dimension (TwoNN / Isomap ≈ 3–5) sits well below the linear PCA
  participation ratio (the gap is a curvature signal).
- **000447:** novel → familiar **transforms the map's geometry at ~fixed
  dimensionality**. The two are different mazes, so compared at matched
  track-relative (**linearized**) positions they share clear geometry yet reshape
  (Procrustes ≪ shuffle null); CA1 and PFC share structure.
- **000978:** each session's manifold is ≈ 3-D but **drifts across the day**
  (pooled dimension ≈ 8; session subspaces rotate), **converging** toward the
  final-session geometry as the animal learns. Sleep-replay geometry was
  inconclusive and set aside.
- Findings reproduce across UMAP and CEBRA (supervised + unsupervised CEBRA-Time)
  and in embedding-independent rate space. Summary: `reports/analysis_summary.html`.

## Layout

```
src/
  common/     shared utils: config, download, preprocessing, linearize,
              dimensionality, lap_baselines, trajectory_labels, embed_smoothed
  000447/     01_extraction … 06_dimensionality (+ 04b linearized, 05 topology,
              figure/movie scripts)
  000978/     01_extraction, 02, 03 (+ 06b session-sequence, 06b linearized,
              06c sleep, 06_dimensionality, 06_dim_drift)
notebooks/
  000447/, 000978/   one notebook per stage (mirror src/<id>/)
reports/      analysis_summary.html (self-contained); figures/ (gitignored)
data/
  raw/<id>/         downloaded NWB (gitignored)
  processed/<id>/   rate matrices, embeddings, results (gitignored)
```

## Data

Raw NWB files (~33 GiB for 000447, ~323 GB for 000978) and processed artifacts
are gitignored. You usually don't need to download anything — `download.py` can
**stream** sessions lazily from the DANDI S3 store (only the bytes you index are
fetched).

```bash
# pass the dandiset id; default is 000447
pixi run python src/common/download.py --list
pixi run python src/common/download.py --stream <asset-path>
```

`src/common/download.py` also exposes `stream_nwb(path, dandiset_id=...)` for use
inside notebooks and pipeline stages.

## Pipeline

Each dandiset's stages run in order under `src/<id>/`, saving rerunnable
intermediates under `data/processed/<id>/`. See [CLAUDE.md](CLAUDE.md) for the
per-stage details and conventions (CA1/PFC kept separate; epoch/condition/
session/subject metadata carried alongside neural data).
