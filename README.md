# NeuroDataReHack

Manifold / dimensionality-reduction analysis of DANDI dandiset
[000447](https://dandiarchive.org/dandiset/000447) — hippocampal (CA1) and
prefrontal cortex (PFC) recordings from rats navigating novel vs. familiar
W-track mazes.

**Core question:** how does population-level neural geometry (the "cognitive
map") transform between novel and familiar contexts, and how do the CA1 and PFC
manifolds relate? See [CLAUDE.md](CLAUDE.md) for the full pipeline and
conventions.

## Data

Raw NWB files (~33 GiB) and processed artifacts live under `data/` and are
gitignored. You usually don't need to download anything — stage 0 can **stream**
sessions lazily from the DANDI S3 store.

```bash
pixi run python src/00_download.py --list                 # list session assets
pixi run python src/00_download.py --stream <asset-path>  # lazily open + summarize
```

`src/00_download.py` also exposes `stream_nwb()` for use inside notebooks and
later pipeline stages (only the bytes you index are fetched).

## Pipeline

Scripts run in order under `src/` (`00_download` → `06_dimensionality`); each
stage saves rerunnable intermediates under `data/processed/`. See
[CLAUDE.md](CLAUDE.md) for details on each stage and the analysis conventions
(CA1/PFC kept separate, epoch/condition/subject metadata carried alongside
neural data).
