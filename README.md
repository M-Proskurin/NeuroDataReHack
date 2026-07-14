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

## Layout

```
src/
  common/     shared utilities (config.py, download.py)
  000447/     000447 pipeline stages (01_extraction … 06_dimensionality)
  000978/     000978 pipeline stages (TBD)
notebooks/
  000447/     one notebook per stage
  000978/     one notebook per stage
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
