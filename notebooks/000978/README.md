# 000978 notebooks — Single-day W-track learning (CA1–PFC)

One notebook per pipeline stage, mirroring `src/000978/`. See the
[top-level notebooks README](../README.md) for layout and the VS Code kernel
setup. This dandiset is analyzed in parallel with 000447 — never merged.

| Notebook | Mirrors | Purpose |
|----------|---------|---------|
| `00_data_sanity_checks.ipynb` | `src/common/download.py` | Stream one session; QC spikes, run/sleep epochs, behavior, learning curve |

Later stages (extraction with run/sleep + run-session metadata, baselines,
embeddings, session-sequence alignment, sleep-replay projection) are TBD under
`src/000978/`.
