# NeuroData ReHack — Project Report

## Project Description
Population-geometry ("cognitive map") analysis of hippocampal CA1 and prefrontal
(PFC) ensemble recordings from the Jadhav lab, using manifold and
dimensionality-reduction methods. Two W-track dandisets from the same lab are
analyzed **in parallel, not merged**: one probing how the map differs between
novel and familiar environments, the other how it evolves as an animal learns a
new environment across a single day. CA1 and PFC are always processed
separately.

## Dandiset(s) Used
- https://dandiarchive.org/dandiset/000978  — "Single Day W-Track Learning" (CA1+PFC, 8 subjects, run sessions interleaved with sleep across one day)
- https://dandiarchive.org/dandiset/000447  — "Novel-familiar-novel WTrack (CA1-PFC)" (CA1+PFC, 5 subjects, novel then familiar epochs)

## Objectives and Approach
**Questions.** (000447) How does the population geometry transform between novel
and familiar contexts, and how do CA1 and PFC relate? (000978) How does the
manifold evolve session-by-session during learning, and does it change its
dimensionality or just its shape?

**Approach.** Data are streamed lazily from DANDI (`pynwb`/`remfile`). A common
pipeline runs per dandiset: (1) extract time-binned spike-rate matrices with
position/velocity and epoch/condition/session labels; (2) linear baselines — PCA,
lap-resolved dPCA (cross-validated regularization + permutation significance),
and GPFA with latents indexed by linearized track position; (3) nonlinear
embeddings — UMAP and CEBRA (supervised and unsupervised CEBRA-Time) on a
validated 50 ms / Gaussian-smoothed / speed-filtered representation; (4)
geometry comparison — Procrustes/CCA on position-matched centroids, with tracks
**linearized onto a common W topology** so different physical mazes are
comparable; (5) intrinsic-dimensionality triangulation — TwoNN, PCA
participation ratio, Isomap residual variance, and a decoding-vs-dimension curve,
all cross-validated.

## Progress and Next Steps
**Done.**
- 000447: novel vs. familiar are physically different mazes; compared in
  track-relative (linearized) coordinates the maps show a **spatially-structured
  transformation** — clear shared geometry plus real reshaping (Procrustes
  disparity ≈ 0.55–0.59 vs. ~0.90 null on the unsupervised UMAP and CEBRA-Time
  embeddings; supervised CEBRA agrees). dPCA confirms a genuine space×condition
  interaction (remap), significant in all animals.
- 000978: the manifold **converges monotonically** toward its final-session
  geometry across the day (disparity-to-final ~0.6→0.16), robust to bin size,
  embedding, region, and 2-D vs. track-relative binning.
- Dimensionality: both maps are **low-dimensional and curved** — TwoNN/Isomap
  give ~3–5 intrinsic dimensions vs. a much higher linear participation ratio
  (the gap is a curvature signal). Intrinsic dimensionality is largely unchanged
  by familiarisation (000447) and, within a session, is stable (~3) across
  learning (000978) — the pooled ~8 reflects cross-session drift, not
  within-session complexity. Conclusion: learning/novelty **reshape the geometry
  of a fixed-low-dimensional map** rather than changing its dimensionality.
- Methods validated: 50 ms + σ=100 ms smoothing + speed filter chosen by
  cross-validated position decoding; findings reproduce in embedding-independent
  rate space; results self-contained in an HTML report and per-stage notebooks.

**Next steps.**
- Topology (persistent homology, `ripser`) to test whether the low-D geometry is
  the expected W/ring structure and whether that topology is preserved across
  the novel→familiar transformation.
- CA1↔PFC comparison in linearized coordinates (currently only raw-2D).
- 000978 sleep/replay via SWR-triggered Bayesian sequence decoding (the
  manifold-occupancy approach was inconclusive and set aside).
- Per-animal vs. pooled alignment, and consistency checks across animals for the
  learning trajectory.

## Background and References
Reference paper: Shin & Jadhav, *Geometric transformation of cognitive maps for
generalization across hippocampal-prefrontal circuits*, Cell Reports 2023
(DOI: 10.1016/j.celrep.2023.112246).

Methods: CEBRA — Schneider, Lee & Mathis, *Nature* 2023; TwoNN — Facco et al.,
*Scientific Reports* 2017; dPCA — Kobak et al., *eLife* 2016; GPFA — Yu et al.,
*J. Neurophysiol.* 2009; bi-cross-validation — Owen & Perry 2009. Tooling:
`pynwb`, `dandi`, `scikit-learn`, `umap-learn`, `cebra`, `elephant` (GPFA),
`track_linearization`.
