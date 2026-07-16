"""Intrinsic-dimensionality estimators for stage 5 (triangulation).

No single method is trusted (CLAUDE.md): a curved manifold makes linear methods
over-count, while local estimators are noise-sensitive, so we report several and
read them together.

  * **TwoNN** (Facco et al. 2017) — primary, model-free intrinsic dimension from
    the ratio of each point's 2nd/1st nearest-neighbour distances. Robust to
    curvature; no embedding fitted. Reported with a bootstrap spread.
  * **PCA participation ratio** — linear, (Σλ)² / Σλ²; an UPPER bound (linear
    methods inflate the dimension of curved manifolds). The PR−TwoNN gap is
    itself a curvature signal.
  * **Isomap residual variance vs. dimension** — 1 − corr(geodesic, embedded)²;
    the knee is a geometry-based dimension estimate.
  * **Decoding-vs-dimension** — sweep the PCA latent dimension and track
    cross-validated decoding of position and of the task label (novelty / run
    session); the saturation point is the behaviourally-relevant dimension.
    (An UNSUPERVISED embedding is swept on purpose — decoding position out of a
    position-supervised CEBRA would be circular.)

All estimators run on random subsamples / cross-validated folds, never the full
noisy matrix at once, per the plan.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)


# ---------------------------------------------------------------- TwoNN
def twonn(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN maximum-likelihood intrinsic dimension for one sample of points."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)                     # col0 = self (0)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu[mu > 1.0])                   # mu>=1 by construction; drop ties
    if mu.size < 10:
        return float("nan")
    keep = max(1, int(mu.size * (1.0 - discard_frac)))   # drop the noisy upper tail
    mu = mu[:keep]
    return float(mu.size / np.sum(np.log(mu)))


def twonn_cv(X: np.ndarray, n_boot: int = 20, sample_frac: float = 0.8,
             rng=None) -> tuple[float, float]:
    """Bootstrap TwoNN: mean ± sd over random subsamples (decorrelates + CV)."""
    rng = rng or np.random.default_rng(0)
    n = len(X)
    m = min(n, max(100, int(n * sample_frac)))
    est = [twonn(X[rng.choice(n, m, replace=False)]) for _ in range(n_boot)]
    est = np.array([e for e in est if np.isfinite(e)])
    if est.size == 0:
        return float("nan"), float("nan")
    return float(est.mean()), float(est.std())


# ---------------------------------------------------------------- participation ratio
def participation_ratio(X: np.ndarray) -> float:
    """Linear participation ratio (Σλ)²/Σλ² — an upper bound on dimensionality."""
    ev = PCA().fit(X).explained_variance_
    return float(ev.sum() ** 2 / np.sum(ev ** 2))


# ---------------------------------------------------------------- Isomap residual variance
def isomap_residual_variance(X: np.ndarray, dims, n_neighbors: int = 15,
                             sub: int = 1500, rng=None) -> dict[int, float]:
    """Residual variance 1 − corr(geodesic, d-dim embedded)² for each d in dims."""
    rng = rng or np.random.default_rng(0)
    if len(X) > sub:
        X = X[rng.choice(len(X), sub, replace=False)]
    iso = Isomap(n_neighbors=n_neighbors, n_components=int(max(dims))).fit(X)
    dg = iso.dist_matrix_[np.triu_indices(len(X), 1)]
    Y = iso.embedding_
    out = {}
    for d in dims:
        de = pdist(Y[:, :d])
        r = np.corrcoef(dg, de)[0, 1]
        out[int(d)] = float(1.0 - r ** 2)
    return out


def _knee(curve: dict[int, float]) -> int:
    """Elbow of the residual-variance curve (max distance to the first→last chord)."""
    dims = np.array(sorted(curve), dtype=float)
    y = np.array([curve[int(d)] for d in dims])
    if len(dims) < 3:
        return int(dims[int(np.argmin(y))])
    x0, x1, y0, y1 = dims[0], dims[-1], y[0], y[-1]
    # perpendicular distance of each point to the chord from (x0,y0) to (x1,y1)
    num = np.abs((y1 - y0) * dims - (x1 - x0) * y + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) + 1e-12
    return int(dims[int(np.argmax(num / den))])


# ---------------------------------------------------------------- decoding vs dimension
def decode_vs_dim(X: np.ndarray, target: np.ndarray, dims, kind: str,
                  cv: int = 5) -> dict[int, float]:
    """CV decoding score vs. PCA latent dimension (unsupervised sweep).

    kind='position' -> kNN regression R² on (x, y); else kNN classification
    accuracy on the label. Contiguous KFold (shuffle=False) respects the temporal
    autocorrelation left by smoothing.
    """
    Z = PCA(n_components=int(max(dims))).fit_transform(X)
    out = {}
    for d in dims:
        if kind == "position":
            sc = cross_val_score(KNeighborsRegressor(15), Z[:, :d], target,
                                 cv=cv, scoring="r2")
        else:
            sc = cross_val_score(KNeighborsClassifier(15), Z[:, :d], target,
                                 cv=cv, scoring="accuracy")
        out[int(d)] = float(np.nanmean(sc))
    return out


def _saturation(curve: dict[int, float], frac: float = 0.95) -> int:
    """Smallest dimension reaching `frac` of the maximum score."""
    dims = sorted(curve)
    vals = np.array([curve[d] for d in dims])
    mx = vals.max()
    if mx <= 0:
        return int(dims[-1])
    return int(dims[int(np.argmax(vals >= frac * mx))])


# ---------------------------------------------------------------- driver per group
def analyze_group(X: np.ndarray, position: np.ndarray, label: np.ndarray,
                  label_kind: str, max_dim: int = 15, max_samples: int = 6000,
                  rng=None):
    """Full triangulation for one (subject/region[/condition]) population sample.

    Returns (summary dict, curves list-of-rows). `label_kind` names the task
    label being decoded ('novelty' for 000447, 'session' for 000978). Points are
    randomly subsampled to `max_samples` first — this both bounds runtime and
    decorrelates the smoothing-induced temporal autocorrelation.
    """
    rng = rng or np.random.default_rng(0)
    n0, N = X.shape
    kmax = int(min(max_dim, N - 1, n0 // 40))
    if kmax < 3:
        return None, []
    dims = list(range(1, kmax + 1))

    # geometry estimators: RANDOM subsample (bounds runtime, decorrelates)
    if n0 > max_samples:
        r = rng.choice(n0, max_samples, replace=False)
        Xg = X[r]
    else:
        Xg = X
    tw_mean, tw_sd = twonn_cv(Xg, rng=rng)
    pr = participation_ratio(Xg)
    iso = isomap_residual_variance(Xg, dims, rng=rng)

    # decoding: STRIDED subsample (order-preserving) so contiguous KFold stays
    # honest — a random subsample would break the temporal block structure and
    # inflate the CV score.
    step = max(1, n0 // max_samples)
    s = np.arange(0, n0, step)
    Xs, poss, labs = X[s], position[s], label[s]
    finite = np.isfinite(poss).all(axis=1)
    pos_curve = decode_vs_dim(Xs[finite], poss[finite], dims, "position")
    lab_ok = labs.astype(str) != "-1"
    lab_curve = decode_vs_dim(Xs[lab_ok], labs[lab_ok], dims, "label") \
        if len(np.unique(labs[lab_ok])) > 1 else {}

    summary = dict(
        n_units=int(N), n_samples=int(X.shape[0]), kmax=kmax,
        twonn=tw_mean, twonn_sd=tw_sd, participation_ratio=pr,
        isomap_knee=_knee(iso),
        decode_pos_sat=_saturation(pos_curve), decode_pos_max=max(pos_curve.values()),
        decode_label_sat=(_saturation(lab_curve) if lab_curve else float("nan")),
        decode_label_max=(max(lab_curve.values()) if lab_curve else float("nan")),
    )
    curves = []
    for d in dims:
        curves.append(dict(dim=d, isomap_resid=iso[d], decode_pos=pos_curve[d],
                           decode_label=lab_curve.get(d, float("nan"))))
    return summary, curves
