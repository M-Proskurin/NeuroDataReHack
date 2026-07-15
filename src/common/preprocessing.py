"""Shared preprocessing for the smoothed fine-bin path.

Low-rate units make hard fine bins mostly-empty; the fix is a fine bin plus a
Gaussian temporal kernel (rate estimate), optionally restricted to movement.
These helpers are used by the extraction scripts (to emit smoothed matrices) and
by the bin/smoothing sensitivity analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor


def smooth_per_epoch(counts: np.ndarray, epoch: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian-smooth each unit over time, WITHIN each epoch (no cross-epoch bleed).

    counts: (T, n_units). epoch: (T,) integer epoch id (contiguous blocks).
    sigma_bins <= 0 returns a copy unchanged.
    """
    out = counts.astype(np.float64, copy=True)
    if sigma_bins <= 0:
        return out
    for e in np.unique(epoch):
        idx = np.flatnonzero(epoch == e)
        out[idx] = gaussian_filter1d(out[idx], sigma=sigma_bins, axis=0, mode="nearest")
    return out


def speed_mask(velocity: np.ndarray, thresh: float = 4.0) -> np.ndarray:
    """Boolean mask of bins with finite speed above `thresh` cm/s (movement)."""
    return np.isfinite(velocity) & (velocity > thresh)


def cv_position_r2(X: np.ndarray, position: np.ndarray, n_neighbors: int = 15,
                   n_splits: int = 5) -> float:
    """Cross-validated kNN position-decoding R^2 from a rate/embedding matrix.

    Uses CONTIGUOUS folds (shuffle=False): smoothing induces temporal
    autocorrelation, so random folds would leak train info into test.
    """
    ok = np.isfinite(position).all(axis=1)
    X, y = X[ok], position[ok]
    if X.shape[0] < n_splits * (n_neighbors + 1):
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = []
    for tr, te in kf.split(X):
        knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X[tr], y[tr])
        # R^2 averaged over x,y
        pred = knn.predict(X[te])
        ss_res = ((y[te] - pred) ** 2).sum(0)
        ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum(0) + 1e-12
        scores.append(float(np.mean(1 - ss_res / ss_tot)))
    return float(np.mean(scores))
