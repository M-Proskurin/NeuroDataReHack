"""Shared paths and conventions for the DANDI manifold analyses.

Common utility code for both dandisets (000447 and 000978), which are analyzed
in parallel and never merged. Artifacts are namespaced by dandiset under
data/raw/<id>/ and data/processed/<id>/. Import this everywhere so pipeline
stages agree on bin size, region names, and where artifacts live.
See CLAUDE.md for the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --- paths (repo-relative; data/ dirs are gitignored) ---
# config.py lives at src/common/, so the repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"          # downloaded NWB files (per dandiset)
DATA_PROCESSED = REPO_ROOT / "data" / "processed"  # rate matrices, embeddings

# --- datasets ---
DANDISETS = ("000447", "000978")
DEFAULT_DANDISET = "000447"
DANDISET_ID = DEFAULT_DANDISET      # back-compat default for download helpers

# --- regions: process separately, never pool units across them ---
REGIONS = ("CA1", "PFC")

# --- time binning (stage 1) ---
# 1000 ms bins: 50 ms was too sparse for these low-rate units (CA1 median
# ~1.2 Hz, PFC ~2 Hz), leaving >90% of bins empty. 1 s bins roughly halve the
# empty-bin fraction while leaving mean firing rate (Hz) unchanged.
BIN_SIZE_S = 1.0
SQRT_TRANSFORM = True    # variance-stabilize binned spike counts

# --- reproducibility ---
RANDOM_SEED = 0


def raw_dir(dandiset_id: str | None = None) -> Path:
    """data/raw/<dandiset_id>/, created if needed."""
    d = DATA_RAW / (dandiset_id or DEFAULT_DANDISET)
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(dandiset_id: str | None = None) -> Path:
    """data/processed/<dandiset_id>/, created if needed."""
    d = DATA_PROCESSED / (dandiset_id or DEFAULT_DANDISET)
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_path(name: str, dandiset_id: str | None = None) -> Path:
    """Resolve a filename under data/processed/<dandiset_id>/."""
    return processed_dir(dandiset_id) / name


def rate_matrix_name(subject: str, region: str, bin_ms: int | None = None) -> str:
    """Canonical filename for a binned rate matrix, e.g. rates_subj01_CA1_50ms.npz.

    Store the neural matrix together with epoch/condition/animal metadata columns
    inside the same .npz (per CLAUDE.md), not as separate untracked arrays.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")
    bin_ms = bin_ms if bin_ms is not None else int(BIN_SIZE_S * 1000)
    return f"rates_{subject}_{region}_{bin_ms}ms.npz"


def load_rate_matrix(subject: str, region: str, bin_ms: int | None = None,
                     dandiset_id: str | None = None) -> dict:
    """Load a stage-1 rate matrix as a dict of arrays."""
    path = processed_path(rate_matrix_name(subject, region, bin_ms), dandiset_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the extraction stage first")
    return dict(np.load(path, allow_pickle=False))


def spatial_grid_labels(position: np.ndarray, n_grid: int = 8) -> np.ndarray:
    """Integer 2-D occupancy-grid bin per timepoint; -1 where position is NaN.

    Shared by stages that compare representations at matched spatial locations
    (dPCA in stage 2, Procrustes/CCA in stage 4).
    """
    x, y = position[:, 0], position[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    lab = np.full(x.shape, -1, dtype=int)
    if valid.sum() == 0:
        return lab
    x0, x1 = np.percentile(x[valid], [0.5, 99.5])
    y0, y1 = np.percentile(y[valid], [0.5, 99.5])
    xi = np.clip(((x[valid] - x0) / max(x1 - x0, 1e-9) * n_grid).astype(int), 0, n_grid - 1)
    yi = np.clip(((y[valid] - y0) / max(y1 - y0, 1e-9) * n_grid).astype(int), 0, n_grid - 1)
    lab[valid] = xi * n_grid + yi
    return lab


def available_rate_matrices(bin_ms: int | None = None,
                            dandiset_id: str | None = None) -> list[tuple[str, str, Path]]:
    """List (subject, region, path) for stage-1 rate matrices at a given bin size."""
    bin_ms = bin_ms if bin_ms is not None else int(BIN_SIZE_S * 1000)
    out = []
    for path in sorted(processed_dir(dandiset_id).glob(f"rates_*_{bin_ms}ms.npz")):
        subject, region, _ = path.stem[len("rates_"):].rsplit("_", 2)
        out.append((subject, region, path))
    return out
