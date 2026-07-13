"""Shared paths and conventions for the dandiset 000447 manifold analysis.

Import this everywhere so pipeline stages agree on bin size, region names,
and where intermediate artifacts live. See CLAUDE.md for the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --- paths (repo-relative; data/ dirs are gitignored) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"          # downloaded NWB files
DATA_PROCESSED = REPO_ROOT / "data" / "processed"  # rate matrices, embeddings

# --- dataset ---
DANDISET_ID = "000447"

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


def processed_path(name: str) -> Path:
    """Resolve a filename under data/processed/, creating the dir if needed."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED / name


def rate_matrix_name(subject: str, region: str, bin_ms: int | None = None) -> str:
    """Canonical filename for a binned rate matrix, e.g. rates_subj01_CA1_50ms.npz.

    Store the neural matrix together with epoch/condition/animal metadata columns
    inside the same .npz (per CLAUDE.md), not as separate untracked arrays.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")
    bin_ms = bin_ms if bin_ms is not None else int(BIN_SIZE_S * 1000)
    return f"rates_{subject}_{region}_{bin_ms}ms.npz"


def load_rate_matrix(subject: str, region: str, bin_ms: int | None = None) -> dict:
    """Load a stage-1 rate matrix as a dict of arrays."""
    path = processed_path(rate_matrix_name(subject, region, bin_ms))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/01_extraction.py first")
    return dict(np.load(path, allow_pickle=False))


def available_rate_matrices(bin_ms: int | None = None) -> list[tuple[str, str, Path]]:
    """List (subject, region, path) for stage-1 rate matrices at a given bin size."""
    bin_ms = bin_ms if bin_ms is not None else int(BIN_SIZE_S * 1000)
    out = []
    for path in sorted(DATA_PROCESSED.glob(f"rates_*_{bin_ms}ms.npz")):
        subject, region, _ = path.stem[len("rates_"):].rsplit("_", 2)
        out.append((subject, region, path))
    return out
