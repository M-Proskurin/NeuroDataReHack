"""Shared paths and conventions for the dandiset 000447 manifold analysis.

Import this everywhere so pipeline stages agree on bin size, region names,
and where intermediate artifacts live. See CLAUDE.md for the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

# --- paths (repo-relative; data/ dirs are gitignored) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"          # downloaded NWB files
DATA_PROCESSED = REPO_ROOT / "data" / "processed"  # rate matrices, embeddings

# --- dataset ---
DANDISET_ID = "000447"

# --- regions: process separately, never pool units across them ---
REGIONS = ("CA1", "PFC")

# --- time binning (stage 1) ---
BIN_SIZE_S = 0.05        # 50 ms bins to start; revisit per open question
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
