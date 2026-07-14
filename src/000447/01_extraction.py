"""Stage 1 — extract time-binned rate matrices from NWB.

For each subject and each region (CA1, PFC processed separately), pull spike
times from `Units`, bin them into BIN_SIZE_S windows within each behavioral
epoch, sqrt-transform for variance stabilization, and align position/velocity
from the behavior `SpatialSeries` to the bin centers. Epoch index, condition
(novel/familiar), subject, and region travel with the neural matrix in one
`.npz` — never as separate untracked arrays (see CLAUDE.md).

Bins are built per epoch and concatenated, so there is no single contiguous
edge array; we store the bin-center `time` instead of `bin_edges`.

Output: data/processed/rates_<subject>_<region>_<bin>ms.npz containing
    rates      (T, n_units) float32  sqrt(spike counts per bin) if SQRT_TRANSFORM
    time       (T,)         float64  bin-center time (s), session clock
    position   (T, 2)       float32  x, y in cm (nan outside tracking coverage)
    velocity   (T,)         float32  speed in cm/s (nan outside coverage)
    epoch      (T,)         int16    epoch index (0..n_epochs-1)
    condition  (T,)         <U8      'novel' | 'familiar'
    unit_ids   (n_units,)   int64    NWB unit ids kept for this region
    subject    ()           <U..     animal ID
    region     ()           <U..     'CA1' | 'PFC'
    bin_size_s ()           float64  bin width used
    source     ()           <U..     DANDI asset path

Usage:
    pixi run python src/000447/01_extraction.py                 # all sessions, both regions
    pixi run python src/000447/01_extraction.py --session <asset-path>
    pixi run python src/000447/01_extraction.py --bin-ms 100
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse
import numpy as np

from config import (
    BIN_SIZE_S,
    REGIONS,
    SQRT_TRANSFORM,
    processed_path,
    rate_matrix_name,
)

# stage-0 helper (module name starts with a digit)
import download as dl                    # streaming helper from src/common

EPOCH_TABLE = "epoch intervals"
CONDITION_WORDS = ("novel", "familiar")


def unit_regions(nwb) -> np.ndarray:  # noqa: ANN001
    """Region label ('CA1'/'PFC') per unit, via the linked electrodes table."""
    n = len(nwb.units)
    out = np.empty(n, dtype=object)
    for i in range(n):
        locs = nwb.units["electrodes"][i]["location"].unique().tolist()
        out[i] = locs[0] if len(locs) == 1 else "mixed"
    return out


def epoch_conditions(session_description: str, n_epochs: int) -> list[str]:
    """Per-epoch condition from the session description, e.g.
    'Novel familiar novel experiment' -> ['novel', 'familiar', 'novel'].

    Falls back to positional 'epoch{i}' labels if the description can't be
    matched to the epoch count.
    """
    words = [w.lower().strip(".,") for w in session_description.split()]
    conds = [w for w in words if w in CONDITION_WORDS]
    if len(conds) >= n_epochs:
        return conds[:n_epochs]
    print(f"  WARNING: could not map description {session_description!r} to "
          f"{n_epochs} epochs; using positional labels")
    return [f"epoch{i}" for i in range(n_epochs)]


def _epoch_bins(start: float, stop: float, dt: float) -> np.ndarray:
    """Bin edges covering [start, stop] with width dt, dropping the partial tail."""
    n = int(np.floor((stop - start) / dt))
    return start + dt * np.arange(n + 1)


def extract_region(nwb, region: str, bin_size_s: float) -> dict:  # noqa: ANN001
    """Build the binned rate matrix + metadata for one region of one session.

    All NWB reads happen here; call inside a `stream_nwb` context.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")

    regions = unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    if unit_idx.size == 0:
        raise ValueError(f"no units in region {region!r}")
    unit_ids = np.asarray(nwb.units.id[:])[unit_idx]
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    epochs = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    conds = epoch_conditions(nwb.session_description, len(epochs))

    # behavior: x, y (cm), speed (cm/s) sampled on the SpatialSeries timestamps
    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    pos_t = np.asarray(ss.timestamps[:])
    pos = np.asarray(ss.data[:])  # (N, 3)

    counts_blocks, time_blocks, epoch_blocks, cond_blocks = [], [], [], []
    for e, row in epochs.iterrows():
        edges = _epoch_bins(row["start_time"], row["stop_time"], bin_size_s)
        if edges.size < 2:
            continue
        centers = edges[:-1] + bin_size_s / 2.0
        block = np.empty((centers.size, unit_idx.size), dtype=np.float64)
        for u, st in enumerate(spike_times):
            block[:, u], _ = np.histogram(st, bins=edges)
        counts_blocks.append(block)
        time_blocks.append(centers)
        epoch_blocks.append(np.full(centers.size, e, dtype=np.int16))
        cond_blocks.append(np.full(centers.size, conds[e], dtype="<U8"))

    counts = np.vstack(counts_blocks)
    time = np.concatenate(time_blocks)
    epoch = np.concatenate(epoch_blocks)
    condition = np.concatenate(cond_blocks)

    rates = np.sqrt(counts) if SQRT_TRANSFORM else counts
    rates = rates.astype(np.float32)

    # align behavior to bin centers (nan outside tracking coverage)
    x = np.interp(time, pos_t, pos[:, 0], left=np.nan, right=np.nan)
    y = np.interp(time, pos_t, pos[:, 1], left=np.nan, right=np.nan)
    velocity = np.interp(time, pos_t, pos[:, 2], left=np.nan, right=np.nan)
    position = np.column_stack([x, y]).astype(np.float32)

    return {
        "rates": rates,
        "time": time,
        "position": position,
        "velocity": velocity.astype(np.float32),
        "epoch": epoch,
        "condition": condition,
        "unit_ids": unit_ids.astype(np.int64),
        "subject": np.asarray(getattr(nwb.subject, "subject_id", "unknown")),
        "region": np.asarray(region),
        "bin_size_s": np.asarray(float(bin_size_s)),
    }


def extract_session(asset_path: str, regions=REGIONS, bin_size_s: float = BIN_SIZE_S):
    """Stream one session and write one rate matrix per region. Returns paths."""
    bin_ms = int(round(bin_size_s * 1000))
    written = []
    with dl.stream_nwb(asset_path) as nwb:
        subject = getattr(nwb.subject, "subject_id", "unknown")
        for region in regions:
            data = extract_region(nwb, region, bin_size_s)
            data["source"] = np.asarray(asset_path)
            out = processed_path(rate_matrix_name(subject, region, bin_ms))
            np.savez_compressed(out, **data)
            T, n = data["rates"].shape
            print(f"  {subject} {region}: {T} bins x {n} units -> {out.name}")
            written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", action="append", metavar="ASSET_PATH",
                        help="asset path to extract (repeatable; default: all)")
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000),
                        help="bin width in milliseconds")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        choices=list(REGIONS), help="regions to extract")
    args = parser.parse_args()

    sessions = args.session or [p for p in dl.list_asset_paths() if p.endswith(".nwb")]
    bin_size_s = args.bin_ms / 1000.0
    print(f"extracting {len(sessions)} session(s), regions={args.regions}, "
          f"bin={args.bin_ms}ms")
    for path in sessions:
        print(path)
        extract_session(path, regions=args.regions, bin_size_s=bin_size_s)


if __name__ == "__main__":
    main()
