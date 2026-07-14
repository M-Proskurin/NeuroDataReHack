"""Stage 1 (000978) — extract time-binned rate matrices from NWB.

Single-day W-track learning: each file has ~17 epochs alternating sleep and
W-track run sessions. For each region (CA1/PFC, never pooled) we bin spikes into
BIN_SIZE_S windows within every epoch (run AND sleep — sleep epochs feed the
replay analysis), sqrt-transform, and align position/velocity. Beyond the shared
metadata we carry the 000978-specific columns the plan needs:

    kind         'run' | 'sleep'   (run epochs contain W-track trials)
    run_session  1..N for run bins in temporal order, -1 for sleep bins

Run vs. sleep is decided by the `trials` table (run epochs contain trials).

**ZT2 note:** subject ZT2's single day is split across two NWB files (9 + 8
epochs). We key each rate matrix by a *session key* from the filename
(e.g. 'JS14', or 'ZT2_obj-u40err') and also store the true subject id, so the
two ZT2 files stay distinct; their run-session order is per-file and must be
stitched in the session-sequence analysis.

Output: data/processed/000978/rates_<session_key>_<region>_<bin>ms.npz with
    rates       (T, n_units) float32  sqrt(spike counts per bin)
    time        (T,)         float64  bin-center time (s), session clock
    position    (T, 2)       float32  x, y cm (nan outside tracking)
    velocity    (T,)         float32  speed cm/s
    epoch       (T,)         int16    epoch index within file (0..n-1)
    kind        (T,)         <U5      'run' | 'sleep'
    run_session (T,)         int16    1..N for run bins, -1 for sleep
    unit_ids    (n_units,)   int64
    subject     ()           <U..     true subject id (e.g. JDS-SingleDay-ZT2)
    session_key ()           <U..     filename-derived key (e.g. ZT2_obj-u40err)
    region      ()           <U..     'CA1' | 'PFC'
    bin_size_s  ()           float64
    source      ()           <U..     DANDI asset path

Usage:
    pixi run python src/000978/01_extraction.py                 # all files, both regions
    pixi run python src/000978/01_extraction.py --session <asset-path>
    pixi run python src/000978/01_extraction.py --bin-ms 1000
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np

import download as dl
from config import (
    BIN_SIZE_S,
    REGIONS,
    SQRT_TRANSFORM,
    processed_path,
    rate_matrix_name,
)

DANDISET = "000978"
EPOCH_TABLE = "epoch intervals"


def session_key(asset_path: str) -> str:
    """Filename-derived key, unique per NWB file (disambiguates ZT2's two files)."""
    name = asset_path.split("/")[-1].replace("_behavior+ecephys.nwb", "")
    return name.replace("sub-JDS-SingleDay-", "")


def unit_regions(nwb) -> np.ndarray:  # noqa: ANN001
    """Region label ('CA1'/'PFC') per unit via the linked electrodes table."""
    n = len(nwb.units)
    out = np.empty(n, dtype=object)
    for i in range(n):
        locs = nwb.units["electrodes"][i]["location"].unique().tolist()
        out[i] = locs[0] if len(locs) == 1 else "mixed"
    return out


def classify_epochs(epochs, trials):  # noqa: ANN001
    """Per-epoch ('run'/'sleep', run_session): run epochs contain W-track trials."""
    tr = trials["start_time"].to_numpy()
    kinds, run_session, r = [], [], 0
    for row in epochs.itertuples():
        is_run = bool(np.any((tr >= row.start_time) & (tr < row.stop_time)))
        if is_run:
            r += 1
            kinds.append("run")
            run_session.append(r)
        else:
            kinds.append("sleep")
            run_session.append(-1)
    return kinds, run_session


def _epoch_bins(start: float, stop: float, dt: float) -> np.ndarray:
    n = int(np.floor((stop - start) / dt))
    return start + dt * np.arange(n + 1)


def extract_region(nwb, region: str, bin_size_s: float) -> dict:  # noqa: ANN001
    """Binned rate matrix + metadata for one region of one file (call in stream)."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")

    regions = unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    if unit_idx.size == 0:
        raise ValueError(f"no units in region {region!r}")
    unit_ids = np.asarray(nwb.units.id[:])[unit_idx]
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    epochs = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    trials = nwb.intervals["trials"].to_dataframe()
    kinds, run_session = classify_epochs(epochs, trials)

    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    pos_t = np.asarray(ss.timestamps[:])
    pos = np.asarray(ss.data[:])  # (N, 3): x, y (cm), speed (cm/s)

    counts_b, time_b, epoch_b, kind_b, sess_b = [], [], [], [], []
    for e, row in epochs.iterrows():
        edges = _epoch_bins(row["start_time"], row["stop_time"], bin_size_s)
        if edges.size < 2:
            continue
        centers = edges[:-1] + bin_size_s / 2.0
        block = np.empty((centers.size, unit_idx.size), dtype=np.float64)
        for u, st in enumerate(spike_times):
            block[:, u], _ = np.histogram(st, bins=edges)
        counts_b.append(block)
        time_b.append(centers)
        epoch_b.append(np.full(centers.size, e, dtype=np.int16))
        kind_b.append(np.full(centers.size, kinds[e], dtype="<U5"))
        sess_b.append(np.full(centers.size, run_session[e], dtype=np.int16))

    counts = np.vstack(counts_b)
    time = np.concatenate(time_b)
    rates = (np.sqrt(counts) if SQRT_TRANSFORM else counts).astype(np.float32)

    x = np.interp(time, pos_t, pos[:, 0], left=np.nan, right=np.nan)
    y = np.interp(time, pos_t, pos[:, 1], left=np.nan, right=np.nan)
    velocity = np.interp(time, pos_t, pos[:, 2], left=np.nan, right=np.nan)

    return {
        "rates": rates,
        "time": time,
        "position": np.column_stack([x, y]).astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "epoch": np.concatenate(epoch_b),
        "kind": np.concatenate(kind_b),
        "run_session": np.concatenate(sess_b),
        "unit_ids": unit_ids.astype(np.int64),
        "bin_size_s": np.asarray(float(bin_size_s)),
    }


def extract_session(asset_path: str, regions=REGIONS, bin_size_s: float = BIN_SIZE_S):
    """Stream one file and write one rate matrix per region. Returns paths."""
    bin_ms = int(round(bin_size_s * 1000))
    key = session_key(asset_path)
    written = []
    with dl.stream_nwb(asset_path, dandiset_id=DANDISET) as nwb:
        subject = getattr(nwb.subject, "subject_id", "unknown")
        for region in regions:
            data = extract_region(nwb, region, bin_size_s)
            data["subject"] = np.asarray(subject)
            data["session_key"] = np.asarray(key)
            data["region"] = np.asarray(region)
            data["source"] = np.asarray(asset_path)
            out = processed_path(rate_matrix_name(key, region, bin_ms), DANDISET)
            np.savez_compressed(out, **data)
            T, n = data["rates"].shape
            n_run = int(np.sum(np.unique(data["run_session"]) > 0))
            print(f"  {key} {region}: {T} bins x {n} units, {n_run} run sessions "
                  f"-> {out.name}")
            written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", action="append", metavar="ASSET_PATH",
                        help="asset path to extract (repeatable; default: all)")
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        choices=list(REGIONS))
    args = parser.parse_args()

    sessions = args.session or [
        p for p in dl.list_asset_paths(dandiset_id=DANDISET) if p.endswith(".nwb")
    ]
    bin_size_s = args.bin_ms / 1000.0
    print(f"000978 extraction: {len(sessions)} file(s), regions={args.regions}, "
          f"bin={args.bin_ms}ms")
    for path in sessions:
        print(path)
        extract_session(path, regions=args.regions, bin_size_s=bin_size_s)


if __name__ == "__main__":
    main()
