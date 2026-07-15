"""Stage 6c-fine (000978) — fine-timescale sleep replay into the awake manifold.

The 1 s sleep projection (06c_sleep_projection.py) found no co-activation signal
because ripple replay lives at ~100-200 ms. This re-runs the projection at a fine
bin (default 20 ms) with Gaussian smoothing (default 50 ms), which is the right
scale for replay, and tests for cross-unit co-activation structure with a
**circular time-shift** null (per-unit random circular shift preserves each
unit's smoothed autocorrelation while destroying cross-unit alignment — the
standard co-activation control; a plain permutation would wreck the smoothing).

Per session file and region:
  1. Fine-bin + smooth run and sleep spike trains; standardize by awake stats.
  2. Awake manifold = PCA on smoothed run; project run and sleep.
  3. occupancy = sleep->awake NN distance / awake self-spacing.
  4. Null: circularly shift each unit's sleep, reproject -> occupancy_shift.
     replay_index = shift / real (>1 => sleep is closer to the awake manifold
     than chance, i.e. genuine fine-timescale population structure).

Only metrics are saved (fine matrices are large and stay in memory).

Output: data/processed/000978/stage6c_fine_replay_<fine_ms>ms.csv

Usage:
    pixi run python src/000978/06c_fine_replay.py                 # all files
    pixi run python src/000978/06c_fine_replay.py --session <asset-path>
    pixi run python src/000978/06c_fine_replay.py --fine-ms 20 --smooth-ms 50
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import download as dl
from config import RANDOM_SEED, REGIONS, processed_path

DANDISET = "000978"
ex = import_module("01_extraction")

K_DIM = 10
REF_SUB = 3000        # awake reference points for NN
QUERY_SUB = 30000     # sleep points queried for NN
N_SHIFT = 5


def _fine_smoothed(spike_times, epochs, kinds, want, fine_s, sigma_bins):
    """(T, n_units) float32 smoothed spike counts for epochs of the wanted kind."""
    blocks = []
    for e, row in epochs.iterrows():
        if kinds[e] != want:
            continue
        s, t = float(row["start_time"]), float(row["stop_time"])
        n = int(np.floor((t - s) / fine_s))
        edges = s + fine_s * np.arange(n + 1)
        block = np.empty((n, len(spike_times)), dtype=np.float32)
        for u, st in enumerate(spike_times):
            block[:, u], _ = np.histogram(st, bins=edges)
        blocks.append(gaussian_filter1d(block, sigma=sigma_bins, axis=0, mode="nearest"))
    return np.vstack(blocks)


def _nn(query, ref, k):
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    d, _ = nn.kneighbors(query)
    return float(d[:, -1].mean())


def analyze_region(nwb, session_key, region, fine_s, sigma_bins, rng):  # noqa: ANN001
    regions = ex.unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    if unit_idx.size <= K_DIM:
        return None
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]
    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    kinds, _ = ex.classify_epochs(epochs, nwb.intervals["trials"].to_dataframe())

    run = _fine_smoothed(spike_times, epochs, kinds, "run", fine_s, sigma_bins)
    mu, sd = run.mean(0), run.std(0) + 1e-9
    run_z = (run - mu) / sd
    k = min(K_DIM, unit_idx.size - 1)
    fit_idx = rng.choice(run_z.shape[0], min(50000, run_z.shape[0]), replace=False)
    pca = PCA(n_components=k).fit(run_z[fit_idx])
    run_p = pca.transform(run_z)
    del run, run_z

    ref = run_p[rng.choice(run_p.shape[0], min(REF_SUB, run_p.shape[0]), replace=False)]
    d_awake = _nn(ref, ref, k=2)

    sleep = _fine_smoothed(spike_times, epochs, kinds, "sleep", fine_s, sigma_bins)
    q_idx = rng.choice(sleep.shape[0], min(QUERY_SUB, sleep.shape[0]), replace=False)
    sleep_p = pca.transform((sleep - mu) / sd)
    d_sleep = _nn(sleep_p[q_idx], ref, k=1)

    d_shift = []
    for _ in range(N_SHIFT):
        sh = np.empty_like(sleep)
        for j in range(sleep.shape[1]):
            sh[:, j] = np.roll(sleep[:, j], int(rng.integers(1, sleep.shape[0])))
        d_shift.append(_nn(pca.transform((sh - mu) / sd)[q_idx], ref, k=1))
    del sleep
    d_shift = float(np.mean(d_shift))

    return dict(
        session_key=session_key, region=region, n_units=int(unit_idx.size),
        occupancy_real=d_sleep / d_awake, occupancy_shift=d_shift / d_awake,
        replay_index=d_shift / d_sleep,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", action="append", metavar="ASSET_PATH")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    parser.add_argument("--fine-ms", type=int, default=20)
    parser.add_argument("--smooth-ms", type=int, default=50)
    args = parser.parse_args()

    fine_s = args.fine_ms / 1000.0
    sigma_bins = args.smooth_ms / args.fine_ms
    sessions = args.session or [
        p for p in dl.list_asset_paths(dandiset_id=DANDISET) if p.endswith(".nwb")
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    print(f"fine replay: {len(sessions)} file(s), fine={args.fine_ms}ms smooth={args.smooth_ms}ms")
    for asset in sessions:
        print(asset)
        with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
            key = ex.session_key(asset)
            for region in args.regions:
                row = analyze_region(nwb, key, region, fine_s, sigma_bins, rng)
                if row is None:
                    print(f"  {key} {region}: skipped")
                    continue
                rows.append(row)
                print(f"  {key} {region}: occupancy real={row['occupancy_real']:.2f} "
                      f"shift={row['occupancy_shift']:.2f} "
                      f"replay index={row['replay_index']:.2f}")

    df = pd.DataFrame(rows)
    out = processed_path(f"stage6c_fine_replay_{args.fine_ms}ms.csv", DANDISET)
    df.to_csv(out, index=False)
    print("\nmean by region:")
    print(df.groupby("region")[["occupancy_real", "occupancy_shift", "replay_index"]]
          .mean().round(3).to_string())
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
