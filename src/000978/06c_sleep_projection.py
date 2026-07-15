"""Stage 6c (000978) — sleep-epoch projection into the awake manifold.

Does sleep-epoch population activity sample the same geometric structure as the
awake W-track manifold (replay-consistent), or does it fall off that manifold?

Per session file and region:
  1. Fit an awake basis (PCA) on the **run** bins; project both run and sleep
     bins into the top-k awake PCs.
  2. Occupancy: for each sleep point, distance to the nearest awake point in the
     awake PC space, normalized by the awake points' own nearest-neighbour
     spacing -> `occupancy_real` (~1 means sleep sits within the awake manifold).
  3. Null: independently permute each unit's sleep activity over time (destroys
     cross-unit co-activation, preserves single-unit rate) and recompute ->
     `occupancy_shuffle`. If real << shuffle, sleep retains awake-like population
     geometry (the replay signature).
  4. Compression: ratio of sleep to awake spread per awake PC (is the sleep
     sampling of the manifold compressed?).

All 9 files are used (ZT2's two files are each valid awake+sleep recordings; the
ZT2 exclusion applies only to the 6b across-session trajectory).

Output: data/processed/000978/stage6c_sleep_projection_<bin>ms.csv

Usage:
    pixi run python src/000978/06c_sleep_projection.py
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from config import (
    BIN_SIZE_S,
    RANDOM_SEED,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
)

DANDISET = "000978"
K_DIM = 10
MAX_REF = 3000       # subsample awake reference for NN baseline (speed)
N_SHUFFLE = 10


def _nn_dist(query: np.ndarray, ref: np.ndarray, k: int = 1) -> float:
    """Mean distance from each query point to its nearest ref point."""
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    d, _ = nn.kneighbors(query)
    return float(d[:, -1].mean())


def analyze(session_key: str, region: str, bin_ms: int, rng) -> dict | None:
    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    kind = d["kind"]
    run, sleep = kind == "run", kind == "sleep"
    if run.sum() < 50 or sleep.sum() < 50:
        return None
    rates = d["rates"].astype(np.float64)
    run_X, sleep_X = rates[run], rates[sleep]

    # awake basis
    k = min(K_DIM, run_X.shape[1] - 1, run_X.shape[0] - 1)
    pca = PCA(n_components=k).fit(run_X)
    run_z = pca.transform(run_X)
    sleep_z = pca.transform(sleep_X)

    # awake reference (subsampled) + its own NN spacing
    ref = run_z
    if ref.shape[0] > MAX_REF:
        ref = ref[rng.choice(ref.shape[0], MAX_REF, replace=False)]
    d_awake = _nn_dist(ref, ref, k=2)                 # 2nd NN (skip self)
    d_sleep = _nn_dist(sleep_z, ref, k=1)

    # shuffle null: permute each unit's sleep activity over time, reproject
    d_shuf = []
    for _ in range(N_SHUFFLE):
        sh = np.empty_like(sleep_X)
        for j in range(sleep_X.shape[1]):
            sh[:, j] = sleep_X[rng.permutation(sleep_X.shape[0]), j]
        d_shuf.append(_nn_dist(pca.transform(sh), ref, k=1))
    d_sleep_shuffle = float(np.mean(d_shuf))

    # compression: sleep vs awake spread per PC (std ratio, averaged)
    compression = float(np.mean(sleep_z.std(0) / (run_z.std(0) + 1e-9)))

    return dict(
        session_key=session_key, region=region, n_run=int(run.sum()),
        n_sleep=int(sleep.sum()), k_dim=k,
        occupancy_real=d_sleep / d_awake,
        occupancy_shuffle=d_sleep_shuffle / d_awake,
        replay_index=d_sleep_shuffle / d_sleep,     # >1 => sleep closer than shuffle
        compression=compression,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms, DANDISET)
                if r in args.regions]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    print(f"6c sleep projection on {len(matrices)} matrices (bin={args.bin_ms}ms)")
    for session_key, region in matrices:
        row = analyze(session_key, region, args.bin_ms, rng)
        if row is None:
            print(f"  {session_key} {region}: skipped (too few run/sleep bins)")
            continue
        rows.append(row)
        print(f"  {session_key} {region}: occupancy real={row['occupancy_real']:.2f} "
              f"vs shuffle={row['occupancy_shuffle']:.2f} "
              f"(replay index {row['replay_index']:.2f}), "
              f"compression={row['compression']:.2f}")

    df = pd.DataFrame(rows)
    out = processed_path(f"stage6c_sleep_projection_{args.bin_ms}ms.csv", DANDISET)
    df.to_csv(out, index=False)
    print("\nmean by region:")
    print(df.groupby("region")[["occupancy_real", "occupancy_shuffle",
                                "replay_index", "compression"]].mean().round(3).to_string())
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
