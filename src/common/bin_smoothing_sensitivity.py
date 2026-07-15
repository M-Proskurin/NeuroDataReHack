"""Bin/smoothing sensitivity check (smoothed fine-bin path, 50 ms).

Tests whether the headline geometric findings survive a better preprocessing
choice than the 1000 ms hard bins: a fine 50 ms bin + Gaussian temporal kernel
(sigma selected by cross-validated position decoding) + a movement (speed) filter.

To keep it embedding-independent and tractable at 50 ms, the headlines are
recomputed in **rate space** (population-vector maps): position-binned centroids
of the smoothed population rate, compared with Procrustes + a shuffled null.

  * sigma selection — sweep Gaussian sigma; pick the value maximizing CV kNN
    position-decoding R^2 on smoothed, speed-filtered awake data.
  * 000447 headline — novel vs. familiar map disparity (per subject/region).
  * 000978 headline — each run session's map vs. the final session, as a function
    of session number (7 clean animals; ZT2 excluded).

000447 fine data is read from the saved 50 ms matrices; 000978 run epochs are
streamed and fine-binned at 50 ms on the fly (no giant matrices persisted).

Usage:
    pixi run python src/common/bin_smoothing_sensitivity.py --dandiset 000447
    pixi run python src/common/bin_smoothing_sensitivity.py --dandiset 000978
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import procrustes

import download as dl
import preprocessing as pp
from config import (
    RANDOM_SEED,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
    spatial_grid_labels,
)

BIN_MS = 50
N_GRID = 6
MIN_COUNT = 5
SPEED = 4.0
SIGMAS_MS = [0, 25, 50, 75, 100, 150]
N_SHUFFLE = 100
EPOCH_TABLE = "epoch intervals"


# ---------------------------------------------------------------- data loaders
def _unit_regions(nwb):  # noqa: ANN001
    return np.array([nwb.units["electrodes"][i]["location"].unique()[0]
                     for i in range(len(nwb.units))], dtype=object)


def _classify(epochs, trials):  # noqa: ANN001
    tr = trials["start_time"].to_numpy()
    kinds, sess, r = [], [], 0
    for row in epochs.itertuples():
        if np.any((tr >= row.start_time) & (tr < row.stop_time)):
            r += 1; kinds.append("run"); sess.append(r)
        else:
            kinds.append("sleep"); sess.append(-1)
    return kinds, sess


def load_awake_447() -> list[dict]:
    out = []
    for subj, region, _ in available_rate_matrices(BIN_MS, "000447"):
        d = load_rate_matrix(subj, region, BIN_MS, "000447")
        out.append(dict(key=subj, region=region,
                        counts=(d["rates"].astype(np.float64) ** 2).astype(np.float32),
                        position=d["position"], velocity=d["velocity"],
                        epoch=d["epoch"], label=d["condition"]))
    return out


def load_awake_978() -> list[dict]:
    fine_s = BIN_MS / 1000.0
    out = []
    assets = [p for p in dl.list_asset_paths(dandiset_id="000978")
              if p.endswith(".nwb") and "ZT2" not in p]           # exclude ZT2 (two days)
    for asset in assets:
        key = asset.split("/")[-1].replace("_behavior+ecephys.nwb", "").replace("sub-JDS-SingleDay-", "")
        print(f"  streaming {key} ...", flush=True)
        with dl.stream_nwb(asset, dandiset_id="000978") as nwb:
            regions = _unit_regions(nwb)
            epochs = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
            kinds, sess = _classify(epochs, nwb.intervals["trials"].to_dataframe())
            ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
            pos_t, pos = np.asarray(ss.timestamps[:]), np.asarray(ss.data[:])
            for region in REGIONS:
                uidx = np.flatnonzero(regions == region)
                spikes = [np.asarray(nwb.units["spike_times"][int(i)]) for i in uidx]
                cb, eb, sb, tb = [], [], [], []
                for e, row in epochs.iterrows():
                    if kinds[e] != "run":
                        continue
                    s, t = float(row["start_time"]), float(row["stop_time"])
                    n = int(np.floor((t - s) / fine_s))
                    edges = s + fine_s * np.arange(n + 1)
                    centers = edges[:-1] + fine_s / 2
                    blk = np.empty((n, uidx.size), dtype=np.float32)
                    for u, st in enumerate(spikes):
                        blk[:, u], _ = np.histogram(st, bins=edges)
                    cb.append(blk); eb.append(np.full(n, e, np.int16))
                    sb.append(np.full(n, sess[e], np.int16)); tb.append(centers)
                counts = np.vstack(cb); time = np.concatenate(tb)
                out.append(dict(
                    key=key, region=region, counts=counts,
                    position=np.column_stack([np.interp(time, pos_t, pos[:, 0], left=np.nan, right=np.nan),
                                              np.interp(time, pos_t, pos[:, 1], left=np.nan, right=np.nan)]).astype(np.float32),
                    velocity=np.interp(time, pos_t, pos[:, 2], left=np.nan, right=np.nan).astype(np.float32),
                    epoch=np.concatenate(eb), label=np.concatenate(sb)))
    return out


# ---------------------------------------------------------------- analysis
def _smoothed_masked(entry: dict, sigma_ms: int):
    sm = pp.smooth_per_epoch(entry["counts"], entry["epoch"], sigma_ms / BIN_MS)
    m = pp.speed_mask(entry["velocity"], SPEED)
    return sm[m], entry["position"][m], entry["label"][m]


def select_sigma(data: list[dict]) -> pd.DataFrame:
    rows = []
    for sigma in SIGMAS_MS:
        r2s = []
        for e in data:
            X, pos, _ = _smoothed_masked(e, sigma)
            r2s.append(pp.cv_position_r2(X, pos))
        rows.append(dict(sigma_ms=sigma, mean_decode_r2=float(np.nanmean(r2s))))
    return pd.DataFrame(rows)


def _centroids(X, labels, mask, bins):
    return np.vstack([X[mask & (labels == b)].mean(0) for b in bins])


def _shared_bins(labels, masks):
    ok = None
    for m in masks:
        cnt = np.bincount(labels[m & (labels >= 0)], minlength=N_GRID * N_GRID)
        ok = (cnt >= MIN_COUNT) if ok is None else (ok & (cnt >= MIN_COUNT))
    return np.flatnonzero(ok)


def _procrustes_disp(A, B, rng):
    _, _, disp = procrustes(A, B)
    null = np.array([procrustes(A, B[rng.permutation(len(B))])[2] for _ in range(N_SHUFFLE)])
    return float(disp), float(null.mean())


def headline_447(data, sigma, rng) -> pd.DataFrame:
    rows = []
    for e in data:
        X, pos, cond = _smoothed_masked(e, sigma)
        labels = spatial_grid_labels(pos, N_GRID)
        nov, fam = cond == "novel", cond == "familiar"
        bins = _shared_bins(labels, [nov, fam])
        if bins.size < 4:
            continue
        A, B = _centroids(X, labels, nov, bins), _centroids(X, labels, fam, bins)
        disp, null = _procrustes_disp(A, B, rng)
        rows.append(dict(subject=e["key"], region=e["region"], n_bins=int(bins.size),
                         disparity=disp, null=null))
    return pd.DataFrame(rows)


def headline_978(data, sigma, rng) -> pd.DataFrame:
    rows = []
    for e in data:
        X, pos, sess = _smoothed_masked(e, sigma)
        labels = spatial_grid_labels(pos, N_GRID)
        sessions = sorted(np.unique(sess[sess > 0]).tolist())
        final = sessions[-1]
        for s in sessions:
            bins = _shared_bins(labels, [sess == s, sess == final])
            if bins.size < 4:
                continue
            A, B = _centroids(X, labels, sess == s, bins), _centroids(X, labels, sess == final, bins)
            disp, _ = _procrustes_disp(A, B, rng)
            rows.append(dict(session_key=e["key"], region=e["region"], session=int(s),
                             disparity=disp))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dandiset", required=True, choices=["000447", "000978"])
    args = ap.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"loading awake fine ({BIN_MS}ms) data for {args.dandiset} ...")
    data = load_awake_447() if args.dandiset == "000447" else load_awake_978()

    print("\nsigma selection (CV position-decoding R^2):")
    sig = select_sigma(data)
    print(sig.to_string(index=False))
    best = int(sig.loc[sig.mean_decode_r2.idxmax(), "sigma_ms"])
    print(f"-> best sigma = {best} ms")
    sig.to_csv(processed_path(f"sensitivity_sigma_{args.dandiset}.csv", args.dandiset), index=False)

    print(f"\nheadline at sigma={best} ms, speed>{SPEED} cm/s, {BIN_MS}ms bins:")
    if args.dandiset == "000447":
        h = headline_447(data, best, rng)
        out = processed_path("sensitivity_headline_000447.csv", "000447")
        h.to_csv(out, index=False)
        print(f"  novel vs familiar disparity: {h.disparity.mean():.3f} "
              f"(null {h.null.mean():.3f}), n={len(h)}")
        print(f"  -> below null and >0 => partial remap holds: "
              f"{h.disparity.mean() < h.null.mean()}")
    else:
        h = headline_978(data, best, rng)
        out = processed_path("sensitivity_headline_000978.csv", "000978")
        h.to_csv(out, index=False)
        per = h.groupby("session")["disparity"].mean()
        print("  disparity to final by session:")
        for s, v in per.items():
            print(f"    session {s}: {v:.3f}")
        early = h[h.session <= 2].disparity.mean()
        late = h[h.session >= 6].disparity.mean()
        print(f"  early(<=2) {early:.3f} vs late(>=6) {late:.3f} "
              f"-> converges: {late < early}")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
