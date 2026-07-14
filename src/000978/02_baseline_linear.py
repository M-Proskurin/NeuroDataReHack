"""Stage 2 (000978) — linear baselines: PCA, GPFA, dPCA.

Single-day W-track learning. Baselines are computed on the **run** epochs (the
awake W-track manifold); sleep epochs are left for the replay analysis (6c). Run
per session file and per region (CA1/PFC never pooled).

  * **PCA** — variance structure of the run-epoch population; scores carry the
    run_session / position metadata so later stages can track drift.
  * **dPCA** — demixed PCA over (space x run_session): splits variance into a
    space-dependent part (the stable map), a session-dependent part (global
    drift across learning), and their interaction (map reshaping during
    learning). This is the 000978 analogue of 000447's space x condition and the
    "factor out position/session variance" step in the plan.
  * **GPFA** — Gaussian-process factor analysis on re-streamed run-epoch spike
    trains, chopped into fixed-length trials labelled by run_session.

**ZT2** is split across two files (session keys ZT2_obj-*), each with 4 run
sessions numbered per file; treat them as separate sessions here.

Usage:
    pixi run python src/000978/02_baseline_linear.py                 # all methods, all matrices
    pixi run python src/000978/02_baseline_linear.py --method pca dpca
    pixi run python src/000978/02_baseline_linear.py --method gpfa --session <asset-path>
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse
from importlib import import_module

import numpy as np
from sklearn.decomposition import PCA

import download as dl
from config import (
    BIN_SIZE_S,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
    spatial_grid_labels,
)

DANDISET = "000978"
ex = import_module("01_extraction")  # 000978 extraction helpers (same folder)


def _run_mask(d: dict) -> np.ndarray:
    return d["kind"] == "run"


# ----------------------------------------------------------------------------
# PCA (run epochs)
# ----------------------------------------------------------------------------

def run_pca(session_key: str, region: str, bin_ms: int, n_keep: int = 20):
    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    run = _run_mask(d)
    rates = d["rates"][run].astype(np.float64)
    n_comp = min(rates.shape)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(rates)
    evr = pca.explained_variance_ratio_
    k = min(n_keep, n_comp)

    out = processed_path(f"pca_{session_key}_{region}_{bin_ms}ms.npz", DANDISET)
    np.savez_compressed(
        out,
        scores=scores[:, :k].astype(np.float32),
        components=pca.components_[:k].astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        run_session=d["run_session"][run], epoch=d["epoch"][run],
        time=d["time"][run], position=d["position"][run], velocity=d["velocity"][run],
        unit_ids=d["unit_ids"], subject=d["subject"], session_key=np.asarray(session_key),
        region=np.asarray(region), bin_size_s=d["bin_size_s"],
    )
    cum = np.cumsum(evr)
    d80 = int(np.searchsorted(cum, 0.80) + 1)
    print(f"  PCA  {session_key} {region}: {rates.shape[1]} units, "
          f"{run.sum()} run bins -> {d80} PCs for 80% var -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# dPCA (space x run_session, run epochs)
# ----------------------------------------------------------------------------

def run_dpca(session_key: str, region: str, bin_ms: int, n_grid: int = 6,
             min_count: int = 5, n_components: int = 10):
    from dPCA import dPCA

    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    run = _run_mask(d)
    rates = d["rates"][run].astype(np.float64)
    rs = d["run_session"][run]
    sessions = sorted(np.unique(rs).tolist())
    if len(sessions) < 2:
        print(f"  dPCA {session_key} {region}: <2 run sessions, skipped")
        return None

    labels = spatial_grid_labels(d["position"][run], n_grid)
    ok = None
    for s in sessions:
        cnt = np.bincount(labels[(rs == s) & (labels >= 0)], minlength=n_grid * n_grid)
        good = cnt >= min_count
        ok = good if ok is None else (ok & good)
    space = np.flatnonzero(ok)
    if space.size < 3:
        print(f"  dPCA {session_key} {region}: only {space.size} shared space bins, skipped")
        return None

    N = rates.shape[1]
    X = np.zeros((N, space.size, len(sessions)), dtype=np.float64)
    for qi, s in enumerate(sessions):
        for si, b in enumerate(space):
            m = (rs == s) & (labels == b)
            X[:, si, qi] = rates[m].mean(axis=0)
    X -= X.mean(axis=(1, 2), keepdims=True)

    k = int(min(n_components, N, X.shape[1] * X.shape[2] - 1))
    dpca = dPCA.dPCA(labels="sq", n_components=k, regularizer=None)
    dpca.protect = None
    Z = dpca.fit_transform(X)
    evr = dpca.explained_variance_ratio_

    out = processed_path(f"dpca_{session_key}_{region}_{bin_ms}ms.npz", DANDISET)
    np.savez_compressed(
        out,
        Z_s=Z["s"].astype(np.float32), Z_q=Z["q"].astype(np.float32),
        Z_sq=Z["sq"].astype(np.float32),
        evr_s=np.asarray(evr["s"], np.float32), evr_q=np.asarray(evr["q"], np.float32),
        evr_sq=np.asarray(evr["sq"], np.float32),
        space_bins=space.astype(np.int64), sessions=np.asarray(sessions, np.int64),
        n_units=np.asarray(N), subject=d["subject"],
        session_key=np.asarray(session_key), region=np.asarray(region),
        bin_size_s=d["bin_size_s"],
    )
    frac = {m: float(np.sum(evr[m])) for m in ("s", "q", "sq")}
    print(f"  dPCA {session_key} {region}: {N} units, {space.size} bins x "
          f"{len(sessions)} sessions -> var space={frac['s']:.2f} "
          f"session={frac['q']:.2f} interact={frac['sq']:.2f} -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# GPFA (run-epoch spike trains)
# ----------------------------------------------------------------------------

def _gpfa_run_trials(nwb, region: str, trial_len_s: float):  # noqa: ANN001
    import neo
    import quantities as pq

    regions = ex.unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    trials_tab = nwb.intervals["trials"].to_dataframe()
    kinds, run_session = ex.classify_epochs(epochs, trials_tab)

    trials, tr_session = [], []
    for e, row in epochs.iterrows():
        if kinds[e] != "run":
            continue
        s, t = float(row["start_time"]), float(row["stop_time"])
        n = int(np.floor((t - s) / trial_len_s))
        for k in range(n):
            w0 = s + k * trial_len_s
            trial = [neo.SpikeTrain((st[(st >= w0) & (st < w0 + trial_len_s)] - w0) * pq.s,
                                    t_start=0 * pq.s, t_stop=trial_len_s * pq.s)
                     for st in spike_times]
            trials.append(trial)
            tr_session.append(run_session[e])
    return trials, np.asarray(tr_session, np.int16), unit_idx


def run_gpfa(nwb, session_key: str, region: str, x_dim: int = 8,
             gpfa_bin_ms: int = 100, trial_len_s: float = 5.0, em_max_iters: int = 50):
    import quantities as pq
    from elephant.gpfa import GPFA

    trials, tr_session, unit_idx = _gpfa_run_trials(nwb, region, trial_len_s)
    if len(trials) < 5 or unit_idx.size <= x_dim:
        print(f"  GPFA {session_key} {region}: too few trials/units, skipped")
        return None

    gpfa = GPFA(bin_size=gpfa_bin_ms * pq.ms, x_dim=x_dim, em_max_iters=em_max_iters)
    trajs = gpfa.fit_transform(trials)
    latents = np.stack(trajs, axis=0).astype(np.float32)

    out = processed_path(f"gpfa_{session_key}_{region}.npz", DANDISET)
    np.savez_compressed(
        out, latents=latents, trial_run_session=tr_session,
        unit_ids=np.asarray(nwb.units.id[:])[unit_idx].astype(np.int64),
        x_dim=np.asarray(x_dim), gpfa_bin_ms=np.asarray(gpfa_bin_ms),
        trial_len_s=np.asarray(trial_len_s),
        session_key=np.asarray(session_key), region=np.asarray(region),
    )
    print(f"  GPFA {session_key} {region}: {latents.shape[0]} trials x {x_dim} dims "
          f"x {latents.shape[2]} bins -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------

def _key_to_asset() -> dict:
    return {ex.session_key(p): p for p in dl.list_asset_paths(dandiset_id=DANDISET)
            if p.endswith(".nwb")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=["pca", "dpca", "gpfa"],
                        choices=["pca", "dpca", "gpfa"])
    parser.add_argument("--session", action="append", metavar="ASSET_PATH")
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    parser.add_argument("--gpfa-xdim", type=int, default=8)
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms, DANDISET)
                if r in args.regions]
    if "pca" in args.method or "dpca" in args.method:
        print(f"PCA/dPCA on {len(matrices)} run-epoch matrices (bin={args.bin_ms}ms)")
        for session_key, region in matrices:
            if "pca" in args.method:
                run_pca(session_key, region, args.bin_ms)
            if "dpca" in args.method:
                run_dpca(session_key, region, args.bin_ms)

    if "gpfa" in args.method:
        k2a = _key_to_asset()
        keys = sorted({s for (s, r) in matrices})
        assets = args.session or [k2a[k] for k in keys if k in k2a]
        print(f"GPFA on {len(assets)} file(s), regions={args.regions}, x_dim={args.gpfa_xdim}")
        for asset in assets:
            print(asset)
            with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
                key = ex.session_key(asset)
                for region in args.regions:
                    run_gpfa(nwb, key, region, x_dim=args.gpfa_xdim)


if __name__ == "__main__":
    main()
