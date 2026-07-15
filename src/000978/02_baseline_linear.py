"""Stage 2 (000978) — linear baselines: PCA, GPFA, dPCA.

Single-day W-track learning. Baselines are computed on the **run** epochs (the
awake W-track manifold); sleep epochs are left for the replay analysis (6c). Run
per session file and per region (CA1/PFC never pooled).

  * **PCA** — variance structure of the run-epoch population; scores carry the
    run_session / position metadata so later stages can track drift.
  * **dPCA** — demixed PCA over (run_session x space): splits variance into a
    space-dependent part (the stable map), a session-dependent part (global
    drift across learning), and their interaction (map reshaping during
    learning). Each **lap** (a trials-table row) is a trial, so we fit a genuine
    single-trial factorial tensor with cross-validated regularization
    (`regularizer='auto'`) and a shuffle `significance_analysis` — the session
    and interaction variance fractions become defensible numbers, not point
    estimates on an over-fit mean. This is the 000978 analogue of 000447's
    condition x space demixing.
  * **GPFA** — Gaussian-process factor analysis on re-streamed run-epoch spike
    trains, one trial per lap; latents are re-expressed as a function of
    linearized track position so trajectories are interpretable and comparable
    across sessions at matched positions.

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
import lap_baselines as lb          # lap-resolved dPCA + GPFA helpers
import linearize as lz              # W-track linearization for GPFA position index
from config import (
    BIN_SIZE_S,
    RANDOM_SEED,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
    spatial_grid_labels,
)

DANDISET = "000978"
N_GRID = 6                          # dPCA spatial grid (per side)
N_POSBINS = 30                      # GPFA linear-position bins
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
# dPCA — lap-resolved (run_session x space), cross-validated + significance
# ----------------------------------------------------------------------------

def _lap_sessions(nwb, lap_start):  # noqa: ANN001
    """run_session index for each lap (via the epoch it starts in; -1 if none)."""
    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    trials_tab = nwb.intervals["trials"].to_dataframe()
    _, run_session = ex.classify_epochs(epochs, trials_tab)
    out = np.full(len(lap_start), -1, dtype=np.int16)
    for i, t0 in enumerate(lap_start):
        for e, row in epochs.iterrows():
            if row.start_time <= t0 < row.stop_time:
                out[i] = run_session[e]
                break
    return out


def run_dpca(nwb, session_key: str, region: str, bin_ms: int,  # noqa: ANN001
             n_perm: int = 200, n_components: int = 10):
    """Demixed PCA over (run_session x space) with laps as trials.

    Each lap gives one single-trial estimate per space bin; the regularizer is
    chosen by lap-level cross-validation and the session / session x space
    variance fractions are tested by permuting the run-session label across laps.
    labels='qs' -> marginalizations 'q' (session drift), 's' (space = stable
    map), 'qs' (interaction = map reshaping during learning).
    """
    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    run = _run_mask(d)
    space = spatial_grid_labels(d["position"][run], N_GRID)
    lt = lb.lap_table(nwb.intervals["trials"].to_dataframe())
    res = lb.lap_dpca(d["rates"][run], d["time"][run], space, d["run_session"][run],
                      lt["start"], lt["stop"], labels="qs",
                      n_components=n_components, n_perm=n_perm, seed=RANDOM_SEED)
    if res is None:
        print(f"  dPCA {session_key} {region}: too few lap-resolved cells, skipped")
        return None
    Z, evr, reg, pv = res["Z"], res["evr"], res["regularizer"], res["pvals"]
    sessions, kept_space, n_laps = res["groups"], res["space"], res["n_laps"]
    N = int(d["rates"].shape[1])

    out = processed_path(f"dpca_{session_key}_{region}_{bin_ms}ms.npz", DANDISET)
    np.savez_compressed(
        out,
        Z_q=Z["q"], Z_s=Z["s"], Z_qs=Z["qs"],
        evr_q=evr["q"], evr_s=evr["s"], evr_qs=evr["qs"],
        p_q=np.asarray(pv["q"]), p_qs=np.asarray(pv["qs"]),
        regularizer=np.asarray(reg), n_laps=np.asarray(n_laps),
        space_bins=kept_space.astype(np.int64), sessions=sessions.astype(np.int64),
        n_grid=np.asarray(N_GRID), n_units=np.asarray(N), subject=d["subject"],
        session_key=np.asarray(session_key), region=np.asarray(region),
        bin_size_s=d["bin_size_s"],
    )
    o = res["obs"]
    print(f"  dPCA {session_key} {region}: {kept_space.size} bins x "
          f"{len(sessions)} sessions, {n_laps} laps, reg={reg:.1e} -> var "
          f"space={o['s']:.2f} session={o['q']:.2f} (p={pv['q']:.3f}) "
          f"interact={o['qs']:.2f} (p={pv['qs']:.3f}) -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# GPFA — lap trials, latents indexed by linearized track position
# ----------------------------------------------------------------------------

def run_gpfa(nwb, session_key: str, region: str, x_dim: int = 6,  # noqa: ANN001
             gpfa_bin_ms: int = 100, min_lap_s: float = 6.0, em_max_iters: int = 50):
    """Fit GPFA with one trial per lap; index latents by linearized position.

    000978 is a single W across sessions, so one track graph (wells from all
    trials) is estimated per file and shared by every lap.
    """
    regions = ex.unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    trials_df = nwb.intervals["trials"].to_dataframe()
    lt = lb.lap_table(trials_df)
    trials, kept = lb.build_lap_spiketrains(spike_times, lt["start"], lt["stop"], min_lap_s)
    if len(trials) < 5 or unit_idx.size <= x_dim:
        print(f"  GPFA {session_key} {region}: too few laps/units, skipped")
        return None

    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    pos_t = np.asarray(ss.timestamps[:]); pos_xy = np.asarray(ss.data[:])[:, :2]
    wells = lz.wells_from_trials(trials_df, pos_t, pos_xy)
    if len(wells) < 3:
        print(f"  GPFA {session_key} {region}: could not build track graph, skipped")
        return None
    g, eo, sp, _ = lz.build_wtrack_graph(pos_xy, wells=wells)
    lin, _ = lz.linearize_position(pos_xy, g, eo, sp)
    fin = np.isfinite(lin)
    lin_edges = np.linspace(0.0, float(np.nanmax(lin[fin])), N_POSBINS + 1)

    def linpos_of_lap(bc, li):
        return np.interp(bc, pos_t[fin], lin[fin], left=np.nan, right=np.nan)

    lap_session = _lap_sessions(nwb, lt["start"])[kept]
    trajs = lb.fit_gpfa_laps(trials, x_dim=x_dim, gpfa_bin_ms=gpfa_bin_ms,
                             em_max_iters=em_max_iters)
    latents = lb.position_index_latents(trajs, kept, lt["start"], linpos_of_lap,
                                        gpfa_bin_ms, lin_edges)

    out = processed_path(f"gpfa_{session_key}_{region}.npz", DANDISET)
    np.savez_compressed(
        out,
        latents_posidx=latents,                    # (n_laps, x_dim, n_posbins)
        posbin_centers=(0.5 * (lin_edges[:-1] + lin_edges[1:])).astype(np.float32),
        lap_run_session=lap_session.astype(np.int16), lap_index=kept.astype(np.int64),
        trajectory_type=lt["trajectory_type"][kept],
        start_well=lt["start_well"][kept], end_well=lt["end_well"][kept],
        unit_ids=np.asarray(nwb.units.id[:])[unit_idx].astype(np.int64),
        x_dim=np.asarray(x_dim), gpfa_bin_ms=np.asarray(gpfa_bin_ms),
        min_lap_s=np.asarray(min_lap_s),
        session_key=np.asarray(session_key), region=np.asarray(region),
    )
    cov = float(np.mean(np.isfinite(latents).any(axis=1)))
    print(f"  GPFA {session_key} {region}: {latents.shape[0]} laps x {x_dim} dims x "
          f"{N_POSBINS} pos-bins ({cov*100:.0f}% cells covered) -> {out.name}")
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
    parser.add_argument("--gpfa-xdim", type=int, default=6)
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms, DANDISET)
                if r in args.regions]

    # PCA operates purely on saved run-epoch rate matrices
    if "pca" in args.method:
        print(f"PCA on {len(matrices)} run-epoch matrices (bin={args.bin_ms}ms)")
        for session_key, region in matrices:
            run_pca(session_key, region, args.bin_ms)

    # dPCA (needs the trials table for laps) and GPFA both re-stream the file
    if "dpca" in args.method or "gpfa" in args.method:
        k2a = _key_to_asset()
        keys = sorted({s for (s, r) in matrices})
        assets = args.session or [k2a[k] for k in keys if k in k2a]
        print(f"dPCA/GPFA on {len(assets)} file(s), regions={args.regions}")
        for asset in assets:
            print(asset)
            with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
                key = ex.session_key(asset)
                for region in args.regions:
                    if "dpca" in args.method:
                        run_dpca(nwb, key, region, args.bin_ms)
                    if "gpfa" in args.method:
                        run_gpfa(nwb, key, region, x_dim=args.gpfa_xdim)


if __name__ == "__main__":
    main()
