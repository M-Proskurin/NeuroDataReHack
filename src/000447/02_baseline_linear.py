"""Stage 2 — linear baselines: PCA, GPFA, dPCA.

Run per subject and per region (CA1/PFC never pooled) on the stage-1 rate
matrices. Each method writes its own artifact to data/processed/ carrying the
epoch/condition/subject/region metadata so later stages stay rerunnable.

  * **PCA** (scikit-learn) — variance structure of the population. Fast; reads
    the saved rate matrix. Saves scores, components, explained-variance ratio.

  * **dPCA** (Kobak et al.) — *demixed* PCA that splits variance into parts that
    depend on space, on condition (novel/familiar), and their interaction. This
    is the "factor out position/novelty variance explicitly" step. Each **lap**
    (a trials-table row) is a trial, so we build a genuine single-trial factorial
    tensor trialX[lap, neuron, condition, space-bin] over a 2-D spatial grid
    (bins visited by >=3 laps under BOTH conditions). That lets us fit dPCA with
    cross-validated regularization (`regularizer='auto'`) and run a shuffle
    `significance_analysis` — so the reported variance fractions are defensible,
    not point estimates on an over-fit mean.

  * **GPFA** (elephant) — Gaussian-process factor analysis on the raw spike
    trains (re-streamed from DANDI), giving smooth latent trajectories. Each lap
    is one trial; the resulting latents are then re-expressed as a function of
    **linearized track position** so trajectories are interpretable and
    averageable across laps/conditions at matched positions.

Usage:
    pixi run python src/000447/02_baseline_linear.py                 # all methods, all sessions
    pixi run python src/000447/02_baseline_linear.py --method pca dpca
    pixi run python src/000447/02_baseline_linear.py --method gpfa --session <asset-path>
    pixi run python src/000447/02_baseline_linear.py --bin-ms 1000
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse
from importlib import import_module

import numpy as np
from sklearn.decomposition import PCA

from config import (
    BIN_SIZE_S,
    RANDOM_SEED,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
    spatial_grid_labels,
)

import download as dl              # stage-0 streaming helper from src/common
import lap_baselines as lb         # lap-resolved dPCA + GPFA helpers
import linearize as lz             # W-track linearization for GPFA position index

EPOCH_TABLE = "epoch intervals"
N_GRID = 6                         # dPCA spatial grid (per side)
N_POSBINS = 30                     # GPFA linear-position bins

# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------

def run_pca(subject: str, region: str, bin_ms: int, n_keep: int = 20) -> "Path":  # noqa: F821
    """Fit PCA on one rate matrix; save scores, components, explained variance."""
    d = load_rate_matrix(subject, region, bin_ms)
    rates = d["rates"].astype(np.float64)          # (T, n_units), sqrt counts
    n_comp = min(rates.shape[1], rates.shape[0])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(rates)              # (T, n_comp), mean-centered inside
    evr = pca.explained_variance_ratio_
    k = min(n_keep, n_comp)

    out = processed_path(f"pca_{subject}_{region}_{bin_ms}ms.npz")
    np.savez_compressed(
        out,
        scores=scores[:, :k].astype(np.float32),
        components=pca.components_[:k].astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        # metadata carried through
        time=d["time"], epoch=d["epoch"], condition=d["condition"],
        position=d["position"], velocity=d["velocity"], unit_ids=d["unit_ids"],
        subject=np.asarray(subject), region=np.asarray(region),
        bin_size_s=d["bin_size_s"],
    )
    cum = np.cumsum(evr)
    d80 = int(np.searchsorted(cum, 0.80) + 1)
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"  PCA  {subject} {region}: {rates.shape[1]} units -> "
          f"{d80} PCs for 80% / {d90} for 90% var -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# dPCA — lap-resolved (space x condition), cross-validated + significance
# ----------------------------------------------------------------------------

def run_dpca(nwb, subject: str, region: str, bin_ms: int,  # noqa: ANN001
             n_perm: int = 200, n_components: int = 10) -> "Path | None":  # noqa: F821
    """Demixed PCA over (condition x space) with laps as trials.

    Each lap gives one single-trial estimate per space bin; the regularizer is
    chosen by lap-level cross-validation and the condition / condition x space
    variance fractions are tested by permuting the condition label across laps.
    labels='cs' -> marginalizations 'c' (condition), 's' (space), 'cs'
    (interaction = map reshaping between novel and familiar).
    """
    d = load_rate_matrix(subject, region, bin_ms)
    space = spatial_grid_labels(d["position"], N_GRID)
    lt = lb.lap_table(nwb.intervals["trials"].to_dataframe())
    res = lb.lap_dpca(d["rates"], d["time"], space, d["condition"],
                      lt["start"], lt["stop"], labels="cs",
                      n_components=n_components, n_perm=n_perm, seed=RANDOM_SEED)
    if res is None:
        print(f"  dPCA {subject} {region}: too few lap-resolved cells, skipped")
        return None
    Z, evr, reg, pv = res["Z"], res["evr"], res["regularizer"], res["pvals"]
    groups, kept_space, n_laps = res["groups"], res["space"], res["n_laps"]
    N = int(d["rates"].shape[1])

    out = processed_path(f"dpca_{subject}_{region}_{bin_ms}ms.npz")
    np.savez_compressed(
        out,
        Z_c=Z["c"], Z_s=Z["s"], Z_cs=Z["cs"],
        evr_c=evr["c"], evr_s=evr["s"], evr_cs=evr["cs"],
        p_c=np.asarray(pv["c"]), p_cs=np.asarray(pv["cs"]),
        regularizer=np.asarray(reg), n_laps=np.asarray(n_laps),
        space_bins=kept_space.astype(np.int64), conditions=np.asarray(groups),
        n_grid=np.asarray(N_GRID), n_units=np.asarray(N),
        subject=np.asarray(subject), region=np.asarray(region),
        bin_size_s=d["bin_size_s"],
    )
    o = res["obs"]
    print(f"  dPCA {subject} {region}: {kept_space.size} bins x {len(groups)} cond, "
          f"{n_laps} laps, reg={reg:.1e} -> var space={o['s']:.2f} "
          f"cond={o['c']:.2f} (p={pv['c']:.3f}) interact={o['cs']:.2f} "
          f"(p={pv['cs']:.3f}) -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# GPFA — lap trials, latents indexed by linearized track position
# ----------------------------------------------------------------------------

def _lin_series_by_condition(nwb, lt):  # noqa: ANN001
    """Per-condition linearized position over the SpatialSeries clock.

    Novel and familiar are physically different mazes, so each condition gets its
    own W graph (wells from that condition's laps); linear position is comparable
    across them because build_wtrack_graph uses a fixed edge order/spacing.
    Returns (pos_t, {condition: linear_position}, lap_condition_fn).
    """
    ex = import_module("01_extraction")
    epochs = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    conds = ex.epoch_conditions(nwb.session_description, len(epochs))
    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    pos_t = np.asarray(ss.timestamps[:]); pos_xy = np.asarray(ss.data[:])[:, :2]
    trials_df = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
    lap_start = lt["start"]

    def lap_condition(t0: float):
        for e, row in epochs.iterrows():
            if row.start_time <= t0 < row.stop_time:
                return conds[e]
        return None

    lin_series = {}
    for c in sorted(set(conds)):
        ep_c = epochs[[conds[e] == c for e in range(len(epochs))]]
        in_c = np.zeros(len(trials_df), bool)
        for _, row in ep_c.iterrows():
            in_c |= (trials_df.start_time.to_numpy() >= row.start_time) & \
                    (trials_df.start_time.to_numpy() < row.stop_time)
        wells = lz.wells_from_trials(trials_df[in_c], pos_t, pos_xy)
        if len(wells) < 3:
            continue
        pos_in_c = np.zeros(len(pos_t), bool)
        for _, row in ep_c.iterrows():
            pos_in_c |= (pos_t >= row.start_time) & (pos_t < row.stop_time)
        g, eo, sp, _ = lz.build_wtrack_graph(pos_xy[pos_in_c], wells=wells)
        lin, _ = lz.linearize_position(pos_xy, g, eo, sp)
        lin_series[c] = lin
    return pos_t, lin_series, lap_condition


def run_gpfa(nwb, subject: str, region: str, x_dim: int = 6,
             gpfa_bin_ms: int = 100, min_lap_s: float = 6.0,
             em_max_iters: int = 50) -> "Path | None":  # noqa: F821
    """Fit GPFA with one trial per lap; index latents by linearized position."""
    ex = import_module("01_extraction")
    regions = ex.unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    lt = lb.lap_table(nwb.intervals["trials"].to_dataframe())
    trials, kept = lb.build_lap_spiketrains(spike_times, lt["start"], lt["stop"], min_lap_s)
    if len(trials) < 5 or unit_idx.size <= x_dim:
        print(f"  GPFA {subject} {region}: too few laps/units, skipped")
        return None

    pos_t, lin_series, lap_condition = _lin_series_by_condition(nwb, lt)
    lap_cond = np.array([lap_condition(lt["start"][i]) for i in kept], dtype="<U8")

    def linpos_of_lap(bc, li):
        s = lin_series.get(lap_condition(lt["start"][li]))
        if s is None:
            return np.full(len(bc), np.nan)
        fin = np.isfinite(s)
        return np.interp(bc, pos_t[fin], s[fin], left=np.nan, right=np.nan)

    all_lin = np.concatenate([s[np.isfinite(s)] for s in lin_series.values()])
    lin_edges = np.linspace(0.0, float(np.nanmax(all_lin)), N_POSBINS + 1)

    trajs = lb.fit_gpfa_laps(trials, x_dim=x_dim, gpfa_bin_ms=gpfa_bin_ms,
                             em_max_iters=em_max_iters)
    latents = lb.position_index_latents(trajs, kept, lt["start"], linpos_of_lap,
                                        gpfa_bin_ms, lin_edges)

    out = processed_path(f"gpfa_{subject}_{region}.npz")
    np.savez_compressed(
        out,
        latents_posidx=latents,                    # (n_laps, x_dim, n_posbins)
        posbin_centers=(0.5 * (lin_edges[:-1] + lin_edges[1:])).astype(np.float32),
        lap_condition=lap_cond, lap_index=kept.astype(np.int64),
        trajectory_type=lt["trajectory_type"][kept],
        start_well=lt["start_well"][kept], end_well=lt["end_well"][kept],
        unit_ids=np.asarray(nwb.units.id[:])[unit_idx].astype(np.int64),
        x_dim=np.asarray(x_dim), gpfa_bin_ms=np.asarray(gpfa_bin_ms),
        min_lap_s=np.asarray(min_lap_s),
        subject=np.asarray(subject), region=np.asarray(region),
    )
    cov = float(np.mean(np.isfinite(latents).any(axis=1)))
    print(f"  GPFA {subject} {region}: {latents.shape[0]} laps x {x_dim} dims x "
          f"{N_POSBINS} pos-bins ({cov*100:.0f}% cells covered) -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------

def _subject_to_asset() -> dict:
    """Map subject id (e.g. 'JDS-NFN-AM2') to its DANDI asset path."""
    out = {}
    for p in dl.list_asset_paths():
        if p.endswith(".nwb"):
            out[p.split("/")[0].replace("sub-", "")] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=["pca", "dpca", "gpfa"],
                        choices=["pca", "dpca", "gpfa"])
    parser.add_argument("--session", action="append", metavar="ASSET_PATH",
                        help="restrict to these asset paths (repeatable)")
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000),
                        help="rate-matrix bin size for PCA/dPCA")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        choices=list(REGIONS))
    parser.add_argument("--gpfa-xdim", type=int, default=6)
    args = parser.parse_args()

    matrices = [(s, r, p) for (s, r, p) in available_rate_matrices(args.bin_ms)
                if r in args.regions]

    # PCA operates purely on saved rate matrices
    if "pca" in args.method:
        print(f"PCA on {len(matrices)} rate matrices (bin={args.bin_ms}ms)")
        for subject, region, _ in matrices:
            run_pca(subject, region, args.bin_ms)

    # dPCA (needs the trials table for laps) and GPFA both re-stream the session
    if "dpca" in args.method or "gpfa" in args.method:
        s2a = _subject_to_asset()
        subjects = sorted({s for (s, r, _) in matrices})
        assets = args.session or [s2a[s] for s in subjects if s in s2a]
        print(f"dPCA/GPFA on {len(assets)} session(s), regions={args.regions}")
        for asset in assets:
            print(asset)
            with dl.stream_nwb(asset) as nwb:
                subject = getattr(nwb.subject, "subject_id", "unknown")
                for region in args.regions:
                    if "dpca" in args.method:
                        run_dpca(nwb, subject, region, args.bin_ms)
                    if "gpfa" in args.method:
                        run_gpfa(nwb, subject, region, x_dim=args.gpfa_xdim)


if __name__ == "__main__":
    main()
