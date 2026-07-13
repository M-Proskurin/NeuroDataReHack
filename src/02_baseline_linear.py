"""Stage 2 — linear baselines: PCA, GPFA, dPCA.

Run per subject and per region (CA1/PFC never pooled) on the stage-1 rate
matrices. Each method writes its own artifact to data/processed/ carrying the
epoch/condition/subject/region metadata so later stages stay rerunnable.

  * **PCA** (scikit-learn) — variance structure of the population. Fast; reads
    the saved rate matrix. Saves scores, components, explained-variance ratio.

  * **dPCA** (Kobak et al.) — *demixed* PCA that splits variance into parts that
    depend on space, on condition (novel/familiar), and their interaction. This
    is the "factor out position/novelty variance explicitly" step. Continuous
    W-track data has no trial structure, so we build the required factorial
    tensor X[neuron, space-bin, condition] by averaging firing within a 2-D
    spatial grid, keeping only grid bins visited under BOTH conditions.

  * **GPFA** (elephant) — Gaussian-process factor analysis on the raw spike
    trains (re-streamed from DANDI), giving smooth latent trajectories. Needs
    trials, so each behavioral epoch is chopped into fixed-length segments.

Usage:
    pixi run python src/02_baseline_linear.py                 # all methods, all sessions
    pixi run python src/02_baseline_linear.py --method pca dpca
    pixi run python src/02_baseline_linear.py --method gpfa --session <asset-path>
    pixi run python src/02_baseline_linear.py --bin-ms 1000
"""
from __future__ import annotations

import argparse
from importlib import import_module

import numpy as np
from sklearn.decomposition import PCA

from config import (
    BIN_SIZE_S,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
)

dl = import_module("00_download")  # stage-0 streaming helper (name starts with a digit)

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
# dPCA
# ----------------------------------------------------------------------------

def _spatial_labels(position: np.ndarray, n_grid: int) -> np.ndarray:
    """Integer 2-D-grid bin per timepoint (-1 where position is undefined)."""
    x, y = position[:, 0], position[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    lab = np.full(x.shape, -1, dtype=int)
    if valid.sum() == 0:
        return lab
    (x0, x1) = np.percentile(x[valid], [0.5, 99.5])
    (y0, y1) = np.percentile(y[valid], [0.5, 99.5])
    xi = np.clip(((x[valid] - x0) / max(x1 - x0, 1e-9) * n_grid).astype(int), 0, n_grid - 1)
    yi = np.clip(((y[valid] - y0) / max(y1 - y0, 1e-9) * n_grid).astype(int), 0, n_grid - 1)
    lab[valid] = xi * n_grid + yi
    return lab


def run_dpca(subject: str, region: str, bin_ms: int, n_grid: int = 8,
             min_count: int = 5, n_components: int = 10) -> "Path | None":  # noqa: F821
    """Demixed PCA over (space x condition); save per-marginalization variance."""
    from dPCA import dPCA

    d = load_rate_matrix(subject, region, bin_ms)
    rates = d["rates"].astype(np.float64)          # (T, N)
    condition = d["condition"]
    conds = sorted(np.unique(condition).tolist())  # e.g. ['familiar', 'novel']
    if len(conds) < 2:
        print(f"  dPCA {subject} {region}: <2 conditions, skipped")
        return None

    labels = _spatial_labels(d["position"], n_grid)
    # keep spatial bins sampled enough under EVERY condition (factorial design)
    ok_bins = None
    for c in conds:
        cnt = np.bincount(labels[(condition == c) & (labels >= 0)],
                          minlength=n_grid * n_grid)
        good = cnt >= min_count
        ok_bins = good if ok_bins is None else (ok_bins & good)
    space = np.flatnonzero(ok_bins)
    if space.size < 3:
        print(f"  dPCA {subject} {region}: only {space.size} shared space bins, skipped")
        return None

    N = rates.shape[1]
    X = np.zeros((N, space.size, len(conds)), dtype=np.float64)
    for ci, c in enumerate(conds):
        for si, s in enumerate(space):
            m = (condition == c) & (labels == s)
            X[:, si, ci] = rates[m].mean(axis=0)
    X -= X.mean(axis=(1, 2), keepdims=True)        # center per neuron

    k = int(min(n_components, N, X.shape[1] * X.shape[2] - 1))
    dpca = dPCA.dPCA(labels="sc", n_components=k, regularizer=None)
    dpca.protect = None
    Z = dpca.fit_transform(X)
    evr = dpca.explained_variance_ratio_

    out = processed_path(f"dpca_{subject}_{region}_{bin_ms}ms.npz")
    np.savez_compressed(
        out,
        Z_s=Z["s"].astype(np.float32), Z_c=Z["c"].astype(np.float32),
        Z_sc=Z["sc"].astype(np.float32),
        evr_s=np.asarray(evr["s"], dtype=np.float32),
        evr_c=np.asarray(evr["c"], dtype=np.float32),
        evr_sc=np.asarray(evr["sc"], dtype=np.float32),
        space_bins=space.astype(np.int64), conditions=np.asarray(conds),
        n_grid=np.asarray(n_grid), n_units=np.asarray(N),
        subject=np.asarray(subject), region=np.asarray(region),
        bin_size_s=d["bin_size_s"],
    )
    frac = {m: float(np.sum(evr[m])) for m in ("s", "c", "sc")}
    print(f"  dPCA {subject} {region}: {N} units, {space.size} space bins x "
          f"{len(conds)} cond -> var space={frac['s']:.2f} "
          f"cond={frac['c']:.2f} interact={frac['sc']:.2f} -> {out.name}")
    return out


# ----------------------------------------------------------------------------
# GPFA
# ----------------------------------------------------------------------------

def _epoch_conditions(nwb, n_epochs):  # noqa: ANN001
    ex = import_module("01_extraction")
    return ex.epoch_conditions(nwb.session_description, n_epochs)


def _gpfa_trials(nwb, region: str, trial_len_s: float):  # noqa: ANN001
    """Chop each epoch into fixed-length trials of neo.SpikeTrains for one region."""
    import neo
    import quantities as pq

    ex = import_module("01_extraction")
    regions = ex.unit_regions(nwb)
    unit_idx = np.flatnonzero(regions == region)
    spike_times = [np.asarray(nwb.units["spike_times"][int(i)]) for i in unit_idx]

    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    conds = _epoch_conditions(nwb, len(epochs))

    trials, tr_epoch, tr_cond = [], [], []
    for e, row in epochs.iterrows():
        s, t = float(row["start_time"]), float(row["stop_time"])
        n = int(np.floor((t - s) / trial_len_s))
        for k in range(n):
            w0 = s + k * trial_len_s
            trial = []
            for st in spike_times:
                seg = st[(st >= w0) & (st < w0 + trial_len_s)] - w0
                trial.append(neo.SpikeTrain(seg * pq.s, t_start=0 * pq.s,
                                            t_stop=trial_len_s * pq.s))
            trials.append(trial)
            tr_epoch.append(e)
            tr_cond.append(conds[e])
    return trials, np.asarray(tr_epoch, dtype=np.int16), np.asarray(tr_cond), unit_idx


def run_gpfa(nwb, subject: str, region: str, x_dim: int = 8,
             gpfa_bin_ms: int = 100, trial_len_s: float = 5.0,
             em_max_iters: int = 50) -> "Path | None":  # noqa: F821
    """Fit GPFA on re-streamed spike trains; save latent trajectories per trial."""
    import quantities as pq
    from elephant.gpfa import GPFA

    trials, tr_epoch, tr_cond, unit_idx = _gpfa_trials(nwb, region, trial_len_s)
    if len(trials) < 5 or unit_idx.size <= x_dim:
        print(f"  GPFA {subject} {region}: too few trials/units, skipped")
        return None

    gpfa = GPFA(bin_size=gpfa_bin_ms * pq.ms, x_dim=x_dim, em_max_iters=em_max_iters)
    trajs = gpfa.fit_transform(trials)             # list of (x_dim, n_bins)
    latents = np.stack(trajs, axis=0).astype(np.float32)  # (n_trials, x_dim, n_bins)

    out = processed_path(f"gpfa_{subject}_{region}.npz")
    np.savez_compressed(
        out,
        latents=latents, trial_epoch=tr_epoch, trial_condition=tr_cond,
        unit_ids=np.asarray(nwb.units.id[:])[unit_idx].astype(np.int64),
        x_dim=np.asarray(x_dim), gpfa_bin_ms=np.asarray(gpfa_bin_ms),
        trial_len_s=np.asarray(trial_len_s),
        subject=np.asarray(subject), region=np.asarray(region),
    )
    print(f"  GPFA {subject} {region}: {latents.shape[0]} trials x {x_dim} dims "
          f"x {latents.shape[2]} bins -> {out.name}")
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
    parser.add_argument("--gpfa-xdim", type=int, default=8)
    args = parser.parse_args()

    # PCA / dPCA operate on saved rate matrices
    matrices = [(s, r, p) for (s, r, p) in available_rate_matrices(args.bin_ms)
                if r in args.regions]
    if "pca" in args.method or "dpca" in args.method:
        print(f"PCA/dPCA on {len(matrices)} rate matrices (bin={args.bin_ms}ms)")
        for subject, region, _ in matrices:
            if "pca" in args.method:
                run_pca(subject, region, args.bin_ms)
            if "dpca" in args.method:
                run_dpca(subject, region, args.bin_ms)

    # GPFA re-streams spike trains per session
    if "gpfa" in args.method:
        s2a = _subject_to_asset()
        subjects = sorted({s for (s, r, _) in matrices})
        assets = args.session or [s2a[s] for s in subjects if s in s2a]
        print(f"GPFA on {len(assets)} session(s), regions={args.regions}, "
              f"x_dim={args.gpfa_xdim}")
        for asset in assets:
            print(asset)
            with dl.stream_nwb(asset) as nwb:
                subject = getattr(nwb.subject, "subject_id", "unknown")
                for region in args.regions:
                    run_gpfa(nwb, subject, region, x_dim=args.gpfa_xdim)


if __name__ == "__main__":
    main()
