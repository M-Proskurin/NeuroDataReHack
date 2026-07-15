"""Lap-resolved baselines: trial-by-trial dPCA + position-indexed GPFA.

Both W-track dandisets record *continuous* behaviour, so the stage-2 baselines
originally faked "trials" — dPCA collapsed each (space x condition) cell to a
single mean (so `regularizer=None`, no significance test), and GPFA chopped
epochs into arbitrary fixed-length windows (so latent trajectories had no
behavioural meaning). The trials table actually gives real trials: each row is
one **lap** (a run between reward wells, with a `trajectory_type`). Using laps as
trials fixes both:

  * **dPCA** — `build_lap_dpca_tensor` builds a genuine single-trial factorial
    tensor `trialX[lap, neuron, group, space]`, so we can run dPCA with
    `regularizer='auto'` (cross-validated) and `significance_analysis`
    (shuffle test) — turning the headline variance fractions into defensible,
    cross-validated numbers instead of point estimates on an over-fit mean.
  * **GPFA** — `build_lap_spiketrains` makes one trial per lap and
    `position_index_latents` re-expresses each lap's latent trajectory as a
    function of linearized track position, so trajectories are interpretable and
    averageable across laps / conditions / sessions at matched positions.

`group` is the task variable that is NOT space: novel/familiar condition for
000447, run-session index for 000978. dPCA's `significance_analysis` skips the
last label's pure marginalization, so callers pass labels with the group first
and space last (e.g. 'cs' / 'qs') to get significance on the group and the
group x space interaction (space alone is trivially significant).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# lap table
# ---------------------------------------------------------------------------

def lap_table(trials_df) -> dict:
    """Per-lap arrays from the NWB trials table (one row = one lap)."""
    n = len(trials_df)
    def col(name, default):
        return trials_df[name].to_numpy() if name in trials_df else np.full(n, default)
    return dict(
        start=trials_df["start_time"].to_numpy(float),
        stop=trials_df["stop_time"].to_numpy(float),
        trajectory_type=col("trajectory_type", -1),
        start_well=col("start_well", -1),
        end_well=col("end_well", -1),
    )


# ---------------------------------------------------------------------------
# dPCA: lap-resolved demixing with cross-validated regularization + permutation
# significance.
#
# We use the dPCA package only for the core closed-form fit with a *fixed*
# regularizer (the code path the original stage-2 already relied on). The
# package's own `regularizer='auto'` / `significance_analysis` assume a
# protected within-trial "time" axis and are buggy for a plain (group x space)
# design (they crash on protect=None and squeeze away a length-1 protected
# axis). Instead we select the regularizer by lap-level K-fold cross-validation
# and assess significance by permuting the group label across laps — both
# transparent and defensible.
# ---------------------------------------------------------------------------

def _lap_records(rates, time, space_label, group, lap_start, lap_stop, min_laps):
    """Per-lap {space_bin: mean-rate vector}, keeping bins with >= min_laps laps
    in every group. Returns (laps, groups, kept_space, N) or None."""
    rates = np.asarray(rates, float)
    N = rates.shape[1]
    groups = [g for g in np.unique(group) if str(g) != "-1"]
    if len(groups) < 2:
        return None
    gidx = {g: i for i, g in enumerate(groups)}

    raw = []                                   # (group_idx, {space: vec})
    per_bin = {gi: {} for gi in range(len(groups))}   # counts for the min_laps test
    for t0, t1 in zip(lap_start, lap_stop):
        m = (time >= t0) & (time < t1)
        if not m.any():
            continue
        gvals = set(group[m].tolist())
        if len(gvals) != 1:
            continue
        g = next(iter(gvals))
        if g not in gidx:
            continue
        sl, rl = space_label[m], rates[m]
        sv = {int(s): rl[sl == s].mean(axis=0) for s in np.unique(sl[sl >= 0])}
        if not sv:
            continue
        gi = gidx[g]
        raw.append((gi, sv))
        for s in sv:
            per_bin[gi][s] = per_bin[gi].get(s, 0) + 1

    all_space = sorted({s for gi in per_bin for s in per_bin[gi]})
    kept = [s for s in all_space
            if all(per_bin[gi].get(s, 0) >= min_laps for gi in range(len(groups)))]
    if len(kept) < 3:
        return None
    keptset = set(kept)
    laps = [(gi, {s: v for s, v in sv.items() if s in keptset})
            for gi, sv in raw]
    laps = [lp for lp in laps if lp[1]]
    return laps, np.asarray(groups), np.asarray(kept), N


def _mean_tensor(laps, group_assign, G, sidx, N):
    """Group x space mean-rate tensor X (N, G, S) and per-cell lap counts (G, S)."""
    S = len(sidx)
    sums = np.zeros((N, G, S))
    cnt = np.zeros((G, S))
    for (_, sv), gi in zip(laps, group_assign):
        for s, v in sv.items():
            j = sidx[s]
            sums[:, gi, j] += v
            cnt[gi, j] += 1
    X = np.zeros((N, G, S))
    nz = cnt > 0
    X[:, nz] = sums[:, nz] / cnt[nz]
    return X, cnt


def _center(X):
    N = X.shape[0]
    return X - X.reshape(N, -1).mean(axis=1).reshape((N,) + (1,) * (X.ndim - 1))


def _fit_dpca(Xc, labels, reg, k):
    """Closed-form dPCA fit with a fixed regularizer; returns (dpca, evr dict)."""
    from dPCA import dPCA
    dpca = dPCA.dPCA(labels=labels, regularizer=reg, n_components=k)
    dpca.protect = None
    Z = dpca.fit_transform(Xc)                 # no trialX -> no buggy CV path
    evr = {key: np.asarray(v, float) for key, v in dpca.explained_variance_ratio_.items()}
    return dpca, Z, evr


def _reconstruct(dpca, Xc):
    """Reconstruct centered data from all demixed components (sum over margs)."""
    Xr = Xc.reshape(Xc.shape[0], -1)
    out = np.zeros_like(Xr)
    for key in dpca.marginalizations:
        out += dpca.P[key] @ (dpca.D[key].T @ Xr)
    return out


def _cv_regularizer(laps, G, sidx, N, labels, k, lams, rng, n_folds=3):
    """Pick the regularizer minimising held-out reconstruction error over lap folds."""
    order = rng.permutation(len(laps))
    folds = np.array_split(order, n_folds)
    err = np.zeros(len(lams))
    for f in folds:
        test_i = set(int(i) for i in f)
        train = [laps[i] for i in range(len(laps)) if i not in test_i]
        test = [laps[i] for i in f]
        Xtr, ctr = _mean_tensor(train, [lp[0] for lp in train], G, sidx, N)
        Xte, cte = _mean_tensor(test, [lp[0] for lp in test], G, sidx, N)
        valid = ((ctr > 0) & (cte > 0)).ravel()
        if valid.sum() < 3:
            continue
        Xtrc, Xtec = _center(Xtr), _center(Xte)
        yte = Xtec.reshape(N, -1)[:, valid]
        for li, lam in enumerate(lams):
            dpca, _, _ = _fit_dpca(Xtrc, labels, lam, k)
            pred = _reconstruct(dpca, Xtec)[:, valid]
            err[li] += float(np.sum((yte - pred) ** 2))
    return float(lams[int(np.argmin(err))])


def lap_dpca(rates, time, space_label, group, lap_start, lap_stop, labels: str,
             min_laps: int = 3, n_components: int = 10, n_perm: int = 200,
             n_folds: int = 3, seed: int = 0):
    """Lap-resolved demixed PCA with CV-chosen regularization + permutation test.

    `labels` is 2 chars ordered (group, space), e.g. 'cs' (000447 condition) or
    'qs' (000978 run-session). Returns None if the design is too small, else a
    dict with per-marginalization variance fractions (`evr`, keys group/'s'/
    interaction), the chosen `regularizer`, permutation p-values for the group
    and interaction marginalizations (`pvals`), the demixed projections (`Z`),
    and design bookkeeping (`groups`, `space`, `n_laps`).
    """
    built = _lap_records(rates, time, space_label, group, lap_start, lap_stop, min_laps)
    if built is None:
        return None
    laps, groups, kept, N = built
    G, S = len(groups), len(kept)
    sidx = {int(s): j for j, s in enumerate(kept)}
    k = int(max(1, min(n_components, N - 1, G * S - 1)))
    rng = np.random.default_rng(seed)
    group_true = [lp[0] for lp in laps]
    grp_key, sp_key, int_key = labels[0], labels[1], labels

    lams = np.logspace(-4, 0, 9)
    reg = _cv_regularizer(laps, G, sidx, N, labels, k, lams, rng, n_folds)

    X, _ = _mean_tensor(laps, group_true, G, sidx, N)
    Xc = _center(X)
    dpca, Z, evr = _fit_dpca(Xc, labels, reg, k)
    obs = {key: float(np.sum(evr[key])) for key in (grp_key, sp_key, int_key)}

    ge = np.zeros(n_perm)
    ie = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(group_true)
        Xp, _ = _mean_tensor(laps, perm, G, sidx, N)
        _, _, evp = _fit_dpca(_center(Xp), labels, reg, k)
        ge[p] = np.sum(evp[grp_key])
        ie[p] = np.sum(evp[int_key])
    pvals = {grp_key: float((1 + np.sum(ge >= obs[grp_key])) / (1 + n_perm)),
             int_key: float((1 + np.sum(ie >= obs[int_key])) / (1 + n_perm))}

    return dict(Z={key: np.asarray(v, np.float32) for key, v in Z.items()},
                evr={key: np.asarray(v, np.float32) for key, v in evr.items()},
                regularizer=reg, pvals=pvals, obs=obs,
                groups=groups, space=kept, n_laps=len(laps))


# ---------------------------------------------------------------------------
# GPFA: lap trials + position indexing
# ---------------------------------------------------------------------------

def build_lap_spiketrains(spike_times, lap_start, lap_stop, min_lap_s: float = 6.0):
    """One GPFA trial per lap (variable length). Returns (trials, kept_lap_idx)."""
    import neo
    import quantities as pq

    trials, kept = [], []
    for li, (t0, t1) in enumerate(zip(lap_start, lap_stop)):
        dur = float(t1 - t0)
        if dur < min_lap_s:
            continue
        trials.append([neo.SpikeTrain((st[(st >= t0) & (st < t1)] - t0) * pq.s,
                                      t_start=0 * pq.s, t_stop=dur * pq.s)
                       for st in spike_times])
        kept.append(li)
    return trials, np.asarray(kept, dtype=int)


def fit_gpfa_laps(trials, x_dim: int = 6, gpfa_bin_ms: int = 100,
                  em_max_iters: int = 50):
    """Fit one GPFA model across all laps; return the list of latent trajectories."""
    import quantities as pq
    from elephant.gpfa import GPFA

    gpfa = GPFA(bin_size=gpfa_bin_ms * pq.ms, x_dim=x_dim, em_max_iters=em_max_iters)
    return gpfa.fit_transform(trials)                    # list of (x_dim, n_bins)


def position_index_latents(trajs, kept, lap_start, linpos_of_lap, gpfa_bin_ms: int,
                           lin_edges):
    """Re-express each lap's latent trajectory as a function of track position.

    `linpos_of_lap(bin_center_times, lap_idx)` returns the linearized position at
    each GPFA bin centre for that lap. Latents are averaged within each linear
    position bin. Returns latents (n_laps, x_dim, n_posbins), NaN where a lap did
    not visit a bin.
    """
    n_pos = len(lin_edges) - 1
    x_dim = trajs[0].shape[0]
    out = np.full((len(trajs), x_dim, n_pos), np.nan, dtype=np.float32)
    dt = gpfa_bin_ms / 1000.0
    for i, (traj, li) in enumerate(zip(trajs, kept)):
        bc = lap_start[li] + (np.arange(traj.shape[1]) + 0.5) * dt
        lp = np.asarray(linpos_of_lap(bc, int(li)))
        for b in range(n_pos):
            m = (lp >= lin_edges[b]) & (lp < lin_edges[b + 1])
            if m.any():
                out[i, :, b] = traj[:, m].mean(axis=1)
    return out
