"""Stage 5 (000978) — is the manifold low-dimensional but DRIFTING across sessions?

Two complementary measures on the smoothed 50 ms run-epoch data, per file/region
(ZT2 excluded by the loader):

  * **cross-session subspace generalization** — fit PCA (top `k`) on the train
    half of session i and measure the fraction of session j's (held-out, own-mean-
    centred) variance those PCs capture. The diagonal being high says each session
    is genuinely low-D; the off-diagonal falling off with |i-j| says the low-D
    subspace rotates/drifts across the day.
  * **cumulative dimensionality** — participation ratio of the pooled first-n
    sessions (session means kept, so translation drift counts) vs n. Rising from
    ~k (n=1) toward the pooled value means adding sessions adds dimensions — the
    drift signature. A label-shuffled null (bins reassigned to pseudo-sessions)
    stays flat/high, confirming the rise is real session structure.

Read together with the per-session TwoNN/PR (flat, low — stage5_dim_by_session.csv)
and 6b (the drift is directional, converging to the final session).

Outputs:
    data/processed/000978/stage5_drift_heatmap.csv      (session_key, region, i, j, ve)
    data/processed/000978/stage5_drift_cumulative.csv   (session_key, region, n, pr, kind)

Usage:
    pixi run python src/000978/06_dim_drift.py
    pixi run python src/000978/06_dim_drift.py --k 3
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from bin_smoothing_sensitivity import _smoothed_masked, load_awake_978
from config import RANDOM_SEED, processed_path
from dimensionality import twonn

DANDISET = "000978"
SIGMA_MS = 100
TWONN_CAP = 5000


def _pr(X):
    """Participation ratio (Σλ)²/Σλ² of column-centred X."""
    ev = PCA().fit(X - X.mean(axis=0)).explained_variance_
    return float(ev.sum() ** 2 / np.sum(ev ** 2))


def _twonn(X, rng):
    """TwoNN intrinsic dim on a bounded subsample (size-comparable, fast)."""
    if len(X) > TWONN_CAP:
        X = X[rng.choice(len(X), TWONN_CAP, replace=False)]
    return twonn(X)


def _subspace_ve(V, Y):
    """Fraction of Y's variance captured by the orthonormal subspace rows of V."""
    Yc = Y - Y.mean(axis=0)
    return float(np.sum((Yc @ V.T) ** 2) / (np.sum(Yc ** 2) + 1e-12))


def analyze(entry, k, rng):
    X, _pos, sess = _smoothed_masked(entry, SIGMA_MS)
    sessions = sorted(np.unique(sess[sess > 0]).tolist())
    if len(sessions) < 3:
        return [], []
    key, region = entry["key"], entry["region"]

    # train/test half per session (random split) + top-k PCs from each train half
    tr, te, V = {}, {}, {}
    for s in sessions:
        idx = np.where(sess == s)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        tr[s], te[s] = X[idx[:h]], X[idx[h:]]
        kk = int(min(k, tr[s].shape[0] - 1, tr[s].shape[1]))
        V[s] = PCA(n_components=kk).fit(tr[s] - tr[s].mean(axis=0)).components_

    heat = [dict(session_key=key, region=region, i=int(i), j=int(j),
                 ve=_subspace_ve(V[i], te[j]))
            for i in sessions for j in sessions]

    cum = []
    for kind, lab in [("real", sess), ("shuffled", rng.permutation(sess))]:
        for n in range(1, len(sessions) + 1):
            sel = np.isin(lab, sessions[:n])
            cum.append(dict(session_key=key, region=region, n_sessions=n,
                            twonn=_twonn(X[sel], rng), pr=_pr(X[sel]), kind=kind))
    real = [c for c in cum if c["kind"] == "real"]
    print(f"  {key} {region}: {len(sessions)} sessions, k={V[sessions[0]].shape[0]}; "
          f"diag VE {np.mean([_subspace_ve(V[s], te[s]) for s in sessions]):.2f}, "
          f"cum TwoNN {real[0]['twonn']:.1f}→{real[-1]['twonn']:.1f}", flush=True)
    return heat, cum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3, help="PCA subspace dim for the heatmap")
    args = ap.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"stage 5 drift {DANDISET} (50 ms, sigma={SIGMA_MS} ms, k={args.k})")
    heat_rows, cum_rows = [], []
    for e in load_awake_978():
        h, c = analyze(e, args.k, rng)
        heat_rows.extend(h); cum_rows.extend(c)

    pd.DataFrame(heat_rows).to_csv(processed_path("stage5_drift_heatmap.csv", DANDISET), index=False)
    pd.DataFrame(cum_rows).to_csv(processed_path("stage5_drift_cumulative.csv", DANDISET), index=False)
    cd = pd.DataFrame(cum_rows)
    print("\ncumulative TwoNN (mean across animals, real):")
    print(cd[cd.kind == "real"].groupby(["region", "n_sessions"]).twonn.mean().round(1).to_string())
    print(f"-> stage5_drift_heatmap.csv ({len(heat_rows)}), stage5_drift_cumulative.csv ({len(cum_rows)})")


if __name__ == "__main__":
    main()
