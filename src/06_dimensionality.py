"""Stage 6 — dimensionality selection by cross-validated reconstruction.

A scree plot always "improves" with more components, so it can't tell you the
true dimensionality. Instead we use **bi-cross-validation** (Owen & Perry):
hold out a block of neurons AND a block of timepoints, fit PCA on the rest, and
predict the held-out block. Because the held-out neurons never touched the
latent and the held-out times never touched the loadings, over-fitting shows up
as *rising* error past the true dimensionality — so the error-vs-dim curve has a
genuine minimum.

For each held-out neuron set N2 and time set T2 (N1/T1 = the rest):
  1. PCA on X[T1, N1]              -> components V, mean m
  2. latent for test times:  Z2 = (X[T2, N1] - m) @ V[:k].T
  3. loadings for held-out neurons from train times: B = lstsq(Z1, X[T1, N2])
  4. predict X[T2, N2] = B applied to Z2; error = unexplained variance
Averaged over a K_n x K_t grid of folds, per subject / region / condition.

Output: data/processed/stage6_dimensionality_<bin>ms.csv

Usage:
    pixi run python src/06_dimensionality.py
    pixi run python src/06_dimensionality.py --bin-ms 1000 --max-dim 25
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from config import (
    BIN_SIZE_S,
    RANDOM_SEED,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
)

N_NEURON_FOLDS = 4
N_TIME_FOLDS = 4


def bicv_curve(X: np.ndarray, max_dim: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bi-cross-validated reconstruction error vs. #components for one matrix.

    Returns (dims, mean_error, sem_error) where error is unexplained variance
    fraction on held-out neuron x time blocks.
    """
    T, N = X.shape
    seed = int(rng.integers(1 << 31))
    nfold = KFold(N_NEURON_FOLDS, shuffle=True, random_state=seed)
    tfold = KFold(N_TIME_FOLDS, shuffle=True, random_state=seed + 1)

    # cap k so it is valid for every fold's train-neuron count
    min_train_neurons = N - int(np.ceil(N / N_NEURON_FOLDS))
    kmax = int(min(max_dim, min_train_neurons - 1))
    dims = np.arange(1, kmax + 1)

    per_fold = []  # each entry: array of error per k for one (N2,T2) fold
    for _, n2 in nfold.split(np.arange(N)):
        n1 = np.setdiff1d(np.arange(N), n2)
        for _, t2 in tfold.split(np.arange(T)):
            t1 = np.setdiff1d(np.arange(T), t2)
            pca = PCA(n_components=kmax).fit(X[np.ix_(t1, n1)])
            m1, V = pca.mean_, pca.components_          # V: (kmax, |n1|)
            Z1 = (X[np.ix_(t1, n1)] - m1) @ V.T          # (|t1|, kmax)
            Z2 = (X[np.ix_(t2, n1)] - m1) @ V.T          # (|t2|, kmax)
            m2 = X[np.ix_(t1, n2)].mean(axis=0)
            Y1 = X[np.ix_(t1, n2)] - m2                  # (|t1|, |n2|)
            Y2 = X[np.ix_(t2, n2)] - m2                  # (|t2|, |n2|)
            ss_tot = float((Y2 ** 2).sum()) + 1e-12
            errs = np.empty(kmax)
            for ki, k in enumerate(dims):
                B, *_ = np.linalg.lstsq(Z1[:, :k], Y1, rcond=None)  # (k, |n2|)
                pred = Z2[:, :k] @ B
                errs[ki] = float(((Y2 - pred) ** 2).sum()) / ss_tot
            per_fold.append(errs)

    per_fold = np.vstack(per_fold)
    return dims, per_fold.mean(axis=0), per_fold.std(axis=0) / np.sqrt(len(per_fold))


def analyze(subject: str, region: str, bin_ms: int, max_dim: int, rng) -> list[dict]:
    d = load_rate_matrix(subject, region, bin_ms)
    rates = d["rates"].astype(np.float64)
    condition = d["condition"]
    rows = []
    subsets = [("all", np.ones(len(condition), bool))]
    subsets += [(c, condition == c) for c in ("novel", "familiar")]
    for cond_name, mask in subsets:
        X = rates[mask]
        if X.shape[0] < 50:
            continue
        dims, err, sem = bicv_curve(X, max_dim, rng)
        opt = int(dims[np.argmin(err)])
        censored = bool(opt == dims[-1])   # minimum at the sweep edge => true dim >= opt
        for k, e, s in zip(dims, err, sem):
            rows.append(dict(subject=subject, region=region, condition=cond_name,
                             dim=int(k), cv_error=float(e), cv_sem=float(s),
                             optimal_dim=opt, kmax=int(dims[-1]), censored=censored))
        flag = "  [CENSORED: curve still falling at kmax]" if censored else ""
        print(f"  {subject} {region} {cond_name:8s}: optimal dim = {opt} "
              f"(min error {err.min():.3f}){flag}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        choices=list(REGIONS))
    parser.add_argument("--max-dim", type=int, default=25)
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms)
                if r in args.regions]
    print(f"stage 6: bi-CV dimensionality on {len(matrices)} matrices "
          f"(bin={args.bin_ms}ms, max_dim={args.max_dim})")
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for subject, region in matrices:
        rows.extend(analyze(subject, region, args.bin_ms, args.max_dim, rng))

    df = pd.DataFrame(rows)
    out = processed_path(f"stage6_dimensionality_{args.bin_ms}ms.csv")
    df.to_csv(out, index=False)
    print("\noptimal dim (median across subjects):")
    opt = df.drop_duplicates(["subject", "region", "condition"])
    print(opt.groupby(["region", "condition"])["optimal_dim"].median().to_string())
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
