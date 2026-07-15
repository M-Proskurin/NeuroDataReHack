"""Stage 6b (000978) — session-sequence alignment (learning trajectory).

For each clean single-day animal and region, align every run session's manifold
to the **final** session and track alignment quality as a function of session
number. A convergence time course (disparity falling toward the last session)
is the geometric signature of the map stabilizing as the animal learns.

Method: as in 000447 stage 4, compare **position-binned centroids** — for each
run session, the centroid of the embedding points in each shared 2-D spatial
bin — so we compare spatial maps rather than raw point clouds. For every session
s we report, relative to the final session:
  * Procrustes disparity (0 = identical) + a shuffled null / empirical p-value
  * mean CCA canonical correlation

**Read CEBRA with care** (same caveat as 000447 6a): CEBRA is behavior-aligned,
so it maps matched positions to matched coordinates by construction and will look
artificially converged. The **unsupervised** embeddings (UMAP/Isomap) give the
honest learning signal.

**ZT2 is excluded** — its two files are separate recording days, not one
8-session day (see CLAUDE.md). We drop session keys starting with 'ZT2'.

Output: data/processed/000978/stage6b_session_sequence_<method>_<bin>ms.csv

Usage:
    pixi run python src/000978/06b_session_sequence.py                 # all methods
    pixi run python src/000978/06b_session_sequence.py --method umap isomap
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA

from config import (
    BIN_SIZE_S,
    RANDOM_SEED,
    REGIONS,
    processed_dir,
    processed_path,
    spatial_grid_labels,
)

DANDISET = "000978"
N_GRID = 6
MIN_COUNT = 5
N_SHUFFLE = 200
EXCLUDE = "ZT2"          # two separate days; excluded from the learning trajectory


def _load_emb(method: str, key: str, region: str, bin_ms: int) -> dict | None:
    f = processed_dir(DANDISET) / f"emb_{method}_{key}_{region}_{bin_ms}ms.npz"
    return dict(np.load(f, allow_pickle=False)) if f.exists() else None


def _shared_bins(labels: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    ok = None
    for m in masks:
        cnt = np.bincount(labels[m & (labels >= 0)], minlength=N_GRID * N_GRID)
        good = cnt >= MIN_COUNT
        ok = good if ok is None else (ok & good)
    return np.flatnonzero(ok)


def _centroids(emb: np.ndarray, labels: np.ndarray, mask: np.ndarray,
               bins: np.ndarray) -> np.ndarray:
    return np.vstack([emb[mask & (labels == b)].mean(axis=0) for b in bins])


def _procrustes_null(A: np.ndarray, B: np.ndarray, rng) -> dict:
    _, _, disp = procrustes(A, B)
    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        _, _, null[i] = procrustes(A, B[rng.permutation(len(B))])
    return {"procrustes_disparity": float(disp), "null_mean": float(null.mean()),
            "p_value": float((np.sum(null <= disp) + 1) / (N_SHUFFLE + 1))}


def _cca_mean_r(A: np.ndarray, B: np.ndarray) -> float:
    k = min(A.shape[1], A.shape[0] - 1)
    if k < 1:
        return np.nan
    Ac, Bc = CCA(n_components=k, max_iter=1000).fit_transform(A, B)
    return float(np.nanmean([np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(k)]))


def analyze(method: str, key: str, region: str, bin_ms: int, rng) -> list[dict]:
    d = _load_emb(method, key, region, bin_ms)
    if d is None:
        return []
    emb, rs = d["embedding"], d["run_session"]
    labels = spatial_grid_labels(d["position"], N_GRID)
    sessions = sorted(np.unique(rs).tolist())
    final = sessions[-1]
    rows = []
    for s in sessions:
        bins = _shared_bins(labels, [rs == s, rs == final])
        A = _centroids(emb, labels, rs == s, bins)
        B = _centroids(emb, labels, rs == final, bins)
        if A.shape[0] < A.shape[1] + 2:
            continue
        row = dict(method=method, session_key=key, region=region, session=int(s),
                   sessions_from_end=int(final - s), n_bins=int(A.shape[0]))
        row.update(_procrustes_null(A, B, rng))
        row["cca_mean_r"] = _cca_mean_r(A, B)
        rows.append(row)
    return rows


def _clean_keys(method: str, bin_ms: int) -> list[str]:
    keys = {f.stem.split("_")[2] for f in
            processed_dir(DANDISET).glob(f"emb_{method}_*_{bin_ms}ms.npz")}
    return sorted(k for k in keys if not k.startswith(EXCLUDE))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=["cebra", "umap", "isomap"],
                        choices=["cebra", "umap", "isomap"])
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    args = parser.parse_args()

    for method in args.method:
        rng = np.random.default_rng(RANDOM_SEED)
        keys = _clean_keys(method, args.bin_ms)
        rows = []
        for key in keys:
            for region in REGIONS:
                rows.extend(analyze(method, key, region, args.bin_ms, rng))
        if not rows:
            print(f"{method}: no embeddings (run stage 3 first)")
            continue
        df = pd.DataFrame(rows)
        out = processed_path(f"stage6b_session_sequence_{method}_{args.bin_ms}ms.csv", DANDISET)
        df.to_csv(out, index=False)

        # convergence: mean disparity-to-final by session, pooled over animals+regions
        print(f"\n=== {method} — {len(keys)} animals, disparity to final session ===")
        summ = df.groupby("session")["procrustes_disparity"].mean()
        for s, v in summ.items():
            print(f"  session {s}: mean disparity {v:.3f}")
        early = df[df.session <= 2]["procrustes_disparity"].mean()
        late = df[df.session >= 6]["procrustes_disparity"].mean()
        print(f"  early (s<=2) {early:.3f} vs late (s>=6) {late:.3f} "
              f"-> {'converges' if late < early else 'no convergence'}")
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
