"""Stage 4 — cross-condition and cross-region comparison.

Quantify how similar two population geometries are, using Procrustes disparity
and CCA on **position-binned centroids**: for each shared 2-D spatial bin we take
the centroid of the embedding points in that bin, giving one matched point per
location. Aligning these matched point-sets is the standard way to compare
spatial maps that live in arbitrary embedding coordinate frames.

Two comparisons (per subject, per embedding method):
  * **novel vs. familiar** within a region — the geometric transformation
  * **CA1 vs. PFC** within a condition (and pooled) — cross-region relationship

For each, we report:
  * Procrustes disparity in [0, 1] (0 = identical shape after best
    translation/scaling/rotation/reflection)
  * a shuffled null (permute the bin correspondence) with an empirical p-value,
    so a low disparity is only meaningful if it beats the null
  * mean CCA canonical correlation

Open question tracked: alignment is done **per animal** here; a pooled variant
would concatenate matched bins across animals before aligning.

Output: data/processed/stage4_alignment_<method>_<bin>ms.csv

Usage:
    pixi run python src/04_cross_condition.py                 # cebra, all subjects
    pixi run python src/04_cross_condition.py --method umap isomap
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA

from config import (
    BIN_SIZE_S,
    DATA_PROCESSED,
    RANDOM_SEED,
    REGIONS,
    processed_path,
    spatial_grid_labels,
)

N_GRID = 8
MIN_COUNT = 5
N_SHUFFLE = 200


def _load_embedding(method: str, subject: str, region: str, bin_ms: int) -> dict | None:
    f = DATA_PROCESSED / f"emb_{method}_{subject}_{region}_{bin_ms}ms.npz"
    return dict(np.load(f, allow_pickle=False)) if f.exists() else None


def _shared_bins(labels: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Grid bins sampled >= MIN_COUNT times under every mask."""
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
    p = float((np.sum(null <= disp) + 1) / (N_SHUFFLE + 1))
    z = float((disp - null.mean()) / (null.std() + 1e-12))
    return {"procrustes_disparity": float(disp), "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p_value": p, "z": z}


def _cca_mean_r(A: np.ndarray, B: np.ndarray) -> float:
    k = min(A.shape[1], A.shape[0] - 1)
    if k < 1:
        return np.nan
    cca = CCA(n_components=k, max_iter=1000)
    Ac, Bc = cca.fit_transform(A, B)
    rs = [np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(k)]
    return float(np.nanmean(rs))


def _compare(A: np.ndarray, B: np.ndarray, rng, **meta) -> dict | None:
    if A.shape[0] < A.shape[1] + 2:          # need more matched bins than dims
        return None
    row = dict(meta, n_bins=A.shape[0])
    row.update(_procrustes_null(A, B, rng))
    row["cca_mean_r"] = _cca_mean_r(A, B)
    return row


def analyze_subject(method: str, subject: str, bin_ms: int, rng) -> list[dict]:
    rows = []
    embs = {r: _load_embedding(method, subject, r, bin_ms) for r in REGIONS}

    # --- novel vs familiar, within each region ---
    for region, d in embs.items():
        if d is None:
            continue
        labels = spatial_grid_labels(d["position"], N_GRID)
        cond = d["condition"]
        novel, familiar = cond == "novel", cond == "familiar"
        bins = _shared_bins(labels, [novel, familiar])
        A = _centroids(d["embedding"], labels, novel, bins)
        B = _centroids(d["embedding"], labels, familiar, bins)
        row = _compare(A, B, rng, comparison="novel_vs_familiar", method=method,
                       subject=subject, group=region)
        if row:
            rows.append(row)

    # --- CA1 vs PFC, within each condition and pooled ---
    ca1, pfc = embs.get("CA1"), embs.get("PFC")
    if ca1 is not None and pfc is not None:
        labels = spatial_grid_labels(ca1["position"], N_GRID)  # shared behavior
        cond = ca1["condition"]
        for name, mask in [("novel", cond == "novel"),
                           ("familiar", cond == "familiar"),
                           ("all", np.ones(len(cond), bool))]:
            bins = _shared_bins(labels, [mask])
            A = _centroids(ca1["embedding"], labels, mask, bins)
            B = _centroids(pfc["embedding"], labels, mask, bins)
            row = _compare(A, B, rng, comparison="CA1_vs_PFC", method=method,
                           subject=subject, group=name)
            if row:
                rows.append(row)
    return rows


def _subjects(method: str, bin_ms: int) -> list[str]:
    subs = set()
    for f in DATA_PROCESSED.glob(f"emb_{method}_*_{bin_ms}ms.npz"):
        subs.add(f.stem.split("_")[2])
    return sorted(subs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=["cebra"],
                        choices=["cebra", "umap", "isomap"])
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    args = parser.parse_args()

    for method in args.method:
        rng = np.random.default_rng(RANDOM_SEED)
        rows = []
        for subject in _subjects(method, args.bin_ms):
            rows.extend(analyze_subject(method, subject, args.bin_ms, rng))
        if not rows:
            print(f"{method}: no embeddings found (run stage 3 first)")
            continue
        df = pd.DataFrame(rows)
        out = processed_path(f"stage4_alignment_{method}_{args.bin_ms}ms.csv")
        df.to_csv(out, index=False)

        print(f"\n=== {method} (bin={args.bin_ms}ms) ===")
        for comp in df["comparison"].unique():
            sub = df[df["comparison"] == comp]
            print(f"{comp}: disparity {sub['procrustes_disparity'].mean():.3f} "
                  f"(null {sub['null_mean'].mean():.3f}), "
                  f"CCA r {sub['cca_mean_r'].mean():.3f}, "
                  f"n={len(sub)} comparisons")
        print(f"-> {out.name}")


if __name__ == "__main__":
    main()
