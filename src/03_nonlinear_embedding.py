"""Stage 3 — nonlinear embeddings.

Run per subject and per region (CA1/PFC never pooled) on the stage-1 rate
matrices. Each embedding is saved with the epoch/condition/behavior metadata so
stage 4 (Procrustes/CCA) and stage 5 (topology) can pick it up.

  * **CEBRA** (primary, behavior-aligned) — trained with the animal's
    position+velocity as the auxiliary continuous variable, so the embedding is
    shaped by behavior rather than by novelty. We deliberately do NOT feed the
    novel/familiar label: colouring the behavior-aligned embedding by condition
    afterwards is how we read out the geometric transformation.
  * **UMAP** (umap-learn) and **Isomap** (scikit-learn) — unsupervised
    comparisons on the same rate matrix.
  * PHATE is listed as optional in CLAUDE.md and is not installed; add `phate`
    to the env if you want it as a further comparison.

Usage:
    pixi run python src/03_nonlinear_embedding.py                    # all methods, all matrices
    pixi run python src/03_nonlinear_embedding.py --method cebra
    pixi run python src/03_nonlinear_embedding.py --dim 3 --bin-ms 1000
"""
from __future__ import annotations

import argparse

import numpy as np

from config import (
    BIN_SIZE_S,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
)


def _behavior(d: dict) -> np.ndarray:
    """Continuous behavior label = [x, y, speed], NaNs filled by interpolation."""
    beh = np.column_stack([d["position"], d["velocity"]]).astype(np.float32)
    # CEBRA can't take NaNs; fill per-column by linear interp over bin index
    idx = np.arange(beh.shape[0])
    for j in range(beh.shape[1]):
        col = beh[:, j]
        bad = ~np.isfinite(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
        beh[:, j] = col
    return beh


def _save(method: str, subject: str, region: str, bin_ms: int,
          embedding: np.ndarray, d: dict, **extra):
    out = processed_path(f"emb_{method}_{subject}_{region}_{bin_ms}ms.npz")
    np.savez_compressed(
        out,
        embedding=embedding.astype(np.float32),
        method=np.asarray(method),
        time=d["time"], epoch=d["epoch"], condition=d["condition"],
        position=d["position"], velocity=d["velocity"],
        subject=np.asarray(subject), region=np.asarray(region),
        bin_size_s=d["bin_size_s"], **extra,
    )
    print(f"  {method:6s} {subject} {region}: {embedding.shape} -> {out.name}")
    return out


def run_cebra(subject: str, region: str, bin_ms: int, dim: int = 3,
              max_iterations: int = 5000):
    from cebra import CEBRA

    d = load_rate_matrix(subject, region, bin_ms)
    rates = d["rates"].astype(np.float32)
    beh = _behavior(d)
    model = CEBRA(
        model_architecture="offset10-model", batch_size=512,
        learning_rate=3e-4, output_dimension=dim, max_iterations=max_iterations,
        conditional="time_delta", distance="cosine", device="cpu", verbose=False,
    )
    model.fit(rates, beh)                       # behavior-aligned (position+speed)
    embedding = model.transform(rates)
    loss = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    return _save("cebra", subject, region, bin_ms, embedding, d, cebra_loss=loss)


def run_umap(subject: str, region: str, bin_ms: int, dim: int = 3,
             n_neighbors: int = 30, min_dist: float = 0.1):
    import umap

    d = load_rate_matrix(subject, region, bin_ms)
    reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=0)
    embedding = reducer.fit_transform(d["rates"].astype(np.float32))
    return _save("umap", subject, region, bin_ms, embedding, d)


def run_isomap(subject: str, region: str, bin_ms: int, dim: int = 3,
               n_neighbors: int = 30):
    from sklearn.manifold import Isomap

    d = load_rate_matrix(subject, region, bin_ms)
    embedding = Isomap(n_components=dim, n_neighbors=n_neighbors).fit_transform(
        d["rates"].astype(np.float64))
    return _save("isomap", subject, region, bin_ms, embedding, d)


RUNNERS = {"cebra": run_cebra, "umap": run_umap, "isomap": run_isomap}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=list(RUNNERS),
                        choices=list(RUNNERS))
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        choices=list(REGIONS))
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--subject", action="append",
                        help="restrict to these subject ids (repeatable)")
    parser.add_argument("--cebra-iters", type=int, default=5000)
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms)
                if r in args.regions and (not args.subject or s in args.subject)]
    print(f"stage 3: methods={args.method} dim={args.dim} on {len(matrices)} "
          f"matrices (bin={args.bin_ms}ms)")
    for subject, region in matrices:
        for method in args.method:
            if method == "cebra":
                run_cebra(subject, region, args.bin_ms, dim=args.dim,
                          max_iterations=args.cebra_iters)
            else:
                RUNNERS[method](subject, region, args.bin_ms, dim=args.dim)


if __name__ == "__main__":
    main()
