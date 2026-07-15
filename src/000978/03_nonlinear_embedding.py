"""Stage 3 (000978) — nonlinear embeddings.

Run per session file and per region (CA1/PFC never pooled) on the **run** epochs
(the awake W-track manifold). Each embedding carries the run_session index so the
learning-trajectory analysis (6b) can track how the manifold drifts across the
day; sleep epochs are embedded/projected separately in 6c.

  * **CEBRA** (primary, behavior-aligned) — trained with position+velocity as the
    continuous auxiliary variable (NOT the session label), so colouring the
    embedding by run_session afterwards reveals learning-related drift.
  * **UMAP** / **Isomap** — unsupervised comparisons on the same rate matrix.

Usage:
    pixi run python src/000978/03_nonlinear_embedding.py                 # all methods, all matrices
    pixi run python src/000978/03_nonlinear_embedding.py --method umap isomap
    pixi run python src/000978/03_nonlinear_embedding.py --method cebra --cebra-iters 2000
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np

from config import (
    BIN_SIZE_S,
    REGIONS,
    available_rate_matrices,
    load_rate_matrix,
    processed_path,
)

DANDISET = "000978"


def _run_mask(d: dict) -> np.ndarray:
    return d["kind"] == "run"


def _behavior(d: dict, m: np.ndarray) -> np.ndarray:
    """Continuous behavior label = [x, y, speed] on run bins, NaNs interpolated."""
    beh = np.column_stack([d["position"][m], d["velocity"][m]]).astype(np.float32)
    idx = np.arange(beh.shape[0])
    for j in range(beh.shape[1]):
        col = beh[:, j]
        bad = ~np.isfinite(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
        beh[:, j] = col
    return beh


def _save(method: str, session_key: str, region: str, bin_ms: int,
          embedding: np.ndarray, d: dict, m: np.ndarray, **extra):
    out = processed_path(f"emb_{method}_{session_key}_{region}_{bin_ms}ms.npz", DANDISET)
    np.savez_compressed(
        out,
        embedding=embedding.astype(np.float32), method=np.asarray(method),
        time=d["time"][m], epoch=d["epoch"][m], run_session=d["run_session"][m],
        position=d["position"][m], velocity=d["velocity"][m],
        subject=d["subject"], session_key=np.asarray(session_key),
        region=np.asarray(region), bin_size_s=d["bin_size_s"], **extra,
    )
    print(f"  {method:6s} {session_key} {region}: {embedding.shape} -> {out.name}")
    return out


def run_cebra(session_key: str, region: str, bin_ms: int, dim: int = 3,
              max_iterations: int = 2000):
    from cebra import CEBRA

    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    m = _run_mask(d)
    rates = d["rates"][m].astype(np.float32)
    model = CEBRA(
        model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
        output_dimension=dim, max_iterations=max_iterations,
        conditional="time_delta", distance="cosine", device="cpu", verbose=False,
    )
    model.fit(rates, _behavior(d, m))
    embedding = model.transform(rates)
    loss = np.asarray(model.state_dict_["loss"], dtype=np.float32)
    return _save("cebra", session_key, region, bin_ms, embedding, d, m, cebra_loss=loss)


def run_umap(session_key: str, region: str, bin_ms: int, dim: int = 3,
             n_neighbors: int = 30, min_dist: float = 0.1):
    import umap

    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    m = _run_mask(d)
    reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=0)
    embedding = reducer.fit_transform(d["rates"][m].astype(np.float32))
    return _save("umap", session_key, region, bin_ms, embedding, d, m)


def run_isomap(session_key: str, region: str, bin_ms: int, dim: int = 3,
               n_neighbors: int = 30):
    from sklearn.manifold import Isomap

    d = load_rate_matrix(session_key, region, bin_ms, DANDISET)
    m = _run_mask(d)
    embedding = Isomap(n_components=dim, n_neighbors=n_neighbors).fit_transform(
        d["rates"][m].astype(np.float64))
    return _save("isomap", session_key, region, bin_ms, embedding, d, m)


RUNNERS = {"cebra": run_cebra, "umap": run_umap, "isomap": run_isomap}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    parser.add_argument("--bin-ms", type=int, default=int(BIN_SIZE_S * 1000))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--session", action="append", help="restrict to these session keys")
    parser.add_argument("--cebra-iters", type=int, default=2000)
    args = parser.parse_args()

    matrices = [(s, r) for (s, r, _) in available_rate_matrices(args.bin_ms, DANDISET)
                if r in args.regions and (not args.session or s in args.session)]
    print(f"000978 stage 3: methods={args.method} dim={args.dim} on {len(matrices)} "
          f"run-epoch matrices (bin={args.bin_ms}ms)")
    for session_key, region in matrices:
        for method in args.method:
            if method == "cebra":
                run_cebra(session_key, region, args.bin_ms, dim=args.dim,
                          max_iterations=args.cebra_iters)
            else:
                RUNNERS[method](session_key, region, args.bin_ms, dim=args.dim)


if __name__ == "__main__":
    main()
