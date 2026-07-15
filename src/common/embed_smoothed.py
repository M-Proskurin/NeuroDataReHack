"""Smoothed fine-bin embeddings (50 ms, Gaussian sigma=100 ms, speed-filtered).

Re-runs the primary embeddings on the validated smoothed fine-bin path so the
headline figures use it instead of the 1000 ms hard bins. Loads awake data once
per dandiset (000447 from the saved 50 ms matrices; 000978 by streaming run
epochs), smooths per epoch with sigma=100 ms, keeps moving bins (speed > 4 cm/s),
and fits the requested methods.

Embeddings are saved as `emb_<method>_<key>_<region>_50ms.npz` (bin tag 50), so
stage 4 (000447) and 6b (000978) pick them up directly with `--bin-ms 50` — no
changes to those scripts. Each file carries `position` and the condition
(000447) or `run_session` (000978) labels the downstream comparisons need.

CEBRA note: speed-filtering breaks strict temporal contiguity, so CEBRA's
time-contrastive sampling is approximate here; CEBRA is the position-baked-in
reference anyway (the unsupervised UMAP embedding carries the honest signal).

Usage:
    pixi run python src/common/embed_smoothed.py --dandiset 000447 --method umap cebra
    pixi run python src/common/embed_smoothed.py --dandiset 000978 --method umap cebra
"""
from __future__ import annotations

import argparse

import numpy as np

import preprocessing as pp
from bin_smoothing_sensitivity import BIN_MS, SPEED, load_awake_447, load_awake_978
from config import processed_path

SIGMA_MS = 100
DIM = 3
CEBRA_ITERS = 2000


def _prep(entry):
    sm = pp.smooth_per_epoch(entry["counts"], entry["epoch"], SIGMA_MS / BIN_MS)
    m = pp.speed_mask(entry["velocity"], SPEED)
    return sm[m].astype(np.float32), entry["position"][m], entry["velocity"][m], entry["label"][m]


def _behavior(pos, vel):
    beh = np.column_stack([pos, vel]).astype(np.float32)
    idx = np.arange(beh.shape[0])
    for j in range(beh.shape[1]):
        col = beh[:, j]; bad = ~np.isfinite(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
        beh[:, j] = col
    return beh


def _fit(method, X, pos, vel):
    if method == "umap":
        import umap
        return umap.UMAP(n_components=DIM, n_neighbors=30, min_dist=0.1,
                         random_state=0).fit_transform(X), None
    if method == "cebra":
        from cebra import CEBRA
        model = CEBRA(model_architecture="offset10-model", batch_size=512,
                      learning_rate=3e-4, output_dimension=DIM,
                      max_iterations=CEBRA_ITERS, conditional="time_delta",
                      distance="cosine", device="cpu", verbose=False)
        model.fit(X, _behavior(pos, vel))
        return model.transform(X), np.asarray(model.state_dict_["loss"], np.float32)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dandiset", required=True, choices=["000447", "000978"])
    ap.add_argument("--method", nargs="+", default=["umap", "cebra"], choices=["umap", "cebra"])
    args = ap.parse_args()
    label_key = "condition" if args.dandiset == "000447" else "run_session"

    print(f"loading awake 50ms data for {args.dandiset} ...", flush=True)
    data = load_awake_447() if args.dandiset == "000447" else load_awake_978()

    for entry in data:
        X, pos, vel, label = _prep(entry)
        for method in args.method:
            emb, loss = _fit(method, X, pos, vel)
            out = processed_path(
                f"emb_{method}_{entry['key']}_{entry['region']}_{BIN_MS}ms.npz", args.dandiset)
            save = dict(embedding=emb.astype(np.float32), method=np.asarray(method),
                        position=pos.astype(np.float32), velocity=vel.astype(np.float32),
                        subject=np.asarray(entry["key"]), region=np.asarray(entry["region"]),
                        bin_size_s=np.asarray(BIN_MS / 1000.0),
                        smooth_ms=np.asarray(SIGMA_MS))
            save[label_key] = label
            if loss is not None:
                save["cebra_loss"] = loss
            np.savez_compressed(out, **save)
            print(f"  {method:6s} {entry['key']} {entry['region']}: {emb.shape} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
