"""Stage 6b-linearized (000978) — learning convergence in track-relative coords.

Same analysis as 06b_session_sequence.py (align each run session's manifold to
the final session, disparity vs. session number), but binning by **linearized
track position** (arm + fraction along it) instead of a raw 2-D grid. 000978 is a
single W across all sessions, so one track graph is estimated per file (wells
from the trials table); linearization just gives cleaner 1-D spatial bins than
the 2-D grid. ZT2 excluded (two separate days); 7 clean animals, both regions,
50 ms UMAP embeddings.

Output: data/processed/000978/stage6b_linearized_<method>.csv
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import procrustes

import download as dl
import linearize as lz
from config import RANDOM_SEED, REGIONS, processed_dir, processed_path

DANDISET = "000978"
BIN_MS = 50
K = 6                 # fractions per arm
N_SHUFFLE = 100
EPOCH_TABLE = "epoch intervals"
EXCLUDE = "ZT2"


def _subject_assets():
    return {p.split("/")[-1].replace("_behavior+ecephys.nwb", "").replace("sub-JDS-SingleDay-", ""): p
            for p in dl.list_asset_paths(dandiset_id=DANDISET) if p.endswith(".nwb")}


def _seg_dist_frac(P, A, B):
    v = B - A
    L2 = float(v @ v) + 1e-12
    t = np.clip((P - A) @ v / L2, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * v), axis=1), t


def arm_bins(P, nodes):
    """Assign each 2-D point to an arm (center/left/right) + fraction bin; base -> None."""
    segs = {
        "center": (nodes["center_junc"], nodes["center_well"]),
        "left": (nodes["left_junc"], nodes["left_well"]),
        "right": (nodes["right_junc"], nodes["right_well"]),
        "_baseL": (nodes["left_junc"], nodes["center_junc"]),
        "_baseR": (nodes["center_junc"], nodes["right_junc"]),
    }
    labels = list(segs)
    dists = np.stack([_seg_dist_frac(P, *segs[l])[0] for l in labels], axis=1)
    fracs = np.stack([_seg_dist_frac(P, *segs[l])[1] for l in labels], axis=1)
    nearest = dists.argmin(axis=1)
    out = []
    for i, ni in enumerate(nearest):
        lab = labels[ni]
        out.append(None if lab.startswith("_base") else (lab, int(min(fracs[i, ni], 0.999) * K)))
    return out


def _centroids(emb, keys):
    by = {}
    for e, k in zip(emb, keys):
        if k is not None:
            by.setdefault(k, []).append(e)
    return {k: np.mean(v, axis=0) for k, v in by.items() if len(v) >= 3}


def analyze_subject(key, asset, method, rng):
    with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
        ep = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
        tr = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
        ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
        pos_t = np.asarray(ss.timestamps[:]); pos_xy = np.asarray(ss.data[:])[:, :2]
    wells = lz.wells_from_trials(tr, pos_t, pos_xy)      # all run trials -> single W graph
    if len(wells) < 3:
        return []

    rows = []
    for region in REGIONS:
        f = processed_dir(DANDISET) / f"emb_{method}_{key}_{region}_{BIN_MS}ms.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=False)
        emb, pos, rs = d["embedding"], d["position"], d["run_session"]
        g, eo, sp, nodes = lz.build_wtrack_graph(pos, wells=wells)
        keys = arm_bins(pos, nodes)
        sessions = sorted(np.unique(rs).tolist())
        final = sessions[-1]
        cent_final = _centroids(emb[rs == final], [k for k, r in zip(keys, rs) if r == final])
        for s in sessions:
            cent_s = _centroids(emb[rs == s], [k for k, r in zip(keys, rs) if r == s])
            shared = sorted(set(cent_s) & set(cent_final))
            if len(shared) < 5:
                continue
            A = np.vstack([cent_s[k] for k in shared])
            B = np.vstack([cent_final[k] for k in shared])
            _, _, disp = procrustes(A, B)
            null = np.array([procrustes(A, B[rng.permutation(len(B))])[2] for _ in range(N_SHUFFLE)])
            rows.append(dict(session_key=key, region=region, session=int(s),
                             n_bins=len(shared), disparity=float(disp),
                             null_mean=float(null.mean())))
        print(f"  {key} {region}: {len(sessions)} sessions linearized", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="umap", choices=["umap", "cebra"])
    args = ap.parse_args()

    assets = _subject_assets()
    keys = sorted({f.stem.split("_")[2] for f in
                   processed_dir(DANDISET).glob(f"emb_{args.method}_*_{BIN_MS}ms.npz")
                   if not f.stem.split("_")[2].startswith(EXCLUDE)})
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    print(f"stage 6b linearized ({args.method}, {BIN_MS}ms): {len(keys)} animals")
    for k in keys:
        if k in assets:
            rows.extend(analyze_subject(k, assets[k], args.method, rng))

    df = pd.DataFrame(rows)
    out = processed_path(f"stage6b_linearized_{args.method}.csv", DANDISET)
    df.to_csv(out, index=False)
    per = df.groupby("session")["disparity"].mean()
    print("\ndisparity to final by session (linearized, track-relative):")
    for s, v in per.items():
        print(f"  session {s}: {v:.3f}")
    early = df[df.session <= 2].disparity.mean()
    late = df[df.session >= 6].disparity.mean()
    print(f"early(<=2) {early:.3f} vs late(>=6) {late:.3f} -> converges: {late < early}")
    print("compare grid-based 6b (50ms umap): 0.60 early -> 0.16 late")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
