"""Stage 4b (000447) — novel vs. familiar comparison in LINEARIZED coordinates.

The novel and familiar W-tracks are physically different mazes in different
orientations, so the stage-4 raw-2-D grid comparison is confounded (its "shared
bins" are wherever two different layouts overlap in camera coords). Here we
instead compare the population geometry at matched **track-relative** positions:

  * per condition, estimate the W graph (wells from the trials table) and assign
    each embedding sample to an arm (center / outer) with a fraction along it;
  * arms are matched across conditions by **well id** (identity), binned into K
    fractions; the base/choice region is dropped (ambiguous to match);
  * Procrustes disparity + shuffled null and CCA on the matched arm-map centroids.

Uses the smoothed 50 ms UMAP embeddings. Compare the disparity here to the raw
2-D stage-4 value (~0.90, ~chance) — linearization should reveal more shared
structure if the maps are genuinely related at matched track positions.

Output: data/processed/000447/stage4b_linearized_<method>.csv
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

import download as dl
import linearize as lz
from config import RANDOM_SEED, REGIONS, processed_dir, processed_path

DANDISET = "000447"
BIN_MS = 50
K = 6                 # fractions per arm
N_SHUFFLE = 100
EPOCH_TABLE = "epoch intervals"


def _subject_assets():
    return {p.split("/")[0].replace("sub-", ""): p
            for p in dl.list_asset_paths(dandiset_id=DANDISET) if p.endswith(".nwb")}


def _seg_dist_frac(P, A, B):
    """Distance of each point to segment AB and fractional position t in [0,1]."""
    v = B - A
    L2 = float(v @ v) + 1e-12
    t = np.clip((P - A) @ v / L2, 0.0, 1.0)
    proj = A + t[:, None] * v
    return np.linalg.norm(P - proj, axis=1), t


def arm_keys(P, nodes, arm_wellid):
    """Assign each 2-D point to an arm (by well id) + fraction bin; base -> None."""
    segs = {  # label -> (A, B) with A=junction, B=well (frac 0..1 = base..well)
        ("arm", arm_wellid["left"]): (nodes["left_junc"], nodes["left_well"]),
        ("arm", arm_wellid["center"]): (nodes["center_junc"], nodes["center_well"]),
        ("arm", arm_wellid["right"]): (nodes["right_junc"], nodes["right_well"]),
        ("base", 0): (nodes["left_junc"], nodes["center_junc"]),
        ("base", 1): (nodes["center_junc"], nodes["right_junc"]),
    }
    labels = list(segs)
    dists = np.stack([_seg_dist_frac(P, *segs[l])[0] for l in labels], axis=1)
    fracs = np.stack([_seg_dist_frac(P, *segs[l])[1] for l in labels], axis=1)
    nearest = dists.argmin(axis=1)
    keys = []
    for i, ni in enumerate(nearest):
        lab = labels[ni]
        if lab[0] == "base":
            keys.append(None)                       # drop base/choice points
        else:
            keys.append((lab[1], int(min(fracs[i, ni], 0.999) * K)))   # (well_id, frac bin)
    return keys


def condition_centroids(emb, pos, nodes, arm_wellid):
    keys = arm_keys(pos, nodes, arm_wellid)
    by = {}
    for e, k in zip(emb, keys):
        if k is not None:
            by.setdefault(k, []).append(e)
    return {k: np.mean(v, axis=0) for k, v in by.items() if len(v) >= 3}


def _nearest_wellid(coord, wells, ids):
    return int(ids[np.argmin(np.linalg.norm(wells - coord, axis=1))])


def analyze_subject(subject, asset, method, rng):
    with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
        ep = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
        tr = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
        ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
        pos_t = np.asarray(ss.timestamps[:]); pos_xy = np.asarray(ss.data[:])[:, :2]
        desc = nwb.session_description
    words = [w.lower().strip(".,") for w in desc.split()]
    conds = [w for w in words if w in ("novel", "familiar")][:len(ep)]
    trs = tr["start_time"].to_numpy()

    def cond_wells(cond):
        idx = [i for i, c in enumerate(conds) if c == cond]
        mask = np.zeros(len(tr), bool)
        for i in idx:
            mask |= (trs >= ep.iloc[i].start_time) & (trs < ep.iloc[i].stop_time)
        return lz.wells_from_trials(tr[mask], pos_t, pos_xy, return_ids=True)

    rows = []
    for region in REGIONS:
        f = processed_dir(DANDISET) / f"emb_{method}_{subject}_{region}_{BIN_MS}ms.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=False)
        emb, pos, cond = d["embedding"], d["position"], d["condition"]
        cents = {}
        for c in ("novel", "familiar"):
            wells, ids = cond_wells(c)
            if len(wells) < 3:
                break
            m = cond == c
            g, eo, sp, nodes = lz.build_wtrack_graph(pos[m], wells=wells)
            arm_wellid = {"left": _nearest_wellid(nodes["left_well"], wells, ids),
                          "center": _nearest_wellid(nodes["center_well"], wells, ids),
                          "right": _nearest_wellid(nodes["right_well"], wells, ids)}
            cents[c] = condition_centroids(emb[m], pos[m], nodes, arm_wellid)
        if len(cents) < 2:
            continue
        shared = sorted(set(cents["novel"]) & set(cents["familiar"]))
        if len(shared) < 5:
            continue
        A = np.vstack([cents["novel"][k] for k in shared])
        B = np.vstack([cents["familiar"][k] for k in shared])
        _, _, disp = procrustes(A, B)
        null = np.array([procrustes(A, B[rng.permutation(len(B))])[2] for _ in range(N_SHUFFLE)])
        kk = min(A.shape[1], A.shape[0] - 1)
        Ac, Bc = CCA(n_components=kk, max_iter=1000).fit_transform(A, B)
        cca = float(np.nanmean([np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(kk)]))
        rows.append(dict(subject=subject, region=region, n_matched_bins=len(shared),
                         procrustes_disparity=float(disp), null_mean=float(null.mean()),
                         p_value=float((np.sum(null <= disp) + 1) / (N_SHUFFLE + 1)),
                         cca_mean_r=cca))
        print(f"  {subject} {region}: {len(shared)} matched arm-bins, "
              f"disparity {disp:.3f} (null {null.mean():.3f}), cca {cca:.3f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="umap", choices=["umap", "cebra", "cebratime"])
    args = ap.parse_args()

    assets = _subject_assets()
    subjects = sorted({f.stem.split("_")[2] for f in
                       processed_dir(DANDISET).glob(f"emb_{args.method}_*_{BIN_MS}ms.npz")})
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    print(f"stage 4b linearized novel-vs-familiar ({args.method}, {BIN_MS}ms): {len(subjects)} subjects")
    for s in subjects:
        if s in assets:
            rows.extend(analyze_subject(s, assets[s], args.method, rng))

    df = pd.DataFrame(rows)
    out = processed_path(f"stage4b_linearized_{args.method}.csv", DANDISET)
    df.to_csv(out, index=False)
    print(f"\nnovel vs familiar (LINEARIZED, arm-matched): "
          f"disparity {df.procrustes_disparity.mean():.3f} "
          f"(null {df.null_mean.mean():.3f}), cca {df.cca_mean_r.mean():.3f}, "
          f"{(df.p_value < 0.05).mean()*100:.0f}% sig, n={len(df)}")
    print("compare raw-2D stage 4 (50ms umap): disparity ~0.90 (null ~0.97), 50% sig")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
