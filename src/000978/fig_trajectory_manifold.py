"""Figure — track trajectory vs. 3-D UMAP manifold for 000978 (last run session).

One two-panel PNG per animal, using that animal's **last** run session (end of the
learning day). Colour modes (`--color-by`):

  * ``position``  — continuous LINEARIZED track position;
  * ``trajtype``  — discrete 6-way arm × direction (paper-style): which arm, and
    into it (toward the well) vs out of it (toward the base).

000978 is a single W across the day, so one track graph per file. ZT2 is excluded
(two separate days; not in the 50 ms embeddings). The 50 ms embeddings are
streamed/fine-binned (no saved rate matrix), so for `trajtype` we reproduce the
same run-epoch fine-binning here to recover per-bin position/time/velocity and
attach the arm/direction labels (verified to align with the embeddings row-for-row).

Usage:
    pixi run python src/000978/fig_trajectory_manifold.py                 # all animals
    pixi run python src/000978/fig_trajectory_manifold.py --color-by trajtype
    pixi run python src/000978/fig_trajectory_manifold.py --key JS14
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse
from importlib import import_module

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import download as dl
import linearize as lz
import trajectory_labels as tl
from config import REPO_ROOT, processed_dir

DANDISET = "000978"
BIN_MS = 50
FINE_S = BIN_MS / 1000.0
CMAP = "turbo"
ex = import_module("01_extraction")


def _key_assets():
    return {ex.session_key(p): p for p in dl.list_asset_paths(dandiset_id=DANDISET)
            if p.endswith(".nwb")}


def _behavior(nwb):
    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    return np.asarray(ss.timestamps[:]), np.asarray(ss.data[:])   # pos_t, (N,3): x,y,speed


def build_track_graph(nwb):
    """Single W-track graph for the file; returns (graph, eo, sp, nodes)."""
    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    trials = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
    kinds, _ = ex.classify_epochs(epochs, trials)
    pos_t, pos = _behavior(nwb)
    run = np.zeros(len(pos_t), bool)
    for e, row in epochs.iterrows():
        if kinds[e] == "run":
            run |= (pos_t >= row.start_time) & (pos_t < row.stop_time)
    wells = lz.wells_from_trials(trials, pos_t, pos[:, :2])
    if len(wells) < 3:
        raise RuntimeError("could not find 3 wells")
    return lz.build_wtrack_graph(pos[run, :2], wells=wells)


def run_epoch_behavior_bins(nwb):
    """Reproduce load_awake_978's run-epoch 50 ms binning (behaviour only, region-independent).

    Returns per-bin (position, time, velocity, run_session) in the exact order the
    embeddings were built from, so a speed mask realigns to them.
    """
    epochs = nwb.intervals[ex.EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    trials = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
    kinds, run_session = ex.classify_epochs(epochs, trials)
    pos_t, pos = _behavior(nwb)
    Pb, Tb, Vb, Sb = [], [], [], []
    for e, row in epochs.iterrows():
        if kinds[e] != "run":
            continue
        s, t = float(row["start_time"]), float(row["stop_time"])
        n = int(np.floor((t - s) / FINE_S))
        if n < 1:
            continue
        centers = (s + FINE_S * np.arange(n + 1))[:-1] + FINE_S / 2
        Pb.append(np.column_stack([np.interp(centers, pos_t, pos[:, 0], left=np.nan, right=np.nan),
                                   np.interp(centers, pos_t, pos[:, 1], left=np.nan, right=np.nan)]))
        Vb.append(np.interp(centers, pos_t, pos[:, 2], left=np.nan, right=np.nan))
        Tb.append(centers); Sb.append(np.full(n, run_session[e], np.int16))
    return np.vstack(Pb), np.concatenate(Tb), np.concatenate(Vb), np.concatenate(Sb)


METHOD_LABEL = {"umap": "UMAP", "cebra": "CEBRA (supervised)", "cebratime": "CEBRA-Time"}
METHOD_AXIS = {"umap": "UMAP", "cebra": "CEBRA", "cebratime": "CEBRA-Time"}


def make_figure(key, asset, region, color_by, method="umap"):
    f = processed_dir(DANDISET) / f"emb_{method}_{key}_{region}_{BIN_MS}ms.npz"
    if not f.exists():
        print(f"  {key} {region}: no {method} embedding, skipped"); return None
    d = np.load(f, allow_pickle=False)
    emb, pos, rs = d["embedding"], d["position"], d["run_session"]
    last = int(rs.max())
    m = rs == last

    with dl.stream_nwb(asset, dandiset_id=DANDISET) as nwb:
        graph, eo, sp, nodes = build_track_graph(nwb)
        if color_by == "position":
            lin, _ = lz.linearize_position(pos[m], graph, eo, sp)
            payload = ("position", lin)
        else:
            Pb, Tb, Vb, Sb = run_epoch_behavior_bins(nwb)
            labels_masked, frac_masked, mmask = tl.masked_labels(Pb, Tb, Vb, nodes)
            keep = Sb[mmask] == last
            lab_last, frac_last = labels_masked[keep], frac_masked[keep]
            assert len(lab_last) == int(m.sum()), f"align {len(lab_last)} vs {int(m.sum())}"
            payload = ("trajtype", lab_last, frac_last)

    ok = np.isfinite(pos[m]).all(axis=1)
    P, E = pos[m][ok], emb[m][ok]

    fig = plt.figure(figsize=(13, 5.6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    if payload[0] == "position":
        L = payload[1][ok]; order = np.argsort(L)
        sc0 = ax0.scatter(P[order, 0], P[order, 1], c=L[order], cmap=CMAP, s=6, alpha=0.85)
        ax1.scatter(E[order, 0], E[order, 1], E[order, 2], c=L[order], cmap=CMAP, s=6, alpha=0.7)
        cb = fig.colorbar(sc0, ax=[ax0, ax1], shrink=0.7, pad=0.02)
        cb.set_label("linearized track position (cm)")
        sub = "shared colour = linearized position"
        tag = ""
    else:
        labels, frac = payload[1][ok], payload[2][ok]
        colors = tl.point_colors(labels, frac)
        base = labels == "base"
        ax0.scatter(P[base, 0], P[base, 1], color=tl.BASE_COLOR, s=6, alpha=0.3)
        ax0.scatter(P[~base, 0], P[~base, 1], c=colors[~base], s=6, alpha=0.9)
        ax1.scatter(E[base, 0], E[base, 1], E[base, 2], color=tl.BASE_COLOR, s=6, alpha=0.25)
        ax1.scatter(E[~base, 0], E[~base, 1], E[~base, 2], c=colors[~base], s=6, alpha=0.8)
        ax0.legend(handles=tl.legend_handles(), fontsize=8, loc="upper right",
                   title="arm · direction\n(dark=base → bright=well)")
        sub = "arm × direction, shaded by location (in = toward well, out = toward base)"
        tag = "_arms"
    ax0.set_aspect("equal"); ax0.set_xlabel("x (cm)"); ax0.set_ylabel("y (cm)")
    ax0.set_title(f"Track trajectory — {key} {region} (session {last})")
    axl = METHOD_AXIS[method]
    ax1.set_xlabel(f"{axl} 1"); ax1.set_ylabel(f"{axl} 2"); ax1.set_zlabel(f"{axl} 3")
    ax1.set_title(f"{region} activity manifold ({METHOD_LABEL[method]} 3-D, {BIN_MS} ms)")
    fig.suptitle(f"000978 {key} — last session: track ↔ neural-manifold ({sub})", fontsize=12)

    outdir = REPO_ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"traj_manifold_000978_{region}_{key}_sess{last}_{method}_{BIN_MS}ms{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  {key} {region} [{color_by}]: session {last}, {int(m.sum())} samples "
          f"({int(ok.sum())} plotted) -> {out.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="single session key (e.g. JS14); default: all animals")
    ap.add_argument("--region", default="CA1", choices=["CA1", "PFC"])
    ap.add_argument("--method", default="umap", choices=["umap", "cebra", "cebratime"])
    ap.add_argument("--color-by", default="position", choices=["position", "trajtype"])
    args = ap.parse_args()

    assets = _key_assets()
    keys = ([args.key] if args.key else
            sorted({p.stem.split("_")[2] for p in
                    processed_dir(DANDISET).glob(f"emb_{args.method}_*_{args.region}_{BIN_MS}ms.npz")}))
    print(f"000978 trajectory/manifold ({args.region}, {args.method}, last session, "
          f"{args.color_by}): {len(keys)} animals")
    for k in keys:
        if k in assets:
            make_figure(k, assets[k], args.region, args.color_by, args.method)


if __name__ == "__main__":
    main()
