"""Figure — track trajectory vs. UMAP manifold, coloured by track position.

Two panels for one 000447 CA1 subject and one condition (a single physical
W-track):

  * left  — the animal's 2-D trajectory on the track;
  * right — the (3-D) UMAP activity manifold for the SAME samples.

Colour modes (`--color-by`):
  * ``position``  — continuous LINEARIZED track position (left arm → base →
    centre → base → right arm); the colour means the same thing in both panels.
  * ``trajtype``  — discrete 6-way arm × direction (paper-style): which arm the
    animal is on and whether it is heading into the arm (toward the well) or out
    of it (toward the base). 50 ms only (needs the smoothed-embedding alignment).

Usage:
    pixi run python src/000447/fig_trajectory_manifold.py
    pixi run python src/000447/fig_trajectory_manifold.py --color-by trajtype
    pixi run python src/000447/fig_trajectory_manifold.py --subject JDS-NFN-JS17 --condition novel
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
from config import REPO_ROOT, load_rate_matrix, processed_dir

DANDISET = "000447"
EPOCH_TABLE = "epoch intervals"
CMAP = "turbo"


def _subject_assets():
    return {p.split("/")[0].replace("sub-", ""): p
            for p in dl.list_asset_paths(dandiset_id=DANDISET) if p.endswith(".nwb")}


def condition_graph(nwb, condition):
    """Build the W-track graph for one condition; returns (graph, edge_order, spacing, nodes)."""
    ex = import_module("01_extraction")
    epochs = nwb.intervals[EPOCH_TABLE].to_dataframe().reset_index(drop=True)
    conds = ex.epoch_conditions(nwb.session_description, len(epochs))
    ss = nwb.processing["behavior"].data_interfaces["Position"].spatial_series["SpatialSeries"]
    pos_t = np.asarray(ss.timestamps[:]); pos_xy = np.asarray(ss.data[:])[:, :2]
    trials = nwb.intervals["trials"].to_dataframe().reset_index(drop=True)
    ep_c = epochs[[conds[e] == condition for e in range(len(epochs))]]
    in_c = np.zeros(len(trials), bool); pos_in_c = np.zeros(len(pos_t), bool)
    for _, row in ep_c.iterrows():
        in_c |= (trials.start_time.to_numpy() >= row.start_time) & \
                (trials.start_time.to_numpy() < row.stop_time)
        pos_in_c |= (pos_t >= row.start_time) & (pos_t < row.stop_time)
    wells = lz.wells_from_trials(trials[in_c], pos_t, pos_xy)
    if len(wells) < 3:
        raise RuntimeError(f"could not find 3 wells for condition {condition!r}")
    return lz.build_wtrack_graph(pos_xy[pos_in_c], wells=wells)


def linearized_for_condition(nwb, emb_pos, condition):
    graph, eo, sp, _ = condition_graph(nwb, condition)
    lin, _ = lz.linearize_position(emb_pos, graph, eo, sp)
    return lin


METHOD_LABEL = {"umap": "UMAP", "cebra": "CEBRA (supervised)", "cebratime": "CEBRA-Time"}
METHOD_AXIS = {"umap": "UMAP", "cebra": "CEBRA", "cebratime": "CEBRA-Time"}


def _manifold_axes(ax1, method, bin_ms):
    ax = METHOD_AXIS[method]
    ax1.set_xlabel(f"{ax} 1"); ax1.set_ylabel(f"{ax} 2"); ax1.set_zlabel(f"{ax} 3")
    ax1.set_title(f"CA1 activity manifold ({METHOD_LABEL[method]} 3-D, {bin_ms} ms)")


def _plot_position(P, E, L, subject, condition, bin_ms, method):
    order = np.argsort(L)                    # draw low→high so colour layering is stable
    fig = plt.figure(figsize=(13, 5.6))
    ax0 = fig.add_subplot(1, 2, 1)
    sc0 = ax0.scatter(P[order, 0], P[order, 1], c=L[order], cmap=CMAP, s=6, alpha=0.85)
    ax0.set_aspect("equal"); ax0.set_xlabel("x (cm)"); ax0.set_ylabel("y (cm)")
    ax0.set_title(f"Track trajectory — {subject} CA1 ({condition})")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(E[order, 0], E[order, 1], E[order, 2], c=L[order], cmap=CMAP, s=6, alpha=0.7)
    _manifold_axes(ax1, method, bin_ms)
    cb = fig.colorbar(sc0, ax=[ax0, ax1], shrink=0.7, pad=0.02)
    cb.set_label("linearized track position (cm)")
    fig.suptitle("Track location ↔ neural-manifold location (shared colour = linearized position)",
                 fontsize=12)
    return fig


def _plot_trajtype(P, E, labels, frac, subject, condition, bin_ms, method):
    colors = tl.point_colors(labels, frac)
    base = labels == "base"
    fig = plt.figure(figsize=(13, 5.6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax0.scatter(P[base, 0], P[base, 1], color=tl.BASE_COLOR, s=6, alpha=0.3)
    ax0.scatter(P[~base, 0], P[~base, 1], c=colors[~base], s=6, alpha=0.9)
    ax1.scatter(E[base, 0], E[base, 1], E[base, 2], color=tl.BASE_COLOR, s=6, alpha=0.25)
    ax1.scatter(E[~base, 0], E[~base, 1], E[~base, 2], c=colors[~base], s=6, alpha=0.8)
    ax0.set_aspect("equal"); ax0.set_xlabel("x (cm)"); ax0.set_ylabel("y (cm)")
    ax0.set_title(f"Track trajectory — {subject} CA1 ({condition})")
    _manifold_axes(ax1, method, bin_ms)
    ax0.legend(handles=tl.legend_handles(), fontsize=8, loc="upper right",
               title="arm · direction\n(dark=base → bright=well)")
    fig.suptitle("Arm × direction, shaded by location (in = toward well, out = toward base)",
                 fontsize=12)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="JDS-NFN-AM2")
    ap.add_argument("--condition", default="familiar", choices=["novel", "familiar"])
    ap.add_argument("--bin-ms", type=int, default=50, help="embedding bin size (50 or 1000)")
    ap.add_argument("--method", default="umap", choices=["umap", "cebra", "cebratime"])
    ap.add_argument("--color-by", default="position", choices=["position", "trajtype"])
    args = ap.parse_args()
    bin_ms = args.bin_ms
    if args.color_by == "trajtype" and bin_ms != 50:
        raise SystemExit("trajtype colouring is only defined for the 50 ms smoothed embeddings")

    f = processed_dir(DANDISET) / f"emb_{args.method}_{args.subject}_CA1_{bin_ms}ms.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} — run the {bin_ms} ms {args.method} embedding first")
    d = np.load(f, allow_pickle=False)
    emb, pos, cond = d["embedding"], d["position"], d["condition"]
    m = cond == args.condition
    if m.sum() < 50:
        raise RuntimeError(f"only {m.sum()} samples for condition {args.condition!r}")

    with dl.stream_nwb(_subject_assets()[args.subject], dandiset_id=DANDISET) as nwb:
        graph, eo, sp, nodes = condition_graph(nwb, args.condition)

    outdir = REPO_ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"traj_manifold_000447_CA1_{args.subject}_{args.condition}_{args.method}_{bin_ms}ms"
    if args.color_by == "position":
        lin, _ = lz.linearize_position(pos[m], graph, eo, sp)
        ok = np.isfinite(lin) & np.isfinite(pos[m]).all(axis=1)
        fig = _plot_position(pos[m][ok], emb[m][ok], lin[ok],
                             args.subject, args.condition, bin_ms, args.method)
        out = outdir / f"{stem}.png"
    else:
        rm = load_rate_matrix(args.subject, "CA1", bin_ms, DANDISET)
        cc = rm["condition"] == args.condition
        labels, frac, _ = tl.masked_labels(rm["position"][cc], rm["time"][cc], rm["velocity"][cc], nodes)
        assert len(labels) == int(m.sum()), f"align mismatch {len(labels)} vs {int(m.sum())}"
        ok = np.isfinite(pos[m]).all(axis=1)
        fig = _plot_trajtype(pos[m][ok], emb[m][ok], labels[ok], frac[ok],
                             args.subject, args.condition, bin_ms, args.method)
        out = outdir / f"{stem}_arms.png"

    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"{args.subject} {args.condition} {args.method} [{args.color_by}]: {m.sum()} samples "
          f"({int(ok.sum())} plotted) -> {out.name}")


if __name__ == "__main__":
    main()
