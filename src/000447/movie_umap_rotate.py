"""Rotating 3-D UMAP manifold movie (GIF) for one 000447 CA1 subject/condition.

Renders the smoothed 50 ms CA1 UMAP manifold as a spinning 3-D scatter, saved as
an animated GIF (ffmpeg unavailable, so matplotlib's pillow writer). Colour modes
(`--color-by`):
  * ``position``  — continuous linearized track position;
  * ``trajtype``  — discrete 6-way arm × direction (into the arm / out of it).

Usage:
    pixi run python src/000447/movie_umap_rotate.py                          # JS17, position
    pixi run python src/000447/movie_umap_rotate.py --color-by trajtype      # arm × direction
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import download as dl
import linearize as lz
import trajectory_labels as tl
from config import REPO_ROOT, load_rate_matrix, processed_dir
from fig_trajectory_manifold import condition_graph, _subject_assets

DANDISET = "000447"
BIN_MS = 50
CMAP = "turbo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="JDS-NFN-JS17")
    ap.add_argument("--condition", default="familiar", choices=["novel", "familiar"])
    ap.add_argument("--color-by", default="position", choices=["position", "trajtype"])
    ap.add_argument("--max-points", type=int, default=4000, help="subsample for a lighter GIF")
    ap.add_argument("--azim-step", type=int, default=3, help="degrees per frame")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    d = np.load(processed_dir(DANDISET) / f"emb_umap_{args.subject}_CA1_{BIN_MS}ms.npz",
                allow_pickle=False)
    emb, pos, cond = d["embedding"], d["position"], d["condition"]
    m = cond == args.condition

    with dl.stream_nwb(_subject_assets()[args.subject], dandiset_id=DANDISET) as nwb:
        graph, eo, sp, nodes = condition_graph(nwb, args.condition)
    if args.color_by == "position":
        lin, _ = lz.linearize_position(pos[m], graph, eo, sp)
        ok = np.isfinite(lin) & np.isfinite(pos[m]).all(axis=1)
        E, L, labels, frac = emb[m][ok], lin[ok], None, None
    else:
        rm = load_rate_matrix(args.subject, "CA1", BIN_MS, DANDISET)
        cc = rm["condition"] == args.condition
        lab, fr, _ = tl.masked_labels(rm["position"][cc], rm["time"][cc], rm["velocity"][cc], nodes)
        assert len(lab) == int(m.sum()), f"align {len(lab)} vs {int(m.sum())}"
        ok = np.isfinite(pos[m]).all(axis=1)
        E, L, labels, frac = emb[m][ok], None, lab[ok], fr[ok]

    if len(E) > args.max_points:
        sel = np.random.default_rng(0).choice(len(E), args.max_points, replace=False)
        E = E[sel]; L = None if L is None else L[sel]
        labels = None if labels is None else labels[sel]
        frac = None if frac is None else frac[sel]

    fig = plt.figure(figsize=(6.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    if labels is None:
        sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=L, cmap=CMAP, s=7, alpha=0.75)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1); cb.set_label("linearized track position (cm)")
        tag = ""
    else:
        colors = tl.point_colors(labels, frac)
        base = labels == "base"
        ax.scatter(E[base, 0], E[base, 1], E[base, 2], color=tl.BASE_COLOR, s=7, alpha=0.3)
        ax.scatter(E[~base, 0], E[~base, 1], E[~base, 2], c=colors[~base], s=7, alpha=0.8)
        ax.legend(handles=tl.legend_handles(), fontsize=7, loc="upper left",
                  title="arm · direction\n(dark=base → bright=well)")
        tag = "_arms"
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
    ax.set_title(f"{args.subject} CA1 ({args.condition}) — UMAP 3-D (50 ms)")

    azims = np.arange(0, 360, args.azim_step)

    def update(az):
        ax.view_init(elev=20, azim=az)
        return ()

    anim = FuncAnimation(fig, update, frames=azims, interval=1000 / args.fps, blit=False)
    outdir = REPO_ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"umap_rotate_000447_CA1_{args.subject}_{args.condition}_{BIN_MS}ms{tag}.gif"
    anim.save(out, writer=PillowWriter(fps=args.fps)); plt.close(fig)
    print(f"{len(E)} points, {len(azims)} frames -> {out}")


if __name__ == "__main__":
    main()
