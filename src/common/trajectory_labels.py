"""Per-bin W-track "arm + direction" labels (6-way) + a location gradient colour.

For each time bin we ask which arm the animal is on (left / center / right), which
way it is moving along that arm — toward the reward well ("in", going into the
arm) or back toward the base/choice point ("out", leaving the arm) — and how far
along the arm it is (fraction from base = 0 to well = 1). Samples on the base
segments are labelled "base".

Colour scheme (`point_colors`): a distinct HUE per (arm × direction) so the two
directions on the same arm differ in hue (not just lightness), and BRIGHTNESS
graded by the along-arm fraction (dark at the base end → bright at the well) so
colour also varies with location within each run.

Arm identity + fraction come from the linearized-track graph nodes; direction
comes from the 2-D velocity (position over time) projected on the arm's
junction→well axis. These align 1:1 with the smoothed 50 ms embeddings after the
same speed mask (verified), so no re-embedding is needed.
"""
from __future__ import annotations

import numpy as np
import matplotlib.colors as _mc

import preprocessing as pp

SPEED_DEFAULT = 4.0
ARMS = ("left", "center", "right")
LABELS6 = ("left-in", "left-out", "center-in", "center-out", "right-in", "right-out")

# Distinct hue per arm×direction (in and out of the same arm get different hues);
# location (fraction along the arm) sets brightness in point_colors().
CAT_HUE = {
    "left-in": 0.62, "left-out": 0.48,      # blue      vs cyan
    "center-in": 0.33, "center-out": 0.17,  # green     vs yellow-green
    "right-in": 0.99, "right-out": 0.83,    # red       vs magenta
}
BASE_COLOR = (0.80, 0.80, 0.80)
_SAT = 0.92
_V_LO, _V_HI = 0.40, 0.97               # base end (dark) → well end (bright)


def _seg_dist_frac(P, A, B):
    v = B - A
    L2 = float(v @ v) + 1e-12
    t = np.clip((P - A) @ v / L2, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * v), axis=1), t


def arm_direction_labels(position: np.ndarray, time: np.ndarray, nodes: dict):
    """Return (labels, frac): 6-way arm×direction label (or 'base') and the
    along-arm fraction (0 = base end, 1 = well; NaN for 'base')."""
    P = np.asarray(position, float)
    segs = [("left", nodes["left_junc"], nodes["left_well"]),
            ("center", nodes["center_junc"], nodes["center_well"]),
            ("right", nodes["right_junc"], nodes["right_well"]),
            ("base", nodes["left_junc"], nodes["center_junc"]),
            ("base", nodes["center_junc"], nodes["right_junc"])]
    dist = np.empty((len(P), len(segs))); frac = np.empty((len(P), len(segs)))
    for j, (_, A, B) in enumerate(segs):
        dist[:, j], frac[:, j] = _seg_dist_frac(P, A, B)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    ni = dist.argmin(axis=1)
    nearest = np.array([segs[i][0] for i in ni], dtype=object)
    frac_near = frac[np.arange(len(P)), ni]

    o = np.argsort(time)
    vx = np.full(len(P), np.nan); vy = np.full(len(P), np.nan)
    with np.errstate(invalid="ignore"):
        vx[o] = np.gradient(P[o, 0], time[o])
        vy[o] = np.gradient(P[o, 1], time[o])

    labels = np.array(["base"] * len(P), dtype=object)
    out_frac = np.full(len(P), np.nan)
    for a in ARMS:
        axis = nodes[f"{a}_well"] - nodes[f"{a}_junc"]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        sel = nearest == a
        v_along = vx[sel] * axis[0] + vy[sel] * axis[1]      # >0 = toward well = "in"
        lab = np.where(v_along >= 0, f"{a}-in", f"{a}-out").astype(object)
        lab[~np.isfinite(v_along)] = "base"
        labels[sel] = lab
        out_frac[sel] = frac_near[sel]
    out_frac[labels == "base"] = np.nan
    return labels, out_frac


def masked_labels(position, time, velocity, nodes, speed=SPEED_DEFAULT):
    """(labels, frac, speed_mask) for the speed-filtered subset (aligns to the embeddings)."""
    lab, frac = arm_direction_labels(position, time, nodes)
    m = pp.speed_mask(velocity, speed)
    return lab[m], frac[m], m


def point_colors(labels, frac):
    """(N, 3) RGB: hue from arm×direction, brightness from along-arm fraction."""
    labels = np.asarray(labels, dtype=object)
    frac = np.asarray(frac, float)
    out = np.tile(np.array(BASE_COLOR), (len(labels), 1))
    for lab in LABELS6:
        s = labels == lab
        if not s.any():
            continue
        fr = np.clip(np.where(np.isfinite(frac[s]), frac[s], 0.0), 0, 1)
        v = _V_LO + (_V_HI - _V_LO) * fr
        hsv = np.stack([np.full(v.shape, CAT_HUE[lab]), np.full(v.shape, _SAT), v], axis=1)
        out[s] = _mc.hsv_to_rgb(hsv)
    return out


def legend_handles(frac=0.85):
    """Proxy handles for the 6 categories (+ base) at a representative brightness."""
    from matplotlib.lines import Line2D
    h = []
    for lab in LABELS6:
        c = _mc.hsv_to_rgb((CAT_HUE[lab], _SAT, _V_LO + (_V_HI - _V_LO) * frac))
        h.append(Line2D([0], [0], marker="o", ls="", color=c, label=lab, ms=7))
    h.append(Line2D([0], [0], marker="o", ls="", color=BASE_COLOR, label="base", ms=7))
    return h
