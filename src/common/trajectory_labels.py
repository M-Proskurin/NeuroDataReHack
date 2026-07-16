"""Per-bin W-track "arm + direction" labels (6-way), paper-style trajectory colouring.

For each time bin we ask which arm the animal is on (left / center / right) and
which way it is moving along that arm — toward the arm's reward well ("in", going
into the arm) or back toward the base/choice point ("out", leaving the arm). That
gives six categories (left-in, left-out, center-in, center-out, right-in,
right-out); samples on the base segments are labelled "base".

Arm identity comes from the linearized-track graph nodes; direction comes from the
2-D velocity (from position over time) projected onto the arm's junction→well axis.
These align 1:1 with the smoothed 50 ms embeddings after the same speed mask
(verified: reload the rate matrix, apply preprocessing.speed_mask, and the row
order matches the saved embedding exactly), so no re-embedding is needed.
"""
from __future__ import annotations

import numpy as np

import preprocessing as pp

SPEED_DEFAULT = 4.0
ARMS = ("left", "center", "right")
LABELS6 = ("left-in", "left-out", "center-in", "center-out", "right-in", "right-out")

# in = toward the well (deeper into the arm); out = toward the base (leaving it)
ARM_COLORS = {
    "left-in":    "#08519c", "left-out":   "#6baed6",
    "center-in":  "#006d2c", "center-out": "#74c476",
    "right-in":   "#a50f15", "right-out":  "#fb6a4a",
    "base":       "#bdbdbd",
}


def _seg_dist(P, A, B):
    v = B - A
    L2 = float(v @ v) + 1e-12
    t = np.clip((P - A) @ v / L2, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * v), axis=1)


def arm_direction_labels(position: np.ndarray, time: np.ndarray, nodes: dict) -> np.ndarray:
    """Six-way arm×direction label per sample (or 'base'); NaN-velocity → 'base'."""
    P = np.asarray(position, float)
    segs = [("left", nodes["left_junc"], nodes["left_well"]),
            ("center", nodes["center_junc"], nodes["center_well"]),
            ("right", nodes["right_junc"], nodes["right_well"]),
            ("base", nodes["left_junc"], nodes["center_junc"]),
            ("base", nodes["center_junc"], nodes["right_junc"])]
    dists = np.stack([_seg_dist(P, A, B) for _, A, B in segs], axis=1)
    dists = np.where(np.isfinite(dists), dists, np.inf)
    nearest = np.array([segs[i][0] for i in dists.argmin(axis=1)], dtype=object)

    # 2-D velocity from position over time (sorted for a clean gradient)
    o = np.argsort(time)
    vx = np.full(len(P), np.nan); vy = np.full(len(P), np.nan)
    with np.errstate(invalid="ignore"):
        vx[o] = np.gradient(P[o, 0], time[o])
        vy[o] = np.gradient(P[o, 1], time[o])

    out = np.array(["base"] * len(P), dtype=object)
    for a in ARMS:
        axis = nodes[f"{a}_well"] - nodes[f"{a}_junc"]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        sel = nearest == a
        v_along = vx[sel] * axis[0] + vy[sel] * axis[1]     # >0 = toward well = "in"
        lab = np.where(v_along >= 0, f"{a}-in", f"{a}-out").astype(object)
        lab[~np.isfinite(v_along)] = "base"
        out[sel] = lab
    return out


def masked_labels(position, time, velocity, nodes, speed=SPEED_DEFAULT):
    """Labels for the speed-filtered subset (aligns to the smoothed embeddings).

    Returns (labels_masked, speed_mask). `position`/`time`/`velocity` are the full
    rate-matrix arrays for the chosen condition; the speed mask reproduces the
    embedding row order.
    """
    lab = arm_direction_labels(position, time, nodes)
    m = pp.speed_mask(velocity, speed)
    return lab[m], m
