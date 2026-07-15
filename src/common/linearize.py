"""W-track linearization: map 2-D position to 1-D track-relative position.

Wraps the Frank-lab `track_linearization` package. The two 000447 tracks (novel,
familiar) are physically different mazes in different orientations, so raw 2-D
position is not comparable between them; linearizing both onto a common W
topology (left / center / right arm, base) yields track-relative coordinates that
ARE comparable, and gives cleaner 1-D spatial binning / decoding on the single
000978 W.

The 6 W nodes (3 arm-end wells + 3 base junctions) are estimated from the
occupancy per track (orientation-agnostic), so no maze coordinates are hardcoded:
  * wells = 3 clusters of low-speed dwell positions (reward wells)
  * arms run perpendicular to the well-line; junctions = wells projected onto the
    base line at the far extent of occupancy
Arms are ordered center/left/right and concatenated in a fixed edge order, so
linear position means the same thing across tracks.

    from linearize import build_wtrack_graph, linearize_position
    graph, edge_order, spacing, nodes = build_wtrack_graph(position, velocity)
    lin, seg = linearize_position(position, graph, edge_order, spacing)
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from track_linearization import get_linearized_position, make_track_graph

EDGE_SPACING = 15.0
DWELL_SPEED = 3.0


def wells_from_trials(trials_df, pos_t: np.ndarray, pos_xy: np.ndarray) -> np.ndarray:
    """Reward-well coordinates (n_wells, 2) from the trials table.

    At each trial's start/stop the animal is at start_well/end_well (reward), so
    each well's position is the median tracked position at those event times.
    This is far more reliable than clustering occupancy (which also piles up at
    the choice point).
    """
    ev_t, ev_w = [], []
    for col_t, col_w in [("start_time", "start_well"), ("stop_time", "end_well")]:
        ev_t.append(trials_df[col_t].to_numpy()); ev_w.append(trials_df[col_w].to_numpy())
    ev_t = np.concatenate(ev_t); ev_w = np.concatenate(ev_w)
    idx = np.searchsorted(pos_t, ev_t).clip(0, len(pos_t) - 1)
    wells = []
    for w in np.unique(ev_w):
        p = pos_xy[idx[ev_w == w]]
        p = p[np.isfinite(p).all(axis=1)]
        if len(p):
            wells.append(np.median(p, axis=0))
    return np.asarray(wells)


def _order_and_junctions(wells: np.ndarray, position: np.ndarray, arm_pctl: float = 97.0):
    """Order wells along the well-line and place junctions at the base."""
    finite = np.isfinite(position).all(axis=1)
    P = position[finite]
    c = wells.mean(axis=0)
    u = np.linalg.svd(wells - c)[2][0]              # well-line direction
    wells = wells[np.argsort((wells - c) @ u)]      # order along the line
    perp = np.array([-u[1], u[0]])
    if (P.mean(axis=0) - c) @ perp < 0:             # point into the occupancy (toward base)
        perp = -perp
    arm_len = np.percentile((P - c) @ perp, arm_pctl)
    junctions = wells + perp * arm_len
    return wells, junctions


def _wells_from_occupancy(position, velocity, dwell_speed=DWELL_SPEED):
    """Fallback well estimate (less reliable than trials): low-speed dwell clusters."""
    finite = np.isfinite(position).all(axis=1)
    dwell = position[finite]
    if velocity is not None:
        v = velocity[finite]
        low = np.isfinite(v) & (v < dwell_speed)
        if low.sum() > 100:
            dwell = position[finite][low]
    return KMeans(n_clusters=3, n_init=10, random_state=0).fit(dwell).cluster_centers_


def build_wtrack_graph(position: np.ndarray, velocity: np.ndarray | None = None,
                       wells: np.ndarray | None = None):
    """Build the W track graph in the data's coordinate frame.

    `wells` (n=3, x/y) should come from `wells_from_trials` when available (much
    more reliable); otherwise a low-speed-occupancy fallback is used. Returns
    (track_graph, edge_order, edge_spacing, nodes) where nodes is a dict of the
    well/junction coordinates for plotting/QC.
    """
    if wells is None:
        wells = _wells_from_occupancy(position, velocity)
    wells, junc = _order_and_junctions(np.asarray(wells), position)
    # node ids: 0 center well, 1 left well, 2 right well, 3 center junc, 4 left junc, 5 right junc
    node_positions = np.array([wells[1], wells[0], wells[2], junc[1], junc[0], junc[2]])
    edges = [(0, 3), (1, 4), (2, 5), (4, 3), (3, 5)]     # 3 arms + 2 base segments
    graph = make_track_graph(node_positions, edges)
    # fixed 1-D layout: left arm, base-left, center arm, base-right, right arm
    edge_order = [(1, 4), (4, 3), (3, 0), (3, 5), (5, 2)]
    nodes = {"center_well": wells[1], "left_well": wells[0], "right_well": wells[2],
             "center_junc": junc[1], "left_junc": junc[0], "right_junc": junc[2]}
    return graph, edge_order, EDGE_SPACING, nodes


def linearize_position(position: np.ndarray, track_graph, edge_order,
                       edge_spacing: float = EDGE_SPACING):
    """Return (linear_position, track_segment_id) for each 2-D sample.

    NaN positions pass through as NaN linear position.
    """
    finite = np.isfinite(position).all(axis=1)
    lin = np.full(position.shape[0], np.nan)
    seg = np.full(position.shape[0], -1, dtype=int)
    df = get_linearized_position(position=position[finite], track_graph=track_graph,
                                 edge_order=edge_order, edge_spacing=edge_spacing)
    lin[finite] = df["linear_position"].to_numpy()
    seg[finite] = df["track_segment_id"].to_numpy()
    return lin, seg
