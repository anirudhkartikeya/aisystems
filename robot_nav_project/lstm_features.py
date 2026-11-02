"""lstm_features.py - the context the LSTM v2 sees at every past step of a person.

Used both for training (train_lstm.py, from the trajectories/ CSV files) and, later,
inside the car (the "lstm" brain), so both compute exactly the same numbers.

For one person at one moment (all in metres / m/s, world frame, then scaled):
  road      kerb    : distance to the road edge (+ on the sidewalk, - on the road)
            to_road : unit vector from the person towards the nearest road centre line
            on_road : 1 if on the asphalt
  crossing  rel     : vector to the nearest painted crossing (cut off at 15 m)
            signal  : 1 if that crossing has lights, walk = 1 if people may cross now (zebras: always 1)
            on_it   : 1 if standing on that crossing
  cars      nearest car (any) and the robot car: relative position and velocity, present flag
  people    the 3 nearest other people within 6 m: relative position and velocity, + crowd count
"""
import numpy as np

N_FEATURES = 33
NAMES = (["kerb", "to_road_x", "to_road_y", "on_road",
          "xing_dx", "xing_dy", "xing_dist", "xing_signal", "xing_walk", "on_xing"]
         + [f"car_{k}" for k in ("dx", "dy", "dvx", "dvy", "present")]
         + [f"robot_{k}" for k in ("dx", "dy", "dvx", "dvy", "present")]
         + [f"ped{i}_{k}" for i in range(3) for k in ("dx", "dy", "dvx", "dvy")]
         + ["crowd"])
assert len(NAMES) == N_FEATURES

CAR_RANGE = 25.0      # m: cars farther than this don't count
PED_RANGE = 6.0       # m: people farther than this don't count
XING_RANGE = 15.0     # m


def _project(points, poly):
    """Closest point on a polyline for many points: (closest_xy (N,2), dist (N,))."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(1), 1e-12)
    best_d = np.full(len(points), np.inf)
    best_c = np.zeros_like(points)
    for i in range(0, len(points), 20000):
        p = points[i:i + 20000]
        ap = p[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / l2[None], 0.0, 1.0)
        c = a[None] + ab[None] * t[..., None]
        d = np.hypot(p[:, None, 0] - c[..., 0], p[:, None, 1] - c[..., 1])
        k = d.argmin(1)
        r = np.arange(len(p))
        best_d[i:i + 20000] = d[r, k]
        best_c[i:i + 20000] = c[r, k]
    return best_c, best_d


def map_features(roadmap, xy, sim_time):
    """Road and crossing features for points xy (N,2) at simulator times sim_time (N,) -> (N, 10)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    t = np.asarray(sim_time, dtype=float).reshape(-1)
    n = len(xy)
    out = np.zeros((n, 10), dtype=np.float32)
    # road edge: distance to the kerb, direction to the road centre
    kerb = np.full(n, np.inf)
    toward = np.zeros((n, 2))
    for e in roadmap.edges:
        c, d = _project(xy, np.array(e.center.pts, dtype=float))
        k = d - e.half
        better = k < kerb
        kerb[better] = k[better]
        toward[better] = c[better] - xy[better]
    for node in roadmap.nodes:
        if node.kind != "junction":
            continue
        dx, dy = xy[:, 0] - node.x, xy[:, 1] - node.y
        inside = (np.abs(dx) <= node.hx) & (np.abs(dy) <= node.hy)
        margin = -np.minimum(node.hx - np.abs(dx), node.hy - np.abs(dy))
        better = inside & (margin < kerb)
        kerb[better] = margin[better]
        toward[better] = np.stack([-dx[better], -dy[better]], 1)
    norm = np.maximum(np.hypot(toward[:, 0], toward[:, 1]), 1e-6)
    out[:, 0] = np.clip(kerb, -8.0, 8.0) / 4.0
    out[:, 1:3] = toward / norm[:, None]
    out[:, 3] = (kerb < 0).astype(np.float32)
    # nearest painted crossing (zebra or with lights)
    xings = [c for c in roadmap.crossings if c.kind in ("zebra", "signal")]
    if xings:
        centers = np.array([c.center for c in xings], dtype=float)
        d = np.hypot(xy[:, None, 0] - centers[None, :, 0], xy[:, None, 1] - centers[None, :, 1])
        j = d.argmin(1)
        rel = centers[j] - xy
        dist = d[np.arange(n), j]
        scale = np.minimum(1.0, XING_RANGE / np.maximum(dist, 1e-6))
        out[:, 4:6] = rel * scale[:, None] / 10.0
        out[:, 6] = np.minimum(dist, XING_RANGE) / 10.0
        for k, c in enumerate(xings):
            sel = j == k
            if not sel.any():
                continue
            if c.kind == "signal" and c.node is not None and c.node.signal is not None:
                out[sel, 7] = 1.0
                sig = c.node.signal
                times = t[sel]
                walk = np.zeros(len(times), dtype=np.float32)
                for u in np.unique(np.round(times, 2)):
                    sig.update(float(u))
                    walk[np.round(times, 2) == u] = float(sig.walk(c))
                out[sel, 8] = walk
            else:
                out[sel, 8] = 1.0
            p = xy[sel]
            ddx, ddy = p[:, 0] - c.center[0], p[:, 1] - c.center[1]
            along = np.abs(ddx * c.along[0] + ddy * c.along[1])
            across = np.abs(ddx * c.across[0] + ddy * c.across[1])
            out[sel, 9] = ((along <= c.width / 2) & (across <= c.half + 0.1)).astype(np.float32)
    return out


def frame_features(p_xy, p_v, cars_xy, cars_v, robot_mask):
    """Car and people features for everyone in ONE moment -> (P, 23).
    p_xy, p_v: (P,2) people; cars_xy, cars_v: (C,2) vehicles (incl. the robot car); robot_mask: (C,) bool."""
    P = len(p_xy)
    out = np.zeros((P, 23), dtype=np.float32)
    if P == 0:
        return out
    if len(cars_xy):
        rel = cars_xy[None, :, :] - p_xy[:, None, :]                       # (P, C, 2)
        dist = np.hypot(rel[..., 0], rel[..., 1])
        for col, mask in ((0, np.ones(len(cars_xy), bool)), (5, robot_mask)):
            if not mask.any():
                continue
            dm = np.where(mask[None, :], dist, np.inf)
            j = dm.argmin(1)
            ok = dm[np.arange(P), j] < CAR_RANGE
            out[ok, col:col + 2] = rel[np.arange(P), j][ok] / 10.0
            out[ok, col + 2:col + 4] = (cars_v[j] - p_v)[ok] / 5.0
            out[ok, col + 4] = 1.0
    if P > 1:
        rel = p_xy[None, :, :] - p_xy[:, None, :]
        dist = np.hypot(rel[..., 0], rel[..., 1])
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)[:, :3]
        for k in range(min(3, P - 1)):
            j = order[:, k]
            dk = dist[np.arange(P), j]
            ok = dk < PED_RANGE
            out[ok, 10 + 4 * k:12 + 4 * k] = rel[np.arange(P), j][ok] / 3.0
            out[ok, 12 + 4 * k:14 + 4 * k] = (p_v[j] - p_v)[ok] / 1.5
        out[:, 22] = (dist < 3.0).sum(1) / 4.0
    return out
