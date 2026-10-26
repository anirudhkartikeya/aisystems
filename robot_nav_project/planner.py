"""planner.py - the shared planner, and the Driver that ties everything together.

    perception (what the car sees) -> brain (where people will be) -> planner -> car

Each control step (0.1 s) the planner:
  1. works out the traffic RULES: speed limit for bends, red lights, waiting
     before a left turn, stopping for people on a crossing (zebra = always stop,
     never swerve), slowing for the goal;
  2. tries CANDIDATE paths: a sideways target (stay on the route, shift a
     little left or right, move into the next lane, or into the oncoming lane)
     combined with a target speed (full, slower ... stop);
  3. rolls every candidate PLAN_HORIZON seconds into the future and checks it
     against where the brain says each person will be at those same times,
     against where every nearby car will be, and against the road (no driving
     onto parking spaces or sidewalks, no swerving inside junctions);
  4. scores the safe ones (closeness to people and cars, distance from the
     route, using the oncoming lane, going slower, changing the plan) and
     drives the cheapest one for one step - then does it all again.
The car only leaves its lane when a person is in the way; for slow cars ahead
it just follows them. If no candidate is safe it brakes as hard as it can.

    driver = Driver(sim, "predictive")      # or "reactive"
    state = sim.reset(seed=3); driver.reset()
    while not done:
        state, info = sim.step(*driver.act(state))
"""
import math

import numpy as np

import config
from brains import make_brain
from perception import Perception

OFF, SAME, ONCOMING, JUNCTION, PARKING = 0, 1, 2, 3, 4


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ======================================================================= the route as a frame
class RouteFrame:
    """The route resampled every `step` m (and extended straight past the goal),
    so a point can be described as (s = distance along the route, d = metres to
    its left) and back."""

    def __init__(self, waypoints, step=0.5, extend=30.0, start=None):
        pts = [tuple(map(float, start if start is not None else waypoints[0]))]
        for p in waypoints:
            if math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-6:
                pts.append((float(p[0]), float(p[1])))
        pts = np.array(pts)
        seg = np.hypot(*(pts[1:] - pts[:-1]).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(cum[-1])
        end_dir = (pts[-1] - pts[-2]) / max(seg[-1], 1e-9)
        n = int(math.ceil((self.length + extend) / step)) + 1
        self.step = step
        self.s = np.arange(n) * step
        inside = np.minimum(self.s, self.length)
        x = np.interp(inside, cum, pts[:, 0]) + end_dir[0] * np.maximum(0.0, self.s - self.length)
        y = np.interp(inside, cum, pts[:, 1]) + end_dir[1] * np.maximum(0.0, self.s - self.length)
        self.xy = np.stack([x, y], axis=1)
        d = np.gradient(self.xy, axis=0)
        self.hdg = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self._hint = 0.0

    def pose(self, s):
        """(xy (..., 2), heading (...)) on the route at distance(s) s."""
        s = np.asarray(s, dtype=float)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        h = np.interp(s, self.s, self.hdg)
        return np.stack([x, y], axis=-1), h

    def to_xy(self, s, d):
        xy, h = self.pose(s)
        d = np.asarray(d, dtype=float)
        return xy + np.stack([-np.sin(h) * d, np.cos(h) * d], axis=-1)

    def project(self, x, y, window=20.0):
        """(s, d) of a point, searching near the last answer (the car moves on)."""
        lo = max(0, int((self._hint - window) / self.step))
        hi = min(len(self.s) - 1, int((self._hint + window) / self.step) + 1)
        a, b = self.xy[lo:hi], self.xy[lo + 1:hi + 1]
        ab = b - a
        ap = np.array([x, y]) - a
        l2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip((ap * ab).sum(1) / l2, 0.0, 1.0)
        dist = np.hypot(*(ap - ab * t[:, None]).T)
        k = int(np.argmin(dist))
        s = float(self.s[lo + k] + t[k] * self.step)
        cross = ab[k, 0] * ap[k, 1] - ab[k, 1] * ap[k, 0]
        d = float(math.copysign(dist[k], cross))
        self._hint = s
        return s, d


# ======================================================================= what is beside the route
def _project_many(points, poly_pts, chunk=4000):
    """Project many points onto one polyline: (s, lateral, distance, heading) each."""
    a, b = poly_pts[:-1], poly_pts[1:]
    ab = b - a
    seg = np.hypot(ab[:, 0], ab[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    hdg = np.arctan2(ab[:, 1], ab[:, 0])
    l2 = np.maximum(seg * seg, 1e-12)
    out_s, out_lat, out_d, out_h = [], [], [], []
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip((ap * ab[None]).sum(2) / l2[None], 0.0, 1.0)
        diff = ap - ab[None] * t[..., None]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        k = np.argmin(dist, axis=1)
        r = np.arange(len(p))
        cross = ab[k, 0] * ap[r, k, 1] - ab[k, 1] * ap[r, k, 0]
        out_s.append(cum[k] + t[r, k] * seg[k])
        out_lat.append(cross / np.maximum(seg[k], 1e-12))
        out_d.append(dist[r, k])
        out_h.append(hdg[k])
    return (np.concatenate(out_s), np.concatenate(out_lat), np.concatenate(out_d), np.concatenate(out_h))


_LANE_CACHE = {}


class LaneGrid:
    """For points beside the route (every 0.5 m along it, every 0.25 m across,
    up to 7 m each side): is it a lane going OUR way, an ONCOMING lane, a
    JUNCTION, PARKING or OFF the road? Worked out once per route."""

    D_MAX, D_STEP = 7.0, 0.25

    def __init__(self, roadmap, frame):
        key = (id(roadmap), round(frame.length, 2), tuple(np.round(frame.xy[::20].ravel(), 2)))
        if key in _LANE_CACHE:
            self.codes = _LANE_CACHE[key]
            return
        d = np.arange(-self.D_MAX, self.D_MAX + 1e-9, self.D_STEP)
        nrm = np.stack([-np.sin(frame.hdg), np.cos(frame.hdg)], axis=1)
        pts = (frame.xy[:, None, :] + nrm[:, None, :] * d[None, :, None]).reshape(-1, 2)
        route_h = np.repeat(frame.hdg, len(d))
        codes = np.full(len(pts), OFF, dtype=np.int8)
        best = np.full(len(pts), np.inf)
        for e in roadmap.edges:
            s, lat, dist, h = _project_many(pts, np.array(e.center.pts))
            ok = (dist <= e.half + 1e-6) & (s >= e.clip_a - 1e-6) & (s <= e.center.length - e.clip_b + 1e-6) \
                & (dist < best)
            if not ok.any():
                continue
            best[ok] = dist[ok]
            lane_h = np.where(lat < 0, h, h + np.pi)            # right of the centre line = forward lanes
            same = np.cos(lane_h - route_h) > 0
            c = np.where(np.abs(lat) > e.drive_half + 1e-6, PARKING, np.where(same, SAME, ONCOMING))
            codes[ok] = c[ok]
        for n in roadmap.nodes:
            if n.kind == "junction":
                inside = (np.abs(pts[:, 0] - n.x) <= n.hx) & (np.abs(pts[:, 1] - n.y) <= n.hy)
                codes[inside] = JUNCTION
        self.codes = codes.reshape(len(frame.s), len(d))
        _LANE_CACHE[key] = self.codes
        self.step = frame.step

    def lookup(self, s, d, step=0.5):
        i = np.clip(np.rint(np.asarray(s) / step).astype(int), 0, self.codes.shape[0] - 1)
        j = np.clip(np.rint((np.asarray(d) + self.D_MAX) / self.D_STEP).astype(int), 0, self.codes.shape[1] - 1)
        return self.codes[i, j]


# ======================================================================= other cars
def predict_vehicles(vehicles, times, robot, me_s=None):
    """Where each sensed car will be at `times`: it follows its lane (and the
    turns it has chosen - like reading its indicators) at its current speed
    and acceleration. Returns circle centres (n, K, 2) and radii (n,).
    A car right behind us in our own lane is left out (it has to brake for us)."""
    centres, radii = [], []
    ch, sh = math.cos(robot.heading), math.sin(robot.heading)
    for v in vehicles:
        dx, dy = v.x - robot.x, v.y - robot.y
        along, across = dx * ch + dy * sh, -dx * sh + dy * ch
        if along < -1.0 and abs(across) < 1.6 and math.cos(v.heading - robot.heading) > 0.7:
            continue
        els = [v.element] + list(v.queue)
        acc = min(1.0, max(-3.0, v.acc))
        pos = []
        for t in times:
            if acc < 0 and v.v + acc * t < 0:
                dist = v.v * v.v / (2 * -acc)
            else:
                dist = v.v * t + 0.5 * acc * t * t
            ss, k = v.s + max(0.0, dist), 0
            while k < len(els) - 1 and ss > els[k].length:
                ss -= els[k].length
                k += 1
            pos.append(els[k].poly.point_at(min(ss, els[k].length)))
        pos = np.array(pos)                                  # (K, 3): x, y, heading
        for o, r in v._offsets:
            centres.append(pos[:, :2] + np.stack([np.cos(pos[:, 2]) * o, np.sin(pos[:, 2]) * o], axis=1))
            radii.append(r)
    if not centres:
        return np.zeros((0, len(times), 2)), np.zeros(0)
    return np.array(centres), np.array(radii)


# ======================================================================= the planner
class Planner:
    def __init__(self, sim):
        self.sim = sim
        self.times = np.arange(0.0, config.PLAN_HORIZON + 1e-9, config.PLAN_DT)
        lo, hi, st = config.PLAN_OFFSETS
        self.offsets = np.round(np.arange(lo, hi + 1e-9, st), 3)
        self.fractions = np.array(config.PLAN_SPEEDS, dtype=float)
        r = sim.robot
        self.car_offsets = np.array([o for o, _ in r._circles])
        self.car_radius = max(rad for _, rad in r._circles)
        self.half_width = config.CAR_WIDTH / 2

    def reset(self, waypoints, start=None):
        self.frame = RouteFrame(waypoints, start=start)
        self.wait = 0.0
        self._lt = None
        self.lanes = LaneGrid(self.sim.world.roadmap, self.frame)
        self.d_target = 0.0
        self.status = "driving"
        self.reacting_to = None
        self.chosen_xy = None
        self.s, self.d = self.frame.project(self.sim.robot.x, self.sim.robot.y)
        self._crossings = self._crossings_on_route()

    def _crossings_on_route(self):
        """Painted crossings the route goes over: (s along the route, crossing)."""
        out = []
        for c in self.sim.world.roadmap.crossings:
            if c.kind not in ("zebra", "signal"):
                continue
            xy = self.frame.xy[:int(self.frame.length / self.frame.step) + 1]
            dist = np.hypot(xy[:, 0] - c.center[0], xy[:, 1] - c.center[1])
            k = int(np.argmin(dist))
            if dist[k] < c.half:
                out.append((float(self.frame.s[k]), c))
        return out

    # ---------- 1. traffic rules: the fastest the car may go, and why ----------
    def rules(self, tracks):
        sim, r = self.sim, self.sim.robot
        vmax = config.ROBOT_MAX_SPEED
        limits = [(vmax, "driving")]
        stops = []                     # route positions (s of the car's centre) the car must not pass
        remaining = self.frame.length - self.s
        limits.append((max(0.0, remaining), "arriving at the goal"))
        # bends: compare the direction now with the route 8 m ahead
        h_now = float(self.frame.pose(self.s)[1])
        h_far = float(self.frame.pose(self.s + 8.0)[1])
        limits.append((vmax * max(config.BEND_MIN_SPEED, math.cos(h_far - h_now)), "slowing for the bend"))
        stop = sim.light_ahead()
        if stop is not None:
            limits.append(((stop - 0.5) * 0.8, "red light"))
            stops.append(self.s + stop)
        if self._lt is not None or float(_wrap(float(self.frame.pose(self.s + 18.0)[1]) - h_now)) > 0.5:
            wait_s = self._left_turn_wait()
            if wait_s is not None:
                gap = wait_s - self.s
                limits.append((max(0.0, gap - 0.3) * 0.8, "waiting for oncoming traffic to turn left"))
                stops.append(wait_s)
        # crossings: someone on (or stepping onto) a crossing ahead -> stop before it, no swerving
        lock = False
        front = config.CAR_LENGTH / 2
        for s_c, c in self._crossings:
            near_edge = s_c - c.width / 2
            ahead = near_edge - self.s - front
            if not -c.width - front < ahead < 25.0:
                continue
            for tr in tracks.values():
                x, y = tr.position
                dx, dy = x - c.center[0], y - c.center[1]
                along = abs(dx * c.along[0] + dy * c.along[1])
                across = abs(dx * c.across[0] + dy * c.across[1])
                on_it = along <= c.width / 2 + 0.5 and across <= c.half + 0.1
                stepping_on = c.kind == "zebra" and along <= c.width / 2 + 0.5 and across <= c.half + config.YIELD_ZONE
                if (on_it or stepping_on) and self._clear_of_us(tr, s_c):
                    continue                                 # already past our lane and walking away
                if on_it or stepping_on:
                    lock = True
                    if ahead > -0.5:                         # not past the line yet
                        limits.append((max(0.0, ahead - 1.0) * 0.8,
                                       "stopping: person on the zebra crossing" if c.kind == "zebra"
                                       else "stopping: person on the crossing"))
                        stops.append(self.s + max(0.0, ahead - 0.5))
                    break
        v_cap, why = min(limits, key=lambda a: a[0])
        self.stop_s = min(stops) if stops else None
        self.stop_why = min(((a, b) for a, b in limits if b not in ("driving", "slowing for the bend",
                                                                     "arriving at the goal")),
                            default=(0.0, "stopping at the line"))[1] if stops else None
        return max(0.0, min(vmax, v_cap)), why, lock

    def _clear_of_us(self, track, s_c):
        """True if a person on a crossing is already past the car's lane plus one
        more lane and walking away from it - no need to keep waiting."""
        from brains import estimate_velocity
        x, y = track.position
        xy, h = self.frame.pose(s_c)
        nx, ny = -math.sin(float(h)), math.cos(float(h))
        lat = (x - float(xy[0])) * nx + (y - float(xy[1])) * ny - self.d      # metres left of the car's path
        vx, vy = estimate_velocity(track)
        v_lat = vx * nx + vy * ny
        return abs(lat) > config.CAR_WIDTH / 2 + config.LANE_WIDTH + 0.5 and lat * v_lat > 0.05

    def _left_turn_wait(self):
        """Turning left: route position (s of the car's centre) where the car must
        wait, or None. It waits at the stop line - or, if it already rolled over
        the line, still straight in its own lane a couple of metres further - until
        it has the green arrow or no oncoming car is within LEFT_TURN_GAP metres
        (the car turns slowly, so it needs a big gap)."""
        sim = self.sim
        front = config.CAR_LENGTH / 2
        if self._lt is None:
            lane, s_l = sim.robot_lane()
            if lane is None or "left" not in getattr(lane, "turns", ()):
                return None
            to_end = lane.length - s_l
            if not front - 0.5 < to_end < 25.0 + front:
                return None
            self._lt = (lane, self.s + to_end - front)            # centre position at the stop line
        lane, s_line = self._lt
        hold = s_line + config.LEFT_TURN_HOLD
        if self.s > hold + 0.3:
            self._lt = None                                         # committed: finish the turn
            return None
        if lane.signal is not None:
            sig, arm = lane.signal
            if sig.axis == "x<" and sig.vehicle_color(arm) == "green":
                return None                                         # protected green arrow
        node, arm = lane.end_node, lane.arm_at_end
        opposite = [l for l in sim.world.roadmap.lanes if l.end_node is node and
                    l.arm_at_end == (-arm[0], -arm[1])]
        wait_at = s_line if self.s <= s_line + 0.2 else hold
        for v in self._vehicles:
            for op in opposite:
                if v.element is op:
                    to_junction = op.length - v.s
                    red = op.signal is not None and op.signal[0].vehicle_color(op.signal[1]) == "red"
                    if red and to_junction > 1.0:
                        continue                                    # they have red and will stop
                    if to_junction < 12.0 or (v.v > 0.5 and to_junction < config.LEFT_TURN_GAP):
                        return wait_at
                elif getattr(v.element, "lane_in", None) is op and v.element.turn != "left":
                    return wait_at                                  # already coming through the junction
        return None

    # ---------- 2-4. candidates, checks, costs ----------
    def plan(self, predictions, vehicles):
        sim, r = self.sim, self.sim.robot
        self.s, self.d = self.frame.project(r.x, r.y)
        s0, d0 = self.s, self.d
        v0 = math.hypot(r.vx, r.vy)
        self._vehicles = vehicles
        v_cap, why, lock = self.rules(self._tracks)
        T, A = self.times, config.ROBOT_MAX_ACCEL

        # speeds: target speeds; slowing down gently (COMFORT_DECEL) or hard (the car's maximum)
        targets = np.unique(np.clip(self.fractions * v_cap, 0.0, None))[::-1]
        v_targets = np.concatenate([targets, targets[targets < v0 - 0.05]])
        hard = np.zeros(len(v_targets), dtype=bool)
        hard[len(targets):] = True
        decel = np.where(hard, A, config.COMFORT_DECEL)[:, None]
        up = np.minimum(v_targets[:, None], v0 + A * T[None, :])
        down = np.maximum(v_targets[:, None], v0 - decel * T[None, :])
        v = np.where(v_targets[:, None] >= v0, up, down)                         # (S, K)
        s = s0 + np.concatenate([np.zeros((len(v_targets), 1)),
                                 np.cumsum((v[:, 1:] + v[:, :-1]) / 2 * np.diff(T)[None], axis=1)], axis=1)
        speed_cost = (v_cap - v_targets) / config.ROBOT_MAX_SPEED
        if self.stop_s is not None and self.stop_s > s0 + 0.2:
            # one more option: keep going, then brake gently to stop right at the line
            vl, sl = [v0], [s0]
            for dt in np.diff(T):
                room = max(0.0, self.stop_s - sl[-1])
                want = min(v_cap, math.sqrt(2 * config.COMFORT_DECEL * room))
                nv = min(want, vl[-1] + A * dt) if want >= vl[-1] else max(want, vl[-1] - A * dt)
                sl.append(sl[-1] + (nv + vl[-1]) / 2 * dt)
                vl.append(nv)
            v = np.vstack([v, vl])
            s = np.vstack([s, sl])
            v_targets = np.append(v_targets, vl[1])
            hard = np.append(hard, False)
            speed_cost = np.append(speed_cost, np.mean(v_cap - np.array(vl)) / config.ROBOT_MAX_SPEED)

        # sideways: smooth move from d0 to each target over a distance that grows with the shift
        d_t = self.offsets[:, None, None]                                       # (D, 1, 1)
        shift = d_t - d0
        L = np.maximum(config.PLAN_SHIFT_LENGTH[0], config.PLAN_SHIFT_LENGTH[1] * np.abs(shift))
        u = np.clip((s[None] - s0) / L, 0.0, 1.0)                               # (D, S, K)
        smooth = u * u * (3 - 2 * u)
        d = d0 + shift * smooth
        slope = shift * 6 * u * (1 - u) / L
        xy_route, h_route = self.frame.pose(s)                                  # (S, K, 2), (S, K)
        nx, ny = -np.sin(h_route), np.cos(h_route)
        px = xy_route[None, ..., 0] + nx[None] * d
        py = xy_route[None, ..., 1] + ny[None] * d
        heading = h_route[None] + np.arctan(slope)
        cx = px[..., None] + np.cos(heading)[..., None] * self.car_offsets            # (D, S, K, 3)
        cy = py[..., None] + np.sin(heading)[..., None] * self.car_offsets

        # the road: no parking lane / sidewalk, no swerving inside junctions, note the oncoming lane
        dd = np.stack([d - self.half_width, d, d + self.half_width], axis=-1)    # (D, S, K, 3)
        codes = self.lanes.lookup(np.broadcast_to(s[None, ..., None], dd.shape), dd)
        bad_road = np.isin(codes, (OFF, PARKING)).any(axis=(2, 3))
        junction_shift = ((codes == JUNCTION).any(axis=3) &
                          (np.abs(d) > max(0.4, abs(d0) + 0.05))).any(axis=2)
        oncoming = (codes == ONCOMING).any(axis=3).mean(axis=2)
        on_route = np.abs(self.offsets) < 0.3
        bad_road[on_route] = False                      # the route itself is always drivable
        junction_shift[on_route] = False
        oncoming[on_route] = 0.0                        # (cutting a tight corner is not "using the oncoming lane")

        # people: closest approach to where the brain says they will be, at the same times.
        # Only the moments when the car is still moving count (a stopped car can't hit anyone),
        # and closeness costs more in the near future than in 3-4 s (the plan is redone every step).
        moving = (v > 0.05)[None, :, :]                                          # (1, S, K)
        moving[..., 0] = True
        weight = np.exp(-T / 4.0)[None, None, :]
        c_ped = np.full(cx.shape[:2], np.inf)
        close_ped = np.zeros(cx.shape[:2])
        who = np.full(cx.shape[:2], -1)
        reach = v0 * T[-1] + 0.5 * A * T[-1] ** 2 + 8.0             # farther than this can't matter
        predictions = [pr for pr in predictions
                       if math.hypot(pr.xy[0][0] - r.x, pr.xy[0][1] - r.y) < reach + 5.0 * T[-1]]
        if predictions:
            P = np.stack([pr.at(T) for pr in predictions])                       # (N, K, 2)
            pr_r = np.array([pr.radius for pr in predictions])
            gx = cx[..., None] - P[:, :, 0].T[None, None, :, None, :]           # (D, S, K, 3, N)
            gy = cy[..., None] - P[:, :, 1].T[None, None, :, None, :]
            gap = (np.hypot(gx, gy) - self.car_radius - pr_r).min(axis=3)        # (D, S, K, N)
            gap = np.where(moving[..., None], gap, np.inf)
            now_ped = gap[0, 0, 0, :].copy()                                     # (N,) gaps right now
            per = gap.min(axis=2)                                                # (D, S, N)
            c_ped = per.min(axis=2)
            who = np.array([pr.id for pr in predictions])[per.argmin(axis=2)]
            close_ped = (np.clip(config.PED_COMFORT - gap.min(axis=3), 0.0, None) * weight).max(axis=2)
        # other cars
        c_veh = np.full(cx.shape[:2], np.inf)
        close_veh = np.zeros(cx.shape[:2])
        V, vr = predict_vehicles(vehicles, T, r)
        if len(vr):
            gx = cx[..., None] - V[:, :, 0].T[None, None, :, None, :]
            gy = cy[..., None] - V[:, :, 1].T[None, None, :, None, :]
            gap = (np.hypot(gx, gy) - self.car_radius - vr).min(axis=(3, 4))    # (D, S, K)
            gap = np.where(moving, gap, np.inf)
            c_veh = gap.min(axis=2)
            close_veh = (np.clip(config.VEH_COMFORT - gap, 0.0, None) * weight).max(axis=2)

        # further ahead: is the lane at each sideways position blocked by someone standing (or
        # predicted to be) in it? The same for every speed - it only helps choose WHICH lane:
        # slowing down behind a person standing in the lane only delays the problem.
        blocked_end = np.zeros(cx.shape[:2], dtype=bool)
        if predictions:
            F = np.stack([pr.at(T[-1:])[0] for pr in predictions])               # (N, 2)
            ahead = s0 + config.CAR_LENGTH / 2 + np.arange(1.0, config.PLAN_CORRIDOR + 1e-9, 1.0)   # (M,)
            ex = self.frame.to_xy(np.broadcast_to(ahead[None], (len(self.offsets), len(ahead))),
                                  np.broadcast_to(self.offsets[:, None], (len(self.offsets), len(ahead))))
            g = np.hypot(ex[..., None, 0] - F[:, 0], ex[..., None, 1] - F[:, 1]) - self.car_radius - pr_r
            lane_blocked = g.min(axis=(1, 2)) < config.PED_MARGIN                # (D,)
            blocked_end = np.broadcast_to(lane_blocked[:, None], cx.shape[:2])

        # people safe: every person keeps PED_MARGIN - except someone the car is ALREADY closer to
        # (it squeezed past them): for them, paths that don't make the gap any smaller are fine,
        # otherwise the car would freeze right next to them
        if predictions:
            margins = np.minimum(config.PED_MARGIN, np.maximum(0.05, now_ped - 0.01))
            ped_ok = (per >= margins[None, None, :]).all(axis=2)
            ped_room = (per - margins[None, None, :]).min(axis=2)
        else:
            ped_ok = np.ones(cx.shape[:2], dtype=bool)
            ped_room = np.full(cx.shape[:2], np.inf)
        safe = ~bad_road & ~junction_shift & ped_ok & (c_veh >= config.VEH_MARGIN)
        # stop lines (red light, waiting to turn left, crossing): don't roll past them -
        # unless the car can't stop in time any more (then it goes through, like at a yellow light)
        if self.stop_s is not None:
            stops_ok = s.max(axis=1) <= self.stop_s + 0.2                        # (S,)
            if stops_ok.any():
                safe &= stops_ok[None, :]

        vmax = config.ROBOT_MAX_SPEED
        cost = (config.COST_SIDEWAYS * np.abs(self.offsets)[:, None]
                + config.COST_ONCOMING * (oncoming > 0) + 2.0 * oncoming
                + (config.COST_SLOW + config.COST_WAIT * self.wait) * speed_cost[None, :]
                + config.COST_HARD_BRAKE * hard[None, :]
                + config.COST_CHANGE * np.abs(self.offsets - self.d_target)[:, None]
                + (config.COST_SLOW + config.COST_WAIT * self.wait) * blocked_end
                + config.COST_PED_CLOSE * close_ped
                + config.COST_VEH_CLOSE * close_veh)

        # only leave the lane for a PERSON in the way (cars ahead: just follow them)
        keep = on_route | (np.abs(self.offsets - self.d_target) < 0.3)
        full = 0                                                    # index of v_cap in v_targets
        conflict = bool(np.any(keep & ((c_ped[:, full] < config.PED_COMFORT) | blocked_end[:, full])))
        allowed = np.ones_like(safe) if (conflict and not lock) else keep[:, None] & np.ones_like(safe)
        options = safe & allowed
        self.debug = dict(c_ped=c_ped, c_veh=c_veh, safe=safe, keep=keep, conflict=conflict, cost=cost,
                          v_targets=v_targets, bad_road=bad_road, junction_shift=junction_shift)
        if options.any():
            k = np.unravel_index(np.argmin(np.where(options, cost, np.inf)), cost.shape)
            emergency = False
        else:
            # nothing keeps the full margins: take the legal path with the most room,
            # and only brake hard (in a straight line) if even that is too close
            legal = ~bad_road & ~junction_shift & allowed
            room = np.where(legal, np.minimum(ped_room, c_veh - config.VEH_MARGIN), -np.inf)
            k = np.unravel_index(np.argmax(room), room.shape)
            emergency = not np.isfinite(room[k]) or min(c_ped[k], c_veh[k]) < 0.15
            if emergency:                                   # the hardest stop there is
                k = (int(np.argmin(np.abs(self.offsets - self.d_target))), int(np.argmin(v_targets - 10 * hard)))
        di, si = int(k[0]), int(k[1])
        d_new, v_new = float(self.offsets[di]), float(v_targets[si])
        if emergency:
            d_new = d0                                  # brake in a straight line, don't steer into anyone

        # who is the car reacting to (for the status text and the metrics)
        lane_keep = int(np.argmin(np.abs(self.offsets - self.d_target)))
        self.reacting_to = int(who[lane_keep, full]) if conflict and who[lane_keep, full] >= 0 else None
        if emergency:
            status = "emergency brake"
        elif not options.any():
            status = "squeezing past (not much room)"
        elif v_new < 0.05 and v0 < 0.5 and conflict:
            status = "stopped for a person"
        elif abs(d_new) > 0.3 and conflict:
            side = "left" if d_new > 0 else "right"
            lane = " into the oncoming lane" if oncoming[di, si] > 0 else ""
            status = f"swerving {side}{lane} around a person"
        elif abs(d_new - d0) > 0.3 and abs(d_new) < 0.3:
            status = "back into the lane"
        elif v_new < v_cap - 0.05 and conflict:
            status = "stopping for a person" if v_new == 0 else "slowing for a person"
        elif v_new < v_cap - 0.05 and self.stop_s is not None and self.stop_s - s0 < 8.0:
            status = self.stop_why
        elif v_new < v_cap - 0.05:
            status = "car ahead"
        else:
            status = why if v_cap < vmax - 0.1 else "driving"
        if v_new < v_cap - 0.05 and v0 < 0.5 and not lock:
            self.wait += config.CONTROL_DT                    # held up (not by a rule): impatience grows
        elif v0 > 1.0:
            self.wait = 0.0
        self.status = status
        self.d_target = d_new
        self.v_target = v_new
        self.chosen_xy = np.stack([px[di, si], py[di, si]], axis=1) if not emergency else \
            self.frame.to_xy(s0 + np.zeros(2), np.full(2, d0))
        self.options = (px, py, options)

        # steer at a point a little ahead on the chosen path
        look = max(config.PLAN_LOOKAHEAD[0], config.PLAN_LOOKAHEAD[1] * v0)
        u_l = min(1.0, look / max(config.PLAN_SHIFT_LENGTH[0], config.PLAN_SHIFT_LENGTH[1] * abs(d_new - d0)))
        d_l = d0 + (d_new - d0) * u_l * u_l * (3 - 2 * u_l)
        tx, ty = self.frame.to_xy(s0 + look, d_l)
        hx, hy = tx - r.x, ty - r.y
        n = math.hypot(hx, hy)
        hx, hy = (hx / n, hy / n) if n > 1e-6 else (math.cos(r.heading), math.sin(r.heading))
        v_cmd = float(np.interp(config.CONTROL_DT, T, v[si]))   # the planned speed one step from now
        return hx * v_cmd, hy * v_cmd


# ======================================================================= the driver
class Driver:
    """Perception + a brain + the planner: drop-in replacement for the placeholder."""

    def __init__(self, sim, brain="predictive"):
        if sim.world.roadmap is None:
            raise ValueError("The planner drives the car on the street and town layouts.")
        self.sim = sim
        self.brain = make_brain(brain) if isinstance(brain, str) else brain
        self.perception = Perception(sim)
        self.planner = Planner(sim)
        self.predictions = []
        self._lines = []

    def reset(self):
        sim = self.sim
        self.perception.reset()
        self.planner.reset(sim.world.route(sim.start, sim.goal), start=(sim.robot.x, sim.robot.y))
        self.predictions = []
        self._lines = []
        self._step = 0

    def act(self, state):
        sim = self.sim
        tracks = self.perception.update()
        self.predictions = self.brain.predict(tracks, sim.t)
        foot = sim.robot.footprint()
        gaps = [min(math.hypot(p.x - fx, p.y - fy) - fr for fx, fy, fr in foot) - p.radius
                for p in sim.peds if abs(p.x - sim.robot.x) < 6 and abs(p.y - sim.robot.y) < 6]
        self.planner._now_gap = min(gaps, default=math.inf)
        self.planner._tracks = tracks
        vx, vy = self.planner.plan(self.predictions, self.perception.vehicles)
        sim.drive_status = f"{self.brain.name}: {self.planner.status}"
        if sim.gui and config.SHOW_PLAN and self._step % config.SHOW_PLAN_EVERY == 0:
            self._draw()
        self._step += 1
        return vx, vy

    # ---------- GUI drawing ----------
    def _line(self, k, a, b, color, width=2.0):
        import pybullet as p
        while len(self._lines) <= k:
            self._lines.append(-1)
        self._lines[k] = p.addUserDebugLine(a, b, color, lineWidth=width, replaceItemUniqueId=self._lines[k])

    def _draw(self):
        z = 0.12
        k = 0
        hide = ([0, 0, -50], [0, 0, -50.01])
        # camera view: two edges of the cone
        cx, cy, h = self.perception.camera_pose()
        half = math.radians(config.PERCEPTION_FOV_DEG) / 2
        for side in (-1, 1):
            a = h + side * half
            self._line(k, [cx, cy, z], [cx + math.cos(a) * config.PERCEPTION_RANGE,
                                        cy + math.sin(a) * config.PERCEPTION_RANGE, z], [0.3, 0.7, 1.0], 1.5)
            k += 1
        # the chosen path (green)
        pts = self.planner.chosen_xy
        step = max(1, (len(pts) - 1) // 4)                   # 4 segments are enough to see it
        for i in range(0, len(pts) - 1, step):
            j = min(i + step, len(pts) - 1)
            self._line(k, [pts[i, 0], pts[i, 1], z], [pts[j, 0], pts[j, 1], z], [0.1, 0.85, 0.2], 4.0)
            k += 1
        # predicted paths of the people the car sees (orange), a short tick for standing people
        r = self.sim.robot
        near = sorted(self.predictions, key=lambda pr: math.hypot(pr.xy[0][0] - r.x, pr.xy[0][1] - r.y))
        for pr in near[:6]:
            a, b = pr.xy[0], pr.xy[-1]
            if np.hypot(*(b - a)) < 0.05:
                b = a + np.array([0.0, 0.3])
            self._line(k, [a[0], a[1], z + 1.9], [b[0], b[1], z + 1.9], [1.0, 0.55, 0.0], 3.0)
            k += 1
        for i in range(k, len(self._lines)):
            self._line(i, *hide, [0, 0, 0], 1.0)
