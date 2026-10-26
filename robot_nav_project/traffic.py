"""traffic.py - other cars driving around the street and town layouts.

Each car follows the lane paths of the road map (roadmap.py): along a lane,
through a junction on a connector (straight, left or right, picked at
random), along the next lane, and so on. Cars that drive off the map come
back in at another map edge a few seconds later, so the number of cars
stays the same.

Speed is chosen with the Intelligent Driver Model (IDM), the standard model
of how drivers follow the car ahead. Cars:
  - keep a safe gap to anything on their path ahead: other cars, the robot,
    people on the road
  - stop at red (and yellow, if they can) traffic lights
  - stop for people waiting at or walking on zebra crossings
  - turning left, wait for oncoming traffic
Like the pedestrians, they are kinematic (placed directly), and every random
choice comes from the generator passed in, so a seed always replays the same.
"""
import math

import numpy as np

import config
import visuals

KIND_SHARES = [("sedan", 0.38), ("hatch", 0.22), ("suv", 0.2), ("van", 0.1), ("taxi", 0.1)]
SAMPLE_STEP = 1.0     # m between the points a driver checks along its path


class Vehicle:
    def __init__(self, vid, kind, color, driver):
        self.id, self.kind = vid, kind
        self.length, self.width, _ = visuals.VEHICLE_KINDS[kind]
        self.driver = driver              # speed factor of this driver (some are faster)
        self.figure = visuals.VehicleFigure(kind, color)
        r = self.width / 2 + 0.05
        n = max(2, int(math.ceil(self.length / self.width)))
        span = self.length / 2 - r
        self._offsets = [(-span + 2 * span * k / (n - 1), r) for k in range(n)]
        self.active = False
        self.element, self.s, self.v, self.acc = None, 0.0, 0.0, 0.0
        self.queue = []                   # path elements chosen ahead
        self.x = self.y = -1000.0
        self.heading = 0.0
        self.yield_to = None              # id of the crossing it is stopping for
        self.stopped_for = 0.0
        self.respawn_in = 0.0

    def place(self):
        self.x, self.y, self.heading = self.element.poly.point_at(self.s)

    def circles(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return [(self.x + c * o, self.y + s * o, r) for o, r in self._offsets]

    def snapshot(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return {"id": self.id, "kind": self.kind, "x": self.x, "y": self.y,
                "vx": self.v * c, "vy": self.v * s, "heading": self.heading,
                "length": self.length, "width": self.width, "circles": self.circles(),
                "is_robot": False, "yield": self.yield_to}


class TrafficManager:
    def __init__(self, roadmap, rng, look_rng, n, avoid=()):
        self.map = roadmap
        self.rng = rng
        self.vehicles = []
        self._samples = {}
        for el in roadmap.lanes + roadmap.connectors:
            pts = el.poly.samples(SAMPLE_STEP)
            self._samples[id(el)] = (np.array([[p[0], p[1]] for p in pts]), np.array([p[3] for p in pts]),
                                     np.array([p[2] for p in pts]))
        self.entries = [l for l in roadmap.lanes if l.entry]
        lanes = [l for l in roadmap.lanes if l.length > 12]
        weights = [l.length for l in lanes]
        for k in range(n):
            kind = self._pick_kind(look_rng)
            if look_rng.random() < config.TRAFFIC_BUS_SHARE:
                kind = "bus"
            color = visuals.CAR_COLORS[int(look_rng.integers(len(visuals.CAR_COLORS)))]
            spread = config.TRAFFIC_DRIVER_SPREAD
            veh = Vehicle(k, kind, color, 1.0 + rng.uniform(-spread, spread))
            self.vehicles.append(veh)
            for _ in range(60):                   # find a free spot on some lane
                lane = lanes[_weighted(rng, weights)]
                if kind == "bus" and not lane.edge.parking:
                    continue
                s = rng.uniform(veh.length, lane.length - 6.0)
                x, y, _ = lane.poly.point_at(s)
                if any(math.hypot(x - ax, y - ay) < d for ax, ay, d in avoid):
                    continue
                if any(o.active and math.hypot(x - o.x, y - o.y) < 12.0 for o in self.vehicles):
                    continue
                self._activate(veh, lane, s, rng.uniform(0.3, 0.9) * lane.speed * veh.driver)
                break
            else:
                veh.respawn_in = rng.uniform(0.0, 5.0)

    @staticmethod
    def _pick_kind(rng):
        r = rng.random()
        for kind, share in KIND_SHARES:
            r -= share
            if r <= 0:
                return kind
        return "sedan"

    # ---------- life cycle ----------
    def _activate(self, veh, lane, s, v):
        veh.active = True
        veh.element, veh.s, veh.v, veh.acc = lane, s, v, 0.0
        veh.queue = []
        veh.yield_to = None
        veh.stopped_for = 0.0
        self._extend_queue(veh)
        veh.place()

    def _deactivate(self, veh):
        veh.active = False
        veh.x = veh.y = -1000.0
        veh.v = 0.0
        veh.respawn_in = self.rng.uniform(1.0, 6.0)
        veh.figure.set_pose(-1000.0, -1000.0, 0.0)

    def _choose_turn(self, veh, lane):
        options = lane.next
        if not options:
            return None
        if veh.kind == "bus":
            straight = [c for c in options if c.turn == "straight"]
            if straight:
                return straight[0]
        weights = [config.TRAFFIC_TURN_WEIGHTS.get(c.turn, 0.2) for c in options]
        return options[_weighted(self.rng, weights)]

    def _extend_queue(self, veh):
        """Choose the next few path elements so the driver can look ahead."""
        ahead = veh.element.length - veh.s + sum(e.length for e in veh.queue)
        while ahead < config.TRAFFIC_LOOKAHEAD + 20:
            last = veh.queue[-1] if veh.queue else veh.element
            if hasattr(last, "lane_out"):
                nxt = last.lane_out
            elif last.exit:
                return
            else:
                nxt = self._choose_turn(veh, last)
                if nxt is None:
                    return
            veh.queue.append(nxt)
            ahead += nxt.length

    # ---------- decisions (control rate) ----------
    def plan(self, ctx):
        """Choose every car's acceleration for the next control step."""
        dt = config.CONTROL_DT
        for veh in self.vehicles:
            if not veh.active:
                veh.respawn_in -= dt
                if veh.respawn_in <= 0:
                    self._try_respawn(veh, ctx)
        others = [v for v in self.vehicles if v.active]
        if not others:
            return
        # everything that can block a car's path, as circles
        obs_xy, obs_r, obs_v, obs_owner, obs_heading = [], [], [], [], []
        for v in others:
            c, s = math.cos(v.heading), math.sin(v.heading)
            for x, y, r in v.circles():
                obs_xy.append((x, y)); obs_r.append(r); obs_v.append((v.v * c, v.v * s))
                obs_owner.append(v.id); obs_heading.append(v.heading)
        for veh in ctx["vehicles"]:
            if veh.get("is_robot"):
                for t in (0.0, 0.7, 1.4):                 # where the robot is, and is about to be
                    for x, y, r in veh["circles"]:
                        obs_xy.append((x + veh["vx"] * t, y + veh["vy"] * t)); obs_r.append(r)
                        obs_v.append((veh["vx"], veh["vy"])); obs_owner.append(-1); obs_heading.append(veh["heading"])
        for ped in ctx["peds"]:
            obs_xy.append((ped.x, ped.y)); obs_r.append(ped.radius); obs_v.append((0.0, 0.0))
            obs_owner.append(-2); obs_heading.append(0.0)
            if ped.vx * ped.vx + ped.vy * ped.vy > 0.09:     # drivers expect people to keep walking
                for t in (0.6, 1.2):
                    obs_xy.append((ped.x + ped.vx * t, ped.y + ped.vy * t)); obs_r.append(ped.radius)
                    obs_v.append((0.0, 0.0)); obs_owner.append(-2); obs_heading.append(0.0)
        obs_xy = np.array(obs_xy)
        obs_r = np.array(obs_r)
        obs_v = np.array(obs_v)
        obs_owner = np.array(obs_owner)
        obs_heading = np.array(obs_heading)
        for veh in others:
            veh.acc = self._idm(veh, ctx, obs_xy, obs_r, obs_v, obs_owner, obs_heading)
            veh.stopped_for = veh.stopped_for + dt if veh.v < 0.2 else 0.0

    def _lookahead(self, veh):
        """Points along the car's path ahead of its front bumper: (xy, distance, heading, chain)."""
        front = veh.s + veh.length / 2
        xs, ds, hs = [], [], []
        offset = 0.0
        chain = [(veh.element, 0.0)]
        el_list = [veh.element] + veh.queue
        for k, el in enumerate(el_list):
            pts, ss, hh = self._samples[id(el)]
            if k == 0:
                mask = ss >= front
                pts, ss, hh = pts[mask], ss[mask] - front, hh[mask]
                base = 0.0
                offset = el.length - front
            else:
                base = offset
                chain.append((el, base))
                offset += el.length
            xs.append(pts); ds.append(ss + base); hs.append(hh)
            if offset > config.TRAFFIC_LOOKAHEAD:
                break
        xs, ds, hs = np.concatenate(xs), np.concatenate(ds), np.concatenate(hs)
        keep = ds >= 0
        return xs[keep], ds[keep], hs[keep], chain, el_list[:len(chain)]

    def _idm(self, veh, ctx, obs_xy, obs_r, obs_v, obs_owner, obs_heading):
        a, b = config.TRAFFIC_ACCEL, config.TRAFFIC_BRAKE
        v = veh.v
        v0 = max(1.0, veh.element.speed * veh.driver)
        acc = a * (1 - (v / v0) ** 4)
        pts, dist, heads, chain, elements = self._lookahead(veh)
        front = veh.s + veh.length / 2
        # 1. things physically on the path ahead
        if len(pts):
            mask_self = obs_owner != veh.id
            near = mask_self & (np.abs(obs_xy[:, 0] - veh.x) < config.TRAFFIC_LOOKAHEAD + 5) & \
                (np.abs(obs_xy[:, 1] - veh.y) < config.TRAFFIC_LOOKAHEAD + 5)
            idx = np.nonzero(near)[0]
            if len(idx):
                d = np.hypot(pts[:, None, 0] - obs_xy[None, idx, 0], pts[:, None, 1] - obs_xy[None, idx, 1])
                hit = d < (obs_r[idx][None, :] + veh.width / 2 + 0.25)
                any_hit = hit.any(axis=0)
                for j in np.nonzero(any_hit)[0]:
                    k = int(np.argmax(hit[:, j]))
                    o = idx[j]
                    if obs_owner[o] >= 0 and veh.stopped_for > 12.0 and \
                            abs(_wrap(obs_heading[o] - heads[k])) > 1.0:
                        continue                  # break a rare junction deadlock
                    if obs_owner[o] == -1 and veh.stopped_for > 4.0 and abs(obs_v[o, 0]) + abs(obs_v[o, 1]) < 0.2 \
                            and not (d[:, j] < obs_r[o] + veh.width / 2 + 0.05).any():
                        continue                  # the robot car is waiting for us and we really fit past it
                    gap = max(0.1, dist[k] - 0.3)
                    lead_v = obs_v[o, 0] * math.cos(heads[k]) + obs_v[o, 1] * math.sin(heads[k])
                    acc = min(acc, _idm_follow(v, v0, gap, max(0.0, lead_v)))
        # 2. traffic lights, zebra crossings, left turns: stop lines on the way
        veh.yield_to = None
        for el, base in chain:
            to_end = base + el.length - (front if el is veh.element else 0.0)
            if el.signal is not None and to_end > -0.5:
                sig, arm = el.signal
                color = sig.vehicle_color(arm)
                if color == "red" or (color == "yellow" and v * v / (2 * b) < to_end - 0.5):
                    acc = min(acc, _idm_stop(v, v0, to_end))
            if config.TRAFFIC_YIELD_ZEBRA:
                for s_c, crossing in el.crossings:
                    d_c = base + s_c - (front if el is veh.element else 0.0)
                    if d_c < -1.0 or d_c > 35.0:
                        continue
                    if self._people_at(crossing, ctx):
                        if crossing.kind == "zebra" or crossing.kind == "signal":
                            stop_at = d_c - crossing.width / 2 - 1.2
                            if stop_at > -0.5:
                                acc = min(acc, _idm_stop(v, v0, stop_at))
                                if crossing.kind == "zebra":
                                    veh.yield_to = crossing.id
            if getattr(el, "turn", None) == "left":
                to_start = base - (front if el is veh.element else 0.0)
                if 0 <= to_start < 20 and self._oncoming(el, ctx):
                    acc = min(acc, _idm_stop(v, v0, to_start))
        return max(-config.TRAFFIC_MAX_BRAKE, min(a, acc))

    def _people_at(self, crossing, ctx):
        """Someone walking on the crossing, or waiting at its kerb to cross."""
        for ped in ctx["peds"]:
            if abs(ped.x - crossing.center[0]) > 12 or abs(ped.y - crossing.center[1]) > 12:
                continue
            if crossing.contains(ped.x, ped.y, 0.4):
                return True
            steps = getattr(ped, "steps", None)
            if ped.mode == "wait" and steps and steps[0][2] is crossing and crossing.kind == "zebra":
                return True
        return False

    def _oncoming(self, conn, ctx):
        """True if a car (or the robot) is coming the other way through the junction."""
        lane_in = conn.lane_in
        node = conn.node
        opposite = [l for l in self.map.lanes if l.end_node is node and
                    l.arm_at_end == (-lane_in.arm_at_end[0], -lane_in.arm_at_end[1])]
        movers = [v for v in self.vehicles if v.active and v.v > 0.5]
        for op in opposite:
            for v in movers:
                if v.element is op and op.length - v.s < 30.0:
                    return True
                if getattr(v.element, "lane_in", None) is op and v.element.turn != "left":
                    return True
            for rv in ctx["vehicles"]:
                if rv.get("is_robot") and math.hypot(rv["vx"], rv["vy"]) > 0.5:
                    s, d, _ = op.poly.project(rv["x"], rv["y"])
                    if d < 2.0 and op.length - s < 30.0:
                        return True
        return False

    def _try_respawn(self, veh, ctx):
        entries = [l for l in self.entries if veh.kind != "bus" or l.edge.parking]
        order = list(self.rng.permutation(len(entries)))
        for i in order:
            lane = entries[int(i)]
            x, y, _ = lane.poly.point_at(0.0)
            busy = any(o.active and math.hypot(o.x - x, o.y - y) < 15.0 for o in self.vehicles)
            busy = busy or any(math.hypot(r["x"] - x, r["y"] - y) < 15.0 for r in ctx["vehicles"])
            if not busy:
                self._activate(veh, lane, veh.length / 2, 0.7 * lane.speed * veh.driver)
                return
        veh.respawn_in = 1.0

    # ---------- motion (physics rate) ----------
    def move(self, dt):
        for veh in self.vehicles:
            if not veh.active:
                continue
            veh.v = max(0.0, veh.v + veh.acc * dt)
            veh.s += veh.v * dt
            while veh.s > veh.element.length:
                if not veh.queue:
                    self._deactivate(veh)
                    break
                veh.s -= veh.element.length
                veh.element = veh.queue.pop(0)
                self._extend_queue(veh)
            if veh.active:
                veh.place()

    # ---------- output ----------
    def snapshots(self):
        return [v.snapshot() for v in self.vehicles if v.active]

    def sync_bodies(self, near=None):
        """Move the drawn cars; near=(x, y, d) skips cars farther than d from (x, y)."""
        for veh in self.vehicles:
            if veh.active and (near is None or (abs(veh.x - near[0]) < near[2] and abs(veh.y - near[1]) < near[2])):
                veh.figure.set_pose(veh.x, veh.y, veh.heading)
                veh.figure.set_brake(veh.acc < -0.6 or veh.v < 0.3)

    def remove(self):
        for veh in self.vehicles:
            veh.figure.remove()
        self.vehicles = []


def _idm_follow(v, v0, gap, lead_v):
    a, b = config.TRAFFIC_ACCEL, config.TRAFFIC_BRAKE
    s_star = config.TRAFFIC_MIN_GAP + v * config.TRAFFIC_HEADWAY + v * (v - lead_v) / (2 * math.sqrt(a * b))
    return a * (1 - (v / v0) ** 4 - (max(0.0, s_star) / max(gap, 0.1)) ** 2)


def _idm_stop(v, v0, dist):
    """Acceleration to stop at a line `dist` metres ahead."""
    a, b = config.TRAFFIC_ACCEL, config.TRAFFIC_BRAKE
    gap = max(dist, 0.05)
    s_star = 0.3 + v * config.TRAFFIC_HEADWAY * 0.5 + v * v / (2 * math.sqrt(a * b))
    return a * (1 - (v / v0) ** 4 - (s_star / gap) ** 2)


def _weighted(rng, weights):
    r = rng.uniform(0.0, sum(weights))
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i
    return len(weights) - 1


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
