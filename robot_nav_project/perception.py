"""perception.py - what the robot car can actually see.

People are seen only through a FRONT CAMERA: up to PERCEPTION_RANGE metres
ahead, inside a PERCEPTION_FOV_DEG wide cone, and only when nothing blocks the
line of sight (buildings, bus shelters, parked cars and other cars all hide
people standing behind them). Close-range sensors (like parking sensors) also
notice anyone within PERCEPTION_NEAR metres of the car, in any direction.

Other cars are seen all around (mirrors + side sensors, up to
PERCEPTION_VEHICLE_RANGE metres), because the car must check the lanes next to
it before it swerves.

For every person it has seen, perception keeps a short TRACK: their last few
positions with times. The brains only ever get these tracks, never the
simulator's own lists, so they work with the same kind of information YOLO
will give in Phase 2. (The positions are exact here; set PERCEPTION_NOISE to
add measurement noise.)
"""
import math
from collections import deque

import numpy as np

import config


class Track:
    """One person the car has seen: id, radius and a history of (t, x, y)."""

    def __init__(self, pid, radius):
        self.id = pid
        self.radius = radius
        self.history = deque(maxlen=int(config.TRACK_HISTORY / config.CONTROL_DT) + 1)
        self.last_seen = -math.inf

    @property
    def position(self):
        return self.history[-1][1], self.history[-1][2]


class Perception:
    def __init__(self, sim, rng=None):
        self.sim = sim
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tracks = {}            # person id -> Track
        self.visible = set()        # ids seen this step
        self.vehicles = []          # vehicles sensed this step (traffic.Vehicle objects)
        self._occluders = None

    def reset(self):
        self.tracks.clear()
        self.visible = set()
        self.vehicles = []
        self._prepare_occluders()

    # ---------- static things that block the view ----------
    def _prepare_occluders(self):
        boxes = []
        roadmap = self.sim.world.roadmap
        for ob in (roadmap.obstacles if roadmap is not None else []):
            if ob["type"] == "box" and ob.get("height", 3.0) >= config.PERCEPTION_OCCLUDER_HEIGHT:
                (cx, cy), (hx, hy) = ob["center"], ob["half_extents"]
                boxes.append((cx - hx, cy - hy, cx + hx, cy + hy))
        self._occluders = np.array(boxes, dtype=float).reshape(-1, 4)

    def camera_pose(self):
        """Camera position (just behind the windscreen) and the direction it faces."""
        r = self.sim.robot
        c, s = math.cos(r.heading), math.sin(r.heading)
        mount = config.PERCEPTION_CAMERA_FORWARD
        return r.x + c * mount, r.y + s * mount, r.heading

    # ---------- one step ----------
    def update(self):
        sim = self.sim
        t = sim.t
        cx, cy, heading = self.camera_pose()
        ch, sh = math.cos(heading), math.sin(heading)
        half_fov = math.radians(config.PERCEPTION_FOV_DEG) / 2
        foot = sim.robot.footprint()

        # cars nearby (all round) - they are also things people can hide behind
        rng_v = config.PERCEPTION_VEHICLE_RANGE
        self.vehicles = [v for v in (sim.traffic.vehicles if sim.traffic else [])
                         if v.active and abs(v.x - cx) < rng_v and abs(v.y - cy) < rng_v
                         and math.hypot(v.x - cx, v.y - cy) < rng_v]
        car_circles = np.array([c for v in self.vehicles for c in v.circles()], dtype=float).reshape(-1, 3)

        # static occluders near the camera
        R = config.PERCEPTION_RANGE
        occ = self._occluders
        if len(occ):
            near = (occ[:, 2] > cx - R) & (occ[:, 0] < cx + R) & (occ[:, 3] > cy - R) & (occ[:, 1] < cy + R)
            occ = occ[near]

        self.visible = set()
        for ped in sim.peds:
            dx, dy = ped.x - cx, ped.y - cy
            if abs(dx) > R + 4 or abs(dy) > R + 4:
                continue
            close = min(math.hypot(ped.x - fx, ped.y - fy) - fr for fx, fy, fr in foot) < config.PERCEPTION_NEAR
            dist = math.hypot(dx, dy)
            in_view = False
            if dist < R:
                along = dx * ch + dy * sh
                across = -dx * sh + dy * ch
                if along > 0 and abs(math.atan2(across, along)) < half_fov:
                    in_view = not self._blocked(cx, cy, ped.x, ped.y, occ, car_circles)
            if not (in_view or close):
                continue
            self.visible.add(ped.ped_id)
            track = self.tracks.get(ped.ped_id)
            if track is None:
                track = self.tracks[ped.ped_id] = Track(ped.ped_id, config.PED_RADIUS)
            x, y = ped.x, ped.y
            if config.PERCEPTION_NOISE > 0:
                x += float(self.rng.normal(0, config.PERCEPTION_NOISE))
                y += float(self.rng.normal(0, config.PERCEPTION_NOISE))
            track.history.append((t, x, y))
            track.last_seen = t

        # forget people not seen for a while
        for pid in [pid for pid, tr in self.tracks.items() if t - tr.last_seen > config.TRACK_FORGET]:
            del self.tracks[pid]
        return self.tracks

    @staticmethod
    def _blocked(ax, ay, bx, by, boxes, circles):
        """True if the straight line from the camera (a) to the person (b)
        passes through a box or a car circle. The last 0.4 m next to the
        person are ignored, so someone right beside a parked car still counts
        as visible when you can see them."""
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return False
        t_end = max(0.0, 1.0 - 0.4 / length)
        if len(boxes):
            # slab test against axis-aligned boxes
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_x = 1.0 / dx if abs(dx) > 1e-9 else np.inf
                inv_y = 1.0 / dy if abs(dy) > 1e-9 else np.inf
                tx1, tx2 = (boxes[:, 0] - ax) * inv_x, (boxes[:, 2] - ax) * inv_x
                ty1, ty2 = (boxes[:, 1] - ay) * inv_y, (boxes[:, 3] - ay) * inv_y
            if abs(dx) <= 1e-9:
                inside_x = (boxes[:, 0] <= ax) & (ax <= boxes[:, 2])
                tx1 = np.where(inside_x, -np.inf, np.inf)
                tx2 = np.where(inside_x, np.inf, -np.inf)
            if abs(dy) <= 1e-9:
                inside_y = (boxes[:, 1] <= ay) & (ay <= boxes[:, 3])
                ty1 = np.where(inside_y, -np.inf, np.inf)
                ty2 = np.where(inside_y, np.inf, -np.inf)
            t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            if np.any((t_in <= t_out) & (t_out > 0.0) & (t_in < t_end)):
                return True
        if len(circles):
            px, py = circles[:, 0] - ax, circles[:, 1] - ay
            t = np.clip((px * dx + py * dy) / (length * length), 0.0, t_end)
            d = np.hypot(px - t * dx, py - t * dy)
            if np.any((d < circles[:, 2] * 0.9) & (t > 0.0) & (t < t_end)):
                return True
        return False
