"""pedestrians.py - simulated people.

Three kinds:

  Pedestrian         - the simple walker used indoors (room, doorway, road):
                       walks to random goals, pauses, avoids walls and people
                       (a "social force" model).
  StreetPedestrian   - realistic city pedestrians for the "street" and "town"
                       layouts. They plan routes along sidewalks and use
                       crossings, wait at the kerb, obey (most of the time)
                       pedestrian lights, look for gaps in traffic, walk in
                       groups, stop now and then, sometimes jaywalk, change
                       their mind halfway across, and - if they are paying
                       attention - slow down, hurry or step back for cars.
  CrossingPedestrian - a scripted person who crosses right in front of the
                       robot, timed to meet it (the "crossing" test scenario).

Everything random comes from the generator passed in, so the same seed
always gives the same people doing the same things.
"""
import math

import config
import visuals

# One colour per pedestrian for the simple walkers and the trails
PED_COLORS = [
    [0.90, 0.20, 0.20, 1], [0.95, 0.60, 0.10, 1], [0.60, 0.20, 0.80, 1],
    [0.10, 0.70, 0.70, 1], [0.85, 0.30, 0.60, 1], [0.50, 0.50, 0.10, 1],
    [0.30, 0.30, 0.30, 1], [0.95, 0.85, 0.10, 1],
]
STRIDE = 1.35        # m walked per full leg cycle (two steps)


class Pedestrian:
    """Simple walker (room / doorway / road), and the base class of the others."""

    def __init__(self, ped_id, start, goal, speed, rng, world, look=None, kind="adult"):
        self.ped_id = ped_id          # stable ID: 0, 1, 2, ... (the LSTM needs this)
        self.kind = kind
        self.radius = config.PED_KINDS.get(kind, {}).get("radius", config.PED_RADIUS)
        self.color = (look or {}).get("shirt", PED_COLORS[ped_id % len(PED_COLORS)])
        self.rng = rng
        self.world = world
        self.x, self.y = start
        self.vx = self.vy = 0.0
        self.preferred_speed = speed
        self.group_id = ped_id
        self.mode = "walk"
        self.pause_left = 0.0
        self.waypoints = world.route(start, goal) if goal is not None else []
        target = self.waypoints[0] if self.waypoints else (self.x + 1.0, self.y)
        self.yaw = math.atan2(target[1] - self.y, target[0] - self.x)   # direction the figure faces
        self.phase = rng.uniform(0, 2 * math.pi) if hasattr(rng, "uniform") else 0.0
        self.swing = 0.0
        # How it looks. Behaviour never uses this: it only uses (x, y) and radius.
        self.figure = visuals.make_person(self.color, self.x, self.y, look)
        self.sync_body()

    @property
    def body_ids(self):
        """PyBullet bodies that draw this pedestrian."""
        return self.figure.body_ids

    @property
    def label_ids(self):
        """The bodies a camera box should cover (not an umbrella, say)."""
        return getattr(self.figure, "label_ids", self.figure.body_ids)

    # ---------- decision (control rate) ----------
    def choose_velocity(self, ctx):
        dt = config.CONTROL_DT
        desired_vx = desired_vy = 0.0
        if self.pause_left > 0:
            self.pause_left -= dt
            self.mode = "pause"
        elif self.rng.random() < config.PED_PAUSE_PROB:
            self.pause_left = self.rng.uniform(config.PED_PAUSE_MIN, config.PED_PAUSE_MAX)
        else:
            self.mode = "walk"
            tx, ty = self.waypoints[0]
            dist = math.hypot(tx - self.x, ty - self.y)
            if dist < config.PED_GOAL_TOLERANCE:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self._new_goal()
                tx, ty = self.waypoints[0]
                dist = math.hypot(tx - self.x, ty - self.y)
            if dist > 1e-6:
                desired_vx = (tx - self.x) / dist * self.preferred_speed
                desired_vy = (ty - self.y) / dist * self.preferred_speed
        fx, fy = self._social_forces(desired_vx, desired_vy, ctx, avoid_vehicles=config.PED_AVOID_ROBOT)
        self._integrate(fx, fy, 1.3 * self.preferred_speed)

    def _social_forces(self, desired_vx, desired_vy, ctx, avoid_vehicles=False, group_scale=1.0):
        # 1. steer toward the desired velocity
        fx = (desired_vx - self.vx) / config.PED_RELAX_TIME
        fy = (desired_vy - self.vy) / config.PED_RELAX_TIME
        # 2. push away from other pedestrians, with a small sideways nudge so
        #    two people walking head-on pass each other instead of getting stuck
        for other in ctx["peds"]:
            if other is self or abs(other.x - self.x) > 2.0 or abs(other.y - self.y) > 2.0:
                continue
            gx, gy = self._push(other.x, other.y, other.radius)
            if other.group_id == self.group_id:
                gx, gy = gx * group_scale, gy * group_scale       # friends walk close together
            fx, fy = fx + gx + 0.4 * gy, fy + gy - 0.4 * gx
        # 3. push away from walls and obstacles
        for px, py, _ in self.world.closest_static_points(self.x, self.y, config.PED_REPULSION_CUTOFF + 0.5):
            gx, gy = self._push(px, py, 0.0)
            fx, fy = fx + gx, fy + gy
        # 4. keep clear of cars: people paying attention avoid all cars,
        #    everyone walks around a car that is standing still
        for veh in ctx["vehicles"]:
            if abs(veh["x"] - self.x) > 8 or abs(veh["y"] - self.y) > 8:
                continue
            if avoid_vehicles or veh["vx"] * veh["vx"] + veh["vy"] * veh["vy"] < 1.0:
                for cx, cy, cr in veh["circles"]:
                    gx, gy = self._push(cx, cy, cr)
                    k = config.PED_CAR_REPULSION / config.PED_REPULSION_STRENGTH if avoid_vehicles else 1.0
                    fx, fy = fx + k * gx, fy + k * gy
        return fx, fy

    def _integrate(self, fx, fy, max_speed):
        dt = config.CONTROL_DT
        self.vx += fx * dt
        self.vy += fy * dt
        speed = math.hypot(self.vx, self.vy)
        if speed > max_speed:
            self.vx, self.vy = self.vx / speed * max_speed, self.vy / speed * max_speed
            speed = max_speed
        if speed < 0.03:
            self.vx = self.vy = 0.0
        if speed > 0.1:                       # face the way they walk
            self.yaw = math.atan2(self.vy, self.vx)

    def _push(self, ox, oy, other_radius):
        """Repulsion from a point/disc at (ox, oy); fades exponentially with the gap."""
        dx, dy = self.x - ox, self.y - oy
        dist = math.hypot(dx, dy)
        gap = dist - self.radius - other_radius
        if dist < 1e-9 or gap > config.PED_REPULSION_CUTOFF:
            return 0.0, 0.0
        strength = config.PED_REPULSION_STRENGTH * math.exp(-gap / config.PED_REPULSION_RANGE)
        return strength * dx / dist, strength * dy / dist

    def _new_goal(self):
        goal = self.world.random_walk_point(self.rng, self.radius + 0.3)
        self.waypoints = self.world.route((self.x, self.y), goal)
        self.preferred_speed = min(config.PED_SPEED_MAX, max(config.PED_SPEED_MIN,
                                   self.preferred_speed * self.rng.uniform(0.9, 1.1)))

    # ---------- motion (physics rate) ----------
    def move(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.stay_out_of_walls()
        speed = math.hypot(self.vx, self.vy)
        self.phase = (self.phase + 2 * math.pi * speed * dt / STRIDE) % (2 * math.pi)
        self.swing = 0.5 * min(1.0, speed / 1.3) * math.sin(self.phase)

    def stay_out_of_walls(self):
        """Never end up inside a wall or obstacle: slide back out to its surface."""
        for px, py, d in self.world.closest_static_points(self.x, self.y, self.radius + 0.1):
            if d < self.radius:
                dx, dy = self.x - px, self.y - py
                n = math.hypot(dx, dy)
                if n > 1e-9:
                    self.x, self.y = px + dx / n * self.radius, py + dy / n * self.radius

    def sync_body(self):
        """Move the drawn figure to the current (x, y), facing its walking
        direction, legs mid-stride, standing on the sidewalk if not on the road."""
        lift = 0.0
        if self.world.roadmap is not None and not self.world.on_road(self.x, self.y):
            lift = config.CURB_HEIGHT
        self.figure.set_pose(self.x, self.y, self.yaw, self.swing, lift)

    def state(self):
        return {"id": self.ped_id, "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
                "group": self.group_id, "kind": self.kind, "mode": self.mode}

    def remove(self):
        self.figure.remove()


# ======================================================================= city pedestrians
class StreetPedestrian(Pedestrian):
    """A realistic city pedestrian (street and town layouts)."""

    def __init__(self, ped_id, start, speed, rng, world, kind, look, leader=None, slot=0):
        self.roadmap = world.roadmap
        self.leader = leader
        self.members = []
        self.slot = slot
        self.steps = []
        super().__init__(ped_id, start, None, speed, rng, world, look, kind)
        if leader is not None:
            self.group_id = leader.group_id
            leader.members.append(self)
        # personality (fixed for the whole trial)
        self.aware = rng.random() < config.PED_AWARE_SHARE
        self.phone = rng.random() < config.PED_PHONE_SHARE
        self.jaywalker = rng.random() < config.PED_JAYWALKER_SHARE
        cautious = rng.random()
        lo, hi = config.PED_GAP_ACCEPT
        self.gap_accept = lo + (hi - lo) * cautious
        self.side_offset = (rng.uniform(-1, 1) * config.PED_LANE_WANDER + 0.15) if leader is None else 0.0
        if self.phone:
            self.preferred_speed *= 0.88
        self.base_speed = self.preferred_speed
        # slow random wobble in direction and speed
        self.noise_heading = 0.0
        self.noise_speed = 0.0
        # crossing bookkeeping
        self.wait_start = None
        self.look_time = 0.0
        self.brave = False
        self.turn_back_at = None
        self.cross_from = None
        self.retreat = False
        self.turned_back = False

    # ---------- planning ----------
    def plan_new_trip(self, ctx=None):
        """Pick somewhere to go and plan the route there."""
        rng = self.rng
        if self.jaywalker and rng.random() < config.PED_JAYWALK_PROB:
            steps = self._plan_jaywalk()
            if steps:
                self.steps = steps
                return
        near_trip = rng.random() < 0.7
        goal = self.world.random_walk_point(rng, self.radius + 0.2, near=(self.x, self.y) if near_trip else None,
                                            max_dist=45.0 if near_trip else None)
        self.steps = self._shift(self.roadmap.plan_walk((self.x, self.y), goal))

    def _plan_jaywalk(self):
        """Cross the road straight away instead of walking to a crossing."""
        we, s_here, _ = self.roadmap.nearest_sidewalk(self.x, self.y)
        road = we.road
        if road is None:
            return None
        s_road, _, _ = road.center.project(self.x, self.y)
        w = config.SIDEWALK_WIDTH
        for shift in (0.0, 3.0, -3.0, 6.0, -6.0):
            s_c = s_road + shift + self.rng.uniform(1.0, 6.0)
            if not (road.clip_a + 8 < s_c < road.center.length - road.clip_b - 8):
                continue
            cx, cy, hdg = road.center.point_at(s_c)
            if any(c.edge is road and math.hypot(c.center[0] - cx, c.center[1] - cy) < 10 for c in self.roadmap.crossings):
                continue                                   # a real crossing is right there: not jaywalking
            crossing = self.roadmap.jaywalk_crossing(road, cx, cy)
            near, far = crossing.curb(we.side), crossing.curb(-we.side)
            if not self.world.line_is_free(near, far, self.radius + 0.05):
                continue                                   # a parked car is in the way
            nx, ny = crossing.across[0] * we.side, crossing.across[1] * we.side
            here = (cx + nx * (road.half + w / 2), cy + ny * (road.half + w / 2))
            s_goal = s_c + self.rng.uniform(-10, 10)
            gx, gy, _ = road.center.point_at(s_goal)
            goal = (gx - nx * (road.half + w / 2), gy - ny * (road.half + w / 2))
            if not self.world.is_free(goal[0], goal[1], self.radius + 0.2) or \
                    not self.world.is_free(here[0], here[1], self.radius + 0.2):
                continue
            to_here = self.roadmap.plan_walk((self.x, self.y), here)
            return self._shift(to_here) + [(near[0], near[1], crossing, "curb"),
                                           (far[0], far[1], crossing, "far"),
                                           (goal[0], goal[1], None, "walk")]
        return None

    def _shift(self, steps):
        """Keep this person's own sideways offset on sidewalks and crossings."""
        out = []
        px, py = self.x, self.y
        for x, y, crossing, role in steps:
            if crossing is not None:
                o = self.side_offset * 0.6
                x, y = x + crossing.along[0] * o, y + crossing.along[1] * o
            elif role == "walk":
                dx, dy = x - px, y - py
                n = math.hypot(dx, dy)
                if n > 1e-6:
                    x, y = x + dy / n * self.side_offset, y - dx / n * self.side_offset   # + = to the right
            out.append((x, y, crossing, role))
            px, py = x, y
        return out

    # ---------- decision (control rate) ----------
    def choose_velocity(self, ctx):
        dt = config.CONTROL_DT
        rng = self.rng
        # slowly drifting wobble: nobody walks a perfect straight line
        k = dt / 2.0
        self.noise_heading += -k * self.noise_heading + rng.normal(0, config.PED_HEADING_NOISE * math.sqrt(2 * k))
        self.noise_speed += -k * self.noise_speed + rng.normal(0, config.PED_SPEED_NOISE * math.sqrt(2 * k))

        if self.leader is not None:
            desired, cap = self._follow_leader()
        else:
            desired, cap = self._decide(ctx)
        aware_now = self.aware and not (self.phone and rng.random() < 0.5)
        fx, fy = self._social_forces(desired[0], desired[1], ctx, avoid_vehicles=aware_now, group_scale=0.3)
        self._integrate(fx, fy, cap)
        if self.mode == "wait" and self.leader is None and math.hypot(self.vx, self.vy) < 0.3:
            c = self.steps[0][2].center                 # waiting at the kerb: look at the road
            self.yaw = math.atan2(c[1] - self.y, c[0] - self.x)

    def _follow_leader(self):
        L = self.leader
        self.mode = L.mode
        face = L.yaw
        if L.mode == "wait" and L.steps and L.steps[0][2] is not None:
            cc = L.steps[0][2].center                  # at the kerb: line up facing the road
            face = math.atan2(cc[1] - L.y, cc[0] - L.x)
        c, s = math.cos(face), math.sin(face)
        row = (self.slot + 1) // 2
        lateral = row * config.PED_GROUP_SPACING * (1 if self.slot % 2 else -1)
        back = 0.45 if L.mode == "wait" else 0.35 * (self.slot // 2)   # at the kerb: stand beside, a bit back
        tx = L.x - c * back - s * lateral
        ty = L.y - s * back + c * lateral
        if L.mode != "cross" and self.world.on_road(tx, ty):
            tx, ty = L.x - c * 0.6 * self.slot, L.y - s * 0.6 * self.slot   # no room beside: queue behind
        dx, dy = tx - self.x, ty - self.y
        lead_speed = math.hypot(L.vx, L.vy)
        cap = max(1.3 * self.preferred_speed, lead_speed + 0.4)
        vx = L.vx + dx / 0.8
        vy = L.vy + dy / 0.8
        n = math.hypot(vx, vy)
        if n > cap:
            vx, vy = vx / n * cap, vy / n * cap
        if n < 0.1 and math.hypot(dx, dy) < 0.3:
            vx = vy = 0.0
        return (vx, vy), cap

    def _decide(self, ctx):
        dt = config.CONTROL_DT
        rng = self.rng
        cap = 1.3 * self.preferred_speed
        if self.pause_left > 0:                     # standing: phone, shop window, chat, arrived
            self.pause_left -= dt
            self.mode = "pause"
            return (0.0, 0.0), cap
        if not self.steps:
            self.plan_new_trip(ctx)
        x, y, crossing, role = self.steps[0]
        dist = math.hypot(x - self.x, y - self.y)

        if role == "walk":
            self.mode = "walk"
            if dist < config.PED_GOAL_TOLERANCE:
                self.steps.pop(0)
                if not self.steps:                   # arrived: stand a while, then go somewhere else
                    self.pause_left = rng.uniform(*config.PED_ARRIVE_PAUSE)
                    self.mode = "pause"
                    return (0.0, 0.0), cap
                return self._decide(ctx)
            if rng.random() < config.PED_PAUSE_RATE * dt * (2.0 if self.phone else 1.0) \
                    and not self.world.on_road(self.x, self.y):
                self.pause_left = rng.uniform(*config.PED_PAUSE_TIME)
            return self._toward(x, y, 1.0), cap

        if role == "curb":
            self.mode = "wait"
            if dist > 0.35 and self.wait_start is None:
                self.mode = "walk"
                return self._toward(x, y, 1.0), cap
            if self.wait_start is None:              # just arrived at the kerb
                self.wait_start = ctx["t"]
                self.look_time = rng.uniform(*config.PED_KERB_LOOK) if self.aware else rng.uniform(0, 0.3)
                self.brave = self.jaywalker and rng.random() < config.PED_RED_LIGHT_PROB
                self.retreat = False
            if self._may_cross(crossing, ctx):
                self.steps.pop(0)
                self.wait_start = None
                self.cross_from = (x, y)
                self.turn_back_at = rng.uniform(0.2, 0.5) if rng.random() < config.PED_TURN_BACK_PROB else None
                self.mode = "cross"
                return self._decide(ctx)
            # wait here, facing the road
            if dist > 0.15:
                return self._toward(x, y, 0.5), cap
            return (0.0, 0.0), cap

        # role == "far": walking across the road
        self.mode = "cross"
        if dist < config.PED_GOAL_TOLERANCE:
            self.steps.pop(0)
            self.cross_from = None
            return self._decide(ctx)
        start = self.cross_from or (self.x, self.y)
        total = max(0.1, math.hypot(x - start[0], y - start[1]))
        progress = 1.0 - dist / total
        if self.turn_back_at is not None and progress > self.turn_back_at:
            # changed their mind: go back to where they came from, then plan something new
            self.turn_back_at = None
            self.turned_back = True
            self.steps = [(start[0], start[1], None, "walk")]
            return self._decide(ctx)
        speed = 1.1
        if self.aware and not self.phone:
            threat = self._threat(ctx)
            if threat is not None:
                if progress < 0.3:                   # barely started: step back to the kerb and wait again
                    self.steps.insert(0, (start[0], start[1], crossing, "curb"))
                    self.wait_start = None
                    return self._decide(ctx)
                speed = config.PED_HURRY            # already out there: hurry across
        if crossing is not None and crossing.kind == "signal" and not self._signal_walk(crossing, ctx):
            speed = max(speed, 1.25)                 # light changed while crossing: walk faster
        vx, vy = self._toward(x, y, speed)
        if self._stopped_car_ahead(vx, vy, ctx):
            self.mode = "wait"                       # a car is standing right in front: wait for it
            return (0.0, 0.0), cap
        return (vx, vy), cap * max(1.0, speed)

    def _stopped_car_ahead(self, vx, vy, ctx):
        """True if a stopped (or crawling) car is right where this person is about to step."""
        sp = math.hypot(vx, vy)
        if sp < 1e-6:
            return False
        ux, uy = vx / sp, vy / sp
        for veh in ctx["vehicles"]:
            if math.hypot(veh["vx"], veh["vy"]) > 0.5:
                continue
            if abs(veh["x"] - self.x) > 6 or abs(veh["y"] - self.y) > 6:
                continue
            for k in (0.3, 0.7):
                px, py = self.x + ux * k, self.y + uy * k
                for cx, cy, cr in veh["circles"]:
                    if math.hypot(px - cx, py - cy) < cr + self.radius + 0.2:
                        return True
        return False

    def _toward(self, x, y, factor):
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        heading = math.atan2(dy, dx) + (self.noise_heading if dist > 1.0 else 0.0)
        speed = self.preferred_speed * factor * (1.0 + self.noise_speed)
        speed = min(speed, dist / 0.5 + 0.2)          # ease in to the target
        return math.cos(heading) * speed, math.sin(heading) * speed

    # ---------- traffic awareness ----------
    def _signal_walk(self, crossing, ctx):
        sig = crossing.node.signal if crossing.node is not None else None
        return True if sig is None else sig.walk(crossing)

    def _may_cross(self, crossing, ctx):
        waited = ctx["t"] - self.wait_start
        if waited < self.look_time:
            return False
        if self._car_on(crossing, ctx):
            return False                               # nobody walks into a car standing on the crossing
        if crossing.kind == "signal":
            if self._signal_walk(crossing, ctx):
                return not self.aware or self._road_clear(crossing, ctx, 1.5)
            # red: law-abiding people wait; brave jaywalkers go if the road looks clear
            return self.brave and waited > 1.5 and self._road_clear(crossing, ctx, self.gap_accept * 1.3)
        if not self.aware:
            return True
        gap = self.gap_accept * (1.3 if crossing.kind == "jaywalk" else 1.0)
        if waited > 25.0:
            gap = 1.5                                  # fed up waiting
        return self._road_clear(crossing, ctx, gap)

    @staticmethod
    def _car_on(crossing, ctx):
        """True if a car (or the robot car) is on the crossing right now."""
        for veh in ctx["vehicles"]:
            if abs(veh["x"] - crossing.center[0]) > 15 or abs(veh["y"] - crossing.center[1]) > 15:
                continue
            for cx, cy, cr in veh["circles"]:
                if crossing.contains(cx, cy, margin=cr + 0.3):
                    return True
        return False

    def _road_clear(self, crossing, ctx, gap):
        """True if no car will reach this crossing within `gap` seconds
        (cars that are stopping for this crossing don't count)."""
        for veh in ctx["vehicles"]:
            speed = math.hypot(veh["vx"], veh["vy"])
            rx, ry = crossing.center[0] - veh["x"], crossing.center[1] - veh["y"]
            if abs(rx) > 60 or abs(ry) > 60:
                continue
            h = veh["heading"]
            ahead = rx * math.cos(h) + ry * math.sin(h)
            side = -rx * math.sin(h) + ry * math.cos(h)
            if abs(side) > crossing.width / 2 + crossing.half + 2.0 and abs(side) > crossing.half + 2:
                continue                                # not on this road
            front = ahead - veh["length"] / 2
            if front < -1.0:
                continue                                # already past
            if crossing.id >= 0 and veh.get("yield") == crossing.id and speed < 4.0:
                continue                                # it is stopping for us
            if speed < 0.3:
                if front < 1.5:
                    return False                        # parked right on the crossing
                continue
            if front / speed < gap:
                return False
        return True

    def _threat(self, ctx):
        """A car that will come too close in the next few seconds, or None."""
        pvx, pvy = self.vx, self.vy
        for veh in ctx["vehicles"]:
            if abs(veh["x"] - self.x) > 30 or abs(veh["y"] - self.y) > 30:
                continue
            speed = math.hypot(veh["vx"], veh["vy"])
            if speed < 1.0:
                continue
            if veh.get("yield") is not None and veh.get("yield") >= 0 and speed < 3.0:
                continue
            rx, ry = veh["x"] - self.x, veh["y"] - self.y
            dvx, dvy = veh["vx"] - pvx, veh["vy"] - pvy
            dv2 = dvx * dvx + dvy * dvy
            t = 0.0 if dv2 < 1e-9 else max(0.0, min(2.5, -(rx * dvx + ry * dvy) / dv2))
            d = math.hypot(rx + dvx * t, ry + dvy * t)
            if d < veh["width"] / 2 + veh["length"] / 2 * 0.6 + self.radius + 0.6:
                return veh
        return None

    def state(self):
        st = super().state()
        st["aware"] = self.aware
        return st


# ======================================================================= scripted crosser
class CrossingPedestrian(Pedestrian):
    """A scripted person who crosses the robot's path right in front of it.

    They wait at `start` (at the side of the robot's path), facing across.
    They watch the robot coming along its route and step out when it is as
    many seconds away as they need to reach the meeting point `meet` (minus
    the chosen timing offset). While walking up to the meeting point they
    adjust their pace a little (as people do), so they still arrive together
    if the robot slows for a turn or a light. Then they carry on to `end`
    on the other side and stop there.

    Like a real person, they only step off the curb while the car is actually
    driving towards them: if it is stopped or crawling (at a light, in a
    queue), they wait on the sidewalk until it gets going again.
    """

    def __init__(self, ped_id, start, end, meet, speed, timing, rng, world, look=None, path=None,
                 behaviour="cross"):
        super().__init__(ped_id, start, end, speed, rng, world, look)
        self.behaviour = behaviour               # "cross" = keep walking, "stop" = stand in the lane
        self.stage = "approach"                  # approach -> (stand ->) walk on
        self.stand_since = None
        self.start = start
        self.end = end
        self.meet = meet
        self.timing = timing                     # s: + arrive after the robot, - before
        self.path = path                         # the robot's route (a roadmap.Polyline) or None
        self.meet_s = path.project(*meet)[0] if path is not None else None
        self.base_speed = speed
        self.started = False
        self.finished = False
        self.mode = "wait"
        self.yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        self.sync_body()

    def _robot_eta(self, robot):
        """Seconds until the FRONT of the robot reaches the meeting point, if it
        speeds up to its top speed and keeps it."""
        car = self.world is not None and getattr(self.world, "roadmap", None) is not None
        front = config.CAR_LENGTH / 2 if car else config.ROBOT_RADIUS
        if self.path is not None:
            d = self.meet_s - self.path.project(robot["x"], robot["y"])[0] - front
        else:
            d = math.hypot(robot["x"] - self.meet[0], robot["y"] - self.meet[1]) - front
        if d <= 0:
            return 0.0
        v0, a, vmax = math.hypot(robot["vx"], robot["vy"]), config.ROBOT_MAX_ACCEL, config.ROBOT_MAX_SPEED
        if v0 >= vmax:
            return d / vmax
        t_acc = (vmax - v0) / a
        d_acc = v0 * t_acc + 0.5 * a * t_acc * t_acc
        if d <= d_acc:
            return (-v0 + math.sqrt(v0 * v0 + 2 * a * d)) / a
        return t_acc + (d - d_acc) / vmax

    @staticmethod
    def _robot_moving(robot):
        return math.hypot(robot["vx"], robot["vy"]) >= config.CROSSER_CAR_MOVING

    def _on_sidewalk(self):
        return self.world is None or not self.world.on_road(self.x, self.y)

    def _slowest(self):
        """Slowest pace when timing the walk to meet the car: on the sidewalk they can
        dawdle, but once on the road people keep walking normally (they don't slow
        down just because the car slows down)."""
        return (0.2 if self._on_sidewalk() else config.CROSSER_ROAD_PACE) * self.base_speed

    def _car_in_the_way(self, robot):
        """True if the car is within 1 m of where this person is about to walk."""
        dx, dy = self.end[0] - self.x, self.end[1] - self.y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return False
        ux, uy = dx / n, dy / n
        for k in (0.0, 0.5, 1.0, 1.5):
            px, py = self.x + ux * k, self.y + uy * k
            for cx, cy, cr in robot["footprint"]:
                if math.hypot(px - cx, py - cy) < cr + self.radius + 1.0:
                    return True
        return False

    def _robot_passed(self, robot):
        if self.path is not None:
            return self.path.project(robot["x"], robot["y"])[0] > self.meet_s + 3.0
        return math.hypot(robot["x"] - self.meet[0], robot["y"] - self.meet[1]) > 4.0 and \
            (robot["x"] - self.meet[0]) * robot["vx"] + (robot["y"] - self.meet[1]) * robot["vy"] > 0

    def choose_velocity(self, ctx):
        robot = ctx.get("robot")
        to_meet = math.hypot(self.meet[0] - self.x, self.meet[1] - self.y)
        if self.behaviour == "stop" and self.started and self.stage != "on":
            # walk to the spot in the lane, stand there until the car has gone by, then walk on
            if self.stage == "approach":
                step = self.base_speed * config.CONTROL_DT
                if to_meet <= step:
                    self.x, self.y = self.meet
                    self.stage, self.stand_since = "stand", ctx["t"]
            if self.stage == "stand":
                self.vx = self.vy = 0.0
                self.mode = "stand"
                if robot is None or self._robot_passed(robot) or \
                        ctx["t"] - self.stand_since > config.CROSSER_STAND_MAX:
                    self.stage = "on"
                return
            if robot is not None and not self._robot_moving(robot) and self._on_sidewalk():
                self.vx = self.vy = 0.0                  # car stopped: wait at the curb
                self.mode = "wait"
                return
            dx, dy = self.meet[0] - self.x, self.meet[1] - self.y
            speed = self.base_speed
            if robot is not None and to_meet > 0.3 and self._robot_moving(robot):
                eta = self._robot_eta(robot) + self.timing
                if eta > 0.2:
                    speed = min(1.5 * self.base_speed, max(self._slowest(), to_meet / eta))
            self.mode = "cross"
            self.vx, self.vy = dx / max(to_meet, 1e-6) * speed, dy / max(to_meet, 1e-6) * speed
            return
        ahead = (self.meet[0] - self.x) * (self.end[0] - self.start[0]) + \
                (self.meet[1] - self.y) * (self.end[1] - self.start[1]) > 0
        if not self.started and robot is not None:
            if self._robot_moving(robot) and self._robot_eta(robot) + self.timing <= to_meet / self.base_speed:
                self.started = True
        waiting = (self.started and not self.finished and ahead and robot is not None
                   and not self._robot_moving(robot) and self._on_sidewalk())
        if self.started and not self.finished and self.stage == "on" and robot is not None \
                and self._car_in_the_way(robot):
            waiting = True                     # walking on after standing: don't walk into the car
        if self.started and not self.finished and not waiting:
            self.mode = "cross"
            speed = self.base_speed
            if robot is not None and ahead and to_meet > 0.3 and self._robot_moving(robot):
                eta = self._robot_eta(robot) + self.timing   # (car stopped: just walk normally)
                if eta > 0.2:                         # keep pace with the robot (within reason)
                    speed = min(1.5 * self.base_speed, max(self._slowest(), to_meet / eta))
            self.preferred_speed = speed
            dx, dy = self.end[0] - self.x, self.end[1] - self.y
            dist = math.hypot(dx, dy)
            step = speed * config.CONTROL_DT
            if dist <= step:                       # last step: land exactly on the end point
                self.vx, self.vy = dx / config.CONTROL_DT, dy / config.CONTROL_DT
                self.finished = True
            else:
                self.vx = dx / dist * speed
                self.vy = dy / dist * speed
        else:
            self.mode = "wait" if not self.started else "pause"
            self.vx = self.vy = 0.0


# ======================================================================= helpers
def separate_pedestrians(peds):
    """People can't walk through each other: push apart any overlapping pair,
    then make sure nobody was pushed into a wall."""
    n = len(peds)
    for i in range(n):
        a = peds[i]
        for j in range(i + 1, n):
            b = peds[j]
            dx, dy = b.x - a.x, b.y - a.y
            lim = a.radius + b.radius
            if abs(dx) > lim or abs(dy) > lim:
                continue
            dist = math.hypot(dx, dy)
            overlap = lim - dist
            if overlap > 0:
                if dist < 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
                shift = overlap / 2
                a.x -= dx / dist * shift
                a.y -= dy / dist * shift
                b.x += dx / dist * shift
                b.y += dy / dist * shift
                a.stay_out_of_walls()
                b.stay_out_of_walls()


def keep_out_of_cars(peds, vehicles):
    """Nobody overlaps a traffic car: push people out of any car they touch.
    Returns the set of (car id, person id) pairs that were touching."""
    touching = set()
    for veh in vehicles:
        if not veh.active:
            continue
        circles = None
        for ped in peds:
            if abs(ped.x - veh.x) > veh.length or abs(ped.y - veh.y) > veh.length:
                continue
            circles = circles or veh.circles()
            for cx, cy, cr in circles:
                dx, dy = ped.x - cx, ped.y - cy
                d = math.hypot(dx, dy)
                lim = cr + ped.radius
                if d < lim:
                    if d < 1e-9:
                        dx, dy, d = 1.0, 0.0, 1.0
                    ped.x, ped.y = cx + dx / d * lim, cy + dy / d * lim
                    ped.stay_out_of_walls()
                    touching.add((veh.id, ped.ped_id))
    return touching


def _pick_kind(rng):
    r = rng.random()
    for kind, spec in config.PED_KINDS.items():
        r -= spec["share"]
        if r <= 0:
            return kind
    return "adult"


def _group_size(rng):
    r = rng.random()
    for size, share in config.PED_GROUP_SIZES.items():
        r -= share
        if r <= 0:
            return size
    return 1


def spawn_pedestrians(n, rng, world, robot_start, robot_goal, existing=(), look_rng=None):
    """Create n pedestrians at free, non-overlapping spots away from the
    robot's start and goal and from `existing` pedestrians. IDs continue
    after the existing ones. Street layouts get StreetPedestrians (in groups),
    the indoor layouts simple walkers."""
    look_rng = look_rng if look_rng is not None else rng
    clear = config.PED_SPAWN_CLEARANCE
    first_id = len(existing)
    peds = []
    if world.roadmap is None:
        for i in range(n):
            avoid = [(robot_start[0], robot_start[1], clear), (robot_goal[0], robot_goal[1], clear)]
            avoid += [(q.x, q.y, 2 * config.PED_RADIUS + 0.4) for q in list(existing) + peds]
            start = world.random_walk_point(rng, config.PED_RADIUS + 0.1, avoid)
            goal = world.random_walk_point(rng, config.PED_RADIUS + 0.3, [(start[0], start[1], 3.0)])
            speed = rng.uniform(config.PED_SPEED_MIN, config.PED_SPEED_MAX)
            look = visuals.random_person_look(look_rng, "adult", look_rng.uniform(1.6, 1.85))
            peds.append(Pedestrian(first_id + i, start, goal, speed, rng, world, look))
        return peds

    while len(peds) < n:
        size = min(_group_size(rng), n - len(peds))
        avoid = [(robot_start[0], robot_start[1], clear + 2), (robot_goal[0], robot_goal[1], clear)]
        avoid += [(q.x, q.y, 2 * config.PED_RADIUS + 0.6) for q in list(existing) + peds]
        start = world.random_walk_point(rng, 0.6 if size > 1 else 0.35, avoid)
        leader = None
        for k in range(size):
            kind = _pick_kind(rng)
            if k == 0 and kind == "child":
                kind = "adult"                       # children don't walk around alone
            spec = config.PED_KINDS[kind]
            speed = rng.uniform(*spec["speed"])
            height = look_rng.uniform(*spec["height"])
            look = visuals.random_person_look(look_rng, kind, height)
            if k == 0:
                pos = start
            else:
                ang = rng.uniform(0, 2 * math.pi)
                pos = (start[0] + math.cos(ang) * 0.7 * k, start[1] + math.sin(ang) * 0.7 * k)
                if not world.is_free(pos[0], pos[1], spec["radius"] + 0.05):
                    pos = (start[0] + 0.3 * k, start[1])
            ped = StreetPedestrian(first_id + len(peds), pos, speed, rng, world, kind, look,
                                   leader=leader, slot=k)
            if k == 0:
                leader = ped
                if rng.random() < 0.25:              # some people start off standing around
                    ped.pause_left = rng.uniform(0.5, 6.0)
            else:
                ped.preferred_speed = max(ped.preferred_speed, leader.preferred_speed)  # keep up
            peds.append(ped)
    for ped in peds:                                 # face the way they will walk
        if ped.leader is None:
            ped.plan_new_trip()
            x, y = ped.steps[0][:2]
            ped.yaw = math.atan2(y - ped.y, x - ped.x)
        else:
            ped.yaw = ped.leader.yaw
        ped.sync_body()
    return peds
