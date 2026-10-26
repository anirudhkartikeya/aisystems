"""simulation.py - ties the world, robot, pedestrians and traffic together.

Use it like this (the planner will do exactly this later):

    sim = Simulation(gui=True)
    state = sim.reset(seed=0)
    while True:
        vx, vy = my_controller(state)
        state, info = sim.step(vx, vy)
        if info["done"]:
            break

The same seed always gives the same pedestrians, cars, goals and pauses, so
the reactive and predictive brains can be tested on identical scenarios.

Simulation(robot=False) leaves the robot out (parked far away): nothing ends
the trial early, which is what the data-collection scripts use.
"""
import math
import time

import numpy as np
import pybullet as p

import config
from pedestrians import keep_out_of_cars, separate_pedestrians
from robot import Robot
from scenarios import make_pedestrians, n_cars, start_and_goal
from traffic import TrafficManager
import visuals
from world import World, connect, top_down_camera

PARKED = (-500.0, -500.0)        # where the robot waits when it is switched off


class Simulation:
    def __init__(self, gui=True, layout=None, scenario=None, robot=True, lighting=None):
        self.gui = gui
        self.layout = layout or config.LAYOUT
        self.scenario = scenario or config.SCENARIO
        self.robot_enabled = robot
        self.lighting = lighting or config.LIGHTING
        if self.layout in ("street", "town"):
            config.use_street_settings()           # car speed, acceleration, camera height
        self.start, self.goal = start_and_goal(self.layout, self.scenario)
        connect(gui)
        if gui:
            # don't redraw while hundreds of objects are being created (much faster start)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if config.GUI_SHADOWS else 0)
            self.set_gui_lighting(self.lighting)
        self.world = World(self.layout, self.goal if robot else PARKED)
        self.robot = Robot(look="car" if self.world.roadmap is not None else "robot")
        self.peds = []
        self.traffic = None
        self.follow = gui and self.world.roadmap is not None and config.FOLLOW_CAMERA and robot
        if gui and not self.follow:
            if self.world.roadmap is not None:
                x0, x1, y0, y1 = self.world.bounds
                top_down_camera(max(x1 - x0, y1 - y0) * 0.9, ((x0 + x1) / 2, (y0 + y1) / 2, 0))
            else:
                top_down_camera(12)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    @staticmethod
    def set_gui_lighting(name):
        d = config.LIGHTING_PRESETS[name]["dir"]
        try:
            p.configureDebugVisualizer(lightPosition=[d[0] * 100, d[1] * 100, d[2] * 100])
            if config.GUI_SHADOWS:                   # sharper shadows over a street-sized area
                p.configureDebugVisualizer(shadowMapResolution=4096, shadowMapWorldSize=60)
        except (TypeError, p.error):
            pass                                     # older PyBullet: keep its default light

    # ---------- start a new trial ----------
    def reset(self, seed=None):
        seed = config.SEED if seed is None else seed
        self.seed = seed
        # separate random streams: people, traffic, looks. Changing how
        # things look never changes what they do.
        self.rng = np.random.default_rng(seed)
        self.rng_traffic = np.random.default_rng([seed, 1])
        self.rng_looks = np.random.default_rng([seed, 2])
        for ped in self.peds:
            ped.remove()
        if self.traffic is not None:
            self.traffic.remove()
            self.traffic = None
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if self.robot_enabled:
            self.robot.reset(self.start, facing=self.goal)
        else:
            self.robot.reset(PARKED)
        self.peds = make_pedestrians(self.scenario, self.world, self.rng, self.start, self.goal, self.rng_looks)
        cars = n_cars(self.scenario, self.layout) if self.world.roadmap is not None else 0
        if cars > 0:
            avoid = [(self.start[0], self.start[1], 25.0)] if self.robot_enabled else []
            avoid += [(ped.meet[0], ped.meet[1], 10.0) for ped in self.peds if hasattr(ped, "meet")]
            self.traffic = TrafficManager(self.world.roadmap, self.rng_traffic, self.rng_looks, cars, avoid)
        for sig in self.signals:
            sig.update(0.0)

        self.t = 0.0
        self.done = False
        self.result = None            # "goal", "collision_pedestrian", "collision_vehicle", "collision_obstacle", "timeout"
        self.collisions = 0
        self.min_gap = math.inf       # closest robot-pedestrian surface gap seen
        self.min_vehicle_gap = math.inf
        self.path_length = 0.0
        self.traffic_ped_contacts = 0  # traffic cars touching people (should stay 0 or close)
        self._touching = set()        # pedestrians currently overlapping the robot
        self._npc_touching = set()
        self._clock_start = None      # for keeping the GUI in real time
        self._light_states = {}
        # what the robot should do for each scripted person, and what it did
        self.crosser_log = [dict(ped.plan, ped_id=ped.ped_id, min_gap=math.inf, action=None, stopped=False)
                            for ped in self.peds if getattr(ped, "plan", None)]

        self.sync_visuals()
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.removeAllUserDebugItems()
            self._heading_line = -1
            self._status_text = -1
            self._draw_crossing_marks()
            self._cam = None
            if self.follow:
                self._follow_camera(0.0)
        return self.get_state()

    def close(self):
        """Shut PyBullet down (one Simulation per process at a time)."""
        if p.isConnected():
            p.disconnect()
        visuals.forget_shapes()

    @property
    def signals(self):
        return self.world.roadmap.signals if self.world.roadmap is not None else []

    # ---------- what the brains will see ----------
    def robot_snapshot(self):
        st = self.robot.state()
        return {"id": -1, "kind": "robot", "x": st["x"], "y": st["y"], "vx": st["vx"], "vy": st["vy"],
                "heading": st["heading"], "length": config.CAR_LENGTH if self.world.roadmap else 2 * config.ROBOT_RADIUS,
                "width": config.CAR_WIDTH if self.world.roadmap else 2 * config.ROBOT_RADIUS,
                "circles": st["footprint"], "is_robot": True, "yield": None}

    def get_state(self):
        return {
            "time": self.t,
            "robot": self.robot.state() if self.robot_enabled else None,
            "goal": self.goal,
            "pedestrians": [ped.state() for ped in self.peds],
            "vehicles": self.traffic.snapshots() if self.traffic else [],
            "lights": [{"junction": (s.node.x, s.node.y), "green_axis": s.axis, "color": s.color}
                       for s in self.signals],
        }

    def info(self):
        return {
            "time": self.t,
            "done": self.done,
            "result": self.result,
            "collisions": self.collisions,
            "min_gap": self.min_gap,
            "min_vehicle_gap": self.min_vehicle_gap,
            "path_length": self.path_length,
            "traffic_ped_contacts": self.traffic_ped_contacts,
            "crossers": [{k: c[k] for k in ("name", "expected", "action", "min_gap", "behaviour", "side",
                                            "contact", "zebra")} for c in self.crosser_log],
            "seed": self.seed,
            "layout": self.layout,
            "scenario": self.scenario,
        }

    def _context(self):
        vehicles = self.traffic.snapshots() if self.traffic else []
        if self.robot_enabled:
            vehicles.append(self.robot_snapshot())
        return {"t": self.t, "peds": self.peds, "vehicles": vehicles,
                "robot": self.robot.state() if self.robot_enabled else None, "world": self.world}

    # ---------- advance one control step (0.1 s) ----------
    def step(self, cmd_vx=0.0, cmd_vy=0.0):
        if self.done:
            return self.get_state(), self.info()

        if self.robot_enabled:
            self.robot.set_command(cmd_vx, cmd_vy)
        for sig in self.signals:
            sig.update(self.t)
        ctx = self._context()
        if self.traffic:
            self.traffic.plan(ctx)
        for ped in self.peds:
            ped.choose_velocity(ctx)

        n = config.PHYSICS_STEPS_PER_CONTROL
        sub_dt = config.CONTROL_DT / n
        start_xy = (self.robot.x, self.robot.y)
        self._npc_now = set()
        ped_start = [(ped.x, ped.y) for ped in self.peds]
        for i in range(n):
            prev = (self.robot.x, self.robot.y)
            if self.robot_enabled:
                self.robot.move(sub_dt)
            if self.traffic:
                self.traffic.move(sub_dt)
            for ped in self.peds:
                ped.move(sub_dt)
            separate_pedestrians(self.peds)
            if self.traffic:
                self._npc_now |= keep_out_of_cars(self.peds, self.traffic.vehicles)
            self.path_length += math.hypot(self.robot.x - prev[0], self.robot.y - prev[1])
            self.t += sub_dt
            if self.robot_enabled:
                self._check_events()
            elif self.t >= config.MAX_TIME - 1e-9:
                self._finish("timeout")
            if self.gui:
                if i % config.GUI_AGENT_EVERY == 0 or i == n - 1:
                    self.sync_visuals()
                if self.follow and i % config.GUI_CAMERA_EVERY == 0:
                    self.robot.sync_body()               # car and camera move together: no judder
                    self._follow_camera(sub_dt * config.GUI_CAMERA_EVERY)
                self._keep_real_time()
            if self.done:
                break
        self._check_traffic_contacts()
        if self.robot_enabled:
            self._check_crossers()

        if self.gui:
            self._draw(start_xy, ped_start)
        return self.get_state(), self.info()

    def sync_visuals(self):
        """Move every drawn body to where its agent is now. The GUI does this
        ~30 times a second; headless runs only before a camera picture."""
        near = None
        if self.gui and self.follow:                 # GUI following the car: only redraw what is in view
            near = (self.robot.x, self.robot.y, 45.0)
        if self.robot_enabled:
            self.robot.sync_body()
        for ped in self.peds:
            if near is None or (abs(ped.x - near[0]) < near[2] and abs(ped.y - near[1]) < near[2]):
                ped.sync_body()
        if self.traffic:
            self.traffic.sync_bodies(near)
        figs = self.world.light_figures
        for lane, fig in figs["vehicle"]:
            sig, arm = lane.signal
            fig.show(sig.vehicle_color(arm))
        for crossing, fig in figs["pedestrian"]:
            fig.show(crossing.node.signal.ped_color(crossing))

    # ---------- collisions, goal, timeout ----------
    def _check_events(self):
        rx, ry = self.robot.x, self.robot.y
        footprint = self.robot.footprint()          # robot as circles (1 for robot, 3 for car)

        for ped in self.peds:
            if abs(ped.x - rx) > 6 or abs(ped.y - ry) > 6:
                continue
            gap = min(math.hypot(cx - ped.x, cy - ped.y) - cr for cx, cy, cr in footprint) - ped.radius
            self.min_gap = min(self.min_gap, gap)
            if gap < 0:
                if ped.ped_id not in self._touching:     # count each contact once
                    self._touching.add(ped.ped_id)
                    self.collisions += 1
                    if config.STOP_ON_COLLISION:
                        self._finish("collision_pedestrian")
                        return
            else:
                self._touching.discard(ped.ped_id)

        if self.traffic:
            for veh in self.traffic.vehicles:
                if not veh.active or abs(veh.x - rx) > 12 or abs(veh.y - ry) > 12:
                    continue
                gap = min(math.hypot(ax - bx, ay - by) - ar - br
                          for ax, ay, ar in footprint for bx, by, br in veh.circles())
                self.min_vehicle_gap = min(self.min_vehicle_gap, gap)
                if gap < 0:
                    self.collisions += 1
                    if config.STOP_ON_COLLISION:
                        self._finish("collision_vehicle")
                        return

        if any(self.world.clearance(cx, cy) < cr for cx, cy, cr in footprint):
            self.collisions += 1
            self._finish("collision_obstacle")
            return

        if math.hypot(rx - self.goal[0], ry - self.goal[1]) < config.GOAL_TOLERANCE:
            self._finish("goal")
            return

        if self.t >= config.MAX_TIME - 1e-9:
            self._finish("timeout")

    def _check_traffic_contacts(self):
        """Count traffic cars touching people (people are pushed back out at
        once; a high count is a sign the behaviour models need tuning)."""
        if not self.traffic:
            return
        self.traffic_ped_contacts += len(self._npc_now - self._npc_touching)
        self._npc_touching = self._npc_now

    def _check_crossers(self):
        """For every scripted person: the closest the robot came, and what it
        did when it reached them - "stop", "swerve_left", "swerve_right" or
        "straight" (lateral move from its route > 0.9 m counts as a swerve)."""
        if not self.crosser_log:
            return
        by_id = {ped.ped_id: ped for ped in self.peds}
        speed = math.hypot(self.robot.vx, self.robot.vy)
        foot = self.robot.footprint()
        for log in self.crosser_log:
            ped = by_id[log["ped_id"]]
            gap = min(math.hypot(cx - ped.x, cy - ped.y) - cr for cx, cy, cr in foot) - ped.radius
            log["min_gap"] = min(log["min_gap"], gap)
            if log["action"] is not None or ped.path is None:
                continue
            s_robot, _, lateral = ped.path.project(self.robot.x, self.robot.y)
            ahead = ped.meet_s - s_robot
            if speed < 0.3 and 0 < ahead < 12:
                log["stopped"] = True
            if ahead < 0:                                  # the robot has reached this person
                if log["stopped"]:
                    log["action"] = "stop"
                elif lateral > 0.9:
                    log["action"] = "swerve_left"
                elif lateral < -0.9:
                    log["action"] = "swerve_right"
                else:
                    log["action"] = "straight"

    def _finish(self, result):
        self.done = True
        self.result = result
        self.robot.vx = self.robot.vy = 0.0

    # ---------- helpers for simple controllers ----------
    def robot_lane(self):
        """The lane the robot is driving in (and how far along it), or (None, 0)."""
        if self.world.roadmap is None:
            return None, 0.0
        best = (None, 0.0, 1.6)
        for lane in self.world.roadmap.lanes:
            s, d, _ = lane.poly.project(self.robot.x, self.robot.y)
            hdg = lane.poly.point_at(s)[2]
            if d < best[2] and abs((hdg - self.robot.heading + math.pi) % (2 * math.pi) - math.pi) < 0.7:
                best = (lane, s, d)
        return best[0], best[1]

    def oncoming_before_left_turn(self):
        """Turning left: distance to the stop line if oncoming traffic is coming
        through the junction (wait for it), else None."""
        lane, s = self.robot_lane()
        if lane is None or self.traffic is None or "left" not in lane.turns:
            return None
        to_line = lane.length - s - config.CAR_LENGTH / 2
        if not -0.5 < to_line < 25.0:
            return None
        node, arm = lane.end_node, lane.arm_at_end
        opposite = [l for l in self.world.roadmap.lanes if l.end_node is node and
                    l.arm_at_end == (-arm[0], -arm[1])]
        for v in self.traffic.vehicles:
            if not v.active:
                continue
            for op in opposite:
                to_junction = op.length - v.s if v.element is op else None
                if to_junction is not None and to_junction > 1.0 and op.signal is not None \
                        and op.signal[0].vehicle_color(op.signal[1]) == "red":
                    continue                     # they have red and will stop at their line
                # coming towards the junction, or waiting at the line and about to go
                if to_junction is not None and (to_junction < 12.0 or (v.v > 0.5 and to_junction < 40.0)):
                    return max(0.0, to_line)
                if getattr(v.element, "lane_in", None) is op and v.element.turn != "left":
                    return max(0.0, to_line)
        return None

    def light_ahead(self, max_dist=30.0):
        """Distance to the stop line if the robot's lane has a red (or yellow)
        light ahead within max_dist, else None."""
        if self.world.roadmap is None:
            return None
        for lane in self.world.roadmap.lanes:
            if lane.signal is None:
                continue
            s, d, _ = lane.poly.project(self.robot.x, self.robot.y)
            hdg = lane.poly.point_at(s)[2]
            if d > 1.5 or abs((hdg - self.robot.heading + math.pi) % (2 * math.pi) - math.pi) > 0.7:
                continue
            to_line = lane.length - s - config.CAR_LENGTH / 2
            if -0.5 < to_line < max_dist:
                sig, arm = lane.signal
                if sig.vehicle_color(arm) in ("red", "yellow"):
                    return max(0.0, to_line)
        return None

    # ---------- pacing (GUI only) ----------
    def _keep_real_time(self):
        """Wait just long enough that simulated time keeps pace with the clock
        (times SIM_SPEED). Unlike sleeping a fixed amount every step, this does
        not fall behind when the computer is busy drawing."""
        if not config.REAL_TIME_GUI:
            return
        now = time.perf_counter()
        if self._clock_start is None:
            self._clock_start, self._sim_start = now, self.t
        ahead = (self.t - self._sim_start) / config.SIM_SPEED - (now - self._clock_start)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.25:                        # far behind (e.g. window dragged): resync
            self._clock_start, self._sim_start = now, self.t

    # ---------- drawing (GUI only) ----------
    def _draw_crossing_marks(self):
        """Crossing scenario: a thin line for each scripted crosser's walk,
        and a yellow X where it meets the robot's path."""
        for ped in self.peds:
            if not hasattr(ped, "meet"):
                continue
            z = 0.05
            p.addUserDebugLine([ped.start[0], ped.start[1], z], [ped.end[0], ped.end[1], z],
                               ped.color[:3], lineWidth=1)
            mx, my, s = ped.meet[0], ped.meet[1], 0.25
            for dx, dy in ((s, s), (s, -s)):
                p.addUserDebugLine([mx - dx, my - dy, z], [mx + dx, my + dy, z],
                                   [0.95, 0.75, 0.0], lineWidth=4)

    def _follow_camera(self, dt):
        """Street/town: a camera that glides after the car. "chase" sits behind
        the car and turns with it (smoothly, through the left turn and the
        curve); "side" looks at the road from the side. dt = 0 jumps straight
        to the car (at the start of a trial)."""
        h = self.robot.heading
        ahead = 4.0
        tx, ty = self.robot.x + math.cos(h) * ahead, self.robot.y + math.sin(h) * ahead
        yaw = math.degrees(h) - 90.0 if config.CAMERA_VIEW == "chase" else 0.0
        if self._cam is None or dt <= 0:
            self._cam = [tx, ty, yaw]
        else:
            a = 1.0 - math.exp(-dt / max(config.CAMERA_SMOOTH, 1e-3))
            cx, cy, cyaw = self._cam
            dyaw = (yaw - cyaw + 180.0) % 360.0 - 180.0          # turn the short way round
            self._cam = [cx + (tx - cx) * a, cy + (ty - cy) * a, cyaw + dyaw * a * 0.7]
        pitch = config.CAMERA_PITCH if config.CAMERA_VIEW == "chase" else -55.0
        dist = config.CAMERA_DISTANCE if config.CAMERA_VIEW == "chase" else 20.0
        p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=self._cam[2], cameraPitch=pitch,
                                     cameraTargetPosition=[self._cam[0], self._cam[1], 0.5])

    def _draw(self, robot_prev, ped_prev):
        rx, ry = self.robot.x, self.robot.y
        z = 0.06
        if self.robot_enabled:
            # robot trail stays on screen for the whole trial
            p.addUserDebugLine([robot_prev[0], robot_prev[1], z], [rx, ry, z],
                               [0.1, 0.35, 0.95], lineWidth=3)
        if config.SHOW_PED_TRAILS:
            for ped, (px, py) in zip(self.peds, ped_prev):
                p.addUserDebugLine([px, py, z], [ped.x, ped.y, z], ped.color[:3], lineWidth=2, lifeTime=3)
        if self.robot_enabled and self.world.roadmap is None:
            h = self.robot.heading
            self._heading_line = p.addUserDebugLine(
                [rx, ry, 0.35], [rx + 0.6 * math.cos(h), ry + 0.6 * math.sin(h), 0.35],
                [0, 0, 0], lineWidth=3, replaceItemUniqueId=self._heading_line)
        speed = math.hypot(self.robot.vx, self.robot.vy)
        text = f"t = {self.t:4.1f} s"
        if self.robot_enabled:
            text += f"   speed = {speed:3.1f} m/s   min gap = {self.min_gap:4.2f} m"
            if getattr(self, "drive_status", None):
                text += f"   [{self.drive_status}]"
        if self.done:
            text += f"   RESULT: {self.result}"
        if self.follow:
            text_pos = [rx, ry, 3.6]                     # floats above the car
        elif self.world.roadmap is not None:
            text_pos = [self.world.bounds[0], self.world.bounds[3] + 2, 0.5]
        else:
            text_pos = [-config.ROOM_SIZE / 2, config.ROOM_SIZE / 2 + 0.6, 0.1]
        self._status_text = p.addUserDebugText(
            text, text_pos, [0, 0, 0], textSize=1.3, replaceItemUniqueId=self._status_text)
