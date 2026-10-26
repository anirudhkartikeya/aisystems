"""robot.py - the robot: either the small blue robot or, on the street, a car.

It is kinematic: we set its position ourselves instead of using motors.
That keeps Phase 1 simple and fully repeatable. Speed and acceleration
limits keep the motion realistic (no instant stops).

For collisions the robot is a set of circles (its "footprint"):
  small robot - one circle of ROBOT_RADIUS
  car         - three overlapping circles along its length, a standard way
                planners model a car (long and narrow, like the real thing)
"""
import math

import pybullet as p

import config
import visuals

ROBOT_COLOR = [0.1, 0.35, 0.95, 1]
CAR_COLOR = [0.10, 0.35, 0.90, 1]


class Robot:
    def __init__(self, look="robot"):
        self.look = look
        self.figure = None
        self.body_id = None
        if look == "car":
            r = config.CAR_WIDTH / 2 + 0.05
            off = config.CAR_LENGTH / 2 - r
            self._circles = [(-off, r), (0.0, r), (off, r)]   # (offset along heading, radius)
            self.front = config.CAR_LENGTH / 2                # centre to front bumper
            self.figure = visuals.make_car(CAR_COLOR)
        else:
            r, h = config.ROBOT_RADIUS, config.ROBOT_HEIGHT
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=ROBOT_COLOR)
            self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                             baseVisualShapeIndex=vis,
                                             basePosition=[0, 0, h / 2])
            self._circles = [(0.0, r)]
            self.front = r
        self.radius = max(abs(o) + r for o, r in self._circles)   # radius of a circle around it all
        self.reset(config.ROBOT_START, config.GOAL)

    def reset(self, start, facing=None):
        """Place the robot at `start`, stopped, facing `facing` (a point) if given."""
        self.x, self.y = start
        self.vx = self.vy = 0.0
        if facing is not None:
            self.heading = math.atan2(facing[1] - self.y, facing[0] - self.x)
        else:
            self.heading = 0.0
        self._sync()

    def set_command(self, vx, vy):
        """Ask for a velocity (m/s). It is clipped to the max speed, and the
        change from the current velocity is clipped to the max acceleration."""
        speed = math.hypot(vx, vy)
        if speed > config.ROBOT_MAX_SPEED:
            vx, vy = vx / speed * config.ROBOT_MAX_SPEED, vy / speed * config.ROBOT_MAX_SPEED
        dvx, dvy = vx - self.vx, vy - self.vy
        dv = math.hypot(dvx, dvy)
        max_dv = config.ROBOT_MAX_ACCEL * config.CONTROL_DT
        if dv > max_dv:
            dvx, dvy = dvx / dv * max_dv, dvy / dv * max_dv
        self.vx += dvx
        self.vy += dvy
        if math.hypot(self.vx, self.vy) > 0.05:
            self.heading = math.atan2(self.vy, self.vx)

    def move(self, dt):
        """Advance the position by one small time step (the drawing is
        updated separately by sync_body, only when something is drawn)."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def sync_body(self):
        self._sync()

    def footprint(self):
        """The circles (x, y, r) that make up the robot right now."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        return [(self.x + c * o, self.y + s * o, r) for o, r in self._circles]

    def _sync(self):
        if self.figure is not None:
            self.figure.set_pose(self.x, self.y, self.heading)
        else:
            orn = p.getQuaternionFromEuler([0, 0, self.heading])
            p.resetBasePositionAndOrientation(self.body_id,
                                              [self.x, self.y, config.ROBOT_HEIGHT / 2], orn)

    def state(self):
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
                "heading": self.heading, "footprint": self.footprint()}
