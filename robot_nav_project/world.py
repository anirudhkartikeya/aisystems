"""world.py - the static part of the simulator: ground, walls, roads,
buildings, street furniture, the goal marker.

It also answers geometry questions ("how far is this point from the nearest
wall or obstacle?") that the pedestrians, collision checks and, later, the
planner all use. That geometry only ever uses plain boxes and circles,
whatever the objects look like on screen. A grid makes those questions fast
even with hundreds of objects in the town.

Layouts:
  town, street          - built from a road map (roadmap.py): lanes, junctions,
                          crossings, sidewalks, buildings, parked cars...
  road, room, doorway   - the small original layouts
"""
import math

import pybullet as p
import pybullet_data

import config
import visuals
from roadmap import RoadMap

WALL_COLOR = [0.75, 0.75, 0.75, 1]
GOAL_COLOR = [0.0, 0.85, 0.2, 1]
LAYOUTS = ("town", "street", "road", "room", "doorway")
GRID = 4.0          # m, cell size of the lookup grid
REACH = 3.0         # m, obstacles farther than this from a query point are ignored


def connect(gui=True):
    """Start PyBullet. gui=True opens a window, gui=False runs headless (fast)."""
    if gui:
        r, g, b = config.GUI_SKY_COLOR
        cid = p.connect(p.GUI, options=f"--background_color_red={r} --background_color_green={g} "
                                       f"--background_color_blue={b}")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)   # hide the side panels
    else:
        cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / config.PHYSICS_HZ)
    return cid


def top_down_camera(distance=12, target=(0, 0, 0)):
    """Overhead view of the whole area (GUI only)."""
    p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=0,
                                 cameraPitch=-89.9, cameraTargetPosition=list(target))


class World:
    """Builds the scene and keeps a 2D description of every static shape."""

    def __init__(self, layout=None, goal=None):
        self.layout = layout or config.LAYOUT
        if self.layout not in LAYOUTS:
            raise ValueError(f"Unknown layout '{self.layout}', use one of {LAYOUTS}")
        self.boxes = []     # (cx, cy, hx, hy): axis-aligned rectangles, for geometry
        self.circles = []   # (cx, cy, r): circles, for geometry
        self.roadmap = None
        self.light_figures = {"vehicle": [], "pedestrian": []}
        self.parked_car_ids = []   # bodies of parked cars (camera labels can use them)

        if self.layout in ("town", "street"):
            self.roadmap = RoadMap(self.layout)
            self.bounds = self.roadmap.bounds
            self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.1])   # 10 cm under the street
            self.light_figures = visuals.make_roadmap(self.roadmap)
            obstacles = self.roadmap.obstacles
        else:
            h = config.ROOM_SIZE / 2
            self.bounds = (-h, h, -h, h)
            self.plane_id = p.loadURDF("plane.urdf")
            self._build_outer_walls()
            if self.layout == "doorway":
                self._build_partition()
            if self.layout == "road":
                visuals.make_road(h)
                obstacles = config.ROAD_OBSTACLES
            else:
                obstacles = config.OBSTACLES
        self.half = config.ROOM_SIZE / 2
        self.half_x = max(abs(self.bounds[0]), abs(self.bounds[1]))
        self.half_y = max(abs(self.bounds[2]), abs(self.bounds[3]))
        self._build_obstacles(obstacles)
        self._build_grid()
        self.goal_id = self._make_goal_marker(goal or config.GOAL)

    # ---------- building ----------
    def _add_wall(self, cx, cy, hx, hy):
        height = config.WALL_HEIGHT
        half_extents = [hx, hy, height / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=WALL_COLOR)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis, basePosition=[cx, cy, height / 2])
        self.boxes.append((cx, cy, hx, hy))

    def _build_outer_walls(self):
        h, t = config.ROOM_SIZE / 2, config.WALL_THICKNESS
        c = h + t / 2          # wall centre line, so the inside is exactly ROOM_SIZE
        long_half = h + t      # walls overlap at the corners
        self._add_wall(0, c, long_half, t / 2)    # north
        self._add_wall(0, -c, long_half, t / 2)   # south
        self._add_wall(c, 0, t / 2, long_half)    # east
        self._add_wall(-c, 0, t / 2, long_half)   # west

    def _build_partition(self):
        """A wall along y = 0 with a door gap centred at DOOR_X."""
        h, t = config.ROOM_SIZE / 2, config.WALL_THICKNESS
        left_end = config.DOOR_X - config.DOOR_WIDTH / 2
        right_start = config.DOOR_X + config.DOOR_WIDTH / 2
        self._add_wall((-h + left_end) / 2, 0, (left_end + h) / 2, t / 2)
        self._add_wall((right_start + h) / 2, 0, (h - right_start) / 2, t / 2)

    def _build_obstacles(self, obstacles):
        for ob in obstacles:
            cx, cy = ob["center"]
            if ob["type"] == "cylinder":
                self.circles.append((cx, cy, ob["radius"]))
            elif ob["type"] == "box":
                hx, hy = ob["half_extents"]
                self.boxes.append((cx, cy, hx, hy))
            else:
                raise ValueError(f"Unknown obstacle type {ob['type']}")
            group = visuals.make_obstacle(ob)
            if ob.get("look") == "parked_car":
                self.parked_car_ids.append(group.body_ids)

    def _build_grid(self):
        """Put every box and circle into the grid cells it touches."""
        self._grid = {}
        items = [("b", i, cx - hx, cy - hy, cx + hx, cy + hy) for i, (cx, cy, hx, hy) in enumerate(self.boxes)]
        items += [("c", i, cx - r, cy - r, cx + r, cy + r) for i, (cx, cy, r) in enumerate(self.circles)]
        for kind, i, x0, y0, x1, y1 in items:
            for gx in range(int(math.floor(x0 / GRID)), int(math.floor(x1 / GRID)) + 1):
                for gy in range(int(math.floor(y0 / GRID)), int(math.floor(y1 / GRID)) + 1):
                    self._grid.setdefault((gx, gy), []).append((kind, i))

    def _make_goal_marker(self, position):
        """Bright green disc. Visual only: nothing collides with it."""
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=config.GOAL_TOLERANCE,
                                  length=0.02, rgbaColor=GOAL_COLOR)
        z = 0.035 if self.roadmap else 0.03          # just above the road paint
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                 basePosition=[position[0], position[1], z])

    # ---------- geometry ----------
    def closest_static_points(self, x, y, reach=REACH):
        """For every wall and obstacle within `reach`: (px, py, dist), the
        closest point on its surface to (x, y) and the distance to it.
        dist <= 0 means inside."""
        seen = set()
        out = []
        for gx in range(int(math.floor((x - reach) / GRID)), int(math.floor((x + reach) / GRID)) + 1):
            for gy in range(int(math.floor((y - reach) / GRID)), int(math.floor((y + reach) / GRID)) + 1):
                for item in self._grid.get((gx, gy), ()):
                    if item in seen:
                        continue
                    seen.add(item)
                    kind, i = item
                    if kind == "b":
                        cx, cy, hx, hy = self.boxes[i]
                        px = min(max(x, cx - hx), cx + hx)
                        py = min(max(y, cy - hy), cy + hy)
                        d = math.hypot(x - px, y - py)
                    else:
                        cx, cy, r = self.circles[i]
                        dx, dy = x - cx, y - cy
                        dc = math.hypot(dx, dy)
                        if dc < 1e-9:
                            dx, dy, dc = 1.0, 0.0, 1.0
                        px, py, d = cx + dx / dc * r, cy + dy / dc * r, dc - r
                    if d <= reach:
                        out.append((px, py, d))
        return out

    def clearance(self, x, y, reach=REACH):
        """Distance from (x, y) to the nearest wall or obstacle surface
        (`reach` if nothing is that close)."""
        pts = self.closest_static_points(x, y, reach)
        return min((d for _, _, d in pts), default=reach)

    def inside(self, x, y, radius=0.0):
        x0, x1, y0, y1 = self.bounds
        return x0 + radius < x < x1 - radius and y0 + radius < y < y1 - radius

    def is_free(self, x, y, radius):
        """True if a disc of this radius at (x, y) is inside the area and touches nothing static."""
        return self.inside(x, y) and self.clearance(x, y) >= radius

    def line_is_free(self, a, b, radius, step=0.1):
        """True if a disc of `radius` can slide from a to b without touching
        any wall or obstacle."""
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        return all(self.is_free(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, radius)
                   for i in range(n + 1))

    def random_free_point(self, rng, radius, avoid=(), tries=2000):
        """Random point where a disc of `radius` fits, and which is at least
        `min_dist` from every (ax, ay, min_dist) in `avoid`."""
        x0, x1, y0, y1 = self.bounds
        for _ in range(tries):
            x = rng.uniform(x0 + radius, x1 - radius)
            y = rng.uniform(y0 + radius, y1 - radius)
            if not self.is_free(x, y, radius):
                continue
            if all(math.hypot(x - ax, y - ay) >= d for ax, ay, d in avoid):
                return (x, y)
        raise RuntimeError("Could not find a free point; the area may be too crowded")

    def random_walk_point(self, rng, radius, avoid=(), tries=2000, near=None, max_dist=None):
        """Where pedestrians may stand or head to: the sidewalks in the street
        layouts (optionally within max_dist of `near`), anywhere free indoors."""
        if self.roadmap is not None:
            for _ in range(tries):
                (x, y), _, _ = self.roadmap.random_sidewalk_point(rng, near, max_dist)
                if not self.inside(x, y, 0.5) or not self.is_free(x, y, radius):
                    continue
                if all(math.hypot(x - ax, y - ay) >= d for ax, ay, d in avoid):
                    return (x, y)
            raise RuntimeError("Could not find a free sidewalk point")
        if self.layout != "road":
            return self.random_free_point(rng, radius, avoid, tries)
        y_min = config.ROAD_WIDTH / 2 + radius + 0.2
        y_max = self.half_y - radius
        for _ in range(tries):
            x = rng.uniform(-self.half_x + radius + 1, self.half_x - radius - 1)
            y = rng.uniform(y_min, y_max) * (1 if rng.random() < 0.5 else -1)
            if not self.is_free(x, y, radius):
                continue
            if all(math.hypot(x - ax, y - ay) >= d for ax, ay, d in avoid):
                return (x, y)
        raise RuntimeError("Could not find a free sidewalk point")

    def route(self, start, goal):
        """Waypoints from start to goal for the robot and the indoor walkers.
        doorway: a path across the dividing wall goes through the door.
        road:    a path across the road goes over the zebra crossing.
        street / town: the robot follows its lane, through junctions (turns)."""
        crosses = start[1] * goal[1] < 0
        s = 1 if start[1] > 0 else -1
        if self.layout == "doorway" and crosses:
            return [(config.DOOR_X, 0.8 * s), (config.DOOR_X, -0.8 * s), tuple(goal)]
        if self.layout == "road" and crosses:
            edge = config.ROAD_WIDTH / 2 + 0.6
            return [(config.CROSSWALK_X, edge * s), (config.CROSSWALK_X, -edge * s), tuple(goal)]
        if self.roadmap is not None:
            via = [tuple(start), tuple(goal)]
            if self.layout == "town" and tuple(start) == tuple(config.TOWN_START) \
                    and tuple(goal) == tuple(config.TOWN_GOAL):
                via = [tuple(p) for p in config.TOWN_ROUTE]     # the planned route, with its lane changes
            path = []
            for a, b in zip(via, via[1:]):
                piece = self.roadmap.drive_route(a, b)          # follow the lanes, turning at junctions
                if not piece:
                    path = None
                    break
                path += piece
            if path:
                return path
        return [tuple(goal)]

    def on_road(self, x, y):
        return self.roadmap is not None and self.roadmap.on_road(x, y)
