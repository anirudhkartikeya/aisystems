"""roadmap.py - the map of the "street" and "town" layouts.

It describes the city as a small road network, the same way driving
simulators such as CARLA do:

  nodes      junctions (with traffic lights) and map edges where roads leave
  edges      roads between nodes (straight or curved), 2 lanes, maybe parking
  lanes      the path a car follows along one side of a road
  connectors the curved path a car follows through a junction (straight/left/right)
  crossings  zebra crossings (no light) and signalised crossings at junctions
  walk graph sidewalks + crossings, which pedestrians plan their routes on
  signals    traffic-light controllers, one per junction

It also decides where buildings, parked cars, trees, lamps, benches, bins,
bus stops and the park go. Everything is fixed for a layout (not random per
seed), so the town is always the same and only the people and traffic change.

Only plain maths here: no PyBullet. visuals.py draws what is described here.
"""
import heapq
import math
import random

import config

AXES = ((1, 0), (0, 1), (-1, 0), (0, -1))


# ======================================================================= polylines
class Polyline:
    """A path made of straight pieces, with distance-along-the-path helpers."""

    def __init__(self, pts):
        clean = []
        for p in pts:
            p = (float(p[0]), float(p[1]))
            if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
                clean.append(p)
        if len(clean) < 2:
            raise ValueError("a polyline needs two different points")
        self.pts = clean
        self.s = [0.0]
        for a, b in zip(clean, clean[1:]):
            self.s.append(self.s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.s[-1]

    def _segment(self, s):
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.s) - 2
        while lo < hi:                      # binary search for the segment holding s
            mid = (lo + hi + 1) // 2
            if self.s[mid] <= s:
                lo = mid
            else:
                hi = mid - 1
        return lo, s

    def point_at(self, s):
        """(x, y, heading) at distance s along the path."""
        i, s = self._segment(s)
        (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
        seg = self.s[i + 1] - self.s[i]
        t = (s - self.s[i]) / seg if seg > 0 else 0.0
        return ax + (bx - ax) * t, ay + (by - ay) * t, math.atan2(by - ay, bx - ax)

    def project(self, x, y):
        """Closest point on the path to (x, y): returns (s, distance, lateral),
        lateral > 0 when (x, y) is on the left of the direction of travel."""
        best = (math.inf, 0.0, 0.0)
        for i, ((ax, ay), (bx, by)) in enumerate(zip(self.pts, self.pts[1:])):
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + dx * t, ay + dy * t
            d = math.hypot(x - px, y - py)
            if d < best[0]:
                seg = math.sqrt(seg2)
                lateral = (dx * (y - ay) - dy * (x - ax)) / seg
                best = (d, self.s[i] + t * seg, lateral)
        return best[1], best[0], best[2]

    def clip(self, s0, s1):
        """The part of the path between s0 and s1 (s0 < s1)."""
        s0, s1 = max(0.0, s0), min(self.length, s1)
        pts = [self.point_at(s0)[:2]]
        pts += [p for p, s in zip(self.pts, self.s) if s0 < s < s1]
        pts.append(self.point_at(s1)[:2])
        return Polyline(pts)

    def offset(self, d):
        """The path moved sideways by d (left of the direction of travel is +)."""
        out = []
        n = len(self.pts)
        for i, (x, y) in enumerate(self.pts):
            normals = []
            if i > 0:
                normals.append(_left_normal(self.pts[i - 1], self.pts[i]))
            if i < n - 1:
                normals.append(_left_normal(self.pts[i], self.pts[i + 1]))
            nx = sum(v[0] for v in normals)
            ny = sum(v[1] for v in normals)
            norm = math.hypot(nx, ny)
            nx, ny = nx / norm, ny / norm
            scale = 1.0 / max(0.5, nx * normals[0][0] + ny * normals[0][1])   # keep corners square
            out.append((x + nx * d * scale, y + ny * d * scale))
        return Polyline(out)

    def clip_pts(self, s0, s1):
        """Like clip, but returns a plain list of points and copes with an
        empty piece (s1 <= s0 gives just one point)."""
        s0, s1 = max(0.0, s0), min(self.length, s1)
        if s1 - s0 < 1e-6:
            return [self.point_at(s0)[:2]]
        return self.clip(s0, s1).pts

    def reversed(self):
        return Polyline(self.pts[::-1])

    def samples(self, step):
        """Points every `step` metres: list of (x, y, heading, s)."""
        n = max(1, int(math.ceil(self.length / step)))
        return [self.point_at(self.length * k / n) + (self.length * k / n,) for k in range(n + 1)]


def _left_normal(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return -dy / n, dx / n


def arc(cx, cy, r, a0, a1, step_deg=6.0):
    """Points on a circle of radius r around (cx, cy), from angle a0 to a1 (radians)."""
    n = max(2, int(abs(math.degrees(a1 - a0)) / step_deg))
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
            for k in range(n + 1)]


def segment_intersection(p, p2, q, q2):
    """Where segment p-p2 crosses segment q-q2: fraction along p-p2, or None."""
    rx, ry = p2[0] - p[0], p2[1] - p[1]
    sx, sy = q2[0] - q[0], q2[1] - q[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    qpx, qpy = q[0] - p[0], q[1] - p[1]
    t = (qpx * sy - qpy * sx) / den
    u = (qpx * ry - qpy * rx) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ======================================================================= map pieces
class Node:
    def __init__(self, nid, x, y, kind):
        self.id, self.x, self.y, self.kind = nid, x, y, kind   # kind: "junction" or "end"
        self.arms = {}          # axis direction (dx, dy) -> (edge, edge_starts_here)
        self.hx = self.hy = 0.0  # half size of the junction box
        self.signal = None

    def reach(self, direction):
        """Distance from the node centre to the edge of the junction box along an arm."""
        if self.kind != "junction":
            return 0.0
        return self.hx if direction[1] == 0 else self.hy


class Edge:
    def __init__(self, eid, a, b, pts, parking, speed, name, lanes=1):
        self.id, self.a, self.b = eid, a, b
        self.center = Polyline(pts)
        self.parking = parking
        self.lanes = lanes                      # driving lanes in each direction
        self.drive_half = lanes * config.LANE_WIDTH
        self.half = self.drive_half + (config.PARKING_WIDTH if parking else 0.0)
        self.speed = speed
        self.name = name
        self.midblock = []      # s positions of zebra crossings along this road
        self.clip_a = self.clip_b = 0.0   # where the junction boxes end along the centre line


class Crossing:
    """Somewhere people walk across a road."""

    def __init__(self, cid, kind, center, along, half, width, edge, node=None, arm=None):
        self.id, self.kind = cid, kind            # "signal", "zebra" or "jaywalk"
        self.center = center                      # (x, y)
        self.along = along                        # unit vector along the road
        self.across = (-along[1], along[0])       # unit vector across the road
        self.half = half                          # half width of the road being crossed
        self.width = width                        # how wide the painted crossing is
        self.edge, self.node, self.arm = edge, node, arm

    def curb(self, side):
        """Point at the curb on one side (+1 = across direction, -1 = other side)."""
        d = self.half + 0.6
        return (self.center[0] + side * self.across[0] * d, self.center[1] + side * self.across[1] * d)

    def road_segment(self, extra=0.3):
        d = self.half + extra
        c, a = self.center, self.across
        return (c[0] - a[0] * d, c[1] - a[1] * d), (c[0] + a[0] * d, c[1] + a[1] * d)

    def contains(self, x, y, margin=0.0):
        """True if (x, y) is on the road part of this crossing (`margin` widens
        it along the road; people standing on the kerb never count)."""
        dx, dy = x - self.center[0], y - self.center[1]
        along = dx * self.along[0] + dy * self.along[1]
        across = dx * self.across[0] + dy * self.across[1]
        return abs(along) <= self.width / 2 + margin and abs(across) <= self.half + 0.1


class PathElement:
    """Something a car drives along: a lane or a junction connector."""

    def __init__(self, eid, poly, speed):
        self.id = eid
        self.poly = poly
        self.length = poly.length
        self.speed = speed
        self.crossings = []     # (s, crossing) sorted by s
        self.next = []          # elements a car can go to after this one
        self.signal = None      # (Signal, arm direction) for lanes ending at a traffic light
        self.exit = False       # lane leaves the map
        self.entry = False      # lane enters the map


class Lane(PathElement):
    def __init__(self, eid, edge, forward, poly):
        super().__init__(eid, poly, edge.speed)
        self.edge, self.forward = edge, forward
        self.start_node = edge.a if forward else edge.b
        self.end_node = edge.b if forward else edge.a
        self.opposite = None


class Connector(PathElement):
    def __init__(self, eid, lane_in, lane_out, poly, turn, node):
        super().__init__(eid, poly, min(lane_in.speed, lane_out.speed) * (1.0 if turn == "straight" else 0.6))
        self.lane_in, self.lane_out, self.turn, self.node = lane_in, lane_out, turn, node
        self.next = [lane_out]


class Signal:
    """Traffic lights at one junction: main street (x) green, then side streets (y).

    The main-street green starts with a few seconds of green ARROW for the
    eastbound cars only ("x<"): the westbound side still has red, so cars
    turning left from the eastbound side (like the robot at the T-junction)
    go first instead of waiting for a gap in the oncoming traffic.
    """

    def __init__(self, node, offset):
        self.node = node
        g1, g2, y, r = (config.LIGHT_GREEN_MAIN, config.LIGHT_GREEN_SIDE,
                        config.LIGHT_YELLOW, config.LIGHT_ALL_RED)
        self.phases = [("x", "green", g1), ("x", "yellow", y), (None, "red", r),
                       ("y", "green", g2), ("y", "yellow", y), (None, "red", r)]
        if config.LIGHT_LEFT_ARROW > 0:
            self.phases.insert(0, ("x<", "green", config.LIGHT_LEFT_ARROW))
        self.cycle = sum(d for _, _, d in self.phases)
        self.offset = offset
        self.update(0.0)

    def update(self, t):
        tc = (t + self.offset) % self.cycle
        for axis, color, dur in self.phases:
            if tc < dur:
                self.axis, self.color, self.left = axis, color, dur - tc
                return
            tc -= dur

    def vehicle_color(self, arm):
        """Colour shown to cars arriving on the arm pointing in `arm` direction."""
        axis = "x" if arm[1] == 0 else "y"
        if self.axis == "x<":                    # head start for the eastbound side (arriving from the west)
            return self.color if axis == "x" and arm[0] < 0 else "red"
        return self.color if self.axis == axis else "red"

    def walk(self, crossing):
        """True while people may start crossing (the cars they cross have red,
        and there are a few seconds of green left)."""
        crossed_axis = "x" if crossing.arm[1] == 0 else "y"      # the road being crossed
        other = "y" if crossed_axis == "x" else "x"
        return self.axis == other and self.color == "green" and self.left > 4.0

    def ped_color(self, crossing):
        return "green" if self.walk(crossing) else "red"


class WalkEdge:
    def __init__(self, wid, u, v, poly, kind, crossing=None, road=None, side=0):
        self.id, self.u, self.v, self.poly = wid, u, v, poly
        self.kind = kind            # "sidewalk" or "crossing"
        self.crossing = crossing
        self.road, self.side = road, side   # sidewalks: which road, which side (+1 left / -1 right)


# ======================================================================= the map
class RoadMap:
    def __init__(self, layout):
        self.layout = layout
        self.nodes, self.edges = [], []
        self.lanes, self.connectors, self.crossings, self.signals = [], [], [], []
        self.walk_nodes, self.walk_edges, self.walk_adj = [], [], {}
        self.obstacles = []          # buildings, parked cars, trees... (world.py turns them into geometry)
        self.parks = []              # (x0, y0, x1, y1) grass areas
        if layout == "town":
            self._town()
        elif layout == "street":
            self._street()
        else:
            raise ValueError(layout)
        self._finish_nodes()
        self._build_crossings_and_sidewalks()
        self._build_lanes()
        self._build_connectors()
        self._attach_crossings()
        self._decorate()

    # ------------------------------------------------------------ layouts
    def _node(self, x, y, kind):
        n = Node(len(self.nodes), x, y, kind)
        self.nodes.append(n)
        return n

    def _edge(self, a, b, pts, parking, speed, name, lanes=1):
        e = Edge(len(self.edges), a, b, [(a.x, a.y)] + list(pts) + [(b.x, b.y)], parking, speed, name, lanes)
        self.edges.append(e)
        return e

    def _street(self):
        h = config.STREET_LENGTH / 2
        w = self._node(-h, 0.0, "end")
        e = self._node(h, 0.0, "end")
        main = self._edge(w, e, [], True, config.TRAFFIC_SPEED_MAIN, "main")
        main.midblock = [x + h for x in config.STREET_CROSSWALKS_X]
        self.bounds = (-h, h, -(main.half + config.SIDEWALK_WIDTH), main.half + config.SIDEWALK_WIDTH)

    def _town(self):
        hx = config.TOWN_HALF_X
        x1, x2 = config.TOWN_JUNCTIONS_X
        top, bottom, r = config.TOWN_NORTH_Y, config.TOWN_SOUTH_Y, config.TOWN_CURVE_RADIUS
        west, east = self._node(-hx, 0.0, "end"), self._node(hx, 0.0, "end")
        j1, j2 = self._node(x1, 0.0, "junction"), self._node(x2, 0.0, "junction")
        south = self._node(x1, bottom, "end")
        main, side = config.TRAFFIC_SPEED_MAIN, config.TRAFFIC_SPEED_SIDE
        n = config.TOWN_ROUTE_LANES                    # lanes each way on the car's route
        e1 = self._edge(west, j1, [], True, main, "main west", n)
        e2 = self._edge(j1, j2, [], True, main, "main centre", n)
        e3 = self._edge(j2, east, [], True, main, "main east", n)
        self._edge(j1, south, [], False, side, "side street")
        loop = ([(x1, top - r)] + arc(x1 + r, top - r, r, math.pi, math.pi / 2)[1:]
                + [(x2 - r, top)] + arc(x2 - r, top - r, r, math.pi / 2, 0.0)[1:])
        loop_edge = self._edge(j1, j2, loop, False, side, "loop", n)
        for x in config.TOWN_MIDBLOCK_X:                 # zebra crossings on the main street
            for e in (e1, e2, e3):
                s, d, _ = e.center.project(x, 0.0)
                if d < 0.01 and 0 < s < e.center.length:
                    e.midblock.append(s)
        for x, y in config.TOWN_LOOP_CROSSINGS:            # zebra crossings on the loop road
            loop_edge.midblock.append(loop_edge.center.project(x, y)[0])
        self.edges[3].midblock.append(abs(bottom) * 0.55)   # side street
        walk = config.SIDEWALK_WIDTH
        self.bounds = (-hx, hx, bottom, top + walk + 4)
        # the park inside the loop, behind the buildings along the main street
        lh, mh = loop_edge.half, e2.half
        self.parks.append((x1 + lh + walk + 10.0, mh + walk + 12.0, x2 - lh - walk - 10.0, top - lh - walk - 1.0))

    # ------------------------------------------------------------ nodes and arms
    def _finish_nodes(self):
        for e in self.edges:
            pts = e.center.pts
            d0 = _axis(pts[0], pts[1])
            d1 = _axis(pts[-1], pts[-2])
            e.a.arms[d0] = (e, True)
            e.b.arms[d1] = (e, False)
        for n in self.nodes:
            if n.kind != "junction":
                continue
            vert = [e.half for d, (e, _) in n.arms.items() if d[0] == 0]
            hor = [e.half for d, (e, _) in n.arms.items() if d[1] == 0]
            n.hx = max(vert) if vert else 0.0
            n.hy = max(hor) if hor else 0.0
            if config.TRAFFIC_LIGHTS:
                k = len(self.signals)
                offsets = config.TOWN_LIGHT_OFFSETS if self.layout == "town" else ()
                n.signal = Signal(n, offset=offsets[k] if k < len(offsets) else 7.0 * k)
                self.signals.append(n.signal)
        for e in self.edges:
            e.clip_a = e.a.reach(_axis(e.center.pts[0], e.center.pts[1]))
            e.clip_b = e.b.reach(_axis(e.center.pts[-1], e.center.pts[-2]))

    def arm_dir(self, edge, at_start):
        pts = edge.center.pts
        return _axis(pts[0], pts[1]) if at_start else _axis(pts[-1], pts[-2])

    def corner(self, node, sx, sy):
        w = config.SIDEWALK_WIDTH / 2
        return (node.x + sx * (node.hx + w), node.y + sy * (node.hy + w))

    def crossing_dist(self, node, arm):
        """Distance from a junction centre to the middle of the crossing on an arm."""
        return node.reach(arm) + config.SIDEWALK_WIDTH / 2

    def stop_dist(self, node, arm):
        """Distance from a node centre to the stop line on an arm (0 at map edges)."""
        if node.kind != "junction":
            return 0.0
        return self.crossing_dist(node, arm) + config.STREET_CROSSWALK_WIDTH / 2 + config.STOP_LINE_GAP

    # ------------------------------------------------------------ crossings, sidewalks, walk graph
    def _walk_node(self, p):
        self.walk_nodes.append((float(p[0]), float(p[1])))
        self.walk_adj[len(self.walk_nodes) - 1] = []
        return len(self.walk_nodes) - 1

    def _walk_edge(self, u, v, pts, kind, crossing=None, road=None, side=0):
        poly = Polyline([self.walk_nodes[u]] + list(pts) + [self.walk_nodes[v]])
        we = WalkEdge(len(self.walk_edges), u, v, poly, kind, crossing, road, side)
        self.walk_edges.append(we)
        self.walk_adj[u].append((v, we))
        self.walk_adj[v].append((u, we))
        return we

    def _new_crossing(self, kind, center, along, half, edge, node=None, arm=None):
        c = Crossing(len(self.crossings), kind, center, along, half,
                     config.STREET_CROSSWALK_WIDTH, edge, node, arm)
        self.crossings.append(c)
        return c

    def _build_crossings_and_sidewalks(self):
        w = config.SIDEWALK_WIDTH
        corners = {}             # (node id, sx, sy) -> walk node
        for n in self.nodes:
            if n.kind != "junction":
                continue
            for sx in (-1, 1):
                for sy in (-1, 1):
                    corners[(n.id, sx, sy)] = self._walk_node(self.corner(n, sx, sy))
            # the four sides of the junction: a crossing where a road leaves, sidewalk otherwise
            for d in AXES:
                if d[1] == 0:
                    a, b = corners[(n.id, d[0], 1)], corners[(n.id, d[0], -1)]
                else:
                    a, b = corners[(n.id, 1, d[1])], corners[(n.id, -1, d[1])]
                if d in n.arms:
                    edge = n.arms[d][0]
                    dist = self.crossing_dist(n, d)
                    center = (n.x + d[0] * dist, n.y + d[1] * dist)
                    kind = "signal" if n.signal else "zebra"
                    c = self._new_crossing(kind, center, d, edge.half, edge, n, d)
                    # walk from corner a to corner b via the two curbs
                    pa, pb = self.walk_nodes[a], self.walk_nodes[b]
                    side_a = 1 if ((pa[0] - center[0]) * c.across[0] + (pa[1] - center[1]) * c.across[1]) > 0 else -1
                    self._walk_edge(a, b, [c.curb(side_a), c.curb(-side_a)], "crossing", crossing=c)
                else:
                    self._walk_edge(a, b, [], "sidewalk")

        for e in self.edges:
            for side in (1, -1):
                off = e.center.offset(side * (e.half + w / 2))
                da, db = self.arm_dir(e, True), self.arm_dir(e, False)
                s0 = self.crossing_dist(e.a, da) if e.a.kind == "junction" else 0.0
                s1 = off.length - (self.crossing_dist(e.b, db) if e.b.kind == "junction" else 0.0)
                walk_line = off.clip(s0, s1)
                # end points: junction corners, or new nodes at the map edge
                ends = []
                for node, d, p in ((e.a, da, walk_line.pts[0]), (e.b, db, walk_line.pts[-1])):
                    if node.kind == "junction":
                        left = (-d[1], d[0])
                        s_here = side if node is e.a else -side   # side seen from the node looking out
                        sx = int(round(d[0] + left[0] * s_here))
                        sy = int(round(d[1] + left[1] * s_here))
                        ends.append(corners[(node.id, sx, sy)])
                    else:
                        ends.append(self._walk_node(p))
                # zebra crossings in the middle of the road split the sidewalk
                cuts = []
                for s in e.midblock:
                    x, y, hdg = e.center.point_at(s)
                    along = (math.cos(hdg), math.sin(hdg))
                    found = [c for c in self.crossings if c.edge is e and c.node is None
                             and math.hypot(c.center[0] - x, c.center[1] - y) < 0.01]
                    c = found[0] if found else self._new_crossing("zebra", (x, y), along, e.half, e)
                    p = (x + c.across[0] * side * (e.half + w / 2), y + c.across[1] * side * (e.half + w / 2))
                    cuts.append((walk_line.project(*p)[0], c, p))
                cuts.sort(key=lambda t: t[0])
                prev_node, prev_s = ends[0], 0.0
                for s, c, p in cuts:
                    k = self._walk_node(p)
                    self._walk_edge(prev_node, k, walk_line.clip(prev_s, s).pts[1:-1], "sidewalk", road=e, side=side)
                    c.sidewalk_nodes = getattr(c, "sidewalk_nodes", {})
                    c.sidewalk_nodes[side] = k
                    prev_node, prev_s = k, s
                self._walk_edge(prev_node, ends[1], walk_line.clip(prev_s, walk_line.length).pts[1:-1],
                                "sidewalk", road=e, side=side)
        for c in self.crossings:                            # the zebra crossings' own walk edges
            if c.node is None:
                a, b = c.sidewalk_nodes[1], c.sidewalk_nodes[-1]
                self._walk_edge(a, b, [c.curb(1), c.curb(-1)], "crossing", crossing=c)
        self.sidewalks = [we for we in self.walk_edges if we.kind == "sidewalk"]
        self._sidewalk_weights = [we.poly.length for we in self.sidewalks]

    # ------------------------------------------------------------ lanes and connectors
    def _build_lanes(self):
        """Lanes of every road: index 0 is the inner lane (next to the centre
        line), the last index the outer lane (next to the curb / parking)."""
        lw = config.LANE_WIDTH
        for e in self.edges:
            da, db = self.arm_dir(e, True), self.arm_dir(e, False)
            sa, sb = self.stop_dist(e.a, da), self.stop_dist(e.b, db)
            made = []
            for k in range(e.lanes):
                off = lw * (k + 0.5)
                fwd_line = e.center.offset(-off)
                bwd_line = e.center.offset(off).reversed()
                fwd = Lane(len(self.lanes), e, True, fwd_line.clip(sa, fwd_line.length - sb))
                self.lanes.append(fwd)
                bwd = Lane(len(self.lanes), e, False, bwd_line.clip(sb, bwd_line.length - sa))
                self.lanes.append(bwd)
                fwd.opposite, bwd.opposite = bwd, fwd
                fwd.index = bwd.index = k
                fwd.offset = bwd.offset = off
                made += [fwd, bwd]
            for lane in made:
                lane.n_lanes = e.lanes
                end, start = lane.end_node, lane.start_node
                lane.exit = end.kind == "end"
                lane.entry = start.kind == "end"
                if end.signal is not None:
                    arm = db if lane.forward else da
                    lane.signal = (end.signal, arm)
                lane.arm_at_end = db if lane.forward else da

    def _build_connectors(self):
        for n in self.nodes:
            if n.kind != "junction":
                continue
            ins = [l for l in self.lanes if l.end_node is n]
            outs = [l for l in self.lanes if l.start_node is n]
            for li in ins:
                for lo in outs:
                    if lo.edge is li.edge:
                        continue                      # no U-turns
                    p0 = li.poly.pts[-1]
                    h0 = li.poly.point_at(li.length)[2]
                    p2 = lo.poly.pts[0]
                    h2 = lo.poly.point_at(0.0)[2]
                    turn_angle = wrap(h2 - h0)
                    turn = "straight" if abs(turn_angle) < 0.4 else ("left" if turn_angle > 0 else "right")
                    # lane rules on multi-lane roads: turn left from the inner lane into the
                    # inner lane, right from the outer lane into the outer lane, go straight
                    # into the lane with the same number
                    if turn == "left" and not (li.index == 0 and lo.index == 0):
                        continue
                    if turn == "right" and not (li.index == li.n_lanes - 1 and lo.index == lo.n_lanes - 1):
                        continue
                    if turn == "straight" and lo.index != min(li.index, lo.n_lanes - 1):
                        continue
                    if turn == "straight":
                        pts = [p0, p2]
                    else:
                        ctrl = _ray_meet(p0, h0, p2, h2)
                        pts = [_bezier(p0, ctrl, p2, k / 16) for k in range(17)]
                    c = Connector(len(self.connectors), li, lo, Polyline(pts), turn, n)
                    self.connectors.append(c)
                    li.next.append(c)
        for lane in self.lanes:
            lane.turns = {c.turn: c for c in lane.next}

    def _attach_crossings(self):
        """Which crossings does each lane / connector drive over, and where."""
        for el in self.lanes + self.connectors:
            hits = []
            for c in self.crossings:
                q0, q1 = c.road_segment()
                for i, (a, b) in enumerate(zip(el.poly.pts, el.poly.pts[1:])):
                    t = segment_intersection(a, b, q0, q1)
                    if t is not None:
                        hits.append((el.poly.s[i] + t * (el.poly.s[i + 1] - el.poly.s[i]), c))
                        break
            el.crossings = sorted(hits, key=lambda h: h[0])

    # ------------------------------------------------------------ pedestrian routes
    def random_sidewalk_point(self, rng, near=None, max_dist=None, tries=200):
        """A random point on a sidewalk (optionally within max_dist of `near`).
        Returns (point, walk_edge, s)."""
        for _ in range(tries):
            we = self.sidewalks[_weighted_index(rng, self._sidewalk_weights)]
            s = rng.uniform(0.0, we.poly.length)
            x, y, _ = we.poly.point_at(s)
            if near is None or max_dist is None or math.hypot(x - near[0], y - near[1]) <= max_dist:
                return (x, y), we, s
        we = self.sidewalks[0]
        return we.poly.pts[0], we, 0.0

    def nearest_sidewalk(self, x, y):
        best = None
        for we in self.sidewalks:
            s, d, _ = we.poly.project(x, y)
            if best is None or d < best[2]:
                best = (we, s, d)
        return best

    def plan_walk(self, start, goal):
        """Route on the sidewalks and crossings from start to goal.
        Returns a list of steps (x, y, crossing, role) where role is
        "walk", "curb" (wait here, then cross) or "far" (the other curb)."""
        ws, ss, _ = self.nearest_sidewalk(*start)
        wg, sg, _ = self.nearest_sidewalk(*goal)
        if ws is wg:                                   # same piece of sidewalk: just walk along it
            line = ws.poly.clip_pts(min(ss, sg), max(ss, sg))
            if ss > sg:
                line = line[::-1]
            return [(x, y, None, "walk") for x, y in line[1:]] + [(goal[0], goal[1], None, "walk")]
        # Dijkstra from both ends of the start piece to both ends of the goal piece
        dist, prev = {}, {}
        heap = []
        for node, d in ((ws.u, ss), (ws.v, ws.poly.length - ss)):
            dist[node] = d
            prev[node] = None
            heapq.heappush(heap, (d, node))
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, we in self.walk_adj[u]:
                nd = d + we.poly.length
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, we)
                    heapq.heappush(heap, (nd, v))
        end_costs = [(dist.get(wg.u, math.inf) + sg, wg.u), (dist.get(wg.v, math.inf) + wg.poly.length - sg, wg.v)]
        _, last = min(end_costs)
        # walk back to collect the walk edges used
        chain = []
        node = last
        while prev.get(node) is not None:
            u, we = prev[node]
            chain.append((u, node, we))
            node = u
        chain.reverse()
        steps = []
        first_node = node
        part = ws.poly.clip_pts(0.0, ss)[::-1] if first_node == ws.u else ws.poly.clip_pts(ss, ws.poly.length)
        steps += [(x, y, None, "walk") for x, y in part[1:]]
        for u, v, we in chain:
            pts = we.poly.pts if we.u == u else we.poly.pts[::-1]
            if we.kind == "crossing":
                steps.append((pts[1][0], pts[1][1], we.crossing, "curb"))
                steps.append((pts[2][0], pts[2][1], we.crossing, "far"))
                steps.append((pts[3][0], pts[3][1], None, "walk"))
            else:
                steps += [(x, y, None, "walk") for x, y in pts[1:]]
        part = wg.poly.clip_pts(0.0, sg) if last == wg.u else wg.poly.clip_pts(sg, wg.poly.length)[::-1]
        steps += [(x, y, None, "walk") for x, y in part[1:]]
        steps.append((goal[0], goal[1], None, "walk"))
        return _dedupe(steps)

    def drive_route(self, start, goal, step=1.0, max_off=2.0):
        """Points along the lanes (and through junctions) from start to goal,
        about `step` m apart, ending exactly at goal. None if start or goal is
        not in a lane, or the goal cannot be reached driving forwards."""
        def nearest(pt):
            best = None
            for lane in self.lanes:
                s, d, _ = lane.poly.project(*pt)
                if best is None or d < best[2]:
                    best = (lane, s, d)
            return best
        ls, ss, ds = nearest(start)
        lg, sg, dg = nearest(goal)
        if ds > max_off or dg > max_off:
            return None
        if ls is not lg and ls.edge is lg.edge and ls.forward == lg.forward:
            # change lane: drift smoothly from one lane to the other along the road
            s_goal = ls.poly.project(*goal)[0]
            if s_goal <= ss:
                return None
            out = []
            n = max(2, int((s_goal - ss) / step))
            for k in range(1, n + 1):
                s = ss + (s_goal - ss) * k / n
                ax, ay, _ = ls.poly.point_at(s)
                bx, by, _ = lg.poly.point_at(lg.poly.project(ax, ay)[0])
                t = k / n
                b = t * t * (3 - 2 * t)                      # smooth start and end
                out.append((ax + (bx - ax) * b, ay + (by - ay) * b))
            out[-1] = tuple(goal)
            return out
        if ls is lg and sg >= ss:
            pieces = [ls.poly.clip_pts(ss, sg)]
        else:
            # cheapest chain of lanes and junction connectors (Dijkstra)
            best = {id(ls): (ls.length - ss, None, ls)}
            heap = [(ls.length - ss, id(ls))]
            elements = {id(ls): ls}
            done = set()
            while heap:
                cost, key = heapq.heappop(heap)
                if key in done:
                    continue
                done.add(key)
                el = elements[key]
                if el is lg:
                    break
                for nxt in el.next:
                    c = cost + (sg if nxt is lg else nxt.length)
                    if c < best.get(id(nxt), (math.inf,))[0]:
                        best[id(nxt)] = (c, key, nxt)
                        elements[id(nxt)] = nxt
                        heapq.heappush(heap, (c, id(nxt)))
            if id(lg) not in best:
                return None
            chain, key = [], id(lg)
            while key is not None:
                chain.append(elements[key])
                key = best[key][1]
            chain.reverse()
            pieces = [ls.poly.clip_pts(ss, ls.length)]
            pieces += [el.poly.pts for el in chain[1:-1]]
            pieces.append(lg.poly.clip_pts(0.0, sg))
        pts = []
        for piece in pieces:
            for p in piece:
                if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-3:
                    pts.append(tuple(p))
        if len(pts) < 2:
            return [tuple(goal)]
        line = Polyline(pts)
        out = [line.point_at(s)[:2] for s in [step * (k + 1) for k in range(int(line.length / step))]]
        out.append(tuple(goal))
        return out

    def jaywalk_crossing(self, edge, x, y):
        """A made-up crossing straight across `edge` at the point nearest (x, y)."""
        s, _, _ = edge.center.project(x, y)
        cx, cy, hdg = edge.center.point_at(s)
        return Crossing(-1, "jaywalk", (cx, cy), (math.cos(hdg), math.sin(hdg)), edge.half, 1.0, edge)

    def on_road(self, x, y):
        """True if (x, y) is on the asphalt (lanes, parking, junction boxes)."""
        for n in self.nodes:
            if n.kind == "junction" and abs(x - n.x) <= n.hx and abs(y - n.y) <= n.hy:
                return True
        for e in self.edges:
            s, d, _ = e.center.project(x, y)
            if d <= e.half and e.clip_a <= s <= e.center.length - e.clip_b:
                return True
        return False

    # ------------------------------------------------------------ buildings, parked cars, furniture
    def _decorate(self):
        rnd = random.Random(1234)               # fixed: the town looks the same every run
        w = config.SIDEWALK_WIDTH
        cw = config.STREET_CROSSWALK_WIDTH / 2
        obs = self.obstacles

        def near_crossing(x, y, margin):
            """Within `margin` of a crossing's painted area or the sidewalk at its ends."""
            for c in self.crossings:
                dx, dy = x - c.center[0], y - c.center[1]
                along = abs(dx * c.along[0] + dy * c.along[1])
                across = abs(dx * c.across[0] + dy * c.across[1])
                if along < cw + margin and across < c.half + w + 1.0:
                    return True
            for edge, s_keep in keep_clear:            # where the scripted people step out
                s_pt, d_pt, _ = edge.center.project(x, y)
                if d_pt < edge.half + w + 1.0 and abs(s_pt - s_keep) < cw + margin + 5.0:
                    return True
            return False

        keep_clear = []
        if self.layout == "town":
            for c in config.TOWN_CROSSERS:
                edge = min(self.edges, key=lambda e: e.center.project(*c["at"])[1])
                keep_clear.append((edge, edge.center.project(*c["at"])[0]))

        def near_junction(x, y, margin):
            return any(n.kind == "junction" and abs(x - n.x) < n.hx + w + margin and abs(y - n.y) < n.hy + w + margin
                       for n in self.nodes)

        k = 0
        for e in self.edges:
            straight = [(a, b) for a, b in zip(e.center.pts, e.center.pts[1:])
                        if math.hypot(b[0] - a[0], b[1] - a[1]) > 8.0]
            for side in (1, -1):
                # lamps along the whole road (also round the curves)
                for x, y, hdg, s in e.center.samples(12.0):
                    if s < e.clip_a + 2 or s > e.center.length - e.clip_b - 2:
                        continue
                    nx, ny = -math.sin(hdg) * side, math.cos(hdg) * side
                    px, py = x + nx * (e.half + 0.45), y + ny * (e.half + 0.45)
                    if not near_crossing(px, py, 1.5) and not near_junction(px, py, 1.0):
                        obs.append({"type": "cylinder", "center": (px, py), "radius": 0.12, "look": "lamp",
                                    "yaw": math.atan2(ny, nx), "z": config.CURB_HEIGHT})
                for a, b in straight:
                    seg = Polyline([a, b])
                    hdg = math.atan2(b[1] - a[1], b[0] - a[0])
                    nx, ny = -math.sin(hdg) * side, math.cos(hdg) * side
                    along_x = abs(math.cos(hdg)) > 0.5
                    # trees at the back of the sidewalk
                    s = 4.0 + (0 if side == 1 else 4.0)
                    while s < seg.length - 2:
                        x, y, _ = seg.point_at(s)
                        px, py = x + nx * (e.half + w - 0.7), y + ny * (e.half + w - 0.7)
                        if not near_crossing(px, py, 1.5) and not near_junction(px, py, 1.0):
                            obs.append({"type": "cylinder", "center": (px, py), "radius": 0.3, "look": "tree",
                                        "scale": rnd.uniform(0.8, 1.25), "z": config.CURB_HEIGHT})
                        s += rnd.uniform(7.5, 10.5)
                    # bins, hydrants and benches here and there
                    s = rnd.uniform(5, 12)
                    while s < seg.length - 3:
                        x, y, _ = seg.point_at(s)
                        look = rnd.choice(["bin", "bin", "hydrant", "bench", "bench", "planter"])
                        if look == "bench":
                            px, py = x + nx * (e.half + w - 0.55), y + ny * (e.half + w - 0.55)
                            he = (0.8, 0.25) if along_x else (0.25, 0.8)
                            item = {"type": "box", "center": (px, py), "half_extents": he, "height": 0.85,
                                    "look": "bench", "yaw": hdg + (0 if side == 1 else math.pi)}
                        elif look == "planter":
                            px, py = x + nx * (e.half + w - 0.6), y + ny * (e.half + w - 0.6)
                            he = (0.6, 0.35) if along_x else (0.35, 0.6)
                            item = {"type": "box", "center": (px, py), "half_extents": he, "height": 0.5,
                                    "look": "planter"}
                        else:
                            px, py = x + nx * (e.half + 0.5), y + ny * (e.half + 0.5)
                            item = {"type": "cylinder", "center": (px, py),
                                    "radius": 0.25 if look == "bin" else 0.15, "look": look}
                        if not near_crossing(px, py, 2.0) and not near_junction(px, py, 1.5) \
                                and not self._too_close(px, py, 1.2):
                            item["z"] = config.CURB_HEIGHT
                            obs.append(item)
                        s += rnd.uniform(9, 18)
                    # parked cars
                    if e.parking and config.PARKED_CARS:
                        s = 3.0 + (0 if side == 1 else 2.5)
                        while s < seg.length - 3:
                            x, y, _ = seg.point_at(s)
                            px, py = x + nx * (e.drive_half + config.PARKING_WIDTH / 2), \
                                y + ny * (e.drive_half + config.PARKING_WIDTH / 2)
                            if not near_crossing(px, py, 4.5) and not near_junction(px, py, 1.5) \
                                    and rnd.random() < config.PARKED_CAR_FILL:
                                length = rnd.choice([3.9, 4.3, 4.5, 4.7, 5.0])
                                kind = {3.9: "hatch", 4.3: "sedan", 4.5: "sedan", 4.7: "suv", 5.0: "van"}[length]
                                he = (length / 2, 0.88) if along_x else (0.88, length / 2)
                                obs.append({"type": "box", "center": (px, py), "half_extents": he,
                                            "height": 1.5, "look": "parked_car", "kind": kind,
                                            "color_index": k, "yaw": hdg + (math.pi if side == 1 else 0.0)})
                            s += 6.0
                            k += 1
        # bus stops on the main street, one each way
        for e in self.edges:
            if e.name in ("main", "main centre", "main west") and e.parking:
                for side, start in ((1, 0.3), (-1, 0.7)):
                    for frac in (start, start + 0.08, start - 0.08, start + 0.16, start - 0.16):
                        x, y, hdg = e.center.point_at(e.center.length * frac)
                        nx, ny = -math.sin(hdg) * side, math.cos(hdg) * side
                        px, py = x + nx * (e.half + w - 0.9), y + ny * (e.half + w - 0.9)
                        if near_crossing(px, py, 3.0) or near_junction(px, py, 2.0) or self._too_close(px, py, 1.6):
                            continue
                        obs.append({"type": "box", "center": (px, py), "half_extents": (1.6, 0.6), "height": 2.4,
                                    "look": "bus_stop", "yaw": 0.0 if side == 1 else math.pi,
                                    "z": config.CURB_HEIGHT})
                        break
        self._place_signs(cw)
        self._place_buildings(rnd)
        self._place_park(rnd)

    def _place_signs(self, cw):
        """Pedestrian-crossing signs before every zebra (for both directions),
        speed-limit signs where roads enter the map, street-name signs at junctions."""
        z = config.CURB_HEIGHT
        for c in self.crossings:
            if c.kind != "zebra" or c.node is not None:
                continue
            for d in (1, -1):                          # traffic travelling along +along or -along
                dx, dy = c.along[0] * d, c.along[1] * d
                rx, ry = dy, -dx                       # the drivers' right
                px = c.center[0] - dx * (cw + 1.2) + rx * (c.half + 0.45)
                py = c.center[1] - dy * (cw + 1.2) + ry * (c.half + 0.45)
                if not self._too_close(px, py, 0.5):
                    self.obstacles.append({"type": "cylinder", "center": (px, py), "radius": 0.08,
                                           "look": "sign_crossing", "yaw": math.atan2(-dy, -dx), "z": z})
        for lane in self.lanes:
            if lane.entry and lane.length > 15:
                x, y, hdg = lane.poly.point_at(9.0)
                d = lane.edge.half - lane.offset + 0.45
                px, py = x + math.sin(hdg) * d, y - math.cos(hdg) * d
                if not self._too_close(px, py, 0.5):
                    self.obstacles.append({"type": "cylinder", "center": (px, py), "radius": 0.08,
                                           "look": "sign_speed", "yaw": hdg + math.pi, "z": z})
        w = config.SIDEWALK_WIDTH
        for n in self.nodes:
            if n.kind == "junction":
                px, py = n.x + n.hx + w - 0.5, n.y + n.hy + w - 0.5
                if not self._too_close(px, py, 0.5):
                    self.obstacles.append({"type": "cylinder", "center": (px, py), "radius": 0.08,
                                           "look": "sign_street", "yaw": math.pi, "z": z})

    def _too_close(self, x, y, d):
        for ob in self.obstacles:
            cx, cy = ob["center"]
            r = ob.get("radius") or max(ob["half_extents"])
            if math.hypot(x - cx, y - cy) < d + r:
                return True
        return False

    def _clear_of_roads(self, x0, y0, x1, y1, margin):
        """True if the rectangle stays clear of every road, sidewalk and junction."""
        w = config.SIDEWALK_WIDTH
        for e in self.edges:
            reach = e.half + w + margin
            for x, y, _, _ in e.center.samples(1.0):
                if x0 - reach < x < x1 + reach and y0 - reach < y < y1 + reach:
                    # distance from the rectangle to this centre-line point
                    dx = max(x0 - x, 0, x - x1)
                    dy = max(y0 - y, 0, y - y1)
                    if math.hypot(dx, dy) < reach:
                        return False
        for n in self.nodes:
            if n.kind == "junction":
                if x0 < n.x + n.hx + w + margin and x1 > n.x - n.hx - w - margin and \
                        y0 < n.y + n.hy + w + margin and y1 > n.y - n.hy - w - margin:
                    return False
        return True

    def _place_buildings(self, rnd):
        w = config.SIDEWALK_WIDTH
        x_lo, x_hi, y_lo, y_hi = self.bounds
        ext = 18.0                             # buildings continue a little past the map edge
        placed = []
        widths = [9, 7, 11, 8, 10, 6, 9.5, 7.5, 12, 8.5]
        heights = [6, 9, 4.5, 12, 7.5, 5, 10, 6.5, 14, 8]
        i = 0
        for e in self.edges:
            for a, b in zip(e.center.pts, e.center.pts[1:]):
                if math.hypot(b[0] - a[0], b[1] - a[1]) < 8.0:
                    continue
                horizontal = abs(b[1] - a[1]) < 1e-6
                for side in (1, -1):
                    front = e.half + w + config.BUILDING_SETBACK
                    lo = min(a[0], b[0]) if horizontal else min(a[1], b[1])
                    hi = max(a[0], b[0]) if horizontal else max(a[1], b[1])
                    if e.a.kind == "end" and (a == e.center.pts[0]):
                        lo, hi = (lo - ext, hi) if (b[0] > a[0] or b[1] > a[1]) else (lo, hi + ext)
                    if e.b.kind == "end" and (b == e.center.pts[-1]):
                        lo, hi = (lo, hi + ext) if (b[0] > a[0] or b[1] > a[1]) else (lo - ext, hi)
                    t = lo
                    homes = e.name in ("side street", "loop")      # quieter streets: mostly houses
                    while t < hi:
                        house = homes and rnd.random() < 0.7
                        if house:                                   # detached house, front garden
                            width = rnd.uniform(8.0, 11.0)
                            depth = rnd.uniform(8.0, 9.5)
                            height = rnd.uniform(5.6, 6.3)          # to the eaves; the roof goes on top
                            setback, gap = rnd.uniform(1.8, 3.0), rnd.uniform(2.0, 3.5)
                        else:
                            width = widths[i % len(widths)] * rnd.uniform(0.9, 1.1)
                            depth = rnd.uniform(7.0, 10.0)
                            height = heights[i % len(heights)] * rnd.uniform(0.85, 1.15)
                            setback, gap = 0.0, 0.0
                        i += 1
                        if horizontal:
                            ny = side if b[0] > a[0] else -side
                            yf = a[1] + ny * (front + setback)
                            x0, x1 = t, t + width
                            y0, y1 = (yf, yf + depth) if ny > 0 else (yf - depth, yf)
                            face = (0, -ny)
                        else:
                            nx = -side if b[1] > a[1] else side
                            xf = a[0] + nx * (front + setback)
                            y0, y1 = t, t + width
                            x0, x1 = (xf, xf + depth) if nx > 0 else (xf - depth, xf)
                            face = (-nx, 0)
                        t += width + gap
                        shrink = 0.3
                        box = (x0 + shrink, y0 + shrink, x1 - shrink, y1 - shrink)
                        if any(_overlap(box, q) for q in placed):
                            continue
                        if not self._clear_of_roads(*box, margin=0.4):
                            continue
                        if any(_overlap(box, p) for p in self.parks):
                            continue
                        placed.append(box)
                        bx0, by0, bx1, by1 = box
                        self.obstacles.append({
                            "type": "box", "center": ((bx0 + bx1) / 2, (by0 + by1) / 2),
                            "half_extents": ((bx1 - bx0) / 2, (by1 - by0) / 2), "height": height,
                            "look": "building", "face": face, "color_index": i,
                            "kind": "house" if house else "block",
                            "shop": (not house) and rnd.random() < (0.2 if homes else 0.45),
                            "balconies": (not house) and rnd.random() < 0.4,
                            "garage": house and rnd.random() < 0.35})

    def _place_park(self, rnd):
        for x0, y0, x1, y1 in self.parks:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.obstacles.append({"type": "cylinder", "center": (cx, cy), "radius": 2.2, "look": "fountain"})
            for _ in range(40):
                x, y = rnd.uniform(x0 + 1.5, x1 - 1.5), rnd.uniform(y0 + 1.5, y1 - 1.5)
                if not self._too_close(x, y, 2.5):
                    self.obstacles.append({"type": "cylinder", "center": (x, y), "radius": 0.3, "look": "tree",
                                           "scale": rnd.uniform(0.9, 1.4)})
            for dx in (-4.5, 4.5):
                self.obstacles.append({"type": "box", "center": (cx + dx, cy - 4.0), "half_extents": (0.8, 0.25),
                                       "height": 0.85, "look": "bench", "yaw": math.pi})


# ======================================================================= small helpers
def _axis(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1, 0)
    return (0, 1 if dy > 0 else -1)


def _ray_meet(p0, h0, p2, h2):
    """Where the line through p0 (heading h0) meets the line through p2 (heading h2)."""
    d0 = (math.cos(h0), math.sin(h0))
    d2 = (math.cos(h2), math.sin(h2))
    den = d0[0] * d2[1] - d0[1] * d2[0]
    if abs(den) < 1e-9:
        return ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
    t = ((p2[0] - p0[0]) * d2[1] - (p2[1] - p0[1]) * d2[0]) / den
    return (p0[0] + d0[0] * t, p0[1] + d0[1] * t)


def _bezier(p0, p1, p2, t):
    u = 1 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def _overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _weighted_index(rng, weights):
    r = rng.uniform(0.0, sum(weights))
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i
    return len(weights) - 1


def _dedupe(steps):
    out = []
    for st in steps:
        if out and math.hypot(st[0] - out[-1][0], st[1] - out[-1][1]) < 0.05 and st[3] == "walk" \
                and out[-1][3] == "walk":
            continue
        out.append(st)
    return out
