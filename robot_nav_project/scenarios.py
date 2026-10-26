"""scenarios.py - who is in the scene and where the robot starts and ends.

random   : pedestrians go about their business, traffic drives around
crossing : scripted people cross the robot's path right in front of it,
           timed so they meet it - the "person steps into the road" test -
           with normal pedestrians and traffic around them
"""
import math

import config
from pedestrians import CrossingPedestrian, spawn_pedestrians
import visuals

SCENARIOS = ("crossing", "random")


def start_and_goal(layout, scenario):
    """Robot start and goal. On the street layouts the robot is a car
    driving along the right-hand lane."""
    if layout == "town":
        return config.TOWN_START, config.TOWN_GOAL
    if layout == "street":
        return config.STREET_START, config.STREET_GOAL
    if scenario == "crossing" and layout == "road":
        return config.ROAD_CAR_START, config.ROAD_CAR_GOAL
    return config.ROBOT_START, config.GOAL


def per_layout(value, layout):
    """Config values can be one number or a {layout: number} table."""
    return value.get(layout, 0) if isinstance(value, dict) else value


def _pick(value, choices, rng):
    return choices[int(rng.integers(len(choices)))] if value == "random" else value


def street_crosser_specs(world, path, rng):
    """The scripted people on the robot's route (town: config.TOWN_CROSSERS,
    street: one per zebra crossing). Works out where each starts (1 m in from
    the curb on their side) and ends (the other sidewalk), relative to the
    robot's route, and settles any "random" choices for this seed."""
    from roadmap import Polyline
    if world.layout == "town":
        wanted = [dict(c) for c in config.TOWN_CROSSERS]
    else:
        wanted = [{"name": f"P{k + 1}", "at": (x, 0.0), "side": side, "behaviour": "cross", "contact": 0.0,
                   "expected": "stop", "zebra": True}
                  for k, (x, side) in enumerate(zip(config.STREET_CROSSWALKS_X, config.STREET_CROSS_SIDES))]
    line = Polyline(path)
    specs = []
    for c in wanted:
        x, y = c["at"]
        s, d, _ = line.project(x, y)
        if d > 6.0 or s <= 1.0 or s >= line.length - 1.0:
            continue                                  # not on the robot's route
        side = _pick(c.get("side", "right"), ["left", "right"], rng)
        behaviour = _pick(c.get("behaviour", "cross"), ["stop", "cross"], rng)
        contact = c.get("contact", 0.0)
        if contact == "random":
            contact = float(rng.uniform(-1, 1) * config.CROSSER_RANDOM_CONTACT)
        mx, my, hdg = line.point_at(s)
        left = (-math.sin(hdg), math.cos(hdg))
        # the road here: where its centre line is, seen from the robot's lane
        edge = min(world.roadmap.edges, key=lambda e: e.center.project(mx, my)[1])
        cx, cy, _ = edge.center.point_at(edge.center.project(mx, my)[0])
        lat_c = (cx - mx) * left[0] + (cy - my) * left[1]
        reach = edge.half + 1.0
        to_left, to_right = lat_c + reach, reach - lat_c
        specs.append({"at": s / line.length, "side": side, "behaviour": behaviour, "contact": contact,
                      "name": c.get("name", f"P{len(specs) + 1}"), "expected": c.get("expected", "stop"),
                      "zebra": c.get("zebra", False),
                      "start_offset": to_left if side == "left" else to_right,
                      "end_offset": to_right if side == "left" else to_left,
                      "speed": config.STREET_CROSS_SPEED})
    return specs


def point_on_path(path, fraction):
    """Point `fraction` (0..1) of the way along a polyline, and the unit
    direction of travel there."""
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    target = max(0.0, min(1.0, fraction)) * sum(lengths)
    last = len(lengths) - 1
    for i, ((a, b), seg) in enumerate(zip(zip(path, path[1:]), lengths)):
        if seg == 0:
            continue
        if target <= seg or i == last:
            t = min(1.0, target / seg)
            dx, dy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), (dx, dy)
        target -= seg
    raise ValueError("the robot's path has no length")


def make_crossers(world, rng, robot_start, robot_goal, look_rng):
    """Build the scripted crossers for this layout."""
    from roadmap import Polyline
    path = [tuple(robot_start)] + world.route(robot_start, robot_goal)
    route = Polyline(path)
    crossers = []
    if world.roadmap is not None:
        specs = street_crosser_specs(world, path, rng)
    else:
        specs = [dict(c, name=f"P{k + 1}", behaviour="cross", contact=0.0, expected="avoid", zebra=False)
                 for k, c in enumerate(config.CROSSERS[world.layout])]
    for i, spec in enumerate(specs):
        on_path, (dx, dy) = point_on_path(path, spec["at"])
        left = (-dy, dx)                                   # robot's left-hand side
        sign = 1 if spec["side"] == "left" else -1
        start = (on_path[0] + sign * spec["start_offset"] * left[0],
                 on_path[1] + sign * spec["start_offset"] * left[1])
        end = (on_path[0] - sign * spec["end_offset"] * left[0],
               on_path[1] - sign * spec["end_offset"] * left[1])
        # where they meet the car's front: `contact` metres left of the car's centre line
        meet = (on_path[0] + spec["contact"] * left[0], on_path[1] + spec["contact"] * left[1])
        if not world.line_is_free(start, end, config.PED_RADIUS):
            raise ValueError(
                f"Crosser {i} in layout '{world.layout}': the line it walks hits a wall or "
                f"obstacle (start {start[0]:.2f},{start[1]:.2f} -> end {end[0]:.2f},{end[1]:.2f}). "
                f"Change 'at', 'side' or the offsets in config.")

        # Small per-seed variation, for experiments (zero by default).
        speed = spec["speed"] * (1 + rng.uniform(-1, 1) * config.CROSSING_SPEED_JITTER)
        timing = config.CROSSING_TIMING + rng.uniform(-1, 1) * config.CROSSING_TIMING_JITTER
        if spec["behaviour"] == "stop":
            timing -= config.CROSSER_STOP_LEAD             # be standing in the lane before the car arrives

        # The person steps out when the robot is as many seconds away as they
        # need to reach the meeting point (minus the timing offset), so both arrive together.
        look = visuals.random_person_look(look_rng, "adult", look_rng.uniform(1.6, 1.85))
        ped = CrossingPedestrian(i, start, end, meet, speed, timing, rng, world, look, path=route,
                                 behaviour=spec["behaviour"])
        ped.plan = {k: spec[k] for k in ("name", "side", "behaviour", "contact", "expected", "zebra")}
        crossers.append(ped)
    return crossers


def make_pedestrians(scenario, world, rng, robot_start, robot_goal, look_rng=None):
    """All pedestrians for one trial."""
    look_rng = look_rng if look_rng is not None else rng
    if scenario == "crossing":
        crossers = make_crossers(world, rng, robot_start, robot_goal, look_rng)
        n = per_layout(config.CROSSING_BACKGROUND_PEDS, world.layout)
        background = spawn_pedestrians(n, rng, world, robot_start, robot_goal,
                                       existing=crossers, look_rng=look_rng)
        return crossers + background
    if scenario == "random":
        n = per_layout(config.N_PEDS, world.layout)
        return spawn_pedestrians(n, rng, world, robot_start, robot_goal, look_rng=look_rng)
    raise ValueError(f"Unknown scenario '{scenario}', use one of {SCENARIOS}")


def n_cars(scenario, layout):
    table = config.CROSSING_CARS if scenario == "crossing" else config.N_CARS
    return per_layout(table, layout)
