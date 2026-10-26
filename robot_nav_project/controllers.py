"""controllers.py - simple stand-in drivers until the brains and planner exist.

placeholder_controller follows the route's waypoints (through junctions and
round curves), slows for bends, stops at red lights, lets oncoming traffic
through before turning left and keeps a gap to cars on its path. It IGNORES
pedestrians unless watch_people=True, so in the crossing scenario it will hit
the scripted people - that is exactly what the reactive and predictive brains
plus the planner have to fix.

Every call also leaves sim.drive_status: why the car is going at the speed it
is ("driving", "red light", "waiting for oncoming traffic", ...). The GUI shows it.
"""
import math

import config


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def placeholder_controller(sim, state, waypoints, watch_people=False):
    rx, ry = state["robot"]["x"], state["robot"]["y"]
    car = sim.world.roadmap is not None
    look = 3.0 if car else 0.3                   # aim this far ahead along the route
    while len(waypoints) > 1 and math.hypot(waypoints[0][0] - rx, waypoints[0][1] - ry) < look:
        waypoints.pop(0)
    tx, ty = waypoints[0]
    dist = math.hypot(tx - rx, ty - ry)
    if dist < 1e-6:
        sim.drive_status = "at the goal"
        return 0.0, 0.0
    hx, hy = (tx - rx) / dist, (ty - ry) / dist
    remaining = dist + sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(waypoints, waypoints[1:]))
    speed, status = config.ROBOT_MAX_SPEED, "driving"

    def limit(value, why):
        nonlocal speed, status
        if value < speed:
            speed = max(0.0, value)
            if value < config.ROBOT_MAX_SPEED - 0.1:     # tiny trims don't count as a reason
                status = why

    limit(remaining, "arriving at the goal")
    if car:
        heading = math.atan2(hy, hx)
        # slow down for bends: compare the direction now with the route 8 m ahead
        far = waypoints[min(len(waypoints) - 1, 8)]
        fx, fy = far[0] - rx, far[1] - ry
        n = math.hypot(fx, fy)
        if n > 1e-6:
            limit(config.ROBOT_MAX_SPEED * max(0.45, (fx * hx + fy * hy) / n), "slowing for the bend")
        stop = sim.light_ahead()                 # red light: stop at the line
        if stop is not None:
            limit((stop - 0.5) * 0.8, "red light")
        # turning left: let oncoming traffic through first
        ahead = waypoints[min(len(waypoints) - 1, 20)]
        turn = _wrap(math.atan2(ahead[1] - ry, ahead[0] - rx) - heading)
        if turn > 0.5:
            wait = sim.oncoming_before_left_turn()
            if wait is not None:
                limit((wait - 0.5) * 0.8, "waiting for oncoming traffic to turn left")
        # changing lane: look in the mirror. A car in the lane we move into that is
        # alongside or just ahead -> ease off and slot in behind it; a faster car
        # coming up from behind -> ease off a little and let it pass. Never a full
        # stop in the middle of the road just to change lane.
        i8 = min(len(waypoints) - 1, 8)
        far8, after8 = waypoints[i8 - 1] if i8 > 0 else (rx, ry), waypoints[i8]
        shift = -(after8[0] - rx) * hy + (after8[1] - ry) * hx      # + = route drifts to our left
        straight = abs(_wrap(math.atan2(after8[1] - far8[1], after8[0] - far8[0]) - heading)) < 0.12
        if abs(shift) > 0.4 and abs(turn) < 0.5 and straight:     # sideways but not a bend: a lane change
            if status == "driving":
                status = "changing lane (getting ready to turn left)" if shift > 0 else "changing lane"
            for v in state["vehicles"]:
                if abs(_wrap(v["heading"] - heading)) > 0.6:
                    continue                             # not going our way
                along = (v["x"] - rx) * hx + (v["y"] - ry) * hy
                side = (-(v["x"] - rx) * hy + (v["y"] - ry) * hx) * math.copysign(1.0, shift)
                if not 1.5 < side < 4.5:
                    continue                             # not in the lane we move into
                v_speed = math.hypot(v["vx"], v["vy"])
                if -4.0 < along < 7.0:
                    if v_speed < 0.5:                    # a queue in that lane: wait our turn
                        limit(0.0, "waiting to change lane (queue in that lane)")
                    else:
                        limit(max(0.5, v_speed - 0.8), "letting a car pass before changing lane")
                elif -15.0 < along <= -4.0 and v_speed > 1.0:
                    limit(0.6 * config.ROBOT_MAX_SPEED, "letting a car pass before changing lane")
        # cars (and, if asked, people) on the route ahead: slow down, stop if close
        things = [(v["x"], v["y"], v["width"] / 2 + 0.3, v["length"] / 2, "car ahead") for v in state["vehicles"]]
        if watch_people:
            things += [(p["x"], p["y"], 0.6, 0.3, "person ahead") for p in state["pedestrians"]]
        path = [(rx, ry)] + waypoints[:16]
        front = config.CAR_LENGTH / 2
        for x, y, half_w, half_l, why in things:
            if abs(x - rx) > 20 or abs(y - ry) > 20:
                continue
            along = 0.0
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                seg = math.hypot(bx - ax, by - ay)
                if seg < 1e-6:
                    continue
                t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (seg * seg)))
                if math.hypot(x - (ax + (bx - ax) * t), y - (ay + (by - ay) * t)) < config.CAR_WIDTH / 2 + half_w:
                    gap = along + t * seg - front - half_l - 1.5
                    limit(gap * 0.7, why)
                    break
                along += seg
    sim.drive_status = status
    return hx * speed, hy * speed
