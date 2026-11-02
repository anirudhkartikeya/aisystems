"""test_near_miss.py - checks the gap / contact / near-miss rule on simple shapes worked out by hand.

    python test_near_miss.py

The rule (metrics.py, used by the simulator, the metrics and the dataset checker):
  gap        = distance between the car's SURFACE and the person's SURFACE (not centre to centre)
  contact    = gap < 0
  near miss  = 0 <= gap < NEAR_MISS_DIST (0.5 m); the same person counts again only after
               the gap has opened past 0.5 + NEAR_MISS_REARM (0.3) = 0.8 m

The car is drawn as 3 circles in a row (robot.py): radius r = CAR_WIDTH/2 + 0.05 = 0.90 m,
centres at -1.0, 0.0, +1.0 m along the heading, so its front edge is 1.0 + 0.9 = 1.9 m
ahead of its centre and its side edge 0.9 m to the side. An adult is a circle of 0.25 m.
"""
import math

import config
from metrics import count_contacts_and_near_misses, surface_gap


def car_footprint(x, y, heading):
    """The same circles robot.py uses for the car (Robot(look="car").footprint())."""
    r = config.CAR_WIDTH / 2 + 0.05
    off = config.CAR_LENGTH / 2 - r
    c, s = math.cos(heading), math.sin(heading)
    return [(x + c * o, y + s * o, r) for o in (-off, 0.0, off)]


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {want}")
    return ok


def main():
    print(f"Car {config.CAR_LENGTH} x {config.CAR_WIDTH} m, person radius {config.PED_RADIUS} m, "
          f"NEAR_MISS_DIST {config.NEAR_MISS_DIST} m, re-arm {config.NEAR_MISS_REARM} m\n")
    foot = car_footprint(0.0, 0.0, 0.0)          # car at the origin, facing +x
    r_car = config.CAR_WIDTH / 2 + 0.05            # 0.90
    front = config.CAR_LENGTH / 2                  # 1.90: centre to front bumper
    pr = config.PED_RADIUS                         # 0.25
    ok = True

    print("1) Surface gap on simple shapes (by hand: centre distance - car circle - person radius)")
    # person straight ahead, 0.40 m in front of the bumper: centre at 1.9 + 0.25 + 0.40 = 2.55 m
    x = front + pr + 0.40
    ok &= check("ahead, centre 2.55 m away -> surface gap 0.40 m", round(surface_gap(foot, x, 0.0, pr), 9), 0.40)
    print(f"        (the centre-to-centre distance here is {x:.2f} m - a 'distance between centres' "
          f"rule would give a very different number)")
    ok &= check("ahead, 0.60 m in front of the bumper -> gap 0.60 m",
                round(surface_gap(foot, front + pr + 0.60, 0.0, pr), 9), 0.60)
    # person beside the car's middle, 0.30 m from the door: centre at 0.9 + 0.25 + 0.30 = 1.45 m
    ok &= check("beside the doors, centre 1.45 m to the side -> gap 0.30 m",
                round(surface_gap(foot, 0.0, r_car + pr + 0.30, pr), 9), 0.30)
    # diagonal: off the front circle at 45 degrees, 0.2 m surface gap
    d = r_car + pr + 0.20
    ok &= check("diagonal off the front corner -> gap 0.20 m",
                round(surface_gap(foot, 1.0 + d / math.sqrt(2), d / math.sqrt(2), pr), 9), 0.20)
    # overlapping: person centre 2.0 m ahead -> 2.0 - 1.0 - 0.9 - 0.25 = -0.15 (touching)
    ok &= check("ahead, centre 2.0 m away -> gap -0.15 m (contact)",
                round(surface_gap(foot, 2.0, 0.0, pr), 9), -0.15)
    # the car turned 90 degrees: the same person 2.55 m ahead along +y
    ok &= check("car facing +y, person 2.55 m ahead -> gap 0.40 m",
                round(surface_gap(car_footprint(0.0, 0.0, math.pi / 2), 0.0, 2.55, pr), 9), 0.40)

    print("\n2) Counting contacts and near misses over time (one person, one value per 0.1 s)")
    cases = [
        ("never closer than 0.5 m", [3.0, 2.0, 1.0, 0.6, 0.55, 1.0], (0, 0)),
        ("comes to 0.4 m once", [2.0, 1.0, 0.4, 0.45, 1.0], (0, 1)),
        ("0.4 -> 0.6 -> 0.4: gap never opened past 0.8 m, still ONE near miss", [1.0, 0.4, 0.6, 0.4, 1.0], (0, 1)),
        ("0.4 -> 0.9 -> 0.4: opened past 0.8 m in between, TWO near misses", [1.0, 0.4, 0.9, 0.4, 1.0], (0, 2)),
        ("touches (gap < 0): a contact; the 0.3 m on the way in is a near miss", [1.0, 0.3, -0.1, -0.2, 0.3, 1.0], (1, 1)),
        ("farther than 15 m (None) is skipped", [None, 0.4, None, 0.4], (0, 1)),
    ]
    for name, gaps, want in cases:
        ok &= check(name, count_contacts_and_near_misses(gaps), want)

    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
