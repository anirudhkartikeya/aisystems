"""reproduce_run.py - re-run ONE saved run and independently recount its near misses from the log.

    python reproduce_run.py run_data/v2                       # picks a run that has near misses
    python reproduce_run.py run_data/v2 reactive_s0006        # a run of your choice

Part A - reproduce: reads the run's brain, seed and varied settings from the saved files, runs it again
         with the same code, and compares the new run with the saved log step by step (car and every
         person), plus the scene fingerprint and the metrics.
Part B - independent recount: from the SAVED trajectory log only (positions, headings, person sizes),
         with its own few lines of arithmetic below (it does not use metrics.py), applies the written
         rule and lists every near miss, so each one can be checked by hand:

         gap       = distance between the car's surface and the person's surface. The car is 3 circles
                     of radius CAR_WIDTH/2 + 0.05 = 0.90 m, centred 1.0 m behind, at, and 1.0 m in front
                     of its centre along its heading; a person is a circle (0.25 m adult, 0.20 m child).
                     gap = min over the 3 circles of (centre distance - 0.90) - person radius
         contact   = gap < 0
         near miss = 0 <= gap < 0.5 m; the same person counts again only after the gap has
                     opened past 0.8 m (0.5 + 0.3)

Saves the report as <folder>/reproduce_<run_id>.txt.
"""
import csv
import gzip
import hashlib
import json
import math
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "run_data/v2"
wanted = sys.argv[2] if len(sys.argv) > 2 else None
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


man = json.load(open(os.path.join(folder, "manifest.json")))
cfg = man["config"]
runs = list(csv.DictReader(open(os.path.join(folder, "runs.csv"))))
if wanted is None:            # a run with near misses (the most interesting to check)
    wanted = max(runs, key=lambda r: (int(r["near_misses"]), int(r["ped_collisions"])))["run_id"]
run = next(r for r in runs if r["run_id"] == wanted)
scen = next(r for r in csv.DictReader(open(os.path.join(folder, "scenarios.csv"))) if r["seed"] == run["seed"])
people = {r["ped_id"]: r for r in csv.DictReader(open(os.path.join(folder, "people_info.csv")))
          if r["run_id"] == wanted}
say(f"Run {wanted} from {folder}: brain {run['brain']}, seed {run['seed']} ({run['split']}), "
    f"{run['steps']} steps, saved result: {run['result']}, near misses {run['near_misses']}, "
    f"contacts {run['ped_collisions']}")
say(f"Saved settings for this seed: {scen['background_people']} background people, {scen['traffic_cars']} cars, "
    f"P5 {scen.get('P5_side')}/{scen.get('P5_behaviour')}, scene fingerprint {scen['scene_hash']}")

robot, peds = [], []
with gzip.open(os.path.join(folder, "robot_steps.csv.gz"), "rt") as f:
    robot = [r for r in csv.DictReader(f) if r["run_id"] == wanted]
with gzip.open(os.path.join(folder, "people_steps.csv.gz"), "rt") as f:
    peds = [r for r in csv.DictReader(f) if r["run_id"] == wanted]

# =========================================================== A. reproduce the run
say("\nA. Reproduce the run from its saved settings")
here = os.path.dirname(os.path.abspath(__file__))
changed = [f for f, h in man["code_files_sha256"].items()
           if os.path.exists(os.path.join(here, f)) and sha256(os.path.join(here, f)) != h]
say(f"  code files changed since the dataset was made: {', '.join(changed) if changed else 'none'}")
import config                                                     # noqa: E402
from make_run_dataset import run_one, seed_variation              # noqa: E402

if man["layout"] in ("street", "town"):
    config.use_street_settings()
var = seed_variation(int(run["seed"]))
same_var = all(str(var[k]) == scen[k] or abs(float(var[k]) - float(scen[k])) < 1e-9 for k in var)
say(f"  varied settings re-derived from the seed = saved scenarios.csv row: {'YES' if same_var else 'NO'}")
res = run_one((run["brain"], int(run["seed"]), run["split"], man["layout"], man["scenario"]))
new = res["row"]
say(f"  scene fingerprint: saved {scen['scene_hash']}, re-run {res['scen']['scene_hash']} -> "
    f"{'SAME' if scen['scene_hash'] == res['scen']['scene_hash'] else 'DIFFERENT'}")
say(f"  steps: saved {run['steps']}, re-run {new['steps']}")
n = min(len(robot), len(res["robot"]))
d_car = max(math.hypot(float(robot[i]["x"]) - res["robot"][i][3], float(robot[i]["y"]) - res["robot"][i][4])
            for i in range(n))
saved_p = {(r["step"], r["ped_id"]): (float(r["x"]), float(r["y"])) for r in peds}
d_ped = max(math.hypot(saved_p[(str(p[1]), str(p[2]))][0] - p[3], saved_p[(str(p[1]), str(p[2]))][1] - p[4])
            for p in res["peds"] if (str(p[1]), str(p[2])) in saved_p)
say(f"  largest difference, car position over all steps: {d_car * 1000:.2f} mm")
say(f"  largest difference, any person over all steps:   {d_ped * 1000:.2f} mm  (log precision 0.1 mm)")
for k in ("result", "near_misses", "ped_collisions", "min_gap", "correct_actions"):
    say(f"  {k:16s} saved {run[k]:>8s}   re-run {str(new[k]):>8s}   {'same' if str(new[k]) == run[k] else 'DIFFERENT'}")
reproduced = (scen["scene_hash"] == res["scen"]["scene_hash"] and int(run["steps"]) == new["steps"]
              and d_car < 1e-3 and d_ped < 1e-3)
say(f"  -> {'REPRODUCED exactly' if reproduced else 'NOT reproduced exactly (see the differences above)'}")

# =========================================================== B. independent recount from the log
say("\nB. Near misses recounted from the saved trajectory log (own arithmetic, written rule)")
R = cfg["CAR_WIDTH"] / 2 + 0.05                 # car circle radius
OFF = cfg["CAR_LENGTH"] / 2 - R                  # circle centres at -OFF, 0, +OFF along the heading
NEAR, REARM = 0.5, 0.3
say(f"  car circles: radius {R:.2f} m at {-OFF:+.2f}, 0, {OFF:+.2f} m along the heading; "
    f"near miss < {NEAR} m, re-armed above {NEAR + REARM} m")
car = {r["step"]: r for r in robot}
series = {}
for r in peds:
    c = car[r["step"]]
    x, y, h = float(c["x"]), float(c["y"]), float(c["heading"])
    px, py = float(r["x"]), float(r["y"])
    rad = float(people[r["ped_id"]]["radius"])
    gaps = []
    for o in (-OFF, 0.0, OFF):
        cx, cy = x + math.cos(h) * o, y + math.sin(h) * o
        gaps.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2) - R - rad)
    series.setdefault(r["ped_id"], []).append((int(r["step"]), float(c["t"]), min(gaps), float(c["speed"])))
events = []                                       # (kind, person, t start, closest gap, t closest, car speed)
for pid, ser in series.items():
    ser.sort()
    in_contact = armed_off = False
    current = None
    for step, t, g, v in ser:
        if g < 0:
            if not in_contact:
                events.append(["CONTACT", pid, t, g, t, v])
                in_contact = True
            elif g < events[-1][3] and events[-1][1] == pid:
                events[-1][3:6] = [g, t, v]
            continue
        in_contact = False
        if g < NEAR:
            if not armed_off:
                current = ["near miss", pid, t, g, t, v]
                events.append(current)
                armed_off = True
            elif current is not None and current[1] == pid and g < current[3]:
                current[3:6] = [g, t, v]
        elif g > NEAR + REARM:
            armed_off = False
say(f"\n  {'event':10s} {'person':18s} {'starts at':>10s} {'closest gap':>12s} {'at':>8s} {'car speed':>10s}")
for kind, pid, t0, g, tg, v in sorted(events, key=lambda e: e[2]):
    who = people[pid]
    name = who["name"] or f"{who['kind']} #{pid}"
    say(f"  {kind:10s} {name:18s} {t0:9.1f}s {g:11.3f}m {tg:7.1f}s {v:8.2f}m/s")
n_near = sum(1 for e in events if e[0] == "near miss")
n_contact = sum(1 for e in events if e[0] == "CONTACT")
say(f"\n  recounted from the log: near misses {n_near}, contacts {n_contact}")
say(f"  saved in runs.csv:      near misses {run['near_misses']}, contacts {run['ped_collisions']}")
ok = n_near == int(run["near_misses"]) and n_contact == int(run["ped_collisions"])
say(f"  -> {'MATCH' if ok else 'MISMATCH'}")
near_events = [e for e in events if e[0] == "near miss"] or events
if near_events:
    kind, pid, t0, g, tg, v = min(near_events, key=lambda e: e[3])
    c = next(r for r in robot if abs(float(r["t"]) - tg) < 1e-6)
    p = next(r for r in peds if r["ped_id"] == pid and r["step"] == c["step"])
    x, y, h = float(c["x"]), float(c["y"]), float(c["heading"])
    px, py, rad = float(p["x"]), float(p["y"]), float(people[pid]["radius"])
    say(f"\n  Check one {kind} by hand (its closest moment, t = {tg:.1f} s):")
    say(f"    car centre ({x:.4f}, {y:.4f}), heading {h:.4f} rad; person ({px:.4f}, {py:.4f}), radius {rad:.2f} m")
    for o, label in ((-OFF, "rear"), (0.0, "middle"), (OFF, "front")):
        cx, cy = x + math.cos(h) * o, y + math.sin(h) * o
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        say(f"    {label:6s} circle centre ({cx:.4f}, {cy:.4f}): distance {dist:.4f} m -> "
            f"gap {dist:.4f} - {R:.2f} - {rad:.2f} = {dist - R - rad:.3f} m")
    say(f"    smallest = {g:.3f} m {'< 0.5 m -> near miss' if 0 <= g < NEAR else '< 0 -> contact'}")

open(os.path.join(folder, f"reproduce_{wanted}.txt"), "w").write("\n".join(out_lines) + "\n")
print(f"\nSaved {folder}/reproduce_{wanted}.txt")
