"""check_run_dataset.py - checks a saved run dataset and recomputes the reported metrics from its logs.

    python check_run_dataset.py run_data/v1

Uses ONLY the saved files (plus the gap / near-miss rule in metrics.py). Checks:
  1. files      every file listed in manifest.json exists and has the same SHA-256; code unchanged?
  2. counts     runs per brain and split, steps, people per step, no missing / repeated steps
  3. split      dev and eval seeds don't overlap; every run is in the split split.csv says
  4. finite     every numeric field: any NaN / infinite values? (which ones, and why)
  5. variation  the range of every varied setting over the seeds; P5's choices
  6. duplicates no two seeds have the same scene; both brains got the SAME scene for a seed;
                no two runs drove the same path
  7. recompute  from the per-step logs: surface gaps from the raw positions, then contacts, near
                misses and min gaps -> compared with runs.csv / crossers.csv
  8. table      the reactive vs predictive comparison computed from the logs, dev and eval separately
Writes <folder>/checks_report.txt and checks.json.
"""
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from metrics import count_contacts_and_near_misses

folder = sys.argv[1] if len(sys.argv) > 1 else "run_data/v1"
lines, results = [], []


def say(s=""):
    print(s)
    lines.append(s)


def verdict(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def read(name):
    path = os.path.join(folder, name)
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


man = json.load(open(os.path.join(folder, "manifest.json")))
cfg = man["config"]
say(f"Dataset {folder}  (made {man['created']}, code version {man['code_version']}, brains {man['brains']})")

# ------------------------------------------------------------------ 1. files
say("\n1. Files and versions")
bad = [n for n, meta in man["files"].items()
       if not os.path.exists(os.path.join(folder, n)) or sha256(os.path.join(folder, n)) != meta["sha256"]]
verdict("every file in the manifest exists with the same SHA-256", not bad, ", ".join(bad) or f"{len(man['files'])} files")
here = os.path.dirname(os.path.abspath(__file__))
changed = [f for f, h in man["code_files_sha256"].items()
           if os.path.exists(os.path.join(here, f)) and sha256(os.path.join(here, f)) != h]
say(f"  [info] code files changed since the dataset was made: {', '.join(changed) if changed else 'none'}")

split = {int(r["seed"]): r["split"] for r in read("split.csv")}
scen = read("scenarios.csv")
runs = read("runs.csv")
crossers = read("crossers.csv")
info = read("people_info.csv")
say("  reading the per-step logs ...")
robot = read("robot_steps.csv.gz")
peds = read("people_steps.csv.gz")
vehicles = read("vehicle_steps.csv.gz")

# ------------------------------------------------------------------ 2. counts
say("\n2. Counts")
by_brain_split = Counter((r["brain"], r["split"]) for r in runs)
for (b, s), n in sorted(by_brain_split.items()):
    say(f"  {b:11s} {s:4s}: {n} runs")
say(f"  seeds: {len(scen)} | runs: {len(runs)} | robot steps: {len(robot)} | person steps: {len(peds)} | "
    f"vehicle steps: {len(vehicles)}")
steps_of = defaultdict(list)
for r in robot:
    steps_of[r["run_id"]].append(int(r["step"]))
bad = [r["run_id"] for r in runs if sorted(steps_of[r["run_id"]]) != list(range(1, int(r["steps"]) + 1))]
verdict("each run's steps are 1..N with none missing or repeated, N = runs.csv steps", not bad, ", ".join(bad[:5]))
people_of = Counter(r["run_id"] for r in info)
ped_rows = Counter(r["run_id"] for r in peds)
bad = [r["run_id"] for r in runs if ped_rows[r["run_id"]] != people_of[r["run_id"]] * int(r["steps"])]
verdict("every person is logged at every step", not bad, ", ".join(bad[:5]))
brains = man["brains"]
seeds_per_brain = {b: sorted(int(r["seed"]) for r in runs if r["brain"] == b) for b in brains}
same = all(seeds_per_brain[b] == seeds_per_brain[brains[0]] for b in brains)
verdict("every brain drove exactly the same seeds", same, f"{len(seeds_per_brain[brains[0]])} seeds each")

# ------------------------------------------------------------------ 3. split
say("\n3. Dev / eval split (whole runs, fixed in split.csv before the runs)")
dev = {s for s, v in split.items() if v == "dev"}
ev = {s for s, v in split.items() if v == "eval"}
verdict("dev and eval seeds don't overlap", not (dev & ev), f"dev {min(dev)}-{max(dev)} ({len(dev)}), "
        f"eval {min(ev)}-{max(ev)} ({len(ev)})" if dev and ev else "")
bad = [r["run_id"] for r in runs if split.get(int(r["seed"])) != r["split"]]
verdict("every run carries the split that split.csv gives its seed", not bad, ", ".join(bad[:5]))

# ------------------------------------------------------------------ 4. finite values
say("\n4. Finite values in every numeric field")
allowed = {"reaction_gap": "the car never reacted to a scripted person in that run",
           "ade": "no prediction could be scored (person left the recorded area)", "fde": "same as ade"}
any_bad = False
for name, rows in [("runs.csv", runs), ("crossers.csv", crossers), ("scenarios.csv", scen),
                   ("people_info.csv", info), ("robot_steps.csv.gz", robot), ("people_steps.csv.gz", peds),
                   ("vehicle_steps.csv.gz", vehicles)]:
    cols = [c for c in rows[0] if c not in ("run_id",)]
    for c in cols:
        vals = [num(r[c]) for r in rows]
        numeric = [v for v in vals if v is not None]
        if len(numeric) < len(vals) * 0.9:        # a text column
            continue
        nan = sum(1 for v in numeric if math.isnan(v))
        inf = sum(1 for v in numeric if math.isinf(v))
        if nan or inf:
            ok = inf == 0 and c in allowed and not name.endswith(".gz")
            any_bad |= not ok
            verdict(f"{name}: {c}", ok, f"{nan} NaN, {inf} infinite of {len(numeric)}"
                    + (f" (allowed: {allowed[c]})" if c in allowed else ""))
verdict("no unexpected NaN / infinite values anywhere", not any_bad,
        "all numeric fields in all 7 files checked; min_ttc uses 99 = 'never closing in'")

# ------------------------------------------------------------------ 5. variation
say("\n5. Scenario variation over the seeds (scenarios.csv)")
say(f"  {'setting':24s} {'min':>8s} {'max':>8s} {'distinct':>9s}")
for c in scen[0]:
    if c in ("seed", "split", "layout", "scenario", "scene_hash"):
        continue
    vals = [num(r[c]) for r in scen]
    if all(v is not None for v in vals):
        say(f"  {c:24s} {min(vals):8.3f} {max(vals):8.3f} {len(set(vals)):9d}")
    else:
        say(f"  {c:24s} {dict(Counter(r[c] for r in scen))}")
p5 = Counter((r.get("P5_side"), r.get("P5_behaviour")) for r in scen)
say(f"  P5 (side, behaviour) combinations: {dict(p5)}")
vary_cols = [c for c in scen[0] if c.endswith(("_timing_s", "_speed_ms")) or c in ("background_people", "traffic_cars")]
verdict("the varied settings really differ between seeds",
        all(len({r[c] for r in scen}) > 1 for c in vary_cols) or len(scen) < 2, f"{len(vary_cols)} settings checked")

# ------------------------------------------------------------------ 6. duplicates
say("\n6. Duplicates")
hashes = [r["scene_hash"] for r in scen]
verdict("no two seeds have the same scene (scene fingerprint)", len(set(hashes)) == len(hashes),
        f"{len(set(hashes))} different scenes for {len(hashes)} seeds")
starts = defaultdict(list)
for r in info:
    starts[r["run_id"]].append((r["ped_id"], r["start_x"], r["start_y"]))
by_seed = defaultdict(set)
for r in runs:
    by_seed[r["seed"]].add((r["scene_hash"], tuple(sorted(starts[r["run_id"]]))))
verdict("for each seed, every brain got the SAME scene (a controlled comparison)",
        all(len(v) == 1 for v in by_seed.values()), f"{len(by_seed)} seeds")
start_sets = [next(iter(v))[1] for v in by_seed.values()]
verdict("no two seeds start with identical people positions", len(set(start_sets)) == len(start_sets))
traj = defaultdict(list)
for r in robot:
    traj[r["run_id"]].append(r["x"] + "," + r["y"])
th = [hashlib.sha256("|".join(v).encode()).hexdigest() for v in traj.values()]
verdict("no two runs drove exactly the same path", len(set(th)) == len(th), f"{len(th)} runs")
dup_steps = len(peds) - len({(r["run_id"], r["step"], r["ped_id"]) for r in peds})
verdict("no repeated (run, step, person) rows in people_steps", dup_steps == 0)

# ------------------------------------------------------------------ 7. recompute from the logs
say("\n7. Metrics recomputed from the per-step logs")
r_car = cfg["CAR_WIDTH"] / 2 + 0.05
off = cfg["CAR_LENGTH"] / 2 - r_car
radius = {(r["run_id"], r["ped_id"]): float(r["radius"]) for r in info}
pose = {(r["run_id"], r["step"]): (float(r["x"]), float(r["y"]), float(r["heading"]), r["done"] == "1") for r in robot}
series = defaultdict(list)             # (run, ped) -> [(step, gap_recomputed, gap_logged, in15, in6, done)]
max_diff = 0.0
for r in peds:
    x, y, h, done = pose[(r["run_id"], r["step"])]
    px, py = float(r["x"]), float(r["y"])
    c, s = math.cos(h), math.sin(h)
    gap = min(math.hypot(x + c * o - px, y + s * o - py) - r_car for o in (-off, 0.0, off)) \
        - radius[(r["run_id"], r["ped_id"])]
    logged = float(r["gap_to_car"])
    max_diff = max(max_diff, abs(gap - logged))
    series[(r["run_id"], r["ped_id"])].append((int(r["step"]), gap, logged,
                                               abs(px - x) <= 15 and abs(py - y) <= 15,
                                               abs(px - x) <= 6 and abs(py - y) <= 6, done))
verdict("surface gap recomputed from raw positions = logged gap", max_diff < 0.003,
        f"largest difference {max_diff * 1000:.2f} mm (positions are saved to 0.1 mm)")
re = defaultdict(lambda: {"contacts": 0, "near": 0, "min6": math.inf})
for (rid, pid), ser in series.items():
    ser.sort()
    gaps = [g if (in15 and not done) else None for _, g, _, in15, _, done in ser]
    c_, n_ = count_contacts_and_near_misses(gaps)
    re[rid]["contacts"] += c_
    re[rid]["near"] += n_
    m6 = min((g for _, g, _, _, in6, _ in ser if in6), default=math.inf)
    re[rid]["min6"] = min(re[rid]["min6"], m6)
mis_c = [r["run_id"] for r in runs if re[r["run_id"]]["contacts"] != int(r["ped_collisions"])]
mis_n = [r["run_id"] for r in runs if re[r["run_id"]]["near"] != int(r["near_misses"])]
verdict("people contacts per run: logs = runs.csv", not mis_c,
        f"{len(runs) - len(mis_c)}/{len(runs)} runs match, total {sum(v['contacts'] for v in re.values())}")
verdict("near misses per run: logs = runs.csv", not mis_n,
        f"{len(runs) - len(mis_n)}/{len(runs)} runs match, total {sum(v['near'] for v in re.values())}")
name_to_pid = {(r["run_id"], r["name"]): r["ped_id"] for r in info if r["role"] == "scripted"}
diffs = []
for r in crossers:
    ser = series[(r["run_id"], name_to_pid[(r["run_id"], r["name"])])]
    diffs.append(abs(min(g for _, g, _, _, _, _ in ser) - float(r["min_gap"])))
verdict("each scripted person's closest gap: logs = crossers.csv", max(diffs) < 0.003,
        f"{len(diffs)} people, largest difference {max(diffs) * 1000:.2f} mm")
d6 = [re[r["run_id"]]["min6"] - float(r["min_gap"]) for r in runs]
verdict("run min gap: logs (every 0.1 s) vs runs.csv (every 0.01 s)", min(d6) > -0.003,
        f"logs are never smaller (as expected); mean difference {np.mean(d6) * 100:.1f} cm, "
        f"largest {max(d6) * 100:.1f} cm - the simulator also checks between the 0.1 s steps")

# ------------------------------------------------------------------ 8. the comparison, from the logs
say("\n8. Reactive vs predictive, computed from the per-step logs only")
goal = cfg.get("TOWN_GOAL") if man["layout"] == "town" else cfg.get("STREET_GOAL")
last = {}
for r in robot:
    if r["done"] == "1" or r["run_id"] not in last or int(r["step"]) > int(last[r["run_id"]]["step"]):
        last[r["run_id"]] = r
table = {}
for sp in ("dev", "eval"):
    say(f"\n  {sp.upper()} seeds")
    say(f"  {'brain':11s} {'runs':>5s} {'reached goal':>13s} {'runs hitting a person':>22s} {'contacts':>9s} "
        f"{'near misses/run':>16s} {'min gap (m)':>12s} {'time (s)':>9s}")
    for b in brains:
        rr = [r for r in runs if r["brain"] == b and r["split"] == sp]
        if not rr:
            continue
        ids = [r["run_id"] for r in rr]
        reached = [math.hypot(float(last[i]["x"]) - goal[0], float(last[i]["y"]) - goal[1]) < cfg["GOAL_TOLERANCE"] + 0.05
                   for i in ids]
        hit = [re[i]["contacts"] > 0 for i in ids]
        row = {"runs": len(ids), "reached_goal_pct": 100 * np.mean(reached), "runs_hitting_person_pct": 100 * np.mean(hit),
               "contacts": sum(re[i]["contacts"] for i in ids), "near_misses_per_run": np.mean([re[i]["near"] for i in ids]),
               "min_gap_m": float(np.mean([re[i]["min6"] for i in ids])),
               "time_s": float(np.mean([float(last[i]["t"]) for i in ids]))}
        table[f"{sp}/{b}"] = row
        say(f"  {b:11s} {row['runs']:5d} {row['reached_goal_pct']:12.0f}% {row['runs_hitting_person_pct']:21.0f}% "
            f"{row['contacts']:9d} {row['near_misses_per_run']:16.2f} {row['min_gap_m']:12.2f} {row['time_s']:9.1f}")

n_fail = sum(1 for r in results if not r["ok"])
say(f"\n{len(results) - n_fail}/{len(results)} checks passed" + (f", {n_fail} FAILED" if n_fail else ""))
open(os.path.join(folder, "checks_report.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"dataset": folder, "code_version": man["code_version"], "checks": results, "table_from_logs": table},
          open(os.path.join(folder, "checks.json"), "w"), indent=1)
print(f"Saved {folder}/checks_report.txt and checks.json")
sys.exit(1 if n_fail else 0)
