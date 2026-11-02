"""make_run_dataset.py - the saved dataset of the core comparison (reactive vs predictive car).

    python make_run_dataset.py                       # 20 dev + 10 eval seeds x 2 brains -> run_data/v1/
    python make_run_dataset.py --split dev           # only the dev seeds (use this while tuning)
    python make_run_dataset.py --brains reactive predictive lstm --tag v3 --eval-start 2000 --eval-count 20
    python check_run_dataset.py run_data/v1          # then check it and recompute the metrics from the logs

Every brain drives exactly the same seeds with exactly the same scene. Uses the simulator's perfect
positions of people (no camera, no YOLO): it tests the PLANNING, not camera-based driving.

Scenario variation (all drawn from the seed, the same for every brain, saved in scenarios.csv):
  - crossers P1-P4 step out up to +/-1.0 s early/late (CROSSING_TIMING_JITTER) and walk +/-20 %
    faster/slower (CROSSING_SPEED_JITTER); P5 picks its side, stop-or-cross and contact point at random
  - 10-24 background people and 4-12 traffic cars
  - the people's behaviour mix: share aware of cars, on the phone, jaywalkers, pausing, turning back

Seeds: dev 0-19 (planner tuning may only look at these), eval = a block of seeds never run before
(v1: 1000-1009; v2: --eval-start 2000 --eval-count 20), looked at once, for the final numbers. The assignment is written to split.csv BEFORE any
run, and is never overwritten.

Saved in run_data/<tag>/:
  split.csv              seed -> dev / eval
  scenarios.csv          one row per seed: every varied setting + each scripted person's plan
  runs.csv               one row per run (brain x seed): result and every metric (metrics.py)
  crossers.csv           one row per scripted person per run: expected vs actual action, min gap
  people_info.csv        one row per person per run: role, kind, radius, personality, start point
  robot_steps.csv.gz     the car every 0.1 s: position, heading, speed, command, planner status
  people_steps.csv.gz    every person every 0.1 s: position, velocity, surface gap to the car
  vehicle_steps.csv.gz   every traffic car every 0.1 s: position, heading, speed
  manifest.json          when, how (command, versions), the code version (hash of every .py file),
                         the full config, the variation ranges and every file with its hash and size
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from multiprocessing import get_context

import numpy as np

import config

DEV_SEEDS = list(range(0, 20))
EVAL_SEEDS = list(range(1000, 1010))

# the documented variation, per seed (ranges saved in the manifest)
VARIATION = {
    "CROSSING_TIMING_JITTER": 1.0,                 # s, +/- for each scripted crosser
    "CROSSING_SPEED_JITTER": 0.2,                  # fraction, +/- for each scripted crosser
    "background_people": (10, 24),                 # inclusive
    "traffic_cars": (4, 12),
    "PED_AWARE_SHARE": (0.6, 0.9),
    "PED_PHONE_SHARE": (0.05, 0.25),
    "PED_JAYWALKER_SHARE": (0.0, 0.25),
    "PED_PAUSE_RATE": (0.01, 0.04),
    "PED_TURN_BACK_PROB": (0.0, 0.08),
}


def seed_variation(seed):
    """The varied settings for one seed (its own random stream: the same for every brain)."""
    rng = np.random.default_rng([seed, 4242])
    v = VARIATION
    return {
        "background_people": int(rng.integers(v["background_people"][0], v["background_people"][1] + 1)),
        "traffic_cars": int(rng.integers(v["traffic_cars"][0], v["traffic_cars"][1] + 1)),
        "PED_AWARE_SHARE": round(float(rng.uniform(*v["PED_AWARE_SHARE"])), 4),
        "PED_PHONE_SHARE": round(float(rng.uniform(*v["PED_PHONE_SHARE"])), 4),
        "PED_JAYWALKER_SHARE": round(float(rng.uniform(*v["PED_JAYWALKER_SHARE"])), 4),
        "PED_PAUSE_RATE": round(float(rng.uniform(*v["PED_PAUSE_RATE"])), 4),
        "PED_TURN_BACK_PROB": round(float(rng.uniform(*v["PED_TURN_BACK_PROB"])), 4),
        "CROSSING_TIMING_JITTER": v["CROSSING_TIMING_JITTER"],
        "CROSSING_SPEED_JITTER": v["CROSSING_SPEED_JITTER"],
    }


def apply_variation(var, layout):
    old = {}
    for k, val in var.items():
        if k == "background_people":
            old["CROSSING_BACKGROUND_PEDS"] = config.CROSSING_BACKGROUND_PEDS
            config.CROSSING_BACKGROUND_PEDS = dict(config.CROSSING_BACKGROUND_PEDS, **{layout: val})
        elif k == "traffic_cars":
            old["CROSSING_CARS"] = config.CROSSING_CARS
            config.CROSSING_CARS = dict(config.CROSSING_CARS, **{layout: val})
        else:
            old[k] = getattr(config, k)
            setattr(config, k, val)
    return old


def scene_hash(sim):
    """Fingerprint of the scene at the start of a run (people, their plans, traffic)."""
    parts = []
    for p in sim.peds:
        parts.append(f"p{p.ped_id}:{p.x:.4f},{p.y:.4f},{p.radius:.3f}")
        if hasattr(p, "timing"):
            parts.append(f"c{p.ped_id}:{p.timing:.4f},{p.base_speed:.4f},{p.behaviour}")
    if sim.traffic:
        for v in sim.traffic.vehicles:
            parts.append(f"v{v.id}:{v.x:.4f},{v.y:.4f},{v.heading:.4f},{int(v.active)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


_SIM = None


def run_one(job):
    brain, seed, split, layout, scenario = job
    config.STOP_ON_COLLISION = False               # keep driving after a contact: every person is tested
    from metrics import EpisodeMetrics, surface_gap
    from planner import Driver
    from simulation import Simulation
    global _SIM
    if brain == "lstm":
        import torch
        torch.set_num_threads(1)                   # many runs side by side: one CPU thread each
    var = seed_variation(seed)
    old = apply_variation(var, layout)
    try:
        if _SIM is None or _SIM.layout != layout:
            _SIM = Simulation(gui=False, layout=layout, scenario=scenario, robot=True)
        sim = _SIM
        sim.scenario = scenario
        state = sim.reset(seed=seed)
        run_id = f"{brain}_s{seed:04d}"
        h = scene_hash(sim)
        # the scene (same for every brain)
        scen = dict(seed=seed, split=split, layout=layout, scenario=scenario, scene_hash=h, **var)
        info_rows = []
        for p in sim.peds:
            plan = getattr(p, "plan", None) or {}
            info_rows.append([run_id, p.ped_id, "scripted" if plan else "background", plan.get("name", ""),
                              p.kind, p.radius, int(getattr(p, "aware", True)), int(getattr(p, "phone", False)),
                              int(getattr(p, "jaywalker", False)), round(p.x, 4), round(p.y, 4)])
            if plan:
                n = plan["name"]
                scen[f"{n}_side"] = plan["side"]
                scen[f"{n}_behaviour"] = plan["behaviour"]
                scen[f"{n}_contact_m"] = round(float(plan["contact"]), 3)
                scen[f"{n}_timing_s"] = round(float(p.timing), 3)
                scen[f"{n}_speed_ms"] = round(float(p.base_speed), 3)
        driver = Driver(sim, brain)
        driver.reset()
        rec = EpisodeMetrics(sim, driver)
        robot_rows, ped_rows, veh_rows = [], [], []
        k = 0
        while True:
            cmd = driver.act(state)
            status = driver.planner.status
            state, info = sim.step(*cmd)
            rec.step()
            k += 1
            r = sim.robot
            foot = r.footprint()
            robot_rows.append([run_id, k, round(sim.t, 2), round(r.x, 4), round(r.y, 4), round(r.heading, 5),
                               round(r.vx, 4), round(r.vy, 4), round(math.hypot(r.vx, r.vy), 4),
                               round(cmd[0], 4), round(cmd[1], 4), status, int(sim.done)])
            for p in sim.peds:
                ped_rows.append([run_id, k, p.ped_id, round(p.x, 4), round(p.y, 4), round(p.vx, 4), round(p.vy, 4),
                                 round(surface_gap(foot, p.x, p.y, p.radius), 5)])
            if sim.traffic:
                for v in sim.traffic.vehicles:
                    if v.active:
                        veh_rows.append([run_id, k, v.id, v.kind, round(v.x, 3), round(v.y, 3),
                                         round(v.heading, 4), round(v.v, 3)])
            if info["done"]:
                break
        row, people = rec.finish(info, brain)
        row = dict(run_id=run_id, split=split, scene_hash=h, steps=k, **row)
        for p in people:
            p["run_id"] = run_id
        return dict(scen=scen, row=row, people=people, info=info_rows, robot=robot_rows, peds=ped_rows,
                    vehicles=veh_rows)
    finally:
        for key, val in old.items():
            setattr(config, key, val)


# ------------------------------------------------------------------ saving
ROBOT_COLS = ["run_id", "step", "t", "x", "y", "heading", "vx", "vy", "speed", "cmd_vx", "cmd_vy", "status", "done"]
PED_COLS = ["run_id", "step", "ped_id", "x", "y", "vx", "vy", "gap_to_car"]
VEH_COLS = ["run_id", "step", "vehicle_id", "kind", "x", "y", "heading", "speed"]
INFO_COLS = ["run_id", "ped_id", "role", "name", "kind", "radius", "aware", "phone", "jaywalker", "start_x", "start_y"]
UNITS = {
    "t": "s since the run started", "x": "m (world)", "y": "m (world)", "heading": "rad", "vx": "m/s",
    "vy": "m/s", "speed": "m/s", "cmd_vx": "m/s, velocity the planner asked for", "cmd_vy": "m/s",
    "gap_to_car": "m, surface-to-surface gap car <-> person (metrics.surface_gap); < 0 = contact",
    "done": "1 on the last step (the run ended here)", "step": "0.1 s control steps, 1 = after the first step",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".py"))
    hashes = {f: sha256(os.path.join(folder, f)) for f in files}
    combined = hashlib.sha256("".join(f + hashes[f] for f in files).encode()).hexdigest()
    return combined[:12], hashes


def config_snapshot():
    out = {}
    for k in sorted(dir(config)):
        if not k.isupper():
            continue
        v = getattr(config, k)
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def write_csv(path, cols, rows, gz=False):
    opener = (lambda p: gzip.open(p, "wt", newline="")) if gz else (lambda p: open(p, "w", newline=""))
    with opener(path) as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    try:
        from importlib.metadata import version
        out["pybullet"] = version("pybullet")
    except Exception:
        out["pybullet"] = "unknown"
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", nargs="+", default=["reactive", "predictive"])
    ap.add_argument("--split", choices=["all", "dev", "eval"], default="all")
    ap.add_argument("--layout", default="town")
    ap.add_argument("--scenario", default="crossing")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--eval-start", type=int, default=1000, help="first evaluation seed (a NEW folder only)")
    ap.add_argument("--eval-count", type=int, default=10, help="how many evaluation seeds (a NEW folder only)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    if args.layout in ("street", "town"):
        config.use_street_settings()     # the car's limits, as the simulator uses them (for the manifest)

    out = os.path.join("run_data", args.tag)
    os.makedirs(out, exist_ok=True)
    # 1) the split, fixed BEFORE anything runs (never overwritten)
    split_path = os.path.join(out, "split.csv")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = {int(r["seed"]): r["split"] for r in csv.DictReader(f)}
        print(f"Using the existing {split_path} ({len(split)} seeds)")
    else:
        split = {s: "dev" for s in DEV_SEEDS}
        split.update({s: "eval" for s in range(args.eval_start, args.eval_start + args.eval_count)})
        with open(split_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "split", "decided_at"])
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            for s in sorted(split):
                w.writerow([s, split[s], stamp])
        print(f"Wrote {split_path}: {len(DEV_SEEDS)} dev seeds, {args.eval_count} eval seeds "
              f"({args.eval_start}-{args.eval_start + args.eval_count - 1})")
    seeds = sorted(s for s, sp in split.items() if args.split == "all" or sp == args.split)
    jobs = [(b, s, split[s], args.layout, args.scenario) for s in seeds for b in args.brains]
    print(f"{len(jobs)} runs ({len(seeds)} seeds x {len(args.brains)} brains) on {args.workers} workers ...")

    t0 = time.time()
    results = []
    if args.workers <= 1:
        it = map(run_one, jobs)
    else:
        pool = get_context("spawn").Pool(args.workers)
        it = pool.imap_unordered(run_one, jobs)
    for n, res in enumerate(it, 1):
        r = res["row"]
        results.append(res)
        print(f"  [{n}/{len(jobs)}] {r['run_id']:18s} {r['split']:4s} {r['result']:20s} steps {r['steps']:5d}  "
              f"min gap {r['min_gap']:5.2f} m  hit {r['ped_collisions']}  near misses {r['near_misses']}", flush=True)
    if args.workers > 1:
        pool.close()
    results.sort(key=lambda res: (res["row"]["seed"], args.brains.index(res["row"]["brain"])))

    # 2) save
    scen = {}
    for res in results:
        scen.setdefault(res["scen"]["seed"], res["scen"])
    scen_rows = [scen[s] for s in sorted(scen)]
    scen_cols = list(dict.fromkeys(k for r in scen_rows for k in r))
    counts = {}
    with open(os.path.join(out, "scenarios.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scen_cols)
        w.writeheader()
        w.writerows(scen_rows)
    counts["scenarios.csv"] = len(scen_rows)
    run_rows = [res["row"] for res in results]
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)
    counts["runs.csv"] = len(run_rows)
    ppl = [p for res in results for p in res["people"]]
    with open(os.path.join(out, "crossers.csv"), "w", newline="") as f:
        cols = ["run_id"] + [k for k in ppl[0] if k != "run_id"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ppl)
    counts["crossers.csv"] = len(ppl)
    counts["people_info.csv"] = write_csv(os.path.join(out, "people_info.csv"), INFO_COLS,
                                          [r for res in results for r in res["info"]])
    counts["robot_steps.csv.gz"] = write_csv(os.path.join(out, "robot_steps.csv.gz"), ROBOT_COLS,
                                             [r for res in results for r in res["robot"]], gz=True)
    counts["people_steps.csv.gz"] = write_csv(os.path.join(out, "people_steps.csv.gz"), PED_COLS,
                                              [r for res in results for r in res["peds"]], gz=True)
    counts["vehicle_steps.csv.gz"] = write_csv(os.path.join(out, "vehicle_steps.csv.gz"), VEH_COLS,
                                               [r for res in results for r in res["vehicles"]], gz=True)

    # 3) the manifest: what made these files
    here = os.path.dirname(os.path.abspath(__file__))
    cv, code_hashes = code_version(here)
    files = {}
    for name in ["split.csv", "scenarios.csv", "runs.csv", "crossers.csv", "people_info.csv",
                 "robot_steps.csv.gz", "people_steps.csv.gz", "vehicle_steps.csv.gz"]:
        path = os.path.join(out, name)
        files[name] = {"sha256": sha256(path), "bytes": os.path.getsize(path),
                       "rows": counts.get(name, len(split))}
    manifest = {
        "dataset": f"run_data/{args.tag}",
        "description": "Core navigation comparison: the same seeds driven by each brain; simulator positions "
                       "of people (perfect sensing, no camera / YOLO).",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "seconds_to_make": round(time.time() - t0, 1),
        "code_version": cv,
        "code_files_sha256": code_hashes,
        "versions": versions(),
        "brains": args.brains, "layout": args.layout, "scenario": args.scenario,
        "seeds": {"dev": [s for s in seeds if split[s] == "dev"], "eval": [s for s in seeds if split[s] == "eval"]},
        "split_rule": "dev seeds: tuning may use these; eval seeds: reserved for the final comparison, chosen "
                      "from a range never run before (see split.csv, written before any run). Whole runs (seeds) "
                      "are assigned, never single steps.",
        "models": ({config.LSTM_MODEL: sha256(config.LSTM_MODEL)} if "lstm" in args.brains else {}),
        "variation_per_seed": {k: list(v) if isinstance(v, tuple) else v for k, v in VARIATION.items()},
        "fixed_settings": {"STOP_ON_COLLISION": False, "CONTROL_DT_s": config.CONTROL_DT,
                           "MAX_TIME_s": config.MAX_TIME},
        "metric_definitions": {
            "gap": "surface-to-surface distance between the car (3 circles) and a person (circle), "
                   "metrics.surface_gap; < 0 = contact",
            "near_miss": f"0 <= gap < {config.NEAR_MISS_DIST} m, counted once per person until the gap opens "
                         f"past {config.NEAR_MISS_DIST + config.NEAR_MISS_REARM} m (metrics.count_contacts_and_near_misses)",
            "min_gap (runs.csv)": "smallest gap to any person within 6 m, checked every 0.01 s (physics sub-steps)",
        },
        "units": UNITS,
        "files": files,
        "config": config_snapshot(),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nSaved {out}/ ({sum(v['bytes'] for v in files.values()) / 1e6:.1f} MB, code version {cv}) "
          f"in {time.time() - t0:.0f} s")
    print(f"Next: python check_run_dataset.py {out}")


if __name__ == "__main__":
    main()
