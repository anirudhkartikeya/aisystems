"""diagnose_contacts.py - for every contact with a person: who, when, and what the car was doing.

    python diagnose_contacts.py --brain lstm --condition "noise 0.1 m" --seeds 1 3 4 9
    python diagnose_contacts.py --brain lstm --condition "noise 0.1 m" --split dev --set LSTM_MODEL=results/lstm_v3/lstm_v3.pt

Same scenes as stress_test.py (the seed's variation + the camera condition). For each contact it
records the person (scripted crosser P1-P5 or a background person, what they were doing, their
speed), the car's speed now and 1 s earlier, and the planner's status. The key question: was the
car DRIVING into the person, or was it (nearly) stopped and the person walked into it?
Writes run_data/diagnose/contacts.csv and prints a summary.
"""
import argparse
import ast
import csv
import math
import os
from collections import deque

import config


def run_one(brain, seed, cond, settings, layout="town", scenario="crossing"):
    config.STOP_ON_COLLISION = False
    from make_run_dataset import apply_variation, seed_variation
    from metrics import surface_gap
    from planner import Driver
    from simulation import Simulation
    from stress_test import CONDITIONS
    old = apply_variation(seed_variation(seed), layout)
    for k, v in list(settings.items()) + list(CONDITIONS[cond].items()):
        old.setdefault(k, getattr(config, k))
        setattr(config, k, v)
    try:
        sim = Simulation(gui=False, layout=layout, scenario=scenario, robot=True)
        state = sim.reset(seed=seed)
        driver = Driver(sim, brain)
        driver.reset()
        names = {log["ped_id"]: log["name"] for log in sim.crosser_log}
        speeds = deque(maxlen=int(1.0 / config.CONTROL_DT) + 1)
        touching, rows = set(), []
        info = {"done": False}
        while not info["done"]:
            state, info = sim.step(*driver.act(state))
            r = sim.robot
            v = math.hypot(r.vx, r.vy)
            speeds.append(v)
            foot = r.footprint()
            for p in sim.peds:
                gap = surface_gap(foot, p.x, p.y, p.radius)
                if gap < 0 and p.ped_id not in touching:
                    touching.add(p.ped_id)
                    seen = p.ped_id in driver.perception.tracks
                    rows.append({"brain": brain, "seed": seed, "condition": cond, "t": round(sim.t, 1),
                                 "person": names.get(p.ped_id, "background"), "ped_id": p.ped_id,
                                 "person_mode": getattr(p, "mode", "?"),
                                 "person_speed": round(math.hypot(p.vx, p.vy), 2),
                                 "car_speed": round(v, 2), "car_speed_1s_before": round(speeds[0], 2),
                                 "car_state": "stopped" if v < 0.2 else ("slow" if v < 0.8 else "driving"),
                                 "seen_by_car": seen, "planner": driver.planner.status})
                elif gap > config.NEAR_MISS_REARM and p.ped_id in touching:
                    touching.discard(p.ped_id)
        result = info.get("result", "?")
        for row in rows:
            row["result"] = result
        return rows, result
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="lstm")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: all seeds of --split")
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--dataset", default="run_data/v3")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default="run_data/diagnose")
    args = ap.parse_args()
    config.use_street_settings()
    settings = {}
    for item in args.set:
        k, v = item.split("=", 1)
        try:
            settings[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            settings[k] = v
    seeds = args.seeds
    if not seeds:
        with open(os.path.join(args.dataset, "split.csv")) as f:
            seeds = sorted(int(r["seed"]) for r in csv.DictReader(f) if r["split"] == args.split)
    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    for s in seeds:
        rows, result = run_one(args.brain, s, args.condition, settings)
        all_rows += rows
        print(f"seed {s:4d}: {result:8s} {len(rows)} contact(s)" +
              "".join(f"\n      t={r['t']:6.1f} s  {r['person']:10s} ({r['person_mode']}, {r['person_speed']} m/s) "
                      f"| car {r['car_state']} {r['car_speed']} m/s (1 s before {r['car_speed_1s_before']}) "
                      f"| seen {r['seen_by_car']} | {r['planner']}" for r in rows), flush=True)
    path = os.path.join(args.out, "contacts.csv")
    if all_rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    n = len(all_rows)
    print(f"\n{n} contacts in {len(seeds)} runs ({args.brain}, {args.condition}, {settings or 'default settings'})")
    if n:
        for key in ("car_state", "person", "seen_by_car"):
            counts = {}
            for r in all_rows:
                counts[r[key]] = counts.get(r[key], 0) + 1
            print(f"  {key:12s}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda a: -a[1])))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
