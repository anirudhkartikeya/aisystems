"""evaluate.py - the controlled comparison: the same runs with each brain.

    python evaluate.py                          # reactive vs predictive, 50 seeds each
    python evaluate.py --seeds 10               # quicker check
    python evaluate.py --brains none reactive predictive   # also the old placeholder driver
    python evaluate.py --workers 8              # use 8 CPU cores (default: all but 2)

Every brain drives exactly the same seeds (same people, same traffic, same P5),
with no window, and keeps driving after a collision so every person is tested.
Writes into results/:
    episodes.csv   one row per run (brain, seed, every metric)
    people.csv     one row per scripted person per run (expected vs actual action)
    summary.csv    the averages, one column per brain (also printed as a table)
"""
import argparse
import csv
import math
import os
import time
from multiprocessing import get_context

import config


def run_one(job):
    brain, seed, layout, scenario = job
    config.STOP_ON_COLLISION = False
    from controllers import placeholder_controller
    from metrics import EpisodeMetrics
    from planner import Driver
    from simulation import Simulation
    global _SIM
    if "_SIM" not in globals() or _SIM.layout != layout:
        _SIM = Simulation(gui=False, layout=layout, scenario=scenario, robot=True)
    sim = _SIM
    sim.scenario = scenario
    state = sim.reset(seed=seed)
    driver = None
    if brain == "none":
        waypoints = sim.world.route(sim.start, sim.goal)
    else:
        driver = Driver(sim, brain)
        driver.reset()
    rec = EpisodeMetrics(sim, driver)
    while True:
        cmd = placeholder_controller(sim, state, waypoints) if driver is None else driver.act(state)
        state, info = sim.step(*cmd)
        rec.step()
        if info["done"]:
            break
    return rec.finish(info, brain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", nargs="+", default=["reactive", "predictive"],
                    choices=["none", "reactive", "predictive", "lstm"])
    ap.add_argument("--seeds", type=int, default=50, help="how many runs per brain")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--layout", default=config.LAYOUT)
    ap.add_argument("--scenario", default="crossing")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    jobs = [(b, s, args.layout, args.scenario) for s in range(args.start, args.start + args.seeds)
            for b in args.brains]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(jobs)} runs ({args.seeds} seeds x {len(args.brains)} brains) on {args.workers} workers ...")
    t0 = time.time()
    rows, people = [], []
    if args.workers <= 1:
        results = map(run_one, jobs)
    else:
        pool = get_context("spawn").Pool(args.workers)
        results = pool.imap_unordered(run_one, jobs)
    for k, (row, ppl) in enumerate(results, 1):
        rows.append(row)
        people += ppl
        print(f"  [{k}/{len(jobs)}] {row['brain']:10s} seed {row['seed']:3d}: {row['result']:20s} "
              f"min gap {row['min_gap']:5.2f} m, hit people {row['ped_collisions']}, "
              f"correct {row['correct_actions']}/{row['graded_people']}", flush=True)
    if args.workers > 1:
        pool.close()
    rows.sort(key=lambda r: (args.brains.index(r["brain"]), r["seed"]))
    people.sort(key=lambda r: (args.brains.index(r["brain"]), r["seed"], r["name"]))

    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "people.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(people[0]))
        w.writeheader()
        w.writerows(people)

    from metrics import summarise
    table = {b: summarise([r for r in rows if r["brain"] == b]) for b in args.brains}
    labels = list(next(iter(table.values())))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + args.brains)
        for lab in labels:
            w.writerow([lab] + [round(table[b][lab], 3) for b in args.brains])

    # the table
    width = max(len(lab) for lab in labels)
    print()
    print(f"{'':{width}}  " + "  ".join(f"{b:>11s}" for b in args.brains))
    for lab in labels:
        cells = "  ".join(f"{table[b][lab]:11.2f}" if not math.isnan(table[b][lab]) else f"{'-':>11s}"
                          for b in args.brains)
        print(f"{lab:{width}}  {cells}")
    # what each brain did for each person
    print()
    for name in sorted({p["name"] for p in people}):
        exp = next(p["expected"] for p in people if p["name"] == name)
        parts = []
        for b in args.brains:
            mine = [p for p in people if p["name"] == name and p["brain"] == b]
            hit = sum(1 for p in mine if p["min_gap"] <= 0)
            acts = {}
            for p in mine:
                acts[p["action"]] = acts.get(p["action"], 0) + 1
            top = ", ".join(f"{a} {n}" for a, n in sorted(acts.items(), key=lambda a: -a[1]))
            parts.append(f"{b}: {top} (hit {hit})")
        print(f"{name} (expected {exp}): " + " | ".join(parts))
    print(f"\nSaved {args.out}/episodes.csv, people.csv, summary.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
