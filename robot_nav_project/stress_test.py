"""stress_test.py - how fast does each brain's safety fall when its "eyes" get worse?

    python stress_test.py                                    # 3 brains x 8 conditions x the 20 eval seeds of run_data/v3
    python stress_test.py --seeds 10 --brains predictive lstm
    python stress_test.py --split dev                        # on the dev seeds instead
    python stress_test.py --split dev --set TRACK_FILTER=none --out run_data/stress_dev_raw
                                                             # any config.py setting, for every run

All runs so far used PERFECT positions of people. A real camera + YOLO is worse, so the camera can be
given faults (perception.py, all off by default):
    noise    every seen position gets a random error (normal, sigma in metres, per axis)
    miss     a share of the camera sightings is lost, like YOLO missing someone in a frame
             (close-range sensors within 3 m still work)
    delay    the brain gets positions that are this many seconds old (camera + detector latency),
             with their real time stamps
Conditions: clean | noise 0.1 / 0.3 m | miss 20 / 50 % | delay 0.2 / 0.5 s | a YOLO-like mix
(noise 0.1 m + miss 20 % + delay 0.2 s). Same seeds, same scenes (the dataset's variation) for all.
The clean runs must equal the saved dataset - checked and reported.
Writes run_data/stress/: stress_runs.csv, stress_summary.csv, stress_report.txt, stress.png, stress.json
"""
import argparse
import ast
import csv
import datetime
import json
import math
import os
import time
from multiprocessing import get_context

import numpy as np

import config

CONDITIONS = {
    "clean": {},
    "noise 0.1 m": {"PERCEPTION_NOISE": 0.1},
    "noise 0.3 m": {"PERCEPTION_NOISE": 0.3},
    "miss 20%": {"PERCEPTION_MISS_RATE": 0.2},
    "miss 50%": {"PERCEPTION_MISS_RATE": 0.5},
    "delay 0.2 s": {"PERCEPTION_DELAY": 0.2},
    "delay 0.5 s": {"PERCEPTION_DELAY": 0.5},
    "YOLO-like mix": {"PERCEPTION_NOISE": 0.1, "PERCEPTION_MISS_RATE": 0.2, "PERCEPTION_DELAY": 0.2},
}
_SIM = None


def run_stress(job):
    brain, seed, cond, layout, scenario, settings = job
    config.STOP_ON_COLLISION = False
    from make_run_dataset import apply_variation, seed_variation
    from metrics import EpisodeMetrics
    from planner import Driver
    from simulation import Simulation
    global _SIM
    if brain == "lstm":
        import torch
        torch.set_num_threads(1)
    old = apply_variation(seed_variation(seed), layout)
    for k, v in settings.items():                       # --set values win over the seed's variation
        old.setdefault(k, getattr(config, k))
        setattr(config, k, v)
    faults = CONDITIONS[cond]
    for k in faults:
        old.setdefault(k, getattr(config, k))
    for k, v in faults.items():
        setattr(config, k, v)
    try:
        if _SIM is None or _SIM.layout != layout:
            _SIM = Simulation(gui=False, layout=layout, scenario=scenario, robot=True)
        sim = _SIM
        state = sim.reset(seed=seed)
        driver = Driver(sim, brain)
        driver.reset()
        rec = EpisodeMetrics(sim, driver)
        while True:
            state, info = sim.step(*driver.act(state))
            rec.step()
            if info["done"]:
                break
        row, _ = rec.finish(info, brain)
        return dict(condition=cond, **row)
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", nargs="+", default=["reactive", "predictive", "lstm"])
    ap.add_argument("--dataset", default="run_data/v3", help="seeds (and the clean-run check) come from here")
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    ap.add_argument("--seeds", type=int, default=0, help="use only the first N seeds (0 = all)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--layout", default="town")
    ap.add_argument("--scenario", default="crossing")
    ap.add_argument("--out", default="run_data/stress")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="change config.py settings for every run, e.g. TRACK_FILTER=none KALMAN_ACCEL=1.0")
    args = ap.parse_args()
    settings = {}
    for item in args.set:
        k, v = item.split("=", 1)
        if not hasattr(config, k):
            raise SystemExit(f"--set: config.py has no setting {k}")
        try:
            settings[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            settings[k] = v                                   # plain text, e.g. none / kalman
    if args.layout in ("street", "town"):
        config.use_street_settings()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.dataset, "split.csv")) as f:
        seeds = sorted(int(r["seed"]) for r in csv.DictReader(f) if r["split"] == args.split)
    if args.seeds:
        seeds = seeds[:args.seeds]
    jobs = [(b, s, c, args.layout, args.scenario, settings)
            for c in args.conditions for s in seeds for b in args.brains]
    filt = settings.get("TRACK_FILTER", config.TRACK_FILTER)
    print(f"tracking filter: {filt}" + (f" | settings: {settings}" if settings else ""))
    print(f"{len(jobs)} runs: {len(args.brains)} brains x {len(args.conditions)} conditions x {len(seeds)} "
          f"{args.split} seeds ({seeds[0]}-{seeds[-1]}) on {args.workers} workers ...")
    t0 = time.time()
    rows = []
    pool = get_context("spawn").Pool(args.workers) if args.workers > 1 else None
    it = pool.imap_unordered(run_stress, jobs) if pool else map(run_stress, jobs)
    for n, r in enumerate(it, 1):
        rows.append(r)
        if n % 20 == 0 or n == len(jobs):
            print(f"  {n}/{len(jobs)} runs ({time.time() - t0:.0f} s)", flush=True)
    if pool:
        pool.close()
    rows.sort(key=lambda r: (args.conditions.index(r["condition"]), args.brains.index(r["brain"]), r["seed"]))
    with open(os.path.join(args.out, "stress_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # clean runs vs the saved dataset (must be identical)
    check = "no dataset runs to compare"
    saved = os.path.join(args.dataset, "runs.csv")
    if "clean" in args.conditions and os.path.exists(saved):
        with open(saved) as f:
            ds = {(r["brain"], int(r["seed"])): r for r in csv.DictReader(f)}
        keys = ("result", "ped_collisions", "near_misses", "min_gap")
        clean = [r for r in rows if r["condition"] == "clean" and (r["brain"], r["seed"]) in ds]
        same = sum(all(str(r[k]) == ds[(r["brain"], r["seed"])][k] for k in keys) for r in clean)
        check = f"clean runs identical to {saved}: {same}/{len(clean)}"
        try:
            made_with = json.load(open(os.path.join(args.dataset, "manifest.json")))["config"].get("TRACK_FILTER", "none")
        except Exception:
            made_with = "?"
        if made_with != filt:
            check += f" (not expected to match: the dataset used tracking filter '{made_with}', this test '{filt}')"

    # summary
    rng = np.random.default_rng(0)

    def ci(vals):
        vals = np.asarray(vals, float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return math.nan, math.nan, math.nan
        s = vals[rng.integers(0, len(vals), (3000, len(vals)))].mean(1)
        return vals.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)

    summ = []
    for c in args.conditions:
        for b in args.brains:
            rr = [r for r in rows if r["condition"] == c and r["brain"] == b]
            touch = ci([100.0 * (r["ped_collisions"] > 0) for r in rr])
            summ.append({"condition": c, "brain": b, "runs": len(rr),
                         "touch a person %": round(touch[0], 1), "touch CI": f"[{touch[1]:.0f}, {touch[2]:.0f}]",
                         "contacts/run": round(float(np.mean([r["ped_collisions"] for r in rr])), 2),
                         "near misses/run": round(float(np.mean([r["near_misses"] for r in rr])), 2),
                         "closest gap (m)": round(float(np.mean([r["min_gap"] for r in rr])), 2),
                         "goal %": round(100 * float(np.mean([r["result"] == "goal" for r in rr])), 1),
                         "time (s)": round(float(np.mean([r["time"] for r in rr])), 1),
                         "hard brakes/run": round(float(np.mean([r["hard_brakes"] for r in rr])), 2),
                         "prediction FDE (m)": round(float(np.nanmean([r["fde"] for r in rr])), 3)})
    with open(os.path.join(args.out, "stress_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    cols = list(summ[0])
    widths = [max(len(c), *(len(str(r[c])) for r in summ)) for c in cols]
    lines = [f"Stress test: {len(seeds)} {args.split} seeds x {len(args.brains)} brains x {len(args.conditions)} camera conditions",
             f"tracking filter: {filt}" + (f" | settings: {settings}" if settings else ""), check, "", "  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)) for r in summ]
    open(os.path.join(args.out, "stress_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    here = os.path.dirname(os.path.abspath(__file__))
    from make_run_dataset import code_version, versions
    json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(here)[0],
               "versions": versions(), "seeds": seeds, "split": args.split, "brains": args.brains,
               "conditions": {c: CONDITIONS[c] for c in args.conditions}, "clean_check": check,
               "tracking_filter": filt, "settings": settings,
               "kalman": {"KALMAN_MEAS_STD": settings.get("KALMAN_MEAS_STD", config.KALMAN_MEAS_STD),
                          "KALMAN_ACCEL": settings.get("KALMAN_ACCEL", config.KALMAN_ACCEL)}},
              open(os.path.join(args.out, "stress.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"reactive": "#c0392b", "predictive": "#2471a3", "lstm": "#229954"}
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, key in zip(axes, ["touch a person %", "near misses/run", "closest gap (m)", "goal %"]):
            for b in args.brains:
                ys = [next(r[key] for r in summ if r["condition"] == c and r["brain"] == b) for c in args.conditions]
                ax.plot(range(len(args.conditions)), ys, "o-", label=b, color=colors.get(b, "gray"))
            ax.set_xticks(range(len(args.conditions)), args.conditions, rotation=35, ha="right", fontsize=8)
            ax.set_title(key)
        axes[0].legend()
        fig.suptitle(f"Camera faults vs safety ({len(seeds)} {args.split} seeds per point, tracking filter: {filt})")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "stress.png"), dpi=120); plt.close()
    except Exception as e:
        print("(no chart:", e, ")")
    print(f"\nSaved {args.out}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
