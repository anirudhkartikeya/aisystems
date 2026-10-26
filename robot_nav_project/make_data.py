"""make_data.py - generate pedestrian trajectories to train the LSTM.

    python make_data.py                          # 500 episodes (town + street) -> trajectories/
    python make_data.py --episodes 20            # a quick test
    python make_data.py --start 0 --episodes 100 # seeds 0..99 (run several terminals with
                                                 # different --start to use more CPU cores)
    python make_data.py --layouts town           # only the town

No window, no real-time waiting. People go about their business while traffic
drives around them. In about 3 of every 10 episodes (--drive-share) the robot
car also drives its route with the predictive brain, so the data includes how
people move around the car the LSTM will be used on. Its rows are in the
vehicles file with kind "robot". Each episode has its own crowd size, traffic
and behaviour mix (see datasets.py).

Output files (one set per run, named after the seed range):
  peds_<a>_<b>.csv      episode, split, layout, t, ped_id, group_id, kind, x, y, vx, vy, mode, on_road
  vehicles_<a>_<b>.csv  episode, split, t, veh_id, kind, x, y, vx, vy, heading, length, width
  episodes_<a>_<b>.csv  one line per episode: seed, split, layout, people, cars, seconds, robot_drives

Positions are logged every --dt seconds (0.4 s = what the LSTM uses). The
first --warmup seconds are not logged (everybody starts standing still).
"split" says train / val / test: whole episodes stay in one split.
"""
import argparse
import csv
import os
import time

import numpy as np

import config
import datasets
from planner import Driver
from simulation import Simulation


def run_episode(sim, seed, seconds, warmup, dt, ped_w, veh_w, driver=None):
    state = sim.reset(seed=seed)
    if driver is not None:
        driver.reset()
    split = datasets.split_for_seed(seed)
    every = max(1, round(dt / config.CONTROL_DT))
    total = round((warmup + seconds) / config.CONTROL_DT)
    first = round(warmup / config.CONTROL_DT)
    world = sim.world
    for step in range(total + 1):
        if step >= first and (step - first) % every == 0:
            t = round((step - first) * config.CONTROL_DT, 2)
            for ped in state["pedestrians"]:
                ped_w.writerow([seed, split, sim.layout, t, ped["id"], ped["group"], ped["kind"],
                                round(ped["x"], 3), round(ped["y"], 3), round(ped["vx"], 3), round(ped["vy"], 3),
                                ped["mode"], int(world.on_road(ped["x"], ped["y"]))])
            cars = state["vehicles"] + ([sim.robot_snapshot()] if driver is not None else [])
            for v in cars:
                veh_w.writerow([seed, split, t, v["id"], "robot" if v.get("is_robot") else v["kind"],
                                round(v["x"], 3), round(v["y"], 3),
                                round(v["vx"], 3), round(v["vy"], 3), round(v["heading"], 4),
                                v["length"], v["width"]])
        if step == total:
            break
        state, info = sim.step(*(driver.act(state) if driver is not None else (0.0, 0.0)))
        if info["done"]:                     # the robot reached its goal: stop logging this episode
            break
    logged = max(0.0, round((step - first) * config.CONTROL_DT, 1))
    return split, len(state["pedestrians"]), len(state["vehicles"]), logged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--start", type=int, default=0, help="first seed")
    parser.add_argument("--layouts", nargs="+", default=["town", "street"])
    parser.add_argument("--seconds", type=float, default=60.0, help="logged length of each episode")
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.4, help="seconds between logged positions")
    parser.add_argument("--out", default="trajectories")
    parser.add_argument("--drive-share", type=float, default=0.3,
                        help="share of episodes where the robot car drives (predictive brain)")
    args = parser.parse_args()

    config.STOP_ON_COLLISION = False
    config.MAX_TIME = args.warmup + args.seconds + 1.0
    os.makedirs(args.out, exist_ok=True)
    a, b = args.start, args.start + args.episodes - 1
    paths = {k: os.path.join(args.out, f"{k}_{a}_{b}.csv") for k in ("peds", "vehicles", "episodes")}
    files = {k: open(p, "w", newline="") for k, p in paths.items()}
    ped_w, veh_w, ep_w = (csv.writer(files[k]) for k in ("peds", "vehicles", "episodes"))
    ped_w.writerow(["episode", "split", "layout", "t", "ped_id", "group_id", "kind",
                    "x", "y", "vx", "vy", "mode", "on_road"])
    veh_w.writerow(["episode", "split", "t", "veh_id", "kind", "x", "y", "vx", "vy", "heading", "length", "width"])
    ep_w.writerow(["seed", "split", "layout", "people", "cars", "seconds", "robot_drives"])

    sim, sim_layout = None, None
    t0 = time.time()
    try:
        for k in range(args.episodes):
            seed = args.start + k
            layout = args.layouts[seed % len(args.layouts)]
            old = datasets.vary_episode(np.random.default_rng([seed, 7]), layout) if config.DATA_VARIETY else {}
            drives = (seed * 7919 % 100) < args.drive_share * 100      # the same choice for a seed every time
            if sim is None or sim_layout != (layout, drives):       # one PyBullet world at a time
                if sim is not None:
                    sim.close()
                sim = Simulation(gui=False, layout=layout, scenario="random", robot=drives)
                sim_layout = (layout, drives)
                driver = Driver(sim, "predictive") if drives else None
            split, n_p, n_c, logged = run_episode(sim, seed, args.seconds, args.warmup, args.dt, ped_w, veh_w,
                                                  driver)
            ep_w.writerow([seed, split, layout, n_p, n_c, logged, int(drives)])
            datasets.restore(old)
            if (k + 1) % 10 == 0 or k + 1 == args.episodes:
                for f in files.values():
                    f.flush()
                rate = (time.time() - t0) / (k + 1)
                print(f"{k + 1}/{args.episodes} episodes done  ({rate:.1f} s per episode)")
    finally:
        for f in files.values():
            f.close()
        if sim is not None:
            sim.close()
    print("Saved:\n  " + "\n  ".join(paths.values()))


if __name__ == "__main__":
    main()
