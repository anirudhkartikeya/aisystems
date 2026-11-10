"""make_yolo_data.py - generate labelled camera images to train YOLO.

    python make_yolo_data.py                     # 120 episodes -> dataset/ (about 2,800 images)
    python make_yolo_data.py --episodes 3        # a quick test
    python make_yolo_data.py --start 0 --episodes 20   # seeds 0..19 (run several terminals
                                                       # with different --start to go faster)

No window. In most episodes the robot is switched off and the camera rides
on the traffic cars, hopping to another car every few seconds, so the
pictures cover the whole town, every kind of street, junction and crossing.
Every 4th episode the robot car itself drives its route (crossing scenario,
driven by the predictive brain + planner) with the camera on its front. Each episode has its own lighting
(noon, morning, evening, overcast, dusk), crowd and traffic, and every
picture gets a slightly different camera height/angle/lens plus a little
noise or blur.

Hard cases (night / shadow / dark clothes), into their own folder with new seeds:
    python make_yolo_data.py --out dataset_night --start 1000 --episodes 45 \
        --lightings night harsh_shadow dusk --weights 0.5 0.3 0.2 --dark-clothes 0.35
(night = moonlight + the car's headlights + street lamps, see camera.night_scene)

Labels come from the simulator's segmentation image, so they are exact and
free. Output (the standard YOLO layout):
  dataset/images/{train,val,test}/*.jpg
  dataset/labels/{train,val,test}/*.txt     one line per person: 0 cx cy w h
  dataset/data.yaml                          for training (yolo train data=dataset/data.yaml)
  dataset/index_<a>_<b>.csv                  image, split, seed, layout, lighting, people
Whole episodes stay in one split.
"""
import argparse
import csv
import os
import time

import numpy as np

import config
import datasets
from camera import RobotCamera, augment, car_boxes, lamp_heads, person_boxes, save_sample, write_data_yaml
from planner import Driver
from simulation import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--start", type=int, default=0, help="first seed")
    parser.add_argument("--layouts", nargs="+", default=["town", "town", "street"])
    parser.add_argument("--seconds", type=float, default=40.0, help="length of each episode")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between pictures")
    parser.add_argument("--hop", type=float, default=8.0, help="seconds before the camera moves to another car")
    parser.add_argument("--min-move", type=float, default=2.0,
                        help="metres the camera must move between two pictures (no near-duplicate pictures)")
    parser.add_argument("--keep-empty", type=float, default=0.25,
                        help="share of pictures with nobody in them that are kept")
    parser.add_argument("--png", action="store_true", help="save .png instead of .jpg")
    parser.add_argument("--out", default=config.DATASET_DIR)
    parser.add_argument("--lightings", nargs="+", default=list(config.YOLO_LIGHTINGS),
                        help="lighting presets to pick from per episode (config.LIGHTING_PRESETS)")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="how often each of --lightings is picked (default: equally)")
    parser.add_argument("--dark-clothes", type=float, default=0.0,
                        help="share of people dressed in dark colours (night: more of them, see below)")
    args = parser.parse_args()
    for name in args.lightings:
        if name not in config.LIGHTING_PRESETS:
            raise SystemExit(f"unknown lighting {name}; choose from {list(config.LIGHTING_PRESETS)}")
    if args.weights is not None and len(args.weights) != len(args.lightings):
        raise SystemExit("--weights needs one number per --lightings entry")
    probs = None if args.weights is None else np.asarray(args.weights, float) / sum(args.weights)
    default_set = args.lightings == list(config.YOLO_LIGHTINGS) and probs is None

    config.STOP_ON_COLLISION = False
    config.MAX_TIME = args.seconds + 5.0
    os.makedirs(args.out, exist_ok=True)
    write_data_yaml(args.out, with_cars=config.YOLO_LABEL_CARS)
    a, b = args.start, args.start + args.episodes - 1
    index_path = os.path.join(args.out, f"index_{a}_{b}.csv")
    index = open(index_path, "w", newline="")
    iw = csv.writer(index)
    iw.writerow(["image", "split", "seed", "layout", "lighting", "people", "camera_on", "dark_clothes", "moon",
                 "unseen_in_dark"])

    sim, sim_key = None, None
    cam = None
    saved = {"train": 0, "val": 0, "test": 0}
    t0 = time.time()
    try:
        for k in range(args.episodes):
            seed = args.start + k
            layout = args.layouts[seed % len(args.layouts)]
            drive = seed % 4 == 3                          # robot car drives with the camera
            scenario = "crossing" if drive else "random"
            rng = np.random.default_rng([seed, 11])
            if default_set:                                # exactly as the first dataset picked it
                lighting = args.lightings[int(rng.integers(len(args.lightings)))]
            else:
                lighting = str(rng.choice(args.lightings, p=probs))
            night = bool(config.LIGHTING_PRESETS[lighting].get("night"))
            moon = float(rng.uniform(*config.NIGHT["moon"])) if night else None
            # dark clothes: more of them at night (people walking home in dark coats)
            config.DARK_CLOTHES_SHARE = min(1.0, args.dark_clothes * (1.3 if night else 1.0))
            old = datasets.vary_episode(np.random.default_rng([seed, 7]), layout) if config.DATA_VARIETY else {}
            key = (layout, scenario, drive)
            if sim is None or sim_key != key:              # one PyBullet world at a time
                if sim is not None:
                    sim.close()
                sim, sim_key = Simulation(gui=False, layout=layout, scenario=scenario, robot=drive), key
                cam = RobotCamera(gui=False)
                driver = Driver(sim, "predictive") if drive else None
                lamps = lamp_heads(sim.world.roadmap)
            split = datasets.split_for_seed(seed)
            state = sim.reset(seed=seed)
            if drive:
                driver.reset()
            carrier, hop_left, last_shot = None, 0.0, None
            steps = int(args.seconds / config.CONTROL_DT)
            every = max(1, round(args.every / config.CONTROL_DT))
            for step in range(steps):
                if drive:
                    vx, vy = driver.act(state)
                else:
                    vx = vy = 0.0
                state, info = sim.step(vx, vy)
                if info["done"] and drive:
                    break
                hop_left -= config.CONTROL_DT
                if step % every:
                    continue
                if drive:
                    pose = sim.robot
                else:
                    cars = [v for v in sim.traffic.vehicles if v.active] if sim.traffic else []
                    if not cars:
                        continue
                    stuck = carrier is not None and carrier.v < 0.3 and carrier.stopped_for > 3.0
                    if carrier is None or not carrier.active or hop_left <= 0 or stuck:
                        moving = [v for v in cars if v.v > 1.0] or cars       # prefer a car that is driving
                        carrier = moving[int(rng.integers(len(moving)))]
                        hop_left, last_shot = args.hop, None
                    pose = carrier
                if last_shot is not None and \
                        ((pose.x - last_shot[0]) ** 2 + (pose.y - last_shot[1]) ** 2) ** 0.5 < args.min_move:
                    continue                               # camera hardly moved: skip a near-duplicate picture
                sim.sync_visuals()                         # headless: move the bodies before the picture
                rgb, _, seg, _ = cam.capture(pose, lighting, config.CAMERA_JITTER, rng, lamps=lamps, moon=moon)
                hard_light = lighting not in config.YOLO_LIGHTINGS        # night / harsh shadow
                unseen = []
                boxes = person_boxes(seg, sim.peds, rgb=rgb if hard_light else None, dark=unseen)
                if config.YOLO_LABEL_CARS:
                    others = [v.figure.body_ids for v in sim.traffic.vehicles if v.active and v is not pose] \
                        if sim.traffic else []
                    boxes += car_boxes(seg, others + sim.world.parked_car_ids)
                people = sum(1 for bx in boxes if bx[0] == 0)
                if people == 0 and rng.random() > args.keep_empty:
                    continue
                aug = dict(config.IMAGE_AUGMENT, brightness=config.NIGHT_AUGMENT_BRIGHTNESS) if night else None
                rgb = augment(rgb, rng, aug)
                name = f"{layout}_s{seed:04d}_t{step:04d}"
                save_sample(rgb, boxes, name, args.out, split, "png" if args.png else "jpg")
                last_shot = (pose.x, pose.y)
                iw.writerow([name, split, seed, layout, lighting, people, "robot" if drive else "traffic",
                             round(config.DARK_CLOTHES_SHARE, 2), "" if moon is None else round(moon, 3),
                             len(unseen)])
                saved[split] += 1
            datasets.restore(old)
            config.DARK_CLOTHES_SHARE = 0.0
            index.flush()
            rate = (time.time() - t0) / (k + 1)
            print(f"{k + 1}/{args.episodes} episodes  |  images so far: {saved}  ({rate:.0f} s per episode)")
    finally:
        index.close()
        if sim is not None:
            sim.close()
    print(f"Saved {sum(saved.values())} images to '{args.out}' ({saved}).  Train with: data={args.out}/data.yaml")


if __name__ == "__main__":
    main()
