"""make_yolo_data.py - generate labelled camera images to train YOLO.

    python make_yolo_data.py                     # 80 episodes -> dataset/ (about 2,800 images)
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
from camera import RobotCamera, augment, car_boxes, person_boxes, save_sample, write_data_yaml
from planner import Driver
from simulation import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--start", type=int, default=0, help="first seed")
    parser.add_argument("--layouts", nargs="+", default=["town", "town", "street"])
    parser.add_argument("--seconds", type=float, default=40.0, help="length of each episode")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between pictures")
    parser.add_argument("--hop", type=float, default=8.0, help="seconds before the camera moves to another car")
    parser.add_argument("--keep-empty", type=float, default=0.25,
                        help="share of pictures with nobody in them that are kept")
    parser.add_argument("--png", action="store_true", help="save .png instead of .jpg")
    parser.add_argument("--out", default=config.DATASET_DIR)
    args = parser.parse_args()

    config.STOP_ON_COLLISION = False
    config.MAX_TIME = args.seconds + 5.0
    os.makedirs(args.out, exist_ok=True)
    write_data_yaml(args.out, with_cars=config.YOLO_LABEL_CARS)
    a, b = args.start, args.start + args.episodes - 1
    index_path = os.path.join(args.out, f"index_{a}_{b}.csv")
    index = open(index_path, "w", newline="")
    iw = csv.writer(index)
    iw.writerow(["image", "split", "seed", "layout", "lighting", "people", "camera_on"])

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
            lighting = list(config.LIGHTING_PRESETS)[int(rng.integers(len(config.LIGHTING_PRESETS)))]
            old = datasets.vary_episode(np.random.default_rng([seed, 7]), layout) if config.DATA_VARIETY else {}
            key = (layout, scenario, drive)
            if sim is None or sim_key != key:              # one PyBullet world at a time
                if sim is not None:
                    sim.close()
                sim, sim_key = Simulation(gui=False, layout=layout, scenario=scenario, robot=drive), key
                cam = RobotCamera(gui=False)
                driver = Driver(sim, "predictive") if drive else None
            split = datasets.split_for_seed(seed)
            state = sim.reset(seed=seed)
            if drive:
                driver.reset()
            carrier, hop_left = None, 0.0
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
                    if carrier is None or not carrier.active or hop_left <= 0:
                        carrier = cars[int(rng.integers(len(cars)))]
                        hop_left = args.hop
                    pose = carrier
                sim.sync_visuals()                         # headless: move the bodies before the picture
                rgb, _, seg, _ = cam.capture(pose, lighting, config.CAMERA_JITTER, rng)
                boxes = person_boxes(seg, sim.peds)
                if config.YOLO_LABEL_CARS:
                    others = [v.figure.body_ids for v in sim.traffic.vehicles if v.active and v is not pose] \
                        if sim.traffic else []
                    boxes += car_boxes(seg, others + sim.world.parked_car_ids)
                people = sum(1 for bx in boxes if bx[0] == 0)
                if people == 0 and rng.random() > args.keep_empty:
                    continue
                rgb = augment(rgb, rng)
                name = f"{layout}_s{seed:04d}_t{step:04d}"
                save_sample(rgb, boxes, name, args.out, split, "png" if args.png else "jpg")
                iw.writerow([name, split, seed, layout, lighting, people, "robot" if drive else "traffic"])
                saved[split] += 1
            datasets.restore(old)
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
