"""run_sim.py - watch the simulator run.

    python run_sim.py                           # settings from config.py (town, crossing scenario)
    python run_sim.py --layout street           # the long straight street with 3 zebra crossings
    python run_sim.py --scenario random         # no scripted crossers: just city life and traffic
    python run_sim.py --robot-off               # watch the town without the robot car (bird's-eye view)
    python run_sim.py --drive-through           # keep driving after a collision, see the whole route
    python run_sim.py --no-follow               # camera stays put (Ctrl+drag to look around)
    python run_sim.py --peds 40 --cars 15       # busier
    python run_sim.py --lighting evening        # noon, morning, evening, overcast, dusk
    python run_sim.py --timing 2                # scripted person arrives 2 s after the robot (near miss)
    python run_sim.py --seed 3                  # a different random draw
    python run_sim.py --camera                  # also save robot-camera images + YOLO labels
    python run_sim.py --headless                # no window, just the result (fast)
    python run_sim.py --layout room             # the original indoor layouts: room, doorway, road

    python run_sim.py --brain reactive          # who drives: predictive (default), reactive, or
    python run_sim.py --brain none              #   none = the old placeholder that ignores people

The car is driven by a brain (brains.py) + the shared planner (planner.py):
it sees people only through its front camera (perception.py), predicts where
they will go (or not, for the reactive brain) and picks a safe path: keep
going, slow down, swerve, or stop. The text above the car says what it is
doing and why. At the end the run's metrics are printed (metrics.py).
"""
import argparse
import time

import pybullet as p

import config
from controllers import placeholder_controller
from metrics import EpisodeMetrics
from scenarios import SCENARIOS
from simulation import Simulation
from world import LAYOUTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--layout", choices=LAYOUTS, default=config.LAYOUT)
    parser.add_argument("--scenario", choices=SCENARIOS, default=config.SCENARIO)
    parser.add_argument("--timing", type=float, default=None,
                        help="crossing: seconds the person arrives after the robot (0 = same moment)")
    parser.add_argument("--peds", type=int, default=None, help="number of (background) pedestrians")
    parser.add_argument("--cars", type=int, default=None, help="number of traffic cars")
    parser.add_argument("--lighting", choices=list(config.LIGHTING_PRESETS), default=None)
    parser.add_argument("--robot-off", action="store_true", help="no robot: just watch the town")
    parser.add_argument("--camera", action="store_true", help="save robot-camera images + labels")
    parser.add_argument("--drive-through", action="store_true",
                        help="don't stop at a collision (to watch the whole drive)")
    parser.add_argument("--no-follow", action="store_true", help="fixed camera you can move")
    parser.add_argument("--speed", type=float, default=None,
                        help="playback speed: 1 = real time, 0.5 = slow motion, 2 = double")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--brain", choices=["predictive", "reactive", "none"], default=config.BRAIN,
                        help="who drives the car (none = old placeholder that ignores people)")
    args = parser.parse_args()
    if args.drive_through:
        config.STOP_ON_COLLISION = False
    if args.no_follow:
        config.FOLLOW_CAMERA = False
    if args.speed is not None:
        config.SIM_SPEED = args.speed
    if args.timing is not None:
        config.CROSSING_TIMING = args.timing
    if args.lighting:
        config.LIGHTING = args.lighting
    if args.peds is not None:
        table = "CROSSING_BACKGROUND_PEDS" if args.scenario == "crossing" else "N_PEDS"
        getattr(config, table)[args.layout] = args.peds
    if args.cars is not None:
        table = "CROSSING_CARS" if args.scenario == "crossing" else "N_CARS"
        getattr(config, table)[args.layout] = args.cars

    sim = Simulation(gui=not args.headless, layout=args.layout, scenario=args.scenario,
                     robot=not args.robot_off)
    state = sim.reset(seed=args.seed)
    waypoints = sim.world.route(sim.start, sim.goal)
    driver = None
    if args.brain != "none" and not args.robot_off and sim.world.roadmap is not None:
        from planner import Driver
        driver = Driver(sim, args.brain)
        driver.reset()
    elif args.brain != "none" and sim.world.roadmap is None:
        print("The brains + planner drive the car on the street and town layouts; "
              "using the placeholder driver here.")
    rec = EpisodeMetrics(sim, driver) if not args.robot_off else None

    cam = None
    if args.camera and not args.robot_off:
        from camera import RobotCamera, person_boxes, save_sample
        cam = RobotCamera(gui=not args.headless)
    saved = 0
    step = 0

    try:
        while True:
            if args.robot_off:
                vx = vy = 0.0
            elif driver is not None:
                vx, vy = driver.act(state)
            else:
                vx, vy = placeholder_controller(sim, state, waypoints)
            state, info = sim.step(vx, vy)
            if rec:
                rec.step()
            if cam and step % config.CAMERA_EVERY == 0:
                sim.sync_visuals()
                rgb, depth, seg, _ = cam.capture(sim.robot)
                boxes = person_boxes(seg, sim.peds)
                save_sample(rgb, boxes, f"{args.layout}_seed{args.seed}_step{step:04d}")
                saved += 1
            step += 1
            if info["done"]:
                break
    except p.error:
        print("Window closed.")
        return

    print(f"Result: {info['result']}  |  time {info['time']:.1f} s  |  "
          f"path {info['path_length']:.2f} m  |  min gap {info['min_gap']:.2f} m  |  "
          f"collisions {info['collisions']}  |  {args.scenario} / {args.layout}  |  seed {info['seed']}")
    if rec and info["crossers"] is not None:
        row, people = rec.finish(info, args.brain)
        print(f"Brain: {args.brain}  |  success {bool(row['success'])}  |  hit people {row['ped_collisions']}  |  "
              f"hit cars {row['vehicle_collisions']}  |  near misses {row['near_misses']}  |  "
              f"hard brakes {row['hard_brakes']}  |  time stopped {row['time_stopped']} s  |  "
              f"ADE {row['ade']} m  FDE {row['fde']} m")
        for p in people:
            ok = "OK " if p["action"] == p["expected"] and p["min_gap"] > 0 else ("-- " if p["expected"] == "decide" else "NO ")
            print(f"   {ok}{p['name']}: expected {p['expected']:12s} did {str(p['action']):12s} "
                  f"closest {p['min_gap']:.2f} m  ({p['behaviour']}, from the {p['side']}, contact {p['contact']:+.1f})")
    if cam:
        print(f"Saved {saved} camera images to '{config.DATASET_DIR}/images' "
              f"and labels to '{config.DATASET_DIR}/labels'")

    if not args.headless:
        print("Close the PyBullet window to exit.")
        while p.isConnected():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
