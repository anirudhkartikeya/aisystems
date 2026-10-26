# Robot navigation simulator - how to run

Open a terminal in this folder and activate your environment first
(`conda activate robotnav`, or whichever one has pybullet, numpy and pillow).

## Watch it
- `python run_sim.py` : the town. The car drives east on the 4-lane main street in the right lane,
  moves ONCE into the left lane to get ready for the left turn, turns left at the T-junction on the green
  arrow, then stays in that lane up the loop road, round its curve and along the top road to the goal.
  5 people step out in front of it one after another:
  - P1: steps into the right half of the lane and stops. The car should swerve left.
  - P2: walks across. The car should swerve right, behind them.
  - P3: crosses on a zebra crossing. The car should stop.
  - P4: steps into the left half of the lane and stops. The car should swerve right.
  - P5: random side, contact point and behaviour each run. The brain decides.
  You set all of this in `config.py` under `TOWN_CROSSERS`: side, contact point on the car's front,
  stop or cross, and the expected action. After each run, `info["crossers"]` lists what the car was
  expected to do and what it actually did.
  - People only step off the curb while the car is driving towards them. If it is stopped (a light,
    a queue), they wait on the sidewalk.
  - The text above the car shows why it is going at that speed, in brackets: `[red light]`,
    `[car ahead]`, `[waiting for oncoming traffic to turn left]`, `[changing lane ...]`.
  - The lights are timed so the car gets green at both junctions. To practise red lights, change
    `TOWN_LIGHT_OFFSETS` in `config.py` (for example `[0.0, 20.0]`).
- `python run_sim.py --scenario random` : no scripted crossers, just city life and traffic
- `python run_sim.py --robot-off` : bird's-eye view of the whole town
- `python run_sim.py --layout street` : the old long street with 3 zebra crossings
- `python run_sim.py --lighting evening` : the other options are noon, morning, overcast and dusk
- `python run_sim.py --drive-through` : keep driving after a collision
- `python run_sim.py --speed 0.5` : slow motion
- Close the window (or press Ctrl+C in the terminal) to stop.

## The brains and the planner (who drives the car)
- `python run_sim.py` : the **predictive** brain drives (set the default with `BRAIN` in `config.py`)
- `python run_sim.py --brain reactive` : the **reactive** brain drives
- `python run_sim.py --brain none` : the old placeholder that ignores people (it hits them)

How it works, every 0.1 s:
1. **Perception** (`perception.py`): the car sees people only through its front camera
   (30 m, 90° wide, nobody hidden behind buildings, parked cars or other cars), plus
   close-range sensors (3 m all round). It senses other cars all round (40 m).
2. **Brain** (`brains.py`): says where each seen person will be.
   - Reactive: where they are now.
   - Predictive: where they will be over the next 3 s, at their current walking velocity.
   - The LSTM brain slots in here in Phase 2.
3. **Planner** (`planner.py`, the same for every brain): tries many candidate paths
   (stay, shift left or right, next lane, oncoming lane only if it is free) at several speeds, checks
   each one 4 s ahead against the people and the other cars, and drives the cheapest safe one.
   - It only leaves its lane for a person. Behind slow cars it just follows.
   - At a zebra crossing it always stops, never swerves.
   - It waits at red lights and before turning left without the green arrow.
   - If nothing is safe it brakes as hard as it can.

What you see in the window:
- Blue lines: the camera's view.
- Green line: the chosen path.
- Orange lines above people's heads: where the brain thinks they will go.
- The text above the car: the brain and what it is doing, e.g. `[predictive: swerving left around a person]`.
- Turn the drawing off with `SHOW_PLAN = False` in `config.py`.

At the end of a run the terminal prints that run's metrics and, for P1-P5, what the
car was expected to do and what it did.

## Compare the brains (the project result)
- `python evaluate.py` : 50 runs with the reactive brain and the same 50 runs with the predictive brain
  (no window). It prints a table and saves `results/episodes.csv`, `results/people.csv` and `results/summary.csv`.
- `python evaluate.py --seeds 10` : a quicker check.
- `python evaluate.py --brains none reactive predictive` : also the old placeholder, as a baseline.
- `python evaluate.py --workers 8` : how many CPU cores to use (default: all but 2).

The metrics:
- **Safety**: runs hitting a person or a car, minimum gap to people, near misses (closer than 0.5 m),
  minimum time-to-collision.
- **Efficiency**: success rate (goal, no collision), time to goal, path length, time stopped.
- **Comfort**: hard brakes (stronger than 1.5 m/s²), strongest braking, steering jerk.
- **Decisions**: correct action for P1-P4, reaction gap (how far away the person was when the car first reacted).
- **Prediction**: ADE and FDE, the average and the final error (in metres) of the predictions over 3 s.

The planner's settings (margins, costs, how far ahead it looks) are all in `config.py` under
"Perception", "Brains" and "Planner".

## Realistic models (Blender)
The `models` folder already holds the 3D models: people, cars, a bus, trees, lamps,
traffic lights, benches, bus stops and a fountain. Buildings get real windows and shop
fronts automatically. To have Blender polish them, with smoothing and rounded edges:
1. Open Blender and click the **Scripting** tab at the top.
2. Click **Open**, choose `make_models.py` in this folder, then click **Run Script** (▶).
3. After a minute the models in `models/` are replaced by the smoothed ones. Blender also
   shows all the models side by side, and saves them as `models/robot_nav_models.blend`.
4. Run the simulator as usual.

- To go back to the simple shapes: set `USE_BLENDER_MODELS = False` in `config.py`.
- If the window is slow: set `GUI_SHADOWS = False` in `config.py`.
- Camera: `CAMERA_VIEW = "chase"` (behind the car, turning with it) or `"side"`. `CAMERA_SMOOTH` sets how
  gently it follows (bigger = smoother), and `CAMERA_DISTANCE` / `CAMERA_PITCH` set where it sits.
- Plain-coloured ground instead of the asphalt, slab and grass textures: set `TEXTURED_GROUND = False`.
- To edit a model by hand: change it in Blender, set `REBUILD = False` at the top of
  `make_models.py`, and run it again. Only the export runs, so your edits are kept.

## Make the datasets
Both scripts run without a window. Whole runs (seeds) go into one split - train / val / test =
14 / 3 / 3 out of every 20 seeds - so nothing from a test run leaks into training.

**LSTM data** (pedestrian positions every 0.4 s):
- `python make_data.py --episodes 3` : a quick test first.
- `python make_data.py --episodes 500` : the full set (60 s per run, town and street).
- Faster: open 4 terminals and give each its own range:
  `python make_data.py --start 0 --episodes 125`, `--start 125 --episodes 125`,
  `--start 250 --episodes 125`, `--start 375 --episodes 125`.
- Output in `trajectories/`: `peds_*.csv` (every person, every 0.4 s: x, y, velocity, what they are
  doing, on the road or not), `vehicles_*.csv` (every car; the robot car is kind "robot") and
  `episodes_*.csv` (one line per run).
- In about 3 of every 10 runs the robot car drives with the predictive brain, so people react to it.

**YOLO data** (camera pictures with exact labels):
- `python make_yolo_data.py --episodes 3` : a quick test first.
- `python make_yolo_data.py --episodes 80` : the full set, about 2,800 pictures.
- Faster: 4 terminals with `--start 0 --episodes 20`, `--start 20 --episodes 20`, and so on.
- Output in `dataset/`: `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`
  (ready for `yolo train data=dataset/data.yaml`) and `index_*.csv`.
- Different lighting, crowds, traffic and camera angles in every run. Every 4th run the camera is
  on the robot car while the predictive brain drives; otherwise it rides on traffic cars all over town.

## Where things are
| File | What it does |
|---|---|
| `config.py` | every number: sizes, speeds, crowd size, traffic, lights, lighting, camera |
| `roadmap.py` | the town and street maps: lanes, junctions, crossings, sidewalks, buildings |
| `world.py` | builds the scene and answers distance questions |
| `visuals.py` | how everything looks (people, cars, buildings, lights) |
| `pedestrians.py` | how people behave |
| `traffic.py` | how the other cars drive |
| `simulation.py` | runs everything step by step (the brains will plug in here) |
| `camera.py` | the car camera, YOLO labels, lighting, image variety |
| `perception.py` | what the car can see (front camera, close-range sensors, other cars) |
| `brains.py` | the reactive and predictive brains (and the slot for the LSTM) |
| `planner.py` | the shared planner and the Driver that connects perception, brain and planner |
| `metrics.py` | the numbers recorded for every run |
| `evaluate.py` | runs the reactive vs predictive comparison |
| `controllers.py` | the old placeholder driver |
| `model_specs.py` | the shapes of all 3D models (people, cars, street objects, buildings) |
| `make_models.py` | the Blender script that polishes and exports the models into `models/` |
| `make_data.py` | the LSTM dataset script |
| `make_yolo_data.py` | the YOLO dataset script |

The version before the Blender models is the `robot_nav_update` folder.
