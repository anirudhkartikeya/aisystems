"""config.py - every number for the simulator lives here.

Units: metres, seconds, m/s, radians (degrees only where it says so).
Change values here, never inside the other files.
"""

# ---------------- Timing ----------------
CONTROL_HZ = 10                                       # robot/pedestrian/traffic decisions per second
CONTROL_DT = 1.0 / CONTROL_HZ                         # = 0.1 s
PHYSICS_STEPS_PER_CONTROL = 10                        # small motion steps inside one decision (100 Hz)
PHYSICS_HZ = CONTROL_HZ * PHYSICS_STEPS_PER_CONTROL
MAX_TIME = 200.0                                      # trial timeout (s)
REAL_TIME_GUI = True                                  # GUI plays at real speed
SIM_SPEED = 1.25             # GUI playback speed: 1 = real time, 0.5 = slow motion, 2 = double
GUI_CAMERA_EVERY = 1         # follow camera (and the robot car) redrawn every motion step = 100 times a second
GUI_AGENT_EVERY = 3          # people and cars are redrawn every 3 motion steps = ~33 times a second
GUI_SHADOWS = True           # shadows look much nicer; set False if the window gets slow
GUI_SKY_COLOR = (0.62, 0.78, 0.95)   # background colour of the window

# ---------------- Layout ----------------
# "town"    = a small city: main street with two signalised junctions (a crossroads
#             and a T-junction), a side street, a loop road with two curves,
#             zebra crossings, traffic lights, moving traffic, parked cars,
#             buildings, a park, sidewalks with trees, lamps, benches, bus stops
# "street"  = one long straight city street with 3 zebra crossings (no junctions)
# "road"    = a short street: road with two lanes, sidewalks and one zebra crossing
# "room"    = open indoor room with furniture
# "doorway" = the room split in two by a wall with a door
LAYOUT = "town"
ROOM_SIZE = 10.0           # room/doorway/road area is ROOM_SIZE x ROOM_SIZE
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

# doorway layout
DOOR_X = 0.0               # x position of the door centre
DOOR_WIDTH = 1.4           # width of the gap

# road layout (the road runs along x through the middle)
ROAD_WIDTH = 4.0           # two lanes of 2 m; sidewalks fill the rest
CROSSWALK_X = 0.0          # x position of the zebra crossing centre
CROSSWALK_WIDTH = 2.0      # how wide the crossing is (along x)

# ---- streets (shared by "street" and "town") ----
# A road is 2 driving lanes (one each way), optionally a parking lane on each
# side, then a curb and a sidewalk. Traffic drives on the right.
LANE_WIDTH = 3.0           # each driving lane
PARKING_WIDTH = 2.0        # parking lane on each side (main streets only)
SIDEWALK_WIDTH = 3.5
CURB_HEIGHT = 0.12         # sidewalks are raised a little above the road
STREET_CROSSWALK_WIDTH = 3.0
STOP_LINE_GAP = 1.0        # m between a stop line and the zebra crossing after it
BUILDING_SETBACK = 0.6     # m between the back of a sidewalk and the buildings
PARKED_CARS = True
PARKED_CAR_FILL = 0.65     # fraction of parking spaces that hold a car

# street layout (long road along x; the car drives in +x in the right-hand lane)
STREET_LENGTH = 80.0       # m, x from -40 to +40
STREET_CROSSWALKS_X = [-18.0, 2.0, 22.0]   # x of the three zebra crossings
STREET_CROSS_SIDES = ["right", "left", "right"]   # side each scripted person comes from
STREET_CROSS_SPEED = 1.3   # m/s walking speed of the scripted crossers
STREET_START = (-36.0, -1.5)   # car starts in the middle of the right-hand lane
STREET_GOAL = (36.0, -1.5)

# town layout (see roadmap.py for the drawing of the map)
TOWN_HALF_X = 70.0         # main street runs from x = -70 to +70 along y = 0
TOWN_JUNCTIONS_X = (-30.0, 30.0)   # crossroads (with a side street going south) and T-junction
TOWN_NORTH_Y = 60.0        # the loop road's top side
TOWN_SOUTH_Y = -50.0       # where the side street leaves the map
TOWN_CURVE_RADIUS = 12.0   # the loop road's two curves
TOWN_ROUTE_LANES = 2       # lanes in EACH direction on the car's route (main street + loop road)
                           # so the car can move into the other lane to get round someone
TOWN_MIDBLOCK_X = [-58.0, 52.0]                  # zebra crossings on the main street (no lights)
TOWN_LOOP_CROSSINGS = [(30.0, 20.0), (-30.0, 28.0)]   # zebra crossings on the loop road

# The car's route: east along the main street in the outer (right) lane, ONE lane
# change into the inner lane to get ready for the LEFT TURN at the T-junction
# (x = 30), then the inner lane all the way: north up the loop road, round its
# curve and west along the top road to the goal. The car follows these points in
# its lanes; between two points in different lanes of the same road it changes lane.
# Lanes: eastbound outer y = -4.5, inner y = -1.5; loop northbound inner x = 31.5,
# outer x = 34.5; top road westbound inner y = 61.5, outer y = 64.5.
TOWN_ROUTE = [(-66.0, -4.5), (-20.0, -4.5), (-6.0, -1.5),     # main street: outer lane, then inner
              (31.5, 25.0), (-14.0, 61.5)]                    # left turn, inner lane to the goal
TOWN_START = TOWN_ROUTE[0]
TOWN_GOAL = TOWN_ROUTE[-1]

# Scripted people in the crossing scenario, one after the other along the route
# (2 before the left turn, 3 after). For each person you choose:
#   at        - where on the car's route they meet it (x, y)
#   side      - "left" or "right": which side of the car they come from
#   behaviour - "stop":  walk into the lane and stand there (blocking part of it),
#                        then walk on once the car has passed
#               "cross": keep walking across, timed to meet the car
#   contact   - where they meet the car's front, in metres LEFT of the car's centre
#               line (-0.6 = right corner, 0 = middle, +0.6 = left corner)
#   expected  - the right action, saved with each run so you can check the brain:
#               "swerve_left", "swerve_right", "stop", or "decide" (no fixed answer)
#   zebra     - True if they cross on a zebra crossing
# Any of side / behaviour / contact can be "random": a new value every seed.
TOWN_CROSSERS = [
    {"name": "P1", "at": (-50.0, -4.5), "side": "right", "behaviour": "stop", "contact": -0.5,
     "expected": "swerve_left", "zebra": False},
    {"name": "P2", "at": (8.0, -1.5), "side": "right", "behaviour": "cross", "contact": 0.0,
     "expected": "swerve_right", "zebra": False},
    {"name": "P3", "at": (30.0, 20.0), "side": "left", "behaviour": "cross", "contact": 0.0,
     "expected": "stop", "zebra": True},
    # P4: after the turn the car is in the left lane, so the free lane to swerve into is on its RIGHT
    {"name": "P4", "at": (31.5, 42.0), "side": "left", "behaviour": "stop", "contact": 0.5,
     "expected": "swerve_right", "zebra": False},
    {"name": "P5", "at": (-6.0, 61.5), "side": "random", "behaviour": "random", "contact": "random",
     "expected": "decide", "zebra": False},
]
CROSSER_RANDOM_CONTACT = 0.8   # m: "random" contact is picked between -0.8 and +0.8
CROSSER_STOP_LEAD = 3.0        # s: people who stop in the lane are standing there this long before the car arrives
CROSSER_CAR_MOVING = 0.6      # m/s: people only step off the curb while the car drives at least this fast
CROSSER_ROAD_PACE = 0.85     # on the road a scripted person walks at least this fraction of their normal pace
CROSSER_STAND_MAX = 12.0       # s: they walk on after the car has passed (or after this long)
TRAFFIC_LIGHTS = True      # signals at the two junctions (cars and people obey them)
LIGHT_GREEN_MAIN = 15.0    # s of green for the main street (east-west)
LIGHT_GREEN_SIDE = 10.0    # s of green for the side streets (north-south)
LIGHT_YELLOW = 3.0
LIGHT_ALL_RED = 1.5
LIGHT_LEFT_ARROW = 12.0    # s: eastbound cars get a green arrow first, so the car turning left at the
                           # T-junction doesn't have to wait for a gap in the oncoming traffic (0 = off)
TOWN_LIGHT_OFFSETS = [0.0, 4.0]   # s, for the lights at x = -30 and x = 30: timed so the car gets green at
                                  # the first and the green arrow at the T-junction (change them for red lights)

FOLLOW_CAMERA = True       # GUI camera follows the car along the street
CAMERA_VIEW = "chase"      # "chase" = behind the car, turning with it; "side" = from the side (road left-to-right)
CAMERA_DISTANCE = 14.0     # m from the camera to the point it looks at
CAMERA_PITCH = -30.0       # degrees, how steeply it looks down
CAMERA_SMOOTH = 0.6        # s, how gently the camera catches up with the car (bigger = smoother)


# ---------------- The car (the robot on "street" and "town") ----------------
# For collisions it is 3 overlapping circles along its length - a standard
# way planners model a car - so it is long and narrow like the real thing.
CAR_LENGTH = 3.8
CAR_WIDTH = 1.7
CAR_HEIGHT = 1.45
CAR_MAX_SPEED = 2.0        # m/s
CAR_MAX_ACCEL = 2.0        # m/s^2
CAR_GOAL_TOLERANCE = 1.0
CAR_CAMERA_HEIGHT = 1.3    # m, robot camera mounted at driver-eye height


def use_street_settings():
    """Swap the robot's limits for the car's. Called by the simulation when
    the layout is "street" or "town"."""
    globals().update(ROBOT_MAX_SPEED=CAR_MAX_SPEED, ROBOT_MAX_ACCEL=CAR_MAX_ACCEL,
                     GOAL_TOLERANCE=CAR_GOAL_TOLERANCE, CAMERA_MOUNT_HEIGHT=CAR_CAMERA_HEIGHT)


# ---------------- Scenario ----------------
# "crossing" = scripted people step out and cross right in front of the robot,
#              timed so their paths meet, plus normal pedestrians and traffic
# "random"   = pedestrians go about their business, traffic drives around
SCENARIO = "crossing"

# Crossing scenario in the room/doorway/road layouts.
#   at           - where on the robot's path they cross (0 = robot start, 1 = goal)
#   side         - "left" or "right" of the robot's direction of travel (where they come FROM)
#   start_offset - how far from the robot's path they start (m)
#   end_offset   - how far past the path, on the other side, they stop (m)
#   speed        - walking speed (m/s)
CROSSERS = {
    "room":    [{"at": 0.6, "side": "left", "start_offset": 3.0, "end_offset": 3.0, "speed": 1.2}],
    "doorway": [{"at": 0.35, "side": "right", "start_offset": 2.5, "end_offset": 1.2, "speed": 1.2}],
    "road":    [{"at": 0.5, "side": "left", "start_offset": 4.0, "end_offset": 3.0, "speed": 1.2}],
}
CROSSING_TIMING = 0.0          # s: 0 = person reaches the robot's path exactly when the robot does,
                               # 2 = person arrives 2 s after the robot, -2 = 2 s before
CROSSING_TIMING_JITTER = 0.0   # s: random +/- added per seed (use e.g. 1.0 for experiments)
CROSSING_SPEED_JITTER = 0.0    # fraction: random +/- speed change per seed (e.g. 0.2 = +/-20%)

# In the crossing scenario on the road, the robot drives along the
# right-hand lane like a car and the person crosses at the zebra crossing.
ROAD_CAR_START = (-4.5, -1.0)
ROAD_CAR_GOAL = (4.5, -1.0)

# ---------------- How many people and cars ----------------
# (per layout; the crossing scenario adds its scripted crossers on top)
N_PEDS = {"town": 30, "street": 12, "road": 3, "room": 3, "doorway": 3}
N_CARS = {"town": 12, "street": 4, "road": 0, "room": 0, "doorway": 0}
CROSSING_BACKGROUND_PEDS = {"town": 16, "street": 6, "road": 0, "room": 0, "doorway": 0}
CROSSING_CARS = {"town": 8, "street": 2, "road": 0, "room": 0, "doorway": 0}

# ---------------- Looks (appearance only, never behaviour) ----------------
# Realistic 3D models made with Blender (run make_models.py in Blender once; it
# writes them to the models/ folder). False = the simple shapes. Anything with
# no model file falls back to the simple shape, so the simulator always runs.
USE_BLENDER_MODELS = True
MODELS_DIR = "models"
TEXTURED_GROUND = True     # asphalt / sidewalk slab / grass textures (with USE_BLENDER_MODELS)
HUMAN_LOOK = True          # pedestrians drawn as people instead of cylinders
FURNITURE_LOOK = True      # obstacles drawn as tables, plants, lamps... instead of plain shapes
ANIMATE_WALK = True        # legs and arms swing while walking

# ---------------- Obstacles (room / doorway / road) ----------------
# Behaviour uses only "type" + its size: boxes are axis-aligned rectangles,
# cylinders are circles. "look" only changes how it is drawn.
OBSTACLES = [
    {"type": "cylinder", "center": (-1.5, 1.5), "radius": 0.4, "height": 1.0, "look": "plant"},
    {"type": "cylinder", "center": (1.0, -1.8), "radius": 0.35, "height": 1.0, "look": "pillar"},
    {"type": "box", "center": (1.0, 3.0), "half_extents": (0.6, 0.4), "height": 0.75, "look": "table"},
    {"type": "box", "center": (-3.2, -1.0), "half_extents": (0.4, 0.6), "height": 0.9, "look": "cabinet"},
]
ROAD_OBSTACLES = [
    {"type": "cylinder", "center": (-2.0, -2.3), "radius": 0.15, "look": "lamp"},
    {"type": "cylinder", "center": (2.0, 2.3), "radius": 0.15, "look": "lamp"},
    {"type": "cylinder", "center": (2.6, -3.9), "radius": 0.2, "look": "tree"},
    {"type": "cylinder", "center": (-3.2, 3.4), "radius": 0.2, "look": "tree"},
    {"type": "box", "center": (-1.2, 4.3), "half_extents": (0.7, 0.25), "height": 0.85, "look": "bench"},
]

# ---------------- Robot (small robot in room/doorway/road) ----------------
ROBOT_RADIUS = 0.3
ROBOT_HEIGHT = 0.3
ROBOT_MAX_SPEED = 1.0      # m/s
ROBOT_MAX_ACCEL = 1.5      # m/s^2, so it cannot stop or turn instantly
ROBOT_START = (-4.0, -4.0)
GOAL = (4.0, 4.0)
GOAL_TOLERANCE = 0.3       # robot has arrived when this close to the goal

# ---------------- Pedestrians ----------------
PED_RADIUS = 0.25          # adults (children are smaller)
PED_HEIGHT = 1.7
PED_SPEED_MIN = 0.8        # m/s, room/doorway/road walkers
PED_SPEED_MAX = 1.5
PED_GOAL_TOLERANCE = 0.4   # pedestrian picks its next waypoint when this close
PED_RELAX_TIME = 0.5       # s, how quickly a pedestrian reaches its preferred velocity
PED_REPULSION_STRENGTH = 2.0   # how hard pedestrians push away from things
PED_REPULSION_RANGE = 0.3      # m, how quickly that push fades with distance
PED_REPULSION_CUTOFF = 1.0     # m, things farther than this are ignored
PED_SPAWN_CLEARANCE = 1.5  # m, no pedestrian starts this close to robot start or goal

# Kinds of people: share of the crowd and walking speed range (m/s)
PED_KINDS = {
    "adult":   {"share": 0.80, "speed": (1.05, 1.6), "height": (1.55, 1.92), "radius": 0.25},
    "elderly": {"share": 0.10, "speed": (0.65, 0.95), "height": (1.50, 1.80), "radius": 0.25},
    "child":   {"share": 0.10, "speed": (0.90, 1.45), "height": (1.05, 1.40), "radius": 0.20},
}
# Group sizes (alone, pairs, threes, fours) and their shares
PED_GROUP_SIZES = {1: 0.60, 2: 0.27, 3: 0.10, 4: 0.03}
PED_GROUP_SPACING = 0.75       # m between people walking side by side

# Behaviour (realistic street pedestrians, "street" and "town" layouts)
PED_AWARE_SHARE = 0.75         # share who notice cars (slow, stop, step back); the rest don't
PED_PHONE_SHARE = 0.15         # share looking at a phone: slower, notice cars late
PED_JAYWALKER_SHARE = 0.12     # share who sometimes cross away from crossings / against red
PED_JAYWALK_PROB = 0.5         # chance a jaywalker crosses directly when their goal is across the road
PED_RED_LIGHT_PROB = 0.35      # chance a jaywalker ignores a red pedestrian light (if the road looks clear)
PED_TURN_BACK_PROB = 0.04      # chance someone changes their mind halfway across and goes back
PED_PAUSE_RATE = 0.02          # chance per second of stopping on the sidewalk (phone, shop window, chat)
PED_PAUSE_TIME = (1.5, 6.0)    # s, how long such a stop lasts
PED_ARRIVE_PAUSE = (2.0, 8.0)  # s, standing at a destination before heading somewhere else
PED_KERB_LOOK = (0.3, 1.5)     # s, looking left/right at the kerb before crossing a zebra
PED_GAP_ACCEPT = (2.5, 5.0)    # s, a car further away in time than this is "far enough" to cross
PED_HEADING_NOISE = 0.08       # rad, slow random wobble in walking direction (nobody walks a perfect line)
PED_SPEED_NOISE = 0.06         # fraction, slow random change in walking speed
PED_HURRY = 1.35               # speed factor when hurrying across in front of a car
PED_CAR_REPULSION = 6.0        # how hard aware people push away from cars that come close
PED_LANE_WANDER = 0.6          # m, people keep their own side offset on the sidewalk (+/-)

# Room/doorway/road walkers only (simple wander model)
PED_AVOID_ROBOT = False    # False = pedestrians ignore the robot (simpler, harder test)
PED_PAUSE_PROB = 0.005     # chance per control step of stopping (~every 20 s)
PED_PAUSE_MIN = 1.0        # s
PED_PAUSE_MAX = 3.0        # s

# ---------------- Traffic (other cars, "street" and "town") ----------------
TRAFFIC_SPEED_MAIN = 8.0       # m/s speed limit on main streets (~29 km/h)
TRAFFIC_SPEED_SIDE = 6.5       # m/s on side streets and the loop
TRAFFIC_DRIVER_SPREAD = 0.15   # +/- fraction: some drivers are faster or slower
TRAFFIC_ACCEL = 1.8            # m/s^2 comfortable acceleration
TRAFFIC_BRAKE = 3.0            # m/s^2 comfortable braking
TRAFFIC_MAX_BRAKE = 7.0        # m/s^2 emergency braking
TRAFFIC_MIN_GAP = 2.0          # m kept to the car ahead when stopped
TRAFFIC_HEADWAY = 1.3          # s time gap kept to the car ahead when driving
TRAFFIC_LOOKAHEAD = 45.0       # m along its path a driver looks for things to stop for
TRAFFIC_TURN_WEIGHTS = {"straight": 0.5, "right": 0.25, "left": 0.25}
TRAFFIC_BUS_SHARE = 0.08       # share of vehicles that are buses (main street routes)
TRAFFIC_YIELD_ZEBRA = True     # cars stop for people waiting at / walking on zebra crossings

# ---------------- Lighting (camera images + GUI) ----------------
# Each YOLO episode picks one of these at random, so the detector sees many conditions.
LIGHTING = "noon"              # the one used for normal runs
LIGHTING_PRESETS = {
    "noon":     {"dir": (0.3, 0.4, 1.0), "color": (1.00, 1.00, 0.97), "ambient": 0.60, "diffuse": 0.50,
                 "specular": 0.10, "shadow": 1, "sky": ((0.45, 0.66, 0.95), (0.80, 0.88, 0.98))},
    "morning":  {"dir": (1.0, 0.5, 0.45), "color": (1.00, 0.95, 0.85), "ambient": 0.50, "diffuse": 0.55,
                 "specular": 0.10, "shadow": 1, "sky": ((0.55, 0.70, 0.92), (0.95, 0.88, 0.78))},
    "evening":  {"dir": (-1.0, 0.3, 0.30), "color": (1.00, 0.78, 0.55), "ambient": 0.40, "diffuse": 0.60,
                 "specular": 0.05, "shadow": 1, "sky": ((0.35, 0.40, 0.65), (0.98, 0.70, 0.45))},
    "overcast": {"dir": (0.0, 0.0, 1.0), "color": (0.88, 0.90, 0.95), "ambient": 0.75, "diffuse": 0.25,
                 "specular": 0.02, "shadow": 0, "sky": ((0.62, 0.64, 0.68), (0.80, 0.81, 0.83))},
    "dusk":     {"dir": (-0.5, -1.0, 0.15), "color": (0.75, 0.70, 0.85), "ambient": 0.32, "diffuse": 0.40,
                 "specular": 0.02, "shadow": 0, "sky": ((0.12, 0.14, 0.30), (0.45, 0.35, 0.50))},
}

# ---------------- Robot camera (Phase 2, for YOLO) ----------------
CAMERA_WIDTH = 640         # pixels
CAMERA_HEIGHT_PX = 480     # pixels
CAMERA_FOV = 60.0          # vertical field of view, degrees
CAMERA_MOUNT_HEIGHT = 0.45 # m above the floor, on top of the robot
CAMERA_NEAR = 0.05         # m, closest distance the camera renders
CAMERA_FAR = 60.0          # m, farthest distance the camera renders
CAMERA_EVERY = 10          # save one image every N control steps (10 = once per second)
CAMERA_MIN_BOX = 6         # px, ignore people smaller than this in the image
CAMERA_MIN_PIXELS = 25     # visible pixels a person needs to get a label (heavily hidden = no label)
DATASET_DIR = "dataset"    # images/ and labels/ are written inside this folder
# Variety for YOLO (applied by make_yolo_data.py, off for normal runs)
CAMERA_JITTER = {"height": 0.10, "pitch_deg": 3.0, "yaw_deg": 2.0, "fov_deg": (52.0, 70.0)}
IMAGE_AUGMENT = {"brightness": 0.15, "contrast": 0.15, "noise": 4.0, "blur_prob": 0.2}
YOLO_LABEL_CARS = False    # True = also label cars (class 1); the project only needs people (class 0)

# ---------------- Datasets (make_data.py, make_yolo_data.py) ----------------
# Whole episodes (seeds) go to one split, never mixed: seed % 20 -> 0-13 train, 14-16 val, 17-19 test
SPLIT_OF_20 = {"train": range(0, 14), "val": range(14, 17), "test": range(17, 20)}
DATA_VARIETY = True        # each episode gets its own crowd size, traffic and behaviour mix

# ---------------- Perception (what the car can see, perception.py) ----------------
PERCEPTION_RANGE = 30.0        # m: the front camera sees people up to this far ahead
PERCEPTION_FOV_DEG = 90.0      # degrees: width of the camera's view
PERCEPTION_NEAR = 3.0          # m: close-range sensors notice anyone this close to the car, any direction
PERCEPTION_VEHICLE_RANGE = 40.0  # m: other cars are sensed all around (mirrors + side sensors)
PERCEPTION_CAMERA_FORWARD = 0.5  # m: camera sits this far ahead of the car's centre
PERCEPTION_OCCLUDER_HEIGHT = 1.4 # m: boxes at least this tall hide people (buildings, parked cars, shelters)
PERCEPTION_NOISE = 0.0         # m: random error added to seen positions (0 = exact; try 0.1 for realism)
TRACK_HISTORY = 3.0            # s of past positions kept for each person
TRACK_FORGET = 1.0             # s: forget a person not seen for this long

# ---------------- Brains (brains.py) ----------------
BRAIN = "predictive"           # run_sim.py default: "reactive", "predictive" or "none" (old placeholder driver)
PREDICTION_HORIZON = 3.0       # s: how far ahead the predictive brain looks (car at 2 m/s: 6 m)
PREDICTION_STEP = 0.25         # s between predicted points
VELOCITY_WINDOW = 0.6          # s of history used to estimate a person's velocity
VELOCITY_STILL = 0.15          # m/s: slower than this counts as standing still

# ---------------- Planner (planner.py) ----------------
PLAN_HORIZON = 4.0             # s: each candidate path is checked this far into the future
PLAN_DT = 0.2                  # s between checked points
PLAN_OFFSETS = (-6.0, 6.0, 0.5)  # m: sideways targets tried, from, to, step (+ = left of the route)
PLAN_SPEEDS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)  # fractions of the allowed speed tried
PLAN_CORRIDOR = 12.0           # m: after the horizon, check this much more of the lane for people blocking it
PLAN_SHIFT_LENGTH = (3.0, 2.5)  # m: a sideways move takes max(3, 2.5 x shift) metres of road
PLAN_LOOKAHEAD = (2.5, 1.2)    # m, s: steer at a point max(2.5 m, 1.2 s x speed) ahead
PED_MARGIN = 0.5               # m: a path closer than this to a (predicted) person is not allowed
PED_COMFORT = 1.2              # m: closer than this is allowed but costs extra
VEH_MARGIN = 0.4               # m: same for other cars
VEH_COMFORT = 1.2
# cost weights: the planner picks the cheapest safe path
COST_SIDEWAYS = 1.0            # per metre away from the route
COST_ONCOMING = 4.0            # for using the oncoming lane (only when it is free)
COST_SLOW = 3.5                # for going slower than allowed (full stop = 2.5)
COST_WAIT = 1.0                # extra per second already spent held up by a person (then it goes round)
COST_HARD_BRAKE = 1.0          # for braking hard instead of gently
COMFORT_DECEL = 1.0            # m/s^2: gentle braking (hard braking uses the car's maximum, 2.0)
COST_CHANGE = 0.5              # per metre of changing the plan (avoids zig-zag)
COST_PED_CLOSE = 8.0           # per metre inside PED_COMFORT
COST_VEH_CLOSE = 3.0           # per metre inside VEH_COMFORT
LEFT_TURN_HOLD = 2.0           # m: if it rolled over the line, it waits this far in (still straight in its lane)
BEND_MIN_SPEED = 0.6           # fraction of top speed it keeps in the tightest bends and turns
LEFT_TURN_GAP = 80.0           # m: turning left without an arrow, no oncoming car may be closer than this
YIELD_ZONE = 2.0               # m: people this close to a zebra crossing on the route make the car stop for it
SHOW_PLAN = True               # GUI: draw the camera view, predictions and the chosen path
SHOW_PLAN_EVERY = 5            # redraw them every N steps (5 = twice a second; more = smoother window)
SAFETY_MARGIN = PED_MARGIN     # (old name)
INFLUENCE_RADIUS = 2.0

# ---------------- Experiments ----------------
SEED = 0
NEAR_MISS_DIST = 0.5       # m, gap between robot and pedestrian surfaces
HARD_BRAKE = 1.5           # m/s^2: braking harder than this counts as a hard brake (the car can do 2.0)
STOP_ON_COLLISION = True   # end the trial at the first collision
SHOW_PED_TRAILS = False    # GUI: short coloured trail behind every pedestrian
