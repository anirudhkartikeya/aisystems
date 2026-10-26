"""visuals.py - what things LOOK like, nothing else.

Every shape made here is visual only. The simulator's collision and distance
maths (world.py, pedestrians.py, traffic.py, simulation.py) treat each
pedestrian as a circle, each car as a few circles and each obstacle as its
plain box or circle, so changing how something looks never changes how it
behaves.

Each object is built from simple parts (boxes, cylinders, spheres) that are
moved together as one group. People swing their arms and legs as they walk.
"""
import atexit
import math
import os
import shutil
import tempfile

import pybullet as p

import config
import model_specs

SKIN_TONES = [[0.96, 0.80, 0.69, 1], [0.93, 0.76, 0.62, 1], [0.84, 0.64, 0.50, 1],
              [0.72, 0.52, 0.38, 1], [0.55, 0.38, 0.26, 1], [0.38, 0.26, 0.18, 1]]
HAIR_COLORS = [[0.10, 0.08, 0.06, 1], [0.25, 0.16, 0.09, 1], [0.45, 0.30, 0.15, 1],
               [0.80, 0.65, 0.35, 1], [0.60, 0.60, 0.60, 1], [0.85, 0.85, 0.85, 1], [0.55, 0.20, 0.10, 1]]
SHIRT_COLORS = [[0.90, 0.20, 0.20, 1], [0.95, 0.60, 0.10, 1], [0.60, 0.20, 0.80, 1], [0.10, 0.70, 0.70, 1],
                [0.85, 0.30, 0.60, 1], [0.50, 0.50, 0.10, 1], [0.30, 0.30, 0.30, 1], [0.95, 0.85, 0.10, 1],
                [0.95, 0.95, 0.95, 1], [0.10, 0.10, 0.12, 1], [0.20, 0.35, 0.75, 1], [0.15, 0.50, 0.25, 1],
                [0.65, 0.15, 0.20, 1], [0.55, 0.75, 0.90, 1], [0.80, 0.70, 0.55, 1], [0.45, 0.30, 0.20, 1]]
PANTS_COLORS = [[0.15, 0.18, 0.30, 1], [0.22, 0.32, 0.55, 1], [0.08, 0.08, 0.09, 1], [0.70, 0.62, 0.45, 1],
                [0.40, 0.40, 0.42, 1], [0.35, 0.25, 0.18, 1], [0.12, 0.20, 0.15, 1], [0.60, 0.60, 0.62, 1]]
COAT_COLORS = [[0.30, 0.25, 0.20, 1], [0.12, 0.12, 0.14, 1], [0.55, 0.45, 0.30, 1], [0.20, 0.25, 0.35, 1],
               [0.50, 0.10, 0.12, 1], [0.70, 0.70, 0.68, 1]]
BAG_COLORS = [[0.10, 0.10, 0.10, 1], [0.35, 0.20, 0.10, 1], [0.15, 0.25, 0.50, 1], [0.60, 0.15, 0.15, 1],
              [0.25, 0.45, 0.25, 1]]
SHOE_COLORS = [[0.08, 0.08, 0.08, 1], [0.92, 0.92, 0.9, 1], [0.35, 0.22, 0.12, 1], [0.25, 0.25, 0.3, 1],
               [0.6, 0.15, 0.15, 1]]
PANTS = PANTS_COLORS[0]
SKIN = SKIN_TONES[1]
WOOD = [0.55, 0.38, 0.22, 1]
DARK_WOOD = [0.40, 0.26, 0.14, 1]
CONCRETE = [0.70, 0.70, 0.68, 1]
METAL = [0.25, 0.25, 0.28, 1]
LEAF = [0.20, 0.55, 0.20, 1]
LEAF2 = [0.28, 0.48, 0.18, 1]
POT = [0.75, 0.40, 0.25, 1]
BARK = [0.40, 0.28, 0.18, 1]
LAMP = [1.0, 0.95, 0.6, 1]
WATER = [0.35, 0.60, 0.85, 1]


# ---------------------------------------------------------------- building blocks
def _visual_body(visual_id, position, orientation=(0, 0, 0, 1)):
    """A body that is only seen, never collides."""
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=visual_id, basePosition=position,
                             baseOrientation=orientation)


def box(half_extents, color, orientation=(0, 0, 0, 1)):
    return p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                               visualFrameOrientation=orientation)


def cylinder(radius, length, color, orientation=(0, 0, 0, 1)):
    return p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color,
                               visualFrameOrientation=orientation)


def sphere(radius, color):
    return p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)


# ---------------------------------------------------------------- Blender models
# The .obj files made by make_models.py. Every function below falls back to
# the simple shapes when a file is missing, so the simulator always runs.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODELS_DIR)
_shape_cache = {}
_tmp_dir = None


def models_on():
    return config.USE_BLENDER_MODELS and os.path.isdir(_MODELS_DIR)


def have_models(*names):
    return models_on() and all(os.path.exists(os.path.join(_MODELS_DIR, n + ".obj")) for n in names)


def mesh(name, color, scale=(1.0, 1.0, 1.0), unique=False):
    """Visual shape for models/<name>.obj, scaled and coloured. Shapes are
    shared between objects unless unique=True (needed when the colour of one
    object changes later, like brake lights and traffic lights)."""
    key = (name, tuple(round(v, 4) for v in scale), tuple(round(c, 3) for c in color))
    if not unique and key in _shape_cache:
        return _shape_cache[key]
    vid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(_MODELS_DIR, name + ".obj"),
                              meshScale=list(scale), rgbaColor=list(color))
    if not unique:
        _shape_cache[key] = vid
    return vid


def generated_mesh(name, mesh_data, color, smooth=False):
    """A visual shape from a mesh made at run time (the buildings)."""
    global _tmp_dir
    if _tmp_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="robotnav_models_")
        atexit.register(shutil.rmtree, _tmp_dir, True)
    path = os.path.join(_tmp_dir, name + ".obj")
    model_specs.write_obj(path, mesh_data, smooth)
    return p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=list(color))


def forget_shapes():
    """Call after p.disconnect(): shape ids belong to the old connection."""
    _shape_cache.clear()


class VisualGroup:
    """Several parts that move together. Each part has an offset (dx, dy, z)
    measured from the point on the ground under the object's centre.
    `swings` (optional) makes some parts limbs that rotate about a pivot:
    {part index: (pivot_z, half_length, factor)}."""

    def __init__(self, parts, x=0.0, y=0.0, yaw=0.0, swings=None):
        # parts: list of (visual_shape_id, (dx, dy, z))
        self.offsets = [offset for _, offset in parts]
        self.body_ids = [_visual_body(vid, [x + dx, y + dy, z]) for vid, (dx, dy, z) in parts]
        self.swings = swings or {}
        self.set_pose(x, y, yaw)

    def set_pose(self, x, y, yaw, swing=0.0, lift=0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        for i, (body, (dx, dy, z)) in enumerate(zip(self.body_ids, self.offsets)):
            limb = self.swings.get(i)
            if limb is None or swing == 0.0:
                p.resetBasePositionAndOrientation(body, [x + c * dx - s * dy, y + s * dx + c * dy, z + lift], orn)
                continue
            pivot_z, half_len, factor = limb
            th = swing * factor
            # the limb hangs from its pivot (hip/shoulder) and rotates about the sideways axis
            lx = dx - half_len * math.sin(th)
            lz = pivot_z - half_len * math.cos(th)
            p.resetBasePositionAndOrientation(body, [x + c * lx - s * dy, y + s * lx + c * dy, lz + lift],
                                              p.getQuaternionFromEuler([0, th, yaw]))

    def set_color(self, index, color):
        p.changeVisualShape(self.body_ids[index], -1, rgbaColor=color)

    def remove(self):
        for body in self.body_ids:
            p.removeBody(body)


# ---------------------------------------------------------------- people
def random_person_look(rng, kind, height):
    """Clothes, skin, hair and accessories for one person (appearance only)."""
    pick = lambda items: items[int(rng.integers(len(items)))]
    look = {
        "kind": kind, "height": height, "build": rng.uniform(0.85, 1.2) if kind != "child" else 0.85,
        "skin": pick(SKIN_TONES), "shirt": pick(SHIRT_COLORS), "pants": pick(PANTS_COLORS),
        "hair": pick(HAIR_COLORS[:4] if kind != "elderly" else HAIR_COLORS[4:6]),
        "coat": pick(COAT_COLORS) if rng.random() < (0.35 if kind == "elderly" else 0.15) else None,
        "hat": rng.random() < 0.10, "backpack": rng.random() < (0.35 if kind == "child" else 0.18),
        "bag": pick(BAG_COLORS) if (kind != "child" and rng.random() < 0.15) else None,
        "umbrella": rng.random() < 0.03,
    }
    look["bag_color"] = pick(BAG_COLORS)
    look["shoes"] = pick(SHOE_COLORS)
    look["hair_style"] = ("long" if rng.random() < 0.4 else "short") if kind != "elderly" else \
        ("short" if rng.random() < 0.8 else "long")
    look["hat_color"] = pick(BAG_COLORS + PANTS_COLORS[:3])
    return look


def make_person(shirt_color, x, y, look=None):
    """A human figure facing +x when yaw = 0, with arms and legs that swing.
    `look` (from random_person_look) gives height, build, clothes and
    accessories; without it you get a plain 1.7 m person in `shirt_color`.
    Returns the figure; figure.label_ids are the parts a camera label covers."""
    if not config.HUMAN_LOOK:
        fig = VisualGroup([(cylinder(config.PED_RADIUS, config.PED_HEIGHT, shirt_color),
                            (0, 0, config.PED_HEIGHT / 2))], x, y)
        fig.label_ids = list(fig.body_ids)
        return fig
    look = look or {"kind": "adult", "height": 1.7, "build": 1.0, "skin": SKIN, "shirt": shirt_color,
                    "pants": PANTS, "hair": HAIR_COLORS[1], "coat": None, "hat": False,
                    "backpack": False, "bag": None, "umbrella": False}
    if have_models("person_leg", "person_torso", "person_head"):
        return _model_person(x, y, look)
    s = look["height"] / 1.6                   # body proportions scale with height
    w = look["build"] * min(1.0, 0.75 + 0.25 * s)
    head_r = 0.115 * s * (1.18 if look["kind"] == "child" else 1.0)
    hip, shoulder = 0.82 * s, 1.32 * s
    leg_len, arm_len = 0.80 * s, 0.60 * s
    torso_color = look["coat"] or look["shirt"]
    parts = [
        (cylinder(0.065 * w, leg_len, look["pants"]), (0, 0.09 * w, hip - leg_len / 2)),        # 0 left leg
        (cylinder(0.065 * w, leg_len, look["pants"]), (0, -0.09 * w, hip - leg_len / 2)),       # 1 right leg
        (cylinder(0.05 * w, arm_len, torso_color), (0, 0.21 * w, shoulder - arm_len / 2)),      # 2 left arm
        (cylinder(0.05 * w, arm_len, torso_color), (0, -0.21 * w, shoulder - arm_len / 2)),     # 3 right arm
        (box([0.11 * w, 0.16 * w, 0.27 * s], look["shirt"]), (0, 0, 1.07 * s)),                # torso
        (sphere(head_r, look["skin"]), (0, 0, 1.47 * s)),                                       # head
    ]
    if look["coat"]:                          # a longer coat over the shirt
        parts.append((box([0.12 * w, 0.165 * w, 0.30 * s], look["coat"]), (0, 0, 0.98 * s)))
    if look["hat"]:
        parts.append((cylinder(head_r * 1.05, 0.08 * s, look["hair"]), (0, 0, 1.47 * s + head_r * 0.75)))
    else:                                     # hair: a slightly bigger sphere, pushed up and back
        parts.append((sphere(head_r * 1.02, look["hair"]), (-0.02 * s, 0, 1.47 * s + 0.035 * s)))
    if look["backpack"]:
        parts.append((box([0.07 * s, 0.12 * w, 0.15 * s], look["bag_color"]), (-0.18 * w, 0, 1.12 * s)))
    if look["bag"]:
        parts.append((box([0.05 * s, 0.11, 0.09 * s], look["bag"]), (0.0, 0.27 * w, 0.82 * s)))
    n_body = len(parts)
    if look["umbrella"]:
        parts.append((cylinder(0.012, 0.9 * s, METAL), (0.05, 0.12, 1.55 * s)))
        parts.append((cylinder(0.55, 0.04, look["bag_color"]), (0.05, 0.12, 2.0 * s)))
    swings = {0: (hip, leg_len / 2, 1.0), 1: (hip, leg_len / 2, -1.0),
              2: (shoulder, arm_len / 2, -0.7), 3: (shoulder, arm_len / 2, 0.7)}
    fig = VisualGroup(parts, x, y, swings=swings if config.ANIMATE_WALK else None)
    fig.label_ids = fig.body_ids[:n_body]       # umbrella not part of the person box
    return fig


def _model_person(x, y, look):
    """A person built from the Blender models (see model_specs.PERSON_LAYOUT)."""
    s = look["height"] / 1.6                          # the models are a 1.6 m person
    w = look["build"] * (0.6 + 0.4 * s)               # width scales a little less than height
    hs = s * (1.12 if look["kind"] == "child" else 1.0)
    colors = {"pants": look["pants"], "shoes": look.get("shoes", SHOE_COLORS[0]), "skin": look["skin"],
              "shirt": look["shirt"], "sleeves": look["coat"] or look["shirt"], "hair": look["hair"],
              "coat": look["coat"], "hat": look.get("hat_color", look["hair"]),
              "backpack": look["bag_color"], "bag": look["bag"] or look["bag_color"], "umbrella": look["bag_color"]}
    head_parts = ("person_neck", "person_head", "person_hair_short", "person_hair_long", "person_cap")
    parts, swings = [], {}

    def add(mesh_name, color, dx, dy, z, swing=None):
        head = mesh_name in head_parts
        scale = (hs, hs, hs) if head else (w, w, s)
        z_pos = z * s
        if head and mesh_name != "person_neck":
            z_pos = 1.39 * s + (z - 1.39) * hs         # bigger head sits on the same neck
        if swing is not None:
            pivot, dist, factor = swing
            swings[len(parts)] = (pivot * s, dist * s, factor)
        parts.append((mesh(mesh_name, color, scale), (dx * w, dy * w, z_pos)))

    for part, mesh_name, ckey, (dx, dy, z), swing in model_specs.PERSON_LAYOUT:
        if part == "torso" and look["coat"]:
            continue                                   # the coat goes over it
        add(mesh_name, colors[ckey], dx, dy, z, swing)
    extras = []
    if look["coat"]:
        extras.append("coat")
    extras.append("cap" if look["hat"] else ("hair_long" if look.get("hair_style") == "long" else "hair_short"))
    if look["backpack"]:
        extras.append("backpack")
    if look["bag"]:
        extras.append("bag")
    for extra in extras:
        mesh_name, ckey, (dx, dy, z) = model_specs.PERSON_EXTRAS[extra]
        if have_models(mesh_name):
            add(mesh_name, colors[ckey], dx, dy, z)
    n_body = len(parts)
    if look["umbrella"] and have_models("person_umbrella"):
        mesh_name, ckey, (dx, dy, z) = model_specs.PERSON_EXTRAS["umbrella"]
        add(mesh_name, colors[ckey], dx, dy, z)
    fig = VisualGroup(parts, x, y, swings=swings if config.ANIMATE_WALK else None)
    fig.label_ids = fig.body_ids[:n_body]
    return fig


# ---------------------------------------------------------------- cars
GLASS = [0.15, 0.20, 0.27, 1]
TYRE = [0.08, 0.08, 0.08, 1]
TAIL_OFF = [0.45, 0.05, 0.05, 1]
TAIL_ON = [1.0, 0.12, 0.10, 1]
CAR_COLORS = [[0.75, 0.12, 0.12, 1], [0.15, 0.30, 0.65, 1], [0.85, 0.85, 0.85, 1],
              [0.20, 0.20, 0.22, 1], [0.95, 0.75, 0.10, 1], [0.25, 0.50, 0.30, 1],
              [0.95, 0.95, 0.95, 1], [0.60, 0.62, 0.65, 1], [0.08, 0.08, 0.10, 1],
              [0.45, 0.10, 0.12, 1], [0.70, 0.65, 0.55, 1], [0.30, 0.45, 0.70, 1]]
VEHICLE_KINDS = {             # length, width, height (m)
    "hatch": (3.9, 1.75, 1.50), "sedan": (4.5, 1.80, 1.45), "suv": (4.7, 1.90, 1.75),
    "van": (5.0, 2.00, 2.10), "bus": (10.5, 2.50, 3.00), "taxi": (4.5, 1.80, 1.45),
}


RIM = [0.72, 0.72, 0.75, 1]
TRIM = [0.07, 0.07, 0.08, 1]


def vehicle_parts(length, width, height, color, lights=True, kind="sedan"):
    """Parts of a vehicle facing +x, centred on (0, 0), and the indices of its
    tail lights. Uses the Blender model of this kind when there is one."""
    model_kind = kind if kind in model_specs.VEHICLE_SIZES else "sedan"
    names = [f"car_{model_kind}_{part}" for part in model_specs.VEHICLE_PARTS]
    if have_models(*names):
        L0, W0, H0 = model_specs.VEHICLE_SIZES[model_kind]
        scale = (length / L0, width / W0, height / H0)
        colors = {"body": color, "glass": GLASS, "trim": TRIM, "tyres": TYRE, "rims": RIM,
                  "headlights": LAMP, "taillights": TAIL_OFF}
        parts, tail = [], []
        for part, name in zip(model_specs.VEHICLE_PARTS, names):
            if not lights and part in ("headlights", "taillights"):
                continue
            if part == "taillights":
                tail.append(len(parts))
            parts.append((mesh(name, colors[part], scale, unique=part == "taillights"), (0, 0, 0)))
        return parts, tail
    parts = car_parts(length, width, height, color, lights, kind)
    return parts, ([len(parts) - 2, len(parts) - 1] if lights else [])


def car_parts(length, width, height, color, lights=True, kind="sedan"):
    """A simple vehicle facing +x, centred on (0, 0): body, cabin with
    windows, wheels and (optionally) head and tail lights.
    Returns the parts list; the tail lights are the last two parts when lights=True."""
    L, W = length, width
    wheel_r = 0.32 if kind != "bus" else 0.48
    body_bottom = 0.25 if kind != "bus" else 0.35
    sideways = p.getQuaternionFromEuler([math.pi / 2, 0, 0])     # wheel axle along y
    if kind == "bus":
        body_h = height - body_bottom
        parts = [(box([L / 2, W / 2, body_h / 2], color), (0, 0, body_bottom + body_h / 2)),
                 (box([L / 2 - 0.3, W / 2 + 0.01, 0.45], GLASS), (0.1, 0, body_bottom + body_h * 0.62)),
                 (box([0.02, W / 2 * 0.85, 0.55], GLASS), (L / 2, 0, body_bottom + body_h * 0.6))]
        axles = (-L * 0.30, L * 0.30)
    else:
        body_half_h = 0.30 if kind != "van" else 0.45
        body_top = body_bottom + 2 * body_half_h
        cabin_half_h = max(0.15, (height - body_top) / 2)
        cabin_len = {"van": 0.40, "suv": 0.32, "hatch": 0.30}.get(kind, 0.27)
        cabin_x = {"van": 0.05, "hatch": -0.08}.get(kind, -0.06)
        parts = [
            (box([L / 2, W / 2, body_half_h], color), (0, 0, body_bottom + body_half_h)),
            (box([L * cabin_len, W / 2 * 0.92, cabin_half_h], GLASS), (L * cabin_x, 0, body_top + cabin_half_h)),
            (box([L * (cabin_len - 0.02), W / 2 * 0.90, 0.03], color), (L * cabin_x, 0, body_top + 2 * cabin_half_h)),
        ]
        if kind == "taxi":
            parts.append((box([0.15, 0.35, 0.08], [0.98, 0.95, 0.80, 1]),
                          (L * cabin_x, 0, body_top + 2 * cabin_half_h + 0.1)))
        axles = (-L * 0.32, L * 0.32)
    for ax in axles:
        for sy in (-1, 1):
            parts.append((cylinder(wheel_r, 0.22, TYRE, sideways), (ax, sy * (W / 2 - 0.10), wheel_r)))
    if lights:
        z = body_bottom + 0.35
        for sy in (-1, 1):
            parts.append((box([0.02, 0.15, 0.06], LAMP), (L / 2, sy * W / 3, z)))
        for sy in (-1, 1):
            parts.append((box([0.02, 0.15, 0.07], TAIL_OFF), (-L / 2, sy * W / 3, z)))
    return parts


def make_car(color, x=0.0, y=0.0, yaw=0.0):
    """The driving car (the robot on the street)."""
    parts, _ = vehicle_parts(config.CAR_LENGTH, config.CAR_WIDTH, config.CAR_HEIGHT, color, kind="hatch")
    return VisualGroup(parts, x, y, yaw)


class VehicleFigure(VisualGroup):
    """A moving traffic car with brake lights."""

    def __init__(self, kind, color):
        L, W, H = VEHICLE_KINDS[kind]
        if kind == "taxi":
            color = [0.98, 0.78, 0.05, 1]
        parts, self.tail = vehicle_parts(L, W, H, color, True, kind)
        super().__init__(parts, -1000.0, -1000.0, 0.0)
        self.braking = False

    def set_brake(self, on):
        if on != self.braking:
            self.braking = on
            for i in self.tail:
                self.set_color(i, TAIL_ON if on else TAIL_OFF)


# ---------------------------------------------------------------- traffic lights
LIGHT_ON = {"red": [1.0, 0.1, 0.08, 1], "yellow": [1.0, 0.8, 0.1, 1], "green": [0.1, 1.0, 0.3, 1]}
LIGHT_OFF = {"red": [0.25, 0.05, 0.05, 1], "yellow": [0.25, 0.2, 0.05, 1], "green": [0.05, 0.22, 0.08, 1]}


class TrafficLightFigure:
    """A pole with a 3-lamp head facing the cars it controls."""

    def __init__(self, x, y, yaw, height=3.2):
        names = [f"traffic_light_{k}" for k in ("pole", "housing", "red", "yellow", "green")]
        if have_models(*names):
            lift = height - 3.2                        # the model's pole is 3.2 m
            parts = [(mesh(names[0], METAL), (0, 0, lift)), (mesh(names[1], [0.1, 0.1, 0.1, 1]), (0, 0, lift))]
            for name, col in zip(names[2:], ("red", "yellow", "green")):
                parts.append((mesh(name, LIGHT_OFF[col], unique=True), (0, 0, lift)))
        else:
            parts = [(cylinder(0.07, height, METAL), (0, 0, height / 2)),
                     (box([0.12, 0.17, 0.45], [0.12, 0.12, 0.12, 1]), (0, 0, height + 0.2))]
            for k, col in enumerate(("red", "yellow", "green")):
                parts.append((sphere(0.11, LIGHT_OFF[col]), (0.11, 0, height + 0.5 - k * 0.3)))
        self.group = VisualGroup(parts, x, y, yaw)
        self.state = None

    def show(self, color):
        if color == self.state:
            return
        for k, col in enumerate(("red", "yellow", "green")):
            self.group.set_color(2 + k, LIGHT_ON[col] if col == color else LIGHT_OFF[col])
        self.state = color


class PedestrianLightFigure:
    """A small pedestrian signal (red / green box) on a short pole."""

    def __init__(self, x, y, yaw, lift):
        if have_models("ped_signal_pole", "ped_signal_box"):
            parts = [(mesh("ped_signal_pole", METAL), (0, 0, lift)),
                     (mesh("ped_signal_box", LIGHT_OFF["red"], unique=True), (0, 0, lift))]
        else:
            parts = [(cylinder(0.05, 2.4, METAL), (0, 0, 1.2 + lift)),
                     (box([0.08, 0.14, 0.18], LIGHT_OFF["red"]), (0.05, 0, 2.45 + lift))]
        self.group = VisualGroup(parts, x, y, yaw)
        self.state = None

    def show(self, color):
        if color != self.state:
            self.group.set_color(1, LIGHT_ON[color] if color in LIGHT_ON else LIGHT_OFF["red"])
            self.state = color


# ---------------------------------------------------------------- furniture & street objects
BUILDING_COLORS = [[0.80, 0.70, 0.55, 1], [0.65, 0.45, 0.38, 1], [0.72, 0.74, 0.78, 1],
                   [0.85, 0.80, 0.68, 1], [0.55, 0.58, 0.62, 1], [0.78, 0.60, 0.45, 1],
                   [0.60, 0.35, 0.30, 1], [0.88, 0.86, 0.80, 1], [0.45, 0.50, 0.55, 1]]
AWNING_COLORS = [[0.75, 0.15, 0.15, 1], [0.15, 0.40, 0.25, 1], [0.15, 0.25, 0.55, 1], [0.85, 0.55, 0.10, 1]]
WINDOW = [0.25, 0.33, 0.45, 1]
SHOP_GLASS = [0.35, 0.45, 0.55, 1]
DOOR = [0.30, 0.20, 0.12, 1]


def make_obstacle(ob):
    """Build the look of one obstacle. The footprint stays inside the
    obstacle's box or circle, so what you see matches what collides."""
    cx, cy = ob["center"]
    look = ob.get("look", "plain") if config.FURNITURE_LOOK else "plain"
    h = ob.get("height", 1.0)
    z0 = ob.get("z", 0.0)          # raised objects stand on the sidewalk
    if models_on():
        made = _model_obstacle(ob, look, h)
        if made is not None:
            parts, yaw = made
            parts = [(vid, (dx, dy, z + z0)) for vid, (dx, dy, z) in parts]
            return VisualGroup(parts, cx, cy, yaw)

    if ob["type"] == "box":
        hx, hy = ob["half_extents"]
        yaw = ob.get("yaw", 0.0)
        if look in ("bench", "parked_car", "bus_stop") and abs(math.sin(yaw)) > 0.5:
            hx, hy = hy, hx           # the box is axis-aligned; draw along the object's own length
        if look == "table":
            lx, ly, leg = hx - 0.06, hy - 0.06, h - 0.06
            parts = [(box([hx, hy, 0.03], WOOD), (0, 0, h - 0.03))]
            parts += [(cylinder(0.04, leg, DARK_WOOD), (sx * lx, sy * ly, leg / 2))
                      for sx in (-1, 1) for sy in (-1, 1)]
        elif look == "cabinet":
            parts = [(box([hx, hy, h / 2 - 0.02], WOOD), (0, 0, h / 2 - 0.02)),
                     (box([hx, hy, 0.02], DARK_WOOD), (0, 0, h - 0.02))]
        elif look == "bench":
            # long side along the group's x; the back rest is at +y (behind the seat)
            seat_h = 0.45
            parts = [(box([hx, hy, 0.03], WOOD), (0, 0, seat_h))]
            parts += [(box([0.04, hy, seat_h / 2], METAL), (sx * (hx - 0.1), 0, seat_h / 2)) for sx in (-1, 1)]
            parts.append((box([hx, 0.03, 0.2], WOOD), (0, hy - 0.03, seat_h + 0.25)))
        elif look == "parked_car":
            kind = ob.get("kind", "sedan")
            color = CAR_COLORS[ob.get("color_index", 0) % len(CAR_COLORS)]
            if "color" in ob:
                color = ob["color"]
            parts = car_parts(2 * hx, 2 * hy * 0.98, VEHICLE_KINDS.get(kind, (0, 0, h))[2] or h,
                              color, lights=False, kind=kind)
        elif look == "bus_stop":
            parts = [(box([hx, hy, 0.04], METAL), (0, 0, h)),                       # roof
                     (box([hx, 0.03, h / 2 - 0.1], [0.65, 0.75, 0.80, 1]), (0, hy - 0.05, h / 2 + 0.05)),
                     (box([0.03, hy, h / 2], METAL), (-hx + 0.05, 0, h / 2)),
                     (box([0.03, hy, h / 2], METAL), (hx - 0.05, 0, h / 2)),
                     (box([hx * 0.8, 0.2, 0.03], WOOD), (0, hy - 0.3, 0.45))]
        elif look == "planter":
            parts = [(box([hx, hy, h / 2], CONCRETE), (0, 0, h / 2)),
                     (box([hx - 0.05, hy - 0.05, 0.1], LEAF2), (0, 0, h + 0.05))]
        elif look == "building":
            parts = _building_parts(ob, hx, hy, h)
            yaw = 0.0
        else:
            parts = [(box([hx, hy, h / 2], WOOD), (0, 0, h / 2))]

    elif ob["type"] == "cylinder":
        r = ob["radius"]
        yaw = ob.get("yaw", 0.0)
        sc = ob.get("scale", 1.0)
        if look == "plant":
            parts = [(cylinder(0.6 * r, 0.45, POT), (0, 0, 0.225)),
                     (sphere(r, LEAF), (0, 0, 0.45 + 0.8 * r))]
        elif look == "lamp":
            parts = [(cylinder(r, 0.10, METAL), (0, 0, 0.05)),
                     (cylinder(0.06, 4.0, METAL), (0, 0, 2.0)),
                     (box([0.35, 0.06, 0.04], METAL), (-0.3, 0, 4.0)),
                     (sphere(0.16, LAMP), (-0.6, 0, 3.92))]
        elif look == "tree":
            # the crown is wider than the trunk but sits high above people's heads
            parts = [(cylinder(r * sc, 2.2 * sc, BARK), (0, 0, 1.1 * sc)),
                     (sphere(1.0 * sc, LEAF), (0, 0, 2.9 * sc)),
                     (sphere(0.75 * sc, LEAF2), (0.35 * sc, 0.25 * sc, 3.55 * sc))]
        elif look == "pillar":
            parts = [(cylinder(r, h, CONCRETE), (0, 0, h / 2))]
        elif look == "bin":
            parts = [(cylinder(r, 0.9, [0.15, 0.35, 0.20, 1]), (0, 0, 0.45)),
                     (cylinder(r * 1.05, 0.05, METAL), (0, 0, 0.92))]
        elif look == "hydrant":
            parts = [(cylinder(r, 0.6, [0.85, 0.10, 0.10, 1]), (0, 0, 0.30)),
                     (sphere(r * 0.9, [0.85, 0.10, 0.10, 1]), (0, 0, 0.62)),
                     (cylinder(0.05, 0.4, [0.85, 0.10, 0.10, 1], p.getQuaternionFromEuler([math.pi / 2, 0, 0])),
                      (0, 0, 0.40))]
        elif look == "sign" or look.startswith("sign_"):
            diamond = p.getQuaternionFromEuler([math.pi / 4, 0, 0])
            parts = [(cylinder(0.04, 2.6, METAL), (0, 0, 1.3)),
                     (box([0.02, 0.28, 0.28], YELLOW, diamond), (0, 0, 2.55))]
        elif look == "fountain":
            parts = [(cylinder(r, 0.5, CONCRETE), (0, 0, 0.25)),
                     (cylinder(r - 0.2, 0.05, WATER), (0, 0, 0.48)),
                     (cylinder(0.25, 1.4, CONCRETE), (0, 0, 0.7)),
                     (cylinder(0.7, 0.12, CONCRETE), (0, 0, 1.4))]
        else:
            parts = [(cylinder(r, h, WOOD), (0, 0, h / 2))]
    else:
        raise ValueError(f"Unknown obstacle type {ob['type']}")

    parts = [(vid, (dx, dy, z + z0)) for vid, (dx, dy, z) in parts]
    return VisualGroup(parts, cx, cy, yaw)


HOUSE_COLORS = [[0.95, 0.93, 0.86, 1], [0.86, 0.80, 0.66, 1], [0.70, 0.76, 0.80, 1], [0.78, 0.55, 0.45, 1],
                [0.92, 0.85, 0.70, 1], [0.66, 0.72, 0.62, 1], [0.88, 0.88, 0.88, 1], [0.80, 0.68, 0.55, 1]]
ROOF_COLORS = [[0.62, 0.25, 0.18, 1], [0.30, 0.30, 0.34, 1], [0.42, 0.28, 0.20, 1], [0.55, 0.22, 0.15, 1],
               [0.22, 0.24, 0.28, 1], [0.48, 0.35, 0.25, 1]]
DOOR_COLORS = [[0.30, 0.20, 0.12, 1], [0.15, 0.25, 0.45, 1], [0.55, 0.12, 0.12, 1], [0.20, 0.35, 0.25, 1],
               [0.92, 0.92, 0.9, 1]]
SIGN_COLORS = [[0.95, 0.92, 0.8, 1], [0.15, 0.25, 0.45, 1], [0.9, 0.85, 0.3, 1], [0.95, 0.95, 0.95, 1],
               [0.2, 0.45, 0.3, 1]]
STREET_COLORS = {"pole": METAL, "blue": [0.10, 0.30, 0.70, 1], "white": [0.96, 0.96, 0.96, 1],
                 "green": [0.08, 0.42, 0.22, 1], "bark": BARK, "leaf": LEAF, "metal": METAL, "light": LAMP, "wood": WOOD,
                 "body": [0.15, 0.35, 0.20, 1], "red": [0.85, 0.10, 0.10, 1], "glass": [0.62, 0.72, 0.80, 1],
                 "ad": [0.92, 0.92, 0.96, 1], "stone": [0.72, 0.71, 0.68, 1], "water": WATER,
                 "concrete": CONCRETE}
LEAF_TONES = [LEAF, LEAF2, [0.24, 0.50, 0.22, 1], [0.30, 0.55, 0.20, 1], [0.18, 0.42, 0.20, 1]]


def _model_obstacle(ob, look, h):
    """Parts and yaw for a street object made from the Blender models, or
    None to use the simple shapes. Sizes are matched to the obstacle's box/circle."""
    yaw = ob.get("yaw", 0.0)
    cx, cy = ob["center"]
    tone = int(abs(cx * 7.3 + cy * 3.1)) % len(LEAF_TONES)

    def parts_of(model, scale, colors=None):
        names = [f"{model}_{part}" for part in model_specs.STREET_PARTS[model]]
        if not have_models(*names):
            return None
        out = []
        for part, name in zip(model_specs.STREET_PARTS[model], names):
            col = (colors or {}).get(part, STREET_COLORS.get(part, WOOD))
            out.append((mesh(name, col, scale), (0, 0, 0)))
        return out

    if ob["type"] == "cylinder":
        r = ob["radius"]
        if look == "tree":
            sc = ob.get("scale", 1.0)
            model = "tree_tall" if int(abs(cx * 13.7 + cy * 5.3)) % 3 == 0 else "tree_round"
            parts = parts_of(model, (sc, sc, sc), {"leaf": LEAF_TONES[tone]})
        elif look == "lamp":
            parts = parts_of("lamp", (1, 1, 1))
        elif look == "bin":
            parts = parts_of("bin", (r / 0.25, r / 0.25, 1))
        elif look == "hydrant":
            parts = parts_of("hydrant", (r / 0.15, r / 0.15, 1))
        elif look == "fountain":
            parts = parts_of("fountain", (r / 2.2, r / 2.2, 1))
        elif look in ("sign_crossing", "sign_speed", "sign_street"):
            parts = parts_of(look, (1, 1, 1))
        else:
            return None
        return (parts, yaw) if parts else None

    hx, hy = ob["half_extents"]
    if look == "building":
        return _model_building(ob, hx, hy, h), 0.0
    if look in ("bench", "bus_stop", "parked_car") and abs(math.sin(yaw)) > 0.5:
        hx, hy = hy, hx
    if look == "bench":
        parts = parts_of("bench", (hx / 0.8, hy / 0.25, 1))
    elif look == "bus_stop":
        parts = parts_of("bus_stop", (hx / 1.6, hy / 0.6, h / 2.4))
    elif look == "planter":
        if hy > hx:
            hx, hy, yaw = hy, hx, math.pi / 2
        parts = parts_of("planter", (hx / 0.6, hy / 0.35, h / 0.5))
    elif look == "parked_car":
        kind = ob.get("kind", "sedan")
        color = ob.get("color", CAR_COLORS[ob.get("color_index", 0) % len(CAR_COLORS)])
        height = VEHICLE_KINDS.get(kind, (0, 0, h))[2] or h
        parts, _ = vehicle_parts(2 * hx, 2 * hy * 0.98, height, color, lights=False, kind=kind)
    else:
        return None
    return (parts, yaw) if parts else None


def _model_building(ob, hx, hy, h):
    """A building with real windows, frames, sills, ledges, a cornice and (for
    shops) a glass front, awning and sign - made fresh for its size."""
    k = ob.get("color_index", 0)
    house = ob.get("kind") == "house"
    wall = (HOUSE_COLORS if house else BUILDING_COLORS)[k % len(HOUSE_COLORS if house else BUILDING_COLORS)]
    trim = [min(1.0, c * 0.82 + 0.1) for c in wall[:3]] + [1]
    frame = [0.95, 0.94, 0.9, 1] if k % 3 else [0.2, 0.2, 0.22, 1]
    colors = {"wall": wall, "trim": trim, "glass": SHOP_GLASS if ob.get("shop") else WINDOW, "frame": frame,
              "awning": AWNING_COLORS[k % len(AWNING_COLORS)], "sign": SIGN_COLORS[k % len(SIGN_COLORS)],
              "door": DOOR_COLORS[k % len(DOOR_COLORS)], "roof": ROOF_COLORS[k % len(ROOF_COLORS)],
              "rail": [0.18, 0.18, 0.2, 1] if k % 2 else [0.55, 0.65, 0.72, 1]}
    meshes = model_specs.building_meshes(hx, hy, h, tuple(ob.get("face", (0, -1))), ob.get("shop", False), seed=k,
                                         kind=ob.get("kind", "block"), balconies=ob.get("balconies", False),
                                         garage=ob.get("garage", False))
    cx, cy = ob["center"]
    tag = f"building_{int(cx * 10)}_{int(cy * 10)}"
    return [(generated_mesh(f"{tag}_{mat}", m, colors[mat]), (0, 0, 0)) for mat, m in meshes.items()]


def _building_parts(ob, hx, hy, h):
    """A block with window rows on the street side, a door, and for shops a
    glass front and an awning. `face` is the direction the front looks."""
    color = BUILDING_COLORS[ob.get("color_index", 0) % len(BUILDING_COLORS)]
    fx, fy = ob.get("face", (0, -1))
    parts = [(box([hx, hy, h / 2], color), (0, 0, h / 2)),
             (box([hx + 0.1, hy + 0.1, 0.12], [c * 0.8 for c in color[:3]] + [1]), (0, 0, h + 0.12))]
    # front wall position and the direction along it
    if fy != 0:
        wall = (0, fy * (hy + 0.03))
        along_half = hx
        win = lambda half_w, half_h: [half_w, 0.04, half_h]
        shift = lambda t: (t, 0)
    else:
        wall = (fx * (hx + 0.03), 0)
        along_half = hy
        win = lambda half_w, half_h: [0.04, half_w, half_h]
        shift = lambda t: (0, t)
    floor = 3.4 if ob.get("shop") else 1.0
    if ob.get("shop"):
        parts.append((box(win(along_half - 0.6, 1.1), SHOP_GLASS), (wall[0], wall[1], 1.4)))
        aw = AWNING_COLORS[ob.get("color_index", 0) % len(AWNING_COLORS)]
        out = 0.55
        ax, ay = (wall[0] + fx * out, wall[1] + fy * out)
        parts.append((box([along_half - 0.4, out, 0.05] if fy != 0 else [out, along_half - 0.4, 0.05], aw),
                      (ax, ay, 2.85)))
    else:
        dx, dy = shift(along_half * 0.4)
        parts.append((box(win(0.55, 1.05), DOOR), (wall[0] + dx, wall[1] + dy, 1.05)))
    while floor + 1.2 < h:
        parts.append((box(win(along_half - 0.7, 0.5), WINDOW), (wall[0], wall[1], floor + 0.7)))
        floor += 3.0
    return parts


# ---------------------------------------------------------------- road surface
ROAD_GREY = [0.22, 0.22, 0.24, 1]
PARKING_GREY = [0.28, 0.28, 0.30, 1]
SIDEWALK = [0.78, 0.76, 0.72, 1]
CURB = [0.62, 0.62, 0.60, 1]
GRASS = [0.42, 0.58, 0.33, 1]
PARK_GRASS = [0.36, 0.60, 0.30, 1]
PAINT = [0.97, 0.97, 0.97, 1]
YELLOW = [0.95, 0.80, 0.15, 1]

Z_ROAD = 0.02          # top of the asphalt
Z_PAINT = 0.035        # top of road paint (1.5 cm above the asphalt so it never flickers)


def make_road(half_room):
    """Flat painted road along the x axis: two lanes, a dashed centre line,
    sidewalks on both sides and a zebra crossing (the small "road" layout)."""
    rw = config.ROAD_WIDTH / 2
    side = (half_room - rw) / 2                      # half-width of one sidewalk
    cw = config.CROSSWALK_WIDTH / 2
    parts = [
        (box([half_room, rw, 0.005], ROAD_GREY), (0, 0, 0.005)),
        (box([half_room, side, 0.01], SIDEWALK), (0, rw + side, 0.01)),
        (box([half_room, side, 0.01], SIDEWALK), (0, -(rw + side), 0.01)),
    ]
    x = -half_room + 0.4
    while x < half_room - 0.4:
        if abs(x - config.CROSSWALK_X) > cw + 0.4:
            parts.append((box([0.4, 0.05, 0.002], YELLOW), (x, 0, 0.012)))
        x += 1.4
    y = -rw + 0.3
    while y < rw - 0.2:
        parts.append((box([0.9 * cw, 0.15, 0.002], PAINT), (config.CROSSWALK_X, y, 0.013)))
        y += 0.6
    return VisualGroup(parts)


def _flat(parts, x, y, yaw, half_len, half_w, z_top, thick, color):
    """A flat slab centred at (x, y), long side along yaw, top at z_top."""
    q = p.getQuaternionFromEuler([0, 0, yaw])
    parts.append((box([half_len, half_w, thick / 2], color, q), (x, y, z_top - thick / 2)))


def _strip(parts, poly, s0, s1, offset, width, z_top, thick, color, overlap=0.03):
    """Lay slabs along a polyline between s0 and s1, `offset` to its left."""
    if s1 - s0 < 0.05:
        return
    line = poly.clip(s0, s1)
    if offset:
        line = line.offset(offset)
    for a, b in zip(line.pts, line.pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-3:
            continue
        _flat(parts, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.atan2(b[1] - a[1], b[0] - a[0]),
              seg / 2 + overlap, width / 2, z_top, thick, color)


def _ranges(lo, hi, holes):
    """Pieces of [lo, hi] outside every (a, b) in holes."""
    out, x = [], lo
    for a, b in sorted(holes):
        if a > x:
            out.append((x, min(a, hi)))
        x = max(x, b)
    if x < hi:
        out.append((x, hi))
    return [(a, b) for a, b in out if b - a > 0.05]


class _Surfaces:
    """Textured ground (asphalt, sidewalk slabs, grass) as a few big meshes.
    Each tile of a texture covers `tile` metres, so the pattern keeps its real
    size; every tile has its own texture coordinates in 0..1."""

    def __init__(self):
        self.tiles = {}                           # material -> list of (4 corners, (u, v) size)

    def quad(self, mat, corners, du=1.0, dv=1.0):
        self.tiles.setdefault(mat, []).append((corners, (du, dv)))

    def rect(self, mat, x0, y0, x1, y1, z, tile):
        nx, ny = max(1, int(math.ceil((x1 - x0) / tile))), max(1, int(math.ceil((y1 - y0) / tile)))
        sx, sy = (x1 - x0) / nx, (y1 - y0) / ny
        for i in range(nx):
            for j in range(ny):
                a, b = x0 + i * sx, y0 + j * sy
                self.quad(mat, [(a, b, z), (a + sx, b, z), (a + sx, b + sy, z), (a, b + sy, z)],
                          sx / tile, sy / tile)

    def strip(self, mat, poly, s0, s1, offset, width, z, tile):
        """Tiles along a polyline. Both edges are offset copies of the line, so
        neighbouring pieces share their edges exactly (no gaps on curves)."""
        if s1 - s0 < 0.05:
            return
        line = poly.clip(s0, s1)
        n_across = max(1, int(math.ceil(width / tile)))
        edges = [line.offset(offset - width / 2 + width * j / n_across) if (offset - width / 2 + width * j / n_across)
                 else line for j in range(n_across + 1)]
        w = width / n_across
        for i in range(len(line.pts) - 1):
            seg = line.s[i + 1] - line.s[i]
            if seg < 1e-3:
                continue
            n_along = max(1, int(math.ceil(seg / tile)))
            for j in range(n_across):
                a0, a1 = edges[j].pts[i], edges[j].pts[i + 1]          # right edge of this column
                b0, b1 = edges[j + 1].pts[i], edges[j + 1].pts[i + 1]  # left edge
                for k in range(n_along):
                    t0, t1 = k / n_along, (k + 1) / n_along
                    lerp = lambda p, q, t: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, z)
                    self.quad(mat, [lerp(a0, a1, t0), lerp(a0, a1, t1), lerp(b0, b1, t1), lerp(b0, b1, t0)],
                              seg / n_along / tile, w / tile)

    def build(self, fallback_colors):
        """Make one body per material and put its texture on it."""
        global _tmp_dir
        import textures
        if _tmp_dir is None:
            _tmp_dir = tempfile.mkdtemp(prefix="robotnav_models_")
            atexit.register(shutil.rmtree, _tmp_dir, True)
        try:
            tex_paths = textures.make_all(_tmp_dir)
        except Exception as err:                              # no PIL etc.: plain colours
            print("Ground textures not made:", err)
            tex_paths = {}
        for mat, tiles in self.tiles.items():
            path = os.path.join(_tmp_dir, f"ground_{mat}.obj")
            lines = ["o ground", "vn 0 0 1"]
            faces = []
            for k, (corners, (du, dv)) in enumerate(tiles):
                for x, y, z in corners:
                    lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
                lines += [f"vt 0 0", f"vt {min(du, 1):.4f} 0", f"vt {min(du, 1):.4f} {min(dv, 1):.4f}",
                          f"vt 0 {min(dv, 1):.4f}"]
                a = 4 * k + 1
                faces.append(f"f {a}/{a}/1 {a + 1}/{a + 1}/1 {a + 2}/{a + 2}/1")
                faces.append(f"f {a}/{a}/1 {a + 2}/{a + 2}/1 {a + 3}/{a + 3}/1")
            with open(path, "w") as f:
                f.write("\n".join(lines + faces) + "\n")
            tex = None
            if mat in tex_paths:
                try:
                    tex = p.loadTexture(tex_paths[mat])
                except Exception as err:
                    print("Texture not loaded:", mat, err)
            color = [1, 1, 1, 1] if tex is not None else fallback_colors[mat]
            vid = p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=color)
            body = _visual_body(vid, [0, 0, 0])
            if tex is not None:
                p.changeVisualShape(body, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1])


def make_roadmap(m):
    """Draw a road map (roadmap.RoadMap): grass, asphalt, parking lanes,
    raised sidewalks with curbs, junction boxes, lane paint, zebra crossings,
    stop lines, the park, traffic lights. Returns the traffic-light figures
    as {"vehicle": [(lane, figure)], "pedestrian": [(crossing, figure)]}."""
    from roadmap import Polyline
    lane, park_w, walk = config.LANE_WIDTH, config.PARKING_WIDTH, config.SIDEWALK_WIDTH
    curb_h = config.CURB_HEIGHT
    cw = config.STREET_CROSSWALK_WIDTH
    ext = 30.0                                  # roads continue past the map edge
    parts = []
    surf = _Surfaces() if (models_on() and config.TEXTURED_GROUND) else None

    def ground(mat, line, s0, s1, offset, width, z_top, color, tile):
        if surf is not None:
            surf.strip(mat, line, s0, s1, offset, width, z_top, tile)
        else:
            _strip(parts, line, s0, s1, offset, width, z_top, min(z_top, 0.02) if z_top < 0.05 else z_top, color)

    def ground_rect(mat, cx, cy, hx, hy, z_top, color, tile):
        if surf is not None:
            surf.rect(mat, cx - hx, cy - hy, cx + hx, cy + hy, z_top, tile)
        else:
            thick = z_top if z_top > 0.05 else 0.02
            parts.append((box([hx, hy, thick / 2], color), (cx, cy, z_top - thick / 2)))

    x0, x1, y0, y1 = m.bounds
    gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
    if surf is not None:
        surf.rect("grass", x0 - 90, y0 - 90, x1 + 90, y1 + 90, 0.0, 6.0)
    else:
        parts.append((box([(x1 - x0) / 2 + 90, (y1 - y0) / 2 + 90, 0.05], GRASS), (gx, gy, -0.05)))
    for px0, py0, px1, py1 in m.parks:
        ground_rect("park", (px0 + px1) / 2, (py0 + py1) / 2, (px1 - px0) / 2, (py1 - py0) / 2, 0.02,
                    PARK_GRASS, 4.0)

    for e in m.edges:
        pts = list(e.center.pts)
        ext_a = ext if e.a.kind == "end" else 0.0
        ext_b = ext if e.b.kind == "end" else 0.0
        if ext_a:
            (ax, ay), (bx, by) = pts[0], pts[1]
            d = math.hypot(bx - ax, by - ay)
            pts.insert(0, (ax - (bx - ax) / d * ext_a, ay - (by - ay) / d * ext_a))
        if ext_b:
            (ax, ay), (bx, by) = pts[-2], pts[-1]
            d = math.hypot(bx - ax, by - ay)
            pts.append((bx + (bx - ax) / d * ext_b, by + (by - ay) / d * ext_b))
        line = Polyline(pts)
        L = line.length
        da, db = m.arm_dir(e, True), m.arm_dir(e, False)
        s_a = ext_a + e.clip_a                         # junction box edges along this drawing line
        s_b = L - ext_b - e.clip_b
        stop_a = ext_a + m.stop_dist(e.a, da)
        stop_b = L - ext_b - m.stop_dist(e.b, db)
        # asphalt for the driving lanes
        drive = e.drive_half                            # half width of the driving lanes
        ground("asphalt", line, s_a, s_b, 0.0, 2 * drive, Z_ROAD, ROAD_GREY, 6.0)
        if e.parking:
            p_a = stop_a + (2.0 if e.a.kind == "junction" else 0.0)
            p_b = stop_b - (2.0 if e.b.kind == "junction" else 0.0)
            for side in (1, -1):
                off = side * (drive + park_w / 2)
                ground("asphalt", line, s_a, p_a, off, park_w, Z_ROAD, ROAD_GREY, 6.0)
                ground("parking", line, p_a, p_b, off, park_w, Z_ROAD, PARKING_GREY, 6.0)
                ground("asphalt", line, p_b, s_b, off, park_w, Z_ROAD, ROAD_GREY, 6.0)
        # curbs and raised sidewalks (they start after the corner squares at junctions)
        w_a = s_a + (walk if e.a.kind == "junction" else 0.0)
        w_b = s_b - (walk if e.b.kind == "junction" else 0.0)
        for side in (1, -1):
            _strip(parts, line, s_a, s_b, side * (e.half + 0.1), 0.2, curb_h + 0.01, curb_h + 0.01, CURB)
            ground("sidewalk", line, w_a, w_b, side * (e.half + 0.2 + (walk - 0.2) / 2), walk - 0.2,
                   curb_h, SIDEWALK, 2.0)
        # paint: dashed yellow centre line and white edge lines, not on crossings or in junctions
        holes = [(0.0, stop_a), (stop_b, L)]
        for s_c, c in [(line.project(*c.center)[0], c) for c in m.crossings if c.edge is e]:
            holes.append((s_c - cw / 2 - 0.6, s_c + cw / 2 + 0.6))
        if e.a.kind == "end":
            holes[0] = (0.0, 0.0)
        if e.b.kind == "end":
            holes[1] = (L, L)
        for r0, r1 in _ranges(0.0, L, holes):
            if e.lanes > 1:                             # wide road: solid double yellow centre line
                for side in (1, -1):
                    _strip(parts, line, r0, r1, side * 0.12, 0.1, Z_PAINT, 0.01, YELLOW)
            else:                                       # two-lane road: dashed yellow centre line
                s = r0 + 0.5
                while s + 2.5 < r1:
                    x, y, hdg = line.point_at(s + 1.25)
                    _flat(parts, x, y, hdg, 1.25, 0.06, Z_PAINT, 0.01, YELLOW)
                    s += 6.0
            for k in range(1, e.lanes):                 # dashed white lines between lanes going the same way
                for side in (1, -1):
                    off = side * lane * k
                    s = r0 + 0.5
                    while s + 3.0 < r1:
                        piece = line.clip(s, s + 3.0).offset(off)
                        for (ax, ay), (bx, by) in zip(piece.pts, piece.pts[1:]):
                            seg = math.hypot(bx - ax, by - ay)
                            if seg > 1e-3:
                                _flat(parts, (ax + bx) / 2, (ay + by) / 2, math.atan2(by - ay, bx - ax),
                                      seg / 2 + 0.02, 0.06, Z_PAINT, 0.01, PAINT)
                        s += 9.0
            for side in (1, -1):
                _strip(parts, line, r0, r1, side * (drive - 0.08), 0.12, Z_PAINT, 0.01, PAINT)

    # junction boxes, corner squares, closed sides of T-junctions
    for n in m.nodes:
        if n.kind != "junction":
            continue
        ground_rect("asphalt", n.x, n.y, n.hx, n.hy, Z_ROAD, ROAD_GREY, 6.0)
        for sx in (-1, 1):
            for sy in (-1, 1):
                ground_rect("sidewalk", n.x + sx * (n.hx + walk / 2), n.y + sy * (n.hy + walk / 2),
                            walk / 2, walk / 2, curb_h, SIDEWALK, 2.0)
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if d in n.arms:
                continue
            if d[1] == 0:
                c = (n.x + d[0] * (n.hx + walk / 2), n.y)
                he = [walk / 2, n.hy, curb_h / 2]
                curb = (n.x + d[0] * (n.hx + 0.1), n.y, [0.1, n.hy, (curb_h + 0.01) / 2])
            else:
                c = (n.x, n.y + d[1] * (n.hy + walk / 2))
                he = [n.hx, walk / 2, curb_h / 2]
                curb = (n.x, n.y + d[1] * (n.hy + 0.1), [n.hx, 0.1, (curb_h + 0.01) / 2])
            ground_rect("sidewalk", c[0], c[1], he[0], he[1], curb_h, SIDEWALK, 2.0)
            parts.append((box(curb[2], CURB), (curb[0], curb[1], (curb_h + 0.01) / 2)))

    # zebra crossings: white bars along the road, repeated across it
    for c in m.crossings:
        yaw = math.atan2(c.along[1], c.along[0])
        t = -c.half + 0.45
        while t < c.half - 0.3:
            x, y = c.center[0] + c.across[0] * t, c.center[1] + c.across[1] * t
            _flat(parts, x, y, yaw, cw * 0.45, 0.25, Z_PAINT, 0.01, PAINT)
            t += 0.9
    # stop lines where lanes end at junctions, and arrows showing where each lane can go
    arrows = model_specs.Mesh()
    for ln in m.lanes:
        if ln.end_node.kind == "junction":
            x, y, hdg = ln.poly.point_at(ln.length)
            _flat(parts, x - math.cos(hdg) * 0.2, y - math.sin(hdg) * 0.2, hdg, 0.2, lane / 2 - 0.1,
                  Z_PAINT, 0.01, PAINT)
            if ln.length > 14 and ln.turns:
                ax, ay, ahdg = ln.poly.point_at(ln.length - 7.0)
                arrows.merge(model_specs.road_arrow(list(ln.turns)).rotate("z", ahdg).move(ax, ay, 0.0))
    if arrows.v and models_on():
        _visual_body(generated_mesh("road_arrows", arrows, PAINT), [0, 0, 0])
    VisualGroup(parts)
    if surf is not None:
        surf.build({"asphalt": ROAD_GREY, "parking": PARKING_GREY, "sidewalk": SIDEWALK, "grass": GRASS,
                    "park": PARK_GRASS})

    # traffic lights: one per lane arriving at a junction, plus pedestrian signals
    figures = {"vehicle": [], "pedestrian": []}
    for ln in m.lanes:
        if ln.signal is None or ln.index != ln.n_lanes - 1:
            continue                                       # one light per road, by the outer lane
        x, y, hdg = ln.poly.point_at(ln.length)
        right = (math.sin(hdg), -math.cos(hdg))
        d = ln.edge.half - ln.offset + 0.45                # from the lane centre to just past the curb
        fig = TrafficLightFigure(x + right[0] * d, y + right[1] * d, hdg + math.pi, height=3.2 + curb_h)
        figures["vehicle"].append((ln, fig))
        # and a second one on the far side of the junction, easy to see from the driver's seat
        node = ln.end_node
        across = 2 * ((node.x - x) * math.cos(hdg) + (node.y - y) * math.sin(hdg)) + 1.0
        exit_arm = node.arms.get((round(math.cos(hdg)), round(math.sin(hdg))))
        d_far = exit_arm[0].half - ln.offset + 0.45 if exit_arm else d     # the curb of the road beyond
        fx, fy = x + math.cos(hdg) * across + right[0] * d_far, y + math.sin(hdg) * across + right[1] * d_far
        fig = TrafficLightFigure(fx, fy, hdg + math.pi, height=3.2 + curb_h)
        figures["vehicle"].append((ln, fig))
    for c in m.crossings:
        if c.kind != "signal":
            continue
        for side in (1, -1):
            px, py = c.curb(side)
            px += c.across[0] * side * 0.5 + c.along[0] * (cw / 2 + 0.3)
            py += c.across[1] * side * 0.5 + c.along[1] * (cw / 2 + 0.3)
            yaw = math.atan2(-c.across[1] * side, -c.across[0] * side)   # faces the people on the other side
            figures["pedestrian"].append((c, PedestrianLightFigure(px, py, yaw, curb_h)))
    return figures
