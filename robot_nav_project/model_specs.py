"""model_specs.py - the 3D models (meshes) for the realistic look.

Plain Python, no PyBullet and no Blender needed. Every model is made of a
few "parts"; each part is one mesh with one colour (PyBullet gives each mesh
one colour, so a car is body + glass + trim + tyres + rims + lights).

Used by:
  make_models.py  - run inside Blender: turns these meshes into Blender
                    objects, smooths/bevels them and exports .obj files to
                    models/  (also works without Blender: `python make_models.py`)
  visuals.py      - loads the .obj files and places the parts (the layout
                    tables below say where each part goes)

Conventions: metres, x = forward, y = left, z = up. People are modelled
1.6 m tall (scaled per person). Vehicles and street objects have their
origin on the ground under their centre. Collisions never use these meshes.
"""
import math
import os
import random


# ======================================================================= mesh basics
class Mesh:
    def __init__(self):
        self.v = []          # [x, y, z]
        self.f = []          # tuples of vertex indices (triangles or quads), outward winding

    def merge(self, other):
        n = len(self.v)
        self.v.extend([list(p) for p in other.v])
        self.f.extend([tuple(i + n for i in face) for face in other.f])
        return self

    def copy(self):
        m = Mesh()
        m.v = [list(p) for p in self.v]
        m.f = list(self.f)
        return m

    def move(self, dx, dy, dz):
        for p in self.v:
            p[0] += dx; p[1] += dy; p[2] += dz
        return self

    def scale(self, sx, sy, sz):
        for p in self.v:
            p[0] *= sx; p[1] *= sy; p[2] *= sz
        return self

    def rotate(self, axis, angle):
        c, s = math.cos(angle), math.sin(angle)
        for p in self.v:
            x, y, z = p
            if axis == "x":
                p[1], p[2] = c * y - s * z, s * y + c * z
            elif axis == "y":
                p[0], p[2] = c * x + s * z, -s * x + c * z
            else:
                p[0], p[1] = c * x - s * y, s * x + c * y
        return self

    def triangles(self):
        for face in self.f:
            for k in range(1, len(face) - 1):
                yield face[0], face[k], face[k + 1]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return [a[0] / n, a[1] / n, a[2] / n]


def _face_normal(m, face):
    a, b, c = m.v[face[0]], m.v[face[1]], m.v[face[2]]
    n = _cross(_sub(b, a), _sub(c, a))
    if len(face) == 4:                      # use both diagonals for quads
        d = m.v[face[3]]
        n2 = _cross(_sub(c, a), _sub(d, a))
        n = [n[0] + n2[0], n[1] + n2[1], n[2] + n2[2]]
    return n


def _orient(m, axis_point=None, axis="z"):
    """Make every face point outwards: away from the centre, or (axis_point
    given) away from the line through axis_point along `axis`."""
    cx = sum(p[0] for p in m.v) / len(m.v)
    cy = sum(p[1] for p in m.v) / len(m.v)
    cz = sum(p[2] for p in m.v) / len(m.v)
    out = []
    for face in m.f:
        fc = [sum(m.v[i][k] for i in face) / len(face) for k in range(3)]
        ref = [cx, cy, cz]
        if axis_point is not None:
            k = {"x": 0, "y": 1, "z": 2}[axis]
            ref = list(axis_point)
            ref[k] = fc[k]
            if math.dist(fc, ref) < 1e-6:     # a cap: compare with the centre instead
                ref = [cx, cy, cz]
        if _dot(_face_normal(m, face), _sub(fc, ref)) < 0:
            face = tuple(reversed(face))
        out.append(face)
    m.f = out
    return m


# ======================================================================= primitives
def box(cx, cy, cz, hx, hy, hz):
    return tapered_box(cx, cy, cz, (hx, hy), (hx, hy), hz)


def tapered_box(cx, cy, cz, bottom, top, hz, shift=(0.0, 0.0)):
    """A box whose top face can be smaller (bottom/top half sizes) and shifted."""
    m = Mesh()
    for k, (hx, hy, z, dx, dy) in enumerate(((bottom[0], bottom[1], -hz, 0, 0),
                                              (top[0], top[1], hz, shift[0], shift[1]))):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            m.v.append([cx + dx + sx * hx, cy + dy + sy * hy, cz + z])
    m.f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return _orient(m)


def loft(rings, seg=12, cap_bottom=True, cap_top=True):
    """A smooth body through horizontal ellipses: rings = [(z, rx, ry, dx)]."""
    m = Mesh()
    for z, rx, ry, dx in rings:
        for k in range(seg):
            a = 2 * math.pi * k / seg
            m.v.append([dx + rx * math.cos(a), ry * math.sin(a), z])
    n = len(rings)
    for r in range(n - 1):
        for k in range(seg):
            a, b = r * seg + k, r * seg + (k + 1) % seg
            m.f.append((a, b, b + seg, a + seg))
    if cap_bottom:
        m.v.append([rings[0][3], 0.0, rings[0][0]])
        c = len(m.v) - 1
        for k in range(seg):
            m.f.append((c, (k + 1) % seg, k))
    if cap_top:
        m.v.append([rings[-1][3], 0.0, rings[-1][0]])
        c = len(m.v) - 1
        base = (n - 1) * seg
        for k in range(seg):
            m.f.append((c, base + k, base + (k + 1) % seg))
    return _orient(m, axis_point=(sum(r[3] for r in rings) / n, 0.0, 0.0), axis="z")


def cylinder(cx, cy, cz, r_bottom, r_top, h, seg=14, axis="z"):
    """Cylinder/cone of height h centred at (cx, cy, cz) along `axis`."""
    m = loft([(-h / 2, r_bottom, r_bottom, 0.0), (h / 2, r_top, r_top, 0.0)], seg)
    if axis == "x":
        m.rotate("y", math.pi / 2)
    elif axis == "y":
        m.rotate("x", math.pi / 2)
    return m.move(cx, cy, cz)


def ellipsoid(cx, cy, cz, rx, ry, rz, seg=12, rings=8):
    m = Mesh()
    m.v.append([0.0, 0.0, -rz])
    for i in range(1, rings):
        t = math.pi * i / rings - math.pi / 2
        for k in range(seg):
            a = 2 * math.pi * k / seg
            m.v.append([rx * math.cos(t) * math.cos(a), ry * math.cos(t) * math.sin(a), rz * math.sin(t)])
    m.v.append([0.0, 0.0, rz])
    top = len(m.v) - 1
    for k in range(seg):
        m.f.append((0, 1 + (k + 1) % seg, 1 + k))
    for i in range(rings - 2):
        for k in range(seg):
            a = 1 + i * seg + k
            b = 1 + i * seg + (k + 1) % seg
            m.f.append((a, b, b + seg, a + seg))
    last = 1 + (rings - 2) * seg
    for k in range(seg):
        m.f.append((top, last + k, last + (k + 1) % seg))
    _orient(m)
    return m.move(cx, cy, cz)


def beam(p0, p1, hw, hd):
    """A thin box from point p0 to point p1 (half width hw, half depth hd)."""
    d = _sub(p1, p0)
    length = math.sqrt(_dot(d, d))
    m = box(0, 0, 0, hd, hw, length / 2)
    # rotate +z onto the direction d
    yaw = math.atan2(d[1], d[0])
    pitch = math.acos(max(-1.0, min(1.0, d[2] / length)))
    m.rotate("y", pitch).rotate("z", yaw)
    return m.move((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)


def jitter(m, amount, seed):
    rnd = random.Random(seed)
    for p in m.v:
        p[0] += rnd.uniform(-amount, amount)
        p[1] += rnd.uniform(-amount, amount)
        p[2] += rnd.uniform(-amount, amount)
    return m


def merged(*meshes):
    out = Mesh()
    for m in meshes:
        out.merge(m)
    return out


# ======================================================================= people
# Placement of each person part (for a 1.6 m person; visuals.py scales it):
#   part: (mesh, colour key, (dx, dy, z), swing)   swing = (pivot z, distance below pivot, factor)
PERSON_LAYOUT = [
    ("leg_l", "person_leg", "pants", (0.0, 0.09, 0.45), (0.82, 0.37, 1.0)),
    ("leg_r", "person_leg", "pants", (0.0, -0.09, 0.45), (0.82, 0.37, -1.0)),
    ("shoe_l", "person_shoe", "shoes", (0.035, 0.09, 0.04), (0.82, 0.78, 1.0)),
    ("shoe_r", "person_shoe", "shoes", (0.035, -0.09, 0.04), (0.82, 0.78, -1.0)),
    ("arm_l", "person_arm", "sleeves", (0.0, 0.215, 1.055), (1.33, 0.275, -0.7)),
    ("arm_r", "person_arm", "sleeves", (0.0, -0.215, 1.055), (1.33, 0.275, 0.7)),
    ("hand_l", "person_hand", "skin", (0.0, 0.215, 0.735), (1.33, 0.595, -0.7)),
    ("hand_r", "person_hand", "skin", (0.0, -0.215, 0.735), (1.33, 0.595, 0.7)),
    ("pelvis", "person_pelvis", "pants", (0.0, 0.0, 0.86), None),
    ("torso", "person_torso", "shirt", (0.0, 0.0, 1.12), None),
    ("neck", "person_neck", "skin", (0.0, 0.0, 1.39), None),
    ("head", "person_head", "skin", (0.0, 0.0, 1.50), None),
]
PERSON_EXTRAS = {          # optional parts: (mesh, colour key, (dx, dy, z))
    "coat": ("person_coat", "coat", (0.0, 0.0, 0.995)),
    "hair_short": ("person_hair_short", "hair", (0.0, 0.0, 1.50)),
    "hair_long": ("person_hair_long", "hair", (0.0, 0.0, 1.50)),
    "cap": ("person_cap", "hat", (0.0, 0.0, 1.50)),
    "backpack": ("person_backpack", "backpack", (-0.18, 0.0, 1.12)),
    "bag": ("person_bag", "bag", (0.0, 0.28, 0.82)),
    "umbrella": ("person_umbrella", "umbrella", (0.05, 0.12, 1.95)),
}


def person_models():
    p = {}
    p["person_leg"] = loft([(-0.37, 0.05, 0.05, 0.0), (-0.18, 0.06, 0.062, 0.005), (0.0, 0.055, 0.056, 0.0),
                            (0.2, 0.072, 0.074, 0.0), (0.37, 0.08, 0.082, 0.0)], seg=10)
    p["person_shoe"] = tapered_box(0, 0, 0, (0.125, 0.047), (0.10, 0.042), 0.04, shift=(-0.02, 0))
    p["person_pelvis"] = loft([(-0.08, 0.095, 0.15, 0.0), (0.0, 0.105, 0.165, 0.0), (0.07, 0.10, 0.155, 0.0)], seg=12)
    p["person_torso"] = loft([(-0.25, 0.10, 0.148, 0.0), (-0.10, 0.104, 0.152, 0.0), (0.06, 0.115, 0.172, 0.005),
                              (0.18, 0.11, 0.19, 0.0), (0.24, 0.085, 0.16, 0.0), (0.27, 0.05, 0.07, 0.0)], seg=14)
    p["person_coat"] = loft([(-0.38, 0.135, 0.19, 0.0), (-0.1, 0.122, 0.168, 0.0), (0.12, 0.122, 0.185, 0.005),
                             (0.3, 0.115, 0.198, 0.0), (0.36, 0.09, 0.165, 0.0), (0.39, 0.055, 0.075, 0.0)], seg=14)
    p["person_arm"] = loft([(-0.29, 0.036, 0.036, 0.0), (-0.12, 0.044, 0.044, 0.0), (0.0, 0.043, 0.043, 0.0),
                            (0.15, 0.052, 0.05, 0.0), (0.28, 0.056, 0.054, 0.0)], seg=10)
    p["person_hand"] = ellipsoid(0, 0, 0, 0.028, 0.04, 0.052, seg=8, rings=6)
    p["person_neck"] = cylinder(0, 0, 0, 0.045, 0.042, 0.09, seg=10)
    p["person_head"] = merged(ellipsoid(0, 0, 0, 0.094, 0.083, 0.112, seg=14, rings=10),
                              ellipsoid(0.088, 0, -0.012, 0.022, 0.016, 0.026, seg=6, rings=4),       # nose
                              ellipsoid(0.0, 0.083, 0.0, 0.012, 0.016, 0.025, seg=6, rings=4),        # ears
                              ellipsoid(0.0, -0.083, 0.0, 0.012, 0.016, 0.025, seg=6, rings=4))
    p["person_hair_short"] = ellipsoid(-0.014, 0, 0.032, 0.1, 0.09, 0.094, seg=14, rings=10)
    p["person_hair_long"] = merged(ellipsoid(-0.014, 0, 0.032, 0.1, 0.09, 0.094, seg=14, rings=10),
                                   ellipsoid(-0.055, 0, -0.08, 0.06, 0.088, 0.15, seg=12, rings=8))
    p["person_cap"] = merged(cylinder(-0.01, 0, 0.07, 0.1, 0.092, 0.07, seg=14),
                             tapered_box(0.105, 0, 0.042, (0.06, 0.085), (0.055, 0.08), 0.008))
    p["person_backpack"] = box(0, 0, 0, 0.07, 0.13, 0.16)
    p["person_bag"] = merged(box(0, 0, 0, 0.05, 0.11, 0.09), beam([0, 0, 0.09], [0, -0.04, 0.42], 0.012, 0.006))
    p["person_umbrella"] = merged(cylinder(0, 0, 0.11, 0.56, 0.02, 0.22, seg=12),
                                  cylinder(0, 0, -0.45, 0.012, 0.012, 0.9, seg=6))
    finish = {k: "smooth" for k in p}
    for k in ("person_shoe", "person_backpack", "person_bag", "person_umbrella", "person_cap"):
        finish[k] = "bevel"
    return p, finish


# ======================================================================= vehicles
VEHICLE_SIZES = {             # length, width, height (m) - same as visuals.VEHICLE_KINDS
    "hatch": (3.9, 1.75, 1.50), "sedan": (4.5, 1.80, 1.45), "suv": (4.7, 1.90, 1.75),
    "van": (5.0, 2.00, 2.10), "bus": (10.5, 2.50, 3.00), "taxi": (4.5, 1.80, 1.45),
}
VEHICLE_PARTS = ["body", "glass", "trim", "tyres", "rims", "headlights", "taillights"]   # taillights last


def _wheels(L, W, rw, width, axles):
    tyres, rims = Mesh(), Mesh()
    for ax in axles:
        for sy in (-1, 1):
            y = sy * (W / 2 - width / 2 + 0.03)          # tyres stick out a little past the body
            tyres.merge(cylinder(ax, y, rw, rw, rw, width, seg=16, axis="y"))
            rims.merge(cylinder(ax, y, rw, rw * 0.58, rw * 0.58, width + 0.02, seg=12, axis="y"))
    return tyres, rims


def car_model(kind):
    L, W, H = VEHICLE_SIZES[kind]
    parts = {k: Mesh() for k in VEHICLE_PARTS}
    if kind == "bus":
        rw, bottom = 0.5, 0.35
        hz = (H - bottom) / 2
        parts["body"].merge(box(0, 0, bottom + hz, L / 2, W / 2, hz))
        parts["glass"].merge(box(0.15, 0, 1.95, L / 2 - 0.45, W / 2 + 0.012, 0.5))                 # window band
        parts["glass"].merge(box(L / 2 + 0.01, 0, 1.75, 0.012, W / 2 - 0.12, 0.72))               # windscreen
        parts["glass"].merge(box(-L / 2 - 0.01, 0, 2.0, 0.012, W / 2 - 0.25, 0.42))               # rear window
        parts["trim"].merge(box(-1.0, 0, H + 0.12, 1.3, 0.75, 0.12))                              # roof unit
        parts["trim"].merge(box(L / 2 + 0.02, 0, 0.55, 0.04, W / 2 - 0.03, 0.18))                 # bumpers
        parts["trim"].merge(box(-L / 2 - 0.02, 0, 0.55, 0.04, W / 2 - 0.03, 0.18))
        parts["trim"].merge(box(L * 0.36, -(W / 2 + 0.006), 1.35, 0.55, 0.008, 1.0))             # door
        t, r = _wheels(L, W, rw, 0.3, (-L * 0.30, L * 0.30))
        hl_z, tl_z = 0.85, 0.9
    else:
        rw = 0.36 if kind in ("suv", "van") else 0.32
        bottom = 0.30
        belt = {"suv": 1.05, "van": 1.0}.get(kind, 0.92)
        mid = bottom + 0.42 if kind != "van" else bottom + 0.45
        hz = (mid - bottom) / 2
        parts["body"].merge(tapered_box(0, 0, bottom + hz, (L / 2, W / 2), (L / 2 - 0.04, W / 2 - 0.015), hz))
        hz2 = (belt - mid) / 2                     # upper body: hood and boot slope down at the ends
        parts["body"].merge(tapered_box(0, 0, mid + hz2, (L / 2 - 0.04, W / 2 - 0.015),
                                        (L / 2 - (0.12 if kind == "van" else 0.3), W / 2 - 0.06), hz2))
        if kind == "van":
            # tall box at the back, sloped glass cab at the front
            parts["body"].merge(box(-L * 0.16, 0, belt + (H - belt) / 2, L * 0.34, W / 2 - 0.03, (H - belt) / 2))
            parts["glass"].merge(tapered_box(L * 0.27, 0, belt + (H - 0.05 - belt) / 2, (L * 0.19, W / 2 - 0.05),
                                             (L * 0.09, W / 2 - 0.08), (H - 0.05 - belt) / 2, shift=(-L * 0.1, 0)))
            parts["glass"].merge(box(-L * 0.12, 0, belt + 0.45, L * 0.22, W / 2 - 0.02, 0.22))   # side windows
            parts["body"].merge(box(L * 0.2, 0, H - 0.04, L * 0.04, W / 2 - 0.08, 0.04))
            roof_x, cab_front, cab_bottom = -L * 0.16, L * 0.46, belt
        else:
            cab = {"sedan": (-0.05, 0.26, 0.17, -0.02), "taxi": (-0.05, 0.26, 0.17, -0.02),
                   "hatch": (-0.07, 0.30, 0.21, -0.05), "suv": (-0.07, 0.31, 0.25, -0.03)}[kind]
            cx, b_half, t_half, t_shift = cab[0] * L, cab[1] * L, cab[2] * L, cab[3] * L
            ch = (H - 0.06 - belt) / 2
            parts["glass"].merge(tapered_box(cx, 0, belt + ch, (b_half, W / 2 - 0.08), (t_half, W / 2 - 0.16), ch,
                                             shift=(t_shift, 0)))
            parts["body"].merge(box(cx + t_shift, 0, H - 0.035, t_half + 0.03, W / 2 - 0.13, 0.035))        # roof
            # pillars: front (A), middle (B), back (C) on both sides
            for sy in (-1, 1):
                yb, yt = sy * (W / 2 - 0.075), sy * (W / 2 - 0.15)
                for xb, xt in ((cx + b_half, cx + t_shift + t_half), (cx - b_half, cx + t_shift - t_half),
                               (cx + 0.1 * b_half, cx + t_shift + 0.1 * t_half)):
                    parts["body"].merge(beam([xb, yb, belt], [xt, yt, H - 0.05], 0.035, 0.04))
                parts["body"].merge(box(cx + b_half - 0.05, sy * (W / 2 + 0.06), belt + 0.06, 0.05, 0.06, 0.045))  # mirror
            if kind == "suv":
                for sy in (-1, 1):
                    parts["trim"].merge(box(cx + t_shift, sy * (W / 2 - 0.2), H + 0.03, t_half, 0.02, 0.03))
            if kind == "taxi":
                parts["trim"].merge(box(cx + t_shift, 0, H + 0.07, 0.13, 0.3, 0.07))
        parts["trim"].merge(box(L / 2 - 0.05, 0, 0.42, 0.08, W / 2 - 0.02, 0.1))                    # bumpers
        parts["trim"].merge(box(-L / 2 + 0.05, 0, 0.42, 0.08, W / 2 - 0.02, 0.1))
        parts["trim"].merge(box(L / 2 + 0.005, 0, 0.66, 0.01, W * 0.2, 0.07))                       # grille
        for sy in (-1, 1):
            parts["trim"].merge(box(0, sy * (W / 2 + 0.005), 0.34, L * 0.3, 0.012, 0.04))         # skirts
        t, r = _wheels(L, W, rw, 0.22, (-L * 0.31, L * 0.31))
        hl_z, tl_z = belt - 0.2, belt - 0.15
    parts["tyres"], parts["rims"] = t, r
    for ax in ((-L * 0.30, L * 0.30) if kind == "bus" else (-L * 0.31, L * 0.31)):
        parts["trim"].merge(cylinder(ax, 0, rw, rw + 0.06, rw + 0.06, W + 0.014, seg=16, axis="y"))   # wheel arches
    for sy in (-1, 1):
        parts["headlights"].merge(box(L / 2 + 0.004, sy * W * 0.32, hl_z, 0.02, 0.17, 0.06))
        parts["taillights"].merge(box(-L / 2 - 0.004, sy * W * 0.33, tl_z, 0.02, 0.15, 0.07))
    return parts


def vehicle_models():
    out, finish = {}, {}
    for kind in VEHICLE_SIZES:
        for part, m in car_model(kind).items():
            name = f"car_{kind}_{part}"
            out[name] = m
            finish[name] = "bevel" if part in ("body", "glass", "trim") else "flat"
    return out, finish


# ======================================================================= street objects
STREET_PARTS = {            # model -> its parts (colour keys used by visuals.py)
    "tree_round": ["bark", "leaf"], "tree_tall": ["bark", "leaf"],
    "lamp": ["metal", "light"], "bench": ["wood", "metal"], "bin": ["body", "metal"],
    "hydrant": ["red"], "bus_stop": ["metal", "glass", "wood", "ad"], "fountain": ["stone", "water"],
    "planter": ["concrete", "leaf"], "traffic_light": ["pole", "housing", "red", "yellow", "green"],
    "ped_signal": ["pole", "box"],
    "sign_crossing": ["pole", "blue", "white"], "sign_speed": ["pole", "red", "white"],
    "sign_street": ["pole", "green", "white"],
}


def flat_triangle(p0, p1, p2, z0, z1):
    """A thin triangular plate lying flat (for road arrows)."""
    m = Mesh()
    for z in (z0, z1):
        m.v += [[p0[0], p0[1], z], [p1[0], p1[1], z], [p2[0], p2[1], z]]
    m.f = [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
    return _orient(m)


def standing_triangle(p0, p1, p2, x0, x1):
    """A thin triangular plate standing up, facing x (points are (y, z))."""
    m = Mesh()
    for x in (x0, x1):
        m.v += [[x, p0[0], p0[1]], [x, p1[0], p1[1]], [x, p2[0], p2[1]]]
    m.f = [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
    return _orient(m)


def road_arrow(turns, z0=0.025, z1=0.035):
    """White painted lane arrow pointing +x for the given turns
    ("straight", "left", "right"), about 4 m long."""
    m = box(-1.0, 0, (z0 + z1) / 2, 1.0, 0.09, (z1 - z0) / 2)                  # shaft
    if "straight" in turns:
        m.merge(box(0.6, 0, (z0 + z1) / 2, 0.6, 0.09, (z1 - z0) / 2))
        m.merge(flat_triangle((1.1, -0.35), (1.1, 0.35), (2.0, 0.0), z0, z1))
    for name, sgn in (("left", 1), ("right", -1)):
        if name in turns:
            m.merge(beam([-0.1, 0, (z0 + z1) / 2], [0.55, sgn * 0.55, (z0 + z1) / 2], 0.09, (z1 - z0) / 2))
            m.merge(flat_triangle((0.25, sgn * 0.85), (0.85, sgn * 0.25), (0.9, sgn * 0.95), z0, z1))
    return m


def street_models():
    s = {}
    # round tree: trunk, three branches, a lumpy crown
    trunk = loft([(0.0, 0.2, 0.2, 0.0), (0.4, 0.15, 0.15, 0.0), (2.3, 0.1, 0.1, 0.0)], seg=10)
    for a in (0.3, 2.4, 4.4):
        trunk.merge(beam([0, 0, 1.9], [0.7 * math.cos(a), 0.7 * math.sin(a), 2.7], 0.04, 0.04))
    s["tree_round_bark"] = trunk
    crown = Mesh()
    for k, (x, y, z, r) in enumerate(((0, 0, 3.0, 0.95), (0.55, 0.2, 2.8, 0.7), (-0.5, 0.3, 2.85, 0.7),
                                      (0.1, -0.55, 2.9, 0.7), (-0.1, 0.1, 3.6, 0.75), (0.3, -0.2, 3.45, 0.6))):
        crown.merge(jitter(ellipsoid(x, y, z, r, r, r * 0.85, seg=10, rings=7), 0.07, k))
    s["tree_round_leaf"] = crown
    # tall tree: straight trunk, stacked cones
    s["tree_tall_bark"] = loft([(0.0, 0.17, 0.17, 0.0), (1.6, 0.11, 0.11, 0.0)], seg=10)
    tall = Mesh()
    for z, r, h in ((1.4, 1.15, 1.6), (2.4, 0.9, 1.5), (3.3, 0.65, 1.4), (4.1, 0.4, 1.2)):
        tall.merge(cylinder(0, 0, z + h / 2, r, 0.05, h, seg=12))
    s["tree_tall_leaf"] = jitter(tall, 0.04, 9)
    # street lamp: base, tapered pole, arm reaching over the road (-x), lamp head
    s["lamp_metal"] = merged(cylinder(0, 0, 0.06, 0.14, 0.13, 0.12),
                             loft([(0.0, 0.07, 0.07, 0.0), (4.0, 0.05, 0.05, 0.0)], seg=10),
                             beam([0, 0, 3.92], [-0.78, 0, 4.1], 0.03, 0.03),
                             tapered_box(-0.85, 0, 4.06, (0.24, 0.13), (0.16, 0.08), 0.06))
    s["lamp_light"] = box(-0.85, 0, 3.985, 0.19, 0.1, 0.015)
    # bench 1.6 x 0.5, back rest at +y
    wood = Mesh()
    for y in (-0.17, -0.05, 0.07):
        wood.merge(box(0, y, 0.45, 0.8, 0.05, 0.02))
    for z in (0.62, 0.77):
        wood.merge(box(0, 0.21, z, 0.8, 0.02, 0.05))
    s["bench_wood"] = wood
    metal = Mesh()
    for x in (-0.68, 0.68):
        metal.merge(box(x, -0.18, 0.22, 0.03, 0.03, 0.22))
        metal.merge(box(x, 0.18, 0.22, 0.03, 0.03, 0.22))
        metal.merge(beam([x, 0.18, 0.43], [x, 0.23, 0.85], 0.03, 0.025))
        metal.merge(box(x, 0.0, 0.43, 0.03, 0.22, 0.02))
    s["bench_metal"] = metal
    # bin
    s["bin_body"] = loft([(0.0, 0.22, 0.22, 0.0), (0.85, 0.25, 0.25, 0.0)], seg=14)
    s["bin_metal"] = merged(cylinder(0, 0, 0.88, 0.26, 0.26, 0.06, seg=14),
                            ellipsoid(0, 0, 0.9, 0.24, 0.24, 0.08, seg=12, rings=6))
    # fire hydrant
    s["hydrant_red"] = merged(cylinder(0, 0, 0.03, 0.16, 0.16, 0.06, seg=12),
                              loft([(0.05, 0.12, 0.12, 0.0), (0.55, 0.11, 0.11, 0.0)], seg=12),
                              ellipsoid(0, 0, 0.56, 0.115, 0.115, 0.09, seg=12, rings=6),
                              cylinder(0, 0, 0.67, 0.04, 0.03, 0.07, seg=8),
                              cylinder(0, 0, 0.4, 0.045, 0.045, 0.34, seg=8, axis="y"),
                              cylinder(0.11, 0, 0.36, 0.05, 0.05, 0.1, seg=8, axis="x"))
    # bus shelter 3.2 x 1.2 x 2.4, back panel at +y
    s["bus_stop_metal"] = merged(box(0, 0, 2.4, 1.65, 0.65, 0.04),
                                 *[box(x, y, 1.2, 0.04, 0.04, 1.2) for x in (-1.55, 1.55) for y in (-0.5, 0.52)])
    s["bus_stop_glass"] = merged(box(0, 0.55, 1.3, 1.5, 0.012, 0.95), box(1.55, 0.05, 1.3, 0.012, 0.42, 0.9))
    s["bus_stop_wood"] = merged(box(0, 0.32, 0.45, 1.2, 0.15, 0.025),
                                box(-1.0, 0.32, 0.22, 0.03, 0.12, 0.22), box(1.0, 0.32, 0.22, 0.03, 0.12, 0.22))
    s["bus_stop_ad"] = box(-1.55, 0.05, 1.3, 0.04, 0.42, 0.9)
    # fountain (radius 2.2)
    s["fountain_stone"] = merged(cylinder(0, 0, 0.22, 2.2, 2.15, 0.44, seg=28),
                                 loft([(0.4, 0.32, 0.32, 0.0), (1.2, 0.2, 0.2, 0.0)], seg=12),
                                 ellipsoid(0, 0, 1.3, 0.85, 0.85, 0.18, seg=16, rings=6))
    s["fountain_water"] = merged(cylinder(0, 0, 0.455, 2.0, 2.0, 0.03, seg=28),
                                 ellipsoid(0, 0, 1.47, 0.7, 0.7, 0.05, seg=16, rings=4))
    # planter 1.2 x 0.7 with shrubs
    s["planter_concrete"] = box(0, 0, 0.25, 0.6, 0.35, 0.25)
    shrubs = Mesh()
    for k, x in enumerate((-0.35, 0.0, 0.35)):
        shrubs.merge(jitter(ellipsoid(x, 0, 0.6, 0.24, 0.24, 0.2, seg=8, rings=6), 0.03, 20 + k))
    s["planter_leaf"] = shrubs
    # traffic light facing +x, pole 3.2 m: lamps at 3.7 / 3.4 / 3.1 m (as the simple version)
    s["traffic_light_pole"] = merged(cylinder(0, 0, 0.06, 0.14, 0.12, 0.12),
                                     loft([(0.0, 0.07, 0.07, 0.0), (3.2, 0.06, 0.06, 0.0)], seg=10))
    housing = box(0.0, 0, 3.4, 0.11, 0.17, 0.47)
    housing.merge(box(-0.13, 0, 3.4, 0.02, 0.24, 0.55))                                   # backboard
    for z in (3.7, 3.4, 3.1):
        housing.merge(tapered_box(0.17, 0, z + 0.1, (0.06, 0.12), (0.06, 0.12), 0.012))    # visors
    s["traffic_light_housing"] = housing
    for name, z in (("red", 3.7), ("yellow", 3.4), ("green", 3.1)):
        s[f"traffic_light_{name}"] = cylinder(0.115, 0, z, 0.095, 0.095, 0.03, seg=14, axis="x")
    s["ped_signal_pole"] = loft([(0.0, 0.05, 0.05, 0.0), (2.4, 0.045, 0.045, 0.0)], seg=8)
    s["ped_signal_box"] = box(0.05, 0, 2.45, 0.08, 0.14, 0.18)
    # signs, all facing +x (towards the traffic they are for)
    pole = loft([(0.0, 0.035, 0.035, 0.0), (2.75, 0.03, 0.03, 0.0)], seg=8)
    s["sign_crossing_pole"] = pole.copy()
    s["sign_crossing_blue"] = box(0.05, 0, 2.45, 0.012, 0.3, 0.3)
    s["sign_crossing_white"] = standing_triangle((-0.22, 2.25), (0.22, 2.25), (0.0, 2.66), 0.062, 0.07)
    s["sign_speed_pole"] = pole.copy()
    s["sign_speed_red"] = cylinder(0.05, 0, 2.42, 0.31, 0.31, 0.024, seg=20, axis="x")
    s["sign_speed_white"] = merged(cylinder(0.066, 0, 2.42, 0.24, 0.24, 0.012, seg=20, axis="x"),
                                   box(0.073, -0.06, 2.42, 0.004, 0.035, 0.11),       # a stylised "30"
                                   box(0.073, 0.07, 2.42, 0.004, 0.06, 0.11))
    s["sign_street_pole"] = loft([(0.0, 0.04, 0.04, 0.0), (2.9, 0.035, 0.035, 0.0)], seg=8)
    s["sign_street_green"] = merged(box(0, 0.45, 2.78, 0.012, 0.45, 0.11), box(0.45, 0, 2.55, 0.45, 0.012, 0.11))
    s["sign_street_white"] = merged(box(0, 0.45, 2.78, 0.016, 0.38, 0.035), box(0.45, 0, 2.55, 0.38, 0.016, 0.035))
    finish = {k: "flat" for k in s}
    for k in ("tree_round_leaf", "tree_tall_leaf", "planter_leaf", "tree_round_bark", "tree_tall_bark",
              "bin_body", "hydrant_red", "fountain_water"):
        finish[k] = "smooth"
    for k in ("bench_wood", "planter_concrete", "bus_stop_metal", "lamp_metal"):
        finish[k] = "bevel"
    return s, finish


def all_models():
    """Every model part: {name: Mesh} and {name: finish} ("smooth", "bevel", "flat")."""
    meshes, finish = {}, {}
    for fn in (person_models, vehicle_models, street_models):
        m, f = fn()
        meshes.update(m)
        finish.update(f)
    return meshes, finish


# ======================================================================= buildings (made at run time)
BUILDING_PARTS = ["wall", "trim", "glass", "frame", "awning", "sign", "door", "roof", "rail"]


def _prism(hl, hd, z0, rise):
    """Triangular gable block along x: base 2*hl x 2*hd at height z0, ridge rise above it."""
    m = Mesh()
    for x in (-hl, hl):
        m.v += [[x, -hd, z0], [x, hd, z0], [x, 0.0, z0 + rise]]
    m.f = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
    return _orient(m)


def _roof_slabs(hl, hd, z0, rise, overhang=0.45, thick=0.08):
    """Two sloping roof plates meeting at the ridge (ridge along x)."""
    roof = Mesh()
    L = math.hypot(hd, rise)
    for s in (-1, 1):
        dy, dz = s * hd / L, -rise / L                     # down the slope, towards the eave
        ny, nz = s * rise / L, hd / L                      # outwards from the roof
        cy = s * hd / 2 + dy * overhang / 2 + ny * thick / 2
        cz = z0 + rise / 2 + dz * overhang / 2 + nz * thick / 2
        slab = box(0, 0, 0, hl + overhang * 0.7, (L + overhang) / 2, thick / 2)
        slab.rotate("x", math.atan2(dz, dy) if s > 0 else math.atan2(-dz, -dy))
        roof.merge(slab.move(0, cy, cz))
    return roof


def _wall_frame(face, hx, hy):
    """For a wall facing `face`: (nx, ny) normal, (tx, ty) direction along it,
    half length along it, distance from the centre."""
    nx, ny = face
    along = hy if nx != 0 else hx
    depth = hx if nx != 0 else hy
    return nx, ny, -ny, nx, along, depth


def _panel(cx, cy, z, nx, half_w, half_h, thick):
    """A thin box lying flat on a wall with normal along x (nx != 0) or y."""
    return box(cx, cy, z, thick, half_w, half_h) if nx != 0 else box(cx, cy, z, half_w, thick, half_h)


def _windows(m, rnd, walls, hx, hy, h, first, floor_h, win_w, win_h, skip, balconies=False):
    for k, face in enumerate(walls):
        nx, ny, tx, ty, along, depth = _wall_frame(face, hx, hy)
        n_win = max(1, int((2 * along - 1.0) / 2.6))
        spacing = (2 * along - 1.0) / n_win
        z, floor = first, 0
        while z + 2 * win_h + 0.3 < h:
            zc = z + win_h + 0.2
            for i in range(n_win):
                t = -along + 0.5 + spacing * (i + 0.5)
                if k == 0 and skip(t, floor):
                    continue
                cx, cy = nx * depth + tx * t, ny * depth + ty * t
                m["glass"].merge(_panel(cx + nx * 0.03, cy + ny * 0.03, zc, nx, win_w, win_h, 0.03))
                m["frame"].merge(_panel(cx + nx * 0.02, cy + ny * 0.02, zc, nx, win_w + 0.07, win_h + 0.07, 0.015))
                m["trim"].merge(_panel(cx + nx * 0.06, cy + ny * 0.06, zc - win_h - 0.09, nx, win_w + 0.1, 0.03, 0.06))
                if balconies and k == 0 and floor >= 1 and i % 2 == 0:
                    # a balcony: floor slab, glass/metal railing at the front and sides
                    bw = min(spacing * 0.48, win_w + 0.6)
                    out = 0.9
                    bx, by = cx + nx * out / 2, cy + ny * out / 2
                    zb = z + 0.05
                    m["trim"].merge(box(bx, by, zb, out / 2, bw, 0.07) if nx != 0 else box(bx, by, zb, bw, out / 2, 0.07))
                    rx, ry = cx + nx * (out - 0.03), cy + ny * (out - 0.03)
                    m["rail"].merge(_panel(rx, ry, zb + 0.55, nx, bw, 0.48, 0.02))
                    for sgn in (-1, 1):
                        sx, sy = bx + tx * sgn * bw, by + ty * sgn * bw
                        m["rail"].merge(box(sx, sy, zb + 0.55, out / 2, 0.02, 0.48) if nx != 0 else
                                        box(sx, sy, zb + 0.55, 0.02, out / 2, 0.48))
            if k == 0 and z > first:                      # a ledge between floors on the front
                lx, ly = (0.05, along) if nx != 0 else (along, 0.05)
                m["trim"].merge(box(nx * (depth + 0.05), ny * (depth + 0.05), z - 0.05, lx, ly, 0.05))
            z += floor_h
            floor += 1


def building_meshes(hx, hy, h, face, shop, seed, kind="block", balconies=False, garage=False):
    """Facade detail for one building of half size hx, hy and height h (to the
    eaves), front facing `face`. kind "block" = flat-roofed town building
    (shop front, balconies optional), "house" = detached house with a gable
    roof, chimney, porch and maybe a garage. Returns {material: Mesh} in
    building-centred coordinates."""
    rnd = random.Random(seed)
    m = {k: Mesh() for k in BUILDING_PARTS}
    fx, fy = face
    walls = [face, (fy, -fx), (-fy, fx)]                    # the front and the two sides
    m["wall"].merge(box(0, 0, h / 2, hx, hy, h / 2))
    m["trim"].merge(box(0, 0, 0.25, hx + 0.04, hy + 0.04, 0.25))                   # plinth
    nx, ny, tx, ty, along, depth = _wall_frame(face, hx, hy)

    if kind == "house":
        # gable roof with the ridge running along the front
        ridge_along_x = fy != 0
        hl, hd = (hx, hy) if ridge_along_x else (hy, hx)
        rise = hd * math.tan(math.radians(rnd.uniform(30, 40)))
        gable = _prism(hl, hd, h, rise)
        roof = _roof_slabs(hl, hd, h, rise)
        chimney = box(hl * rnd.uniform(-0.6, 0.6), hd * 0.35 * rnd.choice((-1, 1)), h + rise * 0.75, 0.28, 0.28, rise * 0.55)
        if not ridge_along_x:
            for part in (gable, roof, chimney):
                part.rotate("z", math.pi / 2)
        m["wall"].merge(gable)
        m["roof"].merge(roof)
        m["trim"].merge(chimney)
        door_t = along * rnd.choice((-0.45, 0.45))
        garage_t = -door_t if garage else None
        win_w, win_h = rnd.choice([(0.45, 0.6), (0.55, 0.65), (0.5, 0.7)])
        _windows(m, rnd, walls, hx, hy, h, 0.9, 2.8, win_w, win_h,
                 skip=lambda t, floor: floor == 0 and (abs(t - door_t) < 1.1 or
                                                       (garage_t is not None and abs(t - garage_t) < 1.8)))
        dcx, dcy = nx * depth + tx * door_t, ny * depth + ty * door_t
        m["door"].merge(_panel(dcx + nx * 0.04, dcy + ny * 0.04, 1.05, nx, 0.48, 1.05, 0.04))
        m["frame"].merge(_panel(dcx + nx * 0.02, dcy + ny * 0.02, 1.12, nx, 0.58, 1.14, 0.02))
        pc_x, pc_y = dcx + nx * 0.55, dcy + ny * 0.55                                 # porch roof and step
        m["roof"].merge(box(pc_x, pc_y, 2.45, 0.55, 0.85, 0.05) if nx != 0 else box(pc_x, pc_y, 2.45, 0.85, 0.55, 0.05))
        st_x, st_y = dcx + nx * 0.4, dcy + ny * 0.4
        m["trim"].merge(box(st_x, st_y, 0.08, 0.4, 0.7, 0.08) if nx != 0 else box(st_x, st_y, 0.08, 0.7, 0.4, 0.08))
        if garage_t is not None:
            gx, gy = nx * depth + tx * garage_t, ny * depth + ty * garage_t
            m["frame"].merge(_panel(gx + nx * 0.04, gy + ny * 0.04, 1.1, nx, 1.25, 1.1, 0.04))
        return {k: v for k, v in m.items() if v.v}

    # ---- flat-roofed town building
    m["trim"].merge(box(0, 0, h + 0.1, hx + 0.12, hy + 0.12, 0.1))                 # cornice
    if h > 5 and rnd.random() < 0.7:
        m["trim"].merge(box(hx * rnd.uniform(-0.4, 0.4), hy * rnd.uniform(-0.4, 0.4), h + 0.55,
                            rnd.uniform(0.6, 1.2), rnd.uniform(0.5, 0.9), 0.35))  # rooftop unit
    if h > 8 and rnd.random() < 0.3:                                               # water tank
        m["roof"].merge(cylinder(hx * 0.3, -hy * 0.3, h + 1.0, 0.9, 0.9, 1.6, seg=12))
    for sgn in (-1, 1):                                                            # corner pilasters
        px, py = nx * (depth + 0.06) + tx * sgn * (along - 0.15), ny * (depth + 0.06) + ty * sgn * (along - 0.15)
        m["trim"].merge(box(px, py, h / 2, 0.15 if nx == 0 else 0.08, 0.15 if nx != 0 else 0.08, h / 2))
    first = 3.4 if shop else 1.2
    win_w, win_h = rnd.choice([(0.5, 0.7), (0.6, 0.75), (0.45, 0.8), (0.7, 0.65), (0.4, 0.85)])
    _windows(m, rnd, walls, hx, hy, h, first, 3.0, win_w, win_h,
             skip=lambda t, floor: (not shop) and floor == 0 and abs(t - along * 0.4) < 1.0,
             balconies=balconies)
    if shop:
        m["glass"].merge(_panel(nx * (depth + 0.03), ny * (depth + 0.03), 1.45, nx, along - 0.7, 1.05, 0.02))
        m["frame"].merge(_panel(nx * (depth + 0.015), ny * (depth + 0.015), 1.45, nx, along - 0.6, 1.15, 0.015))
        out = 0.9
        aw = tapered_box(0, 0, 0, (0.45, along - 0.4) if nx != 0 else (along - 0.4, 0.45),
                         (0.45, along - 0.4) if nx != 0 else (along - 0.4, 0.45), 0.04)
        for p in aw.v:                                      # slope the awning down away from the wall
            d = p[0] * nx + p[1] * ny
            p[2] += -0.25 * (d / 0.45)
        m["awning"].merge(aw.move(nx * (depth + out / 2), ny * (depth + out / 2), 2.85))
        m["sign"].merge(_panel(nx * (depth + 0.06), ny * (depth + 0.06), 3.15, nx, along - 0.8, 0.18, 0.05))
        dcx, dcy = tx * along * 0.55, ty * along * 0.55
        m["door"].merge(_panel(nx * (depth + 0.035) + dcx, ny * (depth + 0.035) + dcy, 1.1, nx, 0.5, 1.1, 0.035))
    else:
        dcx, dcy = tx * along * 0.4, ty * along * 0.4
        m["door"].merge(_panel(nx * (depth + 0.04) + dcx, ny * (depth + 0.04) + dcy, 1.15, nx, 0.6, 1.15, 0.04))
        m["frame"].merge(_panel(nx * (depth + 0.02) + dcx, ny * (depth + 0.02) + dcy, 1.25, nx, 0.72, 1.27, 0.02))
        m["trim"].merge(_panel(nx * (depth + 0.45) + dcx, ny * (depth + 0.45) + dcy, 2.55, nx, 0.9, 0.05, 0.45))
    return {k: v for k, v in m.items() if v.v}


# ======================================================================= .obj files
def write_obj(path, mesh, smooth=False, name=None):
    """Write a mesh as a Wavefront .obj with normals (triangles only)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tris = list(mesh.triangles())
    lines = [f"o {name or os.path.splitext(os.path.basename(path))[0]}"]
    if smooth:
        acc = [[0.0, 0.0, 0.0] for _ in mesh.v]
        for a, b, c in tris:
            n = _cross(_sub(mesh.v[b], mesh.v[a]), _sub(mesh.v[c], mesh.v[a]))
            for i in (a, b, c):
                acc[i][0] += n[0]; acc[i][1] += n[1]; acc[i][2] += n[2]
        lines += [f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in mesh.v]
        lines += ["vn {:.4f} {:.4f} {:.4f}".format(*_norm(n)) for n in acc]
        lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in tris]
    else:
        vi = 0
        for a, b, c in tris:
            n = _norm(_cross(_sub(mesh.v[b], mesh.v[a]), _sub(mesh.v[c], mesh.v[a])))
            for i in (a, b, c):
                p = mesh.v[i]
                lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
            lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
            k = len(lines)
            vi += 3
            nn = vi // 3
            lines.append(f"f {vi - 2}//{nn} {vi - 1}//{nn} {vi}//{nn}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
