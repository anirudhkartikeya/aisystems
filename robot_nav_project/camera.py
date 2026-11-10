"""camera.py - a forward-facing camera on the robot or on any car (Phase 2, YOLO).

Each capture gives three images of what the camera sees:
  rgb   - the colour picture YOLO will read
  depth - distance of every pixel in metres (later: YOLO box -> floor position)
  seg   - which object every pixel belongs to (gives free, exact labels)

For a varied training set, capture() can also change the lighting (time of
day), jitter the camera a little, and augment() can add noise/blur and
brightness changes - so YOLO does not learn one exact look.

Night ("night" lighting): the scene is rendered flat (each surface's own colour)
and then lit by night_scene(): weak moonlight, the car's own headlights (bright
close ahead, fading with distance) and the street lamps (pools of light around
each lamp). Far-away people between lamps stay dark - the hard case for YOLO.

save_sample() writes the picture plus a YOLO label file in the standard
folder layout:  dataset/images/<split>/name.jpg  +  dataset/labels/<split>/name.txt
"""
import math
import os

import numpy as np
import pybullet as p

import config


class RobotCamera:
    def __init__(self, gui=False):
        # The GUI build can render on the graphics card (fast); headless uses
        # PyBullet's built-in software renderer (which also does the lighting below).
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT_PX

    def capture(self, pose, lighting=None, jitter=None, rng=None, lamps=None, moon=None):
        """Render what a camera on `pose` sees. `pose` is the robot, a traffic
        car, or anything with x, y, heading and front (centre to front bumper).
        Night lighting: lamps = street-lamp head positions (N, 3), moon = moonlight level.
        Returns (rgb, depth_m, seg, info)."""
        lighting = config.LIGHTING_PRESETS[lighting or config.LIGHTING]
        height, pitch, yaw_off, fov = config.CAMERA_MOUNT_HEIGHT, 0.0, 0.0, config.CAMERA_FOV
        if jitter and rng is not None:
            height += rng.uniform(-1, 1) * jitter["height"]
            pitch = math.radians(rng.uniform(-1, 1) * jitter["pitch_deg"] - 1.0)
            yaw_off = math.radians(rng.uniform(-1, 1) * jitter["yaw_deg"])
            fov = rng.uniform(*jitter["fov_deg"])
        h = pose.heading + yaw_off
        fx, fy = math.cos(h), math.sin(h)
        front = getattr(pose, "front", None)
        if front is None:
            front = getattr(pose, "length", 4.0) / 2
        # camera sits at the front of the vehicle, looking ahead (slightly up or down)
        eye = [pose.x + fx * (front + 0.05), pose.y + fy * (front + 0.05), height]
        target = [eye[0] + fx * 10 * math.cos(pitch), eye[1] + fy * 10 * math.cos(pitch),
                  height + 10 * math.sin(pitch)]
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=fov, aspect=self.width / self.height,
                                            nearVal=config.CAMERA_NEAR, farVal=config.CAMERA_FAR)
        kwargs = {"renderer": self.renderer}
        if self.renderer == p.ER_TINY_RENDERER:
            kwargs.update(lightDirection=list(lighting["dir"]), lightColor=list(lighting["color"]),
                          lightAmbientCoeff=lighting["ambient"], lightDiffuseCoeff=lighting["diffuse"],
                          lightSpecularCoeff=lighting["specular"], shadow=lighting["shadow"])
        _, _, rgb, depth, seg = p.getCameraImage(self.width, self.height, view, proj, **kwargs)
        rgb = np.reshape(np.asarray(rgb, dtype=np.uint8), (self.height, self.width, 4))[:, :, :3].copy()
        depth = np.reshape(np.asarray(depth, dtype=np.float32), (self.height, self.width))
        seg = np.reshape(np.asarray(seg, dtype=np.int64), (self.height, self.width))
        depth_m = self.depth_to_metres(depth)
        if lighting.get("night"):
            if moon is None:
                moon = float(np.mean(config.NIGHT["moon"]))
            rgb = night_scene(rgb, depth_m, seg, eye, target, fov, self.width / self.height, (fx, fy),
                              lamps, moon, rng)
        paint_sky(rgb, seg, lighting["sky"])
        info = {"fov": fov, "height": height, "pitch_deg": math.degrees(pitch)}
        return rgb, depth_m, seg, info

    @staticmethod
    def depth_to_metres(depth_buffer):
        """PyBullet's depth buffer is 0..1 and non-linear; convert to metres."""
        near, far = config.CAMERA_NEAR, config.CAMERA_FAR
        return far * near / (far - (far - near) * depth_buffer)


def pixel_points(depth_m, eye, target, fov_deg, aspect):
    """World position (H, W, 3) of the surface seen in every pixel, from the depth image."""
    H, W = depth_m.shape
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = math.tan(math.radians(fov_deg) / 2)
    u = ((np.arange(W) + 0.5) / W * 2 - 1) * t * aspect
    v = (1 - (np.arange(H) + 0.5) / H * 2) * t
    rays = fwd[None, None] + u[None, :, None] * right[None, None] + v[:, None, None] * up[None, None]
    return eye[None, None] + depth_m[..., None] * rays           # depth is measured along the view axis


def night_scene(albedo, depth_m, seg, eye, target, fov_deg, aspect, heading_xy, lamps, moon, rng=None, s=None):
    """Light a flat-rendered picture as at night: moonlight + headlights + street lamps.
    The lit share of each surface = moon + headlight/d^2 (inside the beam) + sum of lamp/d^2,
    with warm light colour, plus sensor noise. Sky pixels are left for paint_sky()."""
    s = s or config.NIGHT
    P = pixel_points(depth_m, eye, target, fov_deg, aspect)
    light = np.full(depth_m.shape, float(moon))
    # the car's own headlights: just below the camera, aimed ahead and slightly down
    hx, hy = heading_xy
    beam = np.array([hx, hy, -0.08])
    beam /= np.linalg.norm(beam)
    D = P - (np.asarray(eye, float) + [0.0, 0.0, -0.1])
    dist = np.maximum(np.linalg.norm(D, axis=2), 1.0)
    ang = np.degrees(np.arccos(np.clip((D @ beam) / dist, -1.0, 1.0)))
    light += s["headlight"] * np.exp(-(ang / s["headlight_deg"]) ** 2) / dist ** 2
    # street lamps near the camera
    if lamps is not None and len(lamps):
        L = np.asarray(lamps, float).reshape(-1, 3)
        near = np.hypot(L[:, 0] - eye[0], L[:, 1] - eye[1]) < s["lamp_range"]
        for lx, ly, lz in L[near]:
            d2 = (P[..., 0] - lx) ** 2 + (P[..., 1] - ly) ** 2 + (P[..., 2] - lz) ** 2
            light += s["lamp"] / np.maximum(d2, 0.05)
    light = np.clip(light, 0.0, 1.3)
    out = albedo.astype(np.float32) * light[..., None] * np.asarray(s["tint"], np.float32)[None, None]
    if s["noise"] > 0:
        r = rng if rng is not None else np.random.default_rng(0)
        out += r.normal(0.0, s["noise"], out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[seg < 0] = albedo[seg < 0]                                # sky: painted afterwards
    return out


def lamp_heads(roadmap):
    """Where the street lamps' lights are (N, 3), from the town map (same as visuals.py draws them)."""
    heads = []
    for ob in (roadmap.obstacles if roadmap is not None else []):
        if ob.get("look") == "lamp":
            (cx, cy), yaw = ob["center"], ob.get("yaw", 0.0)
            heads.append((cx - 0.6 * math.cos(yaw), cy - 0.6 * math.sin(yaw), ob.get("z", 0.0) + 3.92))
    return np.array(heads, dtype=float).reshape(-1, 3)


def paint_sky(rgb, seg, colors):
    """Pixels that hit nothing are sky: fill them with a top-to-bottom gradient."""
    sky = seg < 0
    if not sky.any():
        return
    top, bottom = np.array(colors[0]) * 255, np.array(colors[1]) * 255
    t = np.linspace(0.0, 1.0, rgb.shape[0])[:, None]
    grad = (top[None, :] * (1 - t) + bottom[None, :] * t).astype(np.uint8)       # (H, 3)
    rows = np.broadcast_to(grad[:, None, :], rgb.shape)
    rgb[sky] = rows[sky]


def _object_ids(seg):
    return np.where(seg >= 0, seg & ((1 << 24) - 1), -1)     # strip link bits if present


def person_boxes(seg, pedestrians, min_pixels=None, rgb=None, dark=None):
    """Box around the visible part of every pedestrian, read from the
    segmentation image. Returns a list of (class 0, x1, y1, x2, y2, ped_id).
    With rgb (night / shadow pictures), people a camera could not see in the dark are
    left out (config.CAMERA_MIN_VISIBLE): their ids are added to the list `dark`."""
    min_pixels = config.CAMERA_MIN_PIXELS if min_pixels is None else min_pixels
    ids = _object_ids(seg)
    boxes, masks = [], []
    for ped in pedestrians:
        mask = np.isin(ids, ped.label_ids)
        if mask.sum() < min_pixels:
            continue                                  # not in view, or (almost) fully hidden
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        x1, x2, y1, y2 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        if x2 - x1 >= config.CAMERA_MIN_BOX and y2 - y1 >= config.CAMERA_MIN_BOX:
            boxes.append((0, int(x1), int(y1), int(x2), int(y2), ped.ped_id))
            masks.append(mask)
    if rgb is None or not boxes:
        return boxes
    lum = rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    anyone = np.any(masks, axis=0)
    kept = []
    for box, mask in zip(boxes, masks):
        _, x1, y1, x2, y2, pid = box
        w, h = x2 - x1, y2 - y1
        ring = np.zeros_like(mask)                    # just around the person, not other people
        ring[max(0, y1 - h // 4):y2 + h // 4, max(0, x1 - w // 2):x2 + w // 2] = True
        ring &= ~anyone
        me = float(lum[mask].mean())
        behind = float(lum[ring].mean()) if ring.any() else me
        v = config.CAMERA_MIN_VISIBLE
        if me >= v["brightness"] or abs(me - behind) >= v["contrast"]:
            kept.append(box)
        elif dark is not None:
            dark.append(pid)
    return kept


def car_boxes(seg, body_id_lists, min_pixels=None):
    """Same for cars (class 1): body_id_lists is one list of body ids per car."""
    min_pixels = config.CAMERA_MIN_PIXELS if min_pixels is None else min_pixels
    ids = _object_ids(seg)
    boxes = []
    for k, bodies in enumerate(body_id_lists):
        mask = np.isin(ids, bodies)
        if mask.sum() < min_pixels * 2:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        boxes.append((1, int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1), k))
    return boxes


def augment(rgb, rng, settings=None):
    """Camera-like variation: brightness/contrast change, sensor noise,
    sometimes a little motion blur."""
    s = settings or config.IMAGE_AUGMENT
    img = rgb.astype(np.float32)
    gain = 1.0 + rng.uniform(-1, 1) * s["contrast"]
    bias = rng.uniform(-1, 1) * s["brightness"] * 255
    img = (img - 128.0) * gain + 128.0 + bias
    if s["noise"] > 0:
        img += rng.normal(0.0, s["noise"], img.shape)
    if rng.random() < s["blur_prob"]:
        k = int(rng.integers(2, 5))                                 # horizontal motion blur
        img = sum(np.roll(img, i, axis=1) for i in range(k)) / k
    return np.clip(img, 0, 255).astype(np.uint8)


def save_sample(rgb, boxes, name, folder=None, split=None, ext="jpg"):
    """Write images/[split/]<name>.<ext> and labels/[split/]<name>.txt in YOLO
    format: one line per object: class x_center y_center width height (0..1)."""
    from PIL import Image

    folder = folder or config.DATASET_DIR
    img_dir = os.path.join(folder, "images", split) if split else os.path.join(folder, "images")
    lbl_dir = os.path.join(folder, "labels", split) if split else os.path.join(folder, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    h, w = rgb.shape[:2]
    path = os.path.join(img_dir, f"{name}.{ext}")
    if ext == "jpg":
        Image.fromarray(rgb).save(path, quality=92)
    else:
        Image.fromarray(rgb).save(path)
    with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
        for cls, x1, y1, x2, y2, *_ in boxes:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return path


def write_data_yaml(folder, with_cars=False):
    """The small file YOLO training needs: where the images are, class names."""
    names = ["person", "car"] if with_cars else ["person"]
    with open(os.path.join(folder, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(folder)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
