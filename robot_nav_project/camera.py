"""camera.py - a forward-facing camera on the robot or on any car (Phase 2, YOLO).

Each capture gives three images of what the camera sees:
  rgb   - the colour picture YOLO will read
  depth - distance of every pixel in metres (later: YOLO box -> floor position)
  seg   - which object every pixel belongs to (gives free, exact labels)

For a varied training set, capture() can also change the lighting (time of
day), jitter the camera a little, and augment() can add noise/blur and
brightness changes - so YOLO does not learn one exact look.

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

    def capture(self, pose, lighting=None, jitter=None, rng=None):
        """Render what a camera on `pose` sees. `pose` is the robot, a traffic
        car, or anything with x, y, heading and front (centre to front bumper).
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
        paint_sky(rgb, seg, lighting["sky"])
        info = {"fov": fov, "height": height, "pitch_deg": math.degrees(pitch)}
        return rgb, self.depth_to_metres(depth), seg, info

    @staticmethod
    def depth_to_metres(depth_buffer):
        """PyBullet's depth buffer is 0..1 and non-linear; convert to metres."""
        near, far = config.CAMERA_NEAR, config.CAMERA_FAR
        return far * near / (far - (far - near) * depth_buffer)


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


def person_boxes(seg, pedestrians, min_pixels=None):
    """Box around the visible part of every pedestrian, read from the
    segmentation image. Returns a list of (class 0, x1, y1, x2, y2, ped_id)."""
    min_pixels = config.CAMERA_MIN_PIXELS if min_pixels is None else min_pixels
    ids = _object_ids(seg)
    boxes = []
    for ped in pedestrians:
        mask = np.isin(ids, ped.label_ids)
        if mask.sum() < min_pixels:
            continue                                  # not in view, or (almost) fully hidden
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        x1, x2, y1, y2 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        if x2 - x1 >= config.CAMERA_MIN_BOX and y2 - y1 >= config.CAMERA_MIN_BOX:
            boxes.append((0, int(x1), int(y1), int(x2), int(y2), ped.ped_id))
    return boxes


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
