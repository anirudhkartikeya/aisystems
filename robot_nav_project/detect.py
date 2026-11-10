"""detect.py - run the person detector on one picture, optionally with slicing (SAHI-style).

    from detect import load_model, detect
    model = load_model("runs/yolo11n_sim_960/weights/best.pt")
    boxes, conf = detect(model, "picture.jpg")                 # normal: the whole picture at once
    boxes, conf = detect(model, "picture.jpg", slice=True)     # + overlapping tiles, for tiny far people

boxes: (N, 4) x1 y1 x2 y2 as a share of the picture's width / height (0..1); conf: (N,).

Slicing (off by default, config.DETECT_SLICE): the picture is cut into overlapping tiles,
each tile is enlarged to the model's input size (so a far-away person becomes bigger),
YOLO runs on every tile and on the whole picture, and the boxes are merged:
  - a box that touches an inner tile border is dropped (that person is cut in two;
    the whole picture or the next tile has them complete),
  - overlapping boxes of the same person are merged by non-maximum suppression.
It costs about (tiles + 1) times the time, so it is a switch to use only when needed.
"""
import numpy as np
from PIL import Image

import config


def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")


def nms(boxes, scores, iou=0.5):
    """Keep the most confident box of every group of boxes that overlap more than `iou`."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / (area[i] + area[order[1:]] - inter + 1e-12)
        order = order[1:][overlap <= iou]
    return np.array(keep, dtype=int)


def tiles(width, height, tile, overlap):
    """Top-left corners of overlapping square tiles covering the picture."""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, width - tile) + 1, step))
    ys = list(range(0, max(1, height - tile) + 1, step))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y) for y in ys for x in xs]


def _boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return result.boxes.xyxy.cpu().numpy().astype(float), result.boxes.conf.cpu().numpy().astype(float)


def detect(model, image, conf=0.25, imgsz=960, slice=None, tile=None, overlap=None, tile_imgsz=None,
           iou=0.5, device=None):
    """Person boxes for one picture (path or RGB array). See the module text for slicing."""
    s = config.DETECT_SLICE_SETTINGS
    slice = config.DETECT_SLICE if slice is None else slice
    tile = tile or s["tile"]
    overlap = s["overlap"] if overlap is None else overlap
    tile_imgsz = tile_imgsz or s["tile_imgsz"]
    device = device or _device()
    rgb = np.asarray(Image.open(image).convert("RGB")) if isinstance(image, str) else np.asarray(image)
    H, W = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[..., ::-1])                    # ultralytics reads arrays as BGR
    full = model.predict(bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    boxes, scores = _boxes(full)
    if slice:
        tile = min(tile, W, H)
        corners = tiles(W, H, tile, overlap)
        crops = [np.ascontiguousarray(bgr[y:y + tile, x:x + tile]) for x, y in corners]
        results = model.predict(crops, conf=conf, imgsz=tile_imgsz, device=device, verbose=False)
        more_b, more_s = [boxes], [scores]
        edge = 2.0                                                 # px from a tile border = touching it
        for (x0, y0), res in zip(corners, results):
            b, c = _boxes(res)
            if not len(b):
                continue
            cut = np.zeros(len(b), dtype=bool)                    # touches a border INSIDE the picture
            if x0 > 0:
                cut |= b[:, 0] < edge
            if y0 > 0:
                cut |= b[:, 1] < edge
            if x0 + tile < W:
                cut |= b[:, 2] > tile - edge
            if y0 + tile < H:
                cut |= b[:, 3] > tile - edge
            b, c = b[~cut], c[~cut]
            more_b.append(b + [x0, y0, x0, y0])
            more_s.append(c)
        boxes, scores = np.concatenate(more_b), np.concatenate(more_s)
        keep = nms(boxes, scores, iou)
        boxes, scores = boxes[keep], scores[keep]
    boxes = boxes / [W, H, W, H]
    order = np.argsort(-scores)
    return boxes[order], scores[order]
