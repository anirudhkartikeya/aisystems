"""analyze_yolo.py - which people does the trained YOLO miss, and when does it see people who aren't there?

    python analyze_yolo.py
    python analyze_yolo.py --model runs/yolo11n_sim/weights/best.pt --conf 0.25
    python analyze_yolo.py --model runs/yolo11n_sim_960/weights/best.pt --imgsz 960 --data dataset_night --tag _night
    python analyze_yolo.py ... --slice --tag _sliced     # with slicing (detect.py, SAHI-style), timed

Runs the trained detector on the VALIDATION pictures (the test split stays untouched),
matches its boxes to the true boxes (IoU >= 0.5) and reports:
  - recall (share of people found) by person size (far / near), lighting, town or street,
    camera (traffic car or the robot car), crowded pictures, people cut off at the edge
  - precision (share of detections that are real people) by lighting
  - a picture sheet of the frames with the most missed people:
      green = found, red = MISSED person, yellow = false detection (nobody there)
Saves everything in runs/<name>/analysis/.
"""
import argparse
import time
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/yolo11n_sim/weights/best.pt")
ap.add_argument("--data", default="dataset")
ap.add_argument("--split", default="val")
ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for a detection")
ap.add_argument("--iou", type=float, default=0.5, help="overlap needed to count a detection as correct")
ap.add_argument("--imgsz", type=int, default=640, help="picture size the model was trained with")
ap.add_argument("--tag", default="", help="added to the output names, e.g. conf015")
ap.add_argument("--slice", action="store_true", help="detect.py with slicing (tiles), SAHI-style")
args = ap.parse_args()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.model))), "analysis")
os.makedirs(out_dir, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

# picture info (lighting, town/street, camera) from the dataset index
info = {}
for f in glob.glob(os.path.join(args.data, "index_*.csv")):
    for r in csv.DictReader(open(f)):
        info[r["image"]] = r

img_dir = os.path.join(args.data, "images", args.split)
lab_dir = os.path.join(args.data, "labels", args.split)
images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
print(f"{len(images)} {args.split} pictures | model {args.model} | conf >= {args.conf} | device {device}")


def read_labels(name):
    boxes = []
    path = os.path.join(lab_dir, name + ".txt")
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5 and p[0] == "0":
                x, y, w, h = map(float, p[1:])
                boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def size_group(h):
    if h < 0.03:
        return "1 tiny (very far, <3% of picture height)"
    if h < 0.06:
        return "2 small (far, 3-6%)"
    if h < 0.15:
        return "3 medium (6-15%)"
    return "4 large (close, >15%)"


model = YOLO(args.model)
found = defaultdict(lambda: [0, 0])          # group -> [found, total]
dets = defaultdict(lambda: [0, 0])           # lighting -> [correct detections, all detections]
per_image = []


def detections():
    """(picture path, boxes as shares of width/height, confidences) for every picture.
    One picture at a time through detect.py - the same way with and without slicing (a long
    stream of pictures can stall YOLO's box merging on the Mac GPU: "NMS time limit exceeded")."""
    from detect import detect
    for path in images:
        b, c = detect(model, path, conf=args.conf, imgsz=args.imgsz, slice=args.slice, device=device)
        yield path, b, c


t_start = time.time()
for path, pred, conf in detections():
    name = os.path.splitext(os.path.basename(path))[0]
    meta = info.get(name, {})
    gt = read_labels(name)
    order = np.argsort(-conf)
    pred, conf = pred[order], conf[order]
    ious = iou_matrix(pred, gt)
    gt_hit = np.zeros(len(gt), dtype=bool)
    pred_ok = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):                                     # greedy matching, most confident first
        if len(gt) == 0:
            break
        cand = np.where(~gt_hit, ious[i], -1)
        j = int(np.argmax(cand))
        if cand[j] >= args.iou:
            gt_hit[j] = True
            pred_ok[i] = True
    crowded = "crowded (6+ people)" if len(gt) >= 6 else "not crowded (<6 people)"
    for k, b in enumerate(gt):
        h = b[3] - b[1]
        edge = "cut off at picture edge" if (b[0] < 0.01 or b[2] > 0.99 or b[3] > 0.99) else "fully in picture"
        for g in ("ALL", "size: " + size_group(h), "lighting: " + meta.get("lighting", "?"),
                  "place: " + meta.get("layout", "?"), "camera: " + meta.get("camera_on", "?"),
                  "crowd: " + crowded, "edge: " + edge):
            found[g][0] += int(gt_hit[k])
            found[g][1] += 1
    light = meta.get("lighting", "?")
    dets[light][0] += int(pred_ok.sum()); dets[light][1] += len(pred)
    dets["ALL"][0] += int(pred_ok.sum()); dets["ALL"][1] += len(pred)
    per_image.append({"name": name, "path": path, "gt": gt, "gt_hit": gt_hit, "pred": pred,
                      "pred_ok": pred_ok, "misses": int((~gt_hit).sum()), "false": int((~pred_ok).sum()),
                      "lighting": light})

# ---------------- tables ----------------
ms_per_picture = (time.time() - t_start) / max(1, len(per_image)) * 1000
print(f"\n{'SLICED (tiles + whole picture)' if args.slice else 'whole picture'}: "
      f"{ms_per_picture:.0f} ms per picture (incl. reading it)")
rows = [{"group": "ms per picture" + (" (sliced)" if args.slice else ""), "found": None, "people": None,
         "recall": round(ms_per_picture, 1)}]
print(f"\nRECALL (share of people found), {args.split} split, conf >= {args.conf}, IoU >= {args.iou}")
print(f"{'group':55s} {'found':>7s} {'people':>7s} {'recall':>7s}")
for g in sorted(found, key=lambda g: (g != "ALL", g)):
    f, t = found[g]
    rows.append({"group": g, "found": f, "people": t, "recall": round(f / t, 3) if t else None})
    print(f"{g:55s} {f:7d} {t:7d} {f / t if t else float('nan'):7.2f}")
print(f"\nPRECISION (share of detections that are real people) by lighting")
for g in sorted(dets, key=lambda g: (g != "ALL", g)):
    c, t = dets[g]
    rows.append({"group": "precision / lighting: " + g, "found": c, "people": t,
                 "recall": round(c / t, 3) if t else None})
    print(f"{g:15s} {c:6d} correct of {t:6d} detections  -> precision {c / t if t else float('nan'):.2f}")
with open(os.path.join(out_dir, f"recall_by_group{args.tag}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "found", "people", "recall"])
    w.writeheader(); w.writerows(rows)

# ---------------- picture sheet of the worst frames ----------------
worst = sorted(per_image, key=lambda r: -(r["misses"] * 2 + r["false"]))[:8]
tiles = []
for r in worst:
    im = Image.open(r["path"]).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for b, hit in zip(r["gt"], r["gt_hit"]):
        d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(0, 220, 0) if hit else (255, 0, 0),
                    width=2 if hit else 3)
    for b, ok in zip(r["pred"], r["pred_ok"]):
        if not ok:
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(255, 230, 0), width=2)
    d.rectangle([0, 0, W, 16], fill=(0, 0, 0))
    d.text((4, 2), f"{r['name']}  {r['lighting']}  missed {r['misses']}  false {r['false']}", fill=(255, 255, 255))
    tiles.append(im)
if tiles:
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * 2, h * ((len(tiles) + 1) // 2)), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * w, (i // 2) * h))
    sheet.save(os.path.join(out_dir, f"worst_frames{args.tag}.jpg"), quality=90)

total_miss = sum(r["misses"] for r in per_image)
total_false = sum(r["false"] for r in per_image)
print(f"\nmissed people: {total_miss} | false detections: {total_false} | pictures: {len(per_image)}")
print(f"Saved: {out_dir}/recall_by_group{args.tag}.csv and {out_dir}/worst_frames{args.tag}.jpg "
      f"(green = found, red = MISSED, yellow = false detection)")
