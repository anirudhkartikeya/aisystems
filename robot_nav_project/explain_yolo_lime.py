"""explain_yolo_lime.py - LIME for the person detector, plus a check of the automatic labels.

    python explain_yolo_lime.py                                   # the 960 model on the validation pictures
    python explain_yolo_lime.py --model runs/yolo11n_sim/weights/best.pt --imgsz 640
    pip install scikit-image                                      # optional: proper superpixels (else a grid)

PART 1 - LIME (Ribeiro et al. 2016, the method used for the pneumonia X-rays), per person:
  * cut the area around the person (4x the box size) into ~50 superpixels
  * make 300 copies of the picture with random superpixels greyed out (filled with their own mean
    colour) and run YOLO on each; the score is YOLO's confidence for a person at THIS person's
    true box (IoU >= 0.3, 0 if no such box)
  * fit a weighted linear model: score ~ which superpixels were visible. Its weights say how much
    each superpixel helps (+, green) or hurts (-, red) the detection.
  Done for 12 people YOLO found (3 tiny, 3 small, 3 medium, 3 large) and 6 people it MISSED.
  Trust checks per person:
    inside share - share of the positive weight that lies on the person's own box (high = YOLO
                   looks at the person, not at the road / a shadow / the background)
    fit R2       - how well the simple linear explanation matches YOLO (low = don't trust that one)
    deletion     - hide the 3 most important superpixels vs 3 random ones: the confidence should
                   drop much more for the important ones (the explanation is faithful)

PART 2 - label check (the automatic boxes): a person gets a label only if at least 25 pixels are
  visible AND the box is at least 6 px wide and 6 px tall (config CAMERA_MIN_PIXELS / CAMERA_MIN_BOX).
  A far, thin person can be visible and still have no label. So:
    * overlay sheet: 12 random validation pictures with labels (green) and YOLO boxes (blue)
    * every confident YOLO box (conf >= 0.5) with NO label is cut out and enlarged for a human to
      look at: a real person without a label (label error) or a false detection? Boxes narrower or
      shorter than 6 px are flagged as "likely missed by the 6-px rule".

Writes runs/<model name>/lime/: lime_people.png, lime_results.csv, label_overlay.png,
unlabeled_detections.png, unlabeled_detections.csv, summary.txt
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
from PIL import Image, ImageDraw


def make_detector(model_path, imgsz, device, chunk=24):
    from ultralytics import YOLO
    model = YOLO(model_path)

    def detect(images, conf):
        """images: list of HxWx3 uint8 RGB -> list of (boxes xyxy in pixels (N,4), confidences (N,))."""
        out = []
        for i in range(0, len(images), chunk):
            batch = [np.ascontiguousarray(im[..., ::-1]) for im in images[i:i + chunk]]   # ultralytics: BGR
            for r in model.predict(source=batch, conf=conf, imgsz=imgsz, device=device, verbose=False, classes=[0]):
                b = r.boxes
                out.append((b.xyxy.cpu().numpy().reshape(-1, 4), b.conf.cpu().numpy().reshape(-1)))
        return out
    return detect


def iou(a, b):
    """a (N,4), b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def read_labels(path, W, H):
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5 and p[0] == "0":
                x, y, w, h = map(float, p[1:])
                boxes.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def size_group(h_frac):
    return "tiny" if h_frac < 0.03 else "small" if h_frac < 0.06 else "medium" if h_frac < 0.15 else "large"


def superpixels(crop, n):
    try:
        from skimage.segmentation import slic
        return slic(crop, n_segments=n, compactness=10, start_label=0), "SLIC superpixels"
    except ImportError:
        H, W = crop.shape[:2]
        g = int(round(np.sqrt(n)))
        seg = (np.minimum(np.arange(H) * g // H, g - 1)[:, None] * g + np.minimum(np.arange(W) * g // W, g - 1)[None, :])
        return seg, "grid cells (pip install scikit-image for superpixels)"


def score(dets, box, thr=0.3):
    """YOLO's confidence for a person at `box` (best overlapping detection), per picture."""
    out = []
    for b, c in dets:
        if len(b) == 0:
            out.append(0.0)
            continue
        m = iou(np.array([box]), b)[0] >= thr
        out.append(float(c[m].max()) if m.any() else 0.0)
    return np.array(out)


def lime_person(img, box, detect, n_samples, n_segments, rng):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    side = max(4 * max(x2 - x1, y2 - y1), 64)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx1, ry1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    rx2, ry2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = img[ry1:ry2, rx1:rx2]
    seg, how = superpixels(crop, n_segments)
    seg = np.unique(seg, return_inverse=True)[1].reshape(seg.shape)
    M = int(seg.max()) + 1
    means = np.array([crop[seg == k].mean(0) for k in range(M)])
    flat = means[seg].astype(np.uint8)

    def render(z):
        im = img.copy()
        c = crop.copy()
        off = z[seg] == 0
        c[off] = flat[off]
        im[ry1:ry2, rx1:rx2] = c
        return im

    Z = rng.integers(0, 2, size=(n_samples, M))
    Z[0] = 1
    y = score(detect([render(z) for z in Z], conf=0.001), box)
    # LIME: weighted ridge regression, cosine distance to the full picture, kernel width 0.25
    cos = Z.sum(1) / (np.sqrt(np.maximum(Z.sum(1), 1)) * np.sqrt(M))
    w = np.exp(-((1 - cos) ** 2) / 0.25 ** 2)
    A = np.hstack([Z, np.ones((n_samples, 1))]).astype(float)
    sw = np.sqrt(w)[:, None]
    coef = np.linalg.solve((A * sw).T @ (A * sw) + 0.01 * np.eye(M + 1), (A * sw).T @ (y[:, None] * sw))[:, 0]
    weights = coef[:M]
    fit = A @ coef
    r2 = 1 - np.sum(w * (y - fit) ** 2) / max(np.sum(w * (y - np.average(y, weights=w)) ** 2), 1e-12)
    # share of positive weight on the person's own box
    inside = np.zeros((ry2 - ry1, rx2 - rx1), bool)
    inside[int(max(0, y1 - ry1)):int(max(0, y2 - ry1)), int(max(0, x1 - rx1)):int(max(0, x2 - rx1))] = True
    frac_in = np.array([inside[seg == k].mean() for k in range(M)])
    pos = np.clip(weights, 0, None)
    inside_share = float((pos * frac_in).sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    # deletion check: hide the 3 most helpful superpixels vs 3 random ones
    k = min(3, M)
    z_top = np.ones(M, int); z_top[np.argsort(-weights)[:k]] = 0
    rand = [rng.choice(M, k, replace=False) for _ in range(5)]
    zs = [z_top] + [np.where(np.isin(np.arange(M), r), 0, 1) for r in rand]
    s = score(detect([render(z) for z in zs], conf=0.001), box)
    return {"crop": crop, "seg": seg, "weights": weights, "region": (rx1, ry1, rx2, ry2), "how": how,
            "score_full": float(y[0]), "r2": float(r2), "inside_share": inside_share,
            "score_hide_top3": float(s[0]), "score_hide_random3": float(s[1:].mean()), "segments": M}


def panel(res, box, title, size=190):
    crop, seg, wts = res["crop"], res["seg"], res["weights"]
    rx1, ry1 = res["region"][:2]
    scale = size / max(crop.shape[:2])
    big = lambda a: Image.fromarray(a).resize((int(crop.shape[1] * scale), int(crop.shape[0] * scale)), Image.NEAREST)
    lim = max(np.abs(wts).max(), 1e-6)
    heat = np.zeros(crop.shape, float)
    v = wts[seg] / lim
    heat[..., 1] = np.clip(v, 0, 1) * 255
    heat[..., 0] = np.clip(-v, 0, 1) * 255
    mix = (0.45 * crop + 0.55 * heat).astype(np.uint8)
    top = np.argsort(-wts)[:5]
    keep = np.isin(seg, top[wts[top] > 0])
    only = np.where(keep[..., None], crop, (crop * 0.25 + 90).astype(np.uint8))
    tiles = [big(crop), big(mix), big(only)]
    W = sum(t.width for t in tiles) + 20
    out = Image.new("RGB", (W, tiles[0].height + 34), (30, 30, 30))
    x = 0
    for t in tiles:
        d = ImageDraw.Draw(t)
        d.rectangle([(box[0] - rx1) * scale, (box[1] - ry1) * scale, (box[2] - rx1) * scale, (box[3] - ry1) * scale],
                    outline=(255, 255, 0), width=1)
        out.paste(t, (x, 34)); x += t.width + 10
    d = ImageDraw.Draw(out)
    d.text((4, 2), title, fill=(255, 255, 255))
    d.text((4, 17), f"conf {res['score_full']:.2f} | inside {res['inside_share']:.0%} | R2 {res['r2']:.2f} | "
                    f"hide top-3 -> {res['score_hide_top3']:.2f}, 3 random -> {res['score_hide_random3']:.2f}",
           fill=(200, 200, 200))
    return out


def stack(images, cols, bg=(30, 30, 30)):
    if not images:
        return None
    w = max(i.width for i in images); h = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), bg)
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def main(argv=None, detector=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/yolo11n_sim_960/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--segments", type=int, default=50)
    ap.add_argument("--found", type=int, default=3, help="people YOLO found, per size group")
    ap.add_argument("--missed", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.model))), "lime")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    if detector is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
        detector = make_detector(args.model, args.imgsz, device)
    files = sorted(glob.glob(os.path.join(args.data, "images", args.split, "*.jpg")) +
                   glob.glob(os.path.join(args.data, "images", args.split, "*.png")))
    print(f"{len(files)} {args.split} pictures | {args.model} | imgsz {args.imgsz}")

    # ---------------- run YOLO once on every picture, match to the labels ----------------
    found, missed, unlabeled, pics = [], [], [], []
    for i in range(0, len(files), 32):
        imgs = [np.array(Image.open(f).convert("RGB")) for f in files[i:i + 32]]
        dets = detector(imgs, conf=args.conf)
        for f, im, (b, c) in zip(files[i:i + 32], imgs, dets):
            H, W = im.shape[:2]
            name = os.path.splitext(os.path.basename(f))[0]
            gt = read_labels(os.path.join(args.data, "labels", args.split, name + ".txt"), W, H)
            m = iou(gt, b)
            hit = np.zeros(len(gt), bool); used = np.zeros(len(b), bool)
            for j in np.argsort(-c):                              # greedy, most confident first
                if len(gt) == 0:
                    break
                cand = np.where(~hit, m[:, j], -1)
                g = int(np.argmax(cand))
                if cand[g] >= 0.5:
                    hit[g] = True; used[j] = True
            for g, box in enumerate(gt):
                rec = {"file": f, "box": box, "h": (box[3] - box[1]) / H}
                (found if hit[g] else missed).append(rec)
            for j in np.where(~used & (c >= 0.5))[0]:
                bw, bh = b[j, 2] - b[j, 0], b[j, 3] - b[j, 1]
                near_label = bool(len(gt) and iou(b[j:j + 1], gt).max() > 0.1)
                unlabeled.append({"file": f, "box": b[j], "conf": float(c[j]), "w_px": float(bw), "h_px": float(bh),
                                  "flag": "box overlaps a label (duplicate / bad fit)" if near_label else
                                  ("likely a far person missed by the 6-px rule" if min(bw, bh) < 6 else
                                   "no label nearby: look at it")})
            pics.append((f, gt, b, c))
    n_gt = len(found) + len(missed)
    print(f"people labelled: {n_gt} | found at conf {args.conf}: {len(found)} | missed: {len(missed)} | "
          f"confident boxes with no label: {len(unlabeled)} ({time.time() - t0:.0f} s)")

    # ---------------- PART 1: LIME ----------------
    targets = []
    for grp in ("tiny", "small", "medium", "large"):
        pool = [r for r in found if size_group(r["h"]) == grp]
        for k in rng.permutation(len(pool))[:args.found]:
            targets.append(("found", grp, pool[k]))
    pool = [r for r in missed if size_group(r["h"]) != "tiny"] or missed
    for k in rng.permutation(len(pool))[:args.missed]:
        targets.append(("MISSED", size_group(pool[k]["h"]), pool[k]))
    rows, panels, how = [], [], "-"
    for n, (kind, grp, rec) in enumerate(targets, 1):
        img = np.array(Image.open(rec["file"]).convert("RGB"))
        res = lime_person(img, rec["box"], detector, args.samples, args.segments, rng)
        name = os.path.basename(rec["file"])
        how = res["how"]
        rows.append({"person": n, "status": kind, "size": grp, "picture": name,
                     "box_h_px": round(rec["box"][3] - rec["box"][1], 1), "superpixels": res["segments"],
                     "conf": round(res["score_full"], 3), "inside_share": round(res["inside_share"], 3),
                     "fit_R2": round(res["r2"], 3), "conf_hide_top3": round(res["score_hide_top3"], 3),
                     "conf_hide_random3": round(res["score_hide_random3"], 3)})
        panels.append(panel(res, rec["box"], f"#{n} {kind} {grp} person - {name}"))
        print(f"  LIME {n}/{len(targets)}: {kind:6s} {grp:6s} conf {res['score_full']:.2f} inside "
              f"{res['inside_share']:.0%} R2 {res['r2']:.2f} ({time.time() - t0:.0f} s)")
    with open(os.path.join(out, "lime_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sheet = stack(panels, 2)
    if sheet:
        sheet.save(os.path.join(out, "lime_people.png"))

    # ---------------- PART 2: label check ----------------
    over = []
    for k in rng.permutation(len(pics))[:12]:
        f, gt, b, c = pics[k]
        im = Image.open(f).convert("RGB"); d = ImageDraw.Draw(im)
        for box in gt:
            d.rectangle(list(box), outline=(0, 230, 0), width=2)
        for box, cc in zip(b, c):
            d.rectangle(list(box), outline=(60, 140, 255), width=1)
        d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
        d.text((3, 1), f"{os.path.basename(f)}  green = label, blue = YOLO", fill=(255, 255, 255))
        over.append(im.resize((im.width // 2, im.height // 2)))
    s = stack(over, 4)
    if s:
        s.save(os.path.join(out, "label_overlay.png"))
    tiles = []
    for u in sorted(unlabeled, key=lambda u: -u["conf"])[:24]:
        im = Image.open(u["file"]).convert("RGB")
        x1, y1, x2, y2 = u["box"]
        side = max(48, 3 * max(x2 - x1, y2 - y1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        c = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((160, 160), Image.NEAREST)
        sc = 160 / side
        d = ImageDraw.Draw(c)
        d.rectangle([(x1 - cx + side / 2) * sc, (y1 - cy + side / 2) * sc, (x2 - cx + side / 2) * sc,
                     (y2 - cy + side / 2) * sc], outline=(255, 230, 0), width=1)
        t = Image.new("RGB", (160, 190), (30, 30, 30)); t.paste(c, (0, 0))
        dd = ImageDraw.Draw(t)
        dd.text((2, 162), f"conf {u['conf']:.2f} {u['w_px']:.0f}x{u['h_px']:.0f}px", fill=(255, 255, 255))
        dd.text((2, 175), u["flag"][:30], fill=(255, 200, 0))
        tiles.append(t)
    s = stack(tiles, 6)
    if s:
        s.save(os.path.join(out, "unlabeled_detections.png"))
    with open(os.path.join(out, "unlabeled_detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["picture", "x1", "y1", "x2", "y2", "conf", "w_px", "h_px", "flag", "your_verdict"])
        for u in unlabeled:
            w.writerow([os.path.basename(u["file"])] + [round(float(v), 1) for v in u["box"]] +
                       [round(u["conf"], 3), round(u["w_px"], 1), round(u["h_px"], 1), u["flag"], ""])

    # ---------------- summary ----------------
    fr = [r for r in rows if r["status"] == "found"]
    ms = [r for r in rows if r["status"] == "MISSED"]
    mean = lambda rr, k: float(np.nanmean([r[k] for r in rr])) if rr else float("nan")
    flags = {}
    for u in unlabeled:
        flags[u["flag"]] = flags.get(u["flag"], 0) + 1
    lines = [f"YOLO {args.model} on {len(files)} {args.split} pictures (imgsz {args.imgsz}, conf {args.conf})",
             f"LIME: {args.samples} perturbed copies per person, about {args.segments} {how} around each person", "",
             f"PEOPLE YOLO FOUND ({len(fr)}): positive weight on the person's own box {mean(fr, 'inside_share'):.0%} on average; "
             f"fit R2 {mean(fr, 'fit_R2'):.2f}; confidence {mean(fr, 'conf'):.2f} -> hiding the top-3 superpixels "
             f"{mean(fr, 'conf_hide_top3'):.2f} vs 3 random {mean(fr, 'conf_hide_random3'):.2f}",
             f"PEOPLE YOLO MISSED ({len(ms)}): best confidence at their box {mean(ms, 'conf'):.2f}; inside share "
             f"{mean(ms, 'inside_share'):.0%}; fit R2 {mean(ms, 'fit_R2'):.2f}", "",
             "Per person:"]
    lines += [", ".join(f"{k} {v}" for k, v in r.items()) for r in rows]
    lines += ["", f"LABEL CHECK: {len(unlabeled)} confident YOLO boxes (conf >= 0.5) have no label:"]
    lines += [f"  {v:4d}  {k}" for k, v in sorted(flags.items(), key=lambda kv: -kv[1])]
    lines += ["  -> open unlabeled_detections.png, decide for each: real person (label missing) or false detection,",
              "     and write it in the 'your_verdict' column of unlabeled_detections.csv"]
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {out}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
