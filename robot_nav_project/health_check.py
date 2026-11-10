"""health_check.py - is EVERYTHING working, before we package the models (ONNX / API / Docker)?

    python health_check.py            # about 3-6 minutes
    python health_check.py --quick    # skip the slower parts (dataset scripts, dataset checker)

Every check prints PASS, WARN (works, but look at it) or FAIL, and the whole report is saved to
health_report.txt. Nothing in the project is changed: temporary files go to health_check_tmp/
and are removed at the end.

 1. Setup        Python packages, the trained models and the saved datasets are there
 2. Building     the near-miss rule (test_near_miss.py), the Kalman filter (car and training
    blocks      versions agree, it learns the camera noise), slicing (tiles + box merging),
                 the "only label people a camera could see" rule
 3. Driving      every brain drives a dev scene with today's settings and gives the SAME result
                 as the stress tests we based our decisions on (run_data/stress_dev_guard*),
                 and the same scene twice gives the same result (deterministic)
 4. Old results  with the new parts switched off (no Kalman filter, no still-person guard,
                 LSTM v2) the code still reproduces the saved core dataset run_data/v3 exactly
 5. run_sim      the normal command works and writes its run report
 6. YOLO         both detectors load, find people in day and night pictures (normal and sliced)
 7. Datasets     make_yolo_data.py makes day and night pictures with labels; check_run_dataset.py
                 still passes on run_data/v3
"""
import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import config

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="skip the dataset scripts and the dataset checker")
ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
ap.add_argument("--brains", nargs="+", default=["reactive", "predictive", "lstm"])
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
TMP = os.path.join(HERE, "health_check_tmp")
lines, results = [], []
T0 = time.time()


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def check(name, status, detail=""):
    """status: True -> PASS, False -> FAIL, 'warn' -> WARN."""
    tag = "PASS" if status is True else ("WARN" if status == "warn" else "FAIL")
    results.append((tag, name))
    say(f"  [{tag}] {name}" + (f" - {detail}" if detail else ""))


def guarded(name, fn):
    """Run one check; a crash is a FAIL with the error, not the end of the health check."""
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001 - report every problem
        check(name, False, f"crashed: {type(e).__name__}: {e}")


def run(cmd, timeout=600):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


# ===================================================================== 1. setup
def part_setup():
    say("\n1. SETUP")
    vers = []
    for mod in ("numpy", "pybullet", "torch", "ultralytics", "PIL", "matplotlib", "skimage", "pandas"):
        try:
            m = __import__(mod)
            vers.append(f"{mod} {getattr(m, '__version__', 'ok')}")
        except Exception as e:                               # noqa: BLE001
            check(f"package {mod}", False, str(e))
    say(f"  python {sys.version.split()[0]} | " + " | ".join(vers))
    must = {"LSTM v3 (default brain model)": config.LSTM_MODEL,
            "LSTM v2": "results/lstm_v2/lstm_v2.pt",
            "YOLO day (yolo11n_sim_960)": "runs/yolo11n_sim_960/weights/best.pt",
            "YOLO day + night (yolo11n_sim_960_night)": "runs/yolo11n_sim_960_night/weights/best.pt",
            "core run dataset run_data/v3": "run_data/v3/manifest.json",
            "YOLO day dataset": "dataset/data.yaml",
            "YOLO night dataset": "dataset_night/data.yaml",
            "day + night training list": "dataset_mix.yaml"}
    for name, path in must.items():
        check(name, os.path.exists(path), path)
    check("LSTM dataset (trajectories/, only needed to retrain)",
          True if glob.glob("trajectories/peds_*.csv") else "warn", "trajectories/")
    say(f"  settings: TRACK_FILTER={config.TRACK_FILTER}, STILL_GUARD={config.STILL_GUARD}, "
        f"LSTM_MODEL={config.LSTM_MODEL}, DETECT_SLICE={config.DETECT_SLICE}, BRAIN={config.BRAIN}")


# ===================================================================== 2. building blocks
def part_blocks():
    say("\n2. BUILDING BLOCKS")

    def near_miss():
        code, out = run(["test_near_miss.py"], timeout=120)
        fails = out.count("FAIL")
        check("near-miss / contact rule (test_near_miss.py)", code == 0 and fails == 0,
              f"exit {code}, {fails} FAIL lines")
    guarded("near-miss rule", near_miss)

    def kalman():
        from tracking import KalmanCV, SensorNoise, kalman_batch
        rng = np.random.default_rng(0)
        z = np.cumsum(rng.normal(0, 0.1, (4, 45, 2)), 1) + rng.normal(0, 0.1, (4, 45, 2))
        var = np.array([0.02, 0.05, 0.1, 0.1]) ** 2
        fb = kalman_batch(z, 0.1, var, config.KALMAN_ACCEL)
        worst = 0.0
        for i in range(4):
            n = SensorNoise(math.sqrt(var[i]), learn=False)
            kf = KalmanCV(0, *z[i, 0], n.var, config.KALMAN_ACCEL)
            for k in range(1, 45):
                kf.update(0.1 * k, *z[i, k], n)
                worst = max(worst, math.hypot(*(np.array(kf.position) - fb[i, k])))
        check("Kalman filter: training version = car version", worst < 1e-9, f"largest difference {worst:.1e} m")
        # does it learn the camera noise? a person walking, measured with 0 / 0.1 m noise
        learned = {}
        for sigma in (0.0, 0.1):
            noise = SensorNoise(config.KALMAN_MEAS_STD, config.KALMAN_NOISE_FLOOR, config.KALMAN_NOISE_RATE, True)
            t = np.arange(0, 30, 0.1)
            true = np.stack([1.3 * t, 0.3 * np.sin(t / 3)], 1)
            meas = true + rng.normal(0, sigma, true.shape)
            kf = KalmanCV(0, *meas[0], noise.var, config.KALMAN_ACCEL)
            for k in range(1, len(t)):
                kf.update(t[k], *meas[k], noise)
            learned[sigma] = noise.std
        ok = learned[0.0] < 0.04 and 0.06 < learned[0.1] < 0.14
        check("Kalman filter learns the camera noise", ok,
              f"learned {learned[0.0]:.3f} m with exact positions, {learned[0.1]:.3f} m with 0.1 m noise")
    guarded("Kalman filter", kalman)

    def slicing():
        from detect import nms, tiles
        t = tiles(640, 480, 320, 0.25)
        cover = np.zeros((480, 640), bool)
        for x, y in t:
            cover[y:y + 320, x:x + 320] = True
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
        keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
        check("slicing: tiles cover the picture, overlapping boxes merge", cover.all() and sorted(keep) == [0, 2],
              f"{len(t)} tiles, kept boxes {sorted(keep.tolist())}")
    guarded("slicing", slicing)

    def visible():
        import types
        from camera import person_boxes
        seg = np.full((100, 100), -1)
        rgb = np.zeros((100, 100, 3), np.uint8)
        seg[10:40, 10:20] = 5                                     # dark person in the dark
        seg[10:40, 50:60] = 6; rgb[:, 40:70] = 80; rgb[10:40, 50:60] = 6   # dark coat, lit wall
        seg[60:90, 80:90] = 7; rgb[60:90, 80:90] = 60             # lit person
        peds = [types.SimpleNamespace(label_ids=[i], ped_id=i) for i in (5, 6, 7)]
        kept = [b[-1] for b in person_boxes(seg, peds, rgb=rgb)]
        old = [b[-1] for b in person_boxes(seg, peds)]
        check("labels only for people a camera could see (night)", kept == [6, 7] and old == [5, 6, 7],
              f"night rule keeps {kept}, daytime rule keeps {old}")
    guarded("visibility rule", visible)


# ===================================================================== 3 + 4. driving
def _rows(path, brain, cond="clean"):
    if not os.path.exists(path):
        return {}
    return {int(r["seed"]): r for r in csv.DictReader(open(path)) if r["brain"] == brain and r.get("condition", cond) == cond}


def part_driving():
    from multiprocessing import get_context
    from stress_test import run_stress
    keys = ("result", "ped_collisions", "near_misses", "min_gap")
    seeds = [0, 1, 2]
    old = {"TRACK_FILTER": "none", "STILL_GUARD": False, "LSTM_MODEL": "results/lstm_v2/lstm_v2.pt"}
    jobs = []
    for b in args.brains:
        jobs += [("now", (b, s, "clean", "town", "crossing", {})) for s in seeds]
        jobs += [("old", (b, s, "clean", "town", "crossing", old)) for s in (0, 2000)]
    jobs.append(("again", ("predictive", 1, "clean", "town", "crossing", {})))
    say(f"\n3. DRIVING (today's settings) and 4. OLD RESULTS - {len(jobs)} runs on {args.workers} workers ...")
    pool = get_context("spawn").Pool(args.workers)
    out = pool.map(run_stress, [j for _, j in jobs])
    pool.close()
    res = [(tag, job, row) for (tag, job), row in zip(jobs, out)]

    # 3a. each brain drives, numbers are finite
    for b in args.brains:
        rr = [row for tag, job, row in res if tag == "now" and job[0] == b]
        finite = all(math.isfinite(float(r["min_gap"])) and math.isfinite(float(r["time"])) for r in rr)
        summary = ", ".join(f"seed {r['seed']}: {r['result']} {r['ped_collisions']} contact(s)" for r in rr)
        check(f"{b} brain drives with today's settings", finite and all(r["result"] for r in rr), summary)
    # 3b. today's code = what the dev stress tests measured
    ref = {"reactive": "run_data/stress_dev_guard/stress_runs.csv",
           "predictive": "run_data/stress_dev_guard/stress_runs.csv",
           "lstm": "run_data/stress_dev_guard_v3/stress_runs.csv"}
    for b, path in ((b, ref[b]) for b in args.brains):
        saved = _rows(path, b)
        rr = [row for tag, job, row in res if tag == "now" and job[0] == b]
        if not saved:
            check(f"{b}: same as the dev stress test", "warn", f"no {path} to compare with")
            continue
        same = sum(all(str(r[k]) == saved[r["seed"]][k] for k in keys) for r in rr if r["seed"] in saved)
        check(f"{b}: same result as the dev stress test ({os.path.dirname(path)})", same == len(rr),
              f"{same}/{len(rr)} runs identical")
    # 3c. deterministic
    a = next(row for tag, job, row in res if tag == "now" and job[0] == "predictive" and job[1] == 1)
    b2 = next(row for tag, job, row in res if tag == "again")
    same = all(str(a[k]) == str(b2[k]) for k in keys + ("time", "hard_brakes"))
    check("the same scene twice gives the same result", same, f"{a['result']} / {b2['result']}")
    # 4. old switches reproduce the saved core dataset
    saved = {}
    for r in csv.DictReader(open("run_data/v3/runs.csv")):
        saved[(r["brain"], int(r["seed"]))] = r
    rr = [(job, row) for tag, job, row in res if tag == "old"]
    same = sum(all(str(row[k]) == saved[(job[0], job[1])][k] for k in keys) for job, row in rr)
    check("new parts switched off -> run_data/v3 reproduced exactly", same == len(rr),
          f"{same}/{len(rr)} runs identical (no filter, no guard, LSTM v2; dev + eval seeds)")


# ===================================================================== 5. run_sim
def part_run_sim():
    say("\n5. RUN_SIM (the normal command, without a window)")
    before = set(glob.glob("run_reports/*"))
    t = time.time()
    code, out = run(["run_sim.py", "--headless", "--seed", "3"], timeout=600)
    new = sorted(set(glob.glob("run_reports/*")) - before)
    files = os.listdir(new[-1]) if new else []
    ok = code == 0 and {"report.html", "report.txt", "report.png"} <= set(files)
    res_line = next((l.strip() for l in out.splitlines() if "result" in l.lower()), "")
    check("python run_sim.py --headless works and writes its report", ok,
          f"exit {code}, {time.time() - t:.0f} s, report {new[-1] if new else 'missing'} | {res_line[:90]}")
    if code != 0:
        say("    last lines:\n    " + "\n    ".join(out.strip().splitlines()[-6:]))


# ===================================================================== 6. YOLO
def part_yolo():
    say("\n6. YOLO")
    from detect import detect, load_model

    def recall_on(model, imgs, data, sliced):
        """Share of labelled people found (IoU >= 0.5, most confident box first)."""
        found = total = 0
        for path in imgs:
            name = os.path.splitext(os.path.basename(path))[0]
            gt = []
            for line in open(os.path.join(data, "labels", "val", name + ".txt")):
                c, x, y, w, h = map(float, line.split())
                gt.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            gt = np.array(gt).reshape(-1, 4)
            boxes, _ = detect(model, path, conf=0.25, imgsz=960, slice=sliced)
            hit = np.zeros(len(gt), bool)
            for bx in boxes:
                if not len(gt):
                    break
                x1 = np.maximum(bx[0], gt[:, 0]); y1 = np.maximum(bx[1], gt[:, 1])
                x2 = np.minimum(bx[2], gt[:, 2]); y2 = np.minimum(bx[3], gt[:, 3])
                inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                iou = inter / ((bx[2] - bx[0]) * (bx[3] - bx[1]) + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter)
                iou[hit] = -1
                j = int(np.argmax(iou))
                if iou[j] >= 0.5:
                    hit[j] = True
            found += int(hit.sum())
            total += len(gt)
        return (found / total if total else 1.0), found, total

    for name, path in (("day model", "runs/yolo11n_sim_960/weights/best.pt"),
                       ("day + night model", "runs/yolo11n_sim_960_night/weights/best.pt")):
        model = load_model(path)
        for data in ("dataset", "dataset_night"):
            imgs = sorted(glob.glob(f"{data}/images/val/*.jpg"))[::10][:20]
            for sliced in (False, True):
                t = time.time()
                rec, found, total = recall_on(model, imgs, data, sliced)
                ms = (time.time() - t) / max(1, len(imgs)) * 1000
                ok = True if rec >= 0.6 else ("warn" if rec >= 0.4 else False)
                check(f"{name} on {data} ({'sliced' if sliced else 'whole picture'})", ok,
                      f"found {found}/{total} people in {len(imgs)} val pictures ({rec:.2f}), {ms:.0f} ms/picture")


# ===================================================================== 7. dataset scripts
def part_datasets():
    say("\n7. DATASET SCRIPTS")
    os.makedirs(TMP, exist_ok=True)
    for label, extra in (("day", []), ("night + dark clothes", ["--lightings", "night", "--dark-clothes", "0.35"])):
        out_dir = os.path.join(TMP, "yolo_" + label.split()[0])
        code, out = run(["make_yolo_data.py", "--episodes", "1", "--start", "5003", "--seconds", "15",
                         "--out", out_dir] + extra, timeout=600)
        imgs = glob.glob(f"{out_dir}/images/*/*.jpg")
        labels = glob.glob(f"{out_dir}/labels/*/*.txt")
        check(f"make_yolo_data.py ({label})", code == 0 and len(imgs) > 0 and len(imgs) == len(labels),
              f"exit {code}, {len(imgs)} pictures, {len(labels)} label files")
    code, out = run(["check_run_dataset.py", "run_data/v3"], timeout=900)
    summary = next((l.strip() for l in out.splitlines() if "checks passed" in l), "no summary line")
    check("check_run_dataset.py run_data/v3", code == 0, summary)
    shutil.rmtree(TMP, ignore_errors=True)


def main():
    say(f"HEALTH CHECK - {time.strftime('%Y-%m-%d %H:%M')} - {HERE}")
    part_setup()
    part_blocks()
    guarded("driving", part_driving)
    guarded("run_sim", part_run_sim)
    guarded("YOLO", part_yolo)
    if not args.quick:
        guarded("dataset scripts", part_datasets)
    n = {t: sum(1 for r in results if r[0] == t) for t in ("PASS", "WARN", "FAIL")}
    say(f"\n{n['PASS']} PASS | {n['WARN']} WARN | {n['FAIL']} FAIL   ({time.time() - T0:.0f} s)")
    for tag, name in results:
        if tag != "PASS":
            say(f"  {tag}: {name}")
    say("Everything works." if not n["FAIL"] else "Something needs fixing - see the FAIL lines above.")
    open("health_report.txt", "w").write("\n".join(lines) + "\n")
    print("Saved health_report.txt")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
