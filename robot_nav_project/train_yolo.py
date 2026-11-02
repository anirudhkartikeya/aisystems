"""train_yolo.py - fine-tune a small YOLO on the simulator's camera pictures.

    python train_yolo.py              # 50 epochs on the Mac GPU (about 30-60 minutes)
    python train_yolo.py --epochs 3   # a quick test first (a few minutes)
    python train_yolo.py --imgsz 960 --name yolo11n_sim_960   # bigger pictures: better for far-away people

Reads dataset/data.yaml (made with make_yolo_data.py). Uses the train split to learn,
the val split to check progress, and leaves the test split untouched.
Results go to runs/yolo11n_sim/ (the script prints the exact folder) - the detector is weights/best.pt,
plus training curves, precision/recall plots and example predictions.
"""
import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--model", default="yolo11n.pt", help="starting weights (small pretrained YOLO)")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--name", default="yolo11n_sim")
ap.add_argument("--imgsz", type=int, default=640, help="training picture size (960 helps with far-away people)")
args = ap.parse_args()

device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
print("training on:", device)

model = YOLO(args.model)
model.train(data="dataset/data.yaml", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=0,
            device=device, workers=4, project=os.path.abspath("runs"), name=args.name, exist_ok=True,
            plots=True)

save_dir = Path(model.trainer.save_dir)                # wherever this YOLO version saved the run
best_path = save_dir / "weights" / "best.pt"
best = YOLO(str(best_path))
m = best.val(data="dataset/data.yaml", split="val", imgsz=args.imgsz, device=device, plots=True)
print(f"\nVALIDATION (person): precision {m.box.mp:.3f}  recall {m.box.mr:.3f}  "
      f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
print(f"Trained detector: {best_path}")
print(f"Plots and results: {save_dir}")
