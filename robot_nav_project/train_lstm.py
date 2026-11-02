"""train_lstm.py - LSTM v2: pedestrian path prediction WITH context.

    python train_lstm.py                  # 3 runs (v1 replica, v2 without context, v2 with context), CPU
    python train_lstm.py --epochs 8       # quicker
    python train_lstm.py --runs v2        # only the full v2 model

Uses the LSTM dataset (trajectories/*.csv from make_data.py) - nothing new to generate.
Every past step of a person gets the context from lstm_features.py (kerb, crossing and its
light, nearest car and the robot car, 3 nearest people). The model predicts a CORRECTION to
constant velocity, so it starts from the baseline and learns when people stop, turn or cross.

Runs on the CPU by default (so a YOLO training can keep the GPU). Validation examples are the
same as in training_experiments.ipynb (history 12 points, future 8, every 4th point, val split),
so the numbers compare directly. The test split is not used.
Writes results/lstm_v2/: results.csv (every run), hard_cases.csv, lstm_v2.pt (the best model with
everything the car needs to use it), loss curves.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import lstm_features as lf
from roadmap import RoadMap

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="trajectories")
ap.add_argument("--prefix", default="0_499", help="which CSV set: peds_<prefix>.csv etc.")
ap.add_argument("--runs", default="all", choices=["all", "v2"])
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--obs", type=int, default=8, help="history points the model sees (8 = 2.8 s)")
ap.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU free) or mps")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--warmup", type=float, default=5.0, help="seconds make_data.py skipped before logging")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/lstm_v2")
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
DT, OBS_MAX, PRED = 0.4, 12, 8
WIN = OBS_MAX + PRED


# ======================================================================= 1. data + context features
def load_features():
    cache = os.path.join(args.data, f"features_v2_{args.prefix}.npz")
    peds = pd.read_csv(os.path.join(args.data, f"peds_{args.prefix}.csv"))
    peds = peds.sort_values(["episode", "ped_id", "t"]).reset_index(drop=True)      # track order
    if os.path.exists(cache):
        z = np.load(cache)
        if z["n"] == len(peds):
            print(f"context features loaded from {cache}")
            return peds, z["F"]
    t0 = time.time()
    vehs = pd.read_csv(os.path.join(args.data, f"vehicles_{args.prefix}.csv"))
    F = np.zeros((len(peds), lf.N_FEATURES), dtype=np.float32)
    xy = peds[["x", "y"]].to_numpy(float)
    vel = peds[["vx", "vy"]].to_numpy(float)
    sim_t = peds["t"].to_numpy(float) + args.warmup
    for layout in peds["layout"].unique():
        sel = (peds["layout"] == layout).to_numpy()
        F[sel, :10] = lf.map_features(RoadMap(layout), xy[sel], sim_t[sel])
        print(f"  road + crossing context for {layout}: {sel.sum():,} points ({time.time() - t0:.0f} s)")
    # cars and other people: one moment at a time
    vg = {k: g for k, g in vehs.groupby(["episode", "t"])}
    frames = peds.groupby(["episode", "t"]).indices
    for n, (key, idx) in enumerate(frames.items()):
        g = vg.get(key)
        if g is not None:
            cxy, cv = g[["x", "y"]].to_numpy(float), g[["vx", "vy"]].to_numpy(float)
            rmask = (g["kind"] == "robot").to_numpy()
        else:
            cxy, cv, rmask = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, bool)
        F[idx, 10:] = lf.frame_features(xy[idx], vel[idx], cxy, cv, rmask)
        if n % 20000 == 0:
            print(f"  cars + people context: {n:,}/{len(frames):,} moments ({time.time() - t0:.0f} s)")
    np.savez_compressed(cache, F=F, n=len(peds))
    print(f"context features computed in {time.time() - t0:.0f} s, saved to {cache}")
    return peds, F


print("loading data ...")
peds, FEAT = load_features()
XY = peds[["x", "y"]].to_numpy(np.float32)
MODE = peds["mode"].to_numpy()
KIND = peds["kind"].to_numpy()
ROAD = peds["on_road"].to_numpy()


def windows(split, stride):
    """Start index of every 20-point window (12 history + 8 future) inside one track."""
    starts = []
    part = peds[peds["split"] == split]
    for _, idx in part.groupby(["episode", "ped_id"]).indices.items():
        idx = part.index.to_numpy()[idx]
        first, n = idx[0], len(idx)
        if n >= WIN:
            starts.append(first + np.arange(0, n - WIN + 1, stride))
    return np.concatenate(starts)


S_train = windows("train", 2)
S_val = windows("val", 4)
print(f"examples: train {len(S_train):,} | validation {len(S_val):,} (same validation examples as the notebook)")
OFF = np.arange(WIN)
SCALE = float(np.diff(XY[S_train[:, None] + np.arange(OBS_MAX)], axis=1).std())

# context normalisation (training split only)
sample = FEAT[S_train[np.random.default_rng(0).choice(len(S_train), min(50000, len(S_train)), replace=False)] + OBS_MAX - 1]
CTX_MEAN, CTX_STD = sample.mean(0), sample.std(0) + 1e-3
binary = [i for i, n in enumerate(lf.NAMES) if n in ("on_road", "xing_signal", "xing_walk", "on_xing")
          or n.endswith("present")]
CTX_MEAN[binary], CTX_STD[binary] = 0.0, 1.0


def batch(starts, obs, context):
    """Model inputs, future (relative to the last seen point) and the constant-velocity guess."""
    w = XY[starts[:, None] + OFF]                                        # (B, 20, 2)
    hist = w[:, OBS_MAX - obs:OBS_MAX]
    last = w[:, OBS_MAX - 1:OBS_MAX]
    steps = np.diff(hist, axis=1) / SCALE                                # (B, obs-1, 2)
    x = steps
    if context:
        ctx = FEAT[starts[:, None] + np.arange(OBS_MAX - obs + 1, OBS_MAX)]    # (B, obs-1, F)
        x = np.concatenate([steps, (ctx - CTX_MEAN) / CTX_STD], axis=2)
    fut = w[:, OBS_MAX:] - last
    cv = (w[:, OBS_MAX - 1] - w[:, OBS_MAX - 2])[:, None, :] * np.arange(1, PRED + 1)[None, :, None]
    return x.astype(np.float32), fut.astype(np.float32), cv.astype(np.float32)


# ======================================================================= 2. model
class Predictor(nn.Module):
    """LSTM over the history -> 8 future steps. residual=True: a correction added to constant velocity."""

    def __init__(self, n_in, hidden, residual):
        super().__init__()
        self.residual = residual
        self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, PRED * 2)) if residual \
            else nn.Linear(hidden, PRED * 2)
        if residual:                                    # start exactly at constant velocity
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, cv):
        _, (h, _) = self.lstm(x)
        out = torch.cumsum(self.head(h[-1]).view(-1, PRED, 2), dim=1) * SCALE
        return cv + out if self.residual else out


def predict(model, starts, obs, context, bs=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            x, _, cv = batch(starts[i:i + bs], obs, context)
            outs.append(model(torch.from_numpy(x).to(device), torch.from_numpy(cv).to(device)).cpu().numpy())
    return np.concatenate(outs)


def train(name, obs, context, residual, hidden, epochs, lr, max_train=None):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr = S_train if not max_train or len(S_train) <= max_train else rng.choice(S_train, max_train, replace=False)
    n_in = 2 + (lf.N_FEATURES if context else 0)
    model = Predictor(n_in, hidden, residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    _, fut_val, _ = batch(S_val, obs, context)
    best, best_state, hist = math.inf, None, []
    t0 = time.time()
    print(f"\n=== {name}: history {obs} pts, context {'yes' if context else 'no'}, "
          f"{'correction to constant velocity' if residual else 'direct'}, hidden {hidden}, "
          f"{epochs} epochs, {len(tr):,} examples, device {device}")
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        total = 0.0
        for i in range(0, len(perm), args.batch):
            x, fut, cv = batch(perm[i:i + args.batch], obs, context)
            x, fut, cv = (torch.from_numpy(a).to(device) for a in (x, fut, cv))
            loss = ((model(x, cv) - fut) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(x)
        pred = predict(model, S_val, obs, context)
        err = np.linalg.norm(pred - fut_val, axis=2)
        ade, fde = err.mean(), err[:, -1].mean()
        hist.append((ep + 1, total / len(tr), ade, fde))
        if ade < best:
            best, best_state = ade, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep + 1:2d}/{epochs}: train loss {total / len(tr):.4f} | val ADE {ade:.3f} m  FDE {fde:.3f} m "
              f"| {time.time() - t0:.0f} s")
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


# ======================================================================= 3. runs
info_val = pd.DataFrame({
    "mode_now": MODE[S_val + OBS_MAX - 1], "mode_end": MODE[S_val + WIN - 1], "kind": KIND[S_val + OBS_MAX - 1],
    "road_future": ROAD[S_val[:, None] + np.arange(OBS_MAX, WIN)].max(1),
    "car_near": FEAT[S_val + OBS_MAX - 1, lf.NAMES.index("car_present")] *
    (np.hypot(*(FEAT[S_val + OBS_MAX - 1][:, [10, 11]] * 10).T) < 10.0),
})
groups = {
    "all examples": np.ones(len(S_val), bool),
    "HARD: person changes what they do (stop / start crossing / turn back)":
        (info_val.mode_now != info_val.mode_end).to_numpy(),
    "on the road in the next 3.2 s": info_val.road_future.to_numpy() == 1,
    "a car within 10 m": info_val.car_near.to_numpy() > 0,
    "children": (info_val.kind == "child").to_numpy(),
    "elderly": (info_val.kind == "elderly").to_numpy(),
}
_, FUT_VAL, CV_VAL = batch(S_val, 8, False)
rows, curves = [], {}


def evaluate(name, pred, train_s=0.0, latency=float("nan"), change="-"):
    err = np.linalg.norm(pred - FUT_VAL, axis=2)
    r = {"run": name, "change": change, "ADE_m": err.mean(), "FDE_m": err[:, -1].mean(),
         "val_examples": len(S_val), "train_time_s": round(train_s, 1), "infer_ms_30_people": latency}
    for g, m in groups.items():
        if g != "all examples":
            r[f"FDE {g}"] = err[m, -1].mean()
    rows.append(r)
    return err


def latency_ms(model, obs, context, n=30, reps=100):
    x, _, cv = batch(S_val[:n], obs, context)
    x, cv = torch.from_numpy(x).to(device), torch.from_numpy(cv).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x, cv)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x, cv)
    return (time.perf_counter() - t0) / reps * 1000


errors = {"constant velocity": evaluate("constant velocity (predictive brain)", CV_VAL)}
runs = []
if args.runs == "all":
    runs.append(("LSTM v1 (homework run 1)", dict(obs=8, context=False, residual=False, hidden=64, epochs=5,
                                                 lr=1e-3, max_train=200_000), "homework configuration"))
    runs.append(("LSTM v2 without context", dict(obs=args.obs, context=False, residual=True, hidden=args.hidden,
                                                epochs=args.epochs, lr=args.lr),
                 "correction to const. velocity, all data, longer training"))
runs.append(("LSTM v2 with context", dict(obs=args.obs, context=True, residual=True, hidden=args.hidden,
                                         epochs=args.epochs, lr=args.lr),
             "+ context: kerb, crossing + light, cars, nearby people"))

best_model = None
for name, cfg, change in runs:
    model, hist, secs = train(name, **cfg)
    pred = predict(model, S_val, cfg["obs"], cfg["context"])
    errors[name] = evaluate(name, pred, secs, latency_ms(model, cfg["obs"], cfg["context"]), change)
    curves[name] = hist
    if cfg["context"]:
        best_model = (model, cfg)

# ======================================================================= 4. report + save
table = pd.DataFrame(rows)
table.to_csv(os.path.join(args.out, "results.csv"), index=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\nRESULTS on the validation split (lower is better):")
print(table[["run", "ADE_m", "FDE_m", "val_examples", "train_time_s", "infer_ms_30_people"]].round(3).to_string(index=False))
print("\nFDE (error 3.2 s ahead, m) by kind of situation:")
hard = pd.DataFrame({g: [errors[n][m, -1].mean() for n in errors] for g, m in groups.items()}, index=list(errors))
hard.loc["examples"] = [int(m.sum()) for m in groups.values()]
hard.T.round(3).to_csv(os.path.join(args.out, "hard_cases.csv"))
print(hard.T.round(3).to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in curves.items():
        h = np.array(h)
        ax[0].plot(h[:, 0], h[:, 1], "o-", label=name)
        ax[1].plot(h[:, 0], h[:, 2], "o-", label=name)
    ax[1].axhline(rows[0]["ADE_m"], color="k", ls="--", label="constant velocity")
    ax[0].set_title("training loss"); ax[1].set_title("validation ADE (m)")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "curves.png"), dpi=110)
except Exception as e:                                   # plots are optional
    print("(no plot:", e, ")")

if best_model is not None:
    model, cfg = best_model
    path = os.path.join(args.out, "lstm_v2.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg, "obs": cfg["obs"], "pred": PRED, "dt": DT,
                "scale": SCALE, "ctx_mean": CTX_MEAN, "ctx_std": CTX_STD, "feature_names": lf.NAMES,
                "hidden": cfg["hidden"], "n_in": 2 + lf.N_FEATURES}, path)
    print(f"\nSaved the model for the car: {path}")
print(f"Saved: {args.out}/results.csv, hard_cases.csv, curves.png")
