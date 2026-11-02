"""explain_lstm.py - which inputs does the LSTM v2 rely on? (input importance + local explanations)

    python explain_lstm.py                       # 20,000 validation examples, 3 repeats (a few minutes, CPU)
    python explain_lstm.py --examples 5000       # quicker

Uses the trained model (results/lstm_v2/lstm_v2.pt) and the LSTM dataset's VALIDATION split, exactly
as train_lstm.py built it (needs trajectories/features_v2_0_499.npz, made by train_lstm.py).

1. Permutation importance (global): shuffle ONE input between examples (so it no longer belongs
   to that person) and measure how much worse the 3.2 s prediction gets (FDE, metres). A big
   increase = the model relies on that input. Done for each of the 33 context inputs, for the
   movement history, and for the 6 input groups (road, crossing, nearest car, robot car,
   nearby people, movement). Repeated with different shuffles -> mean +/- spread.
2. Which past moments matter: the same, shuffling everything the model saw at one past step.
3. By situation: group importance separately for the hard cases (person changes what they
   do), people going onto the road, a car within 10 m, children, elderly.
4. Local explanations (LIME-style) for single hard cases: turn input groups on/off at random
   (off = a neutral value), refit a weighted linear model, and read how many metres each
   group moves the prediction (the LSTM's correction to constant velocity). Two cases where
   the LSTM beat constant velocity the most, two where it did worst.

Note: constant velocity itself (from the last two points) is always added to the LSTM's
correction, so shuffling the movement input only removes what the LSTM learned from it.
Writes results/lstm_v2/explain/: *.csv, importance.png, steps.png, situations.png, cases.png, summary.txt
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import lstm_features as lf

GROUPS = {
    "road (kerb, direction to road, on road)": list(range(0, 4)),
    "crossing (where, light, walk signal, on it)": list(range(4, 10)),
    "nearest car": list(range(10, 15)),
    "robot car": list(range(15, 20)),
    "nearby people (3 nearest + crowd)": list(range(20, 33)),
}
MOTION = "movement history (own past steps)"
WIN_OBS, PRED = 12, 8


def load_predictor(path):
    """The trained network from the checkpoint (same as brains.LSTMBrain) -> (meta, fn(x, cv) -> pred)."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pred, scale = int(ck["pred"]), float(ck["scale"])

    class Predictor(nn.Module):
        def __init__(self, n_in, hidden):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, pred * 2))

        def forward(self, x, cv):
            _, (h, _) = self.lstm(x)
            return cv + torch.cumsum(self.head(h[-1]).view(-1, pred, 2), dim=1) * scale

    model = Predictor(int(ck["n_in"]), int(ck["hidden"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def fn(x, cv, bs=8192):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), bs):
                out.append(model(torch.from_numpy(x[i:i + bs]), torch.from_numpy(cv[i:i + bs])).numpy())
        return np.concatenate(out)

    meta = {"obs": int(ck["obs"]), "pred": pred, "scale": scale,
            "ctx_mean": np.asarray(ck["ctx_mean"], np.float32), "ctx_std": np.asarray(ck["ctx_std"], np.float32),
            "names": list(ck.get("feature_names", lf.NAMES))}
    return meta, fn


def main(argv=None, predictor=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/lstm_v2/lstm_v2.pt")
    ap.add_argument("--data", default="trajectories")
    ap.add_argument("--prefix", default="0_499")
    ap.add_argument("--examples", type=int, default=20000, help="validation examples used (random, fixed seed)")
    ap.add_argument("--repeats", type=int, default=3, help="shuffles per input")
    ap.add_argument("--samples", type=int, default=400, help="on/off samples per local explanation")
    ap.add_argument("--out", default="results/lstm_v2/explain")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)

    # ---------------- data: the validation examples, as in train_lstm.py ----------------
    peds = pd.read_csv(os.path.join(args.data, f"peds_{args.prefix}.csv"))
    peds = peds.sort_values(["episode", "ped_id", "t"]).reset_index(drop=True)
    cache = os.path.join(args.data, f"features_v2_{args.prefix}.npz")
    z = np.load(cache)
    if int(z["n"]) != len(peds):
        raise SystemExit(f"{cache} does not match the CSV - run train_lstm.py once to rebuild it")
    FEAT = z["F"]
    XY = peds[["x", "y"]].to_numpy(np.float32)
    meta, fn = predictor if predictor is not None else load_predictor(args.model)
    obs, scale = meta["obs"], meta["scale"]
    win = WIN_OBS + PRED
    part = peds[peds["split"] == "val"]
    starts = []
    for _, idx in part.groupby(["episode", "ped_id"]).indices.items():
        idx = part.index.to_numpy()[idx]
        if len(idx) >= win:
            starts.append(idx[0] + np.arange(0, len(idx) - win + 1, 4))
    S = np.concatenate(starts)
    n_all = len(S)
    if args.examples and len(S) > args.examples:
        S = np.sort(rng.choice(S, args.examples, replace=False))
    w = XY[S[:, None] + np.arange(win)]                                       # (B, 20, 2)
    hist = w[:, WIN_OBS - obs:WIN_OBS]
    steps = np.diff(hist, axis=1) / scale
    ctx = FEAT[S[:, None] + np.arange(WIN_OBS - obs + 1, WIN_OBS)]
    X = np.concatenate([steps, (ctx - meta["ctx_mean"]) / meta["ctx_std"]], axis=2).astype(np.float32)
    last = w[:, WIN_OBS - 1:WIN_OBS]
    FUT = (w[:, WIN_OBS:] - last).astype(np.float32)
    CV = ((w[:, WIN_OBS - 1] - w[:, WIN_OBS - 2])[:, None, :] * np.arange(1, PRED + 1)[None, :, None]).astype(np.float32)
    names = meta["names"]
    print(f"{len(S):,} of {n_all:,} validation examples | model {args.model} | inputs: 2 movement + {len(names)} context")

    def fde(pred, sel=None):
        e = np.linalg.norm(pred[:, -1] - FUT[:, -1], axis=1)
        return float(e.mean()) if sel is None else float(e[sel].mean())

    base_pred = fn(X, CV)
    base = fde(base_pred)
    cv_fde = fde(CV)
    print(f"FDE with all inputs: {base:.3f} m (constant velocity alone: {cv_fde:.3f} m)")

    # situations (same definitions as train_lstm.py)
    mode = peds["mode"].to_numpy()
    kind = peds["kind"].to_numpy()
    road = peds["on_road"].to_numpy()
    last_i = S + WIN_OBS - 1
    car_d = np.hypot(*(FEAT[last_i][:, [10, 11]] * 10).T)
    situations = {
        "all": np.ones(len(S), bool),
        "changes": mode[last_i] != mode[S + win - 1],                      # stops / starts crossing / turns back
        "onto road": road[S[:, None] + np.arange(WIN_OBS, win)].max(1) == 1,  # on the road in the next 3.2 s
        "car<10m": (FEAT[last_i, names.index("car_present")] > 0) & (car_d < 10.0),
        "children": kind[last_i] == "child",
        "elderly": kind[last_i] == "elderly",
    }

    def permuted(chans=None, step=None, seed=0):
        p = np.random.default_rng(seed).permutation(len(X))
        x2 = X.copy()
        if step is None:
            x2[:, :, chans] = X[p][:, :, chans]
        else:
            x2[:, step, :] = X[p][:, step, :]
        return fn(x2, CV)

    # ---------------- 1. single inputs and groups ----------------
    items = [(MOTION, [0, 1], "movement")] + [(n, [2 + i], g) for g, idx in GROUPS.items() for i, n in
                                                  ((i, names[i]) for i in idx)]
    group_items = [(MOTION, [0, 1])] + [(g, [2 + i for i in idx]) for g, idx in GROUPS.items()]
    rows = []
    for k, (name, chans, grp) in enumerate(items):
        d = [fde(permuted(chans, seed=r)) - base for r in range(args.repeats)]
        rows.append({"input": name, "group": grp.split(" (")[0], "FDE_increase_m": np.mean(d),
                     "spread_m": np.std(d), "FDE_increase_pct": 100 * np.mean(d) / base})
        if k % 8 == 0:
            print(f"  input {k + 1}/{len(items)} ({time.time() - t0:.0f} s)")
    feat = pd.DataFrame(rows).sort_values("FDE_increase_m", ascending=False)
    feat.round(4).to_csv(os.path.join(args.out, "feature_importance.csv"), index=False)
    grows = []
    for name, chans in group_items:
        preds = [permuted(chans, seed=100 + r) for r in range(args.repeats)]
        row = {"group": name}
        for sname, sel in situations.items():
            b = fde(base_pred, sel)
            d = [fde(p, sel) - b for p in preds]
            row[sname] = np.mean(d)
            if sname == "all":
                row["spread_m"] = np.std(d)
        grows.append(row)
    groups = pd.DataFrame(grows).sort_values("all", ascending=False)
    groups.round(4).to_csv(os.path.join(args.out, "group_importance.csv"), index=False)

    # ---------------- 2. past steps ----------------
    srows = []
    for k in range(X.shape[1]):
        d = [fde(permuted(step=k, seed=200 + r)) - base for r in range(args.repeats)]
        ago = (X.shape[1] - 1 - k) * 0.4
        srows.append({"step": k + 1, "seconds_before_now": round(ago, 1), "FDE_increase_m": np.mean(d),
                      "spread_m": np.std(d)})
    steps_df = pd.DataFrame(srows)
    steps_df.round(4).to_csv(os.path.join(args.out, "step_importance.csv"), index=False)

    # ---------------- 4. local explanations of single hard cases ----------------
    e_lstm = np.linalg.norm(base_pred[:, -1] - FUT[:, -1], axis=1)
    e_cv = np.linalg.norm(CV[:, -1] - FUT[:, -1], axis=1)
    hard = np.where(situations["changes"])[0]
    gain = e_cv[hard] - e_lstm[hard]
    picks = [("LSTM much better than constant velocity", i) for i in hard[np.argsort(-gain)[:2]]] + \
            [("LSTM worse than constant velocity", i) for i in hard[np.argsort(gain)[:2]]]
    all_groups = [(MOTION, [0, 1])] + [(g, [2 + i for i in idx]) for g, idx in GROUPS.items()]
    G = len(all_groups)
    cases = []
    for label, i in picks:
        x0 = X[i:i + 1]
        neutral = x0.copy()
        neutral[0, :, 2:] = 0.0                                        # context at its training average
        neutral[0, :, :2] = x0[0, :, :2].mean(0)                       # movement: steady, no changes
        Z = rng.integers(0, 2, size=(args.samples, G))
        Z[0] = 1
        xs = np.repeat(x0, args.samples, axis=0)
        for g, (_, chans) in enumerate(all_groups):
            off = Z[:, g] == 0
            xs[np.ix_(off, np.arange(xs.shape[1]), chans)] = neutral[0][:, chans]
        out = fn(xs, np.repeat(CV[i:i + 1], args.samples, axis=0))
        corr_full = out[0, -1] - CV[i, -1]
        u = corr_full / max(np.linalg.norm(corr_full), 1e-6)
        y = (out[:, -1] - CV[i, -1]) @ u                               # metres of correction along its direction
        dist = np.sqrt((1 - Z).sum(1) / G)
        wts = np.exp(-dist ** 2 / 0.75 ** 2)
        A = np.hstack([Z, np.ones((len(Z), 1))])
        Wd = np.sqrt(wts)[:, None]
        coef = np.linalg.solve((A * Wd).T @ (A * Wd) + 1e-3 * np.eye(G + 1), (A * Wd).T @ (y[:, None] * Wd[:, 0:1]))[:, 0]
        fit = A @ coef
        r2 = 1 - np.sum(wts * (y - fit) ** 2) / max(np.sum(wts * (y - np.average(y, weights=wts)) ** 2), 1e-12)
        row = {"case": label, "example": int(S[i]), "kind": kind[last_i[i]], "mode_then": f"{mode[last_i[i]]} -> {mode[S[i] + win - 1]}",
               "FDE_lstm_m": e_lstm[i], "FDE_cv_m": e_cv[i], "lstm_correction_m": float(np.linalg.norm(corr_full)),
               "fit_R2": r2}
        for g, (gname, _) in enumerate(all_groups):
            row[gname.split(" (")[0]] = coef[g]
        cases.append((row, w[i], base_pred[i] + last[i], CV[i] + last[i]))
    case_df = pd.DataFrame([c[0] for c in cases])
    case_df.round(3).to_csv(os.path.join(args.out, "local_cases.csv"), index=False)

    # ---------------- report ----------------
    lines = [f"LSTM v2 input importance on {len(S):,} validation examples (of {n_all:,}), {args.repeats} shuffles each",
             f"FDE with all inputs {base:.3f} m; constant velocity {cv_fde:.3f} m", "",
             "Input groups: FDE increase when shuffled (m), overall and by situation",
             "  (changes = person stops / starts crossing / turns back; onto road = on the road in the next 3.2 s;",
             "   car<10m = a car within 10 m; examples: " + ", ".join(f"{k} {int(v.sum()):,}" for k, v in situations.items()) + ")"]
    lines.append(groups.round(3).to_string(index=False))
    lines += ["", "Top 12 single inputs:", feat.head(12).round(4).to_string(index=False), "",
              "Past steps (shuffle everything seen at that moment):", steps_df.round(4).to_string(index=False), "",
              "Local explanations (metres each group adds to the LSTM's correction; R2 = how well the linear "
              "explanation fits):", case_df.round(3).to_string(index=False)]
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = feat.head(15)[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        ax[0].barh(top["input"], top["FDE_increase_m"], xerr=top["spread_m"], color="#3a6ea5")
        ax[0].set_title("Single inputs: FDE increase when shuffled (m)")
        gg = groups[::-1]
        ax[1].barh([g.split(" (")[0] for g in gg["group"]], gg["all"], xerr=gg["spread_m"],
                   color="#5b8c5a")
        ax[1].set_title(f"Input groups (all inputs: FDE {base:.2f} m)")
        for a in ax:
            a.set_xlabel("increase in 3.2 s error (m)")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "importance.png"), dpi=120); plt.close()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([f"-{s:.1f}s" for s in steps_df["seconds_before_now"]], steps_df["FDE_increase_m"],
               yerr=steps_df["spread_m"], color="#a5673a")
        ax.set_xlabel("moment in the history (0 = the latest)"); ax.set_ylabel("FDE increase (m)")
        ax.set_title("Which past moments the LSTM relies on")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "steps.png"), dpi=120); plt.close()

        cols = list(situations)
        M = groups[cols].to_numpy()
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(M, cmap="Oranges")
        ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right")
        ax.set_yticks(range(len(groups)), [g.split(" (")[0] for g in groups["group"]])
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("FDE increase (m) when a group is shuffled, by situation")
        plt.colorbar(im); plt.tight_layout(); plt.savefig(os.path.join(args.out, "situations.png"), dpi=120); plt.close()

        fig, ax = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 8))
        for j, (row, ww, pl, pc) in enumerate(cases):
            a = ax[0, j]
            a.plot(ww[:WIN_OBS, 0], ww[:WIN_OBS, 1], "k.-", label="seen (past)")
            a.plot(ww[WIN_OBS - 1:, 0], ww[WIN_OBS - 1:, 1], "g.-", label="what really happened")
            a.plot(np.r_[ww[WIN_OBS - 1, 0], pc[:, 0]], np.r_[ww[WIN_OBS - 1, 1], pc[:, 1]], "--", color="gray",
                   label="constant velocity")
            a.plot(np.r_[ww[WIN_OBS - 1, 0], pl[:, 0]], np.r_[ww[WIN_OBS - 1, 1], pl[:, 1]], "r.-", label="LSTM")
            a.set_aspect("equal", "datalim"); a.set_title(f"{row['case']}\n{row['kind']}, {row['mode_then']}", fontsize=9)
            if j == 0:
                a.legend(fontsize=7)
            b = ax[1, j]
            gn = [g.split(" (")[0] for g, _ in all_groups]
            vals = [row[g] for g in gn]
            b.barh(gn, vals, color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
            b.set_title(f"metres added to the correction (R2 {row['fit_R2']:.2f})", fontsize=9)
            b.axvline(0, color="k", lw=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "cases.png"), dpi=110); plt.close()
    except Exception as e:                             # plots are optional
        print("(no plots:", e, ")")
    print(f"\nSaved {args.out}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
