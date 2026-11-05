"""brains.py - the robot's "brains": where will each person be?

Every brain gets the same input - the perception TRACKS (the last few seen
positions of each person, with times) - and gives the same output: for each
person, a list of times and the (x, y) where it thinks they will be then.
The planner (planner.py) is shared by all brains, so any difference in how
the car drives comes only from these predictions.

    reactive    - people stay where they were last seen (no prediction)
    predictive  - people keep walking at their current velocity for
                  PREDICTION_HORIZON seconds (constant-velocity model)
    lstm        - the trained LSTM v2 (train_lstm.py): 3.2 s ahead, using the person's
                  last 2.8 s plus context (kerb, crossings and lights, cars, nearby people)

After a brain's last time the planner assumes the person keeps the last predicted motion
(for the reactive brain: stays put).
"""
import math

import numpy as np

import config


class Prediction:
    """Where one person will be: times (K,) from now and positions (K, 2)."""

    __slots__ = ("id", "radius", "times", "xy", "velocity")

    def __init__(self, pid, radius, times, xy, velocity=(0.0, 0.0)):
        self.id, self.radius = pid, radius
        self.times = np.asarray(times, dtype=float)
        self.xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self.velocity = velocity

    def at(self, t):
        """Positions at the times t (array): interpolated between the predicted
        points; after the last one the person keeps the last predicted motion."""
        t = np.asarray(t, dtype=float)
        if len(self.times) == 1:
            return np.repeat(self.xy, len(t), axis=0)
        x = np.interp(t, self.times, self.xy[:, 0])
        y = np.interp(t, self.times, self.xy[:, 1])
        later = t > self.times[-1]
        if later.any():
            v_end = (self.xy[-1] - self.xy[-2]) / (self.times[-1] - self.times[-2])
            extra = t[later] - self.times[-1]
            x[later] = self.xy[-1, 0] + v_end[0] * extra
            y[later] = self.xy[-1, 1] + v_end[1] * extra
        return np.stack([x, y], axis=1)


def estimate_velocity(track, window=None):
    """Velocity (vx, vy): the Kalman filter's estimate when the track has one,
    otherwise a straight-line fit over the last `window` seconds of history.
    (0, 0) if there is not enough history yet."""
    v = getattr(track, "velocity", None)          # Kalman filter velocity (perception.py), if there is one
    if v is not None:
        return (0.0, 0.0) if math.hypot(*v) < config.VELOCITY_STILL else v
    window = config.VELOCITY_WINDOW if window is None else window
    hist = list(track.history)
    t_last = hist[-1][0]
    pts = [h for h in hist if t_last - h[0] <= window + 1e-9]
    if len(pts) < 3:
        return 0.0, 0.0
    t = np.array([h[0] for h in pts]) - t_last
    xy = np.array([(h[1], h[2]) for h in pts])
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom < 1e-9:
        return 0.0, 0.0
    vx = float((tc * (xy[:, 0] - xy[:, 0].mean())).sum() / denom)
    vy = float((tc * (xy[:, 1] - xy[:, 1].mean())).sum() / denom)
    if math.hypot(vx, vy) < config.VELOCITY_STILL:
        return 0.0, 0.0                       # standing still (ignore tiny jitter)
    return vx, vy


class Brain:
    name = "base"
    horizon = 0.0

    def predict(self, tracks, now):
        raise NotImplementedError


class ReactiveBrain(Brain):
    """Only uses where people ARE (last seen position)."""
    name = "reactive"

    def predict(self, tracks, now):
        return [Prediction(tr.id, tr.radius, [0.0], [tr.position]) for tr in tracks.values()]


class PredictiveBrain(Brain):
    """Constant-velocity prediction over PREDICTION_HORIZON seconds."""
    name = "predictive"

    def __init__(self, horizon=None, step=None):
        self.horizon = config.PREDICTION_HORIZON if horizon is None else horizon
        self.step = config.PREDICTION_STEP if step is None else step
        self.times = np.arange(0.0, self.horizon + 1e-9, self.step)

    def predict(self, tracks, now):
        out = []
        for tr in tracks.values():
            vx, vy = estimate_velocity(tr)
            t_seen, x, y = tr.history[-1]
            lag = max(0.0, now - t_seen)              # not seen for a moment: carry them on
            x0, y0 = x + vx * lag, y + vy * lag
            xy = np.stack([x0 + vx * self.times, y0 + vy * self.times], axis=1)
            out.append(Prediction(tr.id, tr.radius, self.times, xy, (vx, vy)))
        return out


class LSTMBrain(Brain):
    """The trained LSTM (config.LSTM_MODEL: v3 = train_lstm.py --noisy -> results/lstm_v3/lstm_v3.pt).

    For every person the car sees, it takes their last 2.8 s (resampled to one point
    every 0.4 s, like the training data), adds the same context the model was trained
    with (lstm_features.py: kerb, nearest crossing and its light, nearest car and the
    robot car, the 3 nearest people the car can see) and predicts their next 3.2 s as a
    correction to constant velocity. People seen for less than 2.8 s get the
    constant-velocity prediction until there is enough history.
    """
    name = "lstm"

    def __init__(self, sim=None, model_path=None):
        import torch
        import torch.nn as nn
        import lstm_features as lf
        from roadmap import RoadMap
        if sim is None:
            raise ValueError("The LSTM brain needs the simulation (make_brain('lstm', sim)).")
        self.torch, self.lf, self.sim = torch, lf, sim
        path = model_path or config.LSTM_MODEL
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.obs, self.pred, self.dt = int(ck["obs"]), int(ck["pred"]), float(ck["dt"])
        self.scale = float(ck["scale"])
        self.ctx_mean = np.asarray(ck["ctx_mean"], dtype=np.float32)
        self.ctx_std = np.asarray(ck["ctx_std"], dtype=np.float32)
        pred, scale = self.pred, self.scale

        class Predictor(nn.Module):                         # the same network as in train_lstm.py
            def __init__(self, n_in, hidden):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, pred * 2))

            def forward(self, x, cv):
                _, (h, _) = self.lstm(x)
                return cv + torch.cumsum(self.head(h[-1]).view(-1, pred, 2), dim=1) * scale

        self.model = Predictor(int(ck["n_in"]), int(ck["hidden"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.roadmap = RoadMap(sim.world.layout)           # own copy: its lights can be "looked up" in the past
        self.horizon = self.pred * self.dt
        self.times = np.arange(0, self.pred + 1) * self.dt
        self.fallback = PredictiveBrain(horizon=self.horizon, step=self.dt)
        self.frames = []                                     # what the car saw at each recent step

    def reset(self):
        self.frames = []

    # ---------- what the car saw a moment ago (for the people / cars context) ----------
    def _remember(self, tracks, now):
        sim = self.sim
        ids = list(tracks)
        p_xy = np.array([tracks[i].position for i in ids], dtype=float).reshape(-1, 2)
        p_v = np.array([estimate_velocity(tracks[i]) for i in ids], dtype=float).reshape(-1, 2)
        cars = [v for v in (sim.traffic.vehicles if sim.traffic else []) if v.active]
        c_xy = [(v.x, v.y) for v in cars]
        c_v = [(v.v * math.cos(v.heading), v.v * math.sin(v.heading)) for v in cars]
        r = sim.robot
        c_xy.append((r.x, r.y))
        c_v.append((r.vx, r.vy))
        robot_mask = np.zeros(len(c_xy), dtype=bool)
        robot_mask[-1] = True
        self.frames.append((now, ids, p_xy, p_v, np.array(c_xy, float), np.array(c_v, float), robot_mask))
        horizon = self.obs * self.dt + 0.5
        while self.frames and now - self.frames[0][0] > horizon:
            self.frames.pop(0)

    def _frame_at(self, t):
        return min(self.frames, key=lambda f: abs(f[0] - t))

    def predict(self, tracks, now):
        self._remember(tracks, now)
        span = (self.obs - 1) * self.dt
        ready, out = [], []
        for tr in tracks.values():
            hist = tr.history
            if hist[-1][0] - hist[0][0] >= span - 1e-6:
                ready.append(tr)
            else:                                            # not enough history yet
                out += self.fallback.predict({tr.id: tr}, now)
        if not ready:
            return out
        lf = self.lf
        X, CV, last_pts, lags, all_pts, all_t = [], [], [], [], [], []
        ctx_rows = []
        for tr in ready:
            h = np.array(tr.history, dtype=float)
            t_last = h[-1, 0]
            tk = t_last - (self.obs - 1 - np.arange(self.obs)) * self.dt          # 0.4 s apart, ending now
            pts = np.stack([np.interp(tk, h[:, 0], h[:, 1]), np.interp(tk, h[:, 0], h[:, 2])], axis=1)
            steps = np.diff(pts, axis=0)                                           # (obs-1, 2)
            X.append(steps / self.scale)
            CV.append(steps[-1][None, :] * np.arange(1, self.pred + 1)[:, None])
            last_pts.append(pts[-1])
            lags.append(max(0.0, now - t_last))
            all_pts.append(pts[1:]); all_t.append(tk[1:])
            # people / cars context at each of those moments
            rows = []
            for k in range(1, self.obs):
                f_t, ids, p_xy, p_v, c_xy, c_v, rmask = self._frame_at(tk[k])
                others = [j for j, pid in enumerate(ids) if pid != tr.id]
                me_v = steps[k - 1] / self.dt
                P = np.vstack([pts[k][None, :], p_xy[others]]) if others else pts[k][None, :]
                V = np.vstack([me_v[None, :], p_v[others]]) if others else me_v[None, :]
                rows.append(lf.frame_features(P, V, c_xy, c_v, rmask)[0])
            ctx_rows.append(np.array(rows))
        n = len(ready)
        mapf = lf.map_features(self.roadmap, np.concatenate(all_pts), np.concatenate(all_t))
        ctx = np.concatenate([mapf.reshape(n, self.obs - 1, 10), np.stack(ctx_rows)], axis=2)
        ctx = (ctx - self.ctx_mean) / self.ctx_std
        x = np.concatenate([np.stack(X), ctx], axis=2).astype(np.float32)
        cv = np.stack(CV).astype(np.float32)
        with self.torch.no_grad():
            offsets = self.model(self.torch.from_numpy(x), self.torch.from_numpy(cv)).numpy()   # (n, pred, 2)
        for tr, off, p0, lag in zip(ready, offsets, last_pts, lags):
            xy = np.vstack([p0[None, :], p0[None, :] + off])
            vel = tuple((off[0] / self.dt).tolist())
            out.append(Prediction(tr.id, tr.radius, self.times - lag, xy, vel))
        return out


BRAINS = {"reactive": ReactiveBrain, "predictive": PredictiveBrain, "lstm": LSTMBrain}


def make_brain(name, sim=None):
    return BRAINS[name](sim) if name == "lstm" else BRAINS[name]()
