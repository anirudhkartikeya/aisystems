"""brains.py - the robot's "brains": where will each person be?

Every brain gets the same input - the perception TRACKS (the last few seen
positions of each person, with times) - and gives the same output: for each
person, a list of times and the (x, y) where it thinks they will be then.
The planner (planner.py) is shared by all brains, so any difference in how
the car drives comes only from these predictions.

    reactive    - people stay where they were last seen (no prediction)
    predictive  - people keep walking at their current velocity for
                  PREDICTION_HORIZON seconds (constant-velocity model)
    lstm        - Phase 2: a trained LSTM predicts the next positions (slot ready)

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
    """Velocity (vx, vy) from the recent history: a straight-line fit over the
    last `window` seconds. (0, 0) if there is not enough history yet."""
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
    """Phase 2 slot. To plug the trained LSTM in:
      1. load the model in __init__ (e.g. torch.load("lstm.pt")),
      2. in predict: for each track, resample its history to the time step the
         LSTM was trained on (make_data.py saves every 0.4 s), feed it, and
         turn the predicted offsets into positions,
      3. return Prediction(track.id, track.radius, times, xy) like the others.
    Nothing else in the project has to change."""
    name = "lstm"

    def __init__(self, model_path=None):
        raise NotImplementedError("The LSTM brain comes in Phase 2 - train the LSTM first.")

    def predict(self, tracks, now):
        raise NotImplementedError


BRAINS = {"reactive": ReactiveBrain, "predictive": PredictiveBrain, "lstm": LSTMBrain}


def make_brain(name):
    return BRAINS[name]()
