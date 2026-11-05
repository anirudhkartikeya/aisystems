"""tracking.py - a Kalman filter that smooths each person's track.

Why: a real camera + detector (YOLO) gives positions that jump around by a few
centimetres from frame to frame. The brains turn the last few positions into a
walking speed and direction, so small jumps become big errors in where they think
the person is going. The filter keeps a running best guess of each person's
position AND velocity, and blends every new measurement into it according to how
much it trusts the measurement versus its own guess.

Model (one per person), state = [x, y, vx, vy]:
  1. Predict: the person keeps walking at the same velocity for dt seconds.
     Uncertainty grows, because people speed up, slow down and turn
     (KALMAN_ACCEL = how hard they can change velocity, m/s^2).
  2. Update:  a new measurement (x, y) arrives. The Kalman gain decides how far to
     move towards it: a noisy sensor or a confident guess -> small step; a precise
     sensor or an unsure guess -> big step.

How noisy is the sensor? SensorNoise LEARNS it while driving (KALMAN_LEARN_NOISE):
the measurements' surprise (innovation) is partly the person's own movement (which
the filter already expects) and partly sensor noise; what is left over is averaged
over all people. So with exact positions the filter barely smooths (no lag), and
with a noisy camera it smooths more - but never as if the camera were worse than
KALMAN_NOISE_CEIL: heavier smoothing makes the car notice too late that someone
has started to walk out (safety over smoothness). KALMAN_MEAS_STD is the starting guess (or the
fixed value when learning is off).

Missed detections need nothing special: no measurement means no update, and the
next prediction just covers a longer dt. Switch it off with TRACK_FILTER = "none"
in config.py.
"""
import numpy as np

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


class SensorNoise:
    """The camera's position noise (variance, m^2), shared by all tracks and learned online."""

    def __init__(self, start_std, floor_std=0.02, rate=0.03, learn=True, ceil_std=None):
        self.var = start_std ** 2
        self.floor = floor_std ** 2
        self.ceil = ceil_std ** 2 if ceil_std else None
        self.rate = rate
        self.learn = learn

    def observe(self, innov, HPH):
        """innov: measurement minus prediction (2,); HPH: the filter's own position uncertainty (2, 2)."""
        if not self.learn:
            return
        left_over = float(innov @ innov) / 2 - float(np.trace(HPH)) / 2      # per axis
        self.var = (1 - self.rate) * self.var + self.rate * max(self.floor, left_over)
        if self.ceil is not None:
            self.var = min(self.var, self.ceil)

    @property
    def std(self):
        return float(np.sqrt(self.var))


class KalmanCV:
    """Constant-velocity Kalman filter in 2-D."""

    def __init__(self, t, x, y, meas_var, accel_std, speed_std=1.5):
        self.t = float(t)
        self.s = np.array([x, y, 0.0, 0.0])                                  # state
        self.P = np.diag([meas_var, meas_var, speed_std ** 2, speed_std ** 2])
        self.q = accel_std ** 2                                               # process noise
        self.n = 1                                                            # measurements so far

    def predict(self, t):
        dt = float(t) - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # white-noise acceleration: position and velocity errors grow together
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = self.q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)

    def update(self, t, x, y, noise):
        """noise: a SensorNoise (learns from this measurement first, then gives R)."""
        self.predict(t)
        innov = np.array([x, y]) - H @ self.s
        HPH = H @ self.P @ H.T
        noise.observe(innov, HPH)
        S = HPH + np.eye(2) * noise.var
        K = self.P @ H.T @ np.linalg.inv(S)                                   # Kalman gain
        self.s = self.s + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P
        self.n += 1

    @property
    def position(self):
        return float(self.s[0]), float(self.s[1])

    @property
    def velocity(self):
        return float(self.s[2]), float(self.s[3])


def kalman_batch(z, dt, meas_var, accel_std, speed_std=1.5):
    """The same filter as KalmanCV (noise not learned), run on many tracks at once.
    z: (B, T, 2) measurements every dt seconds; meas_var: (B,) noise variance per track.
    Returns the filtered positions (B, T, 2). Used by train_lstm.py --noisy to give the
    LSTM exactly the kind of tracks the car's perception gives it."""
    z = np.asarray(z, float)
    B, T, _ = z.shape
    meas_var = np.broadcast_to(np.asarray(meas_var, float), (B,))
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
    Q = accel_std ** 2 * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
    s = np.zeros((B, 4))
    s[:, :2] = z[:, 0]
    P = np.zeros((B, 4, 4))
    P[:, 0, 0] = P[:, 1, 1] = meas_var
    P[:, 2, 2] = P[:, 3, 3] = speed_std ** 2
    out = np.empty((B, T, 2))
    out[:, 0] = z[:, 0]
    eye2 = np.eye(2)
    for k in range(1, T):
        s = s @ F.T
        P = F @ P @ F.T + Q
        S = P[:, :2, :2] + meas_var[:, None, None] * eye2
        K = P[:, :, :2] @ np.linalg.inv(S)                                    # (B, 4, 2)
        s = s + np.einsum("bij,bj->bi", K, z[:, k] - s[:, :2])
        P = P - K @ P[:, :2, :]
        out[:, k] = s[:, :2]
    return out
