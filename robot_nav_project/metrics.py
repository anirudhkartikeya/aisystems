"""metrics.py - the numbers we compare the brains on, recorded during a run.

    rec = EpisodeMetrics(sim, driver)       # driver = planner.Driver (or None for the placeholder)
    ... after every sim.step(): rec.step()
    row = rec.finish(info)                  # one dict of numbers for this run

Safety      collisions with people / cars, minimum gap, near misses, minimum time-to-collision
Efficiency  success (goal, no collision), time to goal, path length, time standing still
Comfort     hard-braking events, strongest braking, steering jerk
Decisions   correct action for each scripted person, reaction distance
Prediction  ADE / FDE: how far the brain's predictions were from where people really went
            (the same check for every brain: the reactive brain "predicts" people stay put)
"""
import math

import numpy as np

import config


class EpisodeMetrics:
    def __init__(self, sim, driver=None):
        self.sim, self.driver = sim, driver
        dt = config.CONTROL_DT
        self.dt = dt
        self.prev_speed = 0.0
        self.prev_heading = sim.robot.heading
        self.prev_yaw_rate = 0.0
        self.prev_gap = {}
        self.touching = set()
        self.near = set()
        self.ped_collisions = 0
        self.near_misses = 0
        self.min_ttc = math.inf
        self.hard_brakes = 0
        self._braking = False
        self.max_decel = 0.0
        self.jerk_sum, self.jerk_n = 0.0, 0
        self.time_stopped = 0.0
        self.vehicle_collisions = 0
        self._veh_touch = False
        self.reaction = {}                 # crosser id -> gap (m) when the car first reacted to them
        self.crosser_ids = {log["ped_id"]: log["name"] for log in sim.crosser_log}
        self.preds = []                    # (t0, Prediction) samples for ADE/FDE
        self.actual = {}                   # ped id -> list of (t, x, y)
        self._k = 0

    def step(self):
        sim, dt = self.sim, self.dt
        r = sim.robot
        speed = math.hypot(r.vx, r.vy)
        # comfort
        if sim.done:                                  # the run is over (the car is parked)
            return
        acc = (speed - self.prev_speed) / dt
        if acc < -config.HARD_BRAKE and not self._braking:
            self.hard_brakes += 1
        self._braking = acc < -config.HARD_BRAKE
        self.max_decel = max(self.max_decel, -acc)
        if speed > 0.3 and self.prev_speed > 0.3:
            yaw_rate = ((r.heading - self.prev_heading + math.pi) % (2 * math.pi) - math.pi) / dt
            self.jerk_sum += abs(yaw_rate - self.prev_yaw_rate) / dt
            self.jerk_n += 1
            self.prev_yaw_rate = yaw_rate
        self.prev_heading = r.heading
        if speed < 0.1 and not sim.done:
            self.time_stopped += dt
        self.prev_speed = speed
        # safety: gaps to people, near misses, time to collision
        foot = r.footprint()
        for ped in sim.peds:
            if abs(ped.x - r.x) > 15 or abs(ped.y - r.y) > 15:
                self.prev_gap.pop(ped.ped_id, None)
                continue
            gap = min(math.hypot(cx - ped.x, cy - ped.y) - cr for cx, cy, cr in foot) - ped.radius
            pid = ped.ped_id
            if gap < 0:
                if pid not in self.touching:
                    self.ped_collisions += 1
                    self.touching.add(pid)
            else:
                self.touching.discard(pid)
                if gap < config.NEAR_MISS_DIST:
                    if pid not in self.near:
                        self.near_misses += 1
                        self.near.add(pid)
                elif gap > config.NEAR_MISS_DIST + 0.3:
                    self.near.discard(pid)
            prev = self.prev_gap.get(pid)
            if prev is not None and gap > 0 and speed > 0.3:
                rate = (gap - prev) / dt
                if rate < -0.1:
                    self.min_ttc = min(self.min_ttc, gap / -rate)
            self.prev_gap[pid] = gap
        # safety: other cars
        touch = sim.min_vehicle_gap < 0 and self._veh_gap_now(foot) < 0
        if touch and not self._veh_touch:
            self.vehicle_collisions += 1
        self._veh_touch = touch
        # decisions: when did the car first react to each scripted person?
        drv = self.driver
        if drv is not None:
            pid = drv.planner.reacting_to
            if pid in self.crosser_ids and pid not in self.reaction:
                ped = next(p for p in sim.peds if p.ped_id == pid)
                self.reaction[pid] = min(math.hypot(cx - ped.x, cy - ped.y) - cr for cx, cy, cr in foot)
            # prediction quality: sample the brain's predictions twice a second
            if self._k % 5 == 0:
                for pr in drv.predictions:
                    if math.hypot(pr.xy[0][0] - r.x, pr.xy[0][1] - r.y) < 20:
                        self.preds.append((sim.t, pr))
        for ped in sim.peds:
            if abs(ped.x - r.x) < 35 and abs(ped.y - r.y) < 35:
                self.actual.setdefault(ped.ped_id, []).append((sim.t, ped.x, ped.y))
        self._k += 1

    def _veh_gap_now(self, foot):
        if not self.sim.traffic:
            return math.inf
        best = math.inf
        for v in self.sim.traffic.vehicles:
            if v.active and abs(v.x - self.sim.robot.x) < 12 and abs(v.y - self.sim.robot.y) < 12:
                for ax, ay, ar in foot:
                    for bx, by, br in v.circles():
                        best = min(best, math.hypot(ax - bx, ay - by) - ar - br)
        return best

    def _prediction_errors(self, t_end):
        """ADE / FDE over the standard horizon, for every brain the same way."""
        H, st = config.PREDICTION_HORIZON, config.PREDICTION_STEP
        times = np.arange(st, H + 1e-9, st)
        ade, fde = [], []
        for t0, pr in self.preds:
            if t0 + H > t_end:
                continue
            track = self.actual.get(pr.id)
            if not track:
                continue
            tt = np.array([a[0] for a in track])
            want = t0 + times
            idx = np.searchsorted(tt, want - 1e-6)
            if np.any(idx >= len(tt)) or np.any(np.abs(tt[np.minimum(idx, len(tt) - 1)] - want) > 0.06):
                continue                                    # person left the recorded area
            real = np.array([(track[i][1], track[i][2]) for i in idx])
            err = np.hypot(*(pr.at(times) - real).T)
            ade.append(err.mean())
            fde.append(err[-1])
        return (float(np.mean(ade)) if ade else math.nan, float(np.mean(fde)) if fde else math.nan)

    def finish(self, info, brain=""):
        sim = self.sim
        crossers = info["crossers"]
        graded = [c for c in crossers if c["expected"] in ("swerve_left", "swerve_right", "stop")]
        correct = sum(1 for c in graded if c["action"] == c["expected"] and c["min_gap"] > 0)
        reacts = [self.reaction[i] for i in self.reaction]
        ade, fde = self._prediction_errors(sim.t)
        row = {
            "brain": brain, "seed": info["seed"], "result": info["result"],
            "success": int(info["result"] == "goal" and self.ped_collisions == 0 and self.vehicle_collisions == 0),
            "ped_collisions": self.ped_collisions, "vehicle_collisions": self.vehicle_collisions,
            "min_gap": round(info["min_gap"], 3), "near_misses": self.near_misses,
            "min_ttc": round(self.min_ttc, 2) if math.isfinite(self.min_ttc) else 99.0,
            "time": round(info["time"], 1), "path_length": round(info["path_length"], 1),
            "time_stopped": round(self.time_stopped, 1),
            "hard_brakes": self.hard_brakes, "max_decel": round(self.max_decel, 2),
            "steer_jerk": round(self.jerk_sum / max(1, self.jerk_n), 3),
            "correct_actions": correct, "graded_people": len(graded),
            "reaction_gap": round(float(np.mean(reacts)), 2) if reacts else math.nan,
            "ade": round(ade, 3), "fde": round(fde, 3),
        }
        people = []
        for c, log in zip(crossers, sim.crosser_log):
            c = dict(c, ped_id=log["ped_id"])
            people.append({"brain": brain, "seed": info["seed"], "name": c["name"], "expected": c["expected"],
                           "action": c["action"], "min_gap": round(c["min_gap"], 3), "behaviour": c["behaviour"],
                           "side": c["side"], "contact": round(float(c["contact"]), 2), "zebra": c["zebra"],
                           "reaction_gap": round(self.reaction[c["ped_id"]], 2) if c["ped_id"] in self.reaction
                           else math.nan})
        return row, people


SUMMARY = [  # (column, label, how to combine over runs)
    ("success", "Success rate (%)", "pct"),
    ("ped_collisions", "Runs hitting a person (%)", "pct_any"),
    ("vehicle_collisions", "Runs hitting a car (%)", "pct_any"),
    ("min_gap", "Min gap to people (m)", "mean"),
    ("near_misses", "Near misses per run", "mean"),
    ("min_ttc", "Min time-to-collision (s)", "mean"),
    ("time", "Time to goal (s)", "mean"),
    ("path_length", "Path length (m)", "mean"),
    ("time_stopped", "Time stopped (s)", "mean"),
    ("hard_brakes", "Hard brakes per run", "mean"),
    ("max_decel", "Strongest braking (m/s2)", "mean"),
    ("steer_jerk", "Steering jerk (rad/s2)", "mean"),
    ("correct_pct", "Correct action (%)", "mean"),
    ("reaction_gap", "Reaction gap (m)", "mean"),
    ("ade", "ADE (m)", "mean"),
    ("fde", "FDE (m)", "mean"),
]


def summarise(rows):
    """Average the per-run numbers: {label: value}."""
    out = {}
    for col, label, how in SUMMARY:
        if col == "correct_pct":
            vals = [100.0 * r["correct_actions"] / r["graded_people"] for r in rows if r["graded_people"]]
        else:
            vals = [r[col] for r in rows if not (isinstance(r[col], float) and math.isnan(r[col]))]
        if not vals:
            out[label] = math.nan
        elif how == "pct":
            out[label] = 100.0 * sum(vals) / len(vals)
        elif how == "pct_any":
            out[label] = 100.0 * sum(1 for v in vals if v > 0) / len(vals)
        else:
            out[label] = float(np.mean(vals))
    return out
