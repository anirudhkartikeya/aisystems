"""run_report.py - the report saved at the end of every run_sim.py run.

    run_reports/<brain>_seed<seed>_<date-time>/
        report.html   open in a browser: every metric explained, the scripted people, the chart
        report.txt    the same as plain text
        report.png    the car's speed and its closest gap to people over the run

Used by run_sim.py (Trace collects what happens at every step, write_report saves it).
"""
import base64
import datetime
import html
import math
import os

import config
from metrics import surface_gap


class Trace:
    """What happened at every 0.1 s step: time, speed, closest gap to a person, the car's status."""

    def __init__(self, sim, driver):
        self.sim, self.driver = sim, driver
        self.t, self.speed, self.gap, self.status = [], [], [], []
        self.closest = {}                    # scripted person name -> (gap, time)
        self.names = {log["ped_id"]: log["name"] for log in sim.crosser_log}

    def step(self):
        sim, r = self.sim, self.sim.robot
        foot = r.footprint()
        best = math.inf
        for ped in sim.peds:
            if abs(ped.x - r.x) > 15 or abs(ped.y - r.y) > 15:
                continue
            g = surface_gap(foot, ped.x, ped.y, ped.radius)
            best = min(best, g)
            name = self.names.get(ped.ped_id)
            if name and g < self.closest.get(name, (math.inf, 0))[0]:
                self.closest[name] = (g, sim.t)
        self.t.append(sim.t)
        self.speed.append(math.hypot(r.vx, r.vy))
        self.gap.append(best if math.isfinite(best) else float("nan"))
        self.status.append(self.driver.planner.status if self.driver is not None else "")


EXPLAIN = [
    ("result", "How the run ended", "goal = reached the goal; collision_* = stopped at a contact; timeout = still "
                                    "driving after the time limit"),
    ("success", "Success", "reached the goal without touching anyone"),
    ("ped_collisions", "People touched", "times the car's body overlapped a person (surface gap < 0)"),
    ("vehicle_collisions", "Cars touched", "times the car touched another car"),
    ("near_misses", "Near misses", f"a person came within {config.NEAR_MISS_DIST} m of the car's surface without "
                                   f"touching (counted again only after the gap opened past "
                                   f"{config.NEAR_MISS_DIST + config.NEAR_MISS_REARM} m)"),
    ("min_gap", "Closest gap (m)", "smallest surface-to-surface distance to any person (negative = overlap)"),
    ("min_ttc", "Shortest time-to-collision (s)", "how soon the car would have reached someone at that moment's "
                                                  "closing speed (99 = never closing in)"),
    ("time", "Time (s)", "time to the goal (or until the run ended)"),
    ("path_length", "Path length (m)", "distance the car drove"),
    ("time_stopped", "Time stopped (s)", "time standing still (below 0.1 m/s)"),
    ("hard_brakes", "Hard brakes", f"times it braked harder than {config.HARD_BRAKE} m/s2"),
    ("max_decel", "Strongest braking (m/s2)", "the hardest it braked"),
    ("steer_jerk", "Steering jerk (rad/s2)", "how jerky the steering was (lower = smoother)"),
    ("correct_actions", "Correct actions", "scripted people handled as planned (swerve left / right or stop) out "
                                           "of those with a planned action (P5 decides for itself, not graded)"),
    ("reaction_gap", "Reaction distance (m)", "how far from a scripted person the car was when it first reacted to "
                                              "them (bigger = earlier)"),
    ("ade", "Prediction error ADE (m)", "average distance between where the brain predicted people would be and "
                                        "where they really were, over 3 s"),
    ("fde", "Prediction error FDE (m)", "the same, at the end of the 3 s"),
]


def _chart(trace, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = trace.t
    ax[0].plot(t, trace.speed, color="#2471a3")
    person = [("person" in s or "emergency" in s or "squeezing" in s) for s in trace.status]
    lo = None
    for i, flag in enumerate(person + [False]):
        if flag and lo is None:
            lo = t[i]
        if not flag and lo is not None:
            ax[0].axvspan(lo, t[i - 1], color="#f5b041", alpha=0.3, lw=0)
            lo = None
    ax[0].set_ylabel("car speed (m/s)")
    ax[0].set_title(title + "   (orange = reacting to a person)")
    ax[1].plot(t, trace.gap, color="#229954", lw=1)
    ax[1].axhline(config.NEAR_MISS_DIST, color="#e67e22", ls="--", lw=1, label=f"near miss ({config.NEAR_MISS_DIST} m)")
    ax[1].axhline(0, color="#c0392b", lw=1, label="contact (0 m)")
    for name, (g, tt) in trace.closest.items():
        if g < 3:
            ax[1].annotate(f"{name} {g:.2f} m", (tt, g), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            ax[1].plot(tt, g, "k.")
    ax[1].set_ylim(-1.2, 4)
    ax[1].set_ylabel("closest gap to a person (m)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return True


def _val(row, key):
    if key == "success":
        return "yes" if row[key] else "no"
    if key == "correct_actions":
        return f"{row[key]} / {row['graded_people']}"
    return str(row[key])


def write_report(info, row, people, trace, args, folder="run_reports"):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(folder, f"{args.brain}_seed{info['seed']}_{stamp}")
    os.makedirs(out, exist_ok=True)
    title = f"{args.brain} brain - {args.layout} / {args.scenario} - seed {info['seed']}"
    settings = (f"brain {args.brain}, layout {args.layout}, scenario {args.scenario}, seed {info['seed']}, "
                f"lighting {config.LIGHTING}, made {stamp.replace('_', ' ')}")
    lines = [f"RUN REPORT - {title}", settings, ""]
    for key, label, text in EXPLAIN:
        if key in row:
            lines.append(f"{label:32s} {_val(row, key):>10s}   {text}")
    lines += ["", "Scripted people (OK = handled as planned, NO = not, -- = not graded):"]
    prow = []
    for p_ in people:
        ok = "OK" if p_["action"] == p_["expected"] and p_["min_gap"] > 0 else ("--" if p_["expected"] == "decide" else "NO")
        prow.append((ok, p_))
        lines.append(f"  {ok} {p_['name']}: expected {p_['expected']:12s} did {str(p_['action']):12s} "
                     f"closest {p_['min_gap']:.2f} m   ({p_['behaviour']}, from the {p_['side']}, "
                     f"{'on a zebra' if p_['zebra'] else 'not on a crossing'})")
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    png = os.path.join(out, "report.png")
    has_chart = _chart(trace, png, title)

    e = html.escape
    rows = "".join(f"<tr><td>{e(label)}</td><td class=v>{e(_val(row, key))}</td>"
                   f"<td class=x>{e(text)}</td></tr>" for key, label, text in EXPLAIN if key in row)
    people_rows = "".join(
        f"<tr class={'ok' if ok == 'OK' else 'bad' if ok == 'NO' else ''}><td>{ok}</td><td>{e(p_['name'])}</td>"
        f"<td>{e(p_['expected'])}</td><td>{e(str(p_['action']))}</td><td class=v>{p_['min_gap']:.2f}</td>"
        f"<td>{e(p_['behaviour'])}, from the {e(p_['side'])}</td></tr>" for ok, p_ in prow)
    img = ""
    if has_chart:
        img = f"<h2>Speed and closest gap over the run</h2><img src='data:image/png;base64," \
              f"{base64.b64encode(open(png, 'rb').read()).decode()}'>"
    good = row.get("success")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Run report</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d2733;--muted:#5d6b7a;--line:#d9dee3;--ok:#e3f4e8;--bad:#fbe4e1;--accent:#2471a3}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16191d;--fg:#e8ecf0;--muted:#9aa7b4;--line:#2c333b;--ok:#183824;--bad:#3e1f1c;--accent:#6fb1e0}}}}
body{{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg);max-width:1050px;margin:24px auto;padding:0 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} .muted{{color:var(--muted)}}
.badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-weight:600;background:{'var(--ok)' if good else 'var(--bad)'}}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}} td.v{{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}} td.x{{color:var(--muted)}}
tr.ok td:first-child{{background:var(--ok)}} tr.bad td:first-child{{background:var(--bad)}} img{{max-width:100%;border:1px solid var(--line);border-radius:6px}}
</style></head><body>
<h1>Run report: {e(title)}</h1>
<p class=muted>{e(settings)}</p>
<p><span class=badge>{'SUCCESS' if good else 'NOT A SUCCESS'}</span> &nbsp; result: <b>{e(str(row['result']))}</b>,
people touched: <b>{row['ped_collisions']}</b>, near misses: <b>{row['near_misses']}</b>, closest gap: <b>{row['min_gap']} m</b>,
time: <b>{row['time']} s</b></p>
<h2>All metrics</h2><table><tr><th>Metric</th><th>Value</th><th>What it means</th></tr>{rows}</table>
<h2>Scripted people</h2><table><tr><th></th><th>Person</th><th>Planned action</th><th>What the car did</th><th>Closest gap (m)</th><th>What they did</th></tr>{people_rows}</table>
{img}
</body></html>"""
    path = os.path.join(out, "report.html")
    open(path, "w").write(page)
    return out, path
