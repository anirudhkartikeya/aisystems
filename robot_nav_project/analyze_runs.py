"""analyze_runs.py - group results and uncertainty from a saved run dataset (no new simulation).

    python analyze_runs.py run_data/v3

Uses only the saved files of make_run_dataset.py. Writes <folder>/analysis/:
  A. main_with_ci.csv     every brain, dev and eval: contact rate, near misses, min gap, goal rate,
                          time - with 95 % bootstrap confidence intervals over the seeds
     paired.csv           brain vs brain on the SAME seeds: average difference + 95 % CI, and in how
                          many seeds each one was better (the honest "is it really better?" check)
  B. by_scripted_person   P1-P5: correct action, contact, closest gap, per brain
  C. by_p5                P5's random choice (stop / cross, left / right) per brain
  D. by_timing            crossers who step out early vs late (+/- 1 s jitter) per brain
  E. by_busyness          quiet vs busy scenes (background people, traffic cars) per brain
  F. by_person_type       background people: contacts and near misses per 100 s spent within 3 m of the
                          car, by kind (adult / child / elderly), aware or not, on the phone, jaywalker
  G. failures.csv         every run that touched a person or did not reach the goal, with the person
                          involved and the kind of failure
  + group_results.txt (all tables) and charts (main_ci.png, groups.png)
Dev = seeds 0-19 (used while tuning), eval = the held-out seeds. Groups B-F pool dev + eval (more
runs per group) and also show eval alone where it matters.
"""
import csv
import gzip
import math
import os
import sys
from collections import defaultdict

import numpy as np

from metrics import count_contacts_and_near_misses

folder = sys.argv[1] if len(sys.argv) > 1 else "run_data/v3"
out = os.path.join(folder, "analysis")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def read(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(os.path.join(folder, name), "rt", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def boot_ci(vals, stat=np.mean, n=4000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return math.nan, math.nan, math.nan
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    s = np.array([stat(vals[i]) for i in idx]) if stat is not np.mean else vals[idx].mean(1)
    return float(stat(vals)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def table(rows, cols, title):
    say(f"\n{title}")
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    say("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        say("  " + "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


runs = read("runs.csv")
crossers = read("crossers.csv")
scen = {int(r["seed"]): r for r in read("scenarios.csv")}
info = read("people_info.csv")
brains = list(dict.fromkeys(r["brain"] for r in runs))
for r in runs:
    r["seed"] = int(r["seed"])
say(f"Dataset {folder}: {len(runs)} runs, brains {brains}, "
    f"dev seeds {sum(1 for s in scen.values() if s['split'] == 'dev')}, eval seeds {sum(1 for s in scen.values() if s['split'] == 'eval')}")

# per-run numbers
per = {}
for r in runs:
    per[r["run_id"]] = {"contact_run": int(int(r["ped_collisions"]) > 0), "contacts": int(r["ped_collisions"]),
                        "near_misses": int(r["near_misses"]), "min_gap": float(r["min_gap"]),
                        "goal": int(r["result"] == "goal"), "time": float(r["time"]),
                        "correct_pct": 100 * int(r["correct_actions"]) / max(1, int(r["graded_people"])),
                        "hard_brakes": int(r["hard_brakes"])}
METRICS = [("contact_run", "runs touching a person (%)", 100), ("contacts", "contacts per run", 1),
           ("near_misses", "near misses per run", 1), ("min_gap", "closest gap (m)", 1),
           ("goal", "reached the goal (%)", 100), ("time", "time (s)", 1),
           ("correct_pct", "correct action (%)", 1), ("hard_brakes", "hard brakes per run", 1)]

# ------------------------------------------------------------------ A. main metrics with CIs
rows = []
for sp in ("dev", "eval"):
    for b in brains:
        ids = [r["run_id"] for r in runs if r["brain"] == b and r["split"] == sp]
        row = {"split": sp, "brain": b, "runs": len(ids)}
        for k, label, mult in METRICS:
            m, lo, hi = boot_ci([per[i][k] for i in ids])
            row[label] = f"{m * mult:.2f} [{lo * mult:.2f}, {hi * mult:.2f}]"
        rows.append(row)
write("main_with_ci.csv", rows)
table(rows, ["split", "brain", "runs"] + [m[1] for m in METRICS[:5]],
      "A. Main results, mean [95% bootstrap CI over seeds]")

prow = []
pairs = [(a, b) for i, a in enumerate(brains) for b in brains[i + 1:]]
for sp in ("dev", "eval", "all"):
    for a, b in pairs:
        ra = {r["seed"]: r["run_id"] for r in runs if r["brain"] == a and (sp == "all" or r["split"] == sp)}
        rb = {r["seed"]: r["run_id"] for r in runs if r["brain"] == b and (sp == "all" or r["split"] == sp)}
        seeds = sorted(set(ra) & set(rb))
        for k, label, mult in (METRICS[1], METRICS[2], METRICS[3], METRICS[5]):
            d = np.array([per[rb[s]][k] - per[ra[s]][k] for s in seeds])
            m, lo, hi = boot_ci(d)
            better_b = int(np.sum(d < 0)) if k != "min_gap" else int(np.sum(d > 0))
            better_a = int(np.sum(d > 0)) if k != "min_gap" else int(np.sum(d < 0))
            prow.append({"split": sp, "comparison": f"{b} - {a}", "metric": label, "seeds": len(seeds),
                         "mean difference [95% CI]": f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                         "seeds second better": better_b, "seeds first better": better_a, "ties": len(seeds) - better_a - better_b,
                         "CI excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
write("paired.csv", prow)
say("\n   Paired comparison on the SAME seeds (difference = second brain minus first; CI excludes 0 = a real difference):")
for p in prow:
    if p["split"] in ("eval", "all") and p["metric"] in ("contacts per run", "closest gap (m)"):
        b_, a_ = p["comparison"].split(" - ")
        say(f"   {p['split']:4s} {p['comparison']:24s} {p['metric']:18s} {p['mean difference [95% CI]']:22s} "
            f"{b_} better in {p['seeds second better']}, {a_} better in {p['seeds first better']}, ties {p['ties']} "
            f"-> CI excludes 0: {p['CI excludes 0']}")

# ------------------------------------------------------------------ B. scripted people
by_run_brain = {r["run_id"]: r for r in runs}
rows = []
for name in sorted({c["name"] for c in crossers}):
    for b in brains:
        cs = [c for c in crossers if c["name"] == name and c["brain"] == b]
        if not cs:
            continue
        rows.append({"person": name, "expected": cs[0]["expected"], "brain": b, "runs": len(cs),
                     "correct %": round(100 * np.mean([c["action"] == c["expected"] and float(c["min_gap"]) > 0
                                                       for c in cs]), 1) if cs[0]["expected"] != "decide" else "-",
                     "contact %": round(100 * np.mean([float(c["min_gap"]) <= 0 for c in cs]), 1),
                     "mean closest gap (m)": round(float(np.mean([float(c["min_gap"]) for c in cs])), 2),
                     "actions": ", ".join(f"{a} {n}" for a, n in sorted(
                         {(a or "never reached"): sum(1 for c in cs if c["action"] == a)
                          for a in {c["action"] for c in cs}}.items(),
                         key=lambda kv: -kv[1]))})
write("by_scripted_person.csv", rows)
table(rows, ["person", "expected", "brain", "runs", "correct %", "contact %", "mean closest gap (m)", "actions"],
      "B. Scripted people (dev + eval pooled)")

# ------------------------------------------------------------------ C-E. scene groups
def group_rows(keyfn, title, fname, only=None):
    rows = []
    groups = defaultdict(list)
    for r in runs:
        if only and r["split"] != only:
            continue
        groups[(keyfn(r), r["brain"])].append(r["run_id"])
    for (g, b) in sorted(groups, key=lambda k: (str(k[0]), brains.index(k[1]))):
        ids = groups[(g, b)]
        rows.append({"group": g, "brain": b, "runs": len(ids),
                     "touch a person %": round(100 * np.mean([per[i]["contact_run"] for i in ids]), 1),
                     "near misses/run": round(float(np.mean([per[i]["near_misses"] for i in ids])), 2),
                     "closest gap (m)": round(float(np.mean([per[i]["min_gap"] for i in ids])), 2),
                     "goal %": round(100 * np.mean([per[i]["goal"] for i in ids]), 1),
                     "time (s)": round(float(np.mean([per[i]["time"] for i in ids])), 1)})
    write(fname, rows)
    table(rows, ["group", "brain", "runs", "touch a person %", "near misses/run", "closest gap (m)", "goal %", "time (s)"], title)
    return rows


group_rows(lambda r: f"P5 {dict(cross='crosses', stop='stops in the lane').get(scen[r['seed']]['P5_behaviour'])}, "
                     f"from the {scen[r['seed']]['P5_side']}",
           "C. By P5's random choice (dev + eval)", "by_p5.csv")
rows = []
for name in ("P1", "P2", "P3", "P4", "P5"):
    tcol = f"{name}_timing_s"
    if tcol not in next(iter(scen.values())):
        continue
    vals = np.array([float(s[tcol]) for s in scen.values()])
    med = float(np.median(vals))
    for b in brains:
        for lab, sel in (("early (steps out sooner)", lambda v: v <= med), ("late", lambda v: v > med)):
            cs = [c for c in crossers if c["name"] == name and c["brain"] == b and sel(float(scen[int(c["seed"])][tcol]))]
            if cs:
                rows.append({"person": name, "timing": f"{lab} (median {med:+.2f} s)", "brain": b, "runs": len(cs),
                             "contact %": round(100 * np.mean([float(c["min_gap"]) <= 0 for c in cs]), 1),
                             "mean closest gap (m)": round(float(np.mean([float(c["min_gap"]) for c in cs])), 2)})
write("by_timing.csv", rows)
table(rows, ["person", "timing", "brain", "runs", "contact %", "mean closest gap (m)"],
      "D. Crossers stepping out earlier vs later than the median (timing jitter, dev + eval)")
bp = np.median([int(s["background_people"]) for s in scen.values()])
tc = np.median([int(s["traffic_cars"]) for s in scen.values()])
group_rows(lambda r: ("busy" if int(scen[r["seed"]]["background_people"]) > bp else "quiet") +
           f" sidewalks (people {'>' if int(scen[r['seed']]['background_people']) > bp else '<='} {bp:g})",
           "E1. Quiet vs busy sidewalks (dev + eval)", "by_people.csv")
group_rows(lambda r: ("heavy" if int(scen[r["seed"]]["traffic_cars"]) > tc else "light") +
           f" traffic (cars {'>' if int(scen[r['seed']]['traffic_cars']) > tc else '<='} {tc:g})",
           "E2. Light vs heavy traffic (dev + eval)", "by_traffic.csv")

# ------------------------------------------------------------------ F. background people by type
say("\n   reading the per-step logs for F and G ...")
pinfo = {(r["run_id"], r["ped_id"]): r for r in info}
robot = {}
for r in read("robot_steps.csv.gz"):
    robot[(r["run_id"], r["step"])] = (float(r["x"]), float(r["y"]), r["done"] == "1", r["status"], float(r["t"]))
series = defaultdict(list)
with gzip.open(os.path.join(folder, "people_steps.csv.gz"), "rt") as f:
    for r in csv.DictReader(f):
        x, y, done, _, _ = robot[(r["run_id"], r["step"])]
        g = float(r["gap_to_car"])
        inbox = abs(float(r["x"]) - x) <= 15 and abs(float(r["y"]) - y) <= 15
        series[(r["run_id"], r["ped_id"])].append((int(r["step"]), g, inbox and not done))
types = defaultdict(lambda: {"exposure_s": 0.0, "contacts": 0, "near": 0, "people": 0})
brain_of = {r["run_id"]: r["brain"] for r in runs}
for (rid, pid), ser in series.items():
    who = pinfo[(rid, pid)]
    if who["role"] != "background":
        continue
    ser.sort()
    c, n = count_contacts_and_near_misses([g if ok else None for _, g, ok in ser])
    expo = 0.1 * sum(1 for _, g, _ in ser if g < 3.0)
    b = brain_of[rid]
    labels = [f"kind: {who['kind']}", "aware of cars" if who["aware"] == "1" else "NOT aware of cars",
              "on the phone" if who["phone"] == "1" else "not on the phone",
              "jaywalker" if who["jaywalker"] == "1" else "not a jaywalker"]
    for lab in labels:
        t = types[(lab, b)]
        t["exposure_s"] += expo; t["contacts"] += c; t["near"] += n; t["people"] += 1
rows = []
for (lab, b) in sorted(types, key=lambda k: (k[0], brains.index(k[1]))):
    t = types[(lab, b)]
    rows.append({"person type": lab, "brain": b, "people": t["people"], "seconds within 3 m": round(t["exposure_s"], 1),
                 "contacts": t["contacts"], "near misses": t["near"],
                 "contacts per 100 s close": round(100 * t["contacts"] / t["exposure_s"], 2) if t["exposure_s"] else "-",
                 "near misses per 100 s close": round(100 * t["near"] / t["exposure_s"], 2) if t["exposure_s"] else "-"})
write("by_person_type.csv", rows)
table(rows, ["person type", "brain", "people", "seconds within 3 m", "contacts", "near misses",
             "contacts per 100 s close", "near misses per 100 s close"],
      "F. Background people by type (dev + eval): events per 100 s they spent within 3 m of the car")

# ------------------------------------------------------------------ G. failures
last_status = {}
for (rid, step), v in robot.items():
    if rid not in last_status or int(step) > last_status[rid][0]:
        last_status[rid] = (int(step), v[3])
fails = []
for r in runs:
    rid = r["run_id"]
    if r["result"] == "goal" and int(r["ped_collisions"]) == 0:
        continue
    mine = {pid: sorted(ser) for (rr, pid), ser in series.items() if rr == rid}
    if r["result"] != "goal":                          # stuck: who is it stuck next to at the end?
        close = min((ser[-1][1], pid) for pid, ser in mine.items())
    else:                                              # touched: the closest person of the run
        close = min((min(g for _, g, _ in ser), pid) for pid, ser in mine.items())
    who = pinfo[(rid, close[1])]
    step_min = max(s for s, g, _ in mine[close[1]] if g == close[0])
    kind = []
    if int(r["ped_collisions"]) > 0:
        kind.append("touched a person")
    if r["result"] != "goal":
        kind.append(f"{r['result']}: last status '{last_status[rid][1]}'" +
                    (" -> standoff with a person in the lane" if ("person" in last_status[rid][1] or
                                                                     "emergency" in last_status[rid][1]) else ""))
    fails.append({"run": rid, "split": r["split"], "failure": "; ".join(kind),
                  "person involved": who["name"] or f"background {who['kind']}",
                  "aware of cars": "yes" if who["aware"] == "1" else "no", "gap (m)": round(close[0], 3),
                  "when (s)": round(robot[(rid, str(step_min))][4], 1)})
write("failures.csv", fails)
counts = defaultdict(lambda: defaultdict(int))
for f_ in fails:
    counts[f_["run"].split("_s")[0]][f_["split"]] += 1
table(fails, ["run", "split", "failure", "person involved", "aware of cars", "gap (m)", "when (s)"],
      "G. Every run that touched a person or did not reach the goal")

open(os.path.join(out, "group_results.txt"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ charts
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = {"reactive": "#c0392b", "predictive": "#2471a3", "lstm": "#229954"}
    for ax, (k, label, mult) in zip(axes, [METRICS[0], METRICS[2], METRICS[3], METRICS[4]]):
        for j, sp in enumerate(("dev", "eval")):
            for i, b in enumerate(brains):
                ids = [r["run_id"] for r in runs if r["brain"] == b and r["split"] == sp]
                m, lo, hi = boot_ci([per[x][k] for x in ids])
                xpos = j * (len(brains) + 1) + i
                ax.bar(xpos, m * mult, color=colors.get(b, "gray"), label=b if j == 0 else None)
                ax.errorbar(xpos, m * mult, yerr=[[m * mult - lo * mult], [hi * mult - m * mult]], color="k", capsize=3)
        ax.set_xticks([(len(brains) - 1) / 2, len(brains) + 1 + (len(brains) - 1) / 2], ["dev", "eval"])
        ax.set_title(label, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean and 95% bootstrap confidence interval over seeds")
    plt.tight_layout(); plt.savefig(os.path.join(out, "main_ci.png"), dpi=120); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    names = sorted({c["name"] for c in crossers})
    w = 0.8 / len(brains)
    for i, b in enumerate(brains):
        vals = [100 * np.mean([float(c["min_gap"]) <= 0 for c in crossers if c["name"] == n and c["brain"] == b]) for n in names]
        ax[0].bar(np.arange(len(names)) + i * w, vals, w, label=b, color=colors.get(b, "gray"))
        gaps = [np.mean([float(c["min_gap"]) for c in crossers if c["name"] == n and c["brain"] == b]) for n in names]
        ax[1].bar(np.arange(len(names)) + i * w, gaps, w, label=b, color=colors.get(b, "gray"))
    for a, t in zip(ax, ("contact with this person (% of runs)", "mean closest gap to this person (m)")):
        a.set_xticks(np.arange(len(names)) + w * (len(brains) - 1) / 2, names); a.set_title(t); a.legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(out, "groups.png"), dpi=120); plt.close()
except Exception as e:
    print("(no charts:", e, ")")
print(f"\nSaved {out}/ (group_results.txt, CSVs, main_ci.png, groups.png)")
