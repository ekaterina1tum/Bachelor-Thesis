"""
Evaluation-only feasibility diagnostics for Phase 2.

The script reads generated emergency scenarios, Phase-1 solution JSON files,
and Phase-2 summaries. It does not call Gurobi. It checks why emergencies are
outsourced by applying the same insertion feasibility logic used in
phase2_experiment.py:

    source time is after emergency release
    and
    C0_T + delta_m,e <= emergency deadline

No service times are introduced.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

from instance import load_coordinates


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTANCES_ROOT = os.path.join(REPO, "data", "MSCDPinstances")
PHASE1_ROOT = os.path.join(REPO, "data", "phase1_solutions")
SCENARIO_ROOT = os.path.join(REPO, "data", "emergency_scenarios")
RESULTS = os.path.join(REPO, "data", "phase2_results")

SIZE_DIR = {10: "010", 15: "015", 25: "025"}
CLASSES = ["C", "R", "RC"]
DEPOT = 0


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pct(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_phase1(size, instance, cache):
    key = (int(size), instance)
    if key not in cache:
        path = os.path.join(PHASE1_ROOT, SIZE_DIR[int(size)], f"{instance}.json")
        with open(path) as fh:
            cache[key] = json.load(fh)
    return cache[key]


def load_coords(size, instance, cache):
    key = (int(size), instance)
    if key not in cache:
        path = os.path.join(INSTANCES_ROOT, SIZE_DIR[int(size)], f"{instance}.txt")
        cache[key] = load_coordinates(path)[0]
    return cache[key]


def arc_data(solution):
    trips = {t["id"]: t for t in solution["trips"]}
    arcs = []
    for trip in solution["trips"]:
        for arc in trip["arcs"]:
            arcs.append({
                "src": arc["src"],
                "tgt": arc["tgt"],
                "trip_id": trip["id"],
                "route_id": trip["route_id"],
            })
    z0 = {r["id"]: r["z0"] for r in solution["regular_requests"]}
    return arcs, trips, z0


def baseline_time_at_source(arc, z0, coords):
    if arc["src"] != DEPOT:
        return z0.get(arc["src"], 0.0)
    target_time = z0.get(arc["tgt"])
    if target_time is None:
        return 0.0
    return max(0.0, target_time - euclid(coords[DEPOT], coords[arc["tgt"]]))


def has_feasible_insertion(emergency, W, arcs, trips, z0, coords):
    em_xy = (emergency["x"], emergency["y"])
    deadline = emergency["release"] + W
    for arc in arcs:
        if baseline_time_at_source(arc, z0, coords) < emergency["release"] - 1e-9:
            continue
        src_xy = coords[arc["src"]]
        tgt_xy = coords[arc["tgt"]]
        delta = euclid(src_xy, em_xy) + euclid(em_xy, tgt_xy) - euclid(src_xy, tgt_xy)
        if trips[arc["trip_id"]]["C0"] + delta <= deadline + 1e-9:
            return True
    return False


def scenario_files():
    return sorted(glob.glob(os.path.join(SCENARIO_ROOT, "W*", "*", "*", "*.json")))


def feasibility_rows():
    phase1_cache = {}
    coord_cache = {}
    rows = []
    for path in scenario_files():
        with open(path) as fh:
            scenario = json.load(fh)
        W = float(scenario["W"])
        solution = load_phase1(scenario["size"], scenario["instance"], phase1_cache)
        coords = load_coords(scenario["size"], scenario["instance"], coord_cache)
        arcs, trips, z0 = arc_data(solution)
        for emergency in scenario["emergencies"]:
            d0m = euclid(coords[DEPOT], (emergency["x"], emergency["y"]))
            feasible = has_feasible_insertion(emergency, W, arcs, trips, z0, coords)
            rows.append({
                "W": W,
                "size": int(scenario["size"]),
                "class": scenario["class"],
                "undeliverable": 2 * d0m > W + 1e-9,
                "infeasible": not feasible,
            })
    return rows


def grouped_rates(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = dict(zip(keys, key))
        rec["n"] = len(vals)
        rec["undeliverable"] = sum(v["undeliverable"] for v in vals) / len(vals)
        rec["infeasible"] = sum(v["infeasible"] for v in vals) / len(vals)
        out.append(rec)
    return out


def distance_scale_rows():
    by_class = defaultdict(list)
    all_values = []
    for path in sorted(glob.glob(os.path.join(INSTANCES_ROOT, "025", "*.txt"))):
        instance = os.path.splitext(os.path.basename(path))[0]
        cls = instance.split("_", 1)[1].rstrip("0123456789")
        coords = load_coordinates(path)[0]
        values = [euclid(coords[DEPOT], coords[j]) for j in coords if j != DEPOT]
        by_class[cls].extend(values)
        all_values.extend(values)
    rows = []
    for cls in CLASSES:
        values = by_class[cls]
        rows.append((cls, len(values), sum(values) / len(values), pct(values, 0.5), pct(values, 0.9), 2 * pct(values, 0.9)))
    rows.append(("global", len(all_values), sum(all_values) / len(all_values), pct(all_values, 0.5), pct(all_values, 0.9), 2 * pct(all_values, 0.9)))
    return rows


def observed_reason_split():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "summary_W*_lam5.0_piS2.csv"))):
        data = read_csv(path)
        if not data:
            continue
        W = float(data[0]["W"])
        for group in ["overall", "10", "15", "25"]:
            vals = data if group == "overall" else [r for r in data if r["size"] == group]
            if not vals:
                continue
            n_em = sum(int(r["n_emergencies"]) for r in vals)
            n_out = sum(int(r["n_outsourced"]) for r in vals)
            n_infeas = sum(int(r.get("n_outsourced_infeasible", 0)) for r in vals)
            n_econ = sum(int(r.get("n_outsourced_economic", 0)) for r in vals)
            n_inserted = sum(int(r.get("n_inserted", 0)) for r in vals)
            rows.append((W, group, n_inserted / n_em, n_infeas / n_em, n_econ / n_em, n_out / n_em, n_infeas / n_out if n_out else 0.0))
    return rows


def write_report():
    rows = feasibility_rows()
    lines = []
    L = lines.append
    L("=" * 78)
    L("PHASE-2 FEASIBILITY DIAGNOSTIC")
    L("  Evaluation-only; no Gurobi solve; no service times.")
    L("=" * 78)

    L("\nFeasibility rates by W:")
    L(f"  {'W':>5} {'n_em':>7} {'undeliv':>9} {'infeas_ins':>11}")
    for rec in grouped_rates(rows, ["W"]):
        L(f"  {rec['W']:>5g} {rec['n']:>7} {rec['undeliverable']:>9.3f} {rec['infeasible']:>11.3f}")

    L("\nFeasibility rates by W and size:")
    L(f"  {'W':>5} {'size':>5} {'n_em':>7} {'undeliv':>9} {'infeas_ins':>11}")
    for rec in grouped_rates(rows, ["W", "size"]):
        L(f"  {rec['W']:>5g} {rec['size']:>5} {rec['n']:>7} {rec['undeliverable']:>9.3f} {rec['infeasible']:>11.3f}")

    L("\nFeasibility rates by W and class:")
    L(f"  {'W':>5} {'class':>5} {'n_em':>7} {'undeliv':>9} {'infeas_ins':>11}")
    for rec in grouped_rates(rows, ["W", "class"]):
        L(f"  {rec['W']:>5g} {rec['class']:>5} {rec['n']:>7} {rec['undeliverable']:>9.3f} {rec['infeasible']:>11.3f}")

    L("\n025-only travel scale for W calibration:")
    L(f"  {'class':>7} {'n':>5} {'mean':>9} {'p50':>9} {'p90':>9} {'W=2*p90':>10}")
    for cls, n, mean, p50, p90, W_cal in distance_scale_rows():
        L(f"  {cls:>7} {n:>5} {mean:>9.2f} {p50:>9.2f} {p90:>9.2f} {W_cal:>10.2f}")

    L("\nObserved lambda=5, pi_S=2 reason split:")
    L(f"  {'W':>5} {'size':>8} {'inserted':>9} {'out_infeas':>11} {'out_econ':>9} {'out_total':>9} {'infeas/out':>11}")
    for W, group, inserted, out_infeas, out_econ, out_total, infeas_out in observed_reason_split():
        L(f"  {W:>5g} {group:>8} {inserted:>9.3f} {out_infeas:>11.3f} {out_econ:>9.3f} {out_total:>9.3f} {infeas_out:>11.3f}")

    path = os.path.join(RESULTS, "feasibility_recalibration_findings.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport: {path}")


if __name__ == "__main__":
    write_report()
