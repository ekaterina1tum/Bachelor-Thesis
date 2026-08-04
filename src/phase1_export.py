"""
Solve ONE Phase-1 instance and export its solution as JSON for Phase 2.

The JSON captures everything Phase 2 treats as fixed input:
routes, trips (with baseline return time C0, idle g, ordering), regular
requests (z0, soft window d, and the arcs B_j traversed before each request),
and per-route baseline duration Lambda0.

Usage
-----
    python phase1_export.py <instance.txt> [time_limit_s] [max_shift] [out_dir]

If out_dir is omitted, the file is written to
    data/phase1_solutions/<size>/<instance>.json
mirroring the input's size folder.
"""

from __future__ import annotations

import os
import sys
import csv
import glob
import json

import gurobipy as gp
from gurobipy import GRB

from instance import load_instance
from graph import build_graph
from model import build_model


def reconstruct_solution(model: gp.Model):
    """Walk the solved arc chain and return the Phase-1 solution as plain dicts.

    Returns a dict ready for JSON serialization (no Gurobi objects).
    """
    inst = model._inst
    graph = model._graph
    x = model._x
    z_var = model._z
    tau_var = model._tau

    depot = inst.depot
    z_val = {j: z_var[j].X for j in inst.pocs}
    tau_val = {j: tau_var[j].X for j in inst.pocs}
    # Per-PoC shift time used so far, including the return leg to the depot.
    # Phase-1 constraint (11) caps this at max_shift, so the route's shift
    # DURATION is its maximum over the PoCs it serves.
    shift_end = {j: tau_val[j] + inst.c(j, depot) for j in inst.pocs}

    used = [e for e in graph.all_arcs if x[e].X > 0.5]
    out_arc = {e.src: e for e in used}  # src -> outgoing used arc

    def follow_route(start):
        chain, cur = [], start
        while True:
            chain.append(cur)
            if cur.tgt == depot:
                break
            cur = out_arc.get(cur.tgt)
            if cur is None:
                break
        return chain

    depot_out = [e for e in used if e.src == depot]

    trips = []
    routes: dict[int, list[int]] = {}
    Lambda0: dict[int, float] = {}
    regular = {}          # j -> dict
    arc_id = 0
    trip_id = 0
    route_id = 0
    seen_starts = set()

    for start in depot_out:
        if id(start) in seen_starts:
            continue
        seen_starts.add(id(start))

        chain = follow_route(start)

        # Expand replenishment arcs i->j (== i -> depot -> j) into two legs, so the
        # path becomes a clean sequence of depot/PoC visits we can split at the depot.
        legs: list[tuple[int, int]] = []
        for a in chain:
            if a.kind == "AR":
                legs.append((a.src, depot))   # close current trip
                legs.append((depot, a.tgt))   # open next trip
            else:
                legs.append((a.src, a.tgt))

        trip_ids_in_route = []
        cur_arcs = []   # list of {"id","src","tgt"} for the trip being built

        for (s, t) in legs:
            cur_arcs.append({"id": arc_id, "src": s, "tgt": t})
            arc_id += 1
            # A PoC is collected when an arc arrives at it
            if t in inst.pocs and t not in regular:
                regular[t] = {
                    "id": t,
                    "trip_id": trip_id,
                    "z0": z_val[t],
                    "d": inst.due_time[t],
                    "B": [pa["id"] for pa in cur_arcs],  # arcs up to & incl the one reaching t
                }
            # Trip closes when we return to the depot
            if t == depot:
                last_poc = cur_arcs[-1]["src"]
                C0_T = (z_val[last_poc] + inst.c(last_poc, depot)) \
                    if last_poc in inst.pocs else 0.0
                trips.append({
                    "id": trip_id,
                    "route_id": route_id,
                    "C0": C0_T,
                    "g": 0.0,                # inter-trip idle not exposed by Phase 1
                    "is_first": len(trip_ids_in_route) == 0,
                    "is_last": False,        # set below
                    "next_trip_id": None,    # set below
                    "arcs": cur_arcs,
                })
                trip_ids_in_route.append(trip_id)
                trip_id += 1
                cur_arcs = []

        # Fix up last/next within this route, and compute inter-trip idle g_T
        for i, tid in enumerate(trip_ids_in_route):
            t = trips[tid]
            t["is_last"] = (i == len(trip_ids_in_route) - 1)
            t["next_trip_id"] = trip_ids_in_route[i + 1] if not t["is_last"] else None
            # g_T = idle slack between this trip's return and the next trip's departure:
            #   departure(T+) = z0(first PoC of T+) - c(depot, first PoC)
            #   g_T = max(0, departure(T+) - C0_T)
            if t["is_last"]:
                t["g"] = 0.0
            else:
                nxt = trips[t["next_trip_id"]]
                first_poc = nxt["arcs"][0]["tgt"]
                if first_poc in inst.pocs:
                    departure_next = z_val[first_poc] - inst.c(depot, first_poc)
                    t["g"] = max(0.0, departure_next - t["C0"])
                else:
                    t["g"] = 0.0

        routes[route_id] = trip_ids_in_route
        # Route shift duration = max shift-end over the PoCs this route serves
        # (matches Phase-1 constraint (11): tau_j + c(j,0) <= max_shift).
        route_pocs = [a["src"] for tid in trip_ids_in_route for a in trips[tid]["arcs"]
                      if a["src"] in inst.pocs]
        Lambda0[route_id] = max((shift_end[p] for p in route_pocs), default=0.0)
        route_id += 1

    return {
        "routes": {str(k): v for k, v in routes.items()},
        "trips": trips,
        "regular_requests": list(regular.values()),
        "Lambda0": {str(k): v for k, v in Lambda0.items()},
    }


def export_instance(path: str, time_limit: float, max_shift: float, out_dir: str | None,
                    params: dict | None = None):
    """Solve one instance and export its Phase-1 solution JSON.

    params : optional dict of extra Gurobi parameters, e.g. {"MIPFocus": 3, "Threads": 16}.
    """
    inst = load_instance(path, max_shift=max_shift)
    g = build_graph(inst)
    m = build_model(inst, g)
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    for key, val in (params or {}).items():
        m.setParam(key, val)
    m.optimize()

    if m.SolCount == 0:
        print(f"No solution found (status {m.Status}); nothing exported.")
        return None

    sol = reconstruct_solution(m)

    # Human-readable Gurobi status names (subset we expect here)
    STATUS_NAMES = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
    }
    status_name = STATUS_NAMES.get(m.Status, f"STATUS_{m.Status}")
    try:
        gap = m.MIPGap
    except Exception:
        gap = float("nan")
    try:
        best_bound = m.ObjBound
    except Exception:
        best_bound = float("nan")

    sum_r = sum(inst.release_time[j] for j in inst.pocs)
    name = os.path.splitext(os.path.basename(path))[0]
    # Trust the actual optimality gap, not Gurobi's status label: a big-M model
    # can occasionally terminate with status OPTIMAL while its dual bound is
    # numerically inconsistent (gap > tolerance). Flag that case explicitly.
    gap_ok = (gap == gap) and gap <= 1e-4          # gap==gap filters NaN
    status_gap_inconsistent = (m.Status == GRB.OPTIMAL) and not gap_ok

    doc = {
        "instance": name,
        "status": status_name,
        "is_optimal": bool(gap_ok),
        "status_gap_inconsistent": bool(status_gap_inconsistent),
        "objective": m.ObjVal,
        "F_prime": m.ObjVal - sum_r,
        "best_bound": best_bound,
        "mip_gap": gap,
        "runtime_s": round(m.Runtime, 2),         # wall-clock solve time (seconds)
        "time_limit_s": time_limit,               # the limit it ran under
        "max_shift": max_shift,
        **sol,
    }

    if out_dir is None:
        size = os.path.basename(os.path.dirname(os.path.abspath(path)))
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "phase1_solutions", size,
        )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)

    flag = ""
    if doc["is_optimal"]:
        flag = "(optimal-quality)"
    elif doc["status_gap_inconsistent"]:
        flag = "(!) status OPTIMAL but gap>tol -- numerically unreliable bound"
    print(f"{name}: obj={doc['objective']:.2f}  F'={doc['F_prime']:.2f}  "
          f"status={doc['status']}  gap={doc['mip_gap']*100:.2f}%  "
          f"time={doc['runtime_s']:.1f}s  {flag}  ->  {out_path}")
    return doc


def export_folder(folder: str, time_limit: float, max_shift: float, out_dir: str | None,
                  params: dict | None = None):
    """Export every *.txt instance in a folder; write a combined summary CSV."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        print(f"No .txt instances found in {folder}")
        return

    size = os.path.basename(os.path.abspath(folder.rstrip("/")))
    print(f"Batch export: {len(files)} instances from {folder} "
          f"(time_limit={time_limit}s, max_shift={max_shift})\n")

    rows = []
    for path in files:
        doc = export_instance(path, time_limit, max_shift, out_dir, params=params)
        if doc is None:
            continue
        rows.append({
            "instance": doc["instance"],
            "status": doc["status"],
            "is_optimal": doc["is_optimal"],
            "status_gap_inconsistent": doc["status_gap_inconsistent"],
            "objective": round(doc["objective"], 4),
            "F_prime": round(doc["F_prime"], 4),
            "best_bound": round(doc["best_bound"], 4),
            "mip_gap": round(doc["mip_gap"], 6),
            "runtime_s": doc["runtime_s"],
            "time_limit_s": doc["time_limit_s"],
        })

    # write the summary next to the JSON solutions
    if out_dir is None:
        sol_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "phase1_solutions", size,
        )
    else:
        sol_dir = out_dir
    os.makedirs(sol_dir, exist_ok=True)
    summary_path = os.path.join(sol_dir, f"summary_{size}.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "instance", "status", "is_optimal", "status_gap_inconsistent",
            "objective", "F_prime", "best_bound", "mip_gap",
            "runtime_s", "time_limit_s",
        ])
        w.writeheader()
        w.writerows(rows)

    n_opt = sum(1 for r in rows if r["is_optimal"])
    avg_t = sum(r["runtime_s"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nSolved {len(rows)} instances: {n_opt} optimal-quality, "
          f"avg runtime {avg_t:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Solve Phase 1 for one instance or a folder and export JSON(s).")
    ap.add_argument("path", help="instance .txt file OR a folder of instances")
    ap.add_argument("time_limit", nargs="?", type=float, default=3600.0,
                    help="Gurobi time limit in seconds (default 3600)")
    ap.add_argument("max_shift", nargs="?", type=float, default=480.0,
                    help="maximum shift duration (default 480)")
    ap.add_argument("--out-dir", default=None, help="override output directory")
    ap.add_argument("--numeric-focus", type=int, default=None, choices=[1, 2, 3],
                    help="Gurobi NumericFocus (1-3) for numerically hard instances")
    ap.add_argument("--mip-focus", type=int, default=None, choices=[1, 2, 3],
                    help="Gurobi MIPFocus (1-3)")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi thread count")
    args = ap.parse_args()

    params = {}
    if args.numeric_focus is not None:
        params["NumericFocus"] = args.numeric_focus
    if args.mip_focus is not None:
        params["MIPFocus"] = args.mip_focus
    if args.threads is not None:
        params["Threads"] = args.threads
    params = params or None

    if os.path.isdir(args.path):
        export_folder(args.path, args.time_limit, args.max_shift, args.out_dir, params=params)
    else:
        export_instance(args.path, args.time_limit, args.max_shift, args.out_dir, params=params)
