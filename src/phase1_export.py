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

        # Fix up last/next within this route
        for i, tid in enumerate(trip_ids_in_route):
            t = trips[tid]
            t["is_last"] = (i == len(trip_ids_in_route) - 1)
            t["next_trip_id"] = trip_ids_in_route[i + 1] if not t["is_last"] else None

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


def export_instance(path: str, time_limit: float, max_shift: float, out_dir: str | None):
    inst = load_instance(path, max_shift=max_shift)
    g = build_graph(inst)
    m = build_model(inst, g)
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
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

    sum_r = sum(inst.release_time[j] for j in inst.pocs)
    name = os.path.splitext(os.path.basename(path))[0]
    doc = {
        "instance": name,
        "status": status_name,
        "is_optimal": bool(m.Status == GRB.OPTIMAL or gap <= 1e-4),
        "mip_gap": gap,
        "objective": m.ObjVal,
        "F_prime": m.ObjVal - sum_r,
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

    print(f"{name}: obj={doc['objective']:.2f}  F'={doc['F_prime']:.2f}  "
          f"status={doc['status']}  gap={doc['mip_gap']*100:.2f}%  "
          f"{'(optimal-quality)' if doc['is_optimal'] else ''}  ->  {out_path}")
    return doc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python phase1_export.py <instance.txt> "
              "[time_limit_s] [max_shift] [out_dir]")
        sys.exit(1)

    path = sys.argv[1]
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    max_shift = float(sys.argv[3]) if len(sys.argv) > 3 else 480.0
    out_dir = sys.argv[4] if len(sys.argv) > 4 else None

    export_instance(path, time_limit, max_shift, out_dir)
