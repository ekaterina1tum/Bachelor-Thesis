"""
Phase-2 experiment runner.

For every emergency scenario (which references a Phase-1 solution), this:
  1. builds the Phase-2 input -- augmented distance matrix with emergency nodes,
     per-emergency candidate arcs A_m (release-feasible), and the
     outsourcing price f_m = lambda * (c_{0,m} + c_{m,0});
  2. solves the ESCP MILP (build_phase2_model);
  3. saves a result JSON per scenario (decisions, costs, delays, violations).

Parameters come from phase2_params (PI_C, PI_S, LAMBDA, outsourcing_cost).
No service times anywhere; an inserted emergency adds only detour travel time.

Usage
-----
    python phase2_experiment.py                       # base case, all sizes
    python phase2_experiment.py --sizes 25            # one size
    python phase2_experiment.py --lam 1.5 --pi-s 5    # override dials
    python phase2_experiment.py --sizes 25 --limit 20 # quick smoke test
"""

from __future__ import annotations

import os
import csv
import json
import math
import glob
import argparse

import gurobipy as gp
from gurobipy import GRB

from instance import load_coordinates
from phase2_instance import (
    load_phase2_solution, EmergencyRequest, Phase2Arc,
)
from phase2_model import build_phase2_model
import phase2_params as P


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTANCES_ROOT = os.path.join(REPO, "data", "MSCDPinstances")
PHASE1_ROOT = os.path.join(REPO, "data", "phase1_solutions")
SCENARIO_ROOT = os.path.join(REPO, "data", "emergency_scenarios")
OUT_ROOT = os.path.join(REPO, "data", "phase2_results")

SIZE_DIR = {10: "010", 15: "015", 25: "025", 30: "030", 35: "035", 40: "040"}
DEPOT = 0
EM_NODE_OFFSET = 100000   # keep emergency node ids clear of instance node ids
TAU_MAX = 480.0

_coord_cache: dict[str, dict] = {}


def instance_coords(size: int, instance: str) -> dict:
    if instance not in _coord_cache:
        path = os.path.join(INSTANCES_ROOT, SIZE_DIR[size], f"{instance}.txt")
        _coord_cache[instance] = load_coordinates(path)[0]
    return _coord_cache[instance]


def euclid(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# --------------------------------------------------------------------------- #
# Build the Phase-2 input (the "bridge") for one scenario
# --------------------------------------------------------------------------- #
def build_inputs(sol_doc: dict, scenario: dict, coords: dict, lam: float):
    """Return (emergency_requests, travel_time, node_of) for one scenario."""

    # ---- arc lookup + z0 from the Phase-1 solution ----
    arc_by_id = {}
    for t in sol_doc["trips"]:
        for a in t["arcs"]:
            arc_by_id[a["id"]] = Phase2Arc(
                id=a["id"], src=a["src"], tgt=a["tgt"],
                trip_id=t["id"], route_id=t["route_id"],
            )
    z0 = {r["id"]: r["z0"] for r in sol_doc["regular_requests"]}

    def baseline_time_at_source(arc: Phase2Arc) -> float:
        if arc.src != DEPOT:
            return z0.get(arc.src, 0.0)
        zb = z0.get(arc.tgt)
        if zb is None:
            return 0.0
        return max(0.0, zb - euclid(coords[DEPOT], coords[arc.tgt]))

    # ---- distance matrix over instance nodes ----
    travel_time: dict[tuple[int, int], float] = {}
    nodes = list(coords)
    for i in nodes:
        for j in nodes:
            travel_time[(i, j)] = euclid(coords[i], coords[j])

    # ---- add emergency nodes + their distances ----
    em_coords = {}
    node_of = {}   # original emergency id -> unique node id
    for em in scenario["emergencies"]:
        nid = EM_NODE_OFFSET + em["id"]
        node_of[em["id"]] = nid
        em_coords[nid] = (em["x"], em["y"])

    all_nodes = nodes + list(em_coords)
    coord_all = dict(coords)
    coord_all.update(em_coords)
    for nid in em_coords:
        for j in all_nodes:
            travel_time[(nid, j)] = euclid(coord_all[nid], coord_all[j])
            travel_time[(j, nid)] = euclid(coord_all[j], coord_all[nid])

    # ---- per-emergency candidate arcs (release-feasible) + f_m ----
    emergencies = {}
    for em in scenario["emergencies"]:
        nid = node_of[em["id"]]
        rho = em["release"]
        A_m = [arc for arc in arc_by_id.values()
               if baseline_time_at_source(arc) >= rho - 1e-9]
        f_m = P.outsourcing_cost(nid, travel_time, lam=lam, depot=DEPOT)
        emergencies[nid] = EmergencyRequest(
            id=nid, rho=rho, d_bar=em["deadline"], f=f_m, A_m=A_m,
        )

    return emergencies, travel_time, node_of


# --------------------------------------------------------------------------- #
# Solve one scenario
# --------------------------------------------------------------------------- #
def solve_scenario(scen_path: str, lam: float, pi_s: float, pi_c: float) -> dict:
    with open(scen_path) as fh:
        scenario = json.load(fh)

    size = scenario["size"] if scenario["size"] in SIZE_DIR else \
        {10: 10, 15: 15, 25: 25}.get(scenario["size"], scenario["size"])
    # size in scenario is the number of customers; map to folder size
    folder_size = next((s for s in SIZE_DIR if s == size), size)
    instance = scenario["instance"]

    sol_path = os.path.join(PHASE1_ROOT, SIZE_DIR[folder_size], f"{instance}.json")
    if not os.path.exists(sol_path):
        return {"instance": instance, "error": "no phase1 solution"}
    with open(sol_path) as fh:
        sol_doc = json.load(fh)

    coords = instance_coords(folder_size, instance)
    emergencies, travel_time, node_of = build_inputs(sol_doc, scenario, coords, lam)

    # per-request penalties: pi_S for TW violation, pi_C for completion increase
    reg_ids = [r["id"] for r in sol_doc["regular_requests"]]
    penalty_tw = {j: pi_s for j in reg_ids}
    penalty_c = {j: pi_c for j in reg_ids}

    inst = load_phase2_solution(
        sol_path, emergencies, travel_time,
        penalty_tw=penalty_tw, penalty_c=penalty_c, tau_max=TAU_MAX,
    )
    model = build_phase2_model(inst)
    model.setParam("OutputFlag", 0)
    model.optimize()

    o, a, v, Delta = model._o, model._a, model._v, model._Delta

    def has_feasible_arc(nid, em):
        """True if some candidate arc can deliver the emergency by its hard deadline."""
        for arc in em.A_m:
            C0 = inst.trips[arc.trip_id].C0
            delta = inst.delta.get((nid, arc.id), 0.0)
            if C0 + delta <= em.d_bar + 1e-9:
                return True
        return False

    # ---- decisions per emergency (with reason) ----
    inv_node = {nid: oid for oid, nid in node_of.items()}
    em_results = []
    n_out = n_out_infeasible = n_out_economic = 0
    for nid, em in inst.emergency_requests.items():
        rec = {"id": inv_node[nid], "node": nid, "f_m": round(em.f, 4)}
        if o[nid].X > 0.5:
            rec["decision"] = "outsourced"
            n_out += 1
            if has_feasible_arc(nid, em):
                rec["reason"] = "economic"      # could be inserted, but outsourcing was cheaper
                n_out_economic += 1
            else:
                rec["reason"] = "infeasible"    # no arc meets release+deadline -> forced
                n_out_infeasible += 1
        else:
            chosen = [eid for (mm, eid), var in a.items() if mm == nid and var.X > 0.5]
            arc = inst.arcs[chosen[0]] if chosen else None
            rec["decision"] = "inserted"
            rec["reason"] = "inserted"
            if arc is not None:
                rec.update({"arc": arc.id, "src": arc.src, "tgt": arc.tgt,
                            "trip": arc.trip_id, "route": arc.route_id,
                            "delta": round(inst.delta.get((nid, arc.id), 0.0), 4)})
        em_results.append(rec)

    # ---- cost breakdown ----
    outsourcing = sum(inst.emergency_requests[nid].f * o[nid].X
                      for nid in inst.emergency_requests)
    tw_pen = sum(penalty_tw[j] * v[j].X for j in reg_ids)
    comp_pen = sum(Delta[tid].X * sum(inst.regular_requests[j].pi_c for j in reg_ids
                                      if inst.regular_requests[j].trip_id == tid)
                   for tid in inst.trips)
    tw_viol = [{"request": j, "v": round(v[j].X, 4)} for j in reg_ids if v[j].X > 1e-6]

    return {
        "instance": instance,
        "size": folder_size,
        "class": scenario["class"],
        "level": scenario["level"],
        "seed_idx": scenario["seed_idx"],
        "W": scenario["W"],
        "params": {"PI_C": pi_c, "PI_S": pi_s, "LAMBDA": lam},
        "status": "OPTIMAL" if model.Status == GRB.OPTIMAL else str(model.Status),
        "objective": round(model.ObjVal, 4) if model.SolCount else None,
        "n_emergencies": len(em_results),
        "n_outsourced": n_out,
        "n_outsourced_infeasible": n_out_infeasible,
        "n_outsourced_economic": n_out_economic,
        "n_inserted": len(em_results) - n_out,
        "frac_outsourced": n_out / len(em_results) if em_results else 0.0,
        "cost_breakdown": {
            "outsourcing": round(outsourcing, 4),
            "tw_penalty": round(tw_pen, 4),
            "completion_penalty": round(comp_pen, 4),
        },
        "emergencies": em_results,
        "tw_violations": tw_viol,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def optimal_instances_for(size: int) -> set:
    """Names of instances whose Phase-1 solution is optimal (is_optimal == true)."""
    names = set()
    for jf in glob.glob(os.path.join(PHASE1_ROOT, SIZE_DIR[size], "*.json")):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if d.get("is_optimal"):
            names.add(d["instance"])
    return names


def run(sizes, lam, pi_s, pi_c, W, limit=None, only_optimal=False):
    """Run one (W, lambda, pi_s) setting; returns the list of per-scenario summary rows.

    only_optimal : if True, skip scenarios whose instance's Phase-1 solution is not
                   optimal (is_optimal == false, or no solution present).
    """
    tag = f"W{W}_lam{lam}_piS{pi_s}"
    scen_root = os.path.join(SCENARIO_ROOT, f"W{W}")
    summary_rows = []
    n_done = 0

    if not os.path.isdir(scen_root):
        raise FileNotFoundError(
            f"No emergency scenarios found for W={W}: {scen_root}. "
            f"Generate them first with: python src/emergency_generator.py --W {W}"
        )

    print(f"Phase-2 experiment: W={W}  LAMBDA={lam}  PI_S={pi_s}  PI_C={pi_c}"
          f"{'  [only optimal Phase-1]' if only_optimal else ''}")
    total_scenarios = 0
    for size in sizes:
        keep = optimal_instances_for(size) if only_optimal else None
        if only_optimal:
            print(f"  size {size}: {len(keep)} instances have an optimal Phase-1 solution")
        scen_files = sorted(glob.glob(os.path.join(scen_root, SIZE_DIR[size], "*", "*.json")))
        if keep is not None:
            scen_files = [f for f in scen_files
                          if os.path.basename(os.path.dirname(f)) in keep]
        if limit:
            scen_files = scen_files[:limit]
        total_scenarios += len(scen_files)
        for sf in scen_files:
            res = solve_scenario(sf, lam, pi_s, pi_c)
            if "error" in res:
                continue

            scen_name = os.path.splitext(os.path.basename(sf))[0]
            out_dir = os.path.join(OUT_ROOT, tag, SIZE_DIR[size], res["instance"])
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{scen_name}.json"), "w") as fh:
                json.dump(res, fh, indent=2)

            summary_rows.append({
                "instance": res["instance"], "size": size, "class": res["class"],
                "level": res["level"], "seed_idx": res["seed_idx"],
                "status": res["status"], "objective": res["objective"],
                "n_emergencies": res["n_emergencies"], "n_outsourced": res["n_outsourced"],
                "n_outsourced_infeasible": res["n_outsourced_infeasible"],
                "n_outsourced_economic": res["n_outsourced_economic"],
                "n_inserted": res["n_inserted"],
                "frac_outsourced": round(res["frac_outsourced"], 4),
                "outsourcing": res["cost_breakdown"]["outsourcing"],
                "tw_penalty": res["cost_breakdown"]["tw_penalty"],
                "completion_penalty": res["cost_breakdown"]["completion_penalty"],
                "W": W, "LAMBDA": lam, "PI_S": pi_s, "PI_C": pi_c,
            })
            n_done += 1
            if n_done % 500 == 0:
                print(f"  [{tag}] solved {n_done} scenarios ...")

    if total_scenarios == 0:
        raise FileNotFoundError(
            f"No scenario JSON files found for W={W} and sizes={sizes} under {scen_root}. "
            f"Generate them first with: python src/emergency_generator.py --W {W}"
        )

    summary_path = os.path.join(OUT_ROOT, f"summary_{tag}.csv")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "instance", "size", "class", "level", "seed_idx", "status", "objective",
            "n_emergencies", "n_outsourced", "n_outsourced_infeasible",
            "n_outsourced_economic", "n_inserted", "frac_outsourced",
            "outsourcing", "tw_penalty", "completion_penalty",
            "W", "LAMBDA", "PI_S", "PI_C",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  [{tag}] solved {n_done}; summary -> {summary_path}")
    return summary_rows


def aggregate_by_lambda(all_rows):
    """Mean/std of frac_outsourced and objective per (LAMBDA, size, class, level)."""
    import statistics
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["LAMBDA"], r["PI_S"], r["size"], r["class"], r["level"])].append(r)

    out = []
    for (lam, pi_s, size, cls, level), rs in sorted(groups.items()):
        fr = [r["frac_outsourced"] for r in rs]
        obj = [r["objective"] for r in rs if r["objective"] is not None]
        out.append({
            "LAMBDA": lam, "PI_S": pi_s, "size": size, "class": cls, "level": level,
            "n_scenarios": len(rs),
            "frac_outsourced_mean": round(statistics.mean(fr), 4),
            "frac_outsourced_std": round(statistics.pstdev(fr) if len(fr) > 1 else 0.0, 4),
            "objective_mean": round(statistics.mean(obj), 4) if obj else None,
            "objective_std": round(statistics.pstdev(obj) if len(obj) > 1 else 0.0, 4),
        })
    return out


def run_sweep(sizes, lam_grid, pi_s, pi_c, W, limit=None, only_optimal=False):
    """Run the full lambda sweep; per-lambda summaries + one combined comparison CSV."""
    all_rows = []
    for lam in lam_grid:
        all_rows.extend(run(sizes, lam, pi_s, pi_c, W, limit, only_optimal=only_optimal))

    if not all_rows:
        raise RuntimeError(
            f"No Phase-2 rows were produced for W={W}, pi_S={pi_s}. "
            "Check that matching scenarios and Phase-1 solutions exist."
        )

    agg = aggregate_by_lambda(all_rows)
    agg_path = os.path.join(OUT_ROOT, f"lambda_sweep_W{W}_piS{pi_s}.csv")
    with open(agg_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "LAMBDA", "PI_S", "size", "class", "level", "n_scenarios",
            "frac_outsourced_mean", "frac_outsourced_std",
            "objective_mean", "objective_std",
        ])
        w.writeheader()
        w.writerows(agg)

    # console: overall outsource fraction per lambda
    import statistics
    print("\n=== lambda sweep: overall mean outsource fraction ===")
    by_lam = {}
    for r in all_rows:
        by_lam.setdefault(r["LAMBDA"], []).append(r["frac_outsourced"])
    for lam in lam_grid:
        vals = by_lam.get(lam, [])
        if vals:
            print(f"  lambda={lam:>5}: mean frac_outsourced = {statistics.mean(vals):.3f}")
        else:
            print(f"  lambda={lam:>5}: no solved scenarios")
    print(f"\nComparison summary: {agg_path}")
    return agg_path


def run_W_sweep(sizes, W_list, lam, pi_s, pi_c, limit=None):
    """Base-case run for each W; one comparison CSV of feasibility/insertion vs W."""
    import statistics
    from collections import defaultdict
    all_rows = []
    for W in W_list:
        print(f"\n########## W = {W} ##########")
        all_rows.extend(run(sizes, lam, pi_s, pi_c, W, limit))

    # aggregate per (W, class): mean fractions of inserted / outsourced(econ) / outsourced(infeasible)
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["W"], r["class"])].append(r)
    agg = []
    for (W, cls), rs in sorted(groups.items()):
        n_em = sum(r["n_emergencies"] for r in rs)
        agg.append({
            "W": W, "class": cls, "n_scenarios": len(rs), "n_emergencies": n_em,
            "frac_inserted": round(sum(r["n_inserted"] for r in rs) / n_em, 4),
            "frac_out_economic": round(sum(r["n_outsourced_economic"] for r in rs) / n_em, 4),
            "frac_out_infeasible": round(sum(r["n_outsourced_infeasible"] for r in rs) / n_em, 4),
            "frac_outsourced": round(sum(r["n_outsourced"] for r in rs) / n_em, 4),
        })
    agg_path = os.path.join(OUT_ROOT, f"W_sweep_lam{lam}_piS{pi_s}.csv")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(agg_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "W", "class", "n_scenarios", "n_emergencies", "frac_inserted",
            "frac_out_economic", "frac_out_infeasible", "frac_outsourced"])
        w.writeheader(); w.writerows(agg)

    print("\n=== W sweep: fractions by W x class ===")
    print(f"  {'W':>5} {'class':>5} {'inserted':>9} {'out_econ':>9} {'out_infeas':>11}")
    for r in agg:
        print(f"  {r['W']:>5} {r['class']:>5} {r['frac_inserted']:>9.3f} "
              f"{r['frac_out_economic']:>9.3f} {r['frac_out_infeasible']:>11.3f}")
    print(f"\nComparison: {agg_path}")
    return agg_path


def run_full_grid(sizes, lam_grid, piS_grid, pi_c, W, limit=None):
    """Sweep both pi_S (PI_S_GRID) and lambda (LAMBDA_GRID); one lambda-sweep per pi_S."""
    for pi_s in piS_grid:
        print(f"\n########## pi_S = {pi_s} ##########")
        run_sweep(sizes, lam_grid, pi_s, pi_c, W, limit)
    print("\nFull pi_S x lambda grid complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the Phase-2 MILP over all scenarios.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 25])
    ap.add_argument("--lam", type=float, default=P.LAMBDA)
    ap.add_argument("--pi-s", type=float, default=P.PI_S, dest="pi_s")
    ap.add_argument("--pi-c", type=float, default=P.PI_C, dest="pi_c")
    ap.add_argument("--W", type=int, default=60, help="urgency window of the scenario set to use")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of scenarios per size (smoke test)")
    ap.add_argument("--lam-grid", action="store_true",
                    help="sweep LAMBDA over phase2_params.LAMBDA_GRID (single pi_S, single W)")
    ap.add_argument("--full-grid", action="store_true",
                    help="sweep W (W_GRID) x pi_S (PI_S_GRID) x LAMBDA (LAMBDA_GRID)")
    ap.add_argument("--W-sweep", type=int, nargs="+", default=None, dest="W_sweep",
                    help="base-case run for each W in the list, e.g. --W-sweep 60 90")
    ap.add_argument("--only-optimal", action="store_true", dest="only_optimal",
                    help="only run instances whose Phase-1 solution is optimal")
    args = ap.parse_args()
    if args.W_sweep:
        run_W_sweep(args.sizes, args.W_sweep, args.lam, args.pi_s, args.pi_c, args.limit)
    elif args.full_grid:
        for W in P.W_GRID:
            print(f"\n==================== W = {W} ====================")
            run_full_grid(args.sizes, P.LAMBDA_GRID, P.PI_S_GRID, args.pi_c, W, args.limit)
        print("\nAll W full grids complete.")
    elif args.lam_grid:
        run_sweep(args.sizes, P.LAMBDA_GRID, args.pi_s, args.pi_c, args.W, args.limit,
                  only_optimal=args.only_optimal)
    else:
        run(args.sizes, args.lam, args.pi_s, args.pi_c, args.W, args.limit,
            only_optimal=args.only_optimal)
