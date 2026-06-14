"""
ESCP Phase-2 model: Emergency Sample Collection Problem.

Given a fixed Phase-1 solution (routes, trips, baseline timings), decides for
each emergency request whether to insert it into an existing route or outsource
it to a 3PL provider, minimising outsourcing costs + penalty on disruption.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from phase2_instance import Phase2Instance


def build_phase2_model(inst: Phase2Instance) -> gp.Model:
    """Build and return the ESCP Gurobi model (not yet solved)."""

    model = gp.Model("ESCP")

    # ------------------------------------------------------------------ #
    # Index sets
    # ------------------------------------------------------------------ #
    E = list(inst.emergency_requests.keys())      # emergency indices m
    P = list(inst.regular_requests.keys())         # regular request indices j
    T_all = list(inst.trips.keys())                # all trip indices
    K = list(inst.routes.keys())                   # route indices k

    # A_m: candidate arcs per emergency (as arc ids)
    A_m: dict[int, list[int]] = {
        m: [arc.id for arc in inst.emergency_requests[m].A_m]
        for m in E
    }

    # ------------------------------------------------------------------ #
    # Decision variables
    # ------------------------------------------------------------------ #

    # o_m ∈ {0,1}: 1 iff emergency m is outsourced  (3)
    o = model.addVars(E, vtype=GRB.BINARY, name="o")

    # a_{m,e} ∈ {0,1}: 1 iff emergency m inserted on arc e  (3)
    # Only create for feasible (m, e) pairs
    a = {}
    for m in E:
        for eid in A_m[m]:
            a[m, eid] = model.addVar(vtype=GRB.BINARY, name=f"a_{m}_{eid}")

    # v_j ≥ 0: TW violation of regular request j  (8)
    v = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="v")

    # Delta_T ≥ 0: completion-time increase of trip T  (7)
    Delta = model.addVars(T_all, vtype=GRB.CONTINUOUS, lb=0.0, name="Delta")

    # D_T ≥ 0: inherited delay at start of trip T  (5,6)
    D = model.addVars(T_all, vtype=GRB.CONTINUOUS, lb=0.0, name="D")

    # Store references on model for post-processing
    model._o = o
    model._a = a
    model._v = v
    model._Delta = Delta
    model._D = D
    model._inst = inst

    # ------------------------------------------------------------------ #
    # Theta_T: total extra travel time inserted into trip T  (1)
    # Theta_T = sum_{e in E_T} sum_{m in E} delta_{m,e} * a_{m,e}
    # ------------------------------------------------------------------ #
    def theta_expr(trip_id: int) -> gp.LinExpr:
        """Linear expression for Theta_T."""
        expr = gp.LinExpr()
        for eid in inst.trip_arcs.get(trip_id, []):
            for m in E:
                if (m, eid) in a:
                    delta_val = inst.delta.get((m, eid), 0.0)
                    if delta_val != 0.0:
                        expr.add(a[m, eid], delta_val)
        return expr

    # ------------------------------------------------------------------ #
    # Objective (2)
    # min  sum_m f_m * o_m
    #    + sum_j pi^TW_j * v_j
    #    + sum_T Pi^C_T * Delta_T
    # ------------------------------------------------------------------ #
    obj = gp.LinExpr()

    # Outsourcing cost
    for m in E:
        obj.add(o[m], inst.emergency_requests[m].f)

    # TW violation penalty
    for j in P:
        obj.add(v[j], inst.regular_requests[j].pi_tw)

    # Completion-time increase penalty
    # Pi^C_T = sum_{j: T(j)=T} pi^C_j
    Pi_C: dict[int, float] = {}
    for j in P:
        tid = inst.regular_requests[j].trip_id
        Pi_C[tid] = Pi_C.get(tid, 0.0) + inst.regular_requests[j].pi_c
    for tid in T_all:
        if Pi_C.get(tid, 0.0) != 0.0:
            obj.add(Delta[tid], Pi_C[tid])

    model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # Constraint (3): each emergency is outsourced or inserted exactly once
    # o_m + sum_{e in A_m} a_{m,e} = 1   for all m in E
    # ------------------------------------------------------------------ #
    for m in E:
        lhs = o[m] + gp.quicksum(a[m, eid] for eid in A_m[m])
        model.addConstr(lhs == 1, name=f"handle_{m}")

    # ------------------------------------------------------------------ #
    # Constraint (4): at most one emergency per arc
    # sum_{m in E} a_{m,e} <= 1   for all e in A
    # ------------------------------------------------------------------ #
    all_candidate_arc_ids: set[int] = set()
    for m in E:
        all_candidate_arc_ids.update(A_m[m])

    for eid in all_candidate_arc_ids:
        lhs = gp.quicksum(a[m, eid] for m in E if (m, eid) in a)
        model.addConstr(lhs <= 1, name=f"one_per_arc_{eid}")

    # ------------------------------------------------------------------ #
    # Constraint (5): first trips start with zero inherited delay
    # D_T = 0   for all T in T_0
    # ------------------------------------------------------------------ #
    for tid in inst.T0:
        model.addConstr(D[tid] == 0.0, name=f"D_init_{tid}")

    # ------------------------------------------------------------------ #
    # Constraint (6): delay propagation to successor trip
    # D_{T+} >= D_T + Theta_T - g_T   for all T not in T^last
    # ------------------------------------------------------------------ #
    for tid, trip in inst.trips.items():
        if trip.is_last:
            continue
        next_tid = trip.next_trip_id
        if next_tid is None:
            continue
        theta_T = theta_expr(tid)
        model.addConstr(
            D[next_tid] >= D[tid] + theta_T - trip.g,
            name=f"delay_prop_{tid}_{next_tid}",
        )

    # ------------------------------------------------------------------ #
    # Constraint (7): completion-time increase per trip
    # Delta_T >= D_T + Theta_T   for all T
    # ------------------------------------------------------------------ #
    for tid in T_all:
        theta_T = theta_expr(tid)
        model.addConstr(
            Delta[tid] >= D[tid] + theta_T,
            name=f"delta_{tid}",
        )

    # ------------------------------------------------------------------ #
    # Constraint (8): soft TW violation of regular requests
    # v_j >= z^0_j + D_{T(j)} + sum_{e in B_j} sum_m delta_{m,e} a_{m,e} - d_j
    # ------------------------------------------------------------------ #
    for j in P:
        rr = inst.regular_requests[j]
        tid = rr.trip_id

        before_expr = gp.LinExpr()
        for arc in rr.B:
            for m in E:
                if (m, arc.id) in a:
                    dval = inst.delta.get((m, arc.id), 0.0)
                    if dval != 0.0:
                        before_expr.add(a[m, arc.id], dval)

        model.addConstr(
            v[j] >= rr.z0 + D[tid] + before_expr - rr.d,
            name=f"tw_viol_{j}",
        )

    # ------------------------------------------------------------------ #
    # Constraint (9): emergency hard delivery deadline
    # C^0_T + D_T + Theta_T <= d_bar_m + M*(1 - sum_{e in E_T} a_{m,e})
    #   for all m in E, T in T_m
    # ------------------------------------------------------------------ #
    for m in E:
        em = inst.emergency_requests[m]
        for tid in inst.T_m[m]:
            trip = inst.trips[tid]
            # arcs of T that are in A_m
            ET_arcs_in_Am = [eid for eid in inst.trip_arcs.get(tid, []) if (m, eid) in a]
            theta_T = theta_expr(tid)
            sum_a_in_T = gp.quicksum(a[m, eid] for eid in ET_arcs_in_Am)
            model.addConstr(
                trip.C0 + D[tid] + theta_T
                <= em.d_bar + inst.M_big * (1 - sum_a_in_T),
                name=f"hard_deadline_{m}_{tid}",
            )

    # ------------------------------------------------------------------ #
    # Constraint (10): shift-duration feasibility
    # Lambda^0_k + D_{T^last_k} + Theta_{T^last_k} <= tau_max   for all k
    # ------------------------------------------------------------------ #
    last_trip_of_route: dict[int, int] = {}
    for tid, trip in inst.trips.items():
        if trip.is_last:
            last_trip_of_route[trip.route_id] = tid

    for k in K:
        tid = last_trip_of_route.get(k)
        if tid is None:
            continue
        theta_T = theta_expr(tid)
        model.addConstr(
            inst.Lambda0[k] + D[tid] + theta_T <= inst.tau_max,
            name=f"shift_max_{k}",
        )

    return model


def solve_and_print(model: gp.Model) -> None:
    """Solve the model and print a summary of the solution."""
    inst: Phase2Instance = model._inst

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print(f"\nSolver status: {model.Status} (no optimal solution found)")
        return

    print("\n" + "=" * 60)
    print(f"ESCP Objective = {model.ObjVal:.4f}")
    print("=" * 60)

    o = model._o
    a = model._a
    v = model._v
    Delta = model._Delta
    D = model._D

    print("\nEmergency handling decisions:")
    for m, em in inst.emergency_requests.items():
        if o[m].X > 0.5:
            print(f"  Emergency {m}: OUTSOURCED  (cost {em.f})")
        else:
            chosen = [eid for (mm, eid), var in a.items() if mm == m and var.X > 0.5]
            if chosen:
                arc = inst.arcs[chosen[0]]
                print(f"  Emergency {m}: inserted on arc {arc.src}->{arc.tgt} "
                      f"(trip {arc.trip_id}, route {arc.route_id})")

    print("\nTrip delays and completion-time increases:")
    print(f"  {'Trip':>5}  {'D_T':>8}  {'Delta_T':>8}")
    print("  " + "-" * 26)
    for tid in sorted(inst.trips.keys()):
        print(f"  {tid:>5}  {D[tid].X:>8.3f}  {Delta[tid].X:>8.3f}")

    print("\nRegular-request TW violations:")
    violated = [(j, v[j].X) for j in inst.regular_requests if v[j].X > 1e-6]
    if violated:
        for j, viol in violated:
            print(f"  Request {j}: violation = {viol:.3f}")
    else:
        print("  None")

    outsource_cost = sum(
        inst.emergency_requests[m].f * o[m].X
        for m in inst.emergency_requests
    )
    tw_penalty = sum(
        inst.regular_requests[j].pi_tw * v[j].X
        for j in inst.regular_requests
    )
    ct_penalty = sum(
        Delta[tid].X * sum(
            inst.regular_requests[j].pi_c
            for j in inst.regular_requests
            if inst.regular_requests[j].trip_id == tid
        )
        for tid in inst.trips
    )
    print(f"\nCost breakdown:")
    print(f"  Outsourcing:            {outsource_cost:.4f}")
    print(f"  TW violation penalty:   {tw_penalty:.4f}")
    print(f"  Completion-time penalty:{ct_penalty:.4f}")
