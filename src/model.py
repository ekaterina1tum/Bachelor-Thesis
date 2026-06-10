
import gurobipy as gp
from gurobipy import GRB

from instance import Instance
from graph import ExtendedGraph, build_graph


def build_model(inst: Instance, graph: ExtendedGraph) -> gp.Model:
    m = gp.Model("CDSP")

    # ---- Index sets (for readability) ----
    P = inst.pocs              # set of PoCs
    depot = inst.depot         # node 0
    A0 = graph.A0              # depot-adjacent arcs
    AP = graph.AP              # PoC-to-PoC arcs
    AR = graph.AR              # replenishment arcs
    A = graph.all_arcs         # full arc set

    # ---- Variables ----
    # x_e ∈ {0, 1} for every arc e
    x = m.addVars(A, vtype=GRB.BINARY, name="x")

    # y_{i,j} ∈ {0, 1} for every ordered pair of PoCs (including i = j)
    y = m.addVars(P, P, vtype=GRB.BINARY, name="y")

    # z_j, tau_j, C_j ≥ 0 for every PoC
    z = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="z")
    tau = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="tau")
    C = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="C")

    # Stash the variables on the model for later access
    m._x, m._y, m._z, m._tau, m._C = x, y, z, tau, C
    m._inst, m._graph = inst, graph

    # ---- Objective: (1) minimize sum of completion times ----
    m.setObjective(gp.quicksum(C[j] for j in P), GRB.MINIMIZE)

    # Constraint (2): flow at the depot
    depot_out = [e for e in A0 if e.src == depot]
    depot_in = [e for e in A0 if e.tgt == depot]

    m.addConstr(
        gp.quicksum(x[e] for e in depot_out)
        == gp.quicksum(x[e] for e in depot_in),
        name="flow_depot_balance"
    )
    m.addConstr(
        gp.quicksum(x[e] for e in depot_out) <= inst.fleet_size,
        name="flow_depot_capacity"
    )

    # Constraint (3): flow at every PoC
    for j in P:
        out_j = [e for e in A if e.src == j]
        in_j = [e for e in A if e.tgt == j]
        m.addConstr(gp.quicksum(x[e] for e in out_j) == 1, name=f"flow_out_poc_{j}")
        m.addConstr(gp.quicksum(x[e] for e in in_j) == 1, name=f"flow_in_poc_{j}")

    # Constraint (4): time propagation along arcs
    for e in AP + AR:
        i, j = e.src, e.tgt
        M_e = max(0.0, inst.due_time[i] + e.cost - inst.release_time[j])
        m.addConstr(
            z[i] + e.cost <= z[j] + M_e * (1 - x[e]),
            name=f"time_prop_{i}_{j}_{e.kind}"
        )

    # Constraint (5): time windows
    for j in P:
        r_j = max(inst.release_time[j], inst.c(depot, j))
        d_j = min(inst.due_time[j], inst.depot_deadline - inst.c(j, depot))

        m.addConstr(z[j] >= r_j, name=f"tw_lower_{j}")
        m.addConstr(z[j] <= d_j, name=f"tw_upper_{j}")

    # Constraint (6): y_{j,j} = 1 for all j in P

    for j in P:
        m.addConstr(y[j, j] == 1, name=f"y_self_{j}")

    # Constraint (7): y propagation along PoC-to-PoC arcs
    for e in AP:
        for j in P:
            if j == e.tgt:
                continue  # skip the case j = t(e) per the formulation
            m.addConstr(
                y[e.src, j] <= y[e.tgt, j] + (1 - x[e]),
                name=f"y_prop_{e.src}_{e.tgt}_j{j}"
            )

    # Constraint (8): completion time lower bound

    for i in P:
        c_i_depot = inst.c(i, depot)
        M_i = inst.due_time[i] + c_i_depot
        for j in P:
            m.addConstr(
                z[i] + c_i_depot <= C[j] + M_i * (1 - y[i, j]),
                name=f"completion_{i}_{j}"
            )

    # Constraint (9): tau initialization

    for j in P:
        m.addConstr(tau[j] >= inst.c(depot, j), name=f"tau_init_{j}")

    # Constraint (10): tau propagation along arcs

    for e in AP + AR:
        i, j = e.src, e.tgt
        M_prime_e = inst.due_time[j] - inst.c(depot, j)
        m.addConstr(
            tau[i] + (z[j] - z[i]) <= tau[j] + M_prime_e * (1 - x[e]),
            name=f"tau_prop_{i}_{j}_{e.kind}"
        )

    # Constraint (11): maximum shift length
    #   tau_j + c_{j,0} <= tau_max   for all j in P
    for j in P:
        m.addConstr(
            tau[j] + inst.c(j, depot) <= inst.max_shift,
            name=f"tau_max_{j}"
        )

    return m
def solve_and_print(m: gp.Model) -> None:
    inst = m._inst
    graph = m._graph
    P = inst.pocs
    depot = inst.depot

    # Solve
    m.optimize()

    # Check status
    if m.Status != GRB.OPTIMAL:
        print(f"\nSolver did not find an optimal solution. Status: {m.Status}")
        return

    print("\n" + "=" * 60)
    print(f"Objective = {m.ObjVal:.2f}")
    print("=" * 60)

    # Used arcs
    print("\nUsed arcs (x_e = 1):")
    used_arcs = [e for e in graph.all_arcs if m._x[e].X > 0.5]
    for e in used_arcs:
        print(f"  {e}")

    # Visit times, shift times, completion times
    print(f"\n{'PoC':>4}  {'z_j':>8}  {'tau_j':>8}  {'C_j':>8}")
    print("  " + "-" * 32)
    for j in P:
        print(
            f"  {j:>2}  "
            f"{m._z[j].X:>8.2f}  "
            f"{m._tau[j].X:>8.2f}  "
            f"{m._C[j].X:>8.2f}"
        )

    # Y matrix (downstream markers)
    print(f"\ny_{{i,j}} matrix (rows = i, columns = j):")
    print("       " + "  ".join(f"j={j}" for j in P))
    for i in P:
        row = "  ".join(f" {int(m._y[i, j].X):>2}" for j in P)
        print(f"  i={i}  {row}")

if __name__ == "__main__":
    from instance import make_2trip_example

    inst = make_2trip_example()
    g = build_graph(inst)
    m = build_model(inst, g)

    m.update()   # <-- flush the lazy queue so the counters update

    print("Model built successfully.")
    print(f"  Variables: {m.NumVars}")
    print(f"  Constraints: {m.NumConstrs}")
    solve_and_print(m)
