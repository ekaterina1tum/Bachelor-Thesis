"""Minimal standalone example for Phase 2 (ESCP).

The toy instance is deliberately tiny and deterministic. It shows both possible
Phase-2 decisions:

* emergency 10 is inserted into an existing trip;
* emergency 11 is outsourced because outsourcing is cheaper than the disruption
  caused by inserting it.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from phase2_instance import (
    Phase2Arc, Trip, RegularRequest, EmergencyRequest, Phase2Instance
)
from phase2_model import build_phase2_model


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUTDIR = os.path.join(REPO, "data", "phase2_results", "phase2_toy_example")
LAMBDA = 2.5
PI_TW = 2.0
PI_C = 1.0
DEPOT = 0
COORDS = {
    0: (0.0, 0.0),
    1: (5.0, 2.0),
    2: (10.0, 2.0),
    3: (2.0, 8.0),
    10: (7.5, 2.7),
    11: (-1.2, 1.0),
}


def euclid(i: int, j: int) -> float:
    xi, yi = COORDS[i]
    xj, yj = COORDS[j]
    return math.hypot(xi - xj, yi - yj)


def make_example() -> Phase2Instance:
    # Travel times are Euclidean distances from the coordinates used in the graph.
    nodes = list(COORDS)
    tt = {}
    for i in nodes:
        for j in nodes:
            tt[(i, j)] = euclid(i, j)

    # ---- Arcs in the Phase-1 solution ----
    # Trip 0 arcs: 0->1, 1->2, 2->0
    arc0 = Phase2Arc(id=0, src=0, tgt=1, trip_id=0, route_id=0)
    arc1 = Phase2Arc(id=1, src=1, tgt=2, trip_id=0, route_id=0)
    arc2 = Phase2Arc(id=2, src=2, tgt=0, trip_id=0, route_id=0)
    # Trip 1 arcs: 0->3, 3->0
    arc3 = Phase2Arc(id=3, src=0, tgt=3, trip_id=1, route_id=0)
    arc4 = Phase2Arc(id=4, src=3, tgt=0, trip_id=1, route_id=0)

    # ---- Trips ----
    z1 = tt[(0, 1)]
    z2 = z1 + tt[(1, 2)]
    C0 = z2 + tt[(2, 0)]
    g0 = 2.0
    z3 = C0 + g0 + tt[(0, 3)]
    C1 = z3 + tt[(3, 0)]

    trip0 = Trip(id=0, route_id=0, arcs=[arc0, arc1, arc2],
                 C0=C0, g=g0, is_first=True, is_last=False, next_trip_id=1)
    trip1 = Trip(id=1, route_id=0, arcs=[arc3, arc4],
                 C0=C1, g=0.0, is_first=False, is_last=True, next_trip_id=None)

    # ---- Regular requests ----
    # P1 has no slack; P2 has one unit of slack; P3 has a wide soft window.
    rr1 = RegularRequest(id=1, trip_id=0, z0=z1, d=z1, pi_tw=PI_TW, pi_c=PI_C,
                         B=[arc0])
    rr2 = RegularRequest(id=2, trip_id=0, z0=z2, d=z2 + 1.0, pi_tw=PI_TW, pi_c=PI_C,
                         B=[arc0, arc1])
    rr3 = RegularRequest(id=3, trip_id=1, z0=z3, d=z3 + 10.0, pi_tw=PI_TW, pi_c=PI_C,
                         B=[arc3])

    # ---- Emergency requests ----
    # Outsourcing cost follows f_m = lambda * (c_0m + c_m0).
    f10 = LAMBDA * (tt[(DEPOT, 10)] + tt[(10, DEPOT)])
    f11 = LAMBDA * (tt[(DEPOT, 11)] + tt[(11, DEPOT)])
    # Emergency 10: insertion is cheaper than outsourcing.
    em10 = EmergencyRequest(id=10, rho=10.0, d_bar=40.0, f=f10, A_m=[arc1])
    # Emergency 11: outsourcing is cheap; inserting it after P3 would miss the
    # emergency hard deadline in this toy example.
    em11 = EmergencyRequest(id=11, rho=0.0, d_bar=39.5, f=f11, A_m=[arc4])

    return Phase2Instance(
        routes={0: [0, 1]},
        trips={0: trip0, 1: trip1},
        regular_requests={1: rr1, 2: rr2, 3: rr3},
        Lambda0={0: C1},
        tau_max=50.0,
        emergency_requests={10: em10, 11: em11},
        travel_time=tt,
    )


def solve_example(inst: Phase2Instance) -> dict:
    model = build_phase2_model(inst)
    model.setParam("OutputFlag", 0)
    model.optimize()

    o, a, v, Delta, D = model._o, model._a, model._v, model._Delta, model._D
    decisions = []
    for m, em in inst.emergency_requests.items():
        chosen = [eid for (mm, eid), var in a.items() if mm == m and var.X > 0.5]
        if chosen:
            arc = inst.arcs[chosen[0]]
            decisions.append({
                "id": m,
                "decision": "inserted",
                "arc": arc.id,
                "src": arc.src,
                "tgt": arc.tgt,
                "trip": arc.trip_id,
                "delta": inst.delta[(m, arc.id)],
                "outsourcing_cost": em.f,
            })
        else:
            has_deadline_feasible_arc = any(
                inst.trips[arc.trip_id].C0 + inst.delta[(m, arc.id)] <= em.d_bar + 1e-9
                for arc in em.A_m
            )
            decisions.append({
                "id": m,
                "decision": "outsourced",
                "reason": "economic" if has_deadline_feasible_arc else "infeasible",
                "outsourcing_cost": em.f,
            })

    outsourcing = sum(inst.emergency_requests[m].f * o[m].X for m in inst.emergency_requests)
    tw_penalty = sum(inst.regular_requests[j].pi_tw * v[j].X for j in inst.regular_requests)
    completion_penalty = sum(
        Delta[tid].X
        * sum(rr.pi_c for rr in inst.regular_requests.values() if rr.trip_id == tid)
        for tid in inst.trips
    )

    return {
        "status": int(model.Status),
        "objective": round(model.ObjVal, 4),
        "params": {"lambda": LAMBDA, "pi_TW": PI_TW, "pi_C": PI_C},
        "decisions": decisions,
        "trip_delays": {
            tid: {"D": round(D[tid].X, 4), "Delta": round(Delta[tid].X, 4)}
            for tid in inst.trips
        },
        "regular_tw_violations": {
            j: round(v[j].X, 4)
            for j in inst.regular_requests
            if v[j].X > 1e-7
        },
        "cost_breakdown": {
            "outsourcing": round(outsourcing, 4),
            "tw_penalty": round(tw_penalty, 4),
            "completion_penalty": round(completion_penalty, 4),
        },
    }


def draw_example(inst: Phase2Instance, result: dict) -> str:
    coords = COORDS
    base_arcs = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 0)]
    inserted = {d["arc"]: d for d in result["decisions"] if d["decision"] == "inserted"}
    arc_ids = {(0, 1): 0, (1, 2): 1, (2, 0): 2, (0, 3): 3, (3, 0): 4}
    travel_time = {(i, j): euclid(i, j) for i in coords for j in coords}
    z1 = euclid(0, 1)
    z2 = z1 + euclid(1, 2)
    trip0_completion = z2 + euclid(2, 0)
    z3 = trip0_completion + 2.0 + euclid(0, 3)
    trip1_completion = z3 + euclid(3, 0)
    rejected_delta = euclid(3, 11) + euclid(11, 0) - euclid(3, 0)
    rejected_arrival = trip1_completion + rejected_delta
    rejected_deadline = inst.emergency_requests[11].d_bar

    def fmt(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def label_edge(src: int, tgt: int, value: float, color: str, offset: float = 0.0):
        x1, y1 = coords[src]
        x2, y2 = coords[tgt]
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx = -dy / length
        ny = dx / length
        ax.text(
            mx + nx * offset,
            my + ny * offset,
            fmt(value),
            color=color,
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=8,
        )

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for src, tgt in base_arcs:
        arc_id = arc_ids[(src, tgt)]
        if arc_id in inserted:
            em = inserted[arc_id]["id"]
            legs = [(src, em), (em, tgt)]
            for leg_src, leg_tgt in legs:
                ax.annotate(
                    "",
                    xy=coords[leg_tgt],
                    xytext=coords[leg_src],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#e67e22",
                        lw=2.3,
                        linestyle="dashed",
                        shrinkA=9,
                        shrinkB=9,
                    ),
                    zorder=3,
                )
                label_edge(leg_src, leg_tgt, travel_time[(leg_src, leg_tgt)], "#b35400", offset=0.2)
        else:
            ax.annotate(
                "",
                xy=coords[tgt],
                xytext=coords[src],
                arrowprops=dict(arrowstyle="-|>", color="0.65", lw=1.4, shrinkA=9, shrinkB=9),
                zorder=1,
            )
            if (src, tgt) != (3, 0):
                label_edge(src, tgt, travel_time[(src, tgt)], "0.35", offset=0.18)

    # Outsourcing distance for E11: the cost is lambda * (depot -> E11 -> depot).
    outsource_color = "#7b3294"
    for src, tgt, rad, label_offset in [
        (0, 11, -0.25, -0.18),
        (11, 0, -0.45, 0.18),
    ]:
        ax.annotate(
            "",
            xy=coords[tgt],
            xytext=coords[src],
            arrowprops=dict(
                arrowstyle="-|>",
                color=outsource_color,
                lw=1.9,
                linestyle=":",
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=11,
                shrinkB=11,
            ),
            zorder=2,
        )
        label_edge(src, tgt, travel_time[(src, tgt)], outsource_color, offset=label_offset)

    # Rejected in-house option for E11: insert it after P3 on arc P3 -> depot.
    rejected_color = "#c51b7d"
    rejected_legs = [(3, 11), (11, 0)]
    for leg_src, leg_tgt in rejected_legs:
        ax.annotate(
            "",
            xy=coords[leg_tgt],
            xytext=coords[leg_src],
            arrowprops=dict(
                arrowstyle="-|>",
                color=rejected_color,
                lw=1.8,
                linestyle="-.",
                shrinkA=12,
                shrinkB=12,
            ),
            zorder=2,
        )
    label_edge(3, 11, travel_time[(3, 11)], rejected_color, offset=-0.22)
    ax.annotate(
        f"Rejected insertion after P3\narrival {fmt(rejected_arrival)} > deadline {fmt(rejected_deadline)}",
        xy=coords[11],
        xytext=(28, -42),
        textcoords="offset points",
        fontsize=9,
        color=rejected_color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=rejected_color, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=rejected_color, lw=1.0),
        zorder=9,
    )

    regular = [1, 2, 3]
    ax.scatter([coords[i][0] for i in regular], [coords[i][1] for i in regular],
               s=115, c="#2d7fb8", edgecolors="black", zorder=5, label="regular request")
    ax.scatter(*coords[0], s=170, marker="s", c="#d62728", edgecolors="black", zorder=6, label="depot / lab")

    node_labels = {
        0: "Depot",
        1: f"P1\nTW[0,{fmt(inst.regular_requests[1].d)}]",
        2: f"P2\nTW[0,{fmt(inst.regular_requests[2].d)}]",
        3: f"P3\nTW[0,{fmt(inst.regular_requests[3].d)}]",
    }
    node_offsets = {
        0: (7, 7),
        1: (7, -20),
        2: (7, -20),
        3: (7, 10),
    }
    for node, label in node_labels.items():
        ax.annotate(
            label,
            coords[node],
            xytext=node_offsets[node],
            textcoords="offset points",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75)
            if node != 0 else None,
        )

    arrival_color = "#005f99"
    arrival_offsets = {
        1: (7, -44),
        2: (7, -44),
        3: (7, -28),
    }
    for node in [1, 2, 3]:
        rr = inst.regular_requests[node]
        ax.annotate(
            f"arr={fmt(rr.z0)}",
            coords[node],
            xytext=arrival_offsets[node],
            textcoords="offset points",
            fontsize=9.5,
            color=arrival_color,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=arrival_color, alpha=0.88),
            zorder=9,
        )

    for d in result["decisions"]:
        x, y = coords[d["id"]]
        if d["decision"] == "inserted":
            ax.scatter(x, y, s=190, marker="^", c="#e67e22", edgecolors="black", zorder=7, label="inserted emergency")
            em = inst.emergency_requests[d["id"]]
            label = f"E{d['id']} inserted\nrho={fmt(em.rho)}, W={fmt(em.d_bar - em.rho)}"
            color = "#b35400"
            offset = (10, -26)
        else:
            ax.scatter(x, y, s=190, marker="X", c="#c0392b", edgecolors="black", zorder=7, label="outsourced emergency")
            em = inst.emergency_requests[d["id"]]
            label = f"E{d['id']} outsourced\nrho={fmt(em.rho)}, W={fmt(em.d_bar - em.rho)}"
            color = "#8e1b12"
            offset = (14, 10)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=10, color=color)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.plot([], [], color="0.65", lw=1.4, label="Phase-1 route")
    ax.plot([], [], color="#e67e22", lw=2.3, linestyle="dashed", label="inserted detour")
    ax.plot([], [], color=outsource_color, lw=1.9, linestyle=":", label="outsourcing distance")
    ax.plot([], [], color=rejected_color, lw=1.8, linestyle="-.", label="rejected insertion")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="center left", bbox_to_anchor=(1.02, 0.55), frameon=True)

    cb = result["cost_breakdown"]
    ax.set_title(
        "Phase 2 toy example: one inserted, one outsourced\n"
        f"objective={result['objective']}, outsourcing={cb['outsourcing']}, "
        f"TW={cb['tw_penalty']}, completion={cb['completion_penalty']} "
        f"(lambda={LAMBDA}, pi_TW={PI_TW}, pi_C={PI_C})",
        fontsize=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.2, 12.0)
    ax.set_ylim(-0.8, 9.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.45)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "phase2_toy_example_graph.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(inst: Phase2Instance, result: dict, graph_path: str) -> str:
    def fmt(x: float) -> str:
        return f"{x:.4f}".rstrip("0").rstrip(".")

    os.makedirs(OUTDIR, exist_ok=True)
    json_path = os.path.join(OUTDIR, "phase2_toy_example_result.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)

    path = os.path.join(OUTDIR, "phase2_toy_example_report.md")
    with open(path, "w") as fh:
        fh.write("# Small Phase 2 Correctness Example\n\n")
        fh.write("This example uses the real Phase 2 MILP implementation on a tiny hand-built instance.\n\n")
        fh.write(f"Parameters: `lambda = {LAMBDA}`, `pi_TW = {PI_TW}`, `pi_C = {PI_C}`.\n\n")
        fh.write("All travel times are Euclidean distances computed from the plotted coordinates.\n\n")
        fh.write("## Coordinates\n\n")
        fh.write("| Node | Meaning | x | y |\n")
        fh.write("|---:|---|---:|---:|\n")
        meanings = {0: "depot", 1: "P1", 2: "P2", 3: "P3", 10: "E10", 11: "E11"}
        for node, (x, y) in COORDS.items():
            fh.write(f"| {node} | {meanings[node]} | {fmt(x)} | {fmt(y)} |\n")
        fh.write("\n")
        fh.write("## Fixed Phase 1 Route\n\n")
        fh.write("- Trip 0: depot -> P1 -> P2 -> depot\n")
        fh.write("- Trip 1: depot -> P3 -> depot\n\n")
        fh.write("## Emergencies\n\n")
        fh.write("| Emergency | Release `rho` | W | Hard deadline | Candidate arc | Detour delta | Outsourcing cost | Expected behavior |\n")
        fh.write("|---:|---:|---:|---:|---|---:|---:|---|\n")
        em10 = inst.emergency_requests[10]
        em11 = inst.emergency_requests[11]
        fh.write(
            f"| E10 | {fmt(em10.rho)} | {fmt(em10.d_bar - em10.rho)} | {fmt(em10.d_bar)} | "
            f"P1 -> P2 | {fmt(inst.delta[(10, 1)])} | "
            f"{fmt(em10.f)} | inserted, because insertion is cheaper |\n"
        )
        fh.write(
            f"| E11 | {fmt(em11.rho)} | {fmt(em11.d_bar - em11.rho)} | {fmt(em11.d_bar)} | "
            f"P3 -> depot | {fmt(inst.delta[(11, 4)])} | "
            f"{fmt(em11.f)} | outsourced, because insertion after P3 misses the hard deadline |\n\n"
        )
        e11_arrival = inst.trips[1].C0 + inst.delta[(11, 4)]
        fh.write(
            f"For E11, inserting after P3 would return to the depot at `{fmt(e11_arrival)}`, "
            f"which is later than its hard deadline `{fmt(em11.d_bar)}`.\n\n"
        )
        fh.write("## Solver Decision\n\n")
        fh.write("| Emergency | Decision | Reason |\n")
        fh.write("|---:|---|---|\n")
        for d in result["decisions"]:
            if d["decision"] == "inserted":
                reason = f"inserted on arc {d['src']} -> {d['tgt']}; delta = {d['delta']}"
            else:
                reason = f"outsourced; reason = {d['reason']}"
            fh.write(f"| E{d['id']} | {d['decision']} | {reason} |\n")
        fh.write("\n## Objective\n\n")
        cb = result["cost_breakdown"]
        fh.write(f"- Objective value: `{result['objective']}`\n")
        fh.write(f"- Outsourcing cost: `{cb['outsourcing']}`\n")
        fh.write(f"- Time-window penalty: `{cb['tw_penalty']}`\n")
        fh.write(f"- Completion penalty: `{cb['completion_penalty']}`\n\n")
        fh.write("## Graph\n\n")
        fh.write(f"![Phase 2 toy example graph]({graph_path})\n")
    return path


def main():
    inst = make_example()
    result = solve_example(inst)
    graph_path = draw_example(inst, result)
    report_path = write_report(inst, result, graph_path)

    print("Small Phase 2 example solved.")
    print(f"Objective: {result['objective']}")
    for d in result["decisions"]:
        print(f"  E{d['id']}: {d['decision']}")
    print(f"Graph: {graph_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
