"""
Plot a Phase-2 result: Phase-1 routes + emergency insert/outsource decisions.

Reads a result JSON written by phase2_experiment.py, then overlays on the
instance geometry:
  - depot (red square) and regular PoCs (blue),
  - the fixed Phase-1 route arcs (light grey),
  - inserted emergencies (orange triangle) with a dashed detour on the broken arc,
  - outsourced emergencies (red triangle, tagged [3PL]) shown at their location.

Usage
-----
    python phase2_plot_results.py data/phase2_results/025/025_C101/025_C101_L2_s0.json
    python phase2_plot_results.py <result.json> --save
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import matplotlib
import matplotlib.pyplot as plt

from instance import load_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTANCES_ROOT = os.path.join(REPO, "data", "MSCDPinstances")
PHASE1_ROOT = os.path.join(REPO, "data", "phase1_solutions")
SCENARIO_ROOT = os.path.join(REPO, "data", "emergency_scenarios")

SIZE_DIR = {10: "010", 15: "015", 25: "025"}
DEPOT = 0


def load_support(res: dict):
    """Load instance coords, Phase-1 solution, and the scenario emergencies."""
    size = res["size"]
    instance = res["instance"]
    coords = load_coordinates(os.path.join(INSTANCES_ROOT, SIZE_DIR[size], f"{instance}.txt"))[0]

    with open(os.path.join(PHASE1_ROOT, SIZE_DIR[size], f"{instance}.json")) as fh:
        sol = json.load(fh)

    scen_name = f"{instance}_L{res['level']}_s{res['seed_idx']}.json"
    with open(os.path.join(SCENARIO_ROOT, SIZE_DIR[size], instance, scen_name)) as fh:
        scen = json.load(fh)
    em_xy = {em["id"]: (em["x"], em["y"]) for em in scen["emergencies"]}
    return coords, sol, em_xy


def main():
    ap = argparse.ArgumentParser(description="Plot a Phase-2 result.")
    ap.add_argument("result", help="path to a phase2_results/*.json file")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    with open(args.result) as fh:
        res = json.load(fh)
    coords, sol, em_xy = load_support(res)

    # arc id -> (src, tgt) of inserted emergencies
    inserted_arc = {e["arc"]: e["id"] for e in res["emergencies"]
                    if e["decision"] == "inserted" and "arc" in e}

    fig, ax = plt.subplots(figsize=(8.5, 8))

    # ---- Phase-1 route arcs (break arcs hosting an insertion) ----
    for t in sol["trips"]:
        for a in t["arcs"]:
            (x1, y1) = coords[a["src"]]
            (x2, y2) = coords[a["tgt"]]
            if a["id"] in inserted_arc:
                m = inserted_arc[a["id"]]
                mx, my = em_xy[m]
                for (xa, ya), (xb, yb) in [((x1, y1), (mx, my)), ((mx, my), (x2, y2))]:
                    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                                arrowprops=dict(arrowstyle="-|>", color="tab:orange",
                                                lw=1.8, linestyle="dashed",
                                                shrinkA=6, shrinkB=6), zorder=3)
            else:
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color="0.75", lw=1.0,
                                            shrinkA=6, shrinkB=6), zorder=1)

    # ---- nodes ----
    px = [coords[i][0] for i in coords if i != DEPOT]
    py = [coords[i][1] for i in coords if i != DEPOT]
    ax.scatter(px, py, c="tab:blue", s=45, zorder=4, label="Regular PoC")
    ax.scatter(*coords[DEPOT], c="tab:red", marker="s", s=170, edgecolors="black",
               zorder=6, label="Depot (lab)")

    # ---- emergencies ----
    for e in res["emergencies"]:
        x, y = em_xy[e["id"]]
        if e["decision"] == "inserted":
            ax.scatter(x, y, c="tab:orange", marker="^", s=140,
                       edgecolors="black", zorder=7)
            tag = f"E{e['id']} (ins)"
        else:
            ax.scatter(x, y, c="red", marker="X", s=150, edgecolors="black", zorder=7)
            tag = f"E{e['id']} [3PL]"
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color="darkorange" if e["decision"] == "inserted" else "darkred")

    # legend proxies
    ax.plot([], [], color="0.75", lw=1.2, label="Phase-1 route arc")
    ax.plot([], [], color="tab:orange", lw=1.8, linestyle="dashed", label="Inserted detour")
    ax.scatter([], [], c="red", marker="X", s=120, label="Outsourced [3PL]")

    cb = res["cost_breakdown"]
    title = (f"{res['instance']}  L{res['level']} s{res['seed_idx']}  "
             f"(obj={res['objective']}, outsourced {res['n_outsourced']}/{res['n_emergencies']})\n"
             f"out={cb['outsourcing']}  TW={cb['tw_penalty']}  comp={cb['completion_penalty']}  "
             f"| lam={res['params']['LAMBDA']} piS={res['params']['PI_S']}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    if args.save:
        out = os.path.splitext(args.result)[0] + ".png"
        matplotlib.use("Agg")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
