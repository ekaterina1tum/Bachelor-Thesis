"""
Run the Phase-1 model on every instance in a folder and report F'.

F' = sum_{j in P} (C_j - r_j) = (objective) - sum_{j in P} r_j

where r_j is the release time of regular request j. Results are averaged
per Solomon problem type (C1, C2, R1, R2, RC1, RC2).
"""

import os
import re
import sys
import glob

import gurobipy as gp
from gurobipy import GRB

from instance import load_instance
from graph import build_graph
from model import build_model


def instance_type(filename: str) -> str:
    """Map a file like '025_RC201.txt' to its Solomon type 'RC2'."""
    base = os.path.basename(filename)
    m = re.search(r"_(C|R|RC)(\d)\d*\.txt$", base)
    if not m:
        return "?"
    return f"{m.group(1)}{m.group(2)}"


def solve_one(path: str, time_limit: float, max_shift: float):
    """Solve Phase 1 for one instance; return (F', obj, status, gap, runtime)."""
    inst = load_instance(path, max_shift=max_shift)
    g = build_graph(inst)
    m = build_model(inst, g)
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return None, None, status, None, m.Runtime

    obj = m.ObjVal
    sum_r = sum(inst.release_time[j] for j in inst.pocs)
    F_prime = obj - sum_r
    gap = m.MIPGap if status != GRB.OPTIMAL else 0.0
    return F_prime, obj, status, gap, m.Runtime


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/ekaterinatkachenko/PycharmProjects/THESIS/data/MSCDPinstances/025"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    max_shift = float(sys.argv[3]) if len(sys.argv) > 3 else 480.0
    # Optional type filter, e.g. "C1" runs only C1 instances (C2/R1/.. also valid)
    type_filter = sys.argv[4].upper() if len(sys.argv) > 4 else None

    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if type_filter:
        files = [f for f in files if instance_type(f) == type_filter]
    print(f"Folder: {folder}")
    flt = f" | filter: {type_filter}" if type_filter else ""
    print(f"Instances: {len(files)} | time limit: {time_limit}s | max_shift: {max_shift}{flt}\n")

    header = f"{'instance':<16}{'type':<6}{'F_prime':>14}{'obj':>14}{'status':>8}{'gap%':>8}{'time(s)':>9}"
    print(header)
    print("-" * len(header))

    per_type: dict[str, list[float]] = {}
    rows = []

    for path in files:
        typ = instance_type(path)
        F_prime, obj, status, gap, runtime = solve_one(path, time_limit, max_shift)
        name = os.path.basename(path)

        if F_prime is None:
            print(f"{name:<16}{typ:<6}{'NO SOL':>14}{'':>14}{status:>8}{'':>8}{runtime:>9.1f}")
            continue

        status_str = "OPT" if status == GRB.OPTIMAL else str(status)
        print(f"{name:<16}{typ:<6}{F_prime:>14.2f}{obj:>14.2f}"
              f"{status_str:>8}{gap*100:>8.2f}{runtime:>9.1f}")

        per_type.setdefault(typ, []).append(F_prime)
        rows.append((name, typ, F_prime, obj, status_str, gap, runtime))

    # ---- Averages per type ----
    print("\n" + "=" * 40)
    print("Average F' per type")
    print("=" * 40)
    print(f"{'type':<8}{'#inst':>7}{'avg F_prime':>16}")
    print("-" * 31)
    for typ in ["C1", "C2", "R1", "R2", "RC1", "RC2"]:
        if typ in per_type:
            vals = per_type[typ]
            print(f"{typ:<8}{len(vals):>7}{sum(vals) / len(vals):>16.2f}")

    # Write CSV
    suffix = f"_{type_filter}" if type_filter else ""
    out_csv = os.path.join(folder, f"phase1_results{suffix}.csv")
    with open(out_csv, "w") as fh:
        fh.write("instance,type,F_prime,objective,status,gap,runtime_s\n")
        for name, typ, F_prime, obj, st, gap, rt in rows:
            fh.write(f"{name},{typ},{F_prime:.4f},{obj:.4f},{st},{gap:.6f},{rt:.2f}\n")
    print(f"\nPer-instance results written to {out_csv}")


if __name__ == "__main__":
    main()
