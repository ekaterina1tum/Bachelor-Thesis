"""
Recompute inter-trip idle time g_T in existing Phase-1 solution JSONs, in place.

g_T is derivable from the solution alone -- no MILP re-solve:
    departure(T+) = z0(first PoC of T+) - c(depot, first PoC)
    g_T = max(0, departure(T+) - C0_T)        (0 for the last trip of a route)

Needs instance coordinates only for the depot->firstPoC distance.

Usage
-----
    python patch_gT.py                 # patch sizes 010, 015, 025
    python patch_gT.py --sizes 25
"""

from __future__ import annotations

import os
import json
import glob
import math
import argparse

from instance import load_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTANCES_ROOT = os.path.join(REPO, "data", "MSCDPinstances")
PHASE1_ROOT = os.path.join(REPO, "data", "phase1_solutions")
SIZE_DIR = {10: "010", 15: "015", 25: "025"}
DEPOT = 0

_coord_cache: dict[str, dict] = {}


def coords_for(size_dir: str, instance: str) -> dict:
    if instance not in _coord_cache:
        path = os.path.join(INSTANCES_ROOT, size_dir, f"{instance}.txt")
        _coord_cache[instance] = load_coordinates(path)[0]
    return _coord_cache[instance]


def euclid(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def patch_file(path: str, size_dir: str) -> tuple[int, float]:
    with open(path) as fh:
        doc = json.load(fh)

    instance = doc["instance"]
    coords = coords_for(size_dir, instance)
    trips = {t["id"]: t for t in doc["trips"]}
    z0 = {r["id"]: r["z0"] for r in doc["regular_requests"]}

    n_changed = 0
    total_g = 0.0
    for t in doc["trips"]:
        old = t.get("g", 0.0)
        if t["is_last"] or t["next_trip_id"] is None:
            new = 0.0
        else:
            nxt = trips[t["next_trip_id"]]
            first_poc = nxt["arcs"][0]["tgt"]
            if first_poc in z0:
                departure_next = z0[first_poc] - euclid(coords[DEPOT], coords[first_poc])
                new = max(0.0, departure_next - t["C0"])
            else:
                new = 0.0
        t["g"] = new
        total_g += new
        if abs(new - old) > 1e-9:
            n_changed += 1

    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return n_changed, total_g


def main():
    ap = argparse.ArgumentParser(description="Recompute g_T in existing Phase-1 JSONs.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 25])
    args = ap.parse_args()

    grand_changed = 0
    grand_g = 0.0
    n_files = 0
    for size in args.sizes:
        sd = SIZE_DIR[size]
        files = sorted(glob.glob(os.path.join(PHASE1_ROOT, sd, "*.json")))
        size_changed = 0
        size_g = 0.0
        for path in files:
            nc, tg = patch_file(path, sd)
            size_changed += nc
            size_g += tg
            n_files += 1
        print(f"size {sd}: {len(files)} files, {size_changed} trips got g_T>0 set, "
              f"sum g_T = {size_g:.1f}")
        grand_changed += size_changed
        grand_g += size_g

    print(f"\nPatched {n_files} files. {grand_changed} trips now carry nonzero g_T; "
          f"total idle slack recovered = {grand_g:.1f} min")


if __name__ == "__main__":
    main()
