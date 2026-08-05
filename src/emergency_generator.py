"""
Emergency-request scenario generator for the Phase-2 insert-vs-outsource MILP.

Generates emergency requests on top of the existing MSCDP instances (Ferone's
four-column format: x, y, ready, due). It produces ONLY the demand-side scenario
data -- locations and time attributes -- and deliberately does NOT assign
outsourcing costs (F_m), shift/penalty weights (pi^S, pi^C), or candidate
insertion arcs (A_m). Those belong to cost/penalty calibration and to the
Phase-1-solution-dependent feasibility filter, kept strictly separate.

Per emergency request, three attributes are generated:
  * Location -- a NEW point: anchor on a randomly chosen existing customer
    (depot excluded, distinct anchor per emergency within a scenario), then jitter
    by a small offset drawn uniformly from the disk of radius R_jit. This keeps the
    C / R / RC spatial structure (clustered / uniform / mixed) intact -- we do NOT
    sample over the whole bounding box.
  * Release time r ~ Uniform[0, horizon - W_release_base], where `horizon` is
    the instance's lab-closing time (the depot's due time d_0). The release base
    is fixed across W values, so W comparisons use the same emergency locations
    and releases; only the hard deadline changes.
  * Hard deadline d = r + W, with W the urgent window (default 60).

Scenarios are generated for sizes 10 / 15 / 25, with emergency counts ("levels"):
  n = 10 : {1, 2, 3}
  n = 15 : {1, 2, 3}
  n = 25 : {2, 4, 6}
For each (instance x level), N_seeds independent scenarios (default 10) are produced
with deterministic, recorded seeds. Each scenario is saved as JSON, plus a manifest
(CSV) listing instance, level, seed, and W.

Usage
-----
    python emergency_generator.py
    python emergency_generator.py --sizes 10 15 25 --W 60 --R-jit 3.0 --seeds 10
    python emergency_generator.py --W 80 --release-base-W 240
"""

from __future__ import annotations

import os
import re
import csv
import json
import math
import zlib
import random
import argparse
import glob

from instance import load_coordinates


# Emergency counts ("levels") per instance size
LEVELS_BY_SIZE = {
    10: [1, 2, 3],
    15: [1, 2, 3],
    25: [2, 4, 6],
    30: [2, 4, 6],
    35: [3, 6, 9],
    40: [3, 6, 9],
}

# Size -> instance folder name (zero-padded)
SIZE_DIR = {10: "010", 15: "015", 25: "025", 30: "030", 35: "035", 40: "040"}
DEFAULT_RELEASE_BASE_W = 240.0


def instance_class(name: str) -> str:
    """Return 'C', 'R', or 'RC' from an instance filename like '025_RC101'."""
    m = re.search(r"_(RC|R|C)\d", name)
    return m.group(1) if m else "?"


def geometry_hash(coords: dict) -> int:
    """Stable hash of the customer geometry (depot excluded).

    Keyed on the ordered list of customer (x, y) points the instance actually
    uses, so any two instances with identical geometry -- regardless of their
    name or class label -- receive the SAME emergency scenarios. Differences in
    Phase-2 outcomes across such instances are then attributable purely to their
    regular time windows / resulting Phase-1 routes.
    """
    customers = [coords[i] for i in sorted(coords) if i != 0]
    s = ";".join(f"{x:.4f},{y:.4f}" for x, y in customers)
    return zlib.crc32(s.encode())  # stable 32-bit value, independent of PYTHONHASHSEED


def scenario_seed(geom_hash: int, level: int, idx: int) -> int:
    """Deterministic, reproducible seed for one scenario, keyed on geometry."""
    key = f"{geom_hash}|level={level}|seed_idx={idx}".encode()
    return zlib.crc32(key)


def jitter(rng: random.Random, x: float, y: float, R_jit: float):
    """Offset (x, y) by a vector drawn uniformly from the disk of radius R_jit."""
    radius = R_jit * math.sqrt(rng.random())  # sqrt -> uniform over the area
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return x + radius * math.cos(theta), y + radius * math.sin(theta)


def generate_scenario(
    path: str,
    level: int,
    idx: int,
    W: float,
    R_jit: float,
    release_base_W: float = DEFAULT_RELEASE_BASE_W,
) -> dict:
    """Generate one emergency scenario (a list of `level` emergencies) for an instance.

    The seed is derived from the instance geometry, so instances that share the
    same customer coordinates get identical scenarios for a given (level, idx).
    """
    coords, tw = load_coordinates(path)
    name = os.path.splitext(os.path.basename(path))[0]
    cls = instance_class(name)

    horizon = tw[0][1]  # depot due time d_0 == lab-closing time (operating horizon)
    # Keep releases identical across W values. Use max(W, release_base_W) so
    # release + W never exceeds the operating horizon when W is larger than the
    # chosen base.
    release_cap_W = max(W, release_base_W)
    rel_upper = max(0.0, horizon - release_cap_W)

    geom = geometry_hash(coords)
    seed = scenario_seed(geom, level, idx)

    customers = [i for i in coords if i != 0]  # exclude depot (node 0)
    rng = random.Random(seed)

    # Distinct anchor per emergency within this scenario
    anchors = rng.sample(customers, level)

    emergencies = []
    for m, anchor in enumerate(anchors):
        ax, ay = coords[anchor]
        ex, ey = jitter(rng, ax, ay, R_jit)
        r = rng.uniform(0.0, rel_upper)
        d = r + W
        emergencies.append({
            "id": m,
            "x": round(ex, 4),
            "y": round(ey, 4),
            "anchor": anchor,
            "release": round(r, 4),
            "deadline": round(d, 4),
        })

    return {
        "instance": name,
        "size": len(customers),
        "class": cls,
        "level": level,
        "seed_idx": idx,
        "seed": seed,
        "geometry_hash": geom,
        "W": W,
        "release_base_W": release_base_W,
        "R_jit": R_jit,
        "horizon": horizon,
        "emergencies": emergencies,
    }


def generate_all(
    data_root: str,
    out_root: str,
    sizes,
    W: float,
    R_jit: float,
    n_seeds: int,
    release_base_W: float = DEFAULT_RELEASE_BASE_W,
):
    """Batch-generate scenarios for every instance of the requested sizes."""
    # tag the output by urgency window W so multiple W values coexist
    out_root = os.path.join(out_root, f"W{int(W)}")
    os.makedirs(out_root, exist_ok=True)
    manifest_rows = []
    n_files = 0

    for size in sizes:
        if size not in LEVELS_BY_SIZE:
            print(f"  (skipping size {size}: no levels defined)")
            continue
        folder = os.path.join(data_root, SIZE_DIR[size])
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            print(f"  (no instances found in {folder})")
            continue

        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]
            for level in LEVELS_BY_SIZE[size]:
                for idx in range(n_seeds):
                    scen = generate_scenario(path, level, idx, W, R_jit, release_base_W)

                    out_dir = os.path.join(out_root, SIZE_DIR[size], name)
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"{name}_L{level}_s{idx}.json"
                    out_path = os.path.join(out_dir, fname)
                    with open(out_path, "w") as fh:
                        json.dump(scen, fh, indent=2)
                    n_files += 1

                    manifest_rows.append({
                        "instance": name,
                        "size": size,
                        "class": scen["class"],
                        "level": level,
                        "seed_idx": idx,
                        "seed": scen["seed"],
                        "geometry_hash": scen["geometry_hash"],
                        "W": W,
                        "release_base_W": release_base_W,
                        "R_jit": R_jit,
                        "n_emergencies": level,
                        "path": os.path.relpath(out_path, out_root),
                    })

    manifest_path = os.path.join(out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "instance", "size", "class", "level", "seed_idx", "seed",
            "geometry_hash", "W", "release_base_W", "R_jit", "n_emergencies", "path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {n_files} scenario files across {len(manifest_rows)} (instance x level x seed) combos.")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    default_data = os.path.join(repo_root, "data", "MSCDPinstances")
    default_out = os.path.join(repo_root, "data", "emergency_scenarios")

    ap = argparse.ArgumentParser(description="Generate emergency-request scenarios for Phase 2.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 25],
                    help="instance sizes to process (default: 10 15 25)")
    ap.add_argument("--W", type=float, default=60.0,
                    help="urgent-window width; deadline = release + W (default: 60)")
    ap.add_argument("--R-jit", type=float, default=3.0, dest="R_jit",
                    help="jitter radius around the anchor customer (default: 3.0)")
    ap.add_argument("--release-base-W", type=float, default=DEFAULT_RELEASE_BASE_W,
                    help="fixed W used to cap release generation across all W values "
                         "(default: 240); keeps releases comparable across W")
    ap.add_argument("--seeds", type=int, default=10, dest="n_seeds",
                    help="independent scenarios per (instance x level) (default: 10)")
    ap.add_argument("--data-root", default=default_data, help="MSCDP instances root")
    ap.add_argument("--out-root", default=default_out, help="output root for scenarios")
    args = ap.parse_args()

    print(f"sizes={args.sizes}  W={args.W}  R_jit={args.R_jit}  seeds={args.n_seeds}  "
          f"release_base_W={args.release_base_W}  (release horizon = depot due time)")
    generate_all(
        args.data_root,
        args.out_root,
        args.sizes,
        args.W,
        args.R_jit,
        args.n_seeds,
        args.release_base_W,
    )


if __name__ == "__main__":
    main()
