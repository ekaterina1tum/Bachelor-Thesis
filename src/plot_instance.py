"""
Plot the nodes (depot + points of care) of MSCDP instance files.

Reads the raw txt format directly (coordinates are not kept by load_instance):
    line 1:  c <num_nodes> <max_vehicles>
    line k:  x  y  tw_start  tw_end      (first node = depot)

Usage
-----
    # one instance -> show on screen
    python plot_instance.py "data/MSCDPinstances/025/025_C101.txt"

    # one instance -> save PNG
    python plot_instance.py "data/.../025_C101.txt" --save

    # every txt in a folder -> save one PNG each into <folder>/plots/
    python plot_instance.py "data/MSCDPinstances/025" --all
"""

from __future__ import annotations

import os
import sys
import glob

import matplotlib
import matplotlib.pyplot as plt


def read_nodes(path: str):
    """Return (coords, tw, num_vehicles) for one instance txt file.

    coords : dict node_idx -> (x, y)   (node 0 is the depot)
    tw     : dict node_idx -> (tw_start, tw_end)
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    header = lines[0].split()
    num_nodes = int(header[1])
    num_vehicles = int(header[2])

    coords, tw = {}, {}
    for idx, ln in enumerate(lines[1:1 + num_nodes]):
        x, y, ts, te = ln.split()
        coords[idx] = (float(x), float(y))
        tw[idx] = (float(ts), float(te))

    return coords, tw, num_vehicles


def plot_instance(path: str, ax=None, annotate: bool = True):
    """Plot one instance's nodes on a matplotlib Axes."""
    coords, tw, num_vehicles = read_nodes(path)
    name = os.path.splitext(os.path.basename(path))[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    depot = 0
    poc_x = [coords[i][0] for i in coords if i != depot]
    poc_y = [coords[i][1] for i in coords if i != depot]

    # Points of care
    ax.scatter(poc_x, poc_y, c="tab:blue", s=45, zorder=3, label="Points of care")
    # Depot
    dx, dy = coords[depot]
    ax.scatter([dx], [dy], c="tab:red", marker="s", s=160, zorder=4,
               edgecolors="black", label="Depot (lab)")

    if annotate:
        for i, (x, y) in coords.items():
            label = "0" if i == depot else str(i)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    n_pocs = len(coords) - 1
    ax.set_title(f"{name}  ({n_pocs} PoCs, {num_vehicles} vehicles)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    return ax


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    target = args[0]
    save = "--save" in args
    do_all = "--all" in args

    if do_all or os.path.isdir(target):
        folder = target
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        out_dir = os.path.join(folder, "plots")
        os.makedirs(out_dir, exist_ok=True)
        matplotlib.use("Agg")  # no display needed when batch-saving
        for path in files:
            fig, ax = plt.subplots(figsize=(7, 7))
            plot_instance(path, ax=ax)
            name = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(out_dir, f"{name}.png")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out}")
        print(f"\n{len(files)} plots written to {out_dir}")
    else:
        plot_instance(target)
        if save:
            out = os.path.splitext(target)[0] + ".png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            print(f"saved {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
