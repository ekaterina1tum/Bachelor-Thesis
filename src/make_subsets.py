"""
Build smaller instance sets (n = 30, 35, 40) from the n=50 instances by
cluster-proportional downsampling -- NOT tail truncation.

This preserves the C / R / RC spatial character: clustered instances stay
clustered, just with fewer points spread across all their clusters.

Output: data/MSCDPinstances/0NN/0NN_<name>.txt  with header "c <N+1> <vehicles>".

Usage
-----
    python make_subsets.py                 # sizes 30 35 40 from 050, +verify plots
    python make_subsets.py --sizes 30 40   # only some
    python make_subsets.py --no-plots
"""

from __future__ import annotations

import os
import glob
import argparse
import numpy as np

from instance import load_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTANCES_ROOT = os.path.join(REPO, "data", "MSCDPinstances")
SRC_SIZE = 50
TARGET_SIZES = [40, 35, 30]      # descending -> nested


def vehicles_for(n: int) -> int:
    """Interpolate the paper's fleet sizes: 25->10, 50->15 (upper bound on vehicles)."""
    return int(round(10 + (n - 25) * (15 - 10) / (50 - 25)))


# --------------------------------------------------------------------------- #
# Minimal KMeans + silhouette (numpy only)
# --------------------------------------------------------------------------- #
def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    # k-means++ style init
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d / d.sum() if d.sum() > 0 else None
        centers.append(X[rng.choice(len(X), p=probs)])
    centers = np.array(centers, dtype=float)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                centers[j] = pts.mean(axis=0)
    return labels


def silhouette(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sil = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for lab in uniq:
            if lab == labels[i]:
                continue
            other = labels == lab
            if other.any():
                b = min(b, D[i, other].mean())
        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return sil.mean()


def best_clustering(X, kmax=10, seed=0):
    """Pick k (2..kmax) maximizing silhouette; return labels."""
    n = len(X)
    kmax = min(kmax, n - 1)
    best_k, best_score, best_labels = 1, -1.0, np.zeros(n, dtype=int)
    for k in range(2, kmax + 1):
        labels = kmeans(X, k, seed=seed)
        score = silhouette(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    # if nothing clusters well (e.g. uniform R), treat as one cluster
    if best_score < 0.25:
        return np.zeros(n, dtype=int)
    return best_labels


# --------------------------------------------------------------------------- #
# Proportional per-cluster removal
# --------------------------------------------------------------------------- #
def remove_one_densest(keep_idx, coords, labels, target_cluster):
    """Remove the densest (most redundant) point of target_cluster among keep_idx."""
    members = [i for i in keep_idx if labels[i] == target_cluster]
    if len(members) <= 1:
        members = list(keep_idx)  # fallback: whole set
    pts = np.array([coords[i] for i in members])
    # nearest-neighbour distance within the cluster; remove the smallest (densest)
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    worst = members[int(nn.argmin())]
    return worst


def downsample(coords_arr, labels, keep_idx, target_n):
    """Remove (len-target_n) points, allocated proportionally across clusters."""
    keep = list(keep_idx)
    to_remove = len(keep) - target_n
    for _ in range(to_remove):
        # current cluster sizes among kept
        sizes = {}
        for i in keep:
            sizes[labels[i]] = sizes.get(labels[i], 0) + 1
        # remove from the cluster that is currently most over-represented
        # relative to its share (largest size wins -> proportional over many steps)
        target_cluster = max(sizes, key=lambda c: sizes[c])
        worst = remove_one_densest(keep, coords_arr, labels, target_cluster)
        keep.remove(worst)
    return keep


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def read_raw(path):
    """Return (header_vehicles, node_lines[str]) with node 0 = depot."""
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    veh = int(lines[0].split()[2])
    node_lines = lines[1:]  # depot + customers
    return veh, node_lines


def build_for_instance(path, sizes):
    name = os.path.splitext(os.path.basename(path))[0]          # e.g. 050_C101
    base = name.split("_", 1)[1]                                # C101
    veh_src, node_lines = read_raw(path)
    depot_line = node_lines[0]
    cust_lines = node_lines[1:]                                 # 50 customers
    coords = np.array([[float(t.split()[0]), float(t.split()[1])] for t in cust_lines])

    labels = best_clustering(coords)
    n_clusters = len(np.unique(labels))

    # incremental nested downsampling
    keep = list(range(len(cust_lines)))                         # indices into cust_lines
    written = []
    for target in sorted(sizes, reverse=True):                 # 40, 35, 30
        keep = downsample(coords, labels, keep, target)
        size_dir = os.path.join(INSTANCES_ROOT, f"{target:03d}")
        os.makedirs(size_dir, exist_ok=True)
        out_name = f"{target:03d}_{base}.txt"
        header = f"c {target + 1} {vehicles_for(target)}"
        kept_lines = [cust_lines[i] for i in sorted(keep)]
        with open(os.path.join(size_dir, out_name), "w") as fh:
            fh.write(header + "\n" + depot_line + "\n" + "\n".join(kept_lines) + "\n")
        written.append((target, out_name, len(kept_lines)))
    return name, n_clusters, written


def main():
    ap = argparse.ArgumentParser(description="Cluster-proportional subsets of the n=50 instances.")
    ap.add_argument("--sizes", type=int, nargs="+", default=TARGET_SIZES)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.join(INSTANCES_ROOT, f"{SRC_SIZE:03d}")
    files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    print(f"Source: {len(files)} instances of n={SRC_SIZE}")
    print(f"Building sizes {sorted(args.sizes, reverse=True)} (nested), vehicles per size: "
          f"{ {s: vehicles_for(s) for s in args.sizes} }\n")

    for path in files:
        name, n_clusters, written = build_for_instance(path, args.sizes)
        made = ", ".join(f"n={t}({c})" for t, _, c in written)
        print(f"  {name}: {n_clusters} clusters -> {made}")

    print("\nDone. Verify a few on the map, e.g.:")
    print("  python src/plot_instance.py data/MSCDPinstances/030/030_C101.txt --save")


if __name__ == "__main__":
    main()
