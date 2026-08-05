"""
Generate appendix figures from saved Phase-2 summary CSV files.

The script is intentionally read-only with respect to experiment results: it does
not solve any instances. It aggregates the existing summary_W*_lam*_piS*.csv
files and writes PNG/PDF figures under data/phase2_results/appendix_figures/.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "data", "phase2_results")
OUTDIR = os.path.join(RESULTS, "appendix_figures")
INCLUDED_W = {60.0, 80.0, 90.0}

SUMMARY_RE = re.compile(
    r"summary_W(?P<W>[0-9]+(?:\.[0-9]+)?)_"
    r"lam(?P<lam>[0-9]+(?:\.[0-9]+)?)_"
    r"piS(?P<piS>[0-9]+(?:\.[0-9]+)?)\.csv"
)

CLASSES = ["C", "R", "RC"]
CLASS_COLOR = {"C": "#1f77b4", "R": "#2ca02c", "RC": "#ff7f0e"}
W_COLOR = {60.0: "#d95f02", 80.0: "#1b9e77", 90.0: "#0b3c5d", 120.0: "#7570b3"}


def fnum(value: str | float | int) -> float:
    return float(value)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def read_rows() -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(RESULTS, "summary_W*_lam*_piS*.csv"))):
        m = SUMMARY_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (float(m.group("W")), float(m.group("lam")), float(m.group("piS")))
        if key[0] not in INCLUDED_W:
            continue
        if key in seen:
            continue
        seen.add(key)
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row["W"] = float(row.get("W") or m.group("W"))
                row["LAMBDA"] = float(row.get("LAMBDA") or m.group("lam"))
                row["PI_S"] = float(row.get("PI_S") or m.group("piS"))
                row["size"] = int(row["size"])
                for col in [
                    "objective",
                    "n_emergencies",
                    "n_outsourced",
                    "n_outsourced_infeasible",
                    "n_outsourced_economic",
                    "n_inserted",
                    "frac_outsourced",
                    "outsourcing",
                    "tw_penalty",
                    "completion_penalty",
                ]:
                    row[col] = float(row[col])
                rows.append(row)
    return rows


def group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def rates(rows: list[dict]) -> dict[str, float]:
    n_em = sum(r["n_emergencies"] for r in rows)
    n_inserted = sum(r["n_inserted"] for r in rows)
    n_infeas = sum(r["n_outsourced_infeasible"] for r in rows)
    n_econ = sum(r["n_outsourced_economic"] for r in rows)
    n_out = sum(r["n_outsourced"] for r in rows)
    n_feasible = n_inserted + n_econ
    return {
        "outsource": n_out / n_em if n_em else math.nan,
        "inserted": n_inserted / n_em if n_em else math.nan,
        "infeasible": n_infeas / n_em if n_em else math.nan,
        "economic": n_econ / n_em if n_em else math.nan,
        "feasible_inserted": n_inserted / n_feasible if n_feasible else math.nan,
        "feasible_out_econ": n_econ / n_feasible if n_feasible else math.nan,
        "objective": mean([r["objective"] for r in rows]),
        "outsourcing_cost": mean([r["outsourcing"] for r in rows]),
        "tw_penalty": mean([r["tw_penalty"] for r in rows]),
        "completion_penalty": mean([r["completion_penalty"] for r in rows]),
    }


def pct_axis(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(0, 11, 2)])
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name: str, written: list[str]):
    png = os.path.join(OUTDIR, f"{name}.png")
    pdf = os.path.join(OUTDIR, f"{name}.pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    written.extend([png, pdf])


def plot_w_lambda_lines(rows, written):
    pi = 2.0
    lambdas = sorted({r["LAMBDA"] for r in rows if r["PI_S"] == pi})
    Ws = sorted({r["W"] for r in rows if r["PI_S"] == pi})
    metrics = [
        ("outsource", "Overall outsourced share", "overall_outsource_share_by_W_lambda_piS2"),
        ("infeasible", "Forced infeasible outsourced share", "forced_infeasible_share_by_W_lambda_piS2"),
        ("feasible_inserted", "Inserted share among feasible emergencies", "feasible_inserted_share_by_W_lambda_piS2"),
        ("feasible_out_econ", "Economically outsourced share among feasible emergencies", "feasible_economic_outsource_share_by_W_lambda_piS2"),
    ]
    grouped = group([r for r in rows if r["PI_S"] == pi], ["W", "LAMBDA"])
    for metric, title, name in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for W in Ws:
            y = [rates(grouped.get((W, lam), []))[metric] for lam in lambdas]
            ax.plot(lambdas, y, marker="o", linewidth=2.2, color=W_COLOR.get(W), label=f"W={W:g}")
        ax.set_title(title + " (pi_S=2)")
        ax.set_xlabel("Outsourcing-cost multiplier lambda")
        ax.set_ylabel("Share")
        pct_axis(ax)
        ax.legend(title="Emergency window")
        savefig(fig, name, written)


def plot_class_and_size_lines(rows, written):
    pi = 2.0
    lambdas = sorted({r["LAMBDA"] for r in rows if r["PI_S"] == pi})
    Ws = sorted({r["W"] for r in rows if r["PI_S"] == pi})
    sizes = sorted({r["size"] for r in rows})

    for W in Ws:
        rows_w = [r for r in rows if r["PI_S"] == pi and r["W"] == W]
        by_class = group(rows_w, ["class", "LAMBDA"])
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for cls in CLASSES:
            y = [rates(by_class.get((cls, lam), []))["outsource"] for lam in lambdas]
            ax.plot(lambdas, y, marker="o", linewidth=2.2, color=CLASS_COLOR[cls], label=cls)
        ax.set_title(f"Outsourced share by class (W={W:g}, pi_S=2)")
        ax.set_xlabel("Outsourcing-cost multiplier lambda")
        ax.set_ylabel("Outsourced share")
        pct_axis(ax)
        ax.legend(title="Class")
        savefig(fig, f"outsource_share_by_class_W{W:g}_piS2", written)

        by_size = group(rows_w, ["size", "LAMBDA"])
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for size in sizes:
            y = [rates(by_size.get((size, lam), []))["outsource"] for lam in lambdas]
            ax.plot(lambdas, y, marker="s", linewidth=2.2, label=f"n={size}")
        ax.set_title(f"Outsourced share by instance size (W={W:g}, pi_S=2)")
        ax.set_xlabel("Outsourcing-cost multiplier lambda")
        ax.set_ylabel("Outsourced share")
        pct_axis(ax)
        ax.legend(title="Size")
        savefig(fig, f"outsource_share_by_size_W{W:g}_piS2", written)


def plot_objective_terms(rows, written):
    pi = 2.0
    lambdas = sorted({r["LAMBDA"] for r in rows if r["PI_S"] == pi})
    Ws = sorted({r["W"] for r in rows if r["PI_S"] == pi})
    grouped = group([r for r in rows if r["PI_S"] == pi], ["W", "LAMBDA"])
    terms = [
        ("outsourcing_cost", "Outsourcing cost"),
        ("tw_penalty", "Time-window penalty"),
        ("completion_penalty", "Completion penalty"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, (metric, label) in zip(axes, terms):
        for W in Ws:
            y = [rates(grouped.get((W, lam), []))[metric] for lam in lambdas]
            ax.plot(lambdas, y, marker="o", linewidth=2.0, color=W_COLOR.get(W), label=f"W={W:g}")
        ax.set_title(label)
        ax.set_xlabel("lambda")
        ax.grid(True, axis="y", color="0.88", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean value per scenario")
    axes[-1].legend(title="Window", loc="best")
    fig.suptitle("Objective components by lambda (pi_S=2)", y=1.03)
    savefig(fig, "objective_components_by_W_lambda_piS2", written)


def plot_heatmaps(rows, written):
    Ws = sorted({r["W"] for r in rows})
    full_piS = sorted({r["PI_S"] for r in rows})
    lambdas = sorted({r["LAMBDA"] for r in rows})
    metrics = [
        ("outsource", "Outsourced share", "heatmap_outsource_share"),
        ("economic", "Economically outsourced share", "heatmap_economic_outsource_share"),
        ("infeasible", "Forced infeasible share", "heatmap_forced_infeasible_share"),
        ("objective", "Mean objective value", "heatmap_objective_value"),
    ]

    for W in Ws:
        rows_w = [r for r in rows if r["W"] == W]
        piSs = sorted({r["PI_S"] for r in rows_w})
        if len(piSs) <= 1:
            continue
        grouped = group(rows_w, ["PI_S", "LAMBDA"])
        for metric, title, stem in metrics:
            grid = [[rates(grouped.get((pi, lam), []))[metric] for lam in lambdas] for pi in piSs]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"{title} over pi_S x lambda (W={W:g})")
            ax.set_xlabel("lambda")
            ax.set_ylabel("pi_S")
            ax.set_xticks(range(len(lambdas)))
            ax.set_xticklabels([f"{x:g}" for x in lambdas])
            ax.set_yticks(range(len(piSs)))
            ax.set_yticklabels([f"{x:g}" for x in piSs])
            for i, pi in enumerate(piSs):
                for j, lam in enumerate(lambdas):
                    val = grid[i][j]
                    txt = f"{val:.2f}" if metric == "objective" else f"{100 * val:.0f}%"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
            fig.colorbar(im, ax=ax, label=title)
            savefig(fig, f"{stem}_W{W:g}", written)


def plot_piS_sensitivity(rows, written):
    lambd = 1.0
    Ws = sorted({r["W"] for r in rows})
    for W in Ws:
        rows_w = [r for r in rows if r["W"] == W and abs(r["LAMBDA"] - lambd) < 1e-9]
        piSs = sorted({r["PI_S"] for r in rows_w})
        if len(piSs) <= 1:
            continue
        grouped = group(rows_w, ["class", "PI_S"])
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for cls in CLASSES:
            y = [rates(grouped.get((cls, pi), []))["outsource"] for pi in piSs]
            ax.plot(piSs, y, marker="o", linewidth=2.2, color=CLASS_COLOR[cls], label=cls)
        ax.set_title(f"pi_S sensitivity by class (W={W:g}, lambda=1)")
        ax.set_xlabel("Soft-window penalty pi_S")
        ax.set_ylabel("Outsourced share")
        pct_axis(ax)
        ax.legend(title="Class")
        savefig(fig, f"piS_sensitivity_by_class_W{W:g}_lambda1", written)


def plot_instance_scatter(rows, written):
    pi = 2.0
    lambd = 1.0
    Ws = sorted({r["W"] for r in rows if r["PI_S"] == pi})
    for W in Ws:
        rows_w = [r for r in rows if r["W"] == W and r["PI_S"] == pi and abs(r["LAMBDA"] - lambd) < 1e-9]
        by_inst = group(rows_w, ["class", "size", "instance"])
        points = []
        for (cls, size, inst), rs in by_inst.items():
            points.append((cls, size, inst, rates(rs)["outsource"], rates(rs)["infeasible"], rates(rs)["economic"]))
        points.sort(key=lambda x: (x[0], x[1], x[3]))

        fig, ax = plt.subplots(figsize=(11, 5.2))
        x = list(range(len(points)))
        colors = [CLASS_COLOR[p[0]] for p in points]
        ax.scatter(x, [p[3] for p in points], c=colors, s=32, label=None)
        ax.set_title(f"Instance-level outsourced share (W={W:g}, lambda=1, pi_S=2)")
        ax.set_xlabel("Instances ordered within class and size")
        ax.set_ylabel("Outsourced share")
        pct_axis(ax)
        boundaries = []
        last = None
        for idx, p in enumerate(points):
            key = (p[0], p[1])
            if key != last:
                boundaries.append((idx, key))
                last = key
        for idx, key in boundaries:
            ax.axvline(idx - 0.5, color="0.85", linewidth=0.8)
            ax.text(idx, 1.02, f"{key[0]} n={key[1]}", rotation=45, va="bottom", fontsize=8)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLOR[c], label=c, markersize=7)
            for c in CLASSES
        ]
        ax.legend(handles=handles, title="Class", loc="lower right")
        savefig(fig, f"instance_level_outsource_share_W{W:g}_lambda1_piS2", written)


def write_index(written: list[str]) -> str:
    path = os.path.join(OUTDIR, "APPENDIX_FIGURES_INDEX.md")
    pngs = [p for p in written if p.endswith(".png")]
    with open(path, "w") as fh:
        fh.write("# Phase 2 Appendix Figures\n\n")
        fh.write("Generated from existing `summary_W*_lam*_piS*.csv` files. No optimization runs are performed here.\n\n")
        for png in sorted(pngs):
            fh.write(f"- `{os.path.basename(png)}`\n")
    return path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUTDIR, "*.png")) + glob.glob(os.path.join(OUTDIR, "*.pdf")):
        os.remove(old)
    rows = read_rows()
    if not rows:
        raise SystemExit("No summary_W*_lam*_piS*.csv files found.")
    written: list[str] = []
    plot_w_lambda_lines(rows, written)
    plot_class_and_size_lines(rows, written)
    plot_objective_terms(rows, written)
    plot_heatmaps(rows, written)
    plot_piS_sensitivity(rows, written)
    plot_instance_scatter(rows, written)
    index = write_index(written)
    print(f"Wrote {len([p for p in written if p.endswith('.png')])} PNG figures")
    print(f"Output directory: {OUTDIR}")
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
