"""
Summarize the Phase-2 lambda-sweep experiment and produce figures.

Reads the experiment outputs in data/phase2_results/:
  - lambda_sweep_piS{pi_s}.csv     (aggregated per LAMBDA x size x class x level)
  - summary_lam{lam}_piS{pi_s}.csv (per-scenario rows with cost breakdown)

Writes:
  - data/phase2_results/figures/*.png
  - data/phase2_results/findings.txt   (text summary of the key numbers)

Usage
-----
    python phase2_report.py                # pi_S = 2.0 (default)
    python phase2_report.py --pi-s 2
"""

from __future__ import annotations

import os
import csv
import glob
import argparse
import statistics
from collections import defaultdict
import re

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "data", "phase2_results")
FIGDIR = os.path.join(RESULTS, "figures")

CLASSES = ["C", "R", "RC"]
CLASS_COLOR = {"C": "tab:blue", "R": "tab:green", "RC": "tab:orange"}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


SUMMARY_RE = re.compile(
    r"summary_(?:W(?P<W>[0-9]+(?:\.[0-9]+)?)_)?"
    r"lam(?P<lam>[0-9]+(?:\.[0-9]+)?)_"
    r"piS(?P<piS>[0-9]+(?:\.[0-9]+)?)\.csv"
)
SWEEP_RE = re.compile(r"piS(?P<piS>[0-9]+(?:\.[0-9]+)?)\.csv")


def summary_files():
    return sorted(glob.glob(os.path.join(RESULTS, "summary_*.csv")))


def aggregate_summary_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(
            float(r.get("W", 60)),
            float(r["LAMBDA"]),
            float(r["PI_S"]),
            int(r["size"]),
            r["class"],
            int(r["level"]),
        )].append(r)

    out = []
    for (W, lam, pi_s, size, cls, level), rs in sorted(groups.items()):
        fr = [float(r["frac_outsourced"]) for r in rs]
        obj = [float(r["objective"]) for r in rs if r.get("objective")]
        out.append({
            "W": f"{W:g}",
            "LAMBDA": f"{lam:g}",
            "PI_S": f"{pi_s:g}",
            "size": str(size),
            "class": cls,
            "level": str(level),
            "n_scenarios": str(len(rs)),
            "frac_outsourced_mean": f"{statistics.mean(fr):.4f}",
            "frac_outsourced_std": f"{statistics.pstdev(fr) if len(fr) > 1 else 0.0:.4f}",
            "objective_mean": f"{statistics.mean(obj):.4f}" if obj else "",
            "objective_std": f"{statistics.pstdev(obj) if len(obj) > 1 else 0.0:.4f}",
        })
    return out


def load_sweep(pi_s):
    target = float(pi_s)
    for f in sorted(glob.glob(os.path.join(RESULTS, "lambda_sweep_piS*.csv"))):
        m = SWEEP_RE.search(os.path.basename(f))
        if m and abs(float(m.group("piS")) - target) < 1e-9:
            return read_csv(f)
    rows = []
    seen = set()
    for f in summary_files():
        m = SUMMARY_RE.match(os.path.basename(f))
        if not m or abs(float(m.group("piS")) - target) >= 1e-9:
            continue
        key = (float(m.group("W") or 60), float(m.group("lam")), float(m.group("piS")))
        if key in seen:
            continue
        seen.add(key)
        rows.extend(read_csv(f))
    if rows:
        return aggregate_summary_rows(rows)
    raise FileNotFoundError(f"no sweep csv for pi_S={pi_s}")


def load_all_sweeps():
    """Return dict pi_S(float) -> rows for every lambda_sweep_piS*.csv (deduped)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "lambda_sweep_piS*.csv"))):
        m = SWEEP_RE.search(os.path.basename(f))
        if not m:
            continue
        key = float(m.group("piS"))
        if key not in out:
            out[key] = read_csv(f)
    if out:
        return out

    rows_by_piS = defaultdict(list)
    seen = set()
    for f in summary_files():
        m = SUMMARY_RE.match(os.path.basename(f))
        if not m:
            continue
        W = float(m.group("W") or 60)
        lam = float(m.group("lam"))
        pi_s = float(m.group("piS"))
        key = (W, lam, pi_s)
        if key in seen:
            continue
        seen.add(key)
        rows_by_piS[pi_s].extend(read_csv(f))
    for pi_s, rows in rows_by_piS.items():
        out[pi_s] = aggregate_summary_rows(rows)
    return out


def load_per_scenario(pi_s):
    target = float(pi_s)
    rows = []
    seen = set()
    for f in summary_files():
        name = os.path.basename(f)
        m = SUMMARY_RE.match(name)
        if not m:
            continue
        W = float(m.group("W") or 60)
        lam = float(m.group("lam"))
        file_pi_s = float(m.group("piS"))
        key = (W, lam, file_pi_s)
        if abs(file_pi_s - target) < 1e-9 and key not in seen:
            seen.add(key)
            rows.extend(read_csv(f))
    return rows


def load_all_per_scenario():
    """Return dict pi_S(float) -> per-scenario rows, deduping equivalent tags.

    For example, ``piS2`` and ``piS2.0`` represent the same parameter setting.
    """
    out = defaultdict(list)
    seen = set()
    for f in summary_files():
        name = os.path.basename(f)
        m = SUMMARY_RE.match(name)
        if not m:
            continue
        W = float(m.group("W") or 60)
        lam = float(m.group("lam"))
        pi_s = float(m.group("piS"))
        key = (W, lam, pi_s)
        if key in seen:
            continue
        seen.add(key)
        out[pi_s].extend(read_csv(f))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_outsource_vs_lambda_by_class(sweep, lambdas, pi_s):
    """Mean outsource fraction vs lambda, one line per class (avg over size/level)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in CLASSES:
        ys, es = [], []
        for lam in lambdas:
            vals = [float(r["frac_outsourced_mean"]) for r in sweep
                    if float(r["LAMBDA"]) == lam and r["class"] == cls]
            ys.append(statistics.mean(vals))
            es.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(lambdas, ys, yerr=es, marker="o", capsize=3,
                    color=CLASS_COLOR[cls], label=f"{cls}")
    ax.set_xlabel("λ  (outsourcing-price dial)")
    ax.set_ylabel("mean outsource fraction")
    ax.set_title(f"Outsourcing vs λ by class  (π_S={pi_s}, π_C=1)")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="class")
    out = os.path.join(FIGDIR, "outsource_vs_lambda_by_class.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def fig_outsource_vs_lambda_by_size(sweep, lambdas, pi_s):
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = sorted({int(r["size"]) for r in sweep})
    for size in sizes:
        ys = []
        for lam in lambdas:
            vals = [float(r["frac_outsourced_mean"]) for r in sweep
                    if float(r["LAMBDA"]) == lam and int(r["size"]) == size]
            ys.append(statistics.mean(vals))
        ax.plot(lambdas, ys, marker="s", label=f"n={size}")
    ax.set_xlabel("λ"); ax.set_ylabel("mean outsource fraction")
    ax.set_title(f"Outsourcing vs λ by instance size  (π_S={pi_s})")
    ax.set_ylim(0, 1); ax.grid(True, linestyle=":", alpha=0.5); ax.legend(title="size")
    out = os.path.join(FIGDIR, "outsource_vs_lambda_by_size.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def fig_outsource_by_level(sweep, pi_s, lam_ref=1.0):
    """Bar: outsource fraction by emergency-count level, per class (at lambda=lam_ref)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    levels = sorted({int(r["level"]) for r in sweep})
    width = 0.25
    for i, cls in enumerate(CLASSES):
        ys = []
        for lv in levels:
            vals = [float(r["frac_outsourced_mean"]) for r in sweep
                    if float(r["LAMBDA"]) == lam_ref and r["class"] == cls and int(r["level"]) == lv]
            ys.append(statistics.mean(vals) if vals else 0.0)
        xs = [j + (i - 1) * width for j in range(len(levels))]
        ax.bar(xs, ys, width, color=CLASS_COLOR[cls], label=cls)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
    ax.set_xlabel("emergency-count level"); ax.set_ylabel("mean outsource fraction")
    ax.set_title(f"Outsourcing by emergency level  (λ={lam_ref}, π_S={pi_s})")
    ax.set_ylim(0, 1); ax.grid(True, axis="y", linestyle=":", alpha=0.5); ax.legend(title="class")
    out = os.path.join(FIGDIR, "outsource_by_level.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def fig_cost_breakdown(per_scenario, pi_s, lam_ref=1.0):
    """Stacked bar of mean cost components per class (at lambda=lam_ref)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    comps = ["outsourcing", "tw_penalty", "completion_penalty"]
    colors = ["tab:red", "tab:purple", "tab:gray"]
    means = {c: [] for c in comps}
    for cls in CLASSES:
        rows = [r for r in per_scenario
                if abs(float(r["LAMBDA"]) - lam_ref) < 1e-9 and r["class"] == cls]
        for c in comps:
            means[c].append(statistics.mean([float(r[c]) for r in rows]) if rows else 0.0)
    bottom = [0.0] * len(CLASSES)
    for c, col in zip(comps, colors):
        ax.bar(CLASSES, means[c], bottom=bottom, color=col, label=c)
        bottom = [b + m for b, m in zip(bottom, means[c])]
    ax.set_xlabel("class"); ax.set_ylabel("mean cost per scenario")
    ax.set_title(f"Objective composition by class  (λ={lam_ref}, π_S={pi_s})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5); ax.legend()
    out = os.path.join(FIGDIR, "cost_breakdown_by_class.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def fig_outsource_vs_piS(all_sweeps, lam_ref=1.0):
    """Outsource fraction vs pi_S per class (at lambda=lam_ref): shows insensitivity."""
    piSs = sorted(all_sweeps)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in CLASSES:
        ys = []
        for p in piSs:
            vals = [float(r["frac_outsourced_mean"]) for r in all_sweeps[p]
                    if float(r["LAMBDA"]) == lam_ref and r["class"] == cls]
            ys.append(statistics.mean(vals) if vals else 0.0)
        ax.plot(piSs, ys, marker="o", color=CLASS_COLOR[cls], label=cls)
    ax.set_xlabel("π_S  (soft-window penalty)")
    ax.set_ylabel("mean outsource fraction")
    ax.set_title(f"Outsourcing vs π_S by class  (λ={lam_ref}, π_C=1)")
    ax.set_ylim(0, 1); ax.grid(True, linestyle=":", alpha=0.5); ax.legend(title="class")
    out = os.path.join(FIGDIR, "outsource_vs_piS_by_class.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def fig_heatmap_piS_lambda(all_sweeps):
    """Heatmap of overall mean outsource fraction over the pi_S x lambda grid."""
    piSs = sorted(all_sweeps)
    lambdas = sorted({float(r["LAMBDA"]) for r in next(iter(all_sweeps.values()))})
    grid = []
    for p in piSs:
        row = []
        for lam in lambdas:
            vals = [float(r["frac_outsourced_mean"]) for r in all_sweeps[p]
                    if float(r["LAMBDA"]) == lam]
            row.append(statistics.mean(vals))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels(lambdas)
    ax.set_yticks(range(len(piSs))); ax.set_yticklabels(piSs)
    ax.set_xlabel("λ"); ax.set_ylabel("π_S")
    ax.set_title("Overall mean outsource fraction over π_S × λ")
    for i in range(len(piSs)):
        for j in range(len(lambdas)):
            ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center",
                    color="white" if grid[i][j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="outsource fraction")
    out = os.path.join(FIGDIR, "heatmap_piS_lambda.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #
def write_findings(sweep, per_scenario, lambdas, pi_s):
    lines = []
    L = lines.append
    L("=" * 64)
    L("PHASE-2 CALIBRATION FINDINGS")
    L(f"  pi_C = 1 (numeraire), pi_S = {pi_s}, outsourcing f_m = lambda*(c_0m + c_m0)")
    L(f"  scenarios per (size x class x level x seed); lambda grid = {lambdas}")
    L("=" * 64)

    # overall outsource fraction per lambda
    L("\nOverall mean outsource fraction by lambda:")
    for lam in lambdas:
        vals = [float(r["frac_outsourced_mean"]) for r in sweep if float(r["LAMBDA"]) == lam]
        L(f"  lambda={lam:>5}:  {statistics.mean(vals):.3f}")

    # per-class table
    L("\nMean outsource fraction by class x lambda:")
    L(f"  {'lambda':>7}" + "".join(f"{c:>9}" for c in CLASSES))
    for lam in lambdas:
        row = f"  {lam:>7}"
        for cls in CLASSES:
            vals = [float(r["frac_outsourced_mean"]) for r in sweep
                    if float(r["LAMBDA"]) == lam and r["class"] == cls]
            row += f"{statistics.mean(vals):>9.3f}"
        L(row)

    # cost composition at lambda=1
    L("\nMean objective composition per scenario (lambda=1.0):")
    for cls in CLASSES:
        rows = [r for r in per_scenario if abs(float(r["LAMBDA"]) - 1.0) < 1e-9 and r["class"] == cls]
        if not rows:
            continue
        out = statistics.mean([float(r["outsourcing"]) for r in rows])
        tw = statistics.mean([float(r["tw_penalty"]) for r in rows])
        cp = statistics.mean([float(r["completion_penalty"]) for r in rows])
        L(f"  {cls:>3}: outsourcing={out:7.2f}  tw_penalty={tw:6.2f}  completion={cp:7.2f}")

    L("\nKey takeaways:")
    L("  - Insert/outsource balance is structurally insensitive to lambda:")
    L("    quadrupling the outsourcing price barely shifts the split.")
    L("  - C-class (clustered, many requests per trip) stays ~90% outsourced;")
    L("    R-class (dispersed) is the most insertion-friendly and lambda-responsive.")
    L("  - Binding cost is the WITHIN-TRIP completion penalty: a detour delays a")
    L("    trip's return, so every specimen it delivers arrives later. Idle-time")
    L("    absorption (g_T) only mitigates spillover to later trips, not this.")

    path = os.path.join(RESULTS, "findings.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return path


def write_findings_all_piS(all_sweeps, all_per_scenario):
    lines = []
    L = lines.append
    piSs = sorted(all_sweeps)
    lambdas = sorted({float(r["LAMBDA"]) for rows in all_sweeps.values() for r in rows})
    Ws = sorted({float(r.get("W", 60)) for rows in all_sweeps.values() for r in rows})

    L("=" * 72)
    L("PHASE-2 CALIBRATION FINDINGS")
    L("  pi_C = 1 (numeraire)")
    L(f"  emergency-window grid W = {Ws}")
    L(f"  pi_S grid = {piSs}")
    L(f"  lambda grid = {lambdas}")
    L("  outsourcing f_m = lambda*(c_0m + c_m0)")
    L("=" * 72)

    for W in Ws:
        L(f"\nOverall mean outsource fraction by pi_S x lambda (W={W:g}):")
        L(f"  {'pi_S':>7}" + "".join(f"{lam:>9g}" for lam in lambdas))
        for pi_s in piSs:
            row = f"  {pi_s:>7g}"
            for lam in lambdas:
                vals = [float(r["frac_outsourced_mean"]) for r in all_sweeps[pi_s]
                        if float(r.get("W", 60)) == W and float(r["LAMBDA"]) == lam]
                row += f"{statistics.mean(vals):>9.3f}" if vals else f"{'n/a':>9}"
            L(row)

        L(f"\nMean outsource fraction by class at lambda=1.0 (W={W:g}):")
        L(f"  {'pi_S':>7}" + "".join(f"{c:>9}" for c in CLASSES))
        for pi_s in piSs:
            row = f"  {pi_s:>7g}"
            for cls in CLASSES:
                vals = [float(r["frac_outsourced_mean"]) for r in all_sweeps[pi_s]
                        if float(r.get("W", 60)) == W
                        and abs(float(r["LAMBDA"]) - 1.0) < 1e-9
                        and r["class"] == cls]
                row += f"{statistics.mean(vals):>9.3f}" if vals else f"{'n/a':>9}"
            L(row)

        L(f"\nMean objective composition per scenario at lambda=1.0 (W={W:g}):")
        L(f"  {'pi_S':>7} {'class':>6} {'outsourcing':>13} {'tw_penalty':>12} {'completion':>12}")
        for pi_s in piSs:
            rows_pi = all_per_scenario.get(pi_s, [])
            for cls in CLASSES:
                rows = [r for r in rows_pi
                        if float(r.get("W", 60)) == W
                        and abs(float(r["LAMBDA"]) - 1.0) < 1e-9
                        and r["class"] == cls]
                if not rows:
                    continue
                out = statistics.mean([float(r["outsourcing"]) for r in rows])
                tw = statistics.mean([float(r["tw_penalty"]) for r in rows])
                cp = statistics.mean([float(r["completion_penalty"]) for r in rows])
                L(f"  {pi_s:>7g} {cls:>6} {out:>13.2f} {tw:>12.2f} {cp:>12.2f}")

        baseline_pi_s = 2.0
        if baseline_pi_s in all_per_scenario:
            L(f"\nOutsourcing reason split at pi_S=2 (W={W:g}):")
            L(f"  {'lambda':>7} {'inserted':>9} {'out_infeas':>11} {'out_econ':>9} "
              f"{'out_total':>9} {'infeas/out':>11}")
            rows_pi = [r for r in all_per_scenario[baseline_pi_s]
                       if float(r.get("W", 60)) == W]
            for lam in lambdas:
                rows = [r for r in rows_pi if abs(float(r["LAMBDA"]) - lam) < 1e-9]
                if not rows or "n_outsourced_infeasible" not in rows[0]:
                    continue
                n_em = sum(int(r["n_emergencies"]) for r in rows)
                n_inserted = sum(int(r.get("n_inserted", 0)) for r in rows)
                n_infeas = sum(int(r.get("n_outsourced_infeasible", 0)) for r in rows)
                n_econ = sum(int(r.get("n_outsourced_economic", 0)) for r in rows)
                n_out = sum(int(r["n_outsourced"]) for r in rows)
                infeas_out = n_infeas / n_out if n_out else 0.0
                L(f"  {lam:>7g} {n_inserted / n_em:>9.3f} {n_infeas / n_em:>11.3f} "
                  f"{n_econ / n_em:>9.3f} {n_out / n_em:>9.3f} {infeas_out:>11.3f}")

            L(f"\nFeasible-emergency decision split at pi_S=2 (W={W:g}):")
            L("  Denominator = inserted + economically outsourced emergencies only;")
            L("  forced infeasible outsourcing is excluded from this table.")
            L(f"  {'lambda':>7} {'n_feasible':>11} {'inserted':>10} {'out_econ':>10}")
            for lam in lambdas:
                rows = [r for r in rows_pi if abs(float(r["LAMBDA"]) - lam) < 1e-9]
                if not rows or "n_outsourced_economic" not in rows[0]:
                    continue
                n_inserted = sum(int(r.get("n_inserted", 0)) for r in rows)
                n_econ = sum(int(r.get("n_outsourced_economic", 0)) for r in rows)
                n_feasible = n_inserted + n_econ
                if not n_feasible:
                    continue
                L(f"  {lam:>7g} {n_feasible:>11d} "
                  f"{n_inserted / n_feasible:>10.3f} {n_econ / n_feasible:>10.3f}")

    L("\nKey takeaways:")
    L("  - Across all W and pi_S values, the outsource fraction changes much more")
    L("    with lambda and emergency-window width than with pi_S.")
    L("  - The outsourcing reason split shows how much of outsourcing is forced by")
    L("    infeasible insertion versus chosen economically by the objective.")
    L("  - Conditional on insertion being feasible, most emergencies are inserted")
    L("    once lambda reaches 1; the remaining outsourcing is the economic choice.")
    L("  - Higher pi_S mainly changes the cost composition by increasing the")
    L("    penalty attached to regular-request lateness; it does not overturn the")
    L("    insertion/outsourcing pattern.")
    L("  - C-class instances remain the most outsourcing-heavy; R-class instances")
    L("    remain the most insertion-friendly.")
    L("  - The main operational trade-off is still the trip-level completion delay:")
    L("    inserted emergencies delay the return of all specimens in that trip.")

    path = os.path.join(RESULTS, "findings.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description="Summarize Phase-2 lambda-sweep + figures.")
    ap.add_argument("--pi-s", default="2.0", dest="pi_s")
    args = ap.parse_args()
    pi_s = args.pi_s

    os.makedirs(FIGDIR, exist_ok=True)
    sweep = load_sweep(pi_s)
    per_scenario = load_per_scenario(pi_s)
    lambdas = sorted({float(r["LAMBDA"]) for r in sweep})

    figs = []
    if plt is not None:
        figs = [
            fig_outsource_vs_lambda_by_class(sweep, lambdas, pi_s),
            fig_outsource_vs_lambda_by_size(sweep, lambdas, pi_s),
            fig_outsource_by_level(sweep, pi_s),
            fig_cost_breakdown(per_scenario, pi_s),
        ]

    all_sweeps = load_all_sweeps()
    all_per_scenario = load_all_per_scenario()

    # pi_S-dimension figures (if multiple pi_S sweeps are present)
    if plt is not None and len(all_sweeps) > 1:
        figs.append(fig_outsource_vs_piS(all_sweeps))
        figs.append(fig_heatmap_piS_lambda(all_sweeps))

    if len(all_sweeps) > 1:
        findings = write_findings_all_piS(all_sweeps, all_per_scenario)
    else:
        findings = write_findings(sweep, per_scenario, lambdas, pi_s)

    if plt is None:
        print("\nFigures skipped: matplotlib is not installed in this Python environment.")
    else:
        print("\nFigures:")
        for f in figs:
            print("  " + f)
    print("Findings:", findings)


if __name__ == "__main__":
    main()
