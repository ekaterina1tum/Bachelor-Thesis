"""
Create thesis tables from Phase-2 summary CSV files.

For each W and each size group (10, 15, 25, overall), the script creates
matrix tables with rows = pi_S, columns = lambda, and cells equal to:

- total outsourced fraction
- economically outsourced fraction
- infeasible outsourced fraction
- mean objective function value

Outputs:
    data/phase2_results/tables/phase2_tables.md
    data/phase2_results/tables/*.csv
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "data", "phase2_results")
OUT_DIR = os.path.join(RESULTS, "tables")

SUMMARY_RE = re.compile(
    r"summary_W(?P<W>[0-9]+(?:\.[0-9]+)?)_"
    r"lam(?P<lam>[0-9]+(?:\.[0-9]+)?)_"
    r"piS(?P<piS>[0-9]+(?:\.[0-9]+)?)\.csv"
)

SIZES = [10, 15, 25]
SIZE_GROUPS = ["overall", "10", "15", "25"]
METRICS = [
    ("out_total", "Total Outsourced Fraction"),
    ("out_econ", "Economically Outsourced Fraction"),
    ("out_infeas", "Infeasible Outsourced Fraction"),
    ("objective", "Mean Objective Function Value"),
]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def discover_summary_files():
    files = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "summary_W*_lam*_piS*.csv"))):
        m = SUMMARY_RE.match(os.path.basename(path))
        if not m:
            continue
        files.append((
            float(m.group("W")),
            float(m.group("lam")),
            float(m.group("piS")),
            path,
        ))
    return files


def summarize_rows(rows):
    n_em = sum(int(r["n_emergencies"]) for r in rows)
    if n_em == 0:
        return None
    n_out = sum(int(r["n_outsourced"]) for r in rows)
    n_econ = sum(int(r.get("n_outsourced_economic", 0)) for r in rows)
    n_infeas = sum(int(r.get("n_outsourced_infeasible", 0)) for r in rows)
    objectives = [float(r["objective"]) for r in rows if r.get("objective")]
    return {
        "out_total": n_out / n_em,
        "out_econ": n_econ / n_em,
        "out_infeas": n_infeas / n_em,
        "objective": sum(objectives) / len(objectives) if objectives else 0.0,
    }


def build_cube():
    cube = {}
    for W, lam, pi_s, path in discover_summary_files():
        rows = read_csv(path)
        for group in SIZE_GROUPS:
            if group == "overall":
                group_rows = rows
            else:
                group_rows = [r for r in rows if int(r["size"]) == int(group)]
            vals = summarize_rows(group_rows)
            if vals is not None:
                cube[(W, group, pi_s, lam)] = vals
    return cube


def fmt_value(metric, value):
    if metric == "objective":
        return f"{value:.2f}"
    return f"{value:.3f}"


def csv_safe_num(value):
    return f"{value:g}"


def write_metric_csv(cube, W, group, metric, pi_values, lam_values):
    filename = f"W{csv_safe_num(W)}_{group}_{metric}.csv"
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pi_S"] + [csv_safe_num(lam) for lam in lam_values])
        for pi_s in pi_values:
            row = [csv_safe_num(pi_s)]
            for lam in lam_values:
                vals = cube.get((W, group, pi_s, lam))
                row.append(fmt_value(metric, vals[metric]) if vals else "")
            writer.writerow(row)
    return path


def markdown_table(cube, W, group, metric, title, pi_values, lam_values):
    lines = []
    lines.append(f"### W={csv_safe_num(W)}, size={group}, {title}")
    lines.append("")
    header = ["pi_S"] + [f"lambda={csv_safe_num(lam)}" for lam in lam_values]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for pi_s in pi_values:
        row = [csv_safe_num(pi_s)]
        for lam in lam_values:
            vals = cube.get((W, group, pi_s, lam))
            row.append(fmt_value(metric, vals[metric]) if vals else "")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def write_markdown(cube):
    Ws = sorted({key[0] for key in cube})
    pi_values = sorted({key[2] for key in cube})
    lam_values = sorted({key[3] for key in cube})

    lines = []
    lines.append("# Phase-2 Sensitivity Tables")
    lines.append("")
    lines.append("Rows are `pi_S`; columns are `lambda`. Tables are reported for each `W` and for sizes `10`, `15`, `25`, plus the overall average.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("")
    lines.append("- `Total Outsourced Fraction`: all outsourced emergencies divided by all emergencies.")
    lines.append("- `Economically Outsourced Fraction`: emergencies outsourced even though insertion was feasible.")
    lines.append("- `Infeasible Outsourced Fraction`: emergencies outsourced because no feasible insertion arc existed.")
    lines.append("- `Mean Objective Function Value`: average Phase-2 objective per scenario.")
    lines.append("")

    csv_paths = []
    for W in Ws:
        lines.append(f"## W={csv_safe_num(W)}")
        lines.append("")
        for group in SIZE_GROUPS:
            for metric, title in METRICS:
                lines.extend(markdown_table(cube, W, group, metric, title, pi_values, lam_values))
                csv_paths.append(write_metric_csv(cube, W, group, metric, pi_values, lam_values))

    md_path = os.path.join(OUT_DIR, "phase2_tables.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    return md_path, csv_paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cube = build_cube()
    if not cube:
        raise FileNotFoundError(
            f"No summary_W*_lam*_piS*.csv files found in {RESULTS}."
        )
    md_path, csv_paths = write_markdown(cube)
    print(f"Markdown tables: {md_path}")
    print(f"CSV tables written: {len(csv_paths)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
