# Bachelor Thesis Code

This repository contains the computational work for a two-phase biomedical specimen collection problem.

Phase 1 solves the regular Collection and Delivery Problem of biomedical Specimens (CDSP), following the replenishment-arc formulation of Rocha, Otto, and Goerigk. Phase 2 takes the fixed Phase 1 routes and decides how to handle emergency requests: insert them into existing trips or outsource them to a third-party logistics provider.

The code is intentionally split into model construction, data generation, experiments, and reporting so that results can be reproduced and checked step by step.

## Main Outcomes

The current experiments show that emergency outsourcing is mostly driven by insertion feasibility, not only by outsourcing price.

Key findings:

- Emergency window width `W` is the strongest driver. Increasing `W` lowers the forced-outsourcing floor because more emergencies can be inserted before their hard deadline.
- The outsourcing price multiplier `lambda` has a limited effect when many emergencies are infeasible. Even high outsourcing prices cannot force insertion if no feasible insertion arc exists.
- The soft regular time-window penalty `pi_S` has very little effect. Regular TW violations are rare, so `tw_penalty` is close to zero in most runs.
- Completion delay matters more than TW violation. Inserted emergencies mainly delay trip return times, so the relevant disruption is regular specimen completion delay.
- The calibrated global emergency window from the `025` instances is approximately `W=80`, computed as `2 x p90(d_0j)`.
- `W=90` is a more conservative setting that better covers the `RC` class.

The reason split in `data/phase2_results/findings.txt` shows how much outsourcing is forced by infeasibility:

```text
W=80, lambda=5, pi_S=2:
inserted    = 0.401
out_infeas = 0.545
out_econ   = 0.053
out_total  = 0.599
infeas/out = 0.911
```

Interpretation: at `W=80` and high outsourcing price, about `91.1%` of outsourced emergencies are outsourced because insertion is infeasible, not because outsourcing is cheaper.

For `W=90`, feasibility improves:

```text
W=90, lambda=5, pi_S=2:
inserted    = 0.454
out_infeas = 0.483
out_econ   = 0.063
out_total  = 0.546
infeas/out = 0.885
```

Interpretation: `W=90` reduces total outsourcing and forced outsourcing, but most outsourced emergencies are still infeasibility-driven.

Small-instance results show the same pattern. At `lambda=5` and `pi_S=2`, most outsourcing for sizes `10` and `15` is still forced by infeasibility:

```text
W=80, lambda=5, pi_S=2:
size  inserted  out_infeas  out_econ  out_total  infeas/out
10       0.302       0.646     0.053      0.698       0.925
15       0.418       0.542     0.040      0.582       0.932

W=90, lambda=5, pi_S=2:
size  inserted  out_infeas  out_econ  out_total  infeas/out
10       0.344       0.597     0.059      0.656       0.910
15       0.465       0.486     0.049      0.535       0.909
```

Interpretation: size `15` has more feasible insertion opportunities than size `10`, so it inserts more emergencies and outsources less. However, in both small-size groups, more than `90%` of outsourced emergencies at high lambda are still outsourced because insertion is infeasible.

## Project Structure

```text
.
|-- README.md
|-- test_gurobi.py
|-- colab/
|   `-- THESIS_Colab.ipynb
|-- data/
|   |-- MSCDPinstances/
|   |   |-- 010/
|   |   |-- 015/
|   |   |-- 025/
|   |   |-- 050/
|   |   `-- 100/
|   |-- phase1_solutions/
|   |-- emergency_scenarios/
|   `-- phase2_results/
`-- src/
    |-- instance.py
    |-- graph.py
    |-- model.py
    |-- phase1_export.py
    |-- phase1_batch.py
    |-- emergency_generator.py
    |-- phase2_instance.py
    |-- phase2_model.py
    |-- phase2_params.py
    |-- phase2_experiment.py
    |-- phase2_report.py
    |-- phase2_feasibility_report.py
    |-- phase2_plot_results.py
    |-- plot_instance.py
    `-- verify.py
```

## Main Data Folders

### `data/MSCDPinstances/`

Input benchmark instances. The folders are grouped by number of regular requests:

```text
010, 015, 025, 050, 100
```

Each instance file contains coordinates and time windows. Node `0` is the depot/laboratory. All other nodes are points of care.

### `data/phase1_solutions/`

JSON exports of solved Phase 1 routes. These files are the fixed baseline used by Phase 2. They contain:

- routes and trips
- baseline collection times `z0`
- trip return times `C0`
- regular-request soft windows
- arcs before each regular request
- baseline route duration `Lambda0`

Phase 2 should use these JSON files through `load_phase2_solution()`.

Do not use the deprecated in-memory bridge `extract_phase2_instance()`. It is intentionally disabled because it can lose the correct depot-to-depot trip structure.

### `data/emergency_scenarios/`

Generated emergency requests, grouped by emergency deadline window:

```text
data/emergency_scenarios/W60/
data/emergency_scenarios/W80/
data/emergency_scenarios/W90/
```

Each scenario contains emergency locations, releases, deadlines, and metadata. The emergency deadline is:

```text
deadline = release + W
```

Important: for fair W comparisons, releases should be generated with the same `release_base_W`, so only the deadline width changes across W values.

### `data/phase2_results/`

Phase 2 result files. Each parameter setting gets a folder such as:

```text
W80_lam1.0_piS2/
```

and a corresponding summary file:

```text
summary_W80_lam1.0_piS2.csv
```

The report scripts read these summaries to create `findings.txt`.

## Phase 1: Regular CDSP

Phase 1 builds and solves the regular specimen collection model.

Main files:

- `src/instance.py`: loads benchmark instances and computes travel times.
- `src/graph.py`: builds the extended graph with depot arcs, point-of-care arcs, and replenishment arcs.
- `src/model.py`: builds the Rocha et al. CDSP MIP.
- `src/phase1_export.py`: solves one instance and exports the fixed baseline route solution to JSON.
- `src/phase1_batch.py`: solves many Phase 1 instances and reports objective values.

The Phase 1 model minimizes total request completion time:

```text
min sum_j C_j
```

It includes:

- depot flow and vehicle limit
- exactly one visit per point of care
- hard regular time windows
- completion-time calculation
- maximum shift duration
- replenishment arcs for multi-trip routes

### Solve and Export One Phase 1 Instance

```bash
python src/phase1_export.py data/MSCDPinstances/025/025_C101.txt 3600 480
```

Arguments:

```text
instance path
time limit in seconds
maximum shift duration
```

Output goes to:

```text
data/phase1_solutions/025/025_C101.json
```

### Run Phase 1 Batch

```bash
python src/phase1_batch.py data/MSCDPinstances/025 3600 480
```

Phase 1 does not need to be rerun unless:

- the Phase 1 model changes,
- the benchmark instances change,
- `max_shift` changes,
- or you intentionally want new baseline routes.

## Phase 2: Emergency Sample Collection Problem

Phase 2 starts from fixed Phase 1 routes. It decides for each emergency request whether to:

1. insert it into an existing route arc, or
2. outsource it to a third-party provider.

Main files:

- `src/emergency_generator.py`: creates emergency scenario JSON files.
- `src/phase2_instance.py`: loads Phase 1 JSON and builds Phase 2 input objects.
- `src/phase2_model.py`: builds the emergency insert-vs-outsource MIP.
- `src/phase2_params.py`: defines parameter grids.
- `src/phase2_experiment.py`: runs Phase 2 experiments.
- `src/phase2_report.py`: creates `findings.txt`.
- `src/phase2_feasibility_report.py`: diagnoses forced outsourcing and W calibration.

There are no service times anywhere in Phase 2. Insertion adds only detour travel time:

```text
delta_m,e = t_s(e),m + t_m,t(e) - t_s(e),t(e)
```

## Phase 2 Model Logic

For each emergency request `m`, the model chooses exactly one option:

```text
outsource m
or
insert m on one feasible arc e
```

The objective is:

```text
outsourcing cost
+ regular time-window violation penalty
+ regular completion-time delay penalty
```

The important parameters are:

```text
W        emergency hard deadline width
lambda   outsourcing price multiplier
pi_S     soft regular collection-window penalty
pi_C     regular completion-time penalty
```

Current defaults are defined in `src/phase2_params.py`.

`pi_C` is kept fixed at `1`, the numeraire. The main sensitivity grid uses `W`, `lambda`, and `pi_S`.

## Emergency Scenario Generation

Generate scenarios with:

```bash
python src/emergency_generator.py --W 80 --release-base-W 90
```

The meaning of these two values is different:

```text
W
    The actual emergency deadline width.
    deadline = release + W

release_base_W
    The fixed cap used only when generating releases.
    release ~ Uniform(0, horizon - release_base_W)
```

Why this matters:

If releases were generated as `Uniform(0, horizon - W)`, then changing W would also change release times. That would make W comparisons unfair. Using one fixed `release_base_W` keeps locations and release times comparable across W values.

For example, if the final study compares only `W <= 90`, use:

```bash
python src/emergency_generator.py --W 60 --release-base-W 90
python src/emergency_generator.py --W 80 --release-base-W 90
python src/emergency_generator.py --W 90 --release-base-W 90
```

Then only the emergency deadline changes:

```text
W=60: deadline = release + 60
W=80: deadline = release + 80
W=90: deadline = release + 90
```

## W Calibration

The calibrated emergency window is based on depot-to-customer travel distances:

```text
W = 2 x p-th percentile of d_0j
```

Using only the `025` instances and `p = 90`, the global value is approximately:

```text
p90(d_0j) = 40.05
W = 2 x 40.05 = 80.10
```

So the calibrated global value is:

```text
W = 80
```

Class-specific values from the `025` instances are:

```text
C:  78
R:  67
RC: 90
```

For one clean thesis-wide setting, use the global calibrated value `W=80`. For a more conservative setting that better covers RC instances, `W=90` is also defensible.

## Running Phase 2

### One Parameter Setting

```bash
python src/phase2_experiment.py --W 80 --lam 1.0 --pi-s 2 --pi-c 1
```

### Lambda Sweep at One W

```bash
python src/phase2_experiment.py --W 80 --lam-grid --pi-s 2 --pi-c 1
```

This is useful for testing whether outsourcing responds meaningfully to price.

### Full Grid

```bash
python src/phase2_experiment.py --full-grid
```

The full grid uses:

```text
W_GRID
LAMBDA_GRID
PI_S_GRID
PI_C = 1
```

from `src/phase2_params.py`.

Before running a W value, make sure its scenarios exist:

```bash
python src/emergency_generator.py --W 80 --release-base-W 90
```

If a scenario folder is missing, the experiment runner will now tell you which W must be generated.

## Reporting

### Main Findings Report

```bash
python src/phase2_report.py
```

Output:

```text
data/phase2_results/findings.txt
```

This report includes:

- outsource fraction by `W`, `lambda`, and `pi_S`
- class-level outsource fractions
- objective composition
- outsourcing reason split at `pi_S=2`

The outsourcing reason split is especially important:

```text
inserted
out_infeas
out_econ
out_total
infeas/out
```

Definitions:

```text
out_infeas
    Emergency was outsourced because no feasible insertion arc existed.

out_econ
    Emergency could have been inserted, but outsourcing was cheaper.

infeas/out
    Among outsourced emergencies, the share caused by infeasibility.
```

Example interpretation:

```text
W=80, lambda=5:
out_total  = 0.599
out_infeas = 0.545
out_econ   = 0.053
infeas/out = 0.911
```

This means that at `W=80`, even with high outsourcing price, most outsourced emergencies are forced by infeasibility rather than chosen economically.

### Feasibility and W Recalibration Report

```bash
python src/phase2_feasibility_report.py
```

Output:

```text
data/phase2_results/feasibility_recalibration_findings.txt
```

This report is evaluation-only and does not solve Gurobi models. It computes:

- undeliverable fraction
- infeasible-insertion fraction
- route-cap proxy for small instances
- depot-to-customer travel scale
- W calibration diagnostics
- price-sweep summary if matching results already exist

The key feasibility check is:

```text
release-feasible arc
and
C0_T + delta_m,e <= emergency deadline
```

## Interpreting the Current Results

The current results show that outsourcing is often not primarily a price decision. Much of it is forced by insertion infeasibility.

The main pattern is:

```text
W has a large effect.
lambda has a limited effect.
pi_S has very little effect.
tw_penalty is close to zero.
completion_penalty matters more than tw_penalty.
```

This happens because inserted emergencies mostly delay trip return times, not regular collection times. Therefore, they increase completion delay but often do not violate regular collection windows.

In thesis terms:

> The high outsourcing share is largely driven by the absence of feasible insertion positions under tight emergency deadlines. Increasing W reduces the forced-outsourcing floor. Once more emergencies become insertable, lambda becomes a more meaningful economic lever.

## Useful Checks

Check that Phase 2 summaries exist:

```bash
ls data/phase2_results/summary_W*_lam*_piS*.csv
```

Check one summary file:

```bash
head data/phase2_results/summary_W80_lam1.0_piS2.csv
```

Run syntax checks:

```bash
python -m py_compile src/*.py
```

Check Gurobi:

```bash
python test_gurobi.py
```

## Recommended Reproducible Workflow

A clean workflow for the calibrated W study is:

```bash
# 1. Export Phase 1 solutions, only if needed.
python src/phase1_export.py data/MSCDPinstances/025/025_C101.txt 3600 480

# 2. Generate comparable emergency scenarios.
python src/emergency_generator.py --W 60 --release-base-W 90
python src/emergency_generator.py --W 80 --release-base-W 90
python src/emergency_generator.py --W 90 --release-base-W 90

# 3. Run Phase 2.
python src/phase2_experiment.py --full-grid

# 4. Generate findings.
python src/phase2_report.py
python src/phase2_feasibility_report.py
```

For a faster focused run:

```bash
python src/phase2_experiment.py --W 80 --lam-grid --pi-s 2 --pi-c 1
python src/phase2_report.py
```

## Notes

- Phase 1 requires Gurobi.
- Phase 2 solving requires Gurobi.
- The report scripts can still write text findings without plotting libraries.
- No service times are included in the model or scenario generation.
- Keep Phase 1 JSON files fixed when comparing Phase 2 parameters.
- Regenerate Phase 2 results after regenerating emergency scenarios, because old results correspond to old releases/deadlines.
