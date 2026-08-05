# Source Code Map

This folder is organized around the two-phase workflow.

## Phase 1: Regular CDSP

- `instance.py`: instance loader and distance matrix construction.
- `graph.py`: extended graph construction with original and replenishment arcs.
- `model.py`: Phase 1 CDSP Gurobi model.
- `phase1_export.py`: solve one Phase 1 instance and export the fixed baseline JSON.
- `phase1_batch.py`: batch-run Phase 1 and summarize objective values.
- `patch_gT.py`: maintenance helper for recomputing inter-trip idle slack `g_T` in old Phase 1 JSONs.

## Phase 2: Emergency Insertion vs Outsourcing

- `emergency_generator.py`: generate emergency locations, releases, and deadlines.
- `phase2_instance.py`: build Phase 2 input objects from Phase 1 JSONs.
- `phase2_model.py`: Phase 2 ESCP Gurobi model.
- `phase2_params.py`: experiment grids and cost parameters.
- `phase2_experiment.py`: run Phase 2 solves over scenarios and parameter settings.
- `phase2_example.py`: small hand-built Phase 2 sanity example.

## Reporting and Diagnostics

- `phase2_report.py`: write the main `findings.txt` report.
- `phase2_feasibility_report.py`: evaluation-only feasibility and W-calibration diagnostics.
- `phase2_make_tables.py`: generate Markdown and CSV sensitivity tables.
- `phase2_plot_results.py`: plot one solved Phase 2 scenario.
- `plot_instance.py`: plot benchmark instance geometry.

The main project workflow is documented in the repository-level `README.md`.
