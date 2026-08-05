"""
Phase-2 calibration parameters (presets only).

This module ONLY defines the calibration constants, the sweep grids, and a single
outsourcing-cost helper. It runs nothing: no experiments, no scenario loops, no
Gurobi solves, no file I/O, no reporting. A separate experiment/model file imports
from here and drives the runs.

Cost model the experiment will assemble (for reference only -- not built here):

    minimize  sum_m  f_m * o_m
            + PI_S * sum_j  lateness_j
            + PI_C * sum_j  completion_increase_j

PI_C is fixed at 1 (numéraire). LAMBDA and PI_S are the only dials meant to be
swept later. There are no service times anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Fixed / base parameters
# --------------------------------------------------------------------------- #
PI_C = 1.0      # completion-time penalty; numéraire: 1 cost unit = 1 min of completion increase
PI_S = 2.0      # base soft-window violation penalty (per minute of regular-request lateness)
LAMBDA = 1.0    # base outsourcing-price dial

# --------------------------------------------------------------------------- #
# Sweep grids (defaults for later use -- NOT executed here)
# --------------------------------------------------------------------------- #
PI_S_GRID = [0.5, 1, 2, 5, 10]
LAMBDA_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
W_GRID = [60, 80, 90, 120]   # urgency windows to run the full grid over

# Depot node index in the distance matrix (matches the rest of the codebase).
DEPOT = 0


# --------------------------------------------------------------------------- #
# Outsourcing-cost helper
# --------------------------------------------------------------------------- #
def outsourcing_cost(m_node, dist, lam: float = LAMBDA, depot: int = DEPOT) -> float:
    """Outsourcing price f_m for an emergency, with no median / distribution / preprocessing.

        f_m = lambda * ( depot -> m  +  m -> depot )

    `dist` follows the codebase convention: a mapping keyed by ordered node pairs,
    i.e. ``dist[(i, j)]`` is the travel time/cost from i to j (as in
    ``Instance.travel_time`` / ``Instance.c``). The emergency node ``m_node`` and the
    depot must already be present in ``dist``.
    """
    return lam * (dist[(depot, m_node)] + dist[(m_node, depot)])


# --------------------------------------------------------------------------- #
# Bundled config (import one thing in the experiment file)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Phase2Params:
    """Base Phase-2 calibration values bundled for convenient import."""
    PI_C: float = PI_C
    PI_S: float = PI_S
    LAMBDA: float = LAMBDA
    PI_S_GRID: list = field(default_factory=lambda: list(PI_S_GRID))
    LAMBDA_GRID: list = field(default_factory=lambda: list(LAMBDA_GRID))
    DEPOT: int = DEPOT


# Default instance the experiment file can import directly.
PARAMS = Phase2Params()
