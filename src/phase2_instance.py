from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Phase2Arc:
    """An arc that appears in the fixed Phase-1 solution.

    Each arc belongs to exactly one trip (and therefore one route).
    """
    id: int          # unique arc id across all used arcs
    src: int         # source node
    tgt: int         # target node
    trip_id: int     # T: trip this arc belongs to
    route_id: int    # k: route this arc belongs to


@dataclass
class Trip:
    """A single depot-to-depot segment (trip T)."""
    id: int
    route_id: int
    arcs: list[Phase2Arc]     # ordered arcs in this trip
    C0: float                  # baseline return (lab-arrival) time C^0_T
    g: float                   # idle time to next trip g_T (0 if last)
    is_first: bool             # T in T_0
    is_last: bool              # T in T^last
    next_trip_id: Optional[int]  # T+


@dataclass
class RegularRequest:
    """A regular request j with its Phase-1 baseline data."""
    id: int
    trip_id: int       # T(j)
    z0: float          # baseline collection time z^0_j
    d: float           # soft collection window upper bound d_j
    pi_tw: float       # penalty per unit of TW violation
    pi_c: float        # penalty per unit of completion-time increase
    B: list[Phase2Arc] # arcs in T(j) traversed *before* j is collected


@dataclass
class EmergencyRequest:
    """An emergency request m."""
    id: int
    rho: float          # release time rho_m
    d_bar: float        # hard delivery deadline d_bar_m
    f: float            # outsourcing cost f_m
    A_m: list[Phase2Arc]  # candidate insertion arcs (release+reachability feasible)


@dataclass
class Phase2Instance:
    """All input needed to solve the ESCP (Phase 2)."""

    # ---- Phase-1 derived data ----
    routes: dict[int, list[int]]          # k -> ordered trip ids
    trips: dict[int, Trip]                # trip_id -> Trip
    regular_requests: dict[int, RegularRequest]   # j -> RegularRequest
    Lambda0: dict[int, float]             # k -> baseline route duration Lambda^0_k
    tau_max: float                        # maximum shift duration

    # ---- Emergency requests ----
    emergency_requests: dict[int, EmergencyRequest]   # m -> EmergencyRequest

    # ---- Travel times (for computing delta_{m,e}) ----
    travel_time: dict[tuple[int, int], float]

    # ---- Derived index sets (computed in __post_init__) ----
    arcs: dict[int, Phase2Arc] = field(default_factory=dict)          # arc_id -> Phase2Arc
    arc_trip: dict[int, int] = field(default_factory=dict)            # arc_id -> trip_id
    trip_arcs: dict[int, list[int]] = field(default_factory=dict)     # trip_id -> [arc_ids]
    T0: list[int] = field(default_factory=list)                       # first-trip ids
    T_last: list[int] = field(default_factory=list)                   # last-trip ids
    T_m: dict[int, list[int]] = field(default_factory=dict)           # m -> trip ids with >=1 candidate arc
    delta: dict[tuple[int, int], float] = field(default_factory=dict) # (m, arc_id) -> extra travel time
    M_big: float = field(default=0.0)                                  # big-M constant

    def __post_init__(self) -> None:
        # Collect all arcs
        for trip in self.trips.values():
            for arc in trip.arcs:
                self.arcs[arc.id] = arc
                self.arc_trip[arc.id] = arc.trip_id
                self.trip_arcs.setdefault(arc.trip_id, []).append(arc.id)

        # T_0 and T_last
        self.T0 = [t.id for t in self.trips.values() if t.is_first]
        self.T_last = [t.id for t in self.trips.values() if t.is_last]

        # delta_{m,e} = t_{s(e),m} + t_{m,t(e)} - t_{s(e),t(e)}
        for m, em in self.emergency_requests.items():
            for arc in em.A_m:
                key = (m, arc.id)
                self.delta[key] = (
                    self.travel_time[(arc.src, em.id)]
                    + self.travel_time[(em.id, arc.tgt)]
                    - self.travel_time[(arc.src, arc.tgt)]
                )

        # T_m: trips that contain at least one arc from A_m
        for m, em in self.emergency_requests.items():
            trip_set = {arc.trip_id for arc in em.A_m}
            self.T_m[m] = list(trip_set)

        # big-M: max route start time + tau_max
        # We approximate route start as 0 for all routes (depot departs at t=0)
        # More precisely, use max C^0_T across all trips + tau_max as a safe upper bound
        max_C0 = max((t.C0 for t in self.trips.values()), default=0.0)
        self.M_big = max_C0 + self.tau_max

    def t(self, i: int, j: int) -> float:
        return self.travel_time[(i, j)]


def load_phase2_solution(
    json_path: str,
    emergency_requests: dict[int, EmergencyRequest],
    travel_time: dict[tuple[int, int], float],
    penalty_tw: dict[int, float] | None = None,
    penalty_c: dict[int, float] | None = None,
    tau_max: float | None = None,
) -> Phase2Instance:
    """Build a Phase2Instance from a Phase-1 solution JSON (written by phase1_export).

    This decouples the phases: Phase 1 is solved once and dumped to disk; Phase 2
    reads the JSON without needing a live Gurobi model in memory.

    Parameters
    ----------
    json_path : path to a phase1_solutions/<size>/<instance>.json file
    emergency_requests : dict m -> EmergencyRequest, with A_m referencing arc ids
        that exist in the solution file
    travel_time : full travel-time dict including emergency nodes
    penalty_tw, penalty_c : per-request penalty weights (default 1.0)
    tau_max : maximum shift duration (defaults to the value stored in the JSON)
    """
    penalty_tw = penalty_tw or {}
    penalty_c = penalty_c or {}

    with open(json_path) as fh:
        doc = json.load(fh)

    if tau_max is None:
        tau_max = float(doc["max_shift"])

    # ---- Rebuild arcs (one Phase2Arc per arc id) ----
    arc_by_id: dict[int, Phase2Arc] = {}
    for t in doc["trips"]:
        for a in t["arcs"]:
            arc_by_id[a["id"]] = Phase2Arc(
                id=a["id"], src=a["src"], tgt=a["tgt"],
                trip_id=t["id"], route_id=t["route_id"],
            )

    # ---- Trips ----
    trips: dict[int, Trip] = {}
    for t in doc["trips"]:
        trips[t["id"]] = Trip(
            id=t["id"],
            route_id=t["route_id"],
            arcs=[arc_by_id[a["id"]] for a in t["arcs"]],
            C0=t["C0"],
            g=t.get("g", 0.0),
            is_first=t["is_first"],
            is_last=t["is_last"],
            next_trip_id=t["next_trip_id"],
        )

    # ---- Regular requests ----
    regular_requests: dict[int, RegularRequest] = {}
    for r in doc["regular_requests"]:
        j = r["id"]
        regular_requests[j] = RegularRequest(
            id=j,
            trip_id=r["trip_id"],
            z0=r["z0"],
            d=r["d"],
            pi_tw=penalty_tw.get(j, 1.0),
            pi_c=penalty_c.get(j, 1.0),
            B=[arc_by_id[aid] for aid in r["B"]],
        )

    routes = {int(k): v for k, v in doc["routes"].items()}
    Lambda0 = {int(k): v for k, v in doc["Lambda0"].items()}

    # Validate that every emergency's candidate arcs exist in this solution
    for m, em in emergency_requests.items():
        for arc in em.A_m:
            if arc.id not in arc_by_id:
                raise ValueError(
                    f"Emergency {m} references arc id {arc.id} not present in {json_path}"
                )

    return Phase2Instance(
        routes=routes,
        trips=trips,
        regular_requests=regular_requests,
        Lambda0=Lambda0,
        tau_max=tau_max,
        emergency_requests=emergency_requests,
        travel_time=travel_time,
    )


def extract_phase2_instance(
    phase1_model,
    emergency_requests: dict[int, EmergencyRequest],
    travel_time: dict[tuple[int, int], float],
    tau_max: float,
    penalty_tw: dict[int, float],
    penalty_c: dict[int, float],
    outsource_cost: dict[int, float],
) -> Phase2Instance:
    """Deprecated in-memory bridge from a solved Phase-1 Gurobi model.

    Use ``phase1_export.export_instance`` followed by ``load_phase2_solution``
    instead. The JSON path expands replenishment arcs into real depot-to-depot
    trips and preserves the baseline values required by the ESCP formulation.

    Parameters
    ----------
    phase1_model : solved gp.Model with _x, _z, _C, _inst, _graph attached
    emergency_requests : dict m -> EmergencyRequest (A_m pre-computed by caller)
    travel_time : full travel-time dict including emergency nodes
    tau_max : maximum shift duration
    penalty_tw : j -> pi^TW_j
    penalty_c  : j -> pi^C_j
    outsource_cost : m -> f_m  (already stored in EmergencyRequest but kept for consistency)
    """
    raise NotImplementedError(
        "extract_phase2_instance is deprecated because it can lose ESCP baseline "
        "trip structure. Export Phase 1 with phase1_export.export_instance(), "
        "then build Phase 2 with load_phase2_solution()."
    )
