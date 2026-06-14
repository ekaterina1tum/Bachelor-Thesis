from __future__ import annotations

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


def extract_phase2_instance(
    phase1_model,
    emergency_requests: dict[int, EmergencyRequest],
    travel_time: dict[tuple[int, int], float],
    tau_max: float,
    penalty_tw: dict[int, float],
    penalty_c: dict[int, float],
    outsource_cost: dict[int, float],
) -> Phase2Instance:
    """Build a Phase2Instance from a solved Phase-1 Gurobi model.

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
    import gurobipy as gp

    inst = phase1_model._inst
    graph = phase1_model._graph
    x = phase1_model._x
    z_var = phase1_model._z
    C_var = phase1_model._C

    # ---- Reconstruct routes as ordered arc sequences ----
    used_arcs = [e for e in graph.all_arcs if x[e].X > 0.5]

    # Build adjacency: src -> arc (for used arcs only)
    out_arc: dict[int, object] = {}
    for e in used_arcs:
        out_arc[e.src] = e

    depot = inst.depot

    # Follow chains from depot to reconstruct all route arcs in order
    # Each depot-out arc starts a new trip or route
    depot_out_arcs = [e for e in used_arcs if e.src == depot]

    trip_id_counter = 0
    arc_id_counter = 0
    route_id_counter = 0

    routes: dict[int, list[int]] = {}
    trips: dict[int, Trip] = {}
    regular_requests: dict[int, RegularRequest] = {}
    Lambda0: dict[int, float] = {}

    # Baseline z values
    z_val = {j: z_var[j].X for j in inst.pocs}

    # We reconstruct by following the chain from depot
    # Replenishment arcs (AR kind) mark trip boundaries
    visited_depot_starts: set = set()

    def follow_route(start_arc) -> list:
        """Return list of arcs from depot back to depot (one full route)."""
        chain = []
        cur = start_arc
        while True:
            chain.append(cur)
            if cur.tgt == depot:
                break
            cur = out_arc.get(cur.tgt)
            if cur is None:
                break
        return chain

    for start in depot_out_arcs:
        if id(start) in visited_depot_starts:
            continue
        visited_depot_starts.add(id(start))

        route_arc_chain = follow_route(start)
        route_id = route_id_counter
        route_id_counter += 1

        # Split chain into trips at replenishment arcs (AR kind)
        # An AR arc means: PoC i -> depot -> PoC j (already merged into one arc in Phase 1)
        # Trip = consecutive arcs between depot visits
        # In the extended graph: depot->PoC (A0), PoC->PoC (AP), PoC->depot (A0), repeat
        # A replenishment arc (AR) represents a mid-route depot visit between two PoCs
        # Trips are separated by: A0 arc ending at depot OR AR arc

        current_trip_arcs_raw: list = []
        trip_ids_in_route: list[int] = []
        trip_start_time = 0.0  # depot departs at 0

        for arc in route_arc_chain:
            current_trip_arcs_raw.append(arc)
            # End of trip when we return to depot (A0 arc back to depot) or hit AR arc
            if arc.tgt == depot or arc.kind == "AR":
                # Build Phase2Arc list for this trip
                p2_arcs = []
                for a in current_trip_arcs_raw:
                    p2arc = Phase2Arc(
                        id=arc_id_counter,
                        src=a.src,
                        tgt=a.tgt,
                        trip_id=trip_id_counter,
                        route_id=route_id,
                    )
                    arc_id_counter += 1
                    p2_arcs.append(p2arc)

                # Baseline return time: travel time of last arc to depot
                # = z of last PoC before depot + travel to depot
                last_poc = current_trip_arcs_raw[-1].src if arc.tgt == depot else arc.src
                if last_poc == depot:
                    # empty trip (shouldn't happen)
                    C0_T = 0.0
                else:
                    C0_T = z_val[last_poc] + travel_time.get((last_poc, depot), inst.c(last_poc, depot))

                is_first = (len(trip_ids_in_route) == 0)
                trip = Trip(
                    id=trip_id_counter,
                    route_id=route_id,
                    arcs=p2_arcs,
                    C0=C0_T,
                    g=0.0,  # filled in below
                    is_first=is_first,
                    is_last=False,  # filled in below
                    next_trip_id=None,
                )
                trips[trip_id_counter] = trip
                trip_ids_in_route.append(trip_id_counter)
                trip_id_counter += 1
                current_trip_arcs_raw = []

        routes[route_id] = trip_ids_in_route

        # Mark first/last and set next_trip_id and g_T
        for i, tid in enumerate(trip_ids_in_route):
            trip = trips[tid]
            is_last = (i == len(trip_ids_in_route) - 1)
            # g_T: idle time between C^0_T and departure of T+
            # For simplicity, g_T = 0 unless you have explicit schedule data
            # (Phase 1 doesn't expose this directly; set 0 as conservative default)
            g = 0.0
            next_id = trip_ids_in_route[i + 1] if not is_last else None
            trips[tid] = Trip(
                id=trip.id,
                route_id=trip.route_id,
                arcs=trip.arcs,
                C0=trip.C0,
                g=g,
                is_first=trip.is_first,
                is_last=is_last,
                next_trip_id=next_id,
            )

        # Route duration = C^0 of last trip
        last_tid = trip_ids_in_route[-1]
        Lambda0[route_id] = trips[last_tid].C0

    # ---- Build arc lookup: PoC -> trip_id + position ----
    # For each PoC j, find its trip and the arcs before it
    poc_trip: dict[int, int] = {}        # j -> trip_id
    poc_arcs_before: dict[int, list[Phase2Arc]] = {}  # j -> B_j

    for trip in trips.values():
        visited_pocs = []
        for p2arc in trip.arcs:
            # If source is a PoC (not depot), it was just visited
            if p2arc.src != inst.depot and p2arc.src in inst.pocs:
                if p2arc.src not in poc_trip:
                    poc_trip[p2arc.src] = trip.id
                    # B_j = arcs traversed before j = arcs up to and including the incoming arc to j
                    # i.e. arcs where tgt comes before j in the sequence
                    poc_arcs_before[p2arc.src] = list(visited_pocs)
            visited_pocs.append(p2arc)
            if p2arc.tgt != inst.depot and p2arc.tgt in inst.pocs:
                if p2arc.tgt not in poc_trip:
                    poc_trip[p2arc.tgt] = trip.id
                    poc_arcs_before[p2arc.tgt] = list(visited_pocs)

    # ---- Build RegularRequest objects ----
    for j in inst.pocs:
        tid = poc_trip.get(j, list(trips.keys())[0])
        regular_requests[j] = RegularRequest(
            id=j,
            trip_id=tid,
            z0=z_val[j],
            d=inst.due_time[j],
            pi_tw=penalty_tw.get(j, 1.0),
            pi_c=penalty_c.get(j, 1.0),
            B=poc_arcs_before.get(j, []),
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
