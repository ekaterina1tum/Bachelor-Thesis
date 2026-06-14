"""
Minimal standalone example for Phase 2 (ESCP).

Constructs a Phase2Instance by hand (no Phase-1 model needed) and solves it.

Scenario
--------
Route 0 has two trips:
  Trip 0: depot(0) -> PoC 1 -> PoC 2 -> depot(0)   (first, not last)
  Trip 1: depot(0) -> PoC 3 -> depot(0)             (last)

Two emergency requests:
  Emergency 10: can be inserted on arc 1->2 (trip 0) or outsourced for cost 50
  Emergency 11: can only be inserted on arc 3->depot (trip 1), outsource cost 20
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from phase2_instance import (
    Phase2Arc, Trip, RegularRequest, EmergencyRequest, Phase2Instance
)
from phase2_model import build_phase2_model, solve_and_print


def make_example() -> Phase2Instance:
    # Travel times (symmetric for simplicity)
    nodes = [0, 1, 2, 3, 10, 11]
    raw = {
        (0, 1): 5,  (0, 2): 8,  (0, 3): 6,  (0, 10): 4,  (0, 11): 7,
        (1, 2): 4,  (1, 3): 7,  (1, 10): 3, (1, 11): 6,
        (2, 3): 5,  (2, 10): 4, (2, 11): 5,
        (3, 10): 3, (3, 11): 4,
        (10, 11): 3,
    }
    tt = {}
    for (i, j), t in raw.items():
        tt[(i, j)] = t
        tt[(j, i)] = t
    # Self loops
    for n in nodes:
        tt[(n, n)] = 0.0

    # ---- Arcs in the Phase-1 solution ----
    # Trip 0 arcs: 0->1, 1->2, 2->0
    arc0 = Phase2Arc(id=0, src=0, tgt=1, trip_id=0, route_id=0)
    arc1 = Phase2Arc(id=1, src=1, tgt=2, trip_id=0, route_id=0)
    arc2 = Phase2Arc(id=2, src=2, tgt=0, trip_id=0, route_id=0)
    # Trip 1 arcs: 0->3, 3->0
    arc3 = Phase2Arc(id=3, src=0, tgt=3, trip_id=1, route_id=0)
    arc4 = Phase2Arc(id=4, src=3, tgt=0, trip_id=1, route_id=0)

    # ---- Trips ----
    # Trip 0: C^0 = z(2) + t(2,0) = 13 + 8 = 21; g=2 (idle before trip 1)
    trip0 = Trip(id=0, route_id=0, arcs=[arc0, arc1, arc2],
                 C0=21.0, g=2.0, is_first=True, is_last=False, next_trip_id=1)
    # Trip 1: C^0 = z(3) + t(3,0) = 29 + 6 = 35
    trip1 = Trip(id=1, route_id=0, arcs=[arc3, arc4],
                 C0=35.0, g=0.0, is_first=False, is_last=True, next_trip_id=None)

    # ---- Regular requests ----
    # PoC 1: z0=5, d=15, in trip 0; B_1 = [arc0] (arc before PoC1 is arc0->1)
    rr1 = RegularRequest(id=1, trip_id=0, z0=5.0, d=15.0, pi_tw=10.0, pi_c=2.0,
                         B=[arc0])
    # PoC 2: z0=13, d=20, in trip 0; B_2 = [arc0, arc1]
    rr2 = RegularRequest(id=2, trip_id=0, z0=13.0, d=20.0, pi_tw=10.0, pi_c=2.0,
                         B=[arc0, arc1])
    # PoC 3: z0=29, d=40, in trip 1; B_3 = [arc3]
    rr3 = RegularRequest(id=3, trip_id=1, z0=29.0, d=40.0, pi_tw=10.0, pi_c=2.0,
                         B=[arc3])

    # ---- Emergency requests ----
    # Emergency 10: release=10, deadline=40, f=50; candidate arc: 1->2 (arc id=1)
    em10 = EmergencyRequest(id=10, rho=10.0, d_bar=40.0, f=50.0, A_m=[arc1])
    # Emergency 11: release=5, deadline=50, f=20; candidate arc: 3->0 (arc id=4)
    em11 = EmergencyRequest(id=11, rho=5.0, d_bar=50.0, f=20.0, A_m=[arc4])

    return Phase2Instance(
        routes={0: [0, 1]},
        trips={0: trip0, 1: trip1},
        regular_requests={1: rr1, 2: rr2, 3: rr3},
        Lambda0={0: 35.0},
        tau_max=50.0,
        emergency_requests={10: em10, 11: em11},
        travel_time=tt,
    )


if __name__ == "__main__":
    inst = make_example()

    print("Phase2Instance built:")
    print(f"  Routes: {inst.routes}")
    print(f"  Trips: {list(inst.trips.keys())}")
    print(f"  T0={inst.T0}, T_last={inst.T_last}")
    print(f"  delta values: {inst.delta}")
    print(f"  M_big={inst.M_big}")
    print()

    model = build_phase2_model(inst)
    model.update()
    print(f"Model built: {model.NumVars} variables, {model.NumConstrs} constraints")
    print()

    solve_and_print(model)
