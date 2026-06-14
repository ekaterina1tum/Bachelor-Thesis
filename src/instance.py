
import math
from dataclasses import dataclass, field


@dataclass
class Instance:
    pocs: list
    depot: int
    travel_time: dict
    release_time: dict
    due_time: dict
    depot_deadline: float
    fleet_size: int
    max_shift: float

    @property
    def n(self) -> int:
        return len(self.pocs)

    @property
    def nodes(self) -> list:
        return [self.depot] + self.pocs

    def c(self, i, j) -> float:
        return self.travel_time[(i, j)]

    def __repr__(self):
        return (
            f"Instance(n={self.n} PoCs, K={self.fleet_size} vehicles, "
            f"tau_max={self.max_shift}, d_0={self.depot_deadline})"
        )


def make_2trip_example() -> Instance:
    depot = 0
    pocs = [1, 2, 3, 4]

    raw = {
        (0, 1): 2, (0, 2): 3, (0, 3): 3, (0, 4): 2,
        (1, 2): 3, (1, 3): 5, (1, 4): 4,
        (2, 3): 4, (2, 4): 5,
        (3, 4): 3,
    }
    travel_time = {}
    for (i, j), t in raw.items():
        travel_time[(i, j)] = t
        travel_time[(j, i)] = t

    release_time = {1: 0, 2: 0, 3: 0, 4: 0}
    due_time = {1: 30, 2: 30, 3: 30, 4: 30}

    return Instance(
        pocs=pocs,
        depot=depot,
        travel_time=travel_time,
        release_time=release_time,
        due_time=due_time,
        depot_deadline=30,
        fleet_size=1,
        max_shift=20,
    )


def load_instance(
    path: str,
    max_shift: float = 480.0,
    round_distances: bool = False,
) -> Instance:
    """Load an MSCDP-format instance file.

    File format
    -----------
    Line 1:  ``c <num_nodes> <max_vehicles>``
    Lines 2..: for each node ``x  y  tw_start  tw_end``
               The first node is the depot.

    Travel times are Euclidean distances between node coordinates.
    The depot's time-window end is used as the depot deadline.
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    header = lines[0].split()
    # header[0] == 'c'
    num_nodes = int(header[1])
    max_vehicles = int(header[2])

    coords: dict[int, tuple[float, float]] = {}
    tw_start: dict[int, float] = {}
    tw_end: dict[int, float] = {}

    node_lines = lines[1:1 + num_nodes]
    for idx, ln in enumerate(node_lines):
        x, y, ts, te = ln.split()
        coords[idx] = (float(x), float(y))
        tw_start[idx] = float(ts)
        tw_end[idx] = float(te)

    depot = 0
    pocs = [i for i in coords if i != depot]

    def dist(i: int, j: int) -> float:
        (xi, yi), (xj, yj) = coords[i], coords[j]
        d = math.hypot(xi - xj, yi - yj)
        return round(d) if round_distances else d

    travel_time: dict[tuple[int, int], float] = {}
    for i in coords:
        for j in coords:
            travel_time[(i, j)] = dist(i, j)

    release_time = {j: tw_start[j] for j in pocs}
    due_time = {j: tw_end[j] for j in pocs}

    return Instance(
        pocs=pocs,
        depot=depot,
        travel_time=travel_time,
        release_time=release_time,
        due_time=due_time,
        depot_deadline=tw_end[depot],
        fleet_size=max_vehicles,
        max_shift=max_shift,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        inst = load_instance(sys.argv[1])
        print(f"Loaded: {inst}")
        print(f"  Nodes: {inst.nodes[:6]}{'...' if inst.n > 5 else ''}")
        print(f"  Depot deadline: {inst.depot_deadline}")
        print(f"  c(0,1) = {inst.c(0, 1):.3f}")
        print(f"  TW of PoC 1: [{inst.release_time[1]}, {inst.due_time[1]}]")
        sys.exit(0)

    inst = make_2trip_example()
    print(inst)
    print(f"Nodes: {inst.nodes}")
    print(f"c(A, B) = c(1, 2) = {inst.c(1, 2)}")
    print(f"c(B, depot) = c(2, 0) = {inst.c(2, 0)}")
    print(f"Time window of B: [{inst.release_time[2]}, {inst.due_time[2]}]")