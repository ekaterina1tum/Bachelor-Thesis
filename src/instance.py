
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


if __name__ == "__main__":
    inst = make_2trip_example()
    print(inst)
    print(f"Nodes: {inst.nodes}")
    print(f"c(A, B) = c(1, 2) = {inst.c(1, 2)}")
    print(f"c(B, depot) = c(2, 0) = {inst.c(2, 0)}")
    print(f"Time window of B: [{inst.release_time[2]}, {inst.due_time[2]}]")