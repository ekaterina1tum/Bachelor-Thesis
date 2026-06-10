

from dataclasses import dataclass
from instance import Instance


@dataclass(frozen=True)
class Arc:
    """A single directed arc in the extended graph.

    Attributes
    ----------
    src : int       source node
    tgt : int       target node
    cost : float    travel time c_e on this arc
    kind : str      one of 'A0', 'AP', 'AR'
    """
    src: int
    tgt: int
    cost: float
    kind: str

    def __repr__(self):
        return f"Arc({self.src}->{self.tgt}, c={self.cost}, {self.kind})"


@dataclass
class ExtendedGraph:
    """Holds all three arc sets for an instance."""
    A0: list   # depot-adjacent arcs
    AP: list   # PoC-to-PoC arcs
    AR: list   # replenishment arcs

    @property
    def all_arcs(self) -> list:
        """A^0 U A^P U A^R, the full arc set."""
        return self.A0 + self.AP + self.AR

    def __repr__(self):
        return (
            f"ExtendedGraph(|A0|={len(self.A0)}, "
            f"|AP|={len(self.AP)}, |AR|={len(self.AR)}, "
            f"total={len(self.all_arcs)})"
        )


def build_graph(inst: Instance) -> ExtendedGraph:
    """Construct the three arc sets from an Instance."""
    depot = inst.depot
    pocs = inst.pocs

    # A^0 : depot -> PoC and PoC -> depot, for every PoC
    A0 = []
    for j in pocs:
        A0.append(Arc(src=depot, tgt=j, cost=inst.c(depot, j), kind="A0"))
        A0.append(Arc(src=j, tgt=depot, cost=inst.c(j, depot), kind="A0"))

    # A^P : every ordered pair of distinct PoCs
    AP = []
    for i in pocs:
        for j in pocs:
            if i != j:
                AP.append(Arc(src=i, tgt=j, cost=inst.c(i, j), kind="AP"))

    # A^R : replenishment arcs, one per ordered pair of distinct PoCs
    # Cost = c_{i,0} + c_{0,j}  (go to depot in between)
    AR = []
    for i in pocs:
        for j in pocs:
            if i != j:
                cost = inst.c(i, depot) + inst.c(depot, j)
                AR.append(Arc(src=i, tgt=j, cost=cost, kind="AR"))

    return ExtendedGraph(A0=A0, AP=AP, AR=AR)


if __name__ == "__main__":
    from instance import make_2trip_example

    inst = make_2trip_example()
    g = build_graph(inst)
    print(g)
    print()

    print("A^0 (depot-adjacent arcs):")
    for arc in g.A0:
        print(f"  {arc}")
    print()

    print("A^P (PoC-to-PoC arcs):")
    for arc in g.AP:
        print(f"  {arc}")
    print()

    print("A^R (replenishment arcs):")
    for arc in g.AR:
        print(f"  {arc}")