import gurobipy as gp
from gurobipy import GRB

m = gp.Model("test")
x = m.addVar(name="x")
m.setObjective(x, GRB.MINIMIZE)
m.addConstr(x >= 5)
m.optimize()
print(f"Optimal x = {x.X}")  # should print 5.0