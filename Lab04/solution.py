from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from itertools import product
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. a)
model = MarkovNetwork()
model.add_nodes_from(["A1","A2","A3","A4","A5"])
model.add_edges_from([
    ("A1","A2"), ("A1","A3"),
    ("A2","A4"), ("A2","A5"),
    ("A3","A4"), ("A4","A5")
])

pos = nx.spring_layout(model, seed=42)
nx.draw(model, with_labels=True, pos=pos, node_size=1500, alpha=0.8)
plt.title("Graph of the Markov Random Field")
plt.show()

# 1. b)
def phi_linear(idxs, vars_):
    cards = [2]*len(vars_)
    vals = []
    for state in product([-1,1], repeat=len(vars_)):
        vals.append(np.exp(sum(i*s for i,s in zip(idxs,state))))
    return DiscreteFactor(variables=vars_, cardinality=cards, values=vals)

f12  = phi_linear([1,2], ["A1","A2"])
f13  = phi_linear([1,3], ["A1","A3"])
f34  = phi_linear([3,4], ["A3","A4"])
f245 = phi_linear([2,4,5], ["A2","A4","A5"])

model.add_factors(f12, f13, f34, f245)

bp = BeliefPropagation(model)
marginals = bp.map_query(variables=["A1","A2","A3","A4","A5"])

print("MAP (probabilitatea max):")
print(marginals)
