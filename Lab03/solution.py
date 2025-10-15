import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

print("------------ Ex. 1 ------------")

model = DiscreteBayesianNetwork([
    ('S', 'O'),
    ('S', 'L'),
    ('S', 'M'),
    ('L', 'M')
])

# tabele de prob
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3], [0.1, 0.7]],
                   evidence=['S'], evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2], [0.3, 0.8]],
                   evidence=['S'], evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1], [0.2, 0.6, 0.5, 0.9]],
                   evidence=['S', 'L'], evidence_card=[2, 2])

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

print("Model valid:", model.check_model())

# 1.a 
independencies = model.get_independencies()
print(independencies)

# 1.b 

infer = VariableElimination(model)

# teste
test_cases = [
    {'O': 1, 'L': 1, 'M': 1},  # offer, linkuri, lung
    {'O': 0, 'L': 0, 'M': 0},  # !offer, !linkuri, scurt
    {'O': 1, 'L': 0, 'M': 0}   # offer, !linkuri, scurt
]

for i, evidence in enumerate(test_cases, 1):
    result = infer.query(variables=['S'], evidence=evidence)
    spam_prob = result.values[1]
    classification = "SPAM" if spam_prob > 0.5 else "NON-SPAM"
    print(f"Test {i}: P(Spam) = {spam_prob:.4f} - {classification}")

plt.figure(figsize=(8, 6))
pos = {'S': (0, 0), 'O': (-1, -1), 'L': (1, -1), 'M': (0, -2)}
nx.draw(model, pos=pos, with_labels=True, node_size=2000,
        node_color='lightblue', arrowsize=20, font_size=12)
plt.title("Spam Detection Network")
plt.show()
