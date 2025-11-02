import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmmlearn.hmm import CategoricalHMM
import math


difficulty_states = ["Hard", "Medium", "Easy"]
grades = ["FB", "B", "S", "NS"]

start_prob = np.full(3, 1/3)

transition_matrix = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
], dtype=float)

emission_matrix = np.array([
    [0.10, 0.20, 0.40, 0.30], 
    [0.15, 0.25, 0.50, 0.10], 
    [0.20, 0.30, 0.40, 0.10]
], dtype=float)

hmm_model = CategoricalHMM(n_components=3, init_params="")
hmm_model.startprob_ = start_prob
hmm_model.transmat_ = transition_matrix
hmm_model.emissionprob_ = emission_matrix

graph = nx.DiGraph()
for s in difficulty_states:
    graph.add_node(s)

for i, s_from in enumerate(difficulty_states):
    for j, s_to in enumerate(difficulty_states):
        p = transition_matrix[i, j]
        if p > 0:
            graph.add_edge(s_from, s_to, label=f"{p:.2f}")

pos = nx.circular_layout(graph)
plt.figure(figsize=(6, 6))
nx.draw(graph, pos, with_labels=True, node_size=1800, font_size=10)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'label'))
plt.title("State Transition Diagram")
plt.show()

grade_to_idx = {g: i for i, g in enumerate(grades)}
observations = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
obs_seq = np.array([[grade_to_idx[g]] for g in observations])

log_prob = hmm_model.score(obs_seq)
prob_sequence = np.exp(log_prob)
print(f"P={prob_sequence:.6e}")

log_viterbi, decoded_states = hmm_model.decode(obs_seq, algorithm="viterbi")
prob_viterbi = np.exp(log_viterbi)
decoded_names = [difficulty_states[s] for s in decoded_states]

print("States:", decoded_names)
print(f"Viterbi path: {prob_viterbi:.6e}")

plt.figure()
plt.plot(decoded_states, "-o", label="Viterbi Path")
plt.yticks(ticks=range(3), labels=difficulty_states)
plt.xlabel("Test index")
plt.ylabel("Difficulty")
plt.legend()
plt.tight_layout()
plt.show()

#bonus
def log_safe(x):
    return -math.inf if x <= 0 else math.log(x)

T = len(obs_seq)
N = len(start_prob)

delta = np.full((T, N), -math.inf)
psi = np.full((T, N), -1, dtype=int)

first_obs = obs_seq[0, 0]
for i in range(N):
    delta[0, i] = log_safe(start_prob[i]) + log_safe(emission_matrix[i, first_obs])

for t in range(1, T):
    obs_t = obs_seq[t, 0]
    for j in range(N):
        best_prev = -1
        best_val = -math.inf
        for i in range(N):
            val = delta[t-1, i] + log_safe(transition_matrix[i, j])
            if val > best_val:
                best_val = val
                best_prev = i
        delta[t, j] = best_val + log_safe(emission_matrix[j, obs_t])
        psi[t, j] = best_prev

last_state = int(np.argmax(delta[T-1]))
log_prob = float(delta[T-1, last_state])

path = [last_state]
for t in range(T-1, 0, -1):
    path.append(int(psi[t, path[-1]]))
path.reverse()

prob_path = math.exp(log_prob)

print("Viterbi states (0=Hard, 1=Medium, 2=Easy):")
print("Path =", path)
print(f"P(Bonus Viterbi path) = {prob_path:.6e}")

