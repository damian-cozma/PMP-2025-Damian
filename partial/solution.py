import networkx as nx
import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

def build_model():
    model = DiscreteBayesianNetwork([
        ('O', 'H'),
        ('O', 'W'),
        ('H', 'R'),
        ('W', 'R'),
        ('H', 'E'),
        ('R', 'C')
    ])

    cpd_O = TabularCPD(
        variable='O', variable_card=2,
        values=[[0.3], [0.7]],
        state_names={'O': ['cold', 'mild']}
    )

    cpd_H = TabularCPD(
        variable='H', variable_card=2,
        values=[[0.9, 0.2], [0.1, 0.8]],
        evidence=['O'], evidence_card=[2],
        state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']}
    )

    cpd_W = TabularCPD(
        variable='W', variable_card=2,
        values=[[0.1, 0.6], [0.9, 0.4]],
        evidence=['O'], evidence_card=[2],
        state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']}
    )

    cpd_R = TabularCPD(
        variable='R', variable_card=2,
        values=[
            [0.6, 0.9, 0.3, 0.5],
            [0.4, 0.1, 0.7, 0.5]
        ],
        evidence = ['H', 'W'], evidence_card = [2, 2],
        state_names = {'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']}
    )

    cpd_E = TabularCPD(
        variable='E', variable_card=2,
        values=[[0.8, 0.2], [0.2, 0.8]],
        evidence=['H'], evidence_card=[2],
        state_names={'E': ['high', 'low'], 'H': ['yes', 'no']}
    )

    cpd_C = TabularCPD(
        variable='C', variable_card=2,
        values=[[0.85, 0.40], [0.15, 0.60]],
        evidence=['R'], evidence_card=[2],
        state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']}
    )

    model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)
    model.check_model()

    return model

def draw_graph(model):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    pos={
        'O': (0, 1),
        'H': (-1, 0),
        'W': (1, 0),
        'R': (0, -1),
        'C': (-1, -2),
        'E': (1, -2)
    }

    plt.figure(figsize=(6,5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size = 2000,
        node_color = 'lightblue',
        font_size=12,
        arrowsize=20
    )
    plt.title("Bayesian Network")
    plt.show()

def perform_inference(model):
    infer = VariableElimination(model)

    p_h = infer.query(
        variables=['H'],
        evidence={'C': 'comfortable'}
    )

    p_e = infer.query(
        variables=['E'],
        evidence={'C': 'comfortable'}
    )

    map_hw = infer.map_query(
        variables=['H', 'W'],
        evidence={'C': 'comfortable'}
    )

    return p_h, p_e, map_hw

'''
c)

W indep de E daca stim H, deoarece drumul W->R<-H->E este blocat de colider R
O este indep de C daca stim R, deoarece toate caile dintre O si C trec prin R si sunt blocate cand R este observat
O si E nu sunt indep fara nicio conditie, deoarece exista calea O->H->E care este deschisa
O devine indep de E daca stim H, deoarece observarea lui H blocheaza calea O->H->E
H si W nu sunt indep fara conditie deoarece sunt legate prin parintele comun O
H si W devin indep daca stim O, deoarece observarea lui O blocheaza furca H<-O->W

'''

#---Punctul 2

def build_hmm():
    start_prob = np.array([0.4, 0.3, 0.3])
    trans_mat = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.2, 0.5]
    ])
    emiss_mat = np.array([
        [0.1, 0.7, 0.2],
        [0.05, 0.25, 0.7],
        [0.8, 0.15, 0.05]
    ])

    model = hmm.CategoricalHMM(n_components=3, init_params="")
    model.startprob_ = start_prob
    model.transmat_ = trans_mat
    model.emissionprob_ = emiss_mat

    return model

def forward_probability(model):
    obs = np.array([[1], [2], [0]])
    return np.exp(model.score(obs))

def viterbi_path(model):
    obs = np.array([[1], [2], [0]])
    logp, states = model.decode(obs, algorithm = 'viterbi')
    mapping = {0: 'W', 1:'R', 2:'S'}
    decoded = [mapping[s] for s in states]
    return decoded, np.exp(logp)
'''
Daca secventa de observatii era mai lunga, Viterbi este preferat deoarece are complexitatea polinomiala O(T* N^2) unde T = lungimea secventei
Brute force ar trebui sa testeze toate combinatiile posibile de stari, ceea ce este N^T (exponential) si devine imposibil de calculat pentru secvente mai mari
'''

def empirical_probability(model, N=10000):
    target = ['M', 'H', 'L']
    rev = {0:'L', 1:'M', 2:'H'}
    count = 0

if __name__ == '__main__':
    model = build_model()
    draw_graph(model)
    p_h, p_e, map_hw = perform_inference(model)
    print("\n1:")
    print("H | C:", p_h)
    print("E | C:", p_e)
    print("MAP(H,W) | C:", map_hw)

    hmm_model = build_hmm()

    print("\n2:")
    print("2.b)", forward_probability(hmm_model))

    path, prob = viterbi_path(hmm_model)
    print("2.c)", path, "prob =", prob)

    print("2.d)", empirical_probability(hmm_model))
