import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import poisson

print("------------ Ex. 1 ------------")

# 1.a
def estimated_experiment(samples):
    red = 0
    prime = [2, 3, 5]

    for _ in range(samples):
        urn = ['red', 'red', 'red',
               'blue', 'blue', 'blue', 'blue',
               'black', 'black']

        roll = random.randint(1, 7)

        if roll in prime:
            urn.append('black')
        elif roll == 6:
            urn.append('red')
        else:
            urn.append('blue')

        ball = random.choice(urn)

        if ball == 'red':
            red += 1

    return red / samples

# 1.b
estimated = estimated_experiment(100000)
print(f"Experiment simulation: {estimated:.4f}")

# 1.c
'''
Probabilitatea Totala pt P(rosu)
P(rosu) = P(prim) * P(rosu | prim) + P(6) * P(rosu | 6) + P(altcv) * P(rosu | altcv)
P(rosu) = 3/6 * 3/10 + 1/6 * 4/10 + 2/6 + 3/10 
'''
theoretical = (3 / 6 * 3 / 10) + (1 / 6 * 4 / 10) + (2 / 6 * 3 / 10)
print("\n")
print(f"Theoretical: {theoretical:.4f}")
print(f"Estimated: {estimated:.4f}")
print(f"Comparison: {theoretical - estimated:.4f} (-)")

print("------------ Ex. 2 ------------")

# 2.1
samples=100
lambda_v = [1,2,5,10]
poisson_fixed = {}

for v in lambda_v:
    key = f'{v}'
    value = poisson.rvs(v, size=samples)
    poisson_fixed[key] = value

# 2.2
random_lambdas = np.random.choice(lambda_v, size=samples)

poisson_randomized = []
for lam in random_lambdas:
    values = poisson.rvs(lam)
    poisson_randomized.append(values)

poisson_randomized = np.array(poisson_randomized)

# 2.a

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(poisson_fixed['1'], bins=10, alpha=0.7)
plt.title('Lambda = 1')

plt.subplot(2, 3, 2)
plt.hist(poisson_fixed['2'], bins=10, alpha=0.7)
plt.title('Lambda = 2')

plt.subplot(2, 3, 3)
plt.hist(poisson_fixed['5'], bins=10, alpha=0.7)
plt.title('Lambda = 5')

plt.subplot(2, 3, 4)
plt.hist(poisson_fixed['10'], bins=10, alpha=0.7)
plt.title('Lambda = 10')

plt.subplot(2, 3, 5)
plt.hist(poisson_randomized, bins=10, alpha=0.7)
plt.title('Poisson Randomizat')

plt.tight_layout()
plt.show()

# 2.b

"""
Distributia randomizata are mai multe varfuri (sau multimodala) spre deosebire de distributiile 
fixe care au un sg varf.

Asta arata ca incertitudinea parametrilor creste variabilitatea si complexitatea distributiei. 
In procesele reale, unde parametrii se schimba des, modelele cu parametri fixi subestimeaza variatia 
reala si nu capteaza intreaga complexitate.
"""

# 2.c Bonus
probabilities = [0.1, 0.2, 0.5, 0.2]

random_lambdas_unequal = np.random.choice(lambda_v, size=samples, p=probabilities)
poisson_unequal = []
for lam in random_lambdas_unequal:
    value = poisson.rvs(lam)
    poisson_unequal.append(value)

poisson_unequal = np.array(poisson_unequal)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(poisson_randomized, bins=10, alpha=0.7)
plt.hist(poisson_unequal, bins=10, alpha=0.7)
plt.title('Comparatie Probabilitati')

plt.subplot(1, 2, 2)
plt.hist(random_lambdas, bins=4, alpha=0.7)
plt.hist(random_lambdas_unequal, bins=4, alpha=0.7)
plt.title('Distributia Lambda')

plt.tight_layout()
plt.show()

'''
Cu probabilitati inegale, distributia se concentreaza in jurul lui 5. Forma devine MAI 
apropiata de Poisson(5)
'''
