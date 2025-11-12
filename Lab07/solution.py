import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
x_bar = data.mean() # in jur de 58
print(f"Sample mean = {x_bar:.2f}, Sample std = {data.std(ddof=1):.2f}")

#a)

with pm.Model() as weak_model:
    mu = pm.Normal("mu", mu=x_bar, sigma=10) # x, x^2
    sigma = pm.HalfNormal("sigma", sigma=10) # half 10
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    #b)
    trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Weak Prior):")
print(summary_weak)

#c)
print("\nFrequentist estimates:")
print(f"Mean: {np.mean(data):.2f}")
print(f"SD:   {np.std(data, ddof=1):.2f}")

'''
Estimarea bayesiana pentru mu = 58.1 si sigma = 1.8 este aproape identica cu cea initiala 58, 2
Asta se intampla pentru ca priorurile alese sunt slabe, nu influenteaza aproape deloc rezultatul final
'''

#d)

with pm.Model() as strong_model:
    mu = pm.Normal("mu", mu=50, sigma=1) # 50, 1^2, prior foarte puternic
    sigma = pm.HalfNormal("sigma", sigma=10) # half 10, prior la fel de slab
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
    summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Strong Prior):")
print(summary_strong)

'''
Priorul puternic pentru mu trage media estimata in jos
Am observat ca (,) cu cat priorul e mai sigur cu atat influenteaza mai tare rezultatul

Deviatia standard tinde sa creasca usor pentru ca modelul vede ca datele
nu se potrivesc perfect cu media impusa. Ca sa compenseze diferenta fata de date, creste sigma,
devenind mai permisiv cu valorile observate

Cand priorul e puternic si gresit fata de date, mu este impins in directia priorului
iar sigma creste ca sa acopere diferenta dintre model si realitate
'''

az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Posterior with Weak Prior", fontsize=14)
plt.show()

az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Posterior with Strong Prior", fontsize=14)
plt.show()
