import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

df = pd.read_csv('date_promovare_examen.csv')

x1 = df["Ore_Studiu"].values
x2 = df["Ore_Somn"].values
y = df["Promovare"].values

# a)
# determinare balansare
print(df["Promovare"].value_counts())

with pm.Model() as model:
    # priors slabe
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta1 = pm.Normal("beta1", mu=0, sigma=5)
    beta2 = pm.Normal("beta2", mu=0, sigma=5)

    logit_p = alpha + beta1 * x1 + beta2 * x2

    # trecem prin logistic (sigmoid)
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

    # likelihood bernoulli
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    # sampling
    idata = pm.sample(2000, tune=2000, target_accept=0.98)

# Datele sunt perfect balansate: 250 promotori, 250 respinsi
# Am rulat sampling-ul cu NUTS (2000 sample-uri, 2000 tuning), 0 divergente.

# b)
posterior = idata.posterior

alpha_s = posterior["alpha"].values.flatten()
beta1_s = posterior["beta1"].values.flatten()
beta2_s = posterior["beta2"].values.flatten()

somn_med = df["Ore_Somn"].mean()

decision_boundary = (-alpha_s - beta2_s * somn_med) / beta1_s

print("\nMedia granitei de decizie (x1 necesar pt p=0.5):")
print(decision_boundary.mean())

# Media granitei de decizie este aprox. 3.82 ore de studiu, daca tinem somnul la valoarea medie.
# Coeficientii au HDI-uri strict pozitive, deci modelul separa bine cele doua clase

# c)

print("\nHDI coeficienti:")
print(az.hdi(idata, var_names=["beta1", "beta2"], hdi_prob=0.95))

print("\nMedia beta1=", beta1_s.mean())
print("Media beta2=", beta2_s.mean())

# Comparand coeficientii, beta2 (somnul) are valoare medie putin mai mare decat beta1 (studiul)
# Deci somnul influenteaza usor mai mult promovabilitatea, dar ambele variabile conteaza
