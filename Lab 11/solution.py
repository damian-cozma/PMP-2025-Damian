import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

df = pd.read_csv('Prices.csv')

# variabila target (pretul)
y  = df["Price"].values

x1 = df["Speed"].values # viteza procesor
x2 = np.log(df["HardDrive"].values) # log(hard disk) cum cere enuntul

df["Premium_bin"] = (df["Premium"] == "yes").astype(int)
x3 = df["Premium_bin"].values

#a)
with pm.Model() as model:

    # priors slabe
    alpha = pm.Normal("alpha", mu=0, sigma=1000)
    beta1 = pm.Normal("beta1", mu=0, sigma=500)
    beta2 = pm.Normal("beta2", mu=0, sigma=500)
    sigma = pm.HalfNormal("sigma", sigma=500)

    mu = alpha + beta1 * x1 + beta2 * x2

    # likelihood: pretul este normal distribuit in jurul lui mu
    price = pm.Normal("price", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(2000, tune=1500, target_accept=0.9, random_seed=42)


# b) HDI 95% pentru coeficienti
hdi_95 = az.hdi(idata, var_names=["alpha", "beta1", "beta2"], hdi_prob=0.95)
print("\n95% HDI")
print(hdi_95)

# c) verificam daca predictorii sunt utili
# basically: daca intervalele pentru beta1 si beta2 NU includ 0 -> sunt utili
# In cazul meu, atat beta1 cat si beta2 au intervale complet pozitive, deci ambele variabile sunt predictori utili ai pretului.

# d) expected price pentru Speed=33, HD=540
new_speed = 33
new_hd    = 540
new_x2    = np.log(new_hd)

posterior = idata.posterior

# calculam mu pentru fiecare sample din posterior
mu_new = (
    posterior["alpha"]
    + posterior["beta1"] * new_speed
    + posterior["beta2"] * new_x2
)

# intervalul 90% HDI pentru mu (prețul asteptat)
hdi_mu_90 = az.hdi(mu_new, hdi_prob=0.90)
print("\n90% HDI pt EXPECTED price")
print(hdi_mu_90)

# e) predictive interval (cu sigma inclus)
rng = np.random.default_rng(42)
pred_dist = (
    posterior["alpha"]
    + posterior["beta1"] * new_speed
    + posterior["beta2"] * new_x2
    + rng.normal(0, posterior["sigma"], size=posterior["sigma"].shape)
)

hdi_pred_90 = az.hdi(pred_dist, hdi_prob=0.90)
print("\n90% PREDICTIVE INTERVAL pt price")
print(hdi_pred_90)

# bonus
with pm.Model() as premium_model:

    # acum avem beta3 pentru premium
    alpha_p = pm.Normal("alpha_p", mu=0,  sigma=1000)
    beta1_p = pm.Normal("beta1_p", mu=0,  sigma=500)
    beta2_p = pm.Normal("beta2_p", mu=0,  sigma=500)
    beta3_p = pm.Normal("beta3_p", mu=0,  sigma=500)
    sigma_p = pm.HalfNormal("sigma_p", sigma=500)

    mu_p = alpha_p + beta1_p * x1 + beta2_p * x2 + beta3_p * x3

    price_p = pm.Normal("price_p", mu=mu_p, sigma=sigma_p, observed=y)

    idata_prem = pm.sample(2000, tune=1500, target_accept=0.9, random_seed=42)

# extragem HDI pt premium
hdi_premium = az.hdi(idata_prem, var_names=["beta3_p"], hdi_prob=0.95)
print("\n95% HDI pt premium")
print(hdi_premium)

print("\nDaca intervalul nu include 0 -> premium are efect real asupra pretului.")
