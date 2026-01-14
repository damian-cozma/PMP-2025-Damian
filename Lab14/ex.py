import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

df = pd.read_csv("date_colesterol.csv")
t = df["Ore_Exercitii"].values
y = df["Colesterol"].values

def fit_mixture(K):
    with pm.Model() as model:

        w = pm.Dirichlet("w", a=np.ones(K))

        alpha = pm.Normal("alpha", 0, 20, shape=K)
        beta  = pm.Normal("beta", 0, 20, shape=K)
        gamma = pm.Normal("gamma", 0, 20, shape=K)

        sigma = pm.HalfNormal("sigma", 10, shape=K)

        mu = alpha[:, None] + beta[:, None] * t + gamma[:, None] * t**2   #(K, N)

        y_obs = pm.Mixture(
            "y_obs",
            w=w,
            comp_dists=[
                pm.Normal.dist(mu=mu[k], sigma=sigma[k])
                for k in range(K)
            ],
            observed=y
        )
      
        idata = pm.sample(
            1500,
            tune=1500,
            target_accept=0.9,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

    return model, idata

#1. models K = 3, 4, 5

models = {}
idatas = {}

for K in [3, 4, 5]:
    print(f"\nFitting mixture with K = {K} ...")
    model, idata = fit_mixture(K)
    models[K] = model
    idatas[K] = idata

#2. print posterior estimates

for K in [3, 4, 5]:
    print(f"\n===== POSTERIOR SUMMARY FOR K = {K} =====")
    print(az.summary(idatas[K], var_names=["alpha", "beta", "gamma", "w", "sigma"]))

#3. WAIC & LOO 

print("\nWAIC comparison")
for K in [3, 4, 5]:
    print(f"K={K}:", az.waic(idatas[K]))

print("\nLOO comparison")
for K in [3, 4, 5]:
    print(f"K={K}:", az.loo(idatas[K]))
