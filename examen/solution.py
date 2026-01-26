import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

az.style.use("arviz-darkgrid")

CSV_PATH = "bike_daily.csv"
df = pd.read_csv(CSV_PATH)

print(df.head())
print("\nmissing values:\n", df.isna().sum())
print("\nseason counts:\n", df["season"].value_counts())

#1
sns.pairplot(df, vars=["rentals", "temp_c", "humidity", "wind_kph"], diag_kind="kde")
plt.show()
plt.show()
sns.boxplot(data=df, x="is_holiday", y="rentals")
plt.show()

#2a
# am standardizat variabilele continue si targetul rentals

df_oh = pd.get_dummies(df, columns=["season"], drop_first=True)
y_raw = df_oh["rentals"].to_numpy()

cont_cols = ["temp_c", "humidity", "wind_kph"]
bin_cols = ["is_holiday"]
season_cols = [c for c in df_oh.columns if c.startswith("season_")]

X_cont = df_oh[cont_cols].to_numpy(dtype=float)
X_bin = df_oh[bin_cols + season_cols].to_numpy(dtype=float)

scX = StandardScaler()
scY = StandardScaler()

X_cont_s = scX.fit_transform(X_cont)
y_s = scY.fit_transform(y_raw.reshape(-1, 1)).ravel()

temp_s = X_cont_s[:,0]
hum_s = X_cont_s[:,1]
wind_s = X_cont_s[:,2]

#2b
# am construit un model de regresie liniara Bayesiana, cu rentals ca variabila raspuns si predictori:
# temp,humidity,wind,holiday si season
X_lin = np.column_stack([temp_s, hum_s, wind_s, X_bin])
feature_names_lin = ["temp_s", "hum_s", "wind_s"] + bin_cols + season_cols
coords_lin = {"feature": feature_names_lin}

with pm.Model(coords=coords_lin) as model_lin:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta",  mu=0, sigma=1, dims="feature")
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha+pm.math.dot(X_lin, beta)

    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

    idata_lin = pm.sample(
        draws=1500, tune=1500, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

print(az.summary(idata_lin, var_names=["alpha", "beta", "sigma"], hdi_prob=0.90))

#2c
# am extins modelul anterior prin adaugarea unui termen patrat pentru temp (temp_c^2),
# pt a captura posibila relatie neliniara observata in EDA. Structura modelului ramane aceeasi,
# diferenta fiind termenul suplimentar.
temp2_s = temp_s**2
X_poly = np.column_stack([temp_s, temp2_s, hum_s, wind_s, X_bin])

feature_names_poly = ["temp_s", "temp2_s", "hum_s", "wind_s"] + bin_cols + season_cols
coords_poly = {"feature": feature_names_poly}

with pm.Model(coords=coords_poly) as model_poly:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta",  mu=0, sigma=1, dims="feature")
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + pm.math.dot(X_poly, beta)

    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

    idata_poly = pm.sample(
        draws=1500, tune=1500, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

print("\nPoly")
print(az.summary(idata_poly, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

#3
# am rulat MCMC cu mai multe lanturi si un numar suficient de iteratii, folosind target_accept ridicat. post sunt
# bine comportate, fara probleme de convergenta. Din coeficienti, temperatura are cel mai mare efect asupra rentals,
# atat in modelul liniar, cat si in cel polinomial.
az.plot_trace(idata_lin, var_names=["alpha", "beta", "sigma"])
plt.show()
az.plot_trace(idata_poly, var_names=["alpha", "beta", "sigma"])
plt.show()

def ranked_betas(idata, feature_names):
    b = idata.posterior["beta"].stack(sample=("chain","draw")).values  # (feature, sample)
    means = b.mean(axis=1)
    order = np.argsort(-np.abs(means))
    return [(feature_names[i], means[i]) for i in order]

print("\nTop beta (linear):", ranked_betas(idata_lin, feature_names_lin)[:6])
print("Top beta (poly):  ", ranked_betas(idata_poly, feature_names_poly)[:7])

#4a
# am comparat modelele folosind LOO (si WAIC), deoarece acestea masoara performanta predictiva out-of-sample. Modelul
# polinomial are scor mai bun, sugerand ca termenul neliniar in temperatura imbunatateste predictia.
print("\nWAIC:")
print("linear:", az.waic(idata_lin, scale="deviance"))
print("poly: ", az.waic(idata_poly, scale="deviance"))

print("\nLOO:")
print("linear:", az.loo(idata_lin, scale="deviance"))
print("poly: ", az.loo(idata_poly, scale="deviance"))

cmp = az.compare({"linear": idata_lin, "poly": idata_poly}, ic="loo", scale="deviance")
print("\nLOO:\n", cmp)

#4b
# am realizat PPC pentru a verifica daca modelul poate reproduce datele observate.
# am vizualizat media prezisa si intervalele de incertitudine in functie de temperatura.
# modelul polinomial urmareste mai bine trendul observat decat modelul liniar.
def ppc_curve_vs_temp(idata, is_poly: bool):
    t_grid = np.linspace(df["temp_c"].min(), df["temp_c"].max(), 120)

    t_s = (t_grid - scX.mean_[0]) / scX.scale_[0]

    hum0 = 0.0
    wind0 = 0.0

    xb = np.zeros(X_bin.shape[1])

    if not is_poly:
        Xg = np.column_stack([t_s, np.full_like(t_s, hum0), np.full_like(t_s, wind0),
                              np.tile(xb, (len(t_s), 1))])
    else:
        Xg = np.column_stack([t_s, t_s**2, np.full_like(t_s, hum0), np.full_like(t_s, wind0),
                              np.tile(xb, (len(t_s), 1))])

    post = idata.posterior
    beta = post["beta"].stack(sample=("chain","draw")).values
    alpha = post["alpha"].stack(sample=("chain","draw")).values
    sigma = post["sigma"].stack(sample=("chain","draw")).values

    mu_samps = alpha[None, :] + Xg @ beta
    mu_mean = mu_samps.mean(axis=1)
    mu_hdi = az.hdi(mu_samps.T, hdi_prob=0.90)

    rng = np.random.default_rng(42)
    ypred_samps = mu_samps + rng.normal(0, sigma[None, :], size=mu_samps.shape)
    y_hdi = az.hdi(ypred_samps.T, hdi_prob=0.90)

    mu_mean_raw = scY.inverse_transform(mu_mean.reshape(-1,1)).ravel()
    mu_low_raw = scY.inverse_transform(mu_hdi[0].reshape(-1,1)).ravel()
    mu_high_raw = scY.inverse_transform(mu_hdi[1].reshape(-1,1)).ravel()

    y_low_raw = scY.inverse_transform(y_hdi[0].reshape(-1,1)).ravel()
    y_high_raw = scY.inverse_transform(y_hdi[1].reshape(-1,1)).ravel()

    return t_grid, mu_mean_raw, (mu_low_raw, mu_high_raw), (y_low_raw, y_high_raw)

t, mu_m, mu_int, pred_int = ppc_curve_vs_temp(idata_poly, is_poly=True)

plt.figure(figsize=(9,5))
plt.scatter(df["temp_c"], df["rentals"], s=12, alpha=0.35, label="data")

plt.plot(t, mu_m, color="black", label="E[rentals | temp]")
plt.fill_between(t, mu_int[0], mu_int[1], color="gray", alpha=0.35, label="90% HDI")
plt.fill_between(t, pred_int[0], pred_int[1], color="C0", alpha=0.15, label="90% PI")

plt.xlabel("temp_c")
plt.ylabel("rentals")
plt.legend()
plt.title("PPC curve vs temp (poly model)")
plt.show()

#5
# am definit variabila binara is_high_demand, unde 1 inseamna ca rentals este peste 75% din datele originale.
# variabila reprezinta zilele cu cerere ridicata.
Q = np.quantile(y_raw, 0.75)
df_oh["is_high_demand"] = (df_oh["rentals"] >= Q).astype(int)
y_bin = df_oh["is_high_demand"].to_numpy()

print("\nQ= ", Q)
print("high-demand =", y_bin.mean())

#6
# am construit un model de regresie logistica Bayesiana folosind aceiasi predictori standardizati.
# am pastrat termenul patrat de temperatura deoarece imbunatateste separarea probabilitatilor pentru zilele cu cerere mare
X_log = X_poly
feature_names_log = feature_names_poly
coords_log = {"feature": feature_names_log}

with pm.Model(coords=coords_log) as model_log:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta",  mu=0, sigma=1, dims="feature")

    eta = alpha + pm.math.dot(X_log, beta)
    p = pm.Deterministic("p", pm.math.sigmoid(eta))

    pm.Bernoulli("y_obs", p=p, observed=y_bin)

    idata_log = pm.sample(
        draws=1500, tune=1500, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

print("\nLogistic")
print(az.summary(idata_log, var_names=["alpha","beta"], hdi_prob=0.95))

#7
# am calculat intervale HDI de 95% pentru coeficientii modelului logistic. temperatura ramane variabila cu
# cel mai mare impact asupra probabilitatii de high demand, urmata de sezon. umiditatea si vantul au efecte mai mici
print("\n95% HDI beta (logistic):")
print(az.hdi(idata_log, var_names=["beta"], hdi_prob=0.95))

print("\nTop betas (logistic):", ranked_betas(idata_log, feature_names_log)[:7])
