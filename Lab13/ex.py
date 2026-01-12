import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

N = 50
x = np.linspace(-3, 3, N)

true_coeffs = [1.5, -2.0, 0.5, 1.0, -0.3, 0.1]

y_true = sum(true_coeffs[i] * x ** i for i in range(len(true_coeffs)))
y = y_true + np.random.normal(0, 2, size=N)

plt.scatter(x, y)
plt.title("Generated data (order 5)")
plt.show()


def poly_design_matrix(x, order):
    return np.vstack([x ** i for i in range(order + 1)]).T

#a)

order = 5
X_poly = poly_design_matrix(x, order)

with pm.Model() as model_sd10:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order + 1)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = pm.math.dot(X_poly, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    idata_sd10 = pm.sample(2000, tune=2000, target_accept=0.9)

beta_mean10 = idata_sd10.posterior["beta"].mean(dim=("chain", "draw")).values
y_pred10 = X_poly @ beta_mean10

plt.scatter(x, y)
plt.plot(x, y_pred10, color="red")
plt.title("Posterior fit – order 5, sd=10")
plt.show()

#b)

with pm.Model() as model_sd100:
    beta = pm.Normal("beta", mu=0, sigma=100, shape=order + 1)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = pm.math.dot(X_poly, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    idata_sd100 = pm.sample(2000, tune=2000, target_accept=0.9)

beta_mean100 = idata_sd100.posterior["beta"].mean(dim=("chain", "draw")).values
y_pred100 = X_poly @ beta_mean100

plt.scatter(x, y)
plt.plot(x, y_pred100, color="orange")
plt.title("Posterior fit – order 5, sd=100 (overfitting)")
plt.show()

#repetam cu 500 data points

N = 500
x_500 = np.linspace(-3, 3, N)
y_true_500 = sum(true_coeffs[i] * x_500 ** i for i in range(len(true_coeffs)))
y_500 = y_true_500 + np.random.normal(0, 2, size=N)

X_poly_500 = poly_design_matrix(x_500, order)

with pm.Model() as model_500:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order + 1)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = pm.math.dot(X_poly_500, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_500)

    idata_500 = pm.sample(2000, tune=2000)

beta_mean500 = idata_500.posterior["beta"].mean(dim=("chain", "draw")).values
y_pred500 = X_poly_500 @ beta_mean500

plt.scatter(x_500, y_500, s=5)
plt.plot(x_500, y_pred500, color="green")
plt.title("Posterior fit – 500 points")
plt.show()

#------------------

def fit_poly_model(x, y, order):
    X_poly = poly_design_matrix(x, order)
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=order+1)
        sigma = pm.HalfNormal("sigma", sigma=5)

        mu = pm.math.dot(X_poly, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            2000,
            tune=2000,
            target_accept=0.9,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return idata


idata_lin = fit_poly_model(x, y, order=1)
idata_quad = fit_poly_model(x, y, order=2)
idata_cubic = fit_poly_model(x, y, order=3)

print("\nWAIC scores:")
print("Linear:", az.waic(idata_lin))
print("Quadratic:", az.waic(idata_quad))
print("Cubic:", az.waic(idata_cubic))

print("\nLOO scores:")
print("Linear:", az.loo(idata_lin))
print("Quadratic:", az.loo(idata_quad))
print("Cubic:", az.loo(idata_cubic))
