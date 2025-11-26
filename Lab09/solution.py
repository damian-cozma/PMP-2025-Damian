import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def build_and_run(y_value, theta_value, mu_prior=10, draws=2000, tune=2000):
    # a) model PyMC pentru posteriorul lui n
    with pm.Model() as model:

        # prior Poisson(10)
        n_var = pm.Poisson("n", mu=mu_prior)

        # likelihood pentru Y observat
        pm.Binomial("Y_obs", n=n_var, p=theta_value, observed=y_value)

        # c) Y_future pentru predictiva
        pm.Binomial("Y_future", n=n_var, p=theta_value)

        # sampler compatibil cu variabile discrete
        step = pm.Metropolis()

        # sampling din posterior (a)
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            step=step,
            random_seed=2025,
            progressbar=False,
            return_inferencedata=True,
        )

        # sampling predictiv pentru Y* (c)
        trace = pm.sample_posterior_predictive(
            trace,
            model=model,
            var_names=["Y_future"],
            extend_inferencedata=True,
        )

    return trace


def main():
    Y_list = [0, 5, 10]
    theta_list = [0.2, 0.5]

    results = {}

    # rulam toate scenariile cerute la (a) si (c)
    for theta in theta_list:
        for y_obs in Y_list:
            idata = build_and_run(y_obs, theta)
            results[(y_obs, theta)] = idata

            # afisam sumarul posteriorului pentru n (a)
            print(f"\nPosterior pentru n | Y={y_obs}, θ={theta}")
            print(az.summary(idata, var_names=["n"], hdi_prob=0.94))

    # plot posterior n (a)
    fig1, axs1 = plt.subplots(len(Y_list), len(theta_list),
                              figsize=(10, 8), constrained_layout=True)

    for i, y_obs in enumerate(Y_list):
        for j, theta in enumerate(theta_list):
            idata = results[(y_obs, theta)]
            ax = axs1[i, j]
            az.plot_posterior(idata, var_names=["n"], ax=ax, hdi_prob=0.94)
            ax.set_title(f"n | Y={y_obs}, θ={theta}")

    # plot predictiva Y* (c)
    fig2, axs2 = plt.subplots(len(Y_list), len(theta_list),
                              figsize=(10, 8), constrained_layout=True)

    for i, y_obs in enumerate(Y_list):
        for j, theta in enumerate(theta_list):
            idata = results[(y_obs, theta)]
            samples = idata.posterior_predictive["Y_future"].values.flatten()
            ax = axs2[i, j]
            az.plot_dist(samples, ax=ax)
            ax.set_title(f"Y* | Y={y_obs}, θ={theta}")
            ax.set_xlabel("Y*")

    plt.show()


if __name__ == "__main__":
    main()
