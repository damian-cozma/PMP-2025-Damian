import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

n = 10        
k_total = 180   
rate_obs = k_total / n

alpha_prior = 9
beta_prior = alpha_prior / 15

alpha_post = alpha_prior + k_total
beta_post = beta_prior + n

mean_post = alpha_post / beta_post
var_post = alpha_post / (beta_post ** 2)
mode_post = (alpha_post - 1) / beta_post if alpha_post > 1 else 0

print(f"Posterior: Gamma({alpha_post:.2f}, β={beta_post:.2f})")
print(f"Media a posteriori: {mean_post:.2f}")
print(f"Modul a posteriori: {mode_post:.2f}")
print(f"Varianta a posteriori: {var_post:.2f}")

hdi_lower, hdi_upper = stats.gamma.ppf([0.03, 0.97], a=alpha_post, scale=1/beta_post)
print(f"Interval HDI 94%: [{hdi_lower:.2f}, {hdi_upper:.2f}]")

lambda_values = np.linspace(10, 25, 500)
posterior_pdf = stats.gamma.pdf(lambda_values, a=alpha_post, scale=1/beta_post)

plt.figure(figsize=(10,6))
plt.plot(lambda_values, posterior_pdf, label='Posterior Gamma')
plt.axvline(mean_post, color='r', linestyle='--', label=f'Mean = {mean_post:.2f}')
plt.axvline(mode_post, color='g', linestyle='--', label=f'Mode = {mode_post:.2f}')
plt.fill_between(lambda_values, posterior_pdf,
                 where=(lambda_values >= hdi_lower) & (lambda_values <= hdi_upper),
                 color='gray', alpha=0.4, label='94% HDI')
plt.title("Posterior distribution of Gamma–Poisson conjugate")
plt.xlabel("Average calls per hour")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
