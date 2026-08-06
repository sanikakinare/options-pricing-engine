import numpy as np
"""
Variance reduction techniques.

WEEK 3. These make your Monte Carlo estimates converge faster (smaller standard
error for the same number of paths). This is the most "quant" week and the
centerpiece of your writeup.

  - Antithetic variates: for each random draw Z, also use -Z. The paired paths
    partially cancel each other's noise.
  - Control variates: use a quantity with a KNOWN answer (the geometric-average
    Asian, which has a closed form) to correct the noisy estimate of the
    quantity you actually want (the arithmetic-average Asian).

Left as placeholders for now.
"""


def price_asian_antithetic(S0, K, r, sigma, T, n_paths, n_steps, option_type="call", seed=None):
    """Week 3 -- antithetic-variates version of the Asian pricer."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths//2, n_steps))
    Z_neg = -Z

    dt = T/n_steps
    drift = (r - 0.5*sigma**2) * dt
    wobble = sigma * np.sqrt(dt) * Z
    wobble_neg = sigma * np.sqrt(dt) * Z_neg

    S_path = S0 * np.exp(np.cumsum(drift + wobble, axis=1))
    S_path_neg = S0 * np.exp(np.cumsum(drift + wobble_neg, axis=1))

    S_average = np.mean(S_path, axis=1)
    S_average_neg = np.mean(S_path_neg, axis=1)

    if option_type == "call":
        payoff = np.maximum(S_average - K, 0)
        payoff_neg = np.maximum(S_average_neg - K, 0)
    else:
        payoff = np.maximum(K - S_average, 0)
        payoff_neg = np.maximum(K - S_average_neg, 0)

    payoff_mean = (payoff + payoff_neg) / 2
    final_payoff = np.mean(payoff_mean)
    discounted_payoff = final_payoff * np.exp(-r*T)

    return discounted_payoff

def price_asian_control_variate(*args, **kwargs):
    """Week 3 -- control-variate version using the geometric-Asian closed form."""
    raise NotImplementedError("Week 3.")
