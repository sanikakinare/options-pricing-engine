import numpy as np
from scipy.stats import norm
"""
Variance reduction techniques.
  - Antithetic variates: for each random draw Z, also use -Z. The paired paths
    partially cancel each other's noise.
  - Control variates: use a quantity with a KNOWN answer to correct the noisy estimate of the
    quantity you actually want
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

def price_asian_control_variate(S0, K, r, sigma, T, n_paths, n_steps, option_type="call", seed=None):
    """Week 3 -- control-variate version using the geometric-Asian closed form."""
    # Kemna Vorst
    # effective_sigma = sigma / np.sqrt(3)
    # effective_drift = 0.5 * (r - sigma**2/6)

    # d1 = (np.log(S0/K) + (effective_drift + effective_sigma**2/2) * T)/(effective_sigma * np.sqrt(T))
    # d2 = d1 - effective_sigma * np.sqrt(T)

    # Kemna Vorst with n steps - unbiased control variate
    dt = T/n_steps
    mean = np.log(S0) + (r - 0.5 * sigma**2) * dt * (n_steps + 1) / 2
    variance = sigma**2 * dt * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps)
    d1 = (mean - np.log(K) + variance) / np.sqrt(variance)
    d2 = d1 - np.sqrt(variance)

    if option_type == "call":
        option_price = np.exp(-r*T) * (np.exp(mean + variance/2) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        option_price = np.exp(-r*T) * (K * norm.cdf(-d2) - np.exp(mean + variance/2) * norm.cdf(-d1))
    
    S_path = simulate_paths(S0, r, sigma, T, n_paths, n_steps, seed=seed)
    geometric_price = geometric_asian_mc(S_path, n_steps, K, r, T, option_type=option_type)
    arithmetic_price = arithmetic_asian_mc(S_path, K, r, T, option_type=option_type)

    coeff = np.cov(arithmetic_price, geometric_price)[0, 1]/np.var(geometric_price, ddof=1)

    control_variate_price = arithmetic_price - coeff * (geometric_price - option_price)

    # standard_error = control_variate_price.std(ddof=1) / np.sqrt(n_paths)

    return control_variate_price.mean()

def simulate_paths(S0, r, sigma, T, n_paths, n_steps, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))

    dt = T/n_steps
    drift = (r - 0.5*sigma**2)*dt
    wobble = sigma * np.sqrt(dt) * Z

    S_path = S0 * np.exp(np.cumsum(drift + wobble, axis=1))
    return S_path

def arithmetic_asian_mc(S_paths, K, r, T, option_type="call"):
    S_average = np.mean(S_paths, axis = 1)

    if option_type == "call":
        payoff = np.maximum(S_average - K,0)
    else:
        payoff = np.maximum(K - S_average, 0)

    discounted_payoff = payoff * np.exp(-r*T)
    return discounted_payoff

def geometric_asian_mc(S_paths, n_steps, K, r, T, option_type = "call"):
    S_geometric = np.exp(np.mean(np.log(S_paths), axis=1))

    if option_type == "call":
        payoff = np.maximum(S_geometric - K, 0)
    else:
        payoff = np.maximum(K - S_geometric, 0)

    discounted_payoff = payoff * np.exp(-r*T)
    return discounted_payoff