import numpy as np
from scipy.stats import norm
"""
Variance reduction techniques.

WEEK 3. These make your Monte Carlo estimates converge faster (smaller standard
error for the same number of paths).

  - Antithetic variates: for each random draw Z, also use -Z. The paired paths
    partially cancel each other's noise.
  - Control variates: use a quantity with a KNOWN answer (the geometric-average
    Asian, which has a closed form) to correct the noisy estimate of the
    quantity you actually want (the arithmetic-average Asian).

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
    effective_sigma = sigma / np.sqrt(3)
    effective_drift = 0.5 * (r - sigma**2/6)

    d1 = (np.log(S0/K) + (effective_drift + effective_sigma**2/2) * T)/(effective_sigma * np.sqrt(T))
    d2 = d1 - effective_sigma * np.sqrt(T)

    if option_type == "call":
        option_price = S0 * np.exp((effective_drift - r)*T)* norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 *  np.exp((effective_drift - r)*T)* norm.cdf(-d1)

    S_path = simulate_paths(S0, r, sigma, T, n_paths, n_steps, seed=seed)
    geometric_price = geometric_asian_mc(S_path, n_steps, K, r, T, option_type=option_type)
    arithmetic_price = arithmetic_asian_mc(S_path, K, r, T, option_type=option_type)

    control_variate_price = arithmetic_price + (option_price - geometric_price)
    return control_variate_price

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

    discounted_payoff = np.mean(payoff) * np.exp(-r*T)
    return discounted_payoff

def geometric_asian_mc(S_paths, n_steps, K, r, T, option_type = "call"):
    S_geometric = np.prod(S_paths, axis = 1) ** (1/n_steps)

    if option_type == "call":
        payoff = np.maximum(S_geometric - K, 0)
    else:
        payoff = np.maximum(K - S_geometric, 0)

    discounted_payoff = np.mean(payoff) * np.exp(-r*T)
    return discounted_payoff