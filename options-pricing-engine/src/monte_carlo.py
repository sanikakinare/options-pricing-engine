"""
Monte Carlo pricing of European options.
"""

import numpy as np
from .gbm import simulate_gbm_terminal


def monte_carlo_price(S0, K, r, sigma, T, n_paths, option_type="call", seed=None):
    """
    Price a European option by Monte Carlo simulation.

    Parameters
    ----------
    S0, K, r, sigma, T : float
        Standard option parameters.
    n_paths : int
        Number of simulated terminal prices.
    option_type : str
        "call" or "put".
    seed : int or None
        Optional random seed.

    Returns
    -------
    float
        The Monte Carlo option price estimate.

    -------------------------------------------------------------------------
    """
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=seed)
    if option_type == "call":
        payoff = np.maximum(ST - K,0)
    if option_type == "put":
        payoff = np.maximum(K - ST,0)

    discounted_payoff = payoff * np.exp(-r*T)

    average_payoff = np.mean(discounted_payoff)
    return average_payoff


def monte_carlo_price_with_ci(S0, K, r, sigma, T, n_paths, option_type="call",
                              z_score=1.96, seed=None):
    """
    Price a European option by Monte Carlo AND return a confidence interval.
    -------------------------------------------------------------------------
    """
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=seed)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    if option_type == "put":
        payoff = np.maximum(K - ST, 0.0)

    discounted_payoff = np.exp(-r*T) * payoff

    average_price = discounted_payoff.mean() 
    standard_error = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)

    half_width = z_score * standard_error 
    ci_low = average_price - half_width
    ci_high = average_price + half_width

    return {
        "price": average_price,
        "std_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "half_width": half_width
    }
