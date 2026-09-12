"""
Black-Scholes closed-form pricing.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_price(S0, K, r, sigma, T, option_type="call"):
    """
    Price a European option using the closed-form Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Current stock price (spot).
    K : float
        Strike price.
    r : float
        Risk-free rate (annual, continuously compounded).
    sigma : float
        Volatility (annual).
    T : float
        Time to maturity in years.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        The option price.

    -------------------------------------------------------------------------

    d1 = (ln(S0/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    call = S0 * N(d1) - K * exp(-r*T) * N(d2)
    put  = K * exp(-r*T) * N(-d2) - S0 * N(-d1)
    -------------------------------------------------------------------------
    """
    # d1 & d2 formulas
    d1 = (np.log(S0/K) +(r+sigma**2 *0.5)*T)/(sigma * T**0.5)
    d2 = d1 - sigma * T**0.5

    option_price = 0

    if option_type == "call":
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price