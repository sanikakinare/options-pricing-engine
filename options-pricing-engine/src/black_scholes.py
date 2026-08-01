"""
Black-Scholes closed-form pricing.

WEEK 1 - MONDAY. This is your analytic ("exact") pricer. Later you'll build a
Monte Carlo pricer that should agree with this one to 3-4 decimals -- that
agreement is your proof both are correct.
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
    WHAT TO IMPLEMENT (Monday):

      d1 = (ln(S0/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)

      call = S0 * N(d1) - K * exp(-r*T) * N(d2)
      put  = K * exp(-r*T) * N(-d2) - S0 * N(-d1)

    where N(.) is the standard normal CDF -> use norm.cdf from scipy.stats.

    INTUITION TO PIN DOWN (one sentence each, for your SOP later):
      - N(d2) is roughly the risk-neutral probability the option finishes
        in the money.
      - Raising sigma raises the price -- more volatility means more upside
        without more downside (the payoff is floored at zero).

    VALIDATION: with S0=100, K=100, r=0.05, sigma=0.2, T=1 the call should
    come out to about 10.4506 (see config.BS_CALL_REFERENCE).
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