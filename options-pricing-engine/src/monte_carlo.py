"""
Monte Carlo pricing of European options.

WEEK 1 - SATURDAY (deep block). This is the payoff of the week: watch this
pricer converge to the Black-Scholes number as you add paths. That convergence
is your validation and the first figure in your README.
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
    WHAT TO IMPLEMENT (Saturday):

      1. Simulate terminal prices S_T with simulate_gbm_terminal(...).
      2. Compute the payoff on each path:
           call: payoff = max(S_T - K, 0)   -> np.maximum(S_T - K, 0.0)
           put:  payoff = max(K - S_T, 0)   -> np.maximum(K - S_T, 0.0)
      3. Take the mean payoff, then discount it back to today:
           price = exp(-r * T) * mean(payoff)
      4. return price

    WHY THIS WORKS (one sentence): an option's price is the discounted expected
    payoff under the risk-neutral measure -- simulating S_T and averaging the
    discounted payoffs is just estimating that expectation numerically.

    VALIDATION: as n_paths grows (1e3 -> 1e6), this should march toward the
    Black-Scholes reference of ~10.4506. Plot that convergence -- it's Figure 1
    of your writeup.
    -------------------------------------------------------------------------
    """
    raise NotImplementedError("Implement monte_carlo_price in Saturday's deep block.")


def monte_carlo_price_with_stderr(S0, K, r, sigma, T, n_paths, option_type="call", seed=None):
    """
    Same as monte_carlo_price, but also return the standard error of the
    estimate. You'll want this in Week 3 to show variance reduction working.

    Returns
    -------
    (float, float)
        (price, standard_error)

    HINT (leave for later if you like): the standard error is
        std(discounted_payoffs) / sqrt(n_paths)
    """
    raise NotImplementedError("Implement in Week 3 when you measure variance reduction.")
