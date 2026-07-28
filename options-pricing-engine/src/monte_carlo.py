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


def monte_carlo_price_with_ci(S0, K, r, sigma, T, n_paths, option_type="call",
                              z_score=1.96, seed=None):
    """
    Price a European option by Monte Carlo AND return a confidence interval.

    This is the "narrow error bound" idea: a Monte Carlo price is an estimate,
    so it should always come with a statement of how confident you are in it.
    Reporting the CI everywhere is what separates "I computed a number" from
    "I computed a number and I know its uncertainty."

    Returns
    -------
    dict with keys:
        "price"       : float  -- the point estimate
        "std_error"   : float  -- standard error of the estimate
        "ci_low"      : float  -- lower bound of the confidence interval
        "ci_high"     : float  -- upper bound of the confidence interval
        "half_width"  : float  -- z_score * std_error (the +/- on the price)

    -------------------------------------------------------------------------
    WHAT TO IMPLEMENT (Saturday, right after monte_carlo_price):

      1. Simulate terminal prices and compute the DISCOUNTED payoff per path:
           disc_payoffs = exp(-r*T) * np.maximum(S_T - K, 0.0)   # call
      2. price     = disc_payoffs.mean()
      3. std_error = disc_payoffs.std(ddof=1) / sqrt(n_paths)
      4. half_width = z_score * std_error
      5. ci_low, ci_high = price - half_width, price + half_width
      6. return the dict above.

    WHY std_error shrinks like 1/sqrt(n): averaging n independent draws divides
    the variance by n, so the standard error falls with the SQUARE ROOT of the
    path count. This is exactly why variance reduction (Week 3) matters -- it
    shrinks the error WITHOUT needing more paths.

    This function is also what makes the Week 3 comparison possible: you'll put
    naive MC, antithetic, and control-variate side by side and show which gives
    the narrowest CI for the same number of paths.
    -------------------------------------------------------------------------
    """
    raise NotImplementedError("Implement monte_carlo_price_with_ci in Saturday's block.")
