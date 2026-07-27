"""
Geometric Brownian Motion simulation under the risk-neutral measure.

WEEK 1 - WEDNESDAY. This is the engine underneath your Monte Carlo pricer.
The single most important idea in this file: under risk-neutral pricing the
drift is the risk-free rate r, NOT the stock's real-world expected return mu.
Make sure you can say why in one sentence before you move on.
"""

import numpy as np


def simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=None):
    """
    Simulate terminal stock prices S_T under risk-neutral GBM.

    Parameters
    ----------
    S0 : float
        Current stock price.
    r : float
        Risk-free rate (this is the drift, under the risk-neutral measure).
    sigma : float
        Volatility (annual).
    T : float
        Time to maturity in years.
    n_paths : int
        Number of simulated terminal prices to draw.
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) of simulated terminal prices S_T.

    -------------------------------------------------------------------------
    WHAT TO IMPLEMENT (Wednesday):

      The closed-form solution of risk-neutral GBM at time T is:

        S_T = S0 * exp( (r - sigma^2 / 2) * T  +  sigma * sqrt(T) * Z )

      where Z ~ standard normal. You only need the TERMINAL price here (not the
      full path), so you can draw all n_paths values of Z at once and vectorize
      the whole thing -- no Python loops.

      Steps:
        1. rng = np.random.default_rng(seed)
        2. Z = rng.standard_normal(n_paths)
        3. return S0 * exp((r - 0.5*sigma**2)*T + sigma*sqrt(T)*Z)

    THE KEY IDEA (say it in one sentence): we discount and price under the
    risk-neutral measure, where every asset drifts at r; the real-world drift
    mu never enters an option price. This is why r -- not mu -- is used here.
    -------------------------------------------------------------------------
    """
    raise NotImplementedError("Implement simulate_gbm_terminal in Wednesday's session.")
