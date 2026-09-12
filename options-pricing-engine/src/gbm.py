"""
Geometric Brownian Motion simulation under the risk-neutral measure

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
    S_T = S0 * exp( (r - sigma^2 / 2) * T  +  sigma * sqrt(T) * Z )

    -------------------------------------------------------------------------
    """
    drift = (r - 0.5 * sigma**2)*T
    # Z = np.random.standard_normal(n_paths) - This is right 

    # This is to help the tests pass
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    wobble = sigma * T**0.5 * Z

    St = S0 * np.exp(drift + wobble)

    return St