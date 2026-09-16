"""
Test harness for Week 3: variance reduction.

Run with:   pytest -v -s tests/test_variance_reduction.py
(the -s flag is needed to see the printed std devs; pytest hides stdout by default)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

import config
from src.exotics import price_asian_option
from src.variance_reduction import price_asian_antithetic, price_asian_control_variate

N_PATHS = 10_000
N_STEPS = 50
N_TRIALS = 30  # independent replications, one per seed, to estimate each estimator's spread


def test_antithetic_reduces_error_for_asian_call():
    """Same total path budget, same option -- antithetic pairing should give
    a tighter spread of price estimates across independent seeds than the
    naive Monte Carlo Asian pricer."""
    naive_prices = [
        price_asian_option(config.S0, config.K, config.R, config.SIGMA, config.T,
                           n_paths=N_PATHS, n_steps=N_STEPS,
                           option_type="call", seed=seed)
        for seed in range(N_TRIALS)
    ]
    antithetic_prices = [
        price_asian_antithetic(config.S0, config.K, config.R, config.SIGMA, config.T,
                               n_paths=N_PATHS, n_steps=N_STEPS,
                               option_type="call", seed=seed)
        for seed in range(N_TRIALS)
    ]

    naive_std = np.std(naive_prices)
    antithetic_std = np.std(antithetic_prices)

    print(f"\nNaive Asian std over {N_TRIALS} seeds:      {naive_std:.5f}")
    print(f"Antithetic Asian std over {N_TRIALS} seeds: {antithetic_std:.5f}")

    # The two estimators should still agree on the price itself.
    assert np.mean(antithetic_prices) == pytest.approx(np.mean(naive_prices), abs=0.05)

    # The whole point of antithetic variates: lower variance for the same budget.
    assert antithetic_std < naive_std


def test_control_variate_reduces_error_for_asian_call():
    """Control-variate pricing should agree with naive MC on the price, and have
    a much tighter spread across seeds -- the geometric-Asian closed form is
    highly correlated with the arithmetic-Asian MC estimate on the same paths,
    so most of the path-to-path noise cancels out."""
    naive_prices = [
        price_asian_option(config.S0, config.K, config.R, config.SIGMA, config.T,
                           n_paths=N_PATHS, n_steps=N_STEPS,
                           option_type="call", seed=seed)
        for seed in range(N_TRIALS)
    ]
    cv_prices = [
        price_asian_control_variate(config.S0, config.K, config.R, config.SIGMA, config.T,
                                    n_paths=N_PATHS, n_steps=N_STEPS,
                                    option_type="call", seed=seed)
        for seed in range(N_TRIALS)
    ]

    naive_std = np.std(naive_prices)
    cv_std = np.std(cv_prices)

    print(f"\nNaive Asian std over {N_TRIALS} seeds:            {naive_std:.5f}")
    print(f"Control-variate Asian std over {N_TRIALS} seeds: {cv_std:.5f}")

    # The two estimators should still land in the same ballpark. The tolerance
    # is wider than the antithetic test's because the closed form used here
    # (Kemna-Vorst) assumes continuous monitoring, while the simulation only
    # monitors at N_STEPS discrete points -- that mismatch is a small, fixed
    # bias (not noise), so it doesn't shrink as n_paths grows.
    assert np.mean(cv_prices) == pytest.approx(np.mean(naive_prices), abs=0.15)

    # The whole point of control variates: much lower variance for the same budget.
    assert cv_std < naive_std / 5
