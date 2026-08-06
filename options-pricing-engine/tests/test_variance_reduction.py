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
from src.variance_reduction import price_asian_antithetic

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
