"""
Test harness (plumbing -- these are written for you).

Run with:   pytest -v

Right now every test FAILS with NotImplementedError -- that's expected. As you
implement each function during the week, its tests start passing. When they're
all green, Week 1 is done. Think of this file as your definition-of-done.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import config
from src.black_scholes import black_scholes_price
from src.gbm import simulate_gbm_terminal
from src.monte_carlo import monte_carlo_price


# ---------- Monday: Black-Scholes closed form ----------

def test_bs_call_matches_reference():
    price = black_scholes_price(config.S0, config.K, config.R,
                                config.SIGMA, config.T, "call")
    assert price == pytest.approx(config.BS_CALL_REFERENCE, abs=1e-3)


def test_bs_put_matches_reference():
    price = black_scholes_price(config.S0, config.K, config.R,
                                config.SIGMA, config.T, "put")
    assert price == pytest.approx(config.BS_PUT_REFERENCE, abs=1e-3)


def test_bs_put_call_parity():
    # C - P should equal S0 - K*exp(-r*T)
    import numpy as np
    call = black_scholes_price(config.S0, config.K, config.R,
                               config.SIGMA, config.T, "call")
    put = black_scholes_price(config.S0, config.K, config.R,
                              config.SIGMA, config.T, "put")
    parity = config.S0 - config.K * np.exp(-config.R * config.T)
    assert (call - put) == pytest.approx(parity, abs=1e-6)


# ---------- Wednesday: GBM simulation ----------

def test_gbm_output_shape():
    S_T = simulate_gbm_terminal(config.S0, config.R, config.SIGMA,
                                config.T, n_paths=10_000, seed=config.RANDOM_SEED)
    assert S_T.shape == (10_000,)
    assert (S_T > 0).all()  # prices under GBM are always positive


def test_gbm_mean_is_forward_price():
    # E[S_T] under risk-neutral GBM should be S0 * exp(r*T)
    import numpy as np
    S_T = simulate_gbm_terminal(config.S0, config.R, config.SIGMA,
                                config.T, n_paths=500_000, seed=config.RANDOM_SEED)
    expected = config.S0 * np.exp(config.R * config.T)
    assert S_T.mean() == pytest.approx(expected, rel=0.01)


# ---------- Saturday: Monte Carlo pricer ----------

def test_mc_call_converges_to_bs():
    price = monte_carlo_price(config.S0, config.K, config.R, config.SIGMA,
                              config.T, n_paths=1_000_000,
                              option_type="call", seed=config.RANDOM_SEED)
    # With 1e6 paths we should be within a few cents of the analytic price.
    assert price == pytest.approx(config.BS_CALL_REFERENCE, abs=0.05)


def test_mc_and_bs_agree():
    """The headline validation: two independent methods, same answer."""
    bs = black_scholes_price(config.S0, config.K, config.R, config.SIGMA,
                             config.T, "call")
    mc = monte_carlo_price(config.S0, config.K, config.R, config.SIGMA,
                           config.T, n_paths=1_000_000,
                           option_type="call", seed=config.RANDOM_SEED)
    assert mc == pytest.approx(bs, abs=0.05)
