"""
Test harness for Week 2: Asian options.

Run with:   pytest -v -s tests/test_exotics.py
(the -s flag is needed to see the printed prices; pytest hides stdout by default)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import config
from src.black_scholes import black_scholes_price
from src.exotics import price_asian_option


def test_asian_call_cheaper_than_vanilla():
    """Sanity check called out in exotics.py: averaging dampens volatility,
    so the Asian call must price below the equivalent vanilla call."""
    bs = black_scholes_price(config.S0, config.K, config.R,
                             config.SIGMA, config.T, "call")
    asian = price_asian_option(config.S0, config.K, config.R, config.SIGMA, config.T,
                               n_paths=100_000, n_steps=252,
                               option_type="call", seed=config.RANDOM_SEED)

    print(f"\nVanilla BS call:   {bs:.4f}")
    print(f"Asian call (MC):   {asian:.4f}")

    assert asian < bs


def test_asian_put_cheaper_than_vanilla():
    """Same sanity check on the put side."""
    bs = black_scholes_price(config.S0, config.K, config.R,
                             config.SIGMA, config.T, "put")
    asian = price_asian_option(config.S0, config.K, config.R, config.SIGMA, config.T,
                               n_paths=100_000, n_steps=252,
                               option_type="put", seed=config.RANDOM_SEED)

    print(f"\nVanilla BS put:    {bs:.4f}")
    print(f"Asian put (MC):    {asian:.4f}")

    assert asian < bs
