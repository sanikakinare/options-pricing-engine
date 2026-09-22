import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.black_scholes import black_scholes_price
from src.implied_vol import implied_vol


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.20, 0.60])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_round_trip(sigma, K, option_type):
    """Price at a known sigma, invert, recover the same sigma."""
    S0, r, T = 100, 0.05, 1.0
    price = black_scholes_price(S0, K, r, sigma, T, option_type)
    assert implied_vol(price, S0, K, r, T, option_type) == pytest.approx(sigma, abs=1e-6)


def test_price_below_intrinsic_returns_nan():
    """No sigma can produce a call price below S0 - K*exp(-rT)."""
    S0, K, r, T = 100, 80, 0.05, 1.0
    assert np.isnan(implied_vol(10.0, S0, K, r, T, "call"))