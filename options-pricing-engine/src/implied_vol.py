"""Implied volatility: invert Black-Scholes to recover the volatility a market price implies."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.black_scholes import black_scholes_price


def implied_vol(price, S0, K, r, T, option_type="call", lo=1e-4, hi=5.0):
    """
    Return the sigma for which black_scholes_price equals `price`, or NaN if none exists.

    Black-Scholes is strictly increasing in sigma (vega > 0), so at most one
    sigma reproduces any given price.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan

    def f(sigma):
        return black_scholes_price(S0, K, r, sigma, T, option_type) - price

    f_lo, f_hi = f(lo), f(hi)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan
    if f_lo > 0 or f_hi < 0:
        return np.nan

    return brentq(f, lo, hi, xtol=1e-12)


def load_chain(path):
    """Read a saved chain CSV and add mid price and time to expiry in years."""
    chain = pd.read_csv(path, parse_dates=["snapshot_utc", "lastTradeDate"])

    # Expiry is the 4pm ET close. Keep the fractional day: truncating to whole
    # days biases IV noticeably on short-dated options, where price depends on
    # sigma * sqrt(T).
    expiry_ts = pd.to_datetime(chain["expiry"], utc=True) + pd.Timedelta(hours=20)
    snapshot = pd.to_datetime(chain["snapshot_utc"], utc=True)

    # Mid, not lastPrice -- a last price can be days stale and reflect a spot
    # level that no longer exists.
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["T"] = (expiry_ts - snapshot).dt.total_seconds() / (365.25 * 24 * 3600)
    return chain


def build_smile(chain, expiry, r, max_spread_frac=0.5, min_open_interest=1):
    """Implied vol for every strike on one expiry. Returns a DataFrame of strike, type, mid, iv."""
    df = chain[chain["expiry"] == expiry].copy()
    if df.empty:
        raise ValueError(f"no rows for expiry {expiry!r}")

    spot = df["spot"].iloc[0]
    T = df["T"].iloc[0]

    # Keep only out-of-the-money quotes: an ITM option is nearly all intrinsic
    # value, so vega is tiny and a one-cent quote error swings the implied vol.
    # Put-call parity means the OTM option at the same strike says the same
    # thing, better posed. Then drop quotes with no market behind them.
    otm = np.where(df["option_type"] == "call", df["strike"] > spot, df["strike"] <= spot)
    df = df[
        otm
        & (df["bid"] > 0)                                        # nobody bidding
        & (df["ask"] > df["bid"])
        & (df["openInterest"] >= min_open_interest)              # nobody holding
        & ((df["ask"] - df["bid"]) / df["mid"] <= max_spread_frac)  # mid is a guess
    ].copy()

    df["iv"] = [
        implied_vol(mid, spot, K, r, T, option_type)
        for mid, K, option_type in zip(df["mid"], df["strike"], df["option_type"])
    ]

    out = df[["strike", "option_type", "mid", "iv"]].dropna(subset=["iv"])
    out = out.sort_values("strike").reset_index(drop=True)
    out.attrs.update(spot=spot, T=T, expiry=expiry, r=r)
    return out
