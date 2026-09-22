"""Implied volatility: invert Black-Scholes to recover the volatility a market price implies."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.black_scholes import black_scholes_price


def implied_vol(price, S0, K, r, T, option_type="call", lo=1e-4, hi=5.0):
    """
    Return the sigma for which black_scholes_price(...) equals `price`, or NaN if none exists.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    S0, K, r, T : float
        Spot, strike, risk-free rate, time to maturity in years.
    option_type : str
        "call" or "put".
    lo, hi : float
        Volatility bracket to search within.

    Returns
    -------
    float
        The implied volatility, or NaN if no sigma in [lo, hi] reproduces `price`.

    -------------------------------------------------------------------------
    Black-Scholes is strictly increasing in sigma (vega > 0), so at most one
    sigma reproduces any given price. Define

        f(sigma) = black_scholes_price(sigma) - price

    and root-find. f is monotone increasing, so f(lo) and f(hi) have opposite
    signs exactly when a solution exists in the bracket. If they share a sign
    the price is outside the no-arbitrage range -- below intrinsic, or above
    the upper bound -- and no sigma can produce it.
    -------------------------------------------------------------------------
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
    """
    Read a saved chain CSV and add mid price and time to expiry in years.

    Adds two columns the raw Yahoo data does not have but Black-Scholes needs:

    mid : (bid + ask) / 2
        The live two-sided quote. Preferred over `lastPrice`, which can be days
        stale and reflect a spot level that no longer exists.
    T : time to expiry in years
        Measured from the snapshot timestamp to the 4pm ET close on the expiry
        date, keeping the fractional day. Truncating to whole days biases IV
        noticeably on short-dated options, where price depends on sigma*sqrt(T).
    """
    chain = pd.read_csv(path, parse_dates=["snapshot_utc", "lastTradeDate"])

    expiry_ts = pd.to_datetime(chain["expiry"], utc=True) + pd.Timedelta(hours=20)
    snapshot = pd.to_datetime(chain["snapshot_utc"], utc=True)

    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["T"] = (expiry_ts - snapshot).dt.total_seconds() / (365.25 * 24 * 3600)
    return chain


def build_smile(chain, expiry, r, max_spread_frac=0.5, min_open_interest=1):
    """
    Implied vol for every strike on one expiry. Returns a DataFrame of strike, type, mid, iv.

    A smile is a curve at FIXED maturity, so this slices to one expiry, discards
    quotes that cannot be trusted, and inverts what survives.

    Quotes are dropped when:
      - bid is zero          -- nobody is bidding, so there is no market
      - open interest is nil -- nobody holds it, the quote is a placeholder
      - the spread is wide   -- the mid is a guess between two untradeable prices
      - the option is ITM    -- almost all intrinsic value, so vega is tiny and
                                a one-cent quote error swings the implied vol.
                                Put-call parity means the OTM option at the same
                                strike carries the same information, better posed.
    """
    df = chain[chain["expiry"] == expiry].copy()
    if df.empty:
        raise ValueError(f"no rows for expiry {expiry!r}")

    spot = df["spot"].iloc[0]
    T = df["T"].iloc[0]

    otm = np.where(df["option_type"] == "call", df["strike"] > spot, df["strike"] <= spot)
    df = df[
        otm
        & (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["openInterest"] >= min_open_interest)
        & ((df["ask"] - df["bid"]) / df["mid"] <= max_spread_frac)
    ].copy()

    df["iv"] = [
        implied_vol(mid, spot, K, r, T, option_type)
        for mid, K, option_type in zip(df["mid"], df["strike"], df["option_type"])
    ]

    out = df[["strike", "option_type", "mid", "iv"]].dropna(subset=["iv"])
    out = out.sort_values("strike").reset_index(drop=True)
    out.attrs.update(spot=spot, T=T, expiry=expiry, r=r)
    return out
