"""Download one snapshot of an options chain and save it to data/.

Usage:  python scripts/fetch_chain.py SPY
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_chain(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    snapshot_time = datetime.now(timezone.utc)

    frames = []
    for expiry in tk.options:
        chain = tk.option_chain(expiry)
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["option_type"] = option_type
            df["expiry"] = expiry
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["spot"] = spot
    out["snapshot_utc"] = snapshot_time.isoformat()
    return out


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = fetch_chain(ticker)

    Path("data").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = Path("data") / f"{ticker}_chain_{stamp}.csv"
    df.to_csv(path, index=False)

    print(f"Saved {len(df)} rows to {path}")
    print(f"Spot: {df['spot'].iloc[0]:.2f}")
    print(f"Expiries: {df['expiry'].nunique()}")
    print(f"Rows with zero bid: {(df['bid'] == 0).sum()}")