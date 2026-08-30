"""EMA (Exponential Moving Average).

Pure pandas implementation — no ta-lib C dependency required.
All functions are causal: value at bar i uses only data up to bar i.
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with the standard span/alpha = 2/(period+1)."""
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def ema_stack(df: pd.DataFrame, fast: int = 20, mid: int = 50, slow: int = 200) -> pd.DataFrame:
    """Return a DataFrame with ema_fast / ema_mid / ema_slow columns.

    `df` must contain a 'close' column.
    """
    out = pd.DataFrame(index=df.index)
    out["ema_fast"] = ema(df["close"], fast)
    out["ema_mid"] = ema(df["close"], mid)
    out["ema_slow"] = ema(df["close"], slow)
    return out


def trend_alignment(row, fast_col="ema_fast", mid_col="ema_mid", slow_col="ema_slow") -> int:
    """+1 bullish stack (fast>mid>slow), -1 bearish stack (fast<mid<slow), 0 mixed."""
    if pd.isna(row[fast_col]) or pd.isna(row[mid_col]) or pd.isna(row[slow_col]):
        return 0
    if row[fast_col] > row[mid_col] > row[slow_col]:
        return 1
    if row[fast_col] < row[mid_col] < row[slow_col]:
        return -1
    return 0
