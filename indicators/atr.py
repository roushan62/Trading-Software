"""ATR (Average True Range), Wilder's smoothing — used for volatility-based stops."""
from __future__ import annotations

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"atr: missing column '{col}'")
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as % of close — comparable volatility filter across symbols."""
    return (atr(df, period) / df["close"]) * 100.0
