"""VWAP (Volume Weighted Average Price).

For intraday timeframes VWAP is anchored to the session (daily reset),
which is how VWAP is used in practice. For daily+ timeframes a rolling
VWAP over `period` bars is returned instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _session_key(index: pd.DatetimeIndex, intraday: bool):
    if not intraday:
        return None
    if getattr(index, "tz", None) is not None:
        return index.tz_localize(None).normalize()
    return index.normalize()


def vwap(df: pd.DataFrame, period: int = 20, intraday: bool | None = None) -> pd.Series:
    """VWAP over OHLCV data.

    intraday=True  -> session-anchored VWAP (resets each calendar day)
    intraday=False -> rolling VWAP over `period` bars
    intraday=None  -> auto-detect from the median bar spacing
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"vwap: missing column '{col}'")

    idx = df.index
    if intraday is None:
        intraday = _infer_intraday(idx)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    if intraday:
        key = _session_key(idx, True)
        cum_pv = pv.groupby(key).cumsum()
        cum_v = df["volume"].groupby(key).cumsum()
        with np.errstate(divide="ignore", invalid="ignore"):
            out = cum_pv / cum_v.replace(0, np.nan)
        # Bars with no volume: fall back to typical price
        out = out.fillna(typical)
        return out

    roll_pv = pv.rolling(period, min_periods=1).sum()
    roll_v = df["volume"].rolling(period, min_periods=1).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = roll_pv / roll_v.replace(0, np.nan)
    return out.fillna(typical)


def _infer_intraday(index: pd.DatetimeIndex) -> bool:
    """Heuristic: if median gap between bars < 20h, treat as intraday."""
    if len(index) < 3:
        return False
    gaps = np.diff(index.values).astype("timedelta64[s]").astype(float)
    median_gap = np.median(gaps)
    return median_gap < 20 * 3600
