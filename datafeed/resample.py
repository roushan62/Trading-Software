"""Timeframe aggregation: build higher timeframes from a base series.

e.g. hold 5m data and derive 15m/1h bars, so one download serves all scans.
"""
from __future__ import annotations

import pandas as pd

from .base import normalize

_TF_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a lower-timeframe OHLCV frame into `timeframe` bars."""
    if timeframe not in _TF_RULE:
        raise ValueError(f"unsupported target timeframe '{timeframe}'")
    rule = _TF_RULE[timeframe]

    agg = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "close"])
    if len(agg) == 0:
        raise ValueError(f"resample produced 0 bars for {timeframe}")
    return normalize(agg)
