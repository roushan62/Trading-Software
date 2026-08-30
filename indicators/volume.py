"""Volume analysis: relative volume (z-score) to detect participation/expansion."""
from __future__ import annotations

import pandas as pd


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of current volume vs its rolling mean/std.

    > 0  = above-average participation
    > 1.5 = clear expansion (breakout confirmation)
    """
    mean = volume.rolling(period, min_periods=max(3, period // 2)).mean()
    std = volume.rolling(period, min_periods=max(3, period // 2)).std()
    std = std.replace(0.0, pd.NA)
    z = (volume - mean) / std
    return z.fillna(0.0)
