"""RSI (Relative Strength Index), Wilder's smoothing — matches TA-Lib / TradingView."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if period < 2:
        raise ValueError("RSI period must be >= 2")
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    avg_loss_safe = avg_loss.replace(0.0, np.nan)
    rs = (avg_gain / avg_loss_safe).astype(float)
    out = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 with avg_gain > 0  -> pure up-moves -> RSI 100
    # avg_loss == 0 with avg_gain == 0 -> perfectly flat  -> RSI 50 (convention)
    both_zero = (avg_loss == 0) & (avg_gain == 0)
    out = out.where(~(avg_loss == 0), np.where(avg_gain > 0, 100.0, 50.0))
    out = pd.Series(out, index=series.index).astype(float)
    out[both_zero] = 50.0
    out.name = None
    return out
