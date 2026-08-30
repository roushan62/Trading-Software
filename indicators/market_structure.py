"""Market structure classifier using HH-HL / LL-LH swing logic.

UPTREND   = Higher Highs AND Higher Lows
DOWNTREND = Lower Lows  AND Lower Highs
RANGE     = anything mixed

Classification is computed causally at each bar using only swings that are
already confirmed (the pivot bar plus `k` bars to its right).
"""
from __future__ import annotations

import pandas as pd

from .support_resistance import Swing, find_swings

UPTREND = "uptrend"
DOWNTREND = "downtrend"
RANGE = "range"


def classify_swings(swings: list[Swing]) -> str:
    """Classify from the most recent confirmed swing highs/lows.

    Standard market-structure read:
      last high > prev high AND last low > prev low  -> UPTREND (HH+HL)
      last high < prev high AND last low < prev low  -> DOWNTREND (LH+LL)
      anything else                                  -> RANGE
    """
    highs = [s.price for s in swings if s.kind == "high"]
    lows = [s.price for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return ""
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return UPTREND
    if lh and ll:
        return DOWNTREND
    return RANGE


def market_structure_series(df: pd.DataFrame, k: int = 3, min_swings: int = 4) -> pd.Series:
    """Causal per-bar structure label: 'uptrend' | 'downtrend' | 'range'.

    At bar i, only swings whose confirmation window (pivot + k right bars)
    ends at or before i are considered. Carries forward the last classification.
    """
    n = len(df)
    out = [None] * n
    swings = find_swings(df, k)

    confirmed_at = []  # (confirmation_bar, swing)
    for s in swings:
        confirmed_at.append((s.index + k, s))
    confirmed_at.sort(key=lambda t: t[0])

    pending: list[Swing] = []
    ptr = 0
    last_label = None
    for i in range(n):
        while ptr < len(confirmed_at) and confirmed_at[ptr][0] <= i:
            pending.append(confirmed_at[ptr][1])
            ptr += 1
        if len(pending) >= min_swings:
            label = classify_swings(pending)
            if label:
                last_label = label
        out[i] = last_label
    return pd.Series(out, index=df.index, dtype="object")
