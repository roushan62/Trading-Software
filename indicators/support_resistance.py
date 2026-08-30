"""Support/Resistance auto-detection via swing highs/lows (fractal method).

A swing high is a bar whose high is the highest within `k` bars on both sides
(a fractal pivot). Swing lows are the mirror image. These pivots form the
auto-detected support/resistance levels and feed structure-based stop-losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Swing:
    index: int          # bar position (iloc)
    timestamp: pd.Timestamp
    price: float
    kind: str           # "high" | "low"


def _fractal_pivots(values: np.ndarray, k: int, is_high: bool) -> list[int]:
    pivots: list[int] = []
    n = len(values)
    for i in range(k, n - k):
        window = values[i - k : i + k + 1]
        center = values[i]
        if np.isnan(center):
            continue
        if is_high:
            if center >= window.max() and (window == center).sum() == 1:
                pivots.append(i)
        else:
            if center <= window.min() and (window == center).sum() == 1:
                pivots.append(i)
    return pivots


def find_swings(df: pd.DataFrame, k: int = 3) -> list[Swing]:
    """Detect swing highs and lows. `k` = bars required on each side of the pivot."""
    if k < 1:
        raise ValueError("swing strength k must be >= 1")
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    ts = df.index

    swings: list[Swing] = []
    for i in _fractal_pivots(highs, k, True):
        swings.append(Swing(i, ts[i], float(highs[i]), "high"))
    for i in _fractal_pivots(lows, k, False):
        swings.append(Swing(i, ts[i], float(lows[i]), "low"))
    return sorted(swings, key=lambda s: s.index)


def support_resistance_levels(
    df: pd.DataFrame, k: int = 3, lookback: int = 200, merge_pct: float = 0.15
) -> dict[str, list[float]]:
    """Auto-detected S/R levels over the last `lookback` bars.

    Levels closer than merge_pct (as % of price) are merged (averaged) so the
    output isn't cluttered with duplicates. Returns {"support": [...], "resistance": [...]}
    relative to the latest close.
    """
    tail = df.tail(lookback) if lookback and len(df) > lookback else df
    swings = find_swings(tail, k)
    last_close = float(tail["close"].iloc[-1])

    highs = sorted(s.price for s in swings if s.kind == "high")
    lows = sorted(s.price for s in swings if s.kind == "low")

    def merge(levels: list[float]) -> list[float]:
        merged: list[float] = []
        for p in levels:
            if merged and abs(p - merged[-1]) / max(merged[-1], 1e-9) * 100 <= merge_pct:
                merged[-1] = (merged[-1] + p) / 2.0
            else:
                merged.append(p)
        return merged

    resistance = [p for p in merge(highs) if p > last_close]
    support = [p for p in merge(lows) if p <= last_close]
    return {"support": support, "resistance": resistance}


def last_swing_low_before(df: pd.DataFrame, bar_pos: int, lookback: int = 20, k: int = 2) -> float | None:
    """Lowest swing low within `lookback` bars ending at `bar_pos` (inclusive).

    Used for structure-based stop-loss on longs: stop goes under the most
    recent significant swing low. Returns None if no pivot exists yet.
    """
    start = max(0, bar_pos - lookback + 1)
    window = df.iloc[start : bar_pos + 1]
    if len(window) < 2 * k + 1:
        return None
    lows = [s.price for s in find_swings(window, k) if s.kind == "low"]
    if not lows:
        return None
    return min(lows)


def last_swing_high_before(df: pd.DataFrame, bar_pos: int, lookback: int = 20, k: int = 2) -> float | None:
    """Highest swing high within `lookback` bars ending at `bar_pos` (inclusive)."""
    start = max(0, bar_pos - lookback + 1)
    window = df.iloc[start : bar_pos + 1]
    if len(window) < 2 * k + 1:
        return None
    highs = [s.price for s in find_swings(window, k) if s.kind == "high"]
    if not highs:
        return None
    return max(highs)
