"""Synthetic OHLCV generator — regime-switching GBM with realistic intraday sessions.

Purpose: offline demos, unit tests and forward-test replay when no market
data connection is available (e.g. sandboxed environments). Synthetic data is
NOT real market data — never use it to judge a strategy's real edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataProvider, normalize

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 390}  # 1d = full US session


def generate_ohlcv(
    symbol: str = "SYNTH",
    timeframe: str = "1h",
    bars: int = 3000,
    start: str = "2023-01-02",
    seed: int | None = None,
    s0: float = 100.0,
    annual_vol: float = 0.35,
) -> pd.DataFrame:
    """Regime-switching random walk -> OHLCV bars.

    Trend regimes (bull/bear/chop) persist for a random number of bars so the
    series contains genuine higher-highs/higher-lows stretches — enough for
    EMA/structure logic to have something to detect. Volume correlates with
    absolute return plus noise.
    """
    if timeframe not in _TF_MINUTES:
        raise ValueError(f"unsupported timeframe '{timeframe}'")
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol + timeframe)) % (2**32))

    step_min = _TF_MINUTES[timeframe]
    freq = f"{step_min}min" if timeframe != "1d" else "1B"
    index = pd.date_range(start=start, periods=bars, freq=freq)

    # --- regime switching drift ---
    n = bars
    drift = np.zeros(n)
    regime = rng.choice([-1, 0, 1], p=[0.3, 0.35, 0.35])
    strength = 0.0
    i = 0
    while i < n:
        length = int(rng.integers(40, 220))
        regime = rng.choice([-1, 0, 1], p=[0.3, 0.35, 0.35])
        strength = float(rng.uniform(0.0004, 0.0022))
        drift[i : i + length] = regime * strength
        i += length

    bar_vol = annual_vol / np.sqrt(252 * (390 / step_min)) if timeframe != "1d" else annual_vol / np.sqrt(252)
    noise = rng.normal(0, bar_vol, n)
    rets = drift + noise
    rets[0] = 0.0

    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]

    spread = np.abs(rng.normal(0, bar_vol * 0.8, n)) + bar_vol * 0.1
    high = np.maximum(open_, close) + spread * rng.uniform(0.3, 1.0, n)
    low = np.minimum(open_, close) - spread * rng.uniform(0.3, 1.0, n)

    base_volume = float(rng.uniform(5e5, 5e6))
    volume = base_volume * (1 + 6 * np.abs(rets) / max(bar_vol, 1e-9) * 0.15)
    volume *= rng.lognormal(0, 0.35, n)
    # U-shaped intraday volume profile
    if timeframe != "1d" and n > 10:
        pos = np.linspace(0, 1, n)
        volume *= 1 + 0.6 * (np.cos(np.pi * (2 * (pos % (78 / max(1, 390 / step_min)) if step_min < 390 else pos))) ** 2)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df.index.name = "datetime"
    return normalize(df)


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self, bars: int = 3000, seed: int | None = None):
        self.bars = bars
        self.seed = seed
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def fetch(self, symbol: str, timeframe: str, period: str = "1y") -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in self._cache:
            # Deterministic per symbol+timeframe, so scanner sees the same history
            seed = self.seed if self.seed is not None else abs(hash(key)) % (2**32)
            self._cache[key] = generate_ohlcv(symbol=symbol, timeframe=timeframe, bars=self.bars, seed=seed)
        return self._cache[key].copy()
