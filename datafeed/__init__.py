"""Data layer facade.

Resolution order for `get_provider()`:
    1. explicit choice via argument ("yfinance" | "csv" | "synthetic")
    2. csv  — if data/historical/{SYMBOL}_{TF}.csv already exists
    3. yfinance — try live download, fall back to synthetic on network failure

Every fetch is cached to data/historical/ so backtests run on identical data.
"""
from __future__ import annotations

import pandas as pd

from .base import COLS, HISTORICAL_DIR, TIMEFRAMES, DataProvider, normalize, save_csv, validate_ohlcv
from .csv_provider import CSVProvider
from .resample import resample_ohlcv
from .suppressed import SuppressedStderr
from .synthetic import SyntheticProvider, generate_ohlcv
from .yfinance_provider import YFinanceProvider

__all__ = [
    "DataProvider", "CSVProvider", "YFinanceProvider", "SyntheticProvider",
    "generate_ohlcv", "resample_ohlcv", "normalize", "validate_ohlcv",
    "save_csv", "fetch_ohlcv", "TIMEFRAMES", "COLS", "HISTORICAL_DIR",
    "data_source_of",
]


def get_provider(kind: str | None = None, **kwargs) -> DataProvider:
    if kind == "yfinance":
        return YFinanceProvider()
    if kind == "csv":
        return CSVProvider()
    if kind == "synthetic":
        return SyntheticProvider(**kwargs)
    return AutoProvider(**kwargs)


class AutoProvider(DataProvider):
    """csv cache -> yfinance -> synthetic (with clear warnings)."""

    name = "auto"

    def __init__(self, bars: int = 3000):
        self.bars = bars
        self._csv = CSVProvider()
        self._synth = SyntheticProvider(bars=bars)
        self.last_source: str | None = None

    def fetch(self, symbol: str, timeframe: str, period: str = "1y") -> pd.DataFrame:
        if self._csv.has(symbol, timeframe):
            self.last_source = _read_sources().get(f"{symbol}_{timeframe}", "csv")
            return self._csv.fetch(symbol, timeframe, period)
        try:
            with SuppressedStderr():
                yf_provider = YFinanceProvider()
                df = yf_provider.fetch(symbol, timeframe, period)
            save_csv(df, symbol, timeframe)
            _mark_source(symbol, timeframe, "yfinance")
            self.last_source = "yfinance"
            return df
        except Exception:
            print(
                f"[datafeed] WARNING: live fetch failed for {symbol} {timeframe}; "
                "using SYNTHETIC data (demo only — not real market data)."
            )
            df = self._synth.fetch(symbol, timeframe, period)
            save_csv(df, symbol, timeframe)
            _mark_source(symbol, timeframe, "synthetic")
            self.last_source = "synthetic"
            return df


SOURCES_PATH = HISTORICAL_DIR / ".sources.json"


def _read_sources() -> dict:
    import json

    try:
        if SOURCES_PATH.exists():
            return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _mark_source(symbol: str, timeframe: str, source: str) -> None:
    import json

    try:
        data = _read_sources()
        data[f"{symbol}_{timeframe}"] = source
        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
        SOURCES_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def data_source_of(symbol: str, timeframe: str) -> str:
    """What produced the cached data: 'yfinance' | 'synthetic' | 'csv' (unknown/manual)."""
    return _read_sources().get(f"{symbol}_{timeframe}", "csv")


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    provider: str | None = None,
    period: str = "1y",
    cache: bool = True,
) -> pd.DataFrame:
    """One-call data access with local caching to data/historical/."""
    p = get_provider(provider)
    df = p.fetch(symbol, timeframe, period=period)
    if cache and provider not in (None, "auto") or (cache and p.name in ("yfinance", "synthetic", "auto")):
        if not CSVProvider().has(symbol, timeframe):
            save_csv(df, symbol, timeframe)
    return df
