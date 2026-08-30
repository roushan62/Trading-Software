"""Data layer base: canonical OHLCV format + provider interface.

Canonical format
----------------
DataFrame indexed by tz-naive UTC DatetimeIndex with columns:
    open, high, low, close, volume   (float64, lowercase)

Timeframe strings: "1m", "5m", "15m", "1h", "1d".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]

COLS = ["open", "high", "low", "close", "volume"]

HISTORICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "data" / "runtime"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into canonical format."""
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    # Flatten MultiIndex columns (yfinance returns e.g. ('Close','AAPL'))
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0].lower() for c in out.columns]

    rename = {"adj_close": "close", "adjclose": "close"}
    out = out.rename(columns=rename)

    missing = [c for c in COLS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")

    out = out[COLS].astype("float64")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if len(out) == 0:
        raise ValueError("OHLCV data is empty")
    return out


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Sanity checks: high >= max(open,close), low <= min(open,close), volume >= 0."""
    if len(df) == 0:
        raise ValueError("empty OHLCV")
    bad_h = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9).any()
    bad_l = (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9).any()
    bad_v = (df["volume"] < 0).any()
    if bad_h or bad_l:
        raise ValueError("invalid OHLC bars: high/low inconsistent with open/close")
    if bad_v:
        raise ValueError("negative volume found")


def save_csv(df: pd.DataFrame, symbol: str, timeframe: str, directory: Path | None = None) -> Path:
    """Persist canonical OHLCV to data/historical/{SYMBOL}_{TF}.csv."""
    directory = directory or HISTORICAL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    path = directory / f"{safe_symbol}_{timeframe}.csv"
    df.to_csv(path, index_label="datetime")
    return path


class DataProvider(ABC):
    """Interface every data source implements."""

    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, period: str = "1y") -> pd.DataFrame:
        """Return canonical OHLCV data for symbol/timeframe."""
        raise NotImplementedError

    def get_cached_or_fetch(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        return None
