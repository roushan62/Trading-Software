"""CSV provider — loads locally stored OHLCV history (data/historical/).

Expected file: data/historical/{SYMBOL}_{TIMEFRAME}.csv with a datetime index
column and open/high/low/close/volume columns (case-insensitive).
The first fetch also persists a local cache so backtests are reproducible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import COLS, HISTORICAL_DIR, DataProvider, normalize, validate_ohlcv


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory) if directory else HISTORICAL_DIR

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "-")
        return self.directory / f"{safe}_{timeframe}.csv"

    def fetch(self, symbol: str, timeframe: str, period: str = "1y") -> pd.DataFrame:
        path = self._path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"no local CSV for {symbol} {timeframe} at {path}. "
                "Run `python main.py fetch ...` or place your own file there."
            )
        df = pd.read_csv(path, index_col=0)
        df = normalize(df)
        if not all(c in df.columns for c in COLS):
            raise ValueError(f"{path}: missing OHLCV columns")
        validate_ohlcv(df)
        return df

    def has(self, symbol: str, timeframe: str) -> bool:
        return self._path(symbol, timeframe).exists()
