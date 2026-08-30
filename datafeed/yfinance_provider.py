"""yfinance provider — free OHLCV for equities, ETFs, crypto (Yahoo tickers).

Note: requires network access to query1/query2.finance.yahoo.com.
In restricted sandboxes this fails — use CSVProvider or SyntheticProvider.
"""
from __future__ import annotations

import pandas as pd

from .base import DataProvider, normalize

TF_TO_YF = {
    "1m": ("1m", "7d"),    # Yahoo caps 1m history at 7 days
    "5m": ("5m", "60d"),
    "15m": ("15m", "60d"),
    "1h": ("1h", "730d"),
    "1d": ("1d", "max"),
}
TF_DEFAULT_PERIOD = {"1m": "5d", "5m": "30d", "15m": "60d", "1h": "2y", "1d": "5y"}


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:
            raise ImportError("yfinance not installed — pip install yfinance") from e

    def fetch(self, symbol: str, timeframe: str, period: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        if timeframe not in TF_TO_YF:
            raise ValueError(f"unsupported timeframe '{timeframe}', choose from {list(TF_TO_YF)}")
        yf_interval, max_period = TF_TO_YF[timeframe]
        period = period or TF_DEFAULT_PERIOD[timeframe]

        # Respect Yahoo's per-interval history cap
        order = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        if period in order and max_period in order and order.index(period) > order.index(max_period):
            period = max_period

        raw = yf.download(symbol, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned no data for {symbol} {timeframe} (check ticker/network)")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = normalize(raw)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df
