"""Unit tests: data layer (normalize, resample, providers, synthetic validity)."""
import numpy as np
import pandas as pd
import pytest

from datafeed.csv_provider import CSVProvider
from datafeed.resample import resample_ohlcv
from datafeed.synthetic import SyntheticProvider, generate_ohlcv
from datafeed.base import normalize, save_csv, validate_ohlcv


class TestNormalize:
    def test_lowercase_and_sort(self):
        idx = pd.date_range("2024-01-02", periods=4, freq="1D")
        df = pd.DataFrame({
            "Close": [101.0, 100.0, 103.0, 102.0],
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [102.0, 101.5, 104.0, 103.5],
            "Low": [99.0, 99.5, 101.0, 101.5],
            "Volume": [10, 11, 12, 13],
        }, index=idx[::-1])  # unsorted
        out = normalize(df)
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]
        assert out.index.is_monotonic_increasing

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError):
            normalize(df)


class TestSynthetic:
    def test_valid_ohlcv(self):
        df = generate_ohlcv("A", "1h", bars=800, seed=1)
        validate_ohlcv(df)
        assert len(df) == 800

    def test_deterministic(self):
        a = generate_ohlcv("A", "1h", bars=500, seed=42)
        b = generate_ohlcv("A", "1h", bars=500, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_contains_trends(self):
        """Structure classifier should find real trends, not pure noise."""
        df = generate_ohlcv("A", "1h", bars=3000, seed=3)
        from indicators.market_structure import market_structure_series
        labels = market_structure_series(df, k=3).value_counts()
        assert labels.get("uptrend", 0) > 100
        assert labels.get("downtrend", 0) > 100


class TestResample:
    def test_hourly_to_daily(self):
        df = generate_ohlcv("A", "1h", bars=24 * 10, seed=4)
        daily = resample_ohlcv(df, "1d")
        assert len(daily) == 10
        # daily high must equal max of that day's highs
        first_day = df[df.index.date == daily.index[0].date()]
        assert daily.iloc[0]["high"] == pytest.approx(first_day["high"].max())
        assert daily.iloc[0]["close"] == pytest.approx(first_day["close"].iloc[-1])
        assert daily.iloc[0]["volume"] == pytest.approx(first_day["volume"].sum())


class TestCSVProvider:
    def test_roundtrip(self, tmp_path):
        df = generate_ohlcv("ZZ", "1h", bars=300, seed=6)
        path = save_csv(df, "ZZ", "1h", directory=tmp_path)
        assert path.exists()
        loaded = CSVProvider(directory=tmp_path).fetch("ZZ", "1h")
        assert len(loaded) == 300
        validate_ohlcv(loaded)


class TestSyntheticProvider:
    def test_cache_consistent(self):
        p = SyntheticProvider(bars=400)
        a = p.fetch("S", "1h")
        b = p.fetch("S", "1h")
        pd.testing.assert_frame_equal(a, b)
