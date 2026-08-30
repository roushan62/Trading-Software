"""Unit tests: indicator engine (math correctness, causality)."""
import numpy as np
import pandas as pd
import pytest

from indicators.atr import atr
from indicators.ema import ema
from indicators.market_structure import classify_swings, market_structure_series
from indicators.rsi import rsi
from indicators.support_resistance import Swing, find_swings, last_swing_low_before
from indicators.volume import volume_zscore
from indicators.vwap import vwap
from datafeed.synthetic import generate_ohlcv


def make_df(closes, start="2024-01-01", freq="1h"):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) * 1.001
    low = np.minimum(opens, closes) * 0.999
    vol = np.full(len(closes), 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes, "volume": vol},
        index=pd.date_range(start, periods=len(closes), freq=freq),
    )


class TestEMA:
    def test_known_value(self):
        # EMA with span=3 on a simple series — verify against manual computation
        s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        alpha = 2 / (3 + 1)
        manual = [10.0]
        for v in s[1:]:
            manual.append(alpha * v + (1 - alpha) * manual[-1])
        out = ema(s, 3)
        assert np.allclose(out.to_numpy(), manual)

    def test_causal(self):
        s = pd.Series(np.random.default_rng(1).normal(100, 1, 300))
        full = ema(s, 20)
        trunc = ema(s.iloc[:150], 20)
        assert abs(full.iloc[149] - trunc.iloc[-1]) < 1e-12

    def test_constant_series(self):
        s = pd.Series([5.0] * 50)
        assert np.allclose(ema(s, 10), 5.0)


class TestRSI:
    def test_bounds(self):
        rng = np.random.default_rng(2)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
        out = rsi(s, 14)
        assert out.iloc[:14].isna().all()
        vals = out.dropna()
        assert ((vals >= 0) & (vals <= 100)).all()

    def test_monotonic_up_gives_high_rsi(self):
        s = pd.Series(np.linspace(100, 200, 100))
        out = rsi(s, 14)
        assert out.iloc[-1] > 90

    def test_flat_gives_50(self):
        s = pd.Series([100.0] * 60)
        out = rsi(s, 14)
        assert out.iloc[-1] == 50.0


class TestATR:
    def test_positive_and_zero_for_flat(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="1h")
        df = pd.DataFrame({
            "open": [100.0] * 30, "high": [100.0] * 30, "low": [100.0] * 30,
            "close": [100.0] * 30, "volume": [1.0] * 30,
        }, index=idx)
        out = atr(df, 14)
        assert (out.dropna() >= 0).all()
        assert out.iloc[-1] < 1e-9

    def test_scales_with_range(self):
        small = atr(make_df([100, 101, 99, 100] * 10), 14).iloc[-1]
        big = atr(make_df([100, 110, 90, 100] * 10), 14).iloc[-1]
        assert big > small * 5


class TestVWAP:
    def test_session_reset(self):
        # 12 hourly bars within one calendar day; typical price == close
        idx = pd.date_range("2024-01-01 09:00", periods=12, freq="1h")
        closes = np.tile([100.0, 104.0], 6)
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes,
            "volume": np.tile([100.0, 300.0], 6),
        }, index=idx)
        out = vwap(df, intraday=True)
        # final cumulative VWAP = (6*100*100 + 6*104*300) / (6*100 + 6*300) = 103
        assert abs(out.iloc[-1] - 103.0) < 1e-9

    def test_resets_next_day(self):
        idx = pd.date_range("2024-01-01 09:00", periods=24, freq="1h")
        closes = np.tile([100.0, 104.0], 12)
        vol = np.tile([100.0, 300.0], 12)
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": vol,
        }, index=idx)
        out = vwap(df, intraday=True)
        day1_last = out[out.index.date == idx[0].date()].iloc[-1]
        day2_first = out[out.index.date == idx[12].date()].iloc[0]
        assert abs(day2_first - day1_last) > 0.1  # anchored fresh at the session open

    def test_no_volume_falls_back(self):
        df = make_df([100, 102, 101, 103])
        df["volume"] = 0.0
        out = vwap(df, intraday=True)
        assert not out.isna().any()


class TestSwings:
    def test_detects_obvious_pivot(self):
        closes = [10, 10, 10, 10, 10, 10, 10]
        df = make_df(closes)
        df.iloc[3, df.columns.get_loc("high")] = 12.0   # spike up at bar 3
        df.iloc[3, df.columns.get_loc("low")] = 9.5
        swings = find_swings(df, k=2)
        highs = [s for s in swings if s.kind == "high"]
        assert len(highs) == 1 and highs[0].index == 3 and highs[0].price == 12.0

    def test_last_swing_low(self):
        df = generate_ohlcv("T", "1h", bars=400, seed=5)
        val = last_swing_low_before(df, len(df) - 1, lookback=50)
        assert val is None or val <= df["low"].tail(50).min() + 1e-9


class TestStructure:
    def test_uptrend(self):
        # HH + HL swing sequence -> uptrend
        swings = [
            Swing(10, pd.Timestamp("2024-01-01 10:00"), 105.0, "high"),
            Swing(14, pd.Timestamp("2024-01-01 14:00"), 100.0, "low"),
            Swing(20, pd.Timestamp("2024-01-01 20:00"), 110.0, "high"),   # higher high
            Swing(26, pd.Timestamp("2024-01-02 02:00"), 103.0, "low"),    # higher low
        ]
        assert classify_swings(swings) == "uptrend"

    def test_downtrend(self):
        swings = [
            Swing(10, pd.Timestamp("2024-01-01 10:00"), 110.0, "high"),
            Swing(14, pd.Timestamp("2024-01-01 14:00"), 104.0, "low"),
            Swing(20, pd.Timestamp("2024-01-01 20:00"), 105.0, "high"),   # lower high
            Swing(26, pd.Timestamp("2024-01-02 02:00"), 100.0, "low"),    # lower low
        ]
        assert classify_swings(swings) == "downtrend"

    def test_range(self):
        swings = [
            Swing(10, pd.Timestamp("2024-01-01 10:00"), 110.0, "high"),
            Swing(14, pd.Timestamp("2024-01-01 14:00"), 100.0, "low"),
            Swing(20, pd.Timestamp("2024-01-01 20:00"), 108.0, "high"),   # lower high
            Swing(26, pd.Timestamp("2024-01-02 02:00"), 103.0, "low"),    # higher low -> mixed
        ]
        assert classify_swings(swings) == "range"

    def test_series_labels_valid(self):
        df = generate_ohlcv("T", "1h", bars=600, seed=9)
        s = market_structure_series(df, k=3)
        assert set(s.dropna().unique()) <= {"uptrend", "downtrend", "range"}


class TestVolumeZ:
    def test_spike_positive(self):
        vol = pd.Series(np.concatenate([np.full(50, 100.0), [10000.0]]))
        z = volume_zscore(vol, 20)
        assert z.iloc[-1] > 3
