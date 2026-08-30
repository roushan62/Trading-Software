"""Unit tests: strategy + signal invariants (SL always present, causality)."""
import numpy as np
import pandas as pd
import pytest

from datafeed.synthetic import generate_ohlcv
from strategy import TrendPullbackStrategy, load_strategy_config
from strategy.base import Signal


class TestSignalInvariant:
    def test_signal_requires_bracket(self):
        with pytest.raises(ValueError):
            Signal(symbol="X", direction="BUY", entry=100, stop_loss=100, target=110,
                   timeframe="1h", confidence=80, setup="t")

    def test_buy_requires_sl_below_entry(self):
        with pytest.raises(ValueError):
            Signal(symbol="X", direction="BUY", entry=100, stop_loss=105, target=110,
                   timeframe="1h", confidence=80, setup="t")

    def test_disclaimer_attached(self):
        s = Signal(symbol="X", direction="BUY", entry=100, stop_loss=98, target=104,
                   timeframe="1h", confidence=80, setup="t")
        assert "Not financial advice" in s.alert_text()
        assert "Not financial advice" in s.to_dict()["disclaimer"]


class TestStrategyConfig:
    def test_invalid_entry_type_rejected(self):
        with pytest.raises(ValueError):
            load_strategy_config({"entry_type": "moonshot"})

    def test_invalid_sl_type_rejected(self):
        with pytest.raises(ValueError):
            load_strategy_config({"sl_type": "vibes"})

    def test_merge_overrides(self):
        cfg = load_strategy_config({"risk_reward": 3.0, "confidence_weights": {"trend_alignment": 40}})
        assert cfg["risk_reward"] == 3.0
        assert cfg["confidence_weights"]["trend_alignment"] == 40
        assert cfg["confidence_weights"]["vwap_position"] == 15  # untouched


class TestStrategyCausality:
    def test_no_lookahead(self):
        """Signal at bar i must be identical whether or not future bars exist."""
        df = generate_ohlcv("T", "1h", bars=1200, seed=11)
        strat = TrendPullbackStrategy()
        prepared = strat.prepare(df)
        i = len(df) - 200
        sig_full = strat.check_signal(prepared, i, "T", "1h")
        sig_trunc = strat.check_signal(strat.prepare(df.iloc[: i + 1]), i, "T", "1h")
        assert (sig_full is None) == (sig_trunc is None)
        if sig_full is not None:
            assert sig_full.entry == sig_trunc.entry
            assert sig_full.stop_loss == sig_trunc.stop_loss
            assert sig_full.confidence == sig_trunc.confidence

    def test_all_signals_have_stops(self):
        df = generate_ohlcv("T", "1h", bars=3000, seed=13)
        strat = TrendPullbackStrategy()
        for i, s in strat.generate_signals(df, "T", "1h"):
            assert s.stop_loss is not None and s.risk > 0
            assert 0 < s.confidence <= 100
            assert s.rr > 0

    def test_warmup_respected(self):
        df = generate_ohlcv("T", "1h", bars=300, seed=17)
        strat = TrendPullbackStrategy()
        prepared = strat.prepare(df)
        for i in range(strat.warmup_bars):
            assert strat.check_signal(prepared, i, "T", "1h") is None

    def test_atr_sl_mode(self):
        df = generate_ohlcv("T", "1h", bars=3000, seed=19)
        strat = TrendPullbackStrategy({"sl_type": "atr", "atr_sl_mult": 2.0})
        sigs = strat.generate_signals(df, "T", "1h")
        prepared = strat.prepare(df)
        for i, s in sigs:
            atr_v = float(prepared["atr"].iloc[i])
            expected = s.entry - 2.0 * atr_v if s.direction == "BUY" else s.entry + 2.0 * atr_v
            assert abs(s.stop_loss - expected) < 1e-6
