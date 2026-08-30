"""Unit tests: backtester mechanics — fills, R math, stats, no lookahead."""
import numpy as np
import pandas as pd
import pytest

from datafeed.synthetic import generate_ohlcv
from engine.backtester import Backtester, compute_stats, sweep_variable
from strategy import DEFAULT_CONFIG


def handcrafted_df():
    """Deterministic price path to verify fill/exit mechanics exactly."""
    rows = []
    base = 100.0
    # bar: open, high, low, close
    path = [
        (100.0, 100.5, 99.5, 100.0),   # 0
        (100.0, 100.5, 99.5, 100.0),   # 1 ... flat warmup
    ]
    for _ in range(240):
        path.append((100.0, 100.5, 99.5, 100.0))
    # signal bar: strong up-close breakout conditions won't matter — we test
    # simulate via a direct call instead; here we just need valid data
    for _ in range(20):
        path.append((101.0, 103.0, 100.8, 102.5))
    idx = pd.date_range("2024-01-01", periods=len(path), freq="1h")
    df = pd.DataFrame(path, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    return df


class TestSimulatePosition:
    def test_stop_hit_first_when_both_touched(self):
        bt = Backtester()
        df = handcrafted_df()
        # long from bar 5: SL 99 (touched: lows are 99.5 -> no) — craft explicitly
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({
            "open": [100, 100, 100], "high": [100.2, 103.0, 100.2],
            "low": [100.0, 98.0, 100.0], "close": [100.1, 102.0, 100.1],
            "volume": [1, 1, 1],
        }, index=idx)
        # entry bar 0; on bar 1 both SL=99 and target=102.5 are inside the range
        px, bar, reason = bt._simulate_position(df, 1, True, sl=99.0, target=102.5, max_bars=10)
        assert reason == "stop_loss" and px == 99.0 and bar == 1

    def test_target_hit(self):
        bt = Backtester()
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({
            "open": [100, 100, 100], "high": [100.2, 104.0, 100.2],
            "low": [99.9, 100.0, 99.9], "close": [100.1, 103.5, 100.1],
            "volume": [1, 1, 1],
        }, index=idx)
        px, bar, reason = bt._simulate_position(df, 1, True, sl=95.0, target=103.0, max_bars=10)
        assert reason == "target" and px == 103.0

    def test_timeout(self):
        bt = Backtester()
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "open": [100.0] * 10, "high": [100.1] * 10, "low": [99.9] * 10,
            "close": [100.0] * 10, "volume": [1] * 10,
        }, index=idx)
        px, bar, reason = bt._simulate_position(df, 1, True, sl=95.0, target=105.0, max_bars=4)
        assert reason == "timeout" and bar == 4


class TestStats:
    def test_stats_math(self):
        rs = [1.0, 1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 3.0]
        df = pd.DataFrame({"result_r": rs})
        s = compute_stats(df)
        assert s.n_trades == 8
        assert s.wins == 4 and s.losses == 4
        assert abs(s.win_rate - 0.5) < 1e-9
        assert abs(s.avg_win_r - (1 + 1 + 2 + 3) / 4) < 1e-9
        assert abs(s.avg_loss_r - (-1.0)) < 1e-9
        assert abs(s.expectancy_r - (7.0 - 4.0) / 8) < 1e-9
        assert abs(s.profit_factor - 7.0 / 4.0) < 1e-9
        assert abs(s.total_r - 3.0) < 1e-9
        assert s.max_losing_streak == 3
        # equity: 1,2,1,3,2,1,0,3 -> maxDD from peak 3 to 0 = 3
        assert abs(s.max_drawdown_r - 3.0) < 1e-9

    def test_empty(self):
        s = compute_stats(pd.DataFrame())
        assert s.n_trades == 0


class TestBacktestRun:
    def test_100_plus_trades_and_mode_separation(self):
        df = generate_ohlcv("SYMB", "1h", bars=9000, seed=23)
        bt = Backtester()
        res = bt.run(df, "SYMB", "1h")
        assert res.stats.n_trades >= 100
        # every trade has SL bracket consistent with direction
        t = res.trades
        longs = t[t.direction == "BUY"]
        shorts = t[t.direction == "SELL"]
        assert (longs["stop_loss"] < longs["entry"]).all()
        assert (longs["entry"] < longs["target"]).all()
        assert (shorts["stop_loss"] > shorts["entry"]).all()
        assert (shorts["entry"] > shorts["target"]).all()
        # entry happens on the bar AFTER the signal bar (no lookahead fills)
        sig_t = pd.to_datetime(t["signal_time"])
        ent_t = pd.to_datetime(t["entry_time"])
        assert (ent_t > sig_t).all()

    def test_sweep_one_variable(self):
        df = generate_ohlcv("SYMB", "1h", bars=4000, seed=29)
        tbl = sweep_variable(df, "SYMB", "1h", DEFAULT_CONFIG, "risk_reward", [1.0, 2.0])
        assert len(tbl) == 2
        assert set(tbl["risk_reward"]) == {1.0, 2.0}
        assert (tbl["n_trades"] > 0).all()
