"""Unit tests: risk management + journal (mode separation, limits)."""
from datetime import datetime

import pandas as pd
import pytest

from journal.logger import Journal
from risk import RiskConfig, RiskGuard, position_size
from strategy.base import Signal


def make_signal(**kw):
    d = dict(symbol="X", direction="BUY", entry=100.0, stop_loss=98.0, target=104.0,
             timeframe="1h", confidence=80, setup="trend_pullback:pullback")
    d.update(kw)
    return Signal(**d)


class TestPositionSizing:
    def test_fixed_pct_risk(self):
        s = position_size(entry=100, stop_loss=98, account_size=10000, risk_per_trade_pct=1.0)
        assert s.risk_amount == pytest.approx(100.0)
        assert s.qty == pytest.approx(50.0)
        assert s.position_value == pytest.approx(5000.0)

    def test_zero_risk_rejected(self):
        with pytest.raises(ValueError):
            position_size(entry=100, stop_loss=100, account_size=10000, risk_per_trade_pct=1.0)

    def test_no_leverage_cap(self):
        # risk 10% with tight stop would need huge size -> capped by account value
        s = position_size(entry=100, stop_loss=99.99, account_size=10000, risk_per_trade_pct=10.0)
        assert s.position_value <= 10000.0 + 1e-6
        assert any("capped" in n for n in s.notes)

    def test_lot_rounding(self):
        # account 500, risk 0.5% = $2.5, risk/unit $1 -> qty 2.5 -> rounds DOWN to 2
        s = position_size(entry=100, stop_loss=99, account_size=500, risk_per_trade_pct=0.5,
                          round_to=1.0)
        assert s.qty == 2.0
        assert s.risk_amount == 2.0
        # account 100, risk 0.5% = $0.5 -> qty 0.5 -> rounds to 0 -> skip trade
        tiny = position_size(entry=100, stop_loss=99, account_size=100, risk_per_trade_pct=0.5,
                             round_to=1.0)
        assert tiny.qty == 0.0
        assert any("0" in n for n in tiny.notes)


class TestRiskGuard:
    def test_max_trades_per_day(self):
        g = RiskGuard(RiskConfig(max_trades_per_day=2, account_size=10000, max_daily_loss_pct=50))
        now = datetime(2024, 1, 10, 10)
        assert g.can_open(when=now).allowed
        g.register_open(when=now)
        g.register_open(when=now)
        d = g.can_open(when=now)
        assert not d.allowed and "max trades/day" in d.reason

    def test_daily_loss_limit_blocks(self):
        g = RiskGuard(RiskConfig(account_size=10000, max_daily_loss_pct=2.0))
        now = datetime(2024, 1, 10, 11)
        g.register_open(when=now)
        g.register_close(-250.0, when=now)   # -2.5% > 2% limit
        d = g.can_open(when=now)
        assert not d.allowed and "daily loss limit" in d.reason

    def test_next_day_resets(self):
        g = RiskGuard(RiskConfig(max_trades_per_day=1, account_size=10000, max_daily_loss_pct=50))
        d1 = datetime(2024, 1, 10, 10)
        d2 = datetime(2024, 1, 11, 10)
        g.register_open(when=d1)
        assert not g.can_open(when=d1).allowed
        assert g.can_open(when=d2).allowed   # rolled to a new day


class TestJournal:
    def test_roundtrip_and_r_math(self, tmp_path):
        j = Journal(tmp_path / "j.db")
        sig = make_signal(timestamp=pd.Timestamp("2024-01-05 10:00"))
        tid = j.open_paper_trade(sig, qty=50, risk_amount=100.0,
                                 entry_time=pd.Timestamp("2024-01-05 10:00"))
        assert j.has_open_trade("X", "1h", mode="paper")
        res = j.close_trade(tid, 104.0, pd.Timestamp("2024-01-05 15:00"), "target")
        assert res["result_r"] == pytest.approx(2.0)      # (104-100)/2
        assert res["pnl_amount"] == pytest.approx(200.0)  # 4 * 50
        assert not j.has_open_trade("X", "1h", mode="paper")

    def test_cannot_close_twice(self, tmp_path):
        j = Journal(tmp_path / "j.db")
        sig = make_signal(timestamp=pd.Timestamp("2024-01-05 10:00"))
        tid = j.open_paper_trade(sig, 10, 20.0)
        j.close_trade(tid, 99.0, pd.Timestamp("2024-01-05 12:00"), "stop_loss")
        with pytest.raises(ValueError):
            j.close_trade(tid, 99.0, pd.Timestamp("2024-01-05 13:00"), "stop_loss")

    def test_mode_separation(self, tmp_path):
        """SPEC RULE: backtest and paper results must never blend."""
        j = Journal(tmp_path / "j.db")
        sig = make_signal(timestamp=pd.Timestamp("2024-01-05 10:00"))
        j.open_paper_trade(sig, 10, 20.0)
        bt = pd.DataFrame([{
            "symbol": "X", "timeframe": "1h", "setup": "s", "direction": "BUY",
            "confidence": 80, "market_condition": "uptrend",
            "signal_time": "2024-01-05 09:00", "entry_time": "2024-01-05 10:00",
            "entry": 100.0, "stop_loss": 98.0, "target": 104.0, "exit_time": "2024-01-05 15:00",
            "exit": 104.0, "exit_reason": "target", "bars_held": 5,
            "risk_per_unit": 2.0, "result_r": 2.0, "time_of_day": "10:00", "hour": 10,
            "day_of_week": "Fri", "month": "2024-01", "entry_type": "pullback",
            "sl_type": "structure", "risk_reward": 2.0,
        }])
        j.record_backtest_trades(bt)
        paper = j.all_trades(mode="paper")
        back = j.all_trades(mode="backtest")
        assert len(paper) == 1 and paper.iloc[0]["status"] == "open"
        assert len(back) == 1 and back.iloc[0]["result_r"] == 2.0
        # resetting paper never touches backtest history
        j.reset_paper()
        assert len(j.all_trades(mode="paper")) == 0
        assert len(j.all_trades(mode="backtest")) == 1

    def test_report_periods(self, tmp_path):
        j = Journal(tmp_path / "j.db")
        for day, r in ((5, 2.0), (6, -1.0), (20, 1.0)):
            sig = make_signal(timestamp=pd.Timestamp(f"2024-01-{day:02d} 10:00"))
            tid = j.open_paper_trade(sig, 10, 20.0, entry_time=pd.Timestamp(f"2024-01-{day:02d} 10:00"))
            exit_px = sig.entry + r * abs(sig.entry - sig.stop_loss)
            j.close_trade(tid, exit_px, pd.Timestamp(f"2024-01-{day:02d} 16:00"),
                          "target" if r > 0 else "stop_loss")
        rep = j.report("monthly", mode="paper")
        assert len(rep) == 1 and rep.iloc[0]["n_trades"] == 3
        rep_w = j.report("weekly", mode="paper")
        assert len(rep_w) >= 2  # Jan 5/6 and Jan 20 are different ISO weeks
