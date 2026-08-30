"""Backtesting engine — event-driven bar simulation with zero lookahead.

Rules
-----
1. A signal is computed on bar i's CLOSE, using only data <= i.
2. Entry fills at bar i+1's OPEN (+ slippage) — never on the signal bar.
3. SL and target are fixed price levels from the signal (bracket-style).
4. If a bar touches BOTH SL and target, the SL is assumed first (conservative).
5. Timeout exit after `max_bars_in_trade` at that bar's close.
6. One position per symbol; `cooldown_bars` must pass after an exit before re-entry.

Results are always reported in R-multiples (risk units) so runs on different
symbols/position sizes are directly comparable.

Backtest results are HISTORICAL. They live in a separate table from paper
trades — never mix the two when judging a system (overfitting bias).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from strategy.base import Signal, load_strategy_config
from strategy.trend_pullback import TrendPullbackStrategy


@dataclass
class BacktestStats:
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    total_r: float = 0.0
    max_drawdown_r: float = 0.0
    max_losing_streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades, "wins": self.wins, "losses": self.losses,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "avg_win_r": round(self.avg_win_r, 3),
            "avg_loss_r": round(self.avg_loss_r, 3),
            "expectancy_r": round(self.expectancy_r, 3),
            "profit_factor": round(self.profit_factor, 2),
            "total_r": round(self.total_r, 2),
            "max_drawdown_r": round(self.max_drawdown_r, 2),
            "max_losing_streak": self.max_losing_streak,
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (
            f"Trades: {d['n_trades']} | Win rate: {d['win_rate_pct']}% | "
            f"AvgWin: +{d['avg_win_r']}R | AvgLoss: {d['avg_loss_r']}R | "
            f"Expectancy: {d['expectancy_r']}R/trade | PF: {d['profit_factor']} | "
            f"Total: {d['total_r']}R | MaxDD: {d['max_drawdown_r']}R | "
            f"Max losing streak: {d['max_losing_streak']}"
        )


@dataclass
class BacktestResult:
    stats: BacktestStats
    trades: pd.DataFrame
    equity_curve: pd.Series
    config: dict
    symbol: str
    timeframe: str

    def analyze_by(self, column: str) -> pd.DataFrame:
        """One-variable-at-a-time analysis: segment results by a trade attribute
        (market_condition, time_of_day, direction, setup, confidence_bucket)."""
        return segment_analysis(self.trades, column)


class Backtester:
    def __init__(self, strategy_config: dict | None = None, fee_pct: float = 0.0, slippage_pct: float = 0.02):
        self.strategy_config = load_strategy_config(strategy_config)
        self.strategy = TrendPullbackStrategy(self.strategy_config)
        self.fee_pct = fee_pct / 100.0
        self.slippage_pct = slippage_pct / 100.0

    # ------------------------------------------------------------------ #
    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signal_filter: Callable[[Signal], bool] | None = None,
    ) -> BacktestResult:
        """Run the strategy over historical OHLCV data.

        signal_filter: optional predicate on each Signal — used for
        one-variable tests like 'only signals between 10:00-12:00'.
        """
        prepared = self.strategy.prepare(df)
        candidates = self.strategy.generate_signals(prepared, symbol, timeframe, source="backtest")
        if signal_filter is not None:
            candidates = [(i, s) for i, s in candidates if signal_filter(s)]

        cfg = self.strategy.cfg
        cooldown = cfg["cooldown_bars"]
        max_bars = cfg["max_bars_in_trade"]

        trades: list[dict] = []
        next_ok_bar = 0          # earliest bar (signal bar) allowed after cooldown
        sig_ptr = 0
        n = len(prepared)

        while sig_ptr < len(candidates):
            sig_bar, sig = candidates[sig_ptr]
            if sig_bar < next_ok_bar:
                sig_ptr += 1
                continue
            if sig_bar + 1 >= n:   # no next bar to fill entry
                break

            entry_bar = sig_bar + 1
            entry_raw = float(prepared["open"].iloc[entry_bar])
            long = sig.direction == "BUY"
            slip = self.slippage_pct
            entry = entry_raw * (1 + slip) if long else entry_raw * (1 - slip)
            sl, target = sig.stop_loss, sig.target
            risk = entry - sl if long else sl - entry
            if risk <= 0:          # gap through the stop at open — skip invalid trade
                sig_ptr += 1
                continue

            exit_price, exit_bar, reason = self._simulate_position(
                prepared, entry_bar, long, sl, target, max_bars
            )
            exit_raw = exit_price
            exit_px = exit_raw * (1 - slip) if long else exit_raw * (1 + slip)
            pnl = (exit_px - entry) if long else (entry - exit_px)
            fees = (self.fee_pct * entry + self.fee_pct * exit_px)
            result_r = (pnl - fees) / risk

            exit_ts = prepared.index[exit_bar]
            entry_ts = prepared.index[entry_bar]
            trades.append({
                "symbol": symbol, "timeframe": timeframe,
                "setup": sig.setup, "direction": sig.direction,
                "confidence": sig.confidence,
                "market_condition": sig.market_condition,
                "signal_time": sig.timestamp, "entry_time": entry_ts,
                "entry": round(entry, 6), "stop_loss": sl, "target": target,
                "exit_time": exit_ts, "exit": round(exit_px, 6),
                "exit_reason": reason,
                "bars_held": exit_bar - entry_bar,
                "risk_per_unit": round(risk, 6),
                "result_r": round(result_r, 4),
                "time_of_day": entry_ts.strftime("%H:%M"),
                "hour": int(entry_ts.hour),
                "day_of_week": entry_ts.strftime("%a"),
                "month": entry_ts.strftime("%Y-%m"),
                "entry_type": self.strategy.cfg["entry_type"],
                "sl_type": self.strategy.cfg["sl_type"],
                "risk_reward": self.strategy.cfg["risk_reward"],
            })
            next_ok_bar = exit_bar + cooldown
            sig_ptr += 1

        trades_df = pd.DataFrame(trades)
        stats = compute_stats(trades_df)
        equity = trades_df["result_r"].cumsum() if len(trades_df) else pd.Series(dtype=float)
        return BacktestResult(stats, trades_df, equity, dict(self.strategy.cfg), symbol, timeframe)

    # ------------------------------------------------------------------ #
    def _simulate_position(
        self, df: pd.DataFrame, entry_bar: int, long: bool,
        sl: float, target: float, max_bars: int,
    ) -> tuple[float, int, str]:
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        n = len(df)

        for b in range(entry_bar, min(n, entry_bar + max_bars)):
            if long:
                if lows[b] <= sl:            # conservative: SL before target
                    return sl, b, "stop_loss"
                if highs[b] >= target:
                    return target, b, "target"
            else:
                if highs[b] >= sl:
                    return sl, b, "stop_loss"
                if lows[b] <= target:
                    return target, b, "target"
        b = min(n - 1, entry_bar + max_bars - 1)
        return float(closes[b]), b, "timeout"


# ---------------------------------------------------------------------- #
def compute_stats(trades_df: pd.DataFrame) -> BacktestStats:
    if len(trades_df) == 0:
        return BacktestStats()
    r = trades_df["result_r"].to_numpy(dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    # max drawdown on the R equity curve
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0

    # max losing streak
    streak = max_streak = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        max_streak = max(max_streak, streak)

    return BacktestStats(
        n_trades=len(r),
        wins=len(wins), losses=len(losses),
        win_rate=len(wins) / len(r),
        avg_win_r=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_r=float(losses.mean()) if len(losses) else 0.0,
        expectancy_r=float(r.mean()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        total_r=float(r.sum()),
        max_drawdown_r=max_dd,
        max_losing_streak=max_streak,
    )


def segment_analysis(trades_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Stats grouped by one trade attribute — for one-variable-at-a-time study."""
    if len(trades_df) == 0 or column not in trades_df.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in trades_df.groupby(column):
        s = compute_stats(grp)
        rows.append({column: key, **s.to_dict()})
    return pd.DataFrame(rows).sort_values("expectancy_r", ascending=False).reset_index(drop=True)


def sweep_variable(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_config: dict,
    variable: str,
    values: list[Any],
    fee_pct: float = 0.0,
    slippage_pct: float = 0.02,
) -> pd.DataFrame:
    """One-variable-at-a-time parameter sweep.

    Only `variable` changes across runs; everything else stays at base_config.
    Returns a comparison table so you can see whether an 'improvement' is
    robust or curve-fit to one lucky value.
    """
    rows = []
    for val in values:
        cfg = {**base_config, variable: val}
        try:
            bt = Backtester(cfg, fee_pct=fee_pct, slippage_pct=slippage_pct)
            res = bt.run(df, symbol, timeframe)
            rows.append({variable: val, **res.stats.to_dict()})
        except Exception as e:
            rows.append({variable: val, "error": str(e)})
    return pd.DataFrame(rows)


DISCLAIMER = "Backtest = historical simulation only. Not financial advice. Based on historical statistical edge, not a guarantee."
