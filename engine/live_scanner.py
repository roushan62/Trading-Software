"""Live scanner + paper-trading (forward test) engine.

Modes
-----
scan_once   : fetch latest data, evaluate the last CLOSED bar, open paper
              trades on new signals (if risk limits allow), manage open trades.
run_loop    : scan_once on a timer (APScheduler-style loop; plain sleep).
replay      : forward-test on held-out historical data — walks bars one by one
              through the EXACT paper-trading pipeline (journal + risk guard +
              alerts), so forward results are produced the same way live ones
              will be. Uses data AFTER the backtest window for honesty.

This engine NEVER places real orders. It only writes paper trades to the
journal and sends alerts. Manual confirmation is always required for real
execution — by design.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from alerts import Notifier
from datafeed.base import DataProvider
from journal.logger import Journal
from risk import RiskGuard, RiskConfig
from strategy.base import Signal
from strategy.trend_pullback import TrendPullbackStrategy


class LiveScanner:
    def __init__(
        self,
        strategy_config: dict | None = None,
        provider: DataProvider | None = None,
        journal: Journal | None = None,
        notifier: Notifier | None = None,
        risk_config: dict | None = None,
        alerts_config: dict | None = None,
    ):
        self.strategy = TrendPullbackStrategy(strategy_config)
        self.provider = provider
        self.journal = journal or Journal()
        self.notifier = notifier or Notifier(alerts_config, journal=self.journal)
        self.risk = RiskGuard(RiskConfig.from_dict(risk_config))

    # ------------------------------------------------------------------ #
    def scan_once(
        self,
        symbols: list[str],
        timeframes: list[str],
        drop_last_bar: bool = True,
        verbose: bool = True,
    ) -> list[Signal]:
        """One scan pass over the watchlist. Returns signals that were taken.

        drop_last_bar=True skips the newest (possibly still-forming) bar so
        signals are only evaluated on CLOSED candles.
        """
        fired: list[Signal] = []
        self._manage_open_trades(symbols, timeframes)

        for symbol in symbols:
            for tf in timeframes:
                try:
                    df = self.provider.fetch(symbol, tf)
                except Exception as e:
                    if verbose:
                        print(f"[scan] {symbol} {tf}: data unavailable ({e})")
                    continue
                if drop_last_bar:
                    df = df.iloc[:-1]
                prepared = self.strategy.prepare(df)
                if len(prepared) <= self.strategy.warmup_bars:
                    continue

                sig = self.strategy.check_signal(
                    prepared, len(prepared) - 1, symbol, tf, source="paper"
                )
                if sig is None:
                    continue
                fired.append(sig)
                self._maybe_open_paper_trade(sig, df.index[-1])
        return fired

    # ------------------------------------------------------------------ #
    def _maybe_open_paper_trade(self, sig: Signal, bar_ts, verbose: bool = True) -> bool:
        # duplicate guard: one open paper position per symbol+timeframe
        if self.journal.has_open_trade(sig.symbol, sig.timeframe, mode="paper"):
            if verbose:
                print(f"[scan] {sig.symbol} {sig.timeframe}: signal skipped (position already open)")
            return False

        decision = self.risk.can_open(sig)
        if not decision.allowed:
            print(f"[risk] BLOCKED {sig.direction} {sig.symbol}: {decision.reason}")
            return False

        size = self.risk.size_for_signal(sig)
        if size.qty <= 0:
            print(f"[risk] {sig.symbol}: position size rounds to 0 — skipped ({size.notes})")
            return False

        self.journal.log_signal(sig, mode="paper", alerted=True)
        trade_id = self.journal.open_paper_trade(sig, size.qty, size.risk_amount, entry_time=bar_ts)
        self.risk.register_open()
        print(
            f"[paper] OPEN #{trade_id} {sig.direction} {sig.symbol} {sig.timeframe} "
            f"qty={size.qty:g} entry={sig.entry:.2f} SL={sig.stop_loss:.2f} "
            f"target={sig.target:.2f} risk=${size.risk_amount:.2f} conf={sig.confidence:.0f}"
        )
        self.notifier.send_signal(sig, source="paper")
        return True

    # ------------------------------------------------------------------ #
    def _manage_open_trades(self, symbols: list[str], timeframes: list[str]) -> None:
        """Check open paper trades against latest bars; close on SL/target/timeout."""
        open_df = self.journal.open_trades(mode="paper")
        if len(open_df) == 0:
            return
        for _, t in open_df.iterrows():
            symbol, tf = t["symbol"], t["timeframe"]
            try:
                df = self.provider.fetch(symbol, tf)
            except Exception:
                continue
            df = df[df.index > pd.to_datetime(t["entry_time"])]
            if len(df) == 0:
                continue
            self._simulate_paper_trade(t, df)

    def _simulate_paper_trade(self, trade: pd.Series, df: pd.DataFrame) -> None:
        long = trade["direction"] == "BUY"
        sl, target = float(trade["stop_loss"]), float(trade["target"])
        max_bars = self.strategy.cfg["max_bars_in_trade"]
        entry_time = pd.to_datetime(trade["entry_time"])

        for b, (ts, row) in enumerate(df.iterrows()):
            hit_sl = row["low"] <= sl if long else row["high"] >= sl
            hit_tgt = row["high"] >= target if long else row["low"] <= target
            if hit_sl:      # conservative: SL first when both touched
                res = self.journal.close_trade(trade["id"], sl, ts, "stop_loss")
                self._on_close(trade, res, sl, ts, "STOP-LOSS")
                return
            if hit_tgt:
                res = self.journal.close_trade(trade["id"], target, ts, "target")
                self._on_close(trade, res, target, ts, "TARGET")
                return
            if b + 1 >= max_bars:
                res = self.journal.close_trade(trade["id"], float(row["close"]), ts, "timeout")
                self._on_close(trade, res, float(row["close"]), ts, "TIMEOUT")
                return

    def _on_close(self, trade, res, exit_price, ts, label) -> None:
        self.risk.register_close(res["pnl_amount"])
        print(
            f"[paper] CLOSE #{trade['id']} {trade['direction']} {trade['symbol']} "
            f"@{exit_price:.2f} ({label}) result={res['result_r']:+.2f}R "
            f"pnl=${res['pnl_amount']:+.2f}"
        )
        self.notifier.send(
            f"CLOSED {trade['direction']} {trade['symbol']} [{trade['timeframe']}] #{trade['id']}\n"
            f"Exit: {exit_price:.2f} ({label})\n"
            f"Result: {res['result_r']:+.2f}R  |  PnL: ${res['pnl_amount']:+.2f}\n"
            f"{datetime.now():%Y-%m-%d %H:%M}",
            title=f"PAPER CLOSE: {trade['symbol']}",
        )

    # ------------------------------------------------------------------ #
    def run_loop(self, symbols: list[str], timeframes: list[str], poll_seconds: int = 60) -> None:
        """Continuous scanning loop (Ctrl-C to stop)."""
        from strategy.base import DISCLAIMER

        print(
            f"[scanner] live loop started — symbols={symbols} timeframes={timeframes} "
            f"poll={poll_seconds}s (paper trading only, no real orders)\n{DISCLAIMER}"
        )
        try:
            while True:
                self.scan_once(symbols, timeframes)
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("\n[scanner] stopped by user")

    # ------------------------------------------------------------------ #
    def replay(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_frac: float = 0.7,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Forward test: replay bars strictly after `start_frac` through the
        paper pipeline. Indicators warm up on earlier bars (allowed), but
        signals/trades only occur on unseen bars. Mirrors live behaviour:
        same risk guard, journal and alert path as scan_once.
        """
        prepared = self.strategy.prepare(df)
        split = int(len(prepared) * start_frac)
        fired: list[Signal] = []
        open_pos: dict = {}
        cooldown_until: dict[tuple, int] = {}
        cd_bars = self.strategy.cfg["cooldown_bars"]

        for i in range(split, len(prepared)):
            ts = prepared.index[i]
            row = prepared.iloc[i]

            # manage open position on this bar (SL checked before target)
            key = (symbol, timeframe)
            pos = open_pos.get(key)
            if pos is not None:
                long = pos["direction"] == "BUY"
                sl, tgt = pos["sl"], pos["target"]
                hit_sl = row["low"] <= sl if long else row["high"] >= sl
                hit_tgt = row["high"] >= tgt if long else row["low"] <= tgt
                if hit_sl:
                    res = self.journal.close_trade(pos["id"], sl, ts, "stop_loss")
                    self.risk.register_close(res["pnl_amount"], when=ts)
                    self._print_close(pos, res, sl, ts, "STOP-LOSS", verbose)
                    del open_pos[key]
                    cooldown_until[key] = i + cd_bars
                elif hit_tgt:
                    res = self.journal.close_trade(pos["id"], tgt, ts, "target")
                    self.risk.register_close(res["pnl_amount"], when=ts)
                    self._print_close(pos, res, tgt, ts, "TARGET", verbose)
                    del open_pos[key]
                    cooldown_until[key] = i + cd_bars
                elif i - pos["entry_bar"] >= self.strategy.cfg["max_bars_in_trade"]:
                    px = float(row["close"])
                    res = self.journal.close_trade(pos["id"], px, ts, "timeout")
                    self.risk.register_close(res["pnl_amount"], when=ts)
                    self._print_close(pos, res, px, ts, "TIMEOUT", verbose)
                    del open_pos[key]
                    cooldown_until[key] = i + cd_bars

            # look for a new signal on the closed bar (cooldown respected)
            if key in open_pos or cooldown_until.get(key, -1) > i:
                continue
            sig = self.strategy.check_signal(prepared, i, symbol, timeframe, source="paper")
            if sig is None:
                continue
            fired.append(sig)
            decision = self.risk.can_open(sig, when=ts)
            if not decision.allowed:
                if verbose:
                    print(f"[risk] BLOCKED {sig.direction} {symbol}: {decision.reason}")
                continue
            size = self.risk.size_for_signal(sig)
            if size.qty <= 0:
                continue
            self.journal.log_signal(sig, mode="paper", alerted=True)
            trade_id = self.journal.open_paper_trade(sig, size.qty, size.risk_amount, entry_time=ts)
            self.risk.register_open(when=ts)
            open_pos[key] = {
                "id": trade_id, "direction": sig.direction, "sl": sig.stop_loss,
                "target": sig.target, "entry_bar": i, "symbol": symbol, "timeframe": timeframe,
                "qty": size.qty,
            }
            if verbose:
                print(
                    f"[paper] OPEN #{trade_id} {sig.direction} {symbol} {timeframe} "
                    f"qty={size.qty:g} entry={sig.entry:.2f} SL={sig.stop_loss:.2f} "
                    f"target={sig.target:.2f} conf={sig.confidence:.0f}"
                )
            self.notifier.send_signal(sig, source="paper")

        trades = self.journal.all_trades(mode="paper", closed_only=True)
        return trades

    def _print_close(self, pos, res, exit_price, ts, label, verbose) -> None:
        if verbose:
            print(
                f"[paper] CLOSE #{pos['id']} {pos['direction']} {pos['symbol']} "
                f"@{exit_price:.2f} ({label}) result={res['result_r']:+.2f}R "
                f"pnl=${res['pnl_amount']:+.2f}"
            )
