"""Trading journal — SQLite-backed, auto-logs every signal and trade.

CRITICAL DESIGN RULE (per spec): backtest trades and paper/live trades are
stored with a `mode` column and every query path keeps them SEPARATE, so
historical (in-sample) performance can never silently blend with forward
(unseen-data) performance.

Tables
------
trades   : one row per trade (open or closed), mode = 'backtest' | 'paper'
signals  : every signal the engine produced (even if not taken)
alerts   : every alert dispatched (channel, payload, success)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from strategy.base import Signal


def _compute_stats(trades_df: pd.DataFrame) -> "BacktestStats":  # noqa: F821
    """Lazy import to avoid circular dependency (engine -> journal -> engine)."""
    from engine.backtester import compute_stats

    return compute_stats(trades_df)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "runtime" / "journal.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL CHECK (mode IN ('backtest','paper')),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    setup TEXT,
    direction TEXT,
    confidence REAL,
    market_condition TEXT,
    signal_time TEXT,
    entry_time TEXT NOT NULL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target REAL NOT NULL,
    qty REAL,
    risk_amount REAL,
    exit_time TEXT,
    exit REAL,
    exit_reason TEXT,
    result_r REAL,
    pnl_amount REAL,
    status TEXT NOT NULL DEFAULT 'open',
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    mode TEXT,
    symbol TEXT,
    timeframe TEXT,
    direction TEXT,
    entry REAL,
    stop_loss REAL,
    target REAL,
    confidence REAL,
    setup TEXT,
    market_condition TEXT,
    reasons TEXT,
    alerted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT (datetime('now')),
    channel TEXT,
    payload TEXT,
    success INTEGER
);
"""


class Journal:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #
    def log_signal(self, sig: Signal, mode: str = "paper", alerted: bool = False) -> int:
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO signals (ts, mode, symbol, timeframe, direction, entry,
                   stop_loss, target, confidence, setup, market_condition, reasons, alerted)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(sig.timestamp or datetime.now()), mode, sig.symbol, sig.timeframe,
                    sig.direction, sig.entry, sig.stop_loss, sig.target, sig.confidence,
                    sig.setup, sig.market_condition, "; ".join(sig.reasons), int(alerted),
                ),
            )
            return cur.lastrowid

    def recent_signals(self, mode: str | None = None, limit: int = 50) -> pd.DataFrame:
        q = "SELECT * FROM signals"
        args: list = []
        if mode:
            q += " WHERE mode = ?"
            args.append(mode)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as con:
            return pd.DataFrame([dict(r) for r in con.execute(q, args).fetchall()])

    # ------------------------------------------------------------------ #
    # Trades
    # ------------------------------------------------------------------ #
    def open_paper_trade(
        self, sig: Signal, qty: float, risk_amount: float, entry_time=None, notes: str = ""
    ) -> int:
        """Open a PAPER trade. No real orders are ever placed by this software."""
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO trades (mode, symbol, timeframe, setup, direction, confidence,
                   market_condition, signal_time, entry_time, entry, stop_loss, target,
                   qty, risk_amount, status, notes)
                   VALUES ('paper',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sig.symbol, sig.timeframe, sig.setup, sig.direction, sig.confidence,
                    sig.market_condition, str(sig.timestamp or datetime.now()),
                    str(entry_time or sig.timestamp or datetime.now()), sig.entry,
                    sig.stop_loss, sig.target, qty, round(risk_amount, 2), "open", notes,
                ),
            )
            return cur.lastrowid

    def close_trade(
        self, trade_id: int, exit_price: float, exit_time, reason: str, notes: str = ""
    ) -> dict:
        """Close a trade and compute R-multiple + money PnL from stored levels."""
        with self._conn() as con:
            row = con.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if row is None:
                raise ValueError(f"trade {trade_id} not found")
            if row["status"] != "open":
                raise ValueError(f"trade {trade_id} already closed")
            long = row["direction"] == "BUY"
            entry, sl = row["entry"], row["stop_loss"]
            risk = abs(entry - sl)
            pnl_price = (exit_price - entry) if long else (entry - exit_price)
            result_r = pnl_price / risk if risk > 0 else 0.0
            qty = row["qty"] or 0.0
            pnl_amount = pnl_price * qty
            con.execute(
                """UPDATE trades SET exit_time=?, exit=?, exit_reason=?, result_r=?,
                   pnl_amount=?, status='closed', notes=notes || ? WHERE id=?""",
                (str(exit_time), exit_price, reason, round(result_r, 4),
                 round(pnl_amount, 2), notes, trade_id),
            )
            return {"id": trade_id, "result_r": round(result_r, 4), "pnl_amount": round(pnl_amount, 2)}

    def record_backtest_trades(self, trades_df: pd.DataFrame, replace: bool = True) -> int:
        """Bulk-store backtest results (mode='backtest'). Cleared separately from paper."""
        if len(trades_df) == 0:
            return 0
        with self._conn() as con:
            if replace:
                con.execute("DELETE FROM trades WHERE mode='backtest'")
            rows = []
            for _, t in trades_df.iterrows():
                rows.append((
                    "backtest", t["symbol"], t["timeframe"], t["setup"], t["direction"],
                    t["confidence"], t["market_condition"], str(t["signal_time"]),
                    str(t["entry_time"]), t["entry"], t["stop_loss"], t["target"],
                    None, t["risk_per_unit"], str(t["exit_time"]), t["exit"],
                    t["exit_reason"], t["result_r"], None, "closed", "",
                ))
            con.executemany(
                """INSERT INTO trades (mode, symbol, timeframe, setup, direction, confidence,
                   market_condition, signal_time, entry_time, entry, stop_loss, target,
                   qty, risk_amount, exit_time, exit, exit_reason, result_r, pnl_amount,
                   status, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            return len(rows)

    def open_trades(self, mode: str = "paper") -> pd.DataFrame:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM trades WHERE status='open' AND mode=? ORDER BY id", (mode,)
            ).fetchall()
            return pd.DataFrame([dict(r) for r in rows])

    def has_open_trade(self, symbol: str, timeframe: str, mode: str = "paper") -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM trades WHERE status='open' AND mode=? AND symbol=? AND timeframe=? LIMIT 1",
                (mode, symbol, timeframe),
            ).fetchone()
            return row is not None

    def all_trades(self, mode: str | None = None, closed_only: bool = False) -> pd.DataFrame:
        q = "SELECT * FROM trades"
        cond, args = [], []
        if mode:
            cond.append("mode = ?")
            args.append(mode)
        if closed_only:
            cond.append("status = 'closed'")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id"
        with self._conn() as con:
            return pd.DataFrame([dict(r) for r in con.execute(q, args).fetchall()])

    # ------------------------------------------------------------------ #
    # Stats & reports
    # ------------------------------------------------------------------ #
    def stats(self, mode: str, since: str | None = None) -> dict:
        df = self.all_trades(mode=mode, closed_only=True)
        if len(df) == 0:
            return _compute_stats(pd.DataFrame()).to_dict()
        if since:
            df = df[pd.to_datetime(df["entry_time"]) >= pd.to_datetime(since)]
        return _compute_stats(df.rename(columns={"exit": "exit_price"})).to_dict()

    def equity_curve(self, mode: str) -> pd.Series:
        df = self.all_trades(mode=mode, closed_only=True)
        if len(df) == 0:
            return pd.Series(dtype=float)
        df = df.sort_values("exit_time")
        return df["result_r"].cumsum().reset_index(drop=True)

    def report(self, period: str = "weekly", mode: str = "paper") -> pd.DataFrame:
        """Weekly/monthly auto-stats: win rate, expectancy, drawdown per period."""
        df = self.all_trades(mode=mode, closed_only=True)
        if len(df) == 0:
            return pd.DataFrame()
        ts = pd.to_datetime(df["entry_time"])
        key = ts.dt.to_period("W" if period == "weekly" else "M").astype(str)
        rows = []
        for k, grp in df.groupby(key):
            s = _compute_stats(grp.rename(columns={"exit": "exit_price"}))
            rows.append({"period": k, **s.to_dict()})
        return pd.DataFrame([dict(r) for r in rows]).sort_values("period").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Alerts log
    # ------------------------------------------------------------------ #
    def log_alert(self, channel: str, payload: str, success: bool) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO alerts (channel, payload, success) VALUES (?,?,?)",
                (channel, payload[:2000], int(success)),
            )

    def reset_paper(self) -> None:
        """Clear paper trades only (backtest history untouched)."""
        with self._conn() as con:
            con.execute("DELETE FROM trades WHERE mode='paper'")
            con.execute("DELETE FROM signals WHERE mode='paper'")
