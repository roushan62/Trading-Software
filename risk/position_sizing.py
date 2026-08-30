"""Risk management: position sizing + hard guard rails.

Position sizing
---------------
qty = (account_size * risk_per_trade_pct/100) / |entry - stop_loss|
Capped so total position value never exceeds account size (no leverage).

Guard rails (all configurable)
------------------------------
- max daily loss (money): once breached, NO new signals until next day
- max trades per day
- max simultaneous open positions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from strategy.base import Signal


@dataclass
class PositionSize:
    qty: float              # units to trade (0 = risk too small / price too high)
    risk_amount: float      # money at risk if SL is hit
    position_value: float   # qty * entry
    risk_per_unit: float    # |entry - sl|
    notes: list[str] = field(default_factory=list)


def position_size(
    entry: float,
    stop_loss: float,
    account_size: float,
    risk_per_trade_pct: float,
    allow_leverage: bool = False,
    round_to: float = 0.0,
) -> PositionSize:
    """Fixed-%-risk position sizing. Every trade risks the same % of account."""
    if account_size <= 0:
        raise ValueError("account_size must be > 0")
    if not 0 < risk_per_trade_pct <= 100:
        raise ValueError("risk_per_trade_pct must be in (0, 100]")
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit <= 0:
        raise ValueError("entry and stop_loss must differ (zero risk is invalid)")

    risk_amount = account_size * risk_per_trade_pct / 100.0
    qty = risk_amount / risk_per_unit
    notes: list[str] = []

    if not allow_leverage and qty * entry > account_size:
        qty = account_size / entry
        notes.append("qty capped by account size (no leverage)")

    if round_to > 0:
        qty = int(qty // round_to) * round_to
        if qty <= 0:
            notes.append("risk too small for min lot size — qty rounds to 0, skip trade")
        risk_amount = qty * risk_per_unit

    return PositionSize(
        qty=round(qty, 8),
        risk_amount=round(qty * risk_per_unit, 2),
        position_value=round(qty * entry, 2),
        risk_per_unit=round(risk_per_unit, 6),
        notes=notes,
    )


@dataclass
class RiskConfig:
    account_size: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 5
    max_open_positions: int = 3

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskConfig":
        d = d or {}
        valid = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**valid)

    @property
    def daily_loss_limit_amount(self) -> float:
        return self.account_size * self.max_daily_loss_pct / 100.0


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


class RiskGuard:
    """Tracks daily state and blocks new trades when a limit is hit.

    Feed it every closed trade's PnL via `register_close()`; ask
    `can_open()` before every new signal.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.today: date | None = None
        self.trades_today = 0
        self.realized_pnl_today = 0.0
        self.open_positions = 0
        self.blocked_reason: str | None = None

    # ------------------------------------------------------------------ #
    def _roll_day(self, day: date) -> None:
        if self.today != day:
            self.today = day
            self.trades_today = 0
            self.realized_pnl_today = 0.0
            self.blocked_reason = None

    def can_open(self, signal: Signal | None = None, when=None) -> RiskDecision:
        day = (when or _now()).date() if when is not None else _now().date()
        self._roll_day(day)

        if self.blocked_reason:
            return RiskDecision(False, f"blocked for the day: {self.blocked_reason}")
        if self.open_positions >= self.config.max_open_positions:
            return RiskDecision(False, f"max open positions reached ({self.config.max_open_positions})")
        if self.trades_today >= self.config.max_trades_per_day:
            return RiskDecision(False, f"max trades/day reached ({self.config.max_trades_per_day})")
        return RiskDecision(True, "ok")

    def register_open(self, when=None) -> None:
        self._roll_day((when or _now()).date())
        self.trades_today += 1
        self.open_positions += 1

    def register_close(self, pnl_amount: float, when=None) -> None:
        self._roll_day((when or _now()).date())
        self.open_positions = max(0, self.open_positions - 1)
        self.realized_pnl_today += pnl_amount
        if self.realized_pnl_today <= -self.config.daily_loss_limit_amount:
            self.blocked_reason = (
                f"daily loss limit hit ({self.config.max_daily_loss_pct}% = "
                f"-{self.config.daily_loss_limit_amount:.2f}); new signals blocked until tomorrow"
            )

    # ------------------------------------------------------------------ #
    def size_for_signal(self, signal: Signal, allow_leverage: bool = False) -> PositionSize:
        return position_size(
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            account_size=self.config.account_size,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            allow_leverage=allow_leverage,
        )

    def status(self) -> dict:
        return {
            "date": str(self.today) if self.today else None,
            "trades_today": self.trades_today,
            "open_positions": self.open_positions,
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "daily_loss_limit": round(self.config.daily_loss_limit_amount, 2),
            "blocked_reason": self.blocked_reason,
        }


def _now():
    import datetime

    return datetime.datetime.now()
