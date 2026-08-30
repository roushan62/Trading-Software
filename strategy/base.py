"""Strategy engine base: Signal object + configurable rule set.

NON-NEGOTIABLE: every Signal carries a stop-loss. `validate()` refuses to
emit a signal without one. Nothing downstream accepts an SL-less signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

DISCLAIMER = "Not financial advice. Based on historical statistical edge, not a guarantee."

DIRECTIONS = ("BUY", "SELL")


@dataclass
class Signal:
    symbol: str
    direction: str                 # "BUY" | "SELL"
    entry: float
    stop_loss: float
    target: float
    timeframe: str
    confidence: float              # 0-100
    setup: str
    reasons: list[str] = field(default_factory=list)
    market_condition: str = "n/a"  # structure label at signal bar
    timestamp: pd.Timestamp | None = None
    bar_index: int | None = None
    risk: float = 0.0              # |entry - stop_loss|
    rr: float = 0.0                # reward:risk at target
    source: str = "backtest"       # "backtest" | "paper"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.risk = abs(self.entry - self.stop_loss)
        if self.risk <= 0:
            raise ValueError("Signal invalid: stop-loss equals entry (zero risk)")
        self.rr = abs(self.target - self.entry) / self.risk
        self.validate()

    def validate(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        for name, val in (("entry", self.entry), ("stop_loss", self.stop_loss), ("target", self.target)):
            if val is None or pd.isna(val):
                raise ValueError(f"Signal invalid: {name} is NaN")
        if self.risk <= 0:
            raise ValueError("Signal invalid: zero/NaN risk — SL must differ from entry")
        if self.direction == "BUY" and not (self.stop_loss < self.entry < self.target):
            raise ValueError("BUY signal requires SL < entry < target")
        if self.direction == "SELL" and not (self.target < self.entry < self.stop_loss):
            raise ValueError("SELL signal requires target < entry < SL")

    @property
    def disclaimer(self) -> str:
        return DISCLAIMER

    def alert_text(self) -> str:
        arrow = "▲" if self.direction == "BUY" else "▼"
        lines = [
            f"{arrow} {self.direction} {self.symbol} [{self.timeframe}] — {self.setup}",
            f"Entry: {self.entry:.2f} | SL: {self.stop_loss:.2f} | Target: {self.target:.2f}",
            f"R:R 1:{self.rr:.1f} | Confidence: {self.confidence:.0f}/100 | Structure: {self.market_condition}",
            f"Why: {'; '.join(self.reasons)}",
            self.disclaimer,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "symbol": self.symbol, "direction": self.direction,
            "entry": round(self.entry, 6), "stop_loss": round(self.stop_loss, 6),
            "target": round(self.target, 6), "timeframe": self.timeframe,
            "confidence": round(self.confidence, 1), "setup": self.setup,
            "market_condition": self.market_condition,
            "risk": round(self.risk, 6), "rr": round(self.rr, 2),
            "reasons": "; ".join(self.reasons), "source": self.source,
            "disclaimer": DISCLAIMER,
        }
        if self.timestamp is not None:
            d["timestamp"] = str(self.timestamp)
        return d


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "trend_pullback",
    "entry_type": "pullback",
    "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
    "use_vwap_filter": True,
    "rsi_period": 14, "rsi_buy_min": 45, "rsi_buy_max": 68,
    "rsi_sell_min": 32, "rsi_sell_max": 55,
    "atr_period": 14, "atr_sl_mult": 1.5,
    "min_sl_atr_mult": 1.0,
    "sl_type": "structure",          # "structure" | "atr"
    "structure_sl_buffer_atr": 0.25,
    "structure_lookback_bars": 20,
    "risk_reward": 2.0,
    "pullback_atr_dist": 0.5,
    "breakout_swing_lookback": 20,
    "volume_period": 20, "min_volume_z": -0.2,
    "min_confidence": 60,
    "min_atr_pct": 0.0,
    "max_bars_in_trade": 100,
    "cooldown_bars": 5,
    "confidence_weights": {
        "trend_alignment": 30, "vwap_position": 15, "rsi_zone": 20,
        "trigger_pattern": 20, "volume_confirmation": 15,
    },
}


def load_strategy_config(overrides: dict[str, Any] | None = None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge overrides onto defaults (and optionally onto a custom base)."""
    cfg = dict(base or DEFAULT_CONFIG)
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    # sanity
    if cfg["entry_type"] not in ("pullback", "breakout"):
        raise ValueError("entry_type must be 'pullback' or 'breakout'")
    if cfg["sl_type"] not in ("structure", "atr"):
        raise ValueError("sl_type must be 'structure' or 'atr'")
    if cfg["risk_reward"] <= 0:
        raise ValueError("risk_reward must be > 0")
    w = cfg["confidence_weights"]
    if any(v < 0 for v in w.values()):
        raise ValueError("confidence weights must be >= 0")
    return cfg
