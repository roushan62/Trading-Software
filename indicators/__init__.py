"""Indicator engine: EMA, VWAP, RSI, ATR, volume, S/R, market structure.

All indicators are causal (no lookahead) so they are safe for both
backtesting and live scanning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .atr import atr, atr_pct
from .ema import ema, ema_stack
from .market_structure import RANGE, UPTREND, DOWNTREND, market_structure_series
from .rsi import rsi
from .support_resistance import find_swings, support_resistance_levels
from .volume import volume_zscore
from .vwap import vwap

__all__ = [
    "ema", "ema_stack", "vwap", "rsi", "atr", "atr_pct",
    "volume_zscore", "find_swings", "support_resistance_levels",
    "market_structure_series", "compute_all", "IndicatorConfig",
    "UPTREND", "DOWNTREND", "RANGE",
]


@dataclass
class IndicatorConfig:
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    volume_period: int = 20
    vwap_period: int = 20
    structure_swing_k: int = 3


def compute_all(df: pd.DataFrame, cfg: IndicatorConfig | None = None) -> pd.DataFrame:
    """Attach every indicator column to a copy of the OHLCV DataFrame.

    Adds: ema_fast, ema_mid, ema_slow, vwap, rsi, atr, atr_pct, vol_z, structure.
    """
    cfg = cfg or IndicatorConfig()
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_mid"] = ema(out["close"], cfg.ema_mid)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)
    out["vwap"] = vwap(out, period=cfg.vwap_period)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = (out["atr"] / out["close"]) * 100.0
    out["vol_z"] = volume_zscore(out["volume"], cfg.volume_period)
    out["structure"] = market_structure_series(out, k=cfg.structure_swing_k)
    return out
