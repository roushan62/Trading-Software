"""Trend-Pullback / Breakout strategy — rule-based, fully configurable.

LONG setup (mirror for SHORT):
  GATE   trend alignment: EMA fast > mid > slow  (mandatory — no trend, no trade)
  TRIG   pullback:  price dipped to EMA fast / VWAP zone (within pullback_atr_dist*ATR)
                    and closed back above it, in an intact uptrend
         breakout:  close breaks the highest high of the last N bars
  CONF   close above VWAP            (weight)
  CONF   RSI inside momentum band     (weight)
  CONF   volume z-score above floor   (weight)

Confidence = sum of weights of aligned conditions (0-100). A signal is
emitted only when GATE + TRIG fire and confidence >= min_confidence.

Every signal includes: entry (signal-bar close), stop-loss (structure-based
swing level - ATR buffer, or ATR multiple — whichever configured) and target
(entry + risk_reward * risk). SL is widened to at least min_sl_atr_mult*ATR
so a 'structure' stop is never meaninglessly tight.
"""
from __future__ import annotations

import pandas as pd

from indicators import IndicatorConfig, compute_all
from indicators.support_resistance import last_swing_high_before, last_swing_low_before

from .base import Signal, load_strategy_config


class TrendPullbackStrategy:
    name = "trend_pullback"

    def __init__(self, config: dict | None = None):
        self.cfg = load_strategy_config(config)
        self.indicator_config = IndicatorConfig(
            ema_fast=self.cfg["ema_fast"],
            ema_mid=self.cfg["ema_mid"],
            ema_slow=self.cfg["ema_slow"],
            rsi_period=self.cfg["rsi_period"],
            atr_period=self.cfg["atr_period"],
            volume_period=self.cfg["volume_period"],
        )
        self.warmup_bars = max(self.cfg["ema_slow"], 200) + 10

    # ------------------------------------------------------------------ #
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all indicator columns (causal)."""
        return compute_all(df, self.indicator_config)

    # ------------------------------------------------------------------ #
    def check_signal(
        self,
        df: pd.DataFrame,
        i: int,
        symbol: str,
        timeframe: str,
        source: str = "backtest",
    ) -> Signal | None:
        """Evaluate the rule set on closed bar `i`. Returns a Signal or None.

        Only rows <= i are read — strictly causal, safe for backtesting.
        """
        cfg = self.cfg
        if i < self.warmup_bars or i >= len(df):
            return None

        row = df.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        ema_f, vwap, rsi = float(row["ema_fast"]), float(row["vwap"]), float(row["rsi"])
        atr, atr_pct, vol_z = float(row["atr"]), float(row["atr_pct"]), float(row["vol_z"])
        ema_m, ema_s = float(row["ema_mid"]), float(row["ema_slow"])
        structure = row["structure"] if pd.notna(row["structure"]) else "n/a"

        if any(pd.isna(x) for x in (ema_f, ema_m, ema_s, vwap, rsi, atr)) or atr <= 0:
            return None
        if atr_pct < cfg["min_atr_pct"]:
            return None

        is_long_trend = ema_f > ema_m > ema_s
        is_short_trend = ema_f < ema_m < ema_s

        # ---------------- LONG ----------------
        if is_long_trend:
            sig = self._evaluate_side(
                df, i, "BUY", close, high, low, ema_f, vwap, rsi, atr, vol_z,
                symbol, timeframe, structure, source,
            )
            if sig is not None:
                return sig
        # ---------------- SHORT ----------------
        if is_short_trend:
            return self._evaluate_side(
                df, i, "SELL", close, high, low, ema_f, vwap, rsi, atr, vol_z,
                symbol, timeframe, structure, source,
            )
        return None

    # ------------------------------------------------------------------ #
    def _evaluate_side(
        self, df, i, direction, close, high, low, ema_f, vwap, rsi, atr, vol_z,
        symbol, timeframe, structure, source,
    ) -> Signal | None:
        cfg = self.cfg
        w = cfg["confidence_weights"]
        long = direction == "BUY"
        reasons: list[str] = []
        score = 0.0

        # GATE: trend alignment (already verified by caller) — full weight
        score += w["trend_alignment"]
        reasons.append(
            f"EMA{cfg['ema_fast']}>{cfg['ema_mid']}>{cfg['ema_slow']} aligned {'up' if long else 'down'}"
        )

        # TRIG: pullback or breakout
        if cfg["entry_type"] == "pullback":
            touch = (
                low <= ema_f + cfg["pullback_atr_dist"] * atr
                if long else high >= ema_f - cfg["pullback_atr_dist"] * atr
            )
            reclaim = close > ema_f if long else close < ema_f
            if not (touch and reclaim):
                return None
            score += w["trigger_pattern"]
            reasons.append("pullback to EMA{} reclaimed on close".format(cfg["ema_fast"]))
        else:  # breakout
            lb = cfg["breakout_swing_lookback"]
            window = df.iloc[max(0, i - lb): i]
            if len(window) < lb // 2:
                return None
            level = float(window["high"].max()) if long else float(window["low"].min())
            broke = close > level if long else close < level
            if not broke:
                return None
            score += w["trigger_pattern"]
            reasons.append(f"breakout of {lb}-bar {'high' if long else 'low'} {level:.2f}")

        # CONF: VWAP position
        if cfg["use_vwap_filter"]:
            if (close > vwap) if long else (close < vwap):
                score += w["vwap_position"]
                reasons.append(f"price {'above' if long else 'below'} VWAP")

        # CONF: RSI momentum band
        lo, hi = (cfg["rsi_buy_min"], cfg["rsi_buy_max"]) if long else (cfg["rsi_sell_min"], cfg["rsi_sell_max"])
        if lo <= rsi <= hi:
            score += w["rsi_zone"]
            reasons.append(f"RSI {rsi:.0f} in momentum band [{lo},{hi}]")

        # CONF: volume
        if vol_z >= cfg["min_volume_z"]:
            score += w["volume_confirmation"]
            reasons.append(f"volume z={vol_z:+.1f}")

        if score < cfg["min_confidence"]:
            return None

        # ---- stop-loss (structure or ATR) ----
        sl = self._stop_loss(df, i, direction, close, atr)
        if sl is None:
            return None

        # ---- target at configured R:R ----
        risk = abs(close - sl)
        target = close + cfg["risk_reward"] * risk if long else close - cfg["risk_reward"] * risk

        return Signal(
            symbol=symbol,
            direction=direction,
            entry=close,
            stop_loss=sl,
            target=target,
            timeframe=timeframe,
            confidence=min(100.0, score),
            setup=f"{cfg['name']}:{cfg['entry_type']}",
            reasons=reasons,
            market_condition=structure,
            timestamp=df.index[i],
            bar_index=i,
            source=source,
            extra={"rsi": round(rsi, 2), "atr": round(atr, 4), "vol_z": round(vol_z, 2),
                   "vwap": round(vwap, 4), "ema_fast": round(ema_f, 4)},
        )

    # ------------------------------------------------------------------ #
    def _stop_loss(self, df: pd.DataFrame, i: int, direction: str, entry: float, atr_val: float) -> float | None:
        cfg = self.cfg
        long = direction == "BUY"
        min_dist = cfg["min_sl_atr_mult"] * atr_val
        atr_stop = entry - cfg["atr_sl_mult"] * atr_val if long else entry + cfg["atr_sl_mult"] * atr_val

        if cfg["sl_type"] == "atr":
            return round(atr_stop, 6)

        lookback = cfg["structure_lookback_bars"]
        swing = (
            last_swing_low_before(df, i, lookback=lookback)
            if long else last_swing_high_before(df, i, lookback=lookback)
        )
        if swing is None:
            return round(atr_stop, 6)

        buffer = cfg["structure_sl_buffer_atr"] * atr_val
        structure_stop = swing - buffer if long else swing + buffer

        # Never tighter than min distance (avoid meaningless stops)
        if long:
            stop = min(structure_stop, entry - min_dist)
        else:
            stop = max(structure_stop, entry + min_dist)
        return round(stop, 6)

    # ------------------------------------------------------------------ #
    def generate_signals(
        self, df: pd.DataFrame, symbol: str, timeframe: str, source: str = "backtest"
    ) -> list[tuple[int, Signal]]:
        """Scan every bar for candidate signals.

        Basic dedup: after a signal, ignore same-direction re-triggers for
        `cooldown_bars` so one setup isn't counted many times on consecutive
        bars. (The backtester/live scanner apply their own position-level
        cooldown on top of this.)
        """
        prepared = df if "ema_fast" in df.columns else self.prepare(df)
        cooldown = self.cfg["cooldown_bars"]
        out: list[tuple[int, Signal]] = []
        last_bar: dict[str, int] = {}
        for i in range(self.warmup_bars, len(prepared)):
            sig = self.check_signal(prepared, i, symbol, timeframe, source)
            if sig is None:
                continue
            key = sig.direction
            if key in last_bar and (i - last_bar[key]) < cooldown:
                continue
            last_bar[key] = i
            out.append((i, sig))
        return out
