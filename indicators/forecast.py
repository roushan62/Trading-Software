"""Monte Carlo price projection — "next N bars" probability cone.

HONESTY FIRST: nobody can predict the market. What this module does is
project a PROBABILITY DISTRIBUTION of future prices by bootstrapping
thousands of simulated paths from the symbol's own recent bar-to-bar
returns (volatility + trend behaviour), then summarising them as
percentile bands (p10/p25/p50/p75/p90).

What you get:
  - a cone of likely price paths for the next N bars
  - P(final bar above current price)
  - P(level A touched before level B)  e.g. target vs stop-loss
  - expected range at the horizon

These are statistics over simulated scenarios — NOT predictions or
guarantees. Always shown with the disclaimer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    symbol: str
    timeframe: str
    horizon: int
    last_close: float
    future_index: pd.DatetimeIndex          # timestamps of the forward bars
    bands: pd.DataFrame                     # p5..p95 per forward bar
    p_up: float                             # P(close_horizon > last_close)
    sims: int
    paths: np.ndarray                       # (sims, horizon+1) simulated prices

    def band_table(self) -> pd.DataFrame:
        t = self.bands.copy()
        t.insert(0, "time", self.future_index)
        return t.round(2)

    def first_touch(self, level_a: float, level_b: float) -> float:
        """P(level_a touched before level_b). e.g. target vs stop-loss."""
        return first_touch_prob(self.paths, level_a, level_b)

    def touch_prob(self, level: float) -> float:
        """P(level touched at any point within the horizon)."""
        return level_touch_prob(self.paths, level)

    def summary(self) -> dict:
        last = self.bands.iloc[-1]
        return {
            "horizon_bars": self.horizon,
            "timeframe": self.timeframe,
            "last_close": round(self.last_close, 2),
            "p10": round(float(last["p10"]), 2),
            "p25": round(float(last["p25"]), 2),
            "median": round(float(last["p50"]), 2),
            "p75": round(float(last["p75"]), 2),
            "p90": round(float(last["p90"]), 2),
            "p_up_pct": round(self.p_up * 100, 1),
            "sims": self.sims,
        }


def monte_carlo_forecast(
    df: pd.DataFrame,
    horizon: int = 15,
    sims: int = 2000,
    lookback: int = 500,
    seed: int | None = None,
    structure_col: str = "structure",
) -> Forecast:
    """Bootstrap Monte Carlo projection of the next `horizon` bars.

    Returns are resampled from the symbol's own recent history. If a
    `structure` column is present, sampling is restricted to bars from the
    CURRENT structure regime (uptrend/downtrend/range) when enough samples
    exist — so the cone reflects the active regime's behaviour.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(df) < 30:
        raise ValueError("need at least 30 bars to forecast")

    rng = np.random.default_rng(seed)
    close = df["close"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(close[-(lookback + 1):]))

    # regime-conditional sampling when possible
    if structure_col in df.columns and len(df) - 1 >= lookback:
        labels = df[structure_col].iloc[-(lookback):].to_numpy()
        labels = labels[1:]  # returns are forward differences
        current = df[structure_col].iloc[-1]
        mask = labels == current if isinstance(current, str) else None
        if mask is not None and mask.sum() >= 100:
            log_ret = log_ret[-len(labels):][mask]

    draws = rng.choice(log_ret, size=(sims, horizon), replace=True)
    cum = np.cumsum(draws, axis=1)
    paths = float(close[-1]) * np.exp(cum)                    # (sims, horizon)
    all_paths = np.hstack([np.full((sims, 1), close[-1]), paths])

    last_close = float(close[-1])
    p_up = float((paths[:, -1] > last_close).mean())

    # forward timestamps from the median bar gap
    idx = df.index
    if len(idx) > 2:
        gaps = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        step = pd.Timedelta(seconds=float(np.median(gaps)))
    else:
        step = pd.Timedelta(hours=1)
    future = pd.date_range(start=idx[-1] + step, periods=horizon, freq=step)

    qs = [5, 10, 25, 50, 75, 90, 95]
    band_vals = {f"p{q}": np.percentile(paths, q, axis=0) for q in qs}
    bands = pd.DataFrame(band_vals, index=future)
    bands.index.name = "datetime"

    sym = getattr(df, "attrs", {}).get("symbol", "")
    tf = getattr(df, "attrs", {}).get("timeframe", "")
    return Forecast(
        symbol=sym, timeframe=tf, horizon=horizon, last_close=last_close,
        future_index=future, bands=bands, p_up=p_up, sims=sims, paths=all_paths,
    )


def first_touch_prob(paths: np.ndarray, level_a: float, level_b: float) -> float:
    """P(price touches `level_a` before `level_b`) across simulated paths.

    `level_a` and `level_b` must be on OPPOSITE sides of the current price
    (paths[:, 0]) — exactly how a target and a stop-loss sit around an entry.
    Paths touching neither within the horizon are excluded from the
    denominator (they favour neither level). Vectorised.
    """
    if level_a == level_b:
        raise ValueError("levels must differ")
    above_a = paths >= level_a
    below_b = paths <= level_b
    hit_a = above_a.argmax(axis=1)
    hit_b = below_b.argmax(axis=1)
    none_a = ~above_a.any(axis=1)
    none_b = ~below_b.any(axis=1)

    ta = np.where(none_a, np.inf, hit_a)
    tb = np.where(none_b, np.inf, hit_b)
    either = ~(none_a & none_b)
    a_first = (ta <= tb) & either
    total = int(either.sum())
    if total == 0:
        return float("nan")
    return float(a_first.sum() / total)


def level_touch_prob(paths: np.ndarray, level: float) -> float:
    """P(path touches `level` at any point within the horizon)."""
    if level >= paths[0, 0]:
        return float((paths >= level).any(axis=1).mean())
    return float((paths <= level).any(axis=1).mean())


DISCLAIMER = ("Projection = Monte Carlo probability cone from historical returns. "
              "Not a prediction. Not financial advice. Based on historical statistical "
              "edge, not a guarantee.")
