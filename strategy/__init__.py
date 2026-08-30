"""Strategy engine: rule-based, configurable setups.

Currently shipped: TrendPullbackStrategy (pullback & breakout variants,
selectable via config `entry_type`). The interface (prepare / check_signal /
generate_signals) is what backtester and live_scanner program against, so new
strategies drop in without touching the engines.
"""
from .base import DEFAULT_CONFIG, DISCLAIMER, Signal, load_strategy_config
from .trend_pullback import TrendPullbackStrategy

__all__ = ["Signal", "TrendPullbackStrategy", "load_strategy_config", "DEFAULT_CONFIG", "DISCLAIMER"]


def build_strategy(config: dict | None = None) -> TrendPullbackStrategy:
    return TrendPullbackStrategy(config)
