"""Unit tests: Monte Carlo forecast module."""
import numpy as np
import pandas as pd
import pytest

from datafeed.synthetic import generate_ohlcv
from indicators import compute_all
from indicators.forecast import (
    Forecast,
    first_touch_prob,
    level_touch_prob,
    monte_carlo_forecast,
)


@pytest.fixture(scope="module")
def prepared():
    return compute_all(generate_ohlcv("F", "1h", bars=2500, seed=21))


class TestMonteCarlo:
    def test_bands_ordered_and_sized(self, prepared):
        fc = monte_carlo_forecast(prepared, horizon=15, sims=500, seed=3)
        assert len(fc.bands) == 15
        b = fc.bands
        for lo, hi in (("p5", "p10"), ("p10", "p25"), ("p25", "p50"),
                       ("p50", "p75"), ("p75", "p90"), ("p90", "p95")):
            assert (b[lo] <= b[hi]).all(), f"{lo} must be <= {hi}"
        assert (b["p50"].iloc[0] > 0).all()

    def test_future_index_follows_bar_gap(self, prepared):
        fc = monte_carlo_forecast(prepared, horizon=10, sims=200, seed=4)
        gap = prepared.index[-1] - prepared.index[-2]
        assert fc.future_index[0] - prepared.index[-1] == gap
        assert len(fc.future_index) == 10

    def test_probabilities_in_range(self, prepared):
        fc = monte_carlo_forecast(prepared, horizon=15, sims=800, seed=5)
        assert 0.0 <= fc.p_up <= 1.0
        d = 0.03 * fc.last_close
        p = fc.first_touch(fc.last_close + d, fc.last_close - d)
        assert 0.0 <= p <= 1.0
        p_up_touch = fc.touch_prob(fc.last_close + d)
        assert 0.0 <= p_up_touch <= 1.0

    def test_deterministic_with_seed(self, prepared):
        a = monte_carlo_forecast(prepared, 12, 300, seed=11)
        b = monte_carlo_forecast(prepared, 12, 300, seed=11)
        assert np.allclose(a.bands.values, b.bands.values)

    def test_cone_widens_with_horizon(self, prepared):
        fc = monte_carlo_forecast(prepared, 30, 600, seed=13)
        width_first = fc.bands["p90"].iloc[0] - fc.bands["p10"].iloc[0]
        width_last = fc.bands["p90"].iloc[-1] - fc.bands["p10"].iloc[-1]
        assert width_last > width_first

    def test_needs_history(self):
        with pytest.raises(ValueError):
            monte_carlo_forecast(generate_ohlcv("F", "1h", bars=10, seed=1), horizon=5)

    def test_summary_shape(self, prepared):
        s = monte_carlo_forecast(prepared, 15, 200, seed=6).summary()
        for key in ("horizon_bars", "median", "p10", "p90", "p_up_pct", "sims"):
            assert key in s


class TestTouchProbs:
    def test_first_touch_symmetric(self):
        # symmetric random walk around 100 -> near 50/50 for ±5
        rng = np.random.default_rng(0)
        steps = rng.normal(0, 1, (20000, 30))
        paths = 100 + np.cumsum(np.hstack([np.zeros((20000, 1)), steps]), axis=1)
        p = first_touch_prob(paths, 105.0, 95.0)
        assert abs(p - 0.5) < 0.05

    def test_first_touch_obvious_case(self):
        # opposite-side levels (as used for target vs SL): near level wins
        # Brownian-motion expectation: P(+1 before -6) ≈ 6/7 ≈ 0.86
        rng = np.random.default_rng(1)
        steps = rng.normal(0, 1, (5000, 50))
        paths = 100 + np.cumsum(np.hstack([np.zeros((5000, 1)), steps]), axis=1)
        p = first_touch_prob(paths, 101.0, 94.0)
        assert p > 0.80

    def test_never_touch_returns_nan(self):
        paths = np.full((10, 5), 100.0)  # flat forever
        assert np.isnan(first_touch_prob(paths, 110.0, 90.0))

    def test_level_touch_bounds(self):
        paths = np.array([[100.0, 102.0, 98.0], [100.0, 101.0, 99.0]])
        assert level_touch_prob(paths, 102.0) == 0.5
        assert level_touch_prob(paths, 98.5) == 0.5
