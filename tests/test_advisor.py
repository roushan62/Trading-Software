"""Unit tests: AI advisor (context builder, local advice, OpenRouter payload).
No network calls — OpenRouter HTTP layer is monkeypatched."""
import json

import pandas as pd
import pytest

from ai.advisor import (
    Advisor,
    build_market_context,
    get_api_key,
    local_advice,
    render_context,
)
from datafeed.synthetic import generate_ohlcv
from indicators import compute_all
from indicators.forecast import monte_carlo_forecast
from strategy.base import DISCLAIMER, Signal
from strategy.trend_pullback import TrendPullbackStrategy


@pytest.fixture(scope="module")
def prepared():
    return compute_all(generate_ohlcv("ADV", "1h", bars=2500, seed=31))


def make_signal(**kw):
    d = dict(symbol="ADV", direction="BUY", entry=100.0, stop_loss=98.0, target=104.0,
             timeframe="1h", confidence=85, setup="trend_pullback:pullback")
    d.update(kw)
    return Signal(**d)


class TestContext:
    def test_contains_everything_ai_needs(self, prepared):
        fc = monte_carlo_forecast(prepared, horizon=15, sims=200, seed=2)
        sig = make_signal()
        ctx = build_market_context(prepared, "ADV", "1h", signal=sig, forecast=fc,
                                   risk_config={"account_size": 10000, "risk_per_trade_pct": 0.5})
        for key in ("symbol", "timeframe", "price", "indicators", "support_levels",
                    "resistance_levels", "monte_carlo_forecast", "active_signal",
                    "account_settings"):
            assert key in ctx, f"missing {key}"
        ind = ctx["indicators"]
        for k in ("ema20", "ema50", "ema200", "vwap", "rsi14", "atr14", "market_structure"):
            assert k in ind
        # signal numbers match the Signal object
        assert ctx["active_signal"]["stop_loss"] == 98.0
        assert ctx["active_signal"]["target"] == 104.0
        # forecast first-touch probability present
        assert "prob_target_before_sl_pct" in ctx["monte_carlo_forecast"]
        # json-renderable
        json.loads(render_context(ctx))

    def test_none_signal_is_explicit(self, prepared):
        ctx = build_market_context(prepared, "ADV", "1h", signal=None)
        assert ctx["active_signal"] is None


class TestLocalAdvice:
    def test_with_signal_has_full_plan(self, prepared):
        fc = monte_carlo_forecast(prepared, horizon=15, sims=200, seed=2)
        sig = make_signal()
        ctx = build_market_context(prepared, "ADV", "1h", signal=sig, forecast=fc,
                                   risk_config={"account_size": 10000, "risk_per_trade_pct": 0.5})
        text = local_advice(ctx, "kya karu?")
        assert "ENTRY" in text and "STOP-LOSS" in text and "TARGET" in text
        assert "98" in text and "104" in text          # SL & target values from the signal
        assert "QUANTITY" in text                       # position sizing included
        assert DISCLAIMER in text

    def test_without_signal_recommends_wait(self, prepared):
        ctx = build_market_context(prepared, "ADV", "1h", signal=None)
        text = local_advice(ctx)
        assert "HOLD" in text or "WAIT" in text
        assert DISCLAIMER in text

    def test_offline_advisor_no_key(self, prepared):
        advisor = Advisor({"openrouter_api_key": ""})   # no key
        assert advisor.has_key is False
        ctx = build_market_context(prepared, "ADV", "1h")
        out = advisor.chat("entry leni chahiye?", ctx)
        assert "rule-based" in out or "ENTRY" in out or "HOLD" in out


class TestOpenRouterClient:
    def test_payload_and_headers(self, prepared, monkeypatch):
        import types

        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers

            class R:
                status_code = 200
                text = '{"ok":true}'

                def json(self):
                    return {"choices": [{"message": {"content": "test answer"}}]}

            return R()

        # `import requests` inside _ask_openrouter resolves via sys.modules
        monkeypatch.setitem(sys_modules(), "requests", types.SimpleNamespace(post=fake_post))

        advisor = Advisor({"model": "deepseek/deepseek-chat"}, api_key="sk-test-123")
        assert advisor.has_key
        ctx = build_market_context(prepared, "ADV", "1h")
        out = advisor.chat("kya scenario hai?", ctx)
        assert "test answer" in out
        assert DISCLAIMER in out                       # appended if model forgot it
        assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-test-123"
        body = captured["json"]
        assert body["model"] == "deepseek/deepseek-chat"
        assert any(m["role"] == "system" for m in body["messages"])
        assert "MARKET CONTEXT" in body["messages"][-1]["content"]
        assert "kya scenario hai?" in body["messages"][-1]["content"]

    def test_error_falls_back_to_local(self, prepared, monkeypatch):
        import types

        def bad_post(*a, **k):
            raise ConnectionError("no internet")

        monkeypatch.setitem(sys_modules(), "requests", types.SimpleNamespace(post=bad_post))
        advisor = Advisor({}, api_key="sk-x")
        ctx = build_market_context(prepared, "ADV", "1h")
        out = advisor.chat("anything", ctx)
        assert "OpenRouter error" in out
        assert DISCLAIMER in out                        # fallback advice still complete


def sys_modules():
    import sys

    return sys.modules
