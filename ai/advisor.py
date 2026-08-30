"""AI trading advisor — OpenRouter API integration + offline rules fallback.

Two brains, one interface:
  1. OPENROUTER (recommended): any large model (DeepSeek, Llama, GPT-4o-mini,
     Claude, Gemini...) via https://openrouter.ai — one key, hundreds of models.
     Set the key in the dashboard sidebar, config.json `ai.openrouter_api_key`,
     the file data/runtime/ai_key.txt, or env OPENROUTER_API_KEY.
  2. LOCAL fallback (no key / no internet): deterministic rule-based advice
     generated straight from the signal engine — still concrete, still has
     stop-losses, still honest.

The AI never sees anything but market context (no keys/logs). Its system
prompt forces: concrete entry/SL/target/R:R, risk warning, reply in the
user's language (Hinglish supported), and the mandatory disclaimer line.
AI output is OPINION for research — this app never places real orders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from strategy.base import DISCLAIMER, Signal
from strategy.trend_pullback import TrendPullbackStrategy

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path(__file__).resolve().parent.parent / "data" / "runtime" / "ai_key.txt"

SYSTEM_PROMPT = """You are an expert technical analysis trading assistant embedded in a
decision-support app. You receive live market context: OHLCV summary, indicators (EMA
20/50/200, VWAP, RSI, ATR, volume z-score), auto support/resistance levels, a HH/HL market
structure label, a Monte Carlo probability forecast, and any active trade signal with its
entry/stop-loss/target and confidence.

RULES YOU MUST FOLLOW:
1. Be concrete. If you discuss a trade, always give: entry zone, exact stop-loss, target,
   risk:reward, and what invalidates the idea. NEVER suggest a trade without a stop-loss.
2. Use ONLY the numbers from the context — do not invent prices.
3. Say clearly when the right action is WAIT / NO TRADE (this is a valid answer, often the best one).
4. Mention position sizing: risk only 0.5-1% of account per trade.
5. Statistics, not certainties: never promise or guarantee anything.
6. Reply in the SAME language the user writes in. If they write Hinglish (Hindi+English mix),
   reply in Hinglish. Keep it simple and practical.
7. End EVERY answer with exactly this line:
   Not financial advice. Based on historical statistical edge, not a guarantee.
"""


# ---------------------------------------------------------------------- #
# API key resolution
# ---------------------------------------------------------------------- #
def get_api_key(config: dict | None = None) -> str:
    """Key priority: env > data/runtime/ai_key.txt > config.json."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key.strip()
    try:
        if KEY_FILE.exists():
            k = KEY_FILE.read_text(encoding="utf-8").strip()
            if k:
                return k
    except OSError:
        pass
    cfg = dict(config or {})
    if "ai" in cfg and isinstance(cfg["ai"], dict):
        cfg = cfg["ai"]
    return str(cfg.get("openrouter_api_key", "")).strip()


def save_api_key(key: str) -> None:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key.strip(), encoding="utf-8")


# ---------------------------------------------------------------------- #
# Market context builder (shared by dashboard + CLI)
# ---------------------------------------------------------------------- #
def build_market_context(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal: Signal | None = None,
    forecast=None,
    risk_config: dict | None = None,
    recent_trades: pd.DataFrame | None = None,
) -> dict:
    """Compact, AI-ready snapshot of everything the engine knows right now."""
    from indicators.support_resistance import support_resistance_levels

    prepared = df if "ema_fast" in df.columns else TrendPullbackStrategy().prepare(df)
    last = prepared.iloc[-1]
    prev = prepared.iloc[-2] if len(prepared) > 1 else last
    chg = (float(last["close"]) - float(prev["close"])) / float(prev["close"]) * 100 if len(prepared) > 1 else 0.0

    levels = support_resistance_levels(prepared, lookback=200)

    ctx = {
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": str(prepared.index[-1]),
        "price": {
            "close": round(float(last["close"]), 2),
            "change_pct": round(chg, 2),
            "day_high": round(float(last["high"]), 2),
            "day_low": round(float(last["low"]), 2),
        },
        "indicators": {
            "ema20": round(float(last["ema_fast"]), 2),
            "ema50": round(float(last["ema_mid"]), 2),
            "ema200": round(float(last["ema_slow"]), 2),
            "vwap": round(float(last["vwap"]), 2),
            "rsi14": round(float(last["rsi"]), 1),
            "atr14": round(float(last["atr"]), 2),
            "atr_pct": round(float(last["atr_pct"]), 2),
            "volume_z": round(float(last["vol_z"]), 2),
            "market_structure": str(last["structure"] or "n/a"),
        },
        "support_levels": [round(p, 2) for p in levels["support"][-3:]],
        "resistance_levels": [round(p, 2) for p in levels["resistance"][:3]],
    }

    if forecast is not None:
        s = forecast.summary()
        ctx["monte_carlo_forecast"] = {
            "horizon_bars": s["horizon_bars"],
            "note": "bootstrap simulation from this symbol's own returns; probabilities, NOT predictions",
            "p10": s["p10"], "median": s["median"], "p90": s["p90"],
            "prob_up_pct": s["p_up_pct"],
        }
        if signal is not None:
            p_tgt = forecast.first_touch(signal.target, signal.stop_loss)
            if p_tgt == p_tgt:  # not NaN
                ctx["monte_carlo_forecast"]["prob_target_before_sl_pct"] = round(p_tgt * 100, 1)

    if signal is not None:
        ctx["active_signal"] = {
            "direction": signal.direction, "entry": round(signal.entry, 2),
            "stop_loss": round(signal.stop_loss, 2), "target": round(signal.target, 2),
            "risk_reward": f"1:{signal.rr:.1f}", "confidence": round(signal.confidence),
            "setup": signal.setup, "market_condition": signal.market_condition,
            "reasons": signal.reasons,
        }
    else:
        ctx["active_signal"] = None

    if risk_config:
        ctx["account_settings"] = {
            "account_size": risk_config.get("account_size"),
            "risk_per_trade_pct": risk_config.get("risk_per_trade_pct"),
            "max_daily_loss_pct": risk_config.get("max_daily_loss_pct"),
        }

    if recent_trades is not None and len(recent_trades):
        ctx["recent_paper_trades"] = [
            {
                "direction": str(t.get("direction")), "result_r": t.get("result_r"),
                "exit_reason": str(t.get("exit_reason")),
            }
            for _, t in recent_trades.tail(5).iterrows()
        ]
    return ctx


def render_context(ctx: dict) -> str:
    return json.dumps(ctx, indent=1, default=str)


# ---------------------------------------------------------------------- #
# Advisor
# ---------------------------------------------------------------------- #
class Advisor:
    def __init__(self, config: dict | None = None, api_key: str | None = None):
        ai_cfg = (config or {}).get("ai", config or {})
        self.model = ai_cfg.get("model", "deepseek/deepseek-chat")
        self.temperature = float(ai_cfg.get("temperature", 0.3))
        self.max_tokens = int(ai_cfg.get("max_tokens", 900))
        self.key = (api_key or get_api_key(config)).strip()

    @property
    def has_key(self) -> bool:
        return bool(self.key)

    # ------------------------------------------------------------------ #
    def chat(self, question: str, ctx: dict, history: list[dict] | None = None) -> str:
        """Ask the AI (or local fallback). Always returns a useful string."""
        if not self.has_key:
            return local_advice(ctx, question, note=(
                "AI model not connected — ye 100% rule-based analysis hai. OpenRouter key "
                "add karne ke baad full AI answers milenge (dashboard sidebar ya "
                "data/runtime/ai_key.txt)."
            ))
        try:
            return self._ask_openrouter(question, ctx, history or [])
        except Exception as e:
            return (
                f"⚠️ OpenRouter error ({type(e).__name__}: {str(e)[:150]}) — "
                "network/key/model check karo. Filhal rule-based analysis:\n\n"
                + local_advice(ctx, question)
            )

    # ------------------------------------------------------------------ #
    def _ask_openrouter(self, question: str, ctx: dict, history: list[dict]) -> str:
        import requests

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-6:]:                       # keep context small
            msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")[:1500]})
        msgs.append({
            "role": "user",
            "content": f"MARKET CONTEXT (JSON):\n{render_context(ctx)}\n\nQUESTION: {question}",
        })
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.trading-software",   # OpenRouter attribution
            "X-Title": "Market Analysis & Trade Signal Software",
        }
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if DISCLAIMER not in content:
            content = content.rstrip() + "\n\n" + DISCLAIMER
        return content


# ---------------------------------------------------------------------- #
# Local (offline) rule-based advisor
# ---------------------------------------------------------------------- #
def local_advice(ctx: dict, question: str = "", note: str = "") -> str:
    """Deterministic advice from the engine itself — works with zero internet."""
    ind = ctx.get("indicators", {})
    px = ctx.get("price", {}).get("close")
    sig = ctx.get("active_signal")
    lines: list[str] = []

    if note:
        lines.append(note)
        lines.append("")

    trend_word = {"uptrend": "UPTREND (HH-HL)", "downtrend": "DOWNTREND (LL-LH)"}.get(
        ind.get("market_structure", ""), "RANGE / sideways"
    )
    ema_up = ind.get("ema20", 0) > ind.get("ema50", 0) > ind.get("ema200", 0)
    ema_down = ind.get("ema20", 0) < ind.get("ema50", 0) < ind.get("ema200", 0)
    rsi = ind.get("rsi14", 50)

    lines.append(f"📊 {ctx.get('symbol')} [{ctx.get('timeframe')}] @ {px}")
    lines.append(f"Structure: {trend_word} | EMA stack: {'BULLISH' if ema_up else 'BEARISH' if ema_down else 'MIXED'} | RSI {rsi} | ATR {ind.get('atr_pct')}%")
    res = ctx.get("resistance_levels") or []
    sup = ctx.get("support_levels") or []
    if res:
        lines.append(f"Nearest resistance: {res[0]}")
    if sup:
        lines.append(f"Nearest support: {sup[-1]}")
    fc = ctx.get("monte_carlo_forecast")
    if fc:
        lines.append(
            f"Monte Carlo (next {fc.get('horizon_bars')} bars): P(up) = {fc.get('prob_up_pct')}% | "
            f"range p10–p90 ≈ {fc.get('p10')} – {fc.get('p90')} (median {fc.get('median')})"
        )
    lines.append("")

    if sig:
        rr = sig.get("risk_reward")
        acct = ctx.get("account_settings", {})
        size_line = ""
        if acct.get("account_size") and acct.get("risk_per_trade_pct"):
            risk_amt = float(acct["account_size"]) * float(acct["risk_per_trade_pct"]) / 100.0
            per_unit = abs(float(sig["entry"]) - float(sig["stop_loss"]))
            qty = risk_amt / per_unit if per_unit > 0 else 0
            size_line = (
                f"5) QUANTITY: ~{qty:.2f} units (risk ₹/${risk_amt:,.0f} = "
                f"{acct.get('risk_per_trade_pct')}% of {acct.get('account_size'):,.0f})"
            )
        lines.append(f"✅ ACTIVE SIGNAL: {sig['direction']} | confidence {sig.get('confidence')}/100 | {sig.get('setup')}")
        lines.append(f"Why: {'; '.join(sig.get('reasons', []))}")
        lines.append("")
        lines.append("EXACT TRADE PLAN (paper/signal ke liye):")
        lines.append(f"1) ENTRY: {sig['entry']} (signal bar close par)")
        lines.append(f"2) STOP-LOSS: {sig['stop_loss']} — ye predefined invalidation hai, ise achhha setup bhi break ho sakta hai isliye SL ke bina trade MAT lo.")
        lines.append(f"3) TARGET: {sig['target']} (risk:reward {rr})")
        lines.append(f"4) INVALIDATION: SL hit, ya structure {trend_word} se reverse, ya RSI extreme ({'>70 overbought' if rsi > 70 else '<30 oversold' if rsi < 30 else 'normal zone'})")
        if size_line:
            lines.append(size_line)
        p = fc.get("prob_target_before_sl_pct") if fc else None
        if p is not None:
            lines.append(f"6) MONTE CARLO: next {fc.get('horizon_bars')} bars me target SL se pehle chhue ki probability ≈ {p}% (statistical, guarantee nahi)")
    else:
        lines.append("⏸ NO ACTIVE SIGNAL = HOLD / WAIT. Ye bhi ek trade decision hai.")
        lines.append("")
        lines.append("Kya watch karein (setup activate hone ke conditions):")
        if ema_up:
            lines.append(f"• LONG setup: price EMA20 ({ind.get('ema20')}) tak pullback karke wapas upar close kare + RSI 45–68 + VWAP ({ind.get('vwap')}) ke upar")
            if res:
                lines.append(f"• Breakout watch: resistance {res[0]} ke upar volume ke saath close = BUY trigger")
        elif ema_down:
            lines.append(f"• SHORT setup: price EMA20 ({ind.get('ema20')}) tak bounce karke wapas niche close kare + RSI 32–55 + VWAP ke niche")
            if sup:
                lines.append(f"• Breakdown watch: support {sup[-1]} ke niche close = SELL trigger")
        else:
            lines.append("• EMA stack mixed hai — range me trade karna low probability hota hai; breakout ka wait karo")
        lines.append("• Daily loss limit / max trades per day ka rule follow karo (risk module enforce karta hai)")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
