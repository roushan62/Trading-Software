"""Streamlit dashboard — market visualisation, signals, forecast, AI assistant, journal.

Run from the repo root:
    streamlit run dashboard/app.py      (or double-click the desktop shortcut)

Decision-support + paper trading only. Every page carries the disclaimer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.advisor import (
    Advisor,
    build_market_context,
    get_api_key,
    render_context,
    save_api_key,
)
from datafeed import data_source_of, resample_ohlcv
from datafeed.csv_provider import CSVProvider
from engine.backtester import Backtester, segment_analysis
from engine.live_scanner import LiveScanner
from indicators import compute_all
from indicators.forecast import monte_carlo_forecast
from indicators.support_resistance import support_resistance_levels
from journal.logger import Journal
from strategy import DEFAULT_CONFIG, DISCLAIMER
from strategy.trend_pullback import TrendPullbackStrategy

st.set_page_config(page_title="Market Analysis & Trade Signals", page_icon="📈", layout="wide")

RUNTIME = ROOT / "data" / "runtime"
TF_ORDER = ["1m", "5m", "15m", "1h", "1d"]


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def strategy_cfg() -> dict:
    return {**DEFAULT_CONFIG, **load_config()["strategy"]}


def available_series() -> list[tuple[str, str]]:
    out = []
    if CSVProvider().directory.exists():
        for p in sorted(CSVProvider().directory.glob("*.csv")):
            symbol, _, tf = p.stem.rpartition("_")
            if symbol and tf:
                out.append((symbol, tf))
    return out or [("SYNTH", "1h")]


@st.cache_data(show_spinner=False, ttl=300)
def load_series(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        df = CSVProvider().fetch(symbol, timeframe)
    except FileNotFoundError:
        from datafeed import generate_ohlcv

        df = generate_ohlcv(symbol, timeframe, bars=6000, seed=abs(hash((symbol, timeframe))) % 2**32)
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def price_chart(df: pd.DataFrame, symbol: str, timeframe: str, last_n: int = 300,
                show_trades: bool = True, journal: Journal | None = None) -> go.Figure:
    """Candles + EMAs + VWAP + auto S/R + (optional) paper trade markers."""
    ind = compute_all(df).tail(last_n)
    sc = strategy_cfg()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ind.index, open=ind["open"], high=ind["high"], low=ind["low"], close=ind["close"],
        name=symbol, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    for col, color, name in (
        ("ema_fast", "#42a5f5", f"EMA{sc['ema_fast']}"),
        ("ema_mid", "#ffb300", f"EMA{sc['ema_mid']}"),
        ("ema_slow", "#ab47bc", f"EMA{sc['ema_slow']}"),
        ("vwap", "#78909c", "VWAP"),
    ):
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], mode="lines", name=name,
                                 line=dict(width=1.4, color=color)))

    levels = support_resistance_levels(df, lookback=200)
    for p in levels["support"][-3:]:
        fig.add_hline(y=p, line_dash="dot", line_color="green",
                      annotation_text=f"S {p:.2f}", annotation_position="bottom left")
    for p in levels["resistance"][:3]:
        fig.add_hline(y=p, line_dash="dot", line_color="red",
                      annotation_text=f"R {p:.2f}", annotation_position="top left")

    if show_trades and journal is not None:
        trades = journal.all_trades(mode="paper")
        if len(trades):
            t0, t1 = ind.index[0], ind.index[-1]
            trades = trades.copy()
            trades["entry_ts"] = pd.to_datetime(trades["entry_time"])
            win = trades[(trades["entry_ts"] >= t0) & (trades["entry_ts"] <= t1) & (trades["symbol"] == symbol)]
            buys = win[win["direction"] == "BUY"]
            sells = win[win["direction"] == "SELL"]
            if len(buys):
                fig.add_trace(go.Scatter(
                    x=buys["entry_ts"], y=buys["entry"], mode="markers", name="BUY entry",
                    marker=dict(symbol="triangle-up", size=12, color="#26a69a", line=dict(width=1, color="#08372e"))))
            if len(sells):
                fig.add_trace(go.Scatter(
                    x=sells["entry_ts"], y=sells["entry"], mode="markers", name="SELL entry",
                    marker=dict(symbol="triangle-down", size=12, color="#ef5350", line=dict(width=1, color="#4b1414"))))
            closed = win[win["status"] == "closed"].dropna(subset=["exit"])
            if len(closed):
                closed = closed.copy()
                closed["exit_ts"] = pd.to_datetime(closed["exit_time"])
                closed = closed[(closed["exit_ts"] >= t0) & (closed["exit_ts"] <= t1)]
                if len(closed):
                    fig.add_trace(go.Scatter(
                        x=closed["exit_ts"], y=closed["exit"], mode="markers", name="exit",
                        marker=dict(symbol="x", size=9, color="#cfd8dc")))

    fig.update_layout(
        title=f"{symbol} · {timeframe} — structure: {ind['structure'].iloc[-1]}",
        xaxis_rangeslider_visible=False, height=540, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.02), template="plotly_dark",
    )
    return fig


def forecast_chart(df: pd.DataFrame, fc, hist_bars: int = 90) -> go.Figure:
    """History candles + Monte Carlo probability cone for the next N bars."""
    hist = df.tail(hist_bars)
    x_hist = hist.index
    x_fut = fc.future_index
    x_bridge = [x_hist[-1]] + list(x_fut)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_hist, open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
        name="history", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        opacity=0.9,
    ))
    b = fc.bands
    fig.add_trace(go.Scatter(
        x=x_bridge + list(x_bridge[::-1]),
        y=list(b["p90"]) + list(b["p10"][::-1]), fill="toself", fillcolor="rgba(66,165,245,0.15)",
        line=dict(width=0), hoverinfo="skip", name="p10–p90 (80% zone)",
    ))
    fig.add_trace(go.Scatter(
        x=x_bridge + list(x_bridge[::-1]),
        y=list(b["p75"]) + list(b["p25"][::-1]), fill="toself", fillcolor="rgba(66,165,245,0.30)",
        line=dict(width=0), hoverinfo="skip", name="p25–p75 (50% zone)",
    ))
    fig.add_trace(go.Scatter(
        x=x_bridge, y=b["p50"], mode="lines", name="median path",
        line=dict(width=2, dash="dash", color="#42a5f5"),
    ))
    fig.add_hline(y=fc.last_close, line_color="#9e9e9e", line_width=1,
                  annotation_text=f"now {fc.last_close:.2f}")
    fig.update_layout(
        title=(f"🔮 Next {fc.horizon} bars projection — Monte Carlo ({fc.sims:,} sims). "
               "PROBABILITY CONE, not a prediction."),
        xaxis_rangeslider_visible=False, height=540, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.02), template="plotly_dark",
    )
    return fig


def mtf_strip(symbol: str, current_tf: str, base_df: pd.DataFrame) -> list[dict]:
    """Mini per-timeframe summaries: 15m / 1h / 1d (resample from base if needed)."""
    rows = []
    for tf in ["15m", "1h", "1d"]:
        if tf == current_tf:
            df = base_df
        else:
            try:
                df = CSVProvider().fetch(symbol, tf)
            except FileNotFoundError:
                try:
                    if TF_ORDER.index(tf) >= TF_ORDER.index(current_tf):
                        df = resample_ohlcv(base_df, tf)
                    else:
                        continue
                except Exception:
                    continue
        if len(df) < 60:
            continue
        ind = compute_all(df)
        last = ind.iloc[-1]
        rows.append({
            "tf": tf, "close": float(last["close"]),
            "structure": last["structure"] or "n/a",
            "rsi": round(float(last["rsi"]), 1),
            "ema_up": bool(last["ema_fast"] > last["ema_mid"] > last["ema_slow"]),
            "ema_down": bool(last["ema_fast"] < last["ema_mid"] < last["ema_slow"]),
            "atr_pct": round(float(last["atr_pct"]), 2),
            "spark": ind["close"].tail(60).tolist(),
        })
    return rows


def kpi_row(stats: dict, label: str):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trades", stats.get("n_trades", 0))
    c2.metric("Win rate", f"{stats.get('win_rate_pct', 0)}%")
    c3.metric("Expectancy", f"{stats.get('expectancy_r', 0)}R")
    c4.metric("Profit factor", stats.get("profit_factor", 0))
    c5.metric("Max DD", f"{stats.get('max_drawdown_r', 0)}R")
    c6.metric("Losing streak", stats.get("max_losing_streak", 0))
    st.caption(f"{label} — historical statistics, not a guarantee of future results.")


# ------------------------------------------------------------------ #
# setup
# ------------------------------------------------------------------ #
cfg = load_config()
journal = Journal(RUNTIME / "journal.db")
series_choices = available_series()
default_ix = 0
for i, (s, tf) in enumerate(series_choices):
    if s == "SYNTH" and tf == "1h":
        default_ix = i

with st.sidebar:
    st.title("📈 Market Analysis")
    st.caption("Decision-support · paper trading · backtesting")
    symbol, timeframe = st.selectbox(
        "Symbol / timeframe", series_choices, index=default_ix,
        format_func=lambda t: f"{t[0]} · {t[1]}"
    )
    window = st.slider("Chart bars", 100, 600, 300, 50)
    horizon = st.slider("Forecast bars ahead", 5, 60, cfg.get("ai", {}).get("forecast_horizon_bars", 15), 5)

    st.divider()
    st.subheader("Actions")
    col_a, col_b = st.columns(2)
    if col_a.button("Backtest", use_container_width=True, help="Store fresh backtest results (mode=backtest)"):
        with st.spinner("Backtesting…"):
            df_bt = load_series(symbol, timeframe)
            res = Backtester(strategy_cfg()).run(df_bt, symbol, timeframe)
            journal.record_backtest_trades(res.trades, replace=True)
        st.toast(f"Backtest: {res.stats.n_trades} trades, {res.stats.to_dict()['expectancy_r']}R expectancy")
        st.rerun()
    if col_b.button("Paper replay", use_container_width=True, help="Forward-test unseen bars (mode=paper)"):
        with st.spinner("Forward-testing…"):
            df_rp = load_series(symbol, timeframe)
            scanner = LiveScanner(
                strategy_config=cfg["strategy"], provider=None, journal=journal,
                risk_config=cfg["risk"], alerts_config={**cfg["alerts"], "console": False},
            )
            scanner.replay(df_rp, symbol, timeframe, start_frac=0.7, verbose=False)
        st.toast("Paper (forward) trades stored")
        st.rerun()

    st.divider()
    st.subheader("🤖 AI (OpenRouter)")
    key_in = st.text_input("API key", value=get_api_key(cfg), type="password",
                           help="openrouter.ai/keys par free bana lo")
    model_in = st.text_input("Model", value=cfg.get("ai", {}).get("model", "deepseek/deepseek-chat"))
    if st.button("Save key (locally)", use_container_width=True):
        if key_in.strip():
            save_api_key(key_in.strip())
            st.success("Key saved → data/runtime/ai_key.txt (gitignored)")
        else:
            st.warning("Key khaali hai")
    st.session_state.setdefault("ai_model", model_in)

    st.divider()
    st.caption(DISCLAIMER)

# data banner
df = load_series(symbol, timeframe)
src = data_source_of(symbol, timeframe)
if src == "synthetic" or symbol == "SYNTH":
    st.warning("⚠️ Ye SYNTHETIC demo data hai (live feed unavailable). Apne laptop par "
               "`python main.py fetch --symbols AAPL --timeframes 1h --provider yfinance` "
               "chala kar real data lao.", icon="🧪")

tabs = st.tabs(["📊 Market & Signal", "🔮 Forecast", "🤖 AI Assistant", "📂 Positions & Journal", "📈 Stats"])

# ================================================================== #
# TAB 1 — MARKET & SIGNAL
# ================================================================== #
with tabs[0]:
    show_trades = st.checkbox("Show paper trades on chart", value=True)
    st.plotly_chart(price_chart(df, symbol, timeframe, window, show_trades, journal),
                    use_container_width=True)

    # multi-timeframe strip
    st.subheader("Multi-timeframe view")
    strip = mtf_strip(symbol, timeframe, df)
    cols = st.columns(max(1, len(strip)))
    for c, row in zip(cols, strip):
        ema_txt = "🟢 BULL" if row["ema_up"] else ("🔴 BEAR" if row["ema_down"] else "⚪ MIX")
        struct_color = {"uptrend": "🟢", "downtrend": "🔴", "range": "🟡"}.get(row["structure"], "⚪")
        c.markdown(
            f"**{row['tf']}** &nbsp; {struct_color} {row['structure']} &nbsp; {ema_txt} "
            f"&nbsp; RSI {row['rsi']} &nbsp; ATR {row['atr_pct']}%"
        )
        spark = go.Figure(go.Scatter(y=row["spark"], mode="lines",
                                     line=dict(width=1.6, color="#42a5f5")))
        spark.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark",
                            xaxis=dict(visible=False), yaxis=dict(visible=False, ticks=""))
        c.plotly_chart(spark, use_container_width=True)

    # signal + exact trade plan
    strat = TrendPullbackStrategy(strategy_cfg())
    prepared = strat.prepare(df.iloc[:-1])          # last CLOSED bar only
    sig = strat.check_signal(prepared, len(prepared) - 1, symbol, timeframe, source="paper")

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.subheader("Latest signal (last closed bar)")
        if sig is None:
            st.info(f"⏸ **HOLD** — {symbol} {timeframe} par abhi koi signal nahi hai "
                    "(conditions aligned nahi hain). No-trade bhi ek decision hai.")
        else:
            d = sig.direction
            color = "#26a69a" if d == "BUY" else "#ef5350"
            st.markdown(
                f"""<div style="border-left:5px solid {color};padding:14px;background:#1b1e23;
                border-radius:8px">
                <b style="font-size:1.45em;color:{color}">{'▲' if d=='BUY' else '▼'} {d} {symbol}</b>
                &nbsp;[{timeframe}] · {sig.setup} · confidence <b>{sig.confidence:.0f}/100</b><br/>
                Entry <b>{sig.entry:.2f}</b> · SL <b style="color:#ef5350">{sig.stop_loss:.2f}</b> ·
                Target <b style="color:#26a69a">{sig.target:.2f}</b> · R:R 1:{sig.rr:.1f} ·
                structure: {sig.market_condition}<br/>
                <span style="color:#9e9e9e">{' · '.join(sig.reasons)}</span></div>""",
                unsafe_allow_html=True,
            )
        st.caption(DISCLAIMER)
    with col2:
        st.subheader("Exact trade plan (kab & kaise)")
        if sig is None:
            watch = mtf_strip(symbol, timeframe, df)
            tips = [
                "Signal nahi hai = trade nahi. Ye discipline hi edge ka hissa hai.",
                f"AI Assistant tab me 'kya watch karna chahiye?' poochh lo.",
                "Backtest + Paper replay buttons sidebar me — stats refresh karke dekho.",
            ]
            for t in tips:
                st.write("•", t)
        else:
            risk_cfg = cfg["risk"]
            risk_amt = risk_cfg["account_size"] * risk_cfg["risk_per_trade_pct"] / 100
            per_unit = abs(sig.entry - sig.stop_loss)
            qty = risk_amt / per_unit if per_unit > 0 else 0
            plan = (
                f"1) DIRECTION : {sig.direction} ({sig.setup}, confidence {sig.confidence:.0f}/100)\n"
                f"2) ENTRY     : {sig.entry:.2f} (signal bar ke close par / agle bar open)\n"
                f"3) STOP-LOSS : {sig.stop_loss:.2f}  ← SL ke bina trade MAT lo, kabhi nahi\n"
                f"4) TARGET    : {sig.target:.2f} (1:{sig.rr:.1f} risk:reward)\n"
                f"5) QUANTITY  : ~{qty:.2f} units (risk {risk_cfg['risk_per_trade_pct']}% = "
                f"{risk_cfg['account_size']:.0f} × {risk_cfg['risk_per_trade_pct']/100:.4f})\n"
                f"6) EXIT RULES: SL hit → bahar. Target hit → book. {cfg['strategy']['max_bars_in_trade']} bars me "
                f"kuch nahi hua → timeout exit.\n"
                f"7) MAX RISK/DAY: {risk_cfg['max_daily_loss_pct']}% loss hote hi naye signals block."
            )
            st.code(plan, language="text")
            st.caption("Ye paper-trade plan hai — real order placement software me built-in nahi hai "
                       "(by design). " + DISCLAIMER)

# ================================================================== #
# TAB 2 — FORECAST
# ================================================================== #
with tabs[1]:
    st.subheader(f"Next {horizon} bars — Monte Carlo probability cone ({symbol} · {timeframe})")
    st.caption(
        "Ye PREDICTION nahi hai — 2,000 simulated paths ka statistics hai, is symbol ke apne "
        "historical returns se bootstrap karke. 80% zone ke bahar jaana unlikely hai, impossible nahi."
    )
    prepared_full = strat.prepare(df)
    fc = monte_carlo_forecast(prepared_full, horizon=horizon, sims=2000, seed=None)
    st.plotly_chart(forecast_chart(df, fc, hist_bars=min(120, window)), use_container_width=True)

    s = fc.summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P(up) at horizon", f"{s['p_up_pct']}%")
    c2.metric("Median close", s["median"])
    c3.metric("80% range", f"{s['p10']} – {s['p90']}")
    c4.metric("50% range", f"{s['p25']} – {s['p75']}")
    c5.metric("Last close", s["last_close"])

    st.markdown("#### Reading: ye cone kaise use karein")
    st.markdown(
        "• Cone **wide** = uncertainty zyada → size chhota rakho ya skip karo.\n"
        "• Median trend ki direction me hai + structure aligned = setup ka probability context strong.\n"
        "• Target/SL levels cone ke andar kahan fall karte hain, ye AI Assistant se poochho — "
        "wo `prob_target_before_sl` bhi calculate karta hai.\n"
        "• **80% zone ke bahar jaana unlikely hai — lekin impossible nahi.** Tail risk hamesha hota hai."
    )

    sig_now = sig
    if sig_now is not None:
        p_t = fc.first_touch(sig_now.target, sig_now.stop_loss)
        if p_t == p_t:
            ok = 0 < p_t < 1
            st.info(
                f"Active {sig_now.direction} signal — next {horizon} bars me (simulated): "
                f"**P(target {sig_now.target:.2f} pehle chhue) ≈ {(1-p_t)*100:.0f}%** · "
                f"**P(SL {sig_now.stop_loss:.2f} pehle chhue) ≈ {p_t*100:.0f}%**"
            )
    st.caption(DISCLAIMER + " Monte Carlo = simulated scenarios, real future nahi.")

    with st.expander("Band table (per-bar percentiles)"):
        st.dataframe(fc.band_table().tail(20), use_container_width=True, hide_index=True)

# ================================================================== #
# TAB 3 — AI ASSISTANT
# ================================================================== #
with tabs[2]:
    st.subheader("🤖 AI Trading Assistant (OpenRouter)")
    advisor = Advisor({**cfg.get("ai", {}), "model": st.session_state.get("ai_model", cfg.get("ai", {}).get("model"))},
                      api_key=get_api_key(cfg) or None)
    if advisor.has_key:
        st.success(f"AI connected · model: `{advisor.model}`")
    else:
        st.info(
            "OpenRouter key nahi mili — **offline rule-based mode** chal raha hai (ye bhi concrete "
            "trade plan deta hai). Full AI ke liye: [openrouter.ai/keys](https://openrouter.ai/keys) "
            "se key banao → sidebar me paste karke Save karo. Phir bhi yaad rakho: AI opinion hai, "
            "guarantee nahi.", icon="🔌"
        )

    # context (rebuilt fresh each ask)
    fc_ctx = monte_carlo_forecast(prepared_full, horizon=horizon, sims=800, seed=7)
    recent = journal.all_trades(mode="paper", closed_only=True).tail(5)
    ctx = build_market_context(
        prepared, symbol, timeframe, signal=sig, forecast=fc_ctx, risk_config=cfg["risk"],
        recent_trades=recent if len(recent) else None,
    )
    with st.expander("Market context AI ko ye data milta hai (JSON)"):
        st.code(render_context(ctx), language="json")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": ("Namaste! Main aapka trading analyst hoon. Market ka poora context mere paas hai "
                        "(candles, indicators, S/R levels, structure, Monte Carlo forecast, active signal). "
                        "Poochho: *'kya abhi trade lena chahiye?'*, *'risk kya hai?'*, "
                        "*'next 15 bars ka outlook?'* — Hinglish ya English, dono chalega."),
        }]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    quick = st.columns(4)
    q_prompts = [
        ("Kya abhi trade lena chahiye?", "kya abhi trade lena chahiye? exact entry, SL, target ke saath"),
        ("Risk kya hai?", "is setup me kya risks hain? kahan galat ho sakta hai?"),
        (f"Next {horizon} bars?", f"next {horizon} bars ka outlook kya hai? probability cone ke hisaab se"),
        ("Position size?", "mera account size ke hisaab se kitni quantity leni chahiye?"),
    ]
    queued = None
    for c, (label, prompt) in zip(quick, q_prompts):
        if c.button(label, use_container_width=True):
            queued = prompt

    user_q = st.chat_input("Apna sawaal likho… (Hinglish OK)")
    question = queued or user_q
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon…" if advisor.has_key else "Rule engine chala raha hoon…"):
                answer = advisor.chat(question, ctx, st.session_state.messages[:-1])
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ================================================================== #
# TAB 4 — POSITIONS & JOURNAL
# ================================================================== #
with tabs[3]:
    mode = st.radio("Journal mode", ["paper", "backtest"], horizontal=True,
                    help="Backtest (historical) aur paper (forward) results hamesha separate rehte hain.")
    st.subheader(f"Open paper positions")
    open_df = journal.open_trades(mode="paper")
    if len(open_df) == 0:
        st.info("Koi open paper position nahi. `scan` command ya replay chalao.")
    else:
        latest = df["close"].iloc[-1]
        rows = []
        for _, t in open_df.iterrows():
            cur = float(latest)
            entry, sl = float(t["entry"]), float(t["stop_loss"])
            risk = abs(entry - sl)
            move = (cur - entry) if t["direction"] == "BUY" else (entry - cur)
            rows.append({
                "#": t["id"], "symbol": t["symbol"], "tf": t["timeframe"], "dir": t["direction"],
                "qty": t["qty"], "entry": entry, "SL": sl, "target": t["target"],
                "last": round(cur, 2), "open R": round(move / risk, 2) if risk else 0.0,
                "conf": t["confidence"], "opened": t["entry_time"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader(f"Trade journal — mode: {mode}")
    trades = journal.all_trades(mode=mode, closed_only=False)
    if len(trades) == 0:
        st.info(f"No {mode} trades yet.")
    else:
        cols = ["id", "symbol", "timeframe", "direction", "confidence", "market_condition",
                "entry_time", "entry", "stop_loss", "target", "exit_time", "exit",
                "exit_reason", "result_r", "status"]
        show = [c for c in cols if c in trades.columns]
        st.dataframe(trades[show].sort_values("id", ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button("Download CSV", trades[show].to_csv(index=False).encode(),
                           f"journal_{mode}.csv", "text/csv")

    st.subheader("Recent signals")
    sigs_log = journal.recent_signals(mode=mode, limit=15)
    if len(sigs_log):
        st.dataframe(sigs_log[["ts", "symbol", "timeframe", "direction", "entry", "stop_loss",
                               "target", "confidence", "setup"]],
                     use_container_width=True, hide_index=True)
    st.caption("Paper positions only — ye software real orders place nahi karta. " + DISCLAIMER)

# ================================================================== #
# TAB 5 — STATS
# ================================================================== #
with tabs[4]:
    mode_s = st.radio("Stats mode", ["paper", "backtest"], horizontal=True, key="stats_mode")
    st.subheader(f"Performance — {mode_s}")
    stats = journal.stats(mode_s)
    kpi_row(stats, mode_s)

    left, right = st.columns(2)
    with left:
        eq = journal.equity_curve(mode_s)
        if len(eq):
            fig = go.Figure(go.Scatter(y=eq, mode="lines", name="cum R"))
            fig.update_layout(title="Equity curve (R-multiples)", height=320,
                              margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        closed = journal.all_trades(mode=mode_s, closed_only=True)
        if len(closed):
            fig = go.Figure(go.Histogram(x=closed["result_r"], nbinsx=30, name="R"))
            fig.update_layout(title="R-multiple distribution", height=320,
                              margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    period = st.radio("Report period", ["weekly", "monthly"], horizontal=True)
    rep = journal.report(period=period, mode=mode_s)
    if len(rep):
        st.subheader(f"{period.capitalize()} report")
        st.dataframe(rep[["period", "n_trades", "win_rate_pct", "expectancy_r",
                          "profit_factor", "max_drawdown_r"]], use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades to report yet.")

    if len(closed):
        st.subheader("One-variable analysis (overfitting se bachne ke liye ek-ek variable badlo)")
        seg_col = st.selectbox("Segment by", ["market_condition", "direction", "symbol",
                                              "setup", "exit_reason"])
        closed = closed.copy()
        closed["hour"] = pd.to_datetime(closed["entry_time"]).dt.hour
        tbl = segment_analysis(closed, seg_col) if seg_col in closed.columns else pd.DataFrame()
        if len(tbl):
            st.dataframe(tbl[[seg_col, "n_trades", "win_rate_pct", "expectancy_r",
                              "profit_factor"]], use_container_width=True, hide_index=True)
            st.caption("Kam trades wale segments noise hote hain — n_trades check karo pehle.")

st.sidebar.divider()
st.sidebar.caption("📊 " + DISCLAIMER)
