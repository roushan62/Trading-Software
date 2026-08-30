"""Streamlit dashboard: charts + live signal panel + open positions + journal + stats.

Run from the repo root:
    streamlit run dashboard/app.py
or: python main.py dashboard

Decision-support tool only — every page carries the disclaimer.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datafeed import generate_ohlcv
from datafeed.csv_provider import CSVProvider
from engine.backtester import Backtester, segment_analysis
from engine.live_scanner import LiveScanner
from indicators import compute_all
from indicators.support_resistance import support_resistance_levels
from journal.logger import Journal
from strategy import DEFAULT_CONFIG, DISCLAIMER
from strategy.trend_pullback import TrendPullbackStrategy

st.set_page_config(page_title="Market Analysis & Trade Signals", page_icon="📈", layout="wide")

RUNTIME = ROOT / "data" / "runtime"


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def load_config() -> dict:
    import json

    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def available_series() -> list[tuple[str, str]]:
    """(symbol, timeframe) pairs found in data/historical/."""
    out = []
    if CSVProvider().directory.exists():
        for p in sorted(CSVProvider().directory.glob("*.csv")):
            stem = p.stem
            symbol, _, tf = stem.rpartition("_")
            if symbol and tf:
                out.append((symbol, tf))
    return out or [("SYNTH", "1h")]


def load_series(symbol: str, timeframe: str, bars: int = 400) -> pd.DataFrame:
    try:
        df = CSVProvider().fetch(symbol, timeframe)
    except FileNotFoundError:
        df = generate_ohlcv(symbol, timeframe, bars=4000, seed=abs(hash((symbol, timeframe))) % 2**32)
    return df


def price_chart(df: pd.DataFrame, symbol: str, timeframe: str, last_n: int = 300) -> go.Figure:
    """Candles + EMAs + VWAP + auto S/R levels."""
    ind = compute_all(df.tail(last_n))
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
        fig.add_hline(y=p, line_dash="dot", line_color="green", annotation_text=f"S {p:.2f}",
                      annotation_position="bottom left")
    for p in levels["resistance"][:3]:
        fig.add_hline(y=p, line_dash="dot", line_color="red", annotation_text=f"R {p:.2f}",
                      annotation_position="top left")

    fig.update_layout(
        title=f"{symbol} · {timeframe} — structure: {ind['structure'].iloc[-1]}",
        xaxis_rangeslider_visible=False, height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.02), template="plotly_dark",
    )
    return fig


def strategy_cfg() -> dict:
    return {**DEFAULT_CONFIG, **load_config()["strategy"]}


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
        "Symbol / timeframe", series_choices, index=default_ix, format_func=lambda t: f"{t[0]} · {t[1]}"
    )
    window = st.slider("Chart bars", 100, 600, 300, 50)
    mode = st.radio("Journal mode", ["paper", "backtest"], horizontal=True,
                    help="Backtest (historical) and paper (forward) results are always kept separate.")
    st.divider()
    st.subheader("Seed demo data")
    if st.button("① Run backtest (stores as backtest)", use_container_width=True):
        with st.spinner("Backtesting…"):
            df = load_series(symbol, timeframe, bars=9000)
            bt = Backtester(strategy_cfg())
            res = bt.run(df, symbol, timeframe)
            journal.record_backtest_trades(res.trades, replace=True)
        st.success(f"Backtest stored: {res.stats.n_trades} trades")
        st.rerun()
    if st.button("② Run paper replay (stores as paper)", use_container_width=True):
        with st.spinner("Forward-testing…"):
            df = load_series(symbol, timeframe, bars=9000)
            scanner = LiveScanner(
                strategy_config=cfg["strategy"], provider=None, journal=journal,
                risk_config=cfg["risk"], alerts_config={**cfg["alerts"], "console": False},
            )
            scanner.replay(df, symbol, timeframe, start_frac=0.7, verbose=False)
        st.success("Paper (forward) trades stored")
        st.rerun()
    st.divider()
    st.caption(DISCLAIMER)

tabs = st.tabs(["📊 Chart & Signals", "📂 Open Positions", "📓 Journal", "📈 Stats & Reports"])

# ------------------------------------------------------------------ #
# Tab 1: chart + signals
# ------------------------------------------------------------------ #
with tabs[0]:
    df = load_series(symbol, timeframe)
    st.plotly_chart(price_chart(df, symbol, timeframe, window), use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Latest signal (last closed bar)")
        strat = TrendPullbackStrategy(strategy_cfg())
        prepared = strat.prepare(df.iloc[:-1])  # drop possibly-forming last bar
        sig = strat.check_signal(prepared, len(prepared) - 1, symbol, timeframe, source="paper")
        if sig is None:
            st.info(f"No signal on {symbol} {timeframe} at the last closed bar — conditions not aligned. "
                    "That is a HOLD.")
        else:
            d = sig.direction
            color = "#26a69a" if d == "BUY" else "#ef5350"
            st.markdown(
                f"""<div style="border-left:5px solid {color};padding:12px;background:#1e1e1e;border-radius:6px">
                <b style="font-size:1.3em;color:{color}">{'▲' if d=='BUY' else '▼'} {d} {symbol}</b>
                &nbsp;[{timeframe}] · {sig.setup} · confidence {sig.confidence:.0f}/100<br/>
                Entry <b>{sig.entry:.2f}</b> · SL <b style="color:#ef5350">{sig.stop_loss:.2f}</b> ·
                Target <b style="color:#26a69a">{sig.target:.2f}</b> · R:R 1:{sig.rr:.1f} ·
                structure: {sig.market_condition}<br/><span style="color:#9e9e9e">
                {' · '.join(sig.reasons)}</span></div>""",
                unsafe_allow_html=True,
            )
        st.caption(DISCLAIMER)
    with col2:
        st.subheader("Market snapshot")
        last = prepared.iloc[-1]
        st.write({
            "Close": round(float(last["close"]), 2),
            "RSI(14)": round(float(last["rsi"]), 1),
            "ATR%": round(float(last["atr_pct"]), 2),
            "Volume z": round(float(last["vol_z"]), 2),
            "Structure": last["structure"] or "n/a",
            "VWAP": round(float(last["vwap"]), 2),
        })

# ------------------------------------------------------------------ #
# Tab 2: open positions
# ------------------------------------------------------------------ #
with tabs[1]:
    st.subheader(f"Open paper positions ({symbol} · {timeframe})")
    open_df = journal.open_trades(mode="paper")
    if len(open_df) == 0:
        st.info("No open paper positions. Signals appear here when the scanner fires "
                "(run `python main.py scan ...` or seed demo data from the sidebar).")
    else:
        latest = df["close"].iloc[-1]
        rows = []
        for _, t in open_df.iterrows():
            cur = float(latest)
            entry, sl = float(t["entry"]), float(t["stop_loss"])
            risk = abs(entry - sl)
            mfe_dist = (cur - entry) if t["direction"] == "BUY" else (entry - cur)
            rows.append({
                "#": t["id"], "symbol": t["symbol"], "tf": t["timeframe"], "dir": t["direction"],
                "qty": t["qty"], "entry": entry, "SL": sl, "target": t["target"],
                "last": round(cur, 2), "open R": round(mfe_dist / risk, 2) if risk else 0.0,
                "confidence": t["confidence"], "opened": t["entry_time"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Paper positions only — this software never places real orders. " + DISCLAIMER)

# ------------------------------------------------------------------ #
# Tab 3: journal
# ------------------------------------------------------------------ #
with tabs[2]:
    st.subheader(f"Trade journal — mode: {mode} (kept separate from the other mode by design)")
    trades = journal.all_trades(mode=mode, closed_only=False)
    if len(trades) == 0:
        st.info(f"No {mode} trades yet. Seed demo data from the sidebar or run the CLI.")
    else:
        c1, c2 = st.columns(2)
        dir_filter = c1.multiselect("Direction", sorted(trades["direction"].dropna().unique()))
        sym_filter = c2.multiselect("Symbol", sorted(trades["symbol"].dropna().unique()))
        view = trades
        if dir_filter:
            view = view[view["direction"].isin(dir_filter)]
        if sym_filter:
            view = view[view["symbol"].isin(sym_filter)]
        cols = ["id", "symbol", "timeframe", "direction", "confidence", "market_condition",
                "entry_time", "entry", "stop_loss", "target", "exit_time", "exit",
                "exit_reason", "result_r", "status"]
        show = [c for c in cols if c in view.columns]
        st.dataframe(view[show].sort_values("id", ascending=False), use_container_width=True,
                     hide_index=True)
        csv = view[show].to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"journal_{mode}.csv", "text/csv")

    st.subheader("Recent signals")
    sigs = journal.recent_signals(mode=mode, limit=15)
    if len(sigs):
        st.dataframe(sigs[["ts", "symbol", "timeframe", "direction", "entry", "stop_loss",
                           "target", "confidence", "setup"]], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Tab 4: stats & reports
# ------------------------------------------------------------------ #
with tabs[3]:
    st.subheader(f"Performance — {mode}")
    stats = journal.stats(mode)
    kpi_row(stats, mode)

    left, right = st.columns(2)
    with left:
        eq = journal.equity_curve(mode)
        if len(eq):
            fig = go.Figure(go.Scatter(y=eq, mode="lines", name="cum R"))
            fig.update_layout(title="Equity curve (R-multiples)", height=320,
                              margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        closed = journal.all_trades(mode=mode, closed_only=True)
        if len(closed):
            fig = go.Figure(go.Histogram(x=closed["result_r"], nbinsx=30, name="R"))
            fig.update_layout(title="R-multiple distribution", height=320,
                              margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    period = st.radio("Report period", ["weekly", "monthly"], horizontal=True)
    rep = journal.report(period=period, mode=mode)
    if len(rep):
        st.subheader(f"{period.capitalize()} report")
        st.dataframe(rep[["period", "n_trades", "win_rate_pct", "expectancy_r",
                          "profit_factor", "max_drawdown_r"]], use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades to report yet.")

    if len(closed):
        st.subheader("One-variable analysis (avoid overfitting: change one thing at a time)")
        seg_col = st.selectbox("Segment by", ["market_condition", "direction", "symbol",
                                              "setup", "exit_reason"])
        closed = closed.copy()
        closed["hour"] = pd.to_datetime(closed["entry_time"]).dt.hour
        tbl = segment_analysis(closed, seg_col) if seg_col in closed.columns else pd.DataFrame()
        if len(tbl):
            st.dataframe(tbl[[seg_col, "n_trades", "win_rate_pct", "expectancy_r",
                              "profit_factor"]], use_container_width=True, hide_index=True)
            st.caption("Segments with very few trades are noise — check n_trades before trusting any edge.")

st.caption(DISCLAIMER)
