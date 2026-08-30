#!/usr/bin/env python3
"""Market Analysis & Trade Signal Software — CLI entry point.

Decision-support + backtesting + paper-trading tool. It never places real
orders. Every signal output carries the disclaimer:
    "Not financial advice. Based on historical statistical edge, not a guarantee."

Quickstart
----------
  python main.py fetch --symbols SYNTH --timeframes 1h --provider synthetic --bars 6000
  python main.py backtest --symbol SYNTH --timeframe 1h --provider csv
  python main.py sweep --symbol SYNTH --timeframe 1h --variable risk_reward --values 1,1.5,2,2.5,3
  python main.py replay --symbol SYNTH --timeframe 1h          # forward test
  python main.py scan --symbols SYNTH --timeframes 1h --once   # paper scan
  python main.py report --mode paper
  python main.py position --entry 100 --stop 98 --account 10000 --risk-pct 0.5
  python main.py dashboard
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "config.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_stats(stats_dict: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    for k, v in stats_dict.items():
        print(f"  {k:20s} {v}")
    print("\nNot financial advice. Based on historical statistical edge, not a guarantee.")


# ---------------------------------------------------------------------- #
def cmd_fetch(args):
    from datafeed import get_provider, save_csv

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    provider = get_provider(args.provider, bars=args.bars) if args.provider != "yfinance" else get_provider("yfinance")

    for symbol in symbols:
        for tf in timeframes:
            df = provider.fetch(symbol, tf, period=args.period)
            path = save_csv(df, symbol, tf)
            print(f"[fetch] {symbol} {tf}: {len(df)} bars -> {path}")


# ---------------------------------------------------------------------- #
def cmd_backtest(args):
    from datafeed import fetch_ohlcv
    from engine.backtester import Backtester
    from journal.logger import Journal

    config = load_config()
    overrides = {"entry_type": args.entry_type, "sl_type": args.sl_type, "risk_reward": args.rr}
    cfg = {**config["strategy"], **{k: v for k, v in overrides.items() if v is not None}}

    df = fetch_ohlcv(args.symbol, args.timeframe, provider=args.provider, period=args.period)
    bt = Backtester(cfg, fee_pct=config["backtest"]["fee_pct"], slippage_pct=config["backtest"]["slippage_pct"])
    res = bt.run(df, args.symbol, args.timeframe)

    print(f"\nBacktest: {args.symbol} {args.timeframe} | bars={len(df)} "
          f"| entry={cfg['entry_type']} sl={cfg['sl_type']} RR=1:{cfg['risk_reward']}")
    print(res.stats.summary())
    print("\n[engine] DISCLAIMER: " + "Backtest = historical simulation only. "
          "Not financial advice. Based on historical statistical edge, not a guarantee.")

    if args.analyze:
        for col in ("market_condition", "direction", "hour"):
            tbl = res.analyze_by(col)
            if len(tbl):
                print(f"\n--- by {col} ---")
                print(tbl[[col, "n_trades", "win_rate_pct", "expectancy_r"]].to_string(index=False))

    if res.stats.n_trades < 100:
        print(f"\n[note] only {res.stats.n_trades} trades — spec requires 100+ for a "
              "meaningful read. Use more bars/symbols (e.g. --bars 8000 or multiple symbols).")

    journal = Journal()
    journal.record_backtest_trades(res.trades, replace=True)
    print(f"[journal] backtest trades stored separately (mode='backtest') in {journal.db_path}")

    if args.csv:
        out = Path(args.csv)
        res.trades.to_csv(out, index=False)
        print(f"[export] trades -> {out}")
    if args.json:
        out = Path(args.json)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"stats": res.stats.to_dict(), "config": cfg, "symbol": args.symbol,
                       "timeframe": args.timeframe}, f, indent=2)
        print(f"[export] stats -> {out}")


# ---------------------------------------------------------------------- #
def cmd_sweep(args):
    from datafeed import fetch_ohlcv
    from engine.backtester import sweep_variable
    from strategy import DEFAULT_CONFIG

    config = load_config()
    base = {**DEFAULT_CONFIG, **config["strategy"]}
    df = fetch_ohlcv(args.symbol, args.timeframe, provider=args.provider, period=args.period)
    values = [v.strip() for v in args.values.split(",")]
    typed = []
    for v in values:
        typed.append(float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v)

    tbl = sweep_variable(df, args.symbol, args.timeframe, base, args.variable, typed,
                         fee_pct=config["backtest"]["fee_pct"],
                         slippage_pct=config["backtest"]["slippage_pct"])
    print(f"\nOne-variable sweep: {args.variable} (everything else fixed)\n")
    print(tbl.to_string(index=False))
    print("\nRobustness check: prefer plateaus across neighbouring values, not one lucky spike "
          "(overfitting guard). Not financial advice.")


# ---------------------------------------------------------------------- #
def cmd_scan(args):
    from datafeed import get_provider
    from engine.live_scanner import LiveScanner

    config = load_config()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()] or config["timeframes"]
    provider = get_provider(args.provider, bars=args.bars)
    scanner = LiveScanner(
        strategy_config=config["strategy"],
        provider=provider,
        risk_config=config["risk"],
        alerts_config=config["alerts"],
    )
    if args.once:
        fired = scanner.scan_once(symbols, timeframes)
        print(f"\n[scan] pass complete — {len(fired)} signal(s) evaluated")
        for s in fired:
            print(s.alert_text())
    else:
        scanner.run_loop(symbols, timeframes, poll_seconds=config["paper"]["poll_seconds"])


# ---------------------------------------------------------------------- #
def cmd_replay(args):
    from datafeed import fetch_ohlcv
    from engine.live_scanner import LiveScanner
    from journal.logger import Journal

    config = load_config()
    df = fetch_ohlcv(args.symbol, args.timeframe, provider=args.provider, period=args.period)
    scanner = LiveScanner(
        strategy_config=config["strategy"],
        provider=None,
        journal=Journal(),
        risk_config=config["risk"],
        alerts_config={**config["alerts"], "console": False},  # keep logs readable
    )
    print(f"[replay] forward test {args.symbol} {args.timeframe}: bars {int(len(df)*args.start):d}..{len(df)} "
          f"(unseen data, paper pipeline)")
    trades = scanner.replay(df, args.symbol, args.timeframe, start_frac=args.start)
    stats = scanner.journal.stats("paper")
    _print_stats(stats, f"PAPER (forward) results — {args.symbol} {args.timeframe}")
    if args.csv and len(trades):
        trades.to_csv(args.csv, index=False)
        print(f"[export] paper trades -> {args.csv}")


# ---------------------------------------------------------------------- #
def cmd_report(args):
    from journal.logger import Journal

    journal = Journal()
    mode = args.mode
    print(f"\n=== TRADE REPORT ({mode}) — {journal.db_path} ===")
    stats = journal.stats(mode)
    _print_stats_stats(stats)
    for period in ("weekly", "monthly"):
        tbl = journal.report(period=period, mode=mode)
        if len(tbl):
            print(f"\n--- {period} ---")
            cols = [c for c in ("period", "n_trades", "win_rate_pct", "expectancy_r",
                                "profit_factor", "max_drawdown_r") if c in tbl.columns]
            print(tbl[cols].tail(8).to_string(index=False))
    print("\nBacktest and paper results are kept SEPARATE by design. "
          "Not financial advice. Based on historical statistical edge, not a guarantee.")


def _print_stats_stats(stats: dict) -> None:
    for k, v in stats.items():
        print(f"  {k:20s} {v}")


# ---------------------------------------------------------------------- #
def cmd_position(args):
    from risk import position_size

    size = position_size(entry=args.entry, stop_loss=args.stop,
                         account_size=args.account, risk_per_trade_pct=args.risk_pct)
    print(f"\nPosition size for entry={args.entry} SL={args.stop}")
    print(f"  qty               {size.qty:g}")
    print(f"  risk per unit     {size.risk_per_unit:.4f}")
    print(f"  risk amount       ${size.risk_amount:.2f} ({args.risk_pct}% of {args.account:.0f})")
    print(f"  position value    ${size.position_value:.2f}")
    for n in size.notes:
        print(f"  note: {n}")
    print("\nNot financial advice. Based on historical statistical edge, not a guarantee.")


# ---------------------------------------------------------------------- #
def cmd_forecast(args):
    from datafeed import fetch_ohlcv
    from indicators import compute_all
    from indicators.forecast import monte_carlo_forecast

    df = fetch_ohlcv(args.symbol, args.timeframe, provider=args.provider, period=args.period)
    prepared = compute_all(df)
    fc = monte_carlo_forecast(prepared, horizon=args.horizon, sims=args.sims, seed=args.seed)
    s = fc.summary()
    print(f"\nMonte Carlo projection: {args.symbol} {args.timeframe} — next {args.horizon} bars "
          f"({args.sims} sims)")
    print(f"  last close      {s['last_close']}")
    print(f"  median          {s['median']}")
    print(f"  50% range       {s['p25']} – {s['p75']}")
    print(f"  80% range       {s['p10']} – {s['p90']}")
    print(f"  P(up)           {s['p_up_pct']}%")
    print("\n" + ("Probability cone from historical returns — NOT a prediction. "
          "Not financial advice. Based on historical statistical edge, not a guarantee."))
    if args.csv:
        fc.band_table().to_csv(args.csv, index=False)
        print(f"[export] bands -> {args.csv}")


# ---------------------------------------------------------------------- #
def cmd_ask(args):
    from datafeed import fetch_ohlcv
    from ai.advisor import Advisor, build_market_context, render_context
    from indicators import compute_all
    from indicators.forecast import monte_carlo_forecast
    from journal.logger import Journal
    from strategy.trend_pullback import TrendPullbackStrategy

    config = load_config()
    df = fetch_ohlcv(args.symbol, args.timeframe, provider=args.provider, period=args.period)
    strat = TrendPullbackStrategy(config["strategy"])
    prepared = strat.prepare(df.iloc[:-1])
    sig = strat.check_signal(prepared, len(prepared) - 1, args.symbol, args.timeframe, source="paper")
    fc = monte_carlo_forecast(compute_all(df), horizon=args.horizon, sims=1500, seed=7)
    recent = Journal().all_trades(mode="paper", closed_only=True).tail(5)
    ctx = build_market_context(prepared, args.symbol, args.timeframe, signal=sig,
                               forecast=fc, risk_config=config["risk"],
                               recent_trades=recent if len(recent) else None)
    advisor = Advisor(config)
    mode = f"OpenRouter ({advisor.model})" if advisor.has_key else "offline rule-based"
    sig_txt = "NONE (HOLD)" if sig is None else f"{sig.direction} conf {round(sig.confidence)}"
    print(f"\n[ai] advisor: {mode} | {args.symbol} {args.timeframe} @ {ctx['price']['close']} | signal: {sig_txt}")
    print("-" * 60)
    print(advisor.chat(args.question, ctx))
    if args.context:
        print("\n--- context sent to AI ---")
        print(render_context(ctx))


# ---------------------------------------------------------------------- #
def cmd_dashboard(args):
    print("Starting Streamlit dashboard (Ctrl-C to stop)...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
           "--server.port", str(args.port), "--server.address", "0.0.0.0"]
    subprocess.run(cmd, cwd=str(ROOT))


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("fetch", help="download/cache OHLCV data")
    sp.add_argument("--symbols", required=True, help="comma separated, e.g. AAPL,BTC-USD")
    sp.add_argument("--timeframes", required=True, help="e.g. 15m,1h,1d")
    sp.add_argument("--provider", default=None, choices=["yfinance", "csv", "synthetic"])
    sp.add_argument("--period", default="2y")
    sp.add_argument("--bars", type=int, default=6000, help="bars for synthetic provider")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("backtest", help="run historical backtest")
    sp.add_argument("--symbol", default="SYNTH")
    sp.add_argument("--timeframe", default="1h")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--period", default="2y")
    sp.add_argument("--entry-type", choices=["pullback", "breakout"], default=None)
    sp.add_argument("--sl-type", choices=["structure", "atr"], default=None)
    sp.add_argument("--rr", type=float, default=None, help="risk:reward target ratio")
    sp.add_argument("--analyze", action="store_true", help="segment results by condition")
    sp.add_argument("--csv", default=None, help="export trades CSV")
    sp.add_argument("--json", default=None, help="export stats JSON")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("sweep", help="one-variable-at-a-time test")
    sp.add_argument("--symbol", default="SYNTH")
    sp.add_argument("--timeframe", default="1h")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--period", default="2y")
    sp.add_argument("--variable", required=True, help="config key, e.g. risk_reward / entry_type / sl_type")
    sp.add_argument("--values", required=True, help="comma separated, e.g. 1,1.5,2,2.5,3 or pullback,breakout")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("scan", help="paper-trade scanner (live loop or single pass)")
    sp.add_argument("--symbols", required=True)
    sp.add_argument("--timeframes", required=True)
    sp.add_argument("--provider", default=None)
    sp.add_argument("--bars", type=int, default=6000)
    sp.add_argument("--once", action="store_true", help="single scan pass (default: loop)")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("replay", help="forward test on unseen bars through paper pipeline")
    sp.add_argument("--symbol", default="SYNTH")
    sp.add_argument("--timeframe", default="1h")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--period", default="2y")
    sp.add_argument("--start", type=float, default=0.7, help="fraction of data used for warmup")
    sp.add_argument("--csv", default=None)
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("report", help="journal stats + weekly/monthly report")
    sp.add_argument("--mode", default="paper", choices=["paper", "backtest"])
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("position", help="position size calculator")
    sp.add_argument("--entry", type=float, required=True)
    sp.add_argument("--stop", type=float, required=True)
    sp.add_argument("--account", type=float, default=10000)
    sp.add_argument("--risk-pct", type=float, default=0.5)
    sp.set_defaults(func=cmd_position)

    sp = sub.add_parser("forecast", help="Monte Carlo projection of next N bars")
    sp.add_argument("--symbol", default="SYNTH")
    sp.add_argument("--timeframe", default="1h")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--period", default="2y")
    sp.add_argument("--horizon", type=int, default=15, help="bars ahead (15 bars on 1m = 15 minutes)")
    sp.add_argument("--sims", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--csv", default=None)
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("ask", help="ask the AI advisor (OpenRouter) or offline fallback")
    sp.add_argument("--symbol", default="SYNTH")
    sp.add_argument("--timeframe", default="1h")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--period", default="2y")
    sp.add_argument("--horizon", type=int, default=15)
    sp.add_argument("--question", required=True, help="e.g. 'kya abhi trade lena chahiye?'")
    sp.add_argument("--context", action="store_true", help="print the JSON context too")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("dashboard", help="launch Streamlit dashboard")
    sp.add_argument("--port", type=int, default=8501)
    sp.set_defaults(func=cmd_dashboard)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
