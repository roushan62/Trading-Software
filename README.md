# Market Analysis & Trade Signal Software

A complete decision-support, backtesting and paper-trading tool. It continuously
analyzes price data across multiple timeframes and generates **BUY / SELL / HOLD**
signals with entry, stop-loss, target, timeframe and a confidence score — plus the
statistics to judge whether the setup actually has an edge.

> ⚠️ **This is not a guaranteed-profit system.** No software can predict the market
> with certainty. All outputs show probability/statistics, never promises.
> **Not financial advice. Based on historical statistical edge, not a guarantee.**
> This tool **never places real-money orders** — every "trade" is a paper trade
> unless you deliberately integrate your own broker API and accept full responsibility.

---

## Feature Overview

| Module | What it does |
|---|---|
| **Data layer** (`datafeed/`) | yfinance / CSV / synthetic providers, 1m–1D timeframes, resampling, local CSV caching |
| **Indicator engine** (`indicators/`) | EMA(20/50/200), session VWAP, RSI(14), ATR(14), volume z-score, auto S/R (swing pivots), HH/HL–LL/LH market-structure classifier — all causal (no lookahead), pure pandas (no ta-lib needed) |
| **Signal engine** (`strategy/`) | Rule-based trend+pullback / breakout setups; every signal outputs direction, entry, SL (structure- or ATR-based), R:R target, timeframe, confidence (0–100) and the reasons list. Rules are config-driven, not hardcoded |
| **Backtester** (`engine/backtester.py`) | Next-bar-open fills, conservative same-bar rule, win rate, avg win/loss (R), expectancy, profit factor, max drawdown, max losing streak + one-variable-at-a-time sweeps and segmentation |
| **Live scanner / paper trading** (`engine/live_scanner.py`) | Live loop scanning closed bars, or `replay` forward-test on unseen data through the exact paper pipeline (journal + risk guard + alerts) |
| **Risk management** (`risk/`) | Fixed-%-risk position sizing, max daily loss limit, max trades/day, max open positions — new signals are blocked once a limit is hit |
| **Alerts** (`alerts/`) | Console + file + Telegram + desktop notifications; every alert carries the disclaimer |
| **Journal & dashboard** (`journal/`, `dashboard/`) | SQLite journal, auto-logged signals/trades, weekly/monthly reports, Streamlit dashboard with charts, open positions, equity curve, stats |

### Non-negotiable rules enforced in code

1. **Every signal MUST include a stop-loss** — `Signal.validate()` raises on an
   SL-less/zero-risk signal; the test suite asserts it.
2. **Backtest and paper results are stored and reported SEPARATELY**
   (`trades.mode = 'backtest' | 'paper'`) — historical performance can never blend
   with forward performance.
3. **No real orders.** The software writes paper trades and alerts only.
4. **Disclaimer on every signal output** — alerts, CLI, dashboard and JSON exports.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Get data (yfinance on your machine; synthetic works fully offline)
python main.py fetch --symbols AAPL,BTC-USD --timeframes 15m,1h,1d --provider yfinance
python main.py fetch --symbols SYNTH --timeframes 1h --provider synthetic --bars 9000   # offline demo

# 2. Backtest (needs 100+ trades to mean anything)
python main.py backtest --symbol SYNTH --timeframe 1h --provider csv --analyze

# 3. One-variable-at-a-time testing (overfitting guard)
python main.py sweep --symbol SYNTH --timeframe 1h --variable risk_reward --values 1,1.5,2,2.5,3
python main.py sweep --symbol SYNTH --timeframe 1h --variable entry_type --values pullback,breakout
python main.py sweep --symbol SYNTH --timeframe 1h --variable sl_type --values structure,atr

# 4. Forward test on unseen bars through the paper pipeline
python main.py replay --symbol SYNTH --timeframe 1h --provider csv

# 5. Live paper scanning (loop) or a single pass
python main.py scan --symbols AAPL,BTC-USD --timeframes 15m,1h --provider yfinance
python main.py scan --symbols SYNTH --timeframes 1h --provider csv --once

# 6. Reports + position sizing + dashboard
python main.py report --mode paper          # weekly/monthly stats
python main.py position --entry 450 --stop 440 --account 10000 --risk-pct 0.5
python main.py dashboard                    # Streamlit on :8501
```

### Telegram alerts (2-minute setup)

1. Message **@BotFather** → `/newbot` → copy the token.
2. Press START on your new bot, then get your chat id from **@userinfobot**.
3. In `config.json`: set `alerts.telegram: true`, paste token + chat id
   (or export `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`).

---

## Configuration (`config.json`)

Everything the strategy does is configurable — change one variable at a time and
re-test; that's how you avoid curve-fitting.

```jsonc
"strategy": {
  "entry_type": "pullback",        // or "breakout"
  "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
  "rsi_buy_min": 45, "rsi_buy_max": 68,   // momentum band, not just OB/OS
  "sl_type": "structure",          // "structure" (swing ± ATR buffer) | "atr" (mult * ATR)
  "atr_sl_mult": 1.5, "min_sl_atr_mult": 1.0,
  "risk_reward": 2.0,              // target = entry ± RR * risk
  "min_confidence": 60,            // minimum aligned-condition score
  "max_bars_in_trade": 100, "cooldown_bars": 5,
  "confidence_weights": { "trend_alignment": 30, "vwap_position": 15,
                          "rsi_zone": 20, "trigger_pattern": 20, "volume_confirmation": 15 }
},
"risk": {
  "account_size": 10000, "risk_per_trade_pct": 0.5,
  "max_daily_loss_pct": 2.0, "max_trades_per_day": 5, "max_open_positions": 3
}
```

### How a signal is scored (confidence)

A signal fires only when the **trend gate** (EMA 20>50>200 stacked, mirrored for
shorts) and a **trigger** (pullback-reclaim or breakout) both occur. Each aligned
condition adds its weight:

```
confidence = 30 (trend) + 20 (trigger) + 15 (VWAP side) + 20 (RSI band) + 15 (volume)
           → 100 max; emitted only if >= min_confidence (default 60)
```

Every signal ships with: direction, entry, stop-loss, target, R:R, timeframe,
confidence, market-structure label and the plain-English reasons that fired.

---

## Backtest methodology (read before trusting any number)

- Signals are computed on **closed bars only**; entries fill at the **next bar's open**
  (+ configurable slippage) — no lookahead.
- SL/target are fixed price levels; if one bar touches both, the **stop is assumed first**.
- Results are in **R-multiples** (risk units) so runs are comparable across symbols/sizes.
- Prefer **100+ trades**; fewer means the stats are noise.
- Use `sweep` to vary **one variable at a time** and look for plateaus across
  neighbouring values — a single lucky spike is curve-fitting, not edge.
- Keep the workflow honest: **backtest → tweak → replay/paper forward-test** on
  unseen data. Backtest and paper trades live in separate journal tables for that reason.

## Project layout

```
Trading-Software/
├── data/
│   ├── historical/          # cached OHLCV CSVs ({SYMBOL}_{TF}.csv)
│   └── runtime/             # journal.db, alerts.log (gitignored)
├── datafeed/                # providers: yfinance | csv | synthetic (+ resample, normalize)
├── indicators/              # ema, vwap, rsi, atr, volume, support_resistance, market_structure
├── strategy/                # base (Signal+config) & trend_pullback
├── engine/                  # backtester.py & live_scanner.py (paper)
├── risk/                    # position_sizing.py (+ RiskGuard daily limits)
├── journal/                 # logger.py (SQLite, mode-separated)
├── alerts/                  # telegram_bot.py + dispatcher
├── dashboard/app.py         # Streamlit UI
├── tests/                   # 54 unit tests (pytest)
├── config.json              # ALL strategy/risk/alert settings
└── main.py                  # CLI entry point
```

## Running tests

```bash
python -m pytest        # 54 tests: indicator math, causality, fills, R math, limits, journal
```

## Extending

- **New data source:** implement `DataProvider.fetch()` (see `datafeed/base.py`).
- **New strategy:** subclass/replace `TrendPullbackStrategy` — implement
  `prepare()`, `check_signal()`, `generate_signals()`; engines program against that interface.
- **Live broker:** intentionally absent. If you add one, real orders must require
  explicit manual confirmation per trade — that's a design rule of this repo.
