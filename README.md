# SmarTrader

An algorithmic FX and commodity trading bot built on the OANDA v20 API. Combines a Donchian Channel breakout strategy with a multi-layer filter stack — Claude AI for sentiment and regime detection, an ML ensemble for signal confidence, and a full risk management system with daily/weekly drawdown kill switches.

---

## Features

### Strategy
- **Donchian Channel Breakout** — 50-period channel with ATR buffer, volume surge, and ADX trend confirmation
- **Multi-timeframe confirmation** — H1 signal validated against H4 trend direction
- **EMA + RSI filters** — 20/50 EMA crossover and RSI directional bias

### Filter stack (applied in order)
1. News blackout — blocks trades ±30 min around high-impact events
2. Session filter — London (07–16 UTC) and New York (13–22 UTC) only
3. Claude sentiment — news sentiment must align with signal direction
4. Claude regime — blocks counter-trend signals in ranging/volatile markets
5. Volatility filter — ATR must be above the 25th percentile (no dead markets)
6. H4 trend — higher-timeframe bias must agree
7. ML confidence — XGBoost/LightGBM voting ensemble, configurable threshold
8. Correlation filter — prevents stacked USD exposure across open positions

### Risk management
- Equity-based position sizing scaled by recent win rate
- Per-instrument unit caps (Gold, Oil)
- Break-even stop at 2R, trailing stop from 2R
- Partial TP (50% off at 2R)
- Daily drawdown kill switch (default 3%)
- Weekly drawdown kill switch (default 8%)

### Instruments
`EUR_USD` `GBP_USD` `USD_JPY` `AUD_USD` `USD_CHF` `USD_CAD` `NZD_USD` `XAU_USD` `WTICO_USD`

### Operations
- **Live dashboard** — FastAPI + SSE streaming, trade journal, equity curve, bot log
- **Telegram alerts** — trade open/close, kill switch, errors, price moves (per-type toggles)
- **Watchdog** — auto-restarts bot on crash with exponential backoff, logs to `bot_output.log`
- **Startup recovery** — journals and records any trades closed while the bot was down
- **Graceful shutdown** — SIGTERM/SIGINT finishes the current cycle before exiting
- **SQLite persistence** — WAL mode allows dashboard to read while bot writes

---

## Requirements

- Python 3.11+
- OANDA v20 account ([oanda.com](https://www.oanda.com)) — start with a practice account
- Anthropic API key — for Claude sentiment, regime detection, and trade journaling
- Telegram bot — optional, for mobile alerts

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/walearo/SmarTrader.git
cd SmarTrader
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice        # change to "live" when ready

ANTHROPIC_API_KEY=your_anthropic_key

TELEGRAM_BOT_TOKEN=               # optional
TELEGRAM_CHAT_ID=                 # optional

DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=choose_a_strong_password
DASHBOARD_SECRET_KEY=choose_a_random_64_character_string
```

### 3. Train the ML model (optional but recommended)

```bash
python train_ml.py --days 730
```

The script prints the recommended confidence threshold at the end. Set `ML_MIN_CONFIDENCE` in `config.py` to that value, or configure it from the dashboard.

### 4. Run

**Windows (interactive):**
```bat
start.bat
```
This opens two windows — the bot (via watchdog) and the dashboard.

**Manual (any platform):**
```bash
# Terminal 1 — bot with watchdog
python watchdog.py

# Terminal 2 — dashboard
python dashboard.py
```

**Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

For remote access, use an SSH tunnel rather than exposing port 8000:
```bash
ssh -L 8000:127.0.0.1:8000 your-server
```

---

## Dashboard

The dashboard requires login (credentials from `.env`). It provides:

| Tab | Contents |
|-----|----------|
| **Dashboard** | Live equity, open positions, drawdown meters, bot log, manual pause/resume |
| **Journal Analytics** | Claude AI trade journal with win/loss analysis, equity curve chart |
| **Settings** | Configurable trading parameters, per-alert-type toggles |

Settings changes (pairs, sessions, risk limits, concurrent trades, etc.) take effect on the next bot cycle without restarting.

---

## Project structure

```
main.py          — main trading loop (12-step cycle)
strategy.py      — Donchian Channel breakout signal + SL/TP calculation
filters.py       — volatility, H4 trend, correlation filters
risk_manager.py  — position sizing, break-even, trailing stop, kill switch
trader.py        — OANDA order placement and trade management
data.py          — OANDA price and candle feeds
monitor.py       — price move alerts
sentiment.py     — Claude AI news sentiment filter
regime.py        — Claude AI market regime detection
journal.py       — Claude AI post-trade analysis
ml_model.py      — XGBoost/LightGBM voting ensemble inference
train_ml.py      — ML model training script
news.py          — economic calendar (news blackout logic)
alerts.py        — Telegram alert dispatch with per-type toggles
bot_control.py   — cross-process pause/resume, alert settings, bot settings
db.py            — SQLite persistence layer (WAL mode)
bot_log.py       — structured event logger → SQLite
dashboard.py     — FastAPI SSE dashboard
watchdog.py      — process watchdog with exponential backoff restart
config.py        — all configuration constants and env var loading
backtest.py      — offline backtesting engine
verify.py        — pre-flight credential and connectivity check
test_suite.py    — 97-test suite (price precision, alert gating, settings validation)
```

---

## Configuration reference

All strategy and risk parameters can be changed in `config.py` or live from the dashboard Settings tab.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAIRS` | 9 instruments | Instruments to trade |
| `SESSIONS` | London + NY | UTC hour ranges for trading |
| `MAX_CONCURRENT_TRADES` | 2 | Maximum open positions at once |
| `UNITS` | 1000 | Base position size |
| `RISK_PCT_PER_TRADE` | 1.0% | Account equity risked per trade |
| `MAX_DAILY_LOSS_PCT` | 3.0% | Daily drawdown kill switch |
| `MAX_WEEKLY_LOSS_PCT` | 8.0% | Weekly drawdown kill switch |
| `ATR_SL_MULTIPLIER` | 1.5× | ATR multiple for stop loss distance |
| `ATR_TP_MULTIPLIER` | 4.0× | ATR multiple for take profit distance |
| `ML_MIN_CONFIDENCE` | 0.0 | ML filter threshold (0 = disabled) |
| `NEWS_BLACKOUT_MINUTES` | 30 | Minutes around news to block trades |
| `ALERT_PRICE_MOVE_PIPS` | 20 | Pips moved to trigger price alert |
| `CHECK_INTERVAL_SECONDS` | 60 | Main loop interval |

---

## Running tests

```bash
python -m unittest test_suite -v
```

97 tests covering price precision, alert gating, bot settings, drawdown regressions, and the full SL/TP pipeline.

---

## Health monitoring

The dashboard exposes a `/health` endpoint (no auth required):

```
GET http://127.0.0.1:8000/health
→ {"status": "ok", "ts": "2026-04-30T12:00:00+00:00"}
```

Point an external monitor (UptimeRobot, Better Uptime) at this URL to get alerted if the dashboard process goes down.

---

## Security notes

- `.env` is in `.gitignore` — never commit it
- Dashboard binds to `127.0.0.1` only — not accessible from the network
- Use `OANDA_ENVIRONMENT=practice` until you have verified strategy performance
- Set a strong `DASHBOARD_PASSWORD` and a random 64-character `DASHBOARD_SECRET_KEY`

---

## Disclaimer

This software is for educational purposes. Algorithmic trading involves substantial risk of financial loss. Past backtest performance does not guarantee future results. Always test on a practice account before using real funds.
