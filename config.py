"""
All configuration is loaded from environment variables (or a .env file).
Copy .env.example → .env and fill in your credentials.
Strategy/risk constants below are safe to commit — they contain no secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()   # loads .env from the project directory if present


# ─── OANDA ────────────────────────────────────────────────────────────────────
OANDA_API_KEY      = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID   = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT  = os.environ.get("OANDA_ENVIRONMENT", "practice")

# ─── ANTHROPIC ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── DASHBOARD AUTH ───────────────────────────────────────────────────────────
DASHBOARD_USERNAME   = os.environ.get("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD   = os.environ.get("DASHBOARD_PASSWORD", "")
DASHBOARD_SECRET_KEY = os.environ.get("DASHBOARD_SECRET_KEY", "")

# ─── PAIRS & TIMEFRAME ────────────────────────────────────────────────────────
PAIRS          = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "USD_CAD", "NZD_USD",
                  "XAU_USD", "WTICO_USD"]
TIMEFRAME      = "H1"
TRAIN_TIMEFRAME = "H1"
HTF            = "H4"
UNITS     = 1000
MIN_UNITS = 500
MAX_UNITS = 5000
MAX_CONCURRENT_TRADES = 2

# ─── INSTRUMENT-SPECIFIC OVERRIDES ───────────────────────────────────────────
# Pip size: smallest price increment used for SL/TP sizing calculations.
INSTRUMENT_PIP: dict[str, float] = {
    "USD_JPY":  0.01,
    "XAU_USD":  0.01,    # Gold: quoted to 2 dp (e.g. 2350.45)
    "WTICO_USD": 0.001,  # Oil:  quoted to 3 dp (e.g. 79.456)
}
INSTRUMENT_PIP_DEFAULT = 0.0001   # all standard forex pairs

# Price decimal places per instrument (used when formatting SL/TP for the OANDA API).
# Standard forex defaults to 5 dp; instruments with a different pip scale need explicit entries.
INSTRUMENT_PRICE_DP: dict[str, int] = {
    "USD_JPY":   3,    # e.g. 150.123
    "XAU_USD":   2,    # e.g. 2350.45
    "WTICO_USD": 3,    # e.g. 79.456
}
INSTRUMENT_PRICE_DP_DEFAULT = 5

# Position size caps per instrument (native units: oz for Gold, barrels for Oil)
INSTRUMENT_MIN_UNITS: dict[str, int] = {
    "XAU_USD":  1,
    "WTICO_USD": 1,
}
INSTRUMENT_MAX_UNITS: dict[str, int] = {
    "XAU_USD":  50,    # 50 oz max (~$115K notional at $2300/oz)
    "WTICO_USD": 300,  # 300 barrels max (~$24K notional at $80/bbl)
}

# ─── STRATEGY PARAMETERS ──────────────────────────────────────────────────────
EMA_FAST  = 20
EMA_SLOW  = 50
RSI_PERIOD      = 14
RSI_OVERBOUGHT  = 70
RSI_OVERSOLD    = 30

# ─── DONCHIAN CHANNEL SIGNAL ──────────────────────────────────────────────────
DONCHIAN_PERIOD     = 50
VOLUME_SURGE_FACTOR = 1.2
ADX_PERIOD          = 14
ADX_MIN_TREND       = 25
BREAKOUT_ATR_BUFFER = 0.15
RSI_BUY_MIN         = 55
RSI_SELL_MAX        = 45
BREAKOUT_CANDLE_CLOSE_PCT = 0.6

# ─── RISK MANAGEMENT ──────────────────────────────────────────────────────────
ATR_PERIOD        = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 4.0

RISK_PCT_PER_TRADE = 1.0

BREAKEVEN_R    = 2.0
TRAIL_START_R  = 2.0
TRAIL_ATR_MULT = 1.5

PARTIAL_TP_R   = 2.0   # take 50% off at 2R
PARTIAL_TP_PCT = 0.5   # fraction of position to close

MAX_DAILY_LOSS_PCT  = 3.0
MAX_WEEKLY_LOSS_PCT = 8.0

# ─── SESSION FILTER ───────────────────────────────────────────────────────────
SESSIONS = [
    (7, 16),   # London
    (13, 22),  # New York
]

# ─── VOLATILITY FILTER ────────────────────────────────────────────────────────
ATR_PERCENTILE_MIN = 25

# ─── NEWS BLACKOUT ────────────────────────────────────────────────────────────
NEWS_BLACKOUT_MINUTES = 30
NEWS_CURRENCIES = ["USD", "EUR", "GBP", "NZD", "JPY", "CHF", "AUD", "CAD"]

# ─── BACKTEST ─────────────────────────────────────────────────────────────────
BACKTEST_SPREAD_PIPS        = 1.0
BACKTEST_MAX_TRADE_DURATION = 48

# ─── ML MODEL ─────────────────────────────────────────────────────────────────
ML_MIN_CONFIDENCE  = 0.0    # set to 0.0 to bypass ML filter
                            # train with: python train_ml.py --days 730
                            # then set to the recommended threshold printed at end
ML_MODEL_PATH      = "model.pkl"

# ─── MONITORING ───────────────────────────────────────────────────────────────
ALERT_PRICE_MOVE_PIPS  = 20
CHECK_INTERVAL_SECONDS = 60
