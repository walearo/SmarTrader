"""
Level 1 — Smart Filters
  - Session filter     (London / New York for forex; extended window for commodities)
  - Volatility filter  (skip low-ATR / ranging markets)
  - Multi-timeframe    (H1 signal must align with H4 trend)
"""

import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import config
from data import get_candles
from strategy import add_indicators

_HTF_CACHE: dict[str, tuple[float, str]] = {}   # pair -> (timestamp, trend)
_HTF_TTL   = 300                                 # 5-minute cache

# Commodities trade ~23h/day — allow a wider session window than forex.
# WTICO_USD has a ~60-min daily close on OANDA around 22:00 UTC; stop at 21:00
# to avoid placing orders into that close window (OANDA returns 500 during it).
_COMMODITY_PAIRS    = {"XAU_USD", "WTICO_USD"}
_COMMODITY_SESSIONS = [(7, 21)]   # covers London open through NY close, clears daily halt


# ─── SESSION FILTER ───────────────────────────────────────────────────────────

def in_session(pair: str = "") -> bool:
    """Return True if current UTC hour falls inside the appropriate trading session.

    Commodities (Gold, Oil) use an extended window; forex uses the configured sessions.
    """
    hour     = datetime.now(timezone.utc).hour
    sessions = _COMMODITY_SESSIONS if pair in _COMMODITY_PAIRS else config.SESSIONS
    for start, end in sessions:
        if start <= hour < end:
            return True
    return False


# ─── VOLATILITY FILTER ────────────────────────────────────────────────────────

def volatility_ok(df: pd.DataFrame) -> bool:
    """
    Return True if current ATR is above the configured percentile threshold.
    Prevents trading during flat / compressed markets.
    """
    df   = add_indicators(df)
    atrs = df["atr"].dropna()
    if atrs.empty:
        return True   # can't tell — allow
    threshold = np.percentile(atrs, config.ATR_PERCENTILE_MIN)
    current   = atrs.iloc[-1]
    return current >= threshold


# ─── MULTI-TIMEFRAME CONFIRMATION ─────────────────────────────────────────────

def _htf_trend(pair: str) -> str:
    """Fetch H4 candles and determine the current trend: 'up', 'down', or 'flat'.

    Result is cached per pair for 5 minutes to avoid an API call every cycle.
    """
    now = time.monotonic()
    cached = _HTF_CACHE.get(pair)
    if cached and now - cached[0] < _HTF_TTL:
        return cached[1]

    df   = get_candles(pair, count=100, timeframe=config.HTF)
    df   = add_indicators(df)
    last = df.iloc[-1]

    if last["ema_fast"] > last["ema_slow"]:
        trend = "up"
    elif last["ema_fast"] < last["ema_slow"]:
        trend = "down"
    else:
        trend = "flat"

    _HTF_CACHE[pair] = (now, trend)
    return trend


def htf_confirms(pair: str, signal: str) -> bool:
    """
    Return True when the H4 trend agrees with the H1 signal direction.
    Flat H4 trend = no confirmation = skip.
    """
    trend = _htf_trend(pair)
    if signal == "buy"  and trend == "up":   return True
    if signal == "sell" and trend == "down": return True
    return False


# ─── CORRELATION FILTER ───────────────────────────────────────────────────────

# Pairs that share the same directional USD exposure.
# USD_SHORT: buying these = short USD; USD_LONG: buying these = long USD.
_USD_SHORT = {"EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"}
_USD_LONG  = {"USD_JPY", "USD_CHF", "USD_CAD"}


def correlation_ok(pair: str, signal: str, open_trades: list) -> tuple[bool, str]:
    """
    Prevent stacking highly correlated USD positions.

    A new BUY on EUR_USD conflicts with an existing BUY on GBP_USD (both short USD).
    A new BUY on USD_JPY conflicts with an existing BUY on EUR_USD (opposing USD side).

    Returns (True, "") when the trade is safe to take.
    Returns (False, reason) when it would stack correlated risk.
    """
    if not open_trades:
        return True, ""

    # Determine the new trade's USD direction
    if pair in _USD_SHORT:
        new_usd_dir = "short" if signal == "buy" else "long"
    elif pair in _USD_LONG:
        new_usd_dir = "long" if signal == "buy" else "short"
    else:
        return True, ""   # exotic or unknown pair — skip check

    for trade in open_trades:
        inst  = trade.get("instrument", "")
        units = float(trade.get("currentUnits", 0))
        existing_signal = "buy" if units > 0 else "sell"

        if inst in _USD_SHORT:
            existing_usd_dir = "short" if existing_signal == "buy" else "long"
        elif inst in _USD_LONG:
            existing_usd_dir = "long" if existing_signal == "buy" else "short"
        else:
            continue

        if new_usd_dir == existing_usd_dir:
            return False, f"already {existing_usd_dir} USD via {inst} — stacking correlated risk"

    return True, ""


# ─── COMBINED GATE ────────────────────────────────────────────────────────────

def all_filters_pass(pair: str, signal: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Run all Level-1 filters.
    Returns (passed: bool, reason: str).
    """
    if not in_session(pair):
        return False, "outside trading session"
    if not volatility_ok(df):
        return False, "ATR too low (ranging market)"
    if not htf_confirms(pair, signal):
        return False, "H4 trend does not confirm signal"
    return True, "ok"
