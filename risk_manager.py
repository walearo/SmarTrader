"""
Risk Management:
  - Equity-based position sizing  (1% of NAV per trade, scaled by win rate)
  - Break-even stop               (move SL to entry at BREAKEVEN_R profit)
  - Trailing stop                 (trail at TRAIL_ATR_MULT × ATR after TRAIL_START_R profit)
  - Daily drawdown kill switch    (MAX_DAILY_LOSS_PCT)
  - Weekly drawdown kill switch   (MAX_WEEKLY_LOSS_PCT)
"""

import logging
from datetime import datetime, timezone

import pandas as pd

import config
from data import get_candles
from strategy import add_indicators
from trade_history import get_history, record_trade_result          # noqa: F401 (re-exported)
from trader import get_account_equity, update_trade_sl, partial_close_trade, close_trade

log = logging.getLogger(__name__)

_daily_start_equity:  float | None = None
_weekly_start_equity: float | None = None
_week_number:         int   | None = None
_partial_tp_done:     set         = set()   # trade IDs that have already had partial TP executed


# ─── DYNAMIC POSITION SIZING (EQUITY-BASED) ───────────────────────────────────

def dynamic_units(pair: str, atr: float, current_price: float = 0.0) -> int:
    """
    Size the trade as a fixed percentage of account equity (RISK_PCT_PER_TRADE),
    then scale by recent win rate:
      > 60% win rate → 1.25x
      40–60%         → 1.00x
      < 40%          → 0.75x

    Formula:
      risk_dollars       = equity × RISK_PCT_PER_TRADE / 100
      sl_pips            = atr × ATR_SL_MULTIPLIER / pip
      pip_value_per_unit = pip                  (USD-quoted: EUR_USD, GBP_USD, AUD_USD, NZD_USD)
                         = pip / current_price   (USD-base:   USD_JPY, USD_CHF, USD_CAD)
      units              = risk_dollars / (sl_pips × pip_value_per_unit)

    USD-base pairs: 1 pip of move = pip/price USD (e.g. USD/JPY at 150 → 1 pip ≈ $0.0000667).
    Falls back to config.UNITS if equity cannot be fetched.
    """
    try:
        equity = get_account_equity()
    except Exception as e:
        log.warning(f"Could not fetch equity for sizing ({e}) — using base units")
        equity = None

    pip = config.INSTRUMENT_PIP.get(pair, config.INSTRUMENT_PIP_DEFAULT)

    _USD_BASE_PAIRS = {"USD_JPY", "USD_CHF", "USD_CAD"}
    if pair in _USD_BASE_PAIRS and current_price > 0:
        pip_value_per_unit = pip / current_price
    else:
        pip_value_per_unit = pip   # USD-quoted pairs (forex + Gold + Oil): exact

    if equity and atr > 0:
        risk_dollars = equity * config.RISK_PCT_PER_TRADE / 100
        sl_pips      = (atr * config.ATR_SL_MULTIPLIER) / pip
        units        = int(risk_dollars / (sl_pips * pip_value_per_unit))
    else:
        units = config.UNITS

    history = get_history()
    if len(history) >= 20:
        win_rate = history.count("win") / len(history)
        if win_rate > 0.60:
            multiplier = 1.25
        elif win_rate < 0.40:
            multiplier = 0.75
        else:
            multiplier = 1.0
        units = int(units * multiplier)

    min_u = config.INSTRUMENT_MIN_UNITS.get(pair, config.MIN_UNITS)
    max_u = config.INSTRUMENT_MAX_UNITS.get(pair, config.MAX_UNITS)
    return max(min_u, min(max_u, units))


# ─── KILL SWITCH (DAILY + WEEKLY) ─────────────────────────────────────────────

def check_kill_switch() -> tuple[bool, str]:
    """
    Return (True, reason) if daily OR weekly drawdown exceeds configured limits.
    Both limits are measured from the equity at the start of the period.
    """
    global _daily_start_equity, _weekly_start_equity, _week_number

    equity = get_account_equity()
    now    = datetime.now(timezone.utc)

    if _daily_start_equity is None:
        _daily_start_equity = equity
    if _weekly_start_equity is None:
        _weekly_start_equity = equity
        _week_number = now.isocalendar().week

    daily_loss_pct = (_daily_start_equity - equity) / _daily_start_equity * 100
    if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
        return True, f"Daily loss {daily_loss_pct:.1f}% ≥ limit {config.MAX_DAILY_LOSS_PCT}%"

    weekly_loss_pct = (_weekly_start_equity - equity) / _weekly_start_equity * 100
    if weekly_loss_pct >= config.MAX_WEEKLY_LOSS_PCT:
        return True, f"Weekly loss {weekly_loss_pct:.1f}% ≥ limit {config.MAX_WEEKLY_LOSS_PCT}%"

    return False, ""


def reset_daily_equity() -> None:
    global _daily_start_equity
    _daily_start_equity = None


def reset_weekly_equity() -> None:
    global _weekly_start_equity, _week_number
    _weekly_start_equity = None
    _week_number = None


def clear_partial_tp(trade_id: str) -> None:
    """Remove a closed trade from the partial-TP tracking set."""
    _partial_tp_done.discard(trade_id)


def mark_partial_tp_done(trade_id: str) -> None:
    """Mark a trade as already partially closed (used on startup recovery)."""
    _partial_tp_done.add(trade_id)


def get_drawdown_status() -> dict:
    """Return current daily and weekly drawdown percentages vs configured limits."""
    try:
        equity = get_account_equity()
        daily_pct  = max((_daily_start_equity  - equity) / _daily_start_equity  * 100, 0.0) if _daily_start_equity  else 0.0
        weekly_pct = max((_weekly_start_equity - equity) / _weekly_start_equity * 100, 0.0) if _weekly_start_equity else 0.0
    except Exception:
        daily_pct  = 0.0
        weekly_pct = 0.0
    return {
        "daily_pct":    round(daily_pct,  2),
        "daily_limit":  config.MAX_DAILY_LOSS_PCT,
        "weekly_pct":   round(weekly_pct, 2),
        "weekly_limit": config.MAX_WEEKLY_LOSS_PCT,
    }


# ─── TRAILING STOP & BREAK-EVEN ───────────────────────────────────────────────

def _pip(instrument: str) -> float:
    return config.INSTRUMENT_PIP.get(instrument, config.INSTRUMENT_PIP_DEFAULT)


def _price_dp(instrument: str) -> int:
    return config.INSTRUMENT_PRICE_DP.get(instrument, config.INSTRUMENT_PRICE_DP_DEFAULT)


def manage_open_trades(
    price_map: dict[str, float],
    open_trades: list,
    df_map: dict[str, pd.DataFrame] | None = None,
) -> None:
    """
    For each open trade:
      - If profit >= BREAKEVEN_R × initial_risk → move SL to entry (break-even)
      - If profit >= TRAIL_START_R × initial_risk → trail SL at TRAIL_ATR_MULT × ATR

    price_map   — {instrument: current_mid_price}, pre-fetched in the main cycle
    open_trades — pre-fetched list of open trade dicts
    df_map      — optional {instrument: DataFrame with indicators} to avoid redundant
                  candle fetches; fetched on demand if not supplied
    """
    for trade in open_trades:
        instrument = trade["instrument"]
        current    = price_map.get(instrument)
        if current is None:
            continue

        trade_id = trade["id"]

        # ── trade timeout ──
        if config.MAX_TRADE_HOURS > 0:
            open_time_str = trade.get("openTime", "")
            if open_time_str:
                open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - open_time).total_seconds() / 3600
                if age_hours >= config.MAX_TRADE_HOURS:
                    log.info(f"{instrument} [{trade_id}]: timed out after {age_hours:.1f}h — closing at market")
                    try:
                        close_trade(trade_id)
                    except Exception as e:
                        log.error(f"Trade timeout close failed for {trade_id}: {e}")
                    continue

        entry    = float(trade["price"])
        units    = float(trade["currentUnits"])
        sl_order = trade.get("stopLossOrder")
        if not sl_order:
            continue

        current_sl   = float(sl_order["price"])
        direction    = "buy" if units > 0 else "sell"
        initial_risk = abs(entry - current_sl)

        if initial_risk == 0:
            continue

        profit     = (current - entry) if direction == "buy" else (entry - current)
        r_multiple = profit / initial_risk

        # ── partial TP ──
        if (r_multiple >= config.PARTIAL_TP_R
                and trade_id not in _partial_tp_done
                and abs(units) > 1):
            close_units = max(1, int(abs(units) * config.PARTIAL_TP_PCT))
            log.info(
                f"{instrument} [{trade_id}]: partial TP at {r_multiple:.2f}R "
                f"— closing {close_units} of {int(abs(units))} units"
            )
            try:
                partial_close_trade(trade_id, close_units)
                _partial_tp_done.add(trade_id)
            except Exception as e:
                log.error(f"Partial TP failed: {e}")

        # ── break-even ──
        dp = _price_dp(instrument)
        if r_multiple >= config.BREAKEVEN_R:
            if direction == "buy" and current_sl < entry:
                be = round(entry, dp)
                log.info(f"{instrument} [{trade['id']}]: moving SL to break-even {be}")
                try:
                    update_trade_sl(trade["id"], be, instrument)
                except Exception as e:
                    log.error(f"Break-even update failed: {e}")
                continue

            if direction == "sell" and current_sl > entry:
                be = round(entry, dp)
                log.info(f"{instrument} [{trade['id']}]: moving SL to break-even {be}")
                try:
                    update_trade_sl(trade["id"], be, instrument)
                except Exception as e:
                    log.error(f"Break-even update failed: {e}")
                continue

        # ── trailing stop ──
        if r_multiple >= config.TRAIL_START_R:
            try:
                df = (df_map or {}).get(instrument)
                if df is None:
                    df = add_indicators(get_candles(instrument, count=50))
                trail_dist = df["atr"].iloc[-1] * config.TRAIL_ATR_MULT
            except Exception:
                trail_dist = initial_risk * config.TRAIL_ATR_MULT

            if direction == "buy":
                new_sl = round(current - trail_dist, dp)
                if new_sl > current_sl:
                    log.info(f"{instrument} [{trade['id']}]: trailing SL → {new_sl}")
                    try:
                        update_trade_sl(trade["id"], new_sl, instrument)
                    except Exception as e:
                        log.error(f"Trailing stop update failed: {e}")
            else:
                new_sl = round(current + trail_dist, dp)
                if new_sl < current_sl:
                    log.info(f"{instrument} [{trade['id']}]: trailing SL → {new_sl}")
                    try:
                        update_trade_sl(trade["id"], new_sl, instrument)
                    except Exception as e:
                        log.error(f"Trailing stop update failed: {e}")
