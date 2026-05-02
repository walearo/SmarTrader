"""
FX Bot — smart main loop.

Every tick:
  1. Kill switch check        (daily drawdown limit)
  2. Price monitoring         (big move alerts)
  3. Open trade management    (break-even + trailing stop)
  4. Trade closure detection  (journal + record win/loss)
  5. News blackout check      (pause near high-impact events)
  6. Claude sentiment filter  (directional bias from news)
  7. Session filter           (London / NY only)
  8. Strategy signal          (EMA crossover + RSI)
  9. Claude regime filter     (trending / ranging / volatile)
 10. Volatility + H4 filter   (ATR percentile + multi-timeframe)
 11. ML confidence filter     (Voting Ensemble >= 60%)
 12. Dynamic sizing + order   (scale units by win rate)
"""

import time
import signal
import logging
from datetime import datetime, timezone

import config
import db
import alerts
import bot_log
from data import get_candles, get_price, get_prices
from strategy import get_signal, get_sl_tp, add_indicators
from trader import place_order, has_open_position, get_open_trades, get_closed_trade, update_trade_orders, get_margin_available
from monitor import check_price_moves
from filters import all_filters_pass, in_session, correlation_ok
from news import news_blackout_active, upcoming_events
from ml_model import ml_filter_passes, check_model_staleness
from risk_manager import (
    dynamic_units,
    check_kill_switch,
    reset_daily_equity,
    reset_weekly_equity,
    manage_open_trades,
    clear_partial_tp,
    mark_partial_tp_done,
)
from trade_history import record_trade_result, consecutive_losses
import bot_control
import sentiment as sentiment_mod
import regime as regime_mod
import journal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_bot_paused    = False
_last_day_utc  = None
_shutdown      = False
_daily_trades  = 0


def _handle_signal(sig, frame):
    global _shutdown
    log.info("Shutdown signal received — finishing current cycle then exiting.")
    _shutdown = True

# ── trade closure tracking ────────────────────────────────────────────────────
_open_trade_ids: set      = set()
_trade_metadata: dict     = {}   # trade_id -> metadata captured at open time


def validate_config() -> None:
    # ── hard failures (bot cannot run without these) ────────────────────────────
    missing = []
    if not config.OANDA_API_KEY:    missing.append("OANDA_API_KEY")
    if not config.OANDA_ACCOUNT_ID: missing.append("OANDA_ACCOUNT_ID")
    if missing:
        raise ValueError(
            f"Missing required credentials: {', '.join(missing)}. "
            "Set them in your .env file (see .env.example)."
        )

    # ── OANDA environment check ─────────────────────────────────────────────────
    if config.OANDA_ENVIRONMENT == "live":
        log.warning(
            "OANDA_ENVIRONMENT=live — bot is connected to a LIVE funded account. "
            "All trades will use real money."
        )
    else:
        log.info(f"OANDA_ENVIRONMENT={config.OANDA_ENVIRONMENT} (paper trading)")

    # ── security warnings (bot can run but should be fixed before going live) ───
    if not config.ANTHROPIC_API_KEY:
        log.warning(
            "ANTHROPIC_API_KEY not set — Claude sentiment, regime, and journal features disabled."
        )

    if not config.DASHBOARD_PASSWORD:
        log.warning(
            "DASHBOARD_PASSWORD is not set — dashboard login is disabled. "
            "Set DASHBOARD_PASSWORD in .env before exposing the dashboard."
        )

    if not config.DASHBOARD_SECRET_KEY:
        log.warning(
            "DASHBOARD_SECRET_KEY is not set — session cookies use an insecure default. "
            "Set DASHBOARD_SECRET_KEY in .env to a random 64-character string."
        )

    # ── ML model check ──────────────────────────────────────────────────────────
    import os
    if config.ML_MIN_CONFIDENCE > 0 and not os.path.exists(config.ML_MODEL_PATH):
        log.warning(
            f"ML_MIN_CONFIDENCE={config.ML_MIN_CONFIDENCE} but model file '{config.ML_MODEL_PATH}' "
            "not found. Run: python train_ml.py --days 730"
        )
    if config.ML_MIN_CONFIDENCE == 0.0:
        log.warning(
            "ML filter is bypassed (ML_MIN_CONFIDENCE=0.0). "
            "Train the model with: python train_ml.py --days 730"
        )


def _daily_reset() -> None:
    """Tasks to run once at the start of each UTC trading day."""
    global _last_day_utc, _daily_trades
    today = datetime.now(timezone.utc).date()
    if _last_day_utc == today:
        return
    _last_day_utc  = today
    _daily_trades  = 0
    reset_daily_equity()
    log.info("Daily equity reset.")

    # reset weekly baseline on Monday (weekday 0)
    if today.weekday() == 0:
        reset_weekly_equity()
        log.info("Weekly equity reset (Monday).")

    db.prune_bot_log()
    check_model_staleness()

    events = upcoming_events(hours=24)
    if events:
        lines = "\n".join(f"  {e['currency']} — {e['title']} @ {e['time']}" for e in events)
        alerts.send(f"*High-Impact News Today:*\n{lines}")


# ── trade closure detection + journaling ─────────────────────────────────────

def _detect_closures(current_open: list) -> None:
    """
    Compare current open trades to the previous cycle.
    For each trade that disappeared, fetch its final details,
    journal it with Claude, and record the win/loss for dynamic sizing.
    """
    global _open_trade_ids

    current_ids = {t["id"] for t in current_open}
    closed_ids  = _open_trade_ids - current_ids

    for trade_id in closed_ids:
        try:
            trade = get_closed_trade(trade_id)
            if not trade:
                continue

            units      = float(trade.get("initialUnits", 0))
            entry      = float(trade.get("price", 0))
            close_px   = float(trade.get("averageClosePrice", 0))
            realized   = float(trade.get("realizedPL", 0))
            financing  = float(trade.get("financing", 0))
            instrument = trade.get("instrument", "")
            direction  = "BUY" if units > 0 else "SELL"
            pip        = config.INSTRUMENT_PIP.get(instrument, config.INSTRUMENT_PIP_DEFAULT)
            pl_pips    = (close_px - entry) / pip if direction == "BUY" \
                         else (entry - close_px) / pip

            meta = _trade_metadata.pop(trade_id, {})

            journal.analyse_trade({
                "instrument":    instrument,
                "direction":     direction,
                "entry":         entry,
                "exit_price":    close_px,
                "pl_pips":       pl_pips,
                "financing":     financing,
                "rsi_at_entry":  meta.get("rsi"),
                "atr_ratio":     meta.get("atr_ratio"),
                "ml_confidence": meta.get("confidence"),
                "sentiment_bias":meta.get("sentiment_bias"),
                "regime":        meta.get("regime"),
                "news_events":   meta.get("news_events"),
            })

            # win/loss uses total economic return including overnight swap fees
            result = "win" if (realized + financing) > 0 else "loss"
            record_trade_result(result)
            clear_partial_tp(trade_id)
            bot_log.trade_close(instrument, result, pl_pips, financing=financing)
            alerts.trade_closed(instrument, result, pl_pips)
            if financing:
                log.info(f"{instrument}: trade {trade_id} closed — {result} ({pl_pips:+.1f} pips, financing={financing:+.4f})")
            else:
                log.info(f"{instrument}: trade {trade_id} closed — {result} ({pl_pips:+.1f} pips)")

        except Exception as e:
            log.error(f"Closure processing error for {trade_id}: {e}")

    _open_trade_ids = current_ids
    db.state_set("open_trade_ids", list(_open_trade_ids))


# ── startup recovery ──────────────────────────────────────────────────────────

def _startup_recovery() -> None:
    """
    Sync internal state with OANDA on (re)start to survive crashes cleanly.

    1. Reads last known open trade IDs from SQLite.
    2. Compares with live trades from OANDA.
    3. Journals and records any trades that closed while the bot was down.
    4. Marks already-partially-closed trades so partial TP doesn't double-fire.
    5. Initialises _open_trade_ids from the live list.
    """
    global _open_trade_ids
    try:
        live_trades = get_open_trades()
        live_ids    = {t["id"] for t in live_trades}
        last_ids    = set(db.state_get("open_trade_ids", []))

        # Process trades that closed while bot was down
        missed = last_ids - live_ids
        for trade_id in missed:
            try:
                trade = get_closed_trade(trade_id)
                if not trade:
                    continue
                units      = float(trade.get("initialUnits", 0))
                entry      = float(trade.get("price", 0))
                close_px   = float(trade.get("averageClosePrice", 0))
                realized   = float(trade.get("realizedPL", 0))
                financing  = float(trade.get("financing", 0))
                instrument = trade.get("instrument", "")
                direction  = "BUY" if units > 0 else "SELL"
                pip        = config.INSTRUMENT_PIP.get(instrument, config.INSTRUMENT_PIP_DEFAULT)
                pl_pips    = (close_px - entry) / pip if direction == "BUY" \
                             else (entry - close_px) / pip

                journal.analyse_trade({
                    "instrument": instrument, "direction": direction,
                    "entry": entry, "exit_price": close_px, "pl_pips": pl_pips,
                    "financing": financing,
                })
                result = "win" if (realized + financing) > 0 else "loss"
                record_trade_result(result)
                bot_log.trade_close(instrument, result, pl_pips, financing=financing)
                alerts.trade_closed(instrument, result, pl_pips)
                log.info(
                    f"Startup recovery: {instrument} trade {trade_id} "
                    f"closed while down — {result} ({pl_pips:+.1f} pips)"
                )
            except Exception as e:
                log.warning(f"Could not recover closure for {trade_id}: {e}")

        # Mark already-partially-closed trades to prevent double partial TP
        for trade in live_trades:
            initial = abs(float(trade.get("initialUnits", 0)))
            current = abs(float(trade.get("currentUnits", 0)))
            if initial > 0 and current < initial:
                mark_partial_tp_done(trade["id"])
                log.info(
                    f"Startup recovery: trade {trade['id']} already partially closed "
                    f"({current}/{initial} units remain)"
                )

        _open_trade_ids = live_ids
        db.state_set("open_trade_ids", list(live_ids))
        log.info(
            f"Startup recovery complete: {len(live_ids)} open trade(s)"
            + (f", {len(missed)} closure(s) processed" if missed else "")
        )
    except Exception as e:
        log.warning(f"Startup recovery failed (non-critical): {e}")


# ── main cycle ────────────────────────────────────────────────────────────────

def run_cycle() -> None:
    global _bot_paused

    # ── 0a. apply any settings saved via dashboard ──────────────────────────
    bot_control.apply_bot_settings()

    # ── 0. dashboard pause ──────────────────────────────────────────────────
    if bot_control.is_paused():
        ctrl = bot_control.status()
        log.info(f"Bot paused by dashboard: {ctrl.get('reason')} at {ctrl.get('at')}")
        return

    # ── 1. kill switch ──────────────────────────────────────────────────────
    killed, reason = check_kill_switch()
    if killed:
        if not _bot_paused:
            log.warning(f"Kill switch triggered: {reason}")
            alerts.kill_switch_alert(reason)
            bot_log.kill_switch(reason)
            _bot_paused = True
        return
    else:
        _bot_paused = False

    # ── 2 + 3. fetch all prices once — shared by monitor and trade management ──
    try:
        price_map = get_prices(config.PAIRS)
    except Exception as e:
        log.error(f"Batch price fetch failed: {e}")
        alerts.error_alert("cycle/price_fetch", e)
        price_map = {}

    # ── 2. price monitoring ─────────────────────────────────────────────────
    check_price_moves(price_map)

    # ── 3. manage open trades (break-even + trailing) ───────────────────────
    open_trades = get_open_trades()   # fetch once — reused for closure detection and manage_open_trades
    manage_open_trades(
        {pair: data["mid"] for pair, data in price_map.items()},
        open_trades=open_trades,
    )

    # ── 4. detect closures → journal + record win/loss ──────────────────────
    _detect_closures(open_trades)

    # ── 5–12. signal pipeline per pair ──────────────────────────────────────
    if len(open_trades) >= config.MAX_CONCURRENT_TRADES:
        log.info(
            f"Max concurrent trades reached "
            f"({len(open_trades)}/{config.MAX_CONCURRENT_TRADES}), skipping new signals."
        )
        return

    if config.MAX_DAILY_TRADES > 0 and _daily_trades >= config.MAX_DAILY_TRADES:
        log.info(f"Daily trade cap reached ({_daily_trades}/{config.MAX_DAILY_TRADES}), skipping new signals.")
        return

    if config.MAX_CONSECUTIVE_LOSSES > 0:
        consec = consecutive_losses()
        if consec >= config.MAX_CONSECUTIVE_LOSSES:
            log.warning(f"Consecutive loss limit hit ({consec}), skipping new signals for rest of day.")
            return

    for pair in config.PAIRS:
        try:
            _evaluate_pair(pair, open_trades)
        except Exception as e:
            log.error(f"{pair}: {e}")
            alerts.error_alert(f"cycle/{pair}", e)


def _evaluate_pair(pair: str, open_trades: list = None) -> None:
    # ── 5. news blackout (hard block for extreme events) ────────────────────
    blocked, news_reason = news_blackout_active(pair)
    if blocked:
        log.info(f"{pair}: news blackout — {news_reason}")
        bot_log.filter_blocked(pair, f"news blackout: {news_reason}")
        return

    # ── 6. session filter ───────────────────────────────────────────────────
    if not in_session(pair):
        log.debug(f"{pair}: outside trading session.")
        return

    # skip if already in a trade for this pair
    if has_open_position(pair, open_trades):
        log.info(f"{pair}: position open, skipping signal check.")
        return

    # ── 7. strategy signal ──────────────────────────────────────────────────
    df     = add_indicators(get_candles(pair))   # compute indicators once
    signal = get_signal(df)
    log.info(f"{pair}: signal = {signal}")
    if signal == "flat":
        return

    # ── 8. Claude sentiment filter ──────────────────────────────────────────
    nearby_events = upcoming_events(hours=4)
    sentiment_blocked, s_reason = sentiment_mod.sentiment_blocks_signal(
        pair, signal, nearby_events
    )
    if sentiment_blocked:
        log.info(f"{pair}: sentiment blocked — {s_reason}")
        bot_log.filter_blocked(pair, s_reason)
        return
    sentiment_result = sentiment_mod.get_news_sentiment(pair, nearby_events)
    sentiment_bias   = sentiment_result.get("bias", "neutral")

    # ── 9. Claude regime filter ─────────────────────────────────────────────
    regime_blocked, r_reason = regime_mod.regime_blocks_signal(pair, signal, df)
    if regime_blocked:
        log.info(f"{pair}: regime blocked — {r_reason}")
        bot_log.filter_blocked(pair, r_reason)
        return
    current_regime, _ = regime_mod.get_market_regime(pair, df)

    # ── 10. volatility + H4 filters ─────────────────────────────────────────
    passed, filter_reason = all_filters_pass(pair, signal, df)
    if not passed:
        log.info(f"{pair}: filtered out — {filter_reason}")
        bot_log.filter_blocked(pair, filter_reason)
        return

    # ── 11. ML confidence filter ─────────────────────────────────────────────
    ml_ok, confidence = ml_filter_passes(df, signal)
    log.info(f"{pair}: ML confidence = {confidence:.1%}")
    bot_log.signal(pair, signal, confidence)
    if not ml_ok:
        log.info(
            f"{pair}: ML filter blocked "
            f"(confidence {confidence:.1%} < {config.ML_MIN_CONFIDENCE:.0%})"
        )
        bot_log.filter_blocked(
            pair, f"ML confidence {confidence:.1%} < {config.ML_MIN_CONFIDENCE:.0%}"
        )
        return

    # ── 11b. correlation filter ───────────────────────────────────────────────
    corr_ok, corr_reason = correlation_ok(pair, signal, open_trades or [])
    if not corr_ok:
        log.info(f"{pair}: correlation blocked — {corr_reason}")
        bot_log.filter_blocked(pair, f"correlation: {corr_reason}")
        return

    # ── 12. place order with dynamic sizing ──────────────────────────────────
    price  = get_price(pair)
    entry  = price["ask"] if signal == "buy" else price["bid"]

    # spread filter — skip if spread is too wide (cost eats the edge)
    pip           = config.INSTRUMENT_PIP.get(pair, config.INSTRUMENT_PIP_DEFAULT)
    spread_pips   = (price["ask"] - price["bid"]) / pip
    spread_limit  = config.INSTRUMENT_SPREAD_MAX_PIPS.get(pair, config.SPREAD_MAX_PIPS)
    if spread_pips > spread_limit:
        log.info(f"{pair}: spread {spread_pips:.1f} pips > limit {spread_limit} — skipping entry")
        bot_log.filter_blocked(pair, f"spread {spread_pips:.1f} pips > limit {spread_limit}")
        return

    sl, tp = get_sl_tp(df, signal, entry_price=entry, pair=pair)
    # pass entry price so sizing is accurate for USD-base pairs (USD/JPY, USD/CHF, USD/CAD)
    units  = dynamic_units(pair, float(df["atr"].iloc[-1]), current_price=entry)

    # margin pre-check: skip if free margin is already below MARGIN_MIN_FREE_PCT of NAV
    try:
        margin_avail, nav = get_margin_available()
        if nav > 0 and (margin_avail / nav * 100) < config.MARGIN_MIN_FREE_PCT:
            pct = margin_avail / nav * 100
            log.warning(
                f"{pair}: skipping order — free margin {pct:.1f}% < "
                f"{config.MARGIN_MIN_FREE_PCT}% of NAV"
            )
            bot_log.filter_blocked(
                pair,
                f"margin {pct:.1f}% < min {config.MARGIN_MIN_FREE_PCT}%"
            )
            return
    except Exception as e:
        log.warning(f"{pair}: margin check failed ({e}) — proceeding with order")

    response = place_order(pair, signal, sl, tp, units=units)
    log.info(f"{pair}: order placed ({units} units) → {response}")

    # slippage correction: if actual fill price differs from quoted entry by > 0.5 pip,
    # recalculate SL/TP from the real fill and update the trade's dependent orders
    fill_tx = response.get("orderFillTransaction", {})
    fill_price_str = fill_tx.get("price")
    if fill_price_str:
        fill_price = float(fill_price_str)
        slippage   = abs(fill_price - entry)
        if slippage > pip * 0.5:
            sl, tp = get_sl_tp(df, signal, entry_price=fill_price, pair=pair)
            slip_trade_id = fill_tx.get("tradeOpened", {}).get("tradeID")
            if slip_trade_id:
                try:
                    update_trade_orders(slip_trade_id, sl, tp, pair)
                    log.info(
                        f"{pair}: SL/TP recalculated for {slippage/pip:.1f} pip slippage "
                        f"(fill={fill_price}, quoted={entry})"
                    )
                except Exception as e:
                    log.warning(f"{pair}: SL/TP slippage adjustment failed: {e}")

    # extract trade ID from OANDA response to track closure later
    trade_id = (
        response.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
        or response.get("relatedTransactionIDs", [None])[0]
    )
    if trade_id:
        atr_ratio = round(
            float(df["atr"].iloc[-1]) / float(df["atr"].tail(20).mean()), 2
        )
        _trade_metadata[trade_id] = {
            "rsi":           round(float(df["rsi"].iloc[-1]), 1),
            "atr_ratio":     atr_ratio,
            "confidence":    f"{confidence:.1%}",
            "sentiment_bias":sentiment_bias,
            "regime":        current_regime,
            "news_events":   [e["title"] for e in nearby_events],
        }

    global _daily_trades
    _daily_trades += 1

    bot_log.trade_open(pair, signal, entry, sl, tp, units)
    alerts.trade_opened(pair, signal, entry, sl, tp)
    alerts.send(
        f"  ML: `{confidence:.1%}` | Regime: `{current_regime}` | "
        f"Sentiment: `{sentiment_bias}` | Units: `{units}`"
    )


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    validate_config()
    db.init()
    log.info("Smart FX Bot starting...")
    _startup_recovery()
    alerts.send(
        "*Smart FX Bot started*\n"
        "Filters: Session + Sentiment + Regime + Volatility + H4 + News + ML\n"
        "Pairs: " + ", ".join(f"`{p}`" for p in config.PAIRS)
    )

    while not _shutdown:
        try:
            _daily_reset()
            run_cycle()
        except Exception as e:
            log.error(f"Main loop error: {e}")
            alerts.error_alert("main loop", e)

        time.sleep(config.CHECK_INTERVAL_SECONDS)

    log.info("Bot shut down cleanly.")
    alerts.send("*Bot shut down cleanly* (SIGTERM/SIGINT received).")


if __name__ == "__main__":
    main()
