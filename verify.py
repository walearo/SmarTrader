"""
End-to-end verification of every module and feature.
Run with: py -X utf8 verify.py
"""

import sys
import traceback
from datetime import datetime, timezone

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"
results = []


def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {name}: {msg}")
        results.append((name, False, msg))


def skip(name, reason):
    print(f"  {SKIP}  {name}: {reason}")
    results.append((name, None, reason))


# ─── 1. CONFIG ────────────────────────────────────────────────────────────────
print("\n[1] Config")

def _config_keys():
    import config
    assert config.OANDA_API_KEY,    "OANDA_API_KEY missing"
    assert config.OANDA_ACCOUNT_ID, "OANDA_ACCOUNT_ID missing"
    assert config.ANTHROPIC_API_KEY,"ANTHROPIC_API_KEY missing"
    assert config.PAIRS,            "PAIRS empty"
    assert config.RISK_PCT_PER_TRADE == 1.0
    assert config.ATR_SL_MULTIPLIER  == 1.5
    assert config.ATR_TP_MULTIPLIER  == 4.0

check("All required config keys present", _config_keys)


# ─── 2. DATA ──────────────────────────────────────────────────────────────────
print("\n[2] Data")

def _candles():
    from data import get_candles
    df = get_candles("EUR_USD", count=50)
    assert len(df) >= 10
    assert "open" in df.columns
    assert "close" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "volume" in df.columns

def _price():
    from data import get_price
    p = get_price("EUR_USD")
    assert "bid" in p and "ask" in p and "mid" in p
    assert p["ask"] > p["bid"] > 0

check("get_candles returns valid OHLCV DataFrame", _candles)
check("get_price returns bid/ask/mid", _price)


# ─── 3. STRATEGY ──────────────────────────────────────────────────────────────
print("\n[3] Strategy")

def _indicators():
    from data import get_candles
    from strategy import add_indicators
    df = get_candles("EUR_USD", count=100)
    df = add_indicators(df)
    for col in ["ema_fast", "ema_slow", "rsi", "atr", "adx",
                "adx_pos", "adx_neg", "dc_high", "dc_low", "vol_ma"]:
        assert col in df.columns, f"missing column: {col}"

def _signal():
    from data import get_candles
    from strategy import add_indicators, get_signal
    df = get_candles("EUR_USD", count=100)
    df = add_indicators(df)
    sig = get_signal(df)
    assert sig in ("buy", "sell", "flat")

def _sl_tp():
    from data import get_candles
    from strategy import add_indicators, get_sl_tp
    df = get_candles("EUR_USD", count=100)
    df = add_indicators(df)
    import config
    entry = 1.10000
    sl, tp = get_sl_tp(df, "buy", entry_price=entry)
    assert sl < entry < tp, f"SL/TP invalid: sl={sl} entry={entry} tp={tp}"

check("add_indicators produces all expected columns", _indicators)
check("get_signal returns buy/sell/flat", _signal)
check("get_sl_tp returns valid SL < entry < TP", _sl_tp)


# ─── 4. FILTERS ───────────────────────────────────────────────────────────────
print("\n[4] Filters")

def _session():
    from filters import in_session
    result = in_session()
    assert isinstance(result, bool)

def _volatility():
    from data import get_candles
    from strategy import add_indicators
    from filters import volatility_ok
    df = add_indicators(get_candles("EUR_USD", count=100))
    result = volatility_ok(df)
    assert result in (True, False)   # accepts numpy.bool_ and Python bool

def _htf():
    from filters import htf_confirms
    result_buy  = htf_confirms("EUR_USD", "buy")
    result_sell = htf_confirms("EUR_USD", "sell")
    assert isinstance(result_buy, bool)
    assert isinstance(result_sell, bool)
    assert not (result_buy and result_sell), "H4 can't confirm both buy and sell"

def _all_filters():
    from data import get_candles
    from strategy import add_indicators
    from filters import all_filters_pass
    df = add_indicators(get_candles("EUR_USD", count=100))
    passed, reason = all_filters_pass("EUR_USD", "buy", df)
    assert isinstance(passed, bool)
    assert isinstance(reason, str)

check("in_session returns bool", _session)
check("volatility_ok returns bool", _volatility)
check("htf_confirms: not both buy and sell confirmed", _htf)
check("all_filters_pass returns (bool, str)", _all_filters)


# ─── 5. NEWS ──────────────────────────────────────────────────────────────────
print("\n[5] News")

def _blackout():
    from news import news_blackout_active
    blocked, reason = news_blackout_active("EUR_USD")
    assert isinstance(blocked, bool)
    assert isinstance(reason, str)

def _upcoming():
    from news import upcoming_events
    events = upcoming_events(hours=24)
    assert isinstance(events, list)
    for e in events:
        assert "currency" in e and "title" in e and "time" in e

check("news_blackout_active returns (bool, str)", _blackout)
check("upcoming_events returns list of dicts", _upcoming)


# ─── 6. REGIME (Claude) ───────────────────────────────────────────────────────
print("\n[6] Regime (Claude API)")

def _regime():
    from data import get_candles
    from strategy import add_indicators
    from regime import get_market_regime
    df = add_indicators(get_candles("EUR_USD", count=100))
    regime, reason = get_market_regime("EUR_USD", df)
    assert regime in ("trending_up", "trending_down", "ranging", "volatile"), \
        f"unexpected regime: {regime}"
    assert isinstance(reason, str)

def _regime_blocks():
    from data import get_candles
    from strategy import add_indicators
    from regime import regime_blocks_signal
    df = add_indicators(get_candles("EUR_USD", count=100))
    blocked, reason = regime_blocks_signal("EUR_USD", "buy", df)
    assert isinstance(blocked, bool)
    assert isinstance(reason, str)

check("get_market_regime returns valid regime string", _regime)
check("regime_blocks_signal returns (bool, str)", _regime_blocks)


# ─── 7. SENTIMENT (Claude) ────────────────────────────────────────────────────
print("\n[7] Sentiment (Claude API)")

def _sentiment_no_events():
    from sentiment import get_news_sentiment
    result = get_news_sentiment("EUR_USD", [])
    assert result["bias"] == "neutral"
    assert result["avoid"] is False

def _sentiment_with_events():
    from sentiment import get_news_sentiment
    events = [{"currency": "USD", "title": "Non-Farm Payrolls", "time": "13:30 UTC"}]
    result = get_news_sentiment("EUR_USD", events)
    assert result["bias"] in ("bullish", "bearish", "neutral")
    assert result["confidence"] in ("low", "medium", "high")
    assert isinstance(result["avoid"], bool)

def _sentiment_blocks():
    from sentiment import sentiment_blocks_signal
    blocked, reason = sentiment_blocks_signal("EUR_USD", "buy", [])
    assert blocked is False  # no events -> never blocks

check("get_news_sentiment: no events -> neutral", _sentiment_no_events)
check("get_news_sentiment: with events returns valid structure", _sentiment_with_events)
check("sentiment_blocks_signal: no events -> not blocked", _sentiment_blocks)


# ─── 8. RISK MANAGER ─────────────────────────────────────────────────────────
print("\n[8] Risk Manager")

def _equity():
    from trader import get_account_equity
    eq = get_account_equity()
    assert eq > 0, f"equity={eq}"

def _dynamic_units():
    from risk_manager import dynamic_units
    import config
    units = dynamic_units("EUR_USD", atr=0.0010, current_price=1.10)
    assert config.MIN_UNITS <= units <= config.MAX_UNITS, \
        f"units {units} out of bounds [{config.MIN_UNITS}, {config.MAX_UNITS}]"

def _dynamic_units_jpy():
    from risk_manager import dynamic_units
    import config
    units = dynamic_units("USD_JPY", atr=0.15, current_price=150.0)
    assert config.MIN_UNITS <= units <= config.MAX_UNITS

def _kill_switch():
    from risk_manager import check_kill_switch
    killed, reason = check_kill_switch()
    assert isinstance(killed, bool)
    assert isinstance(reason, str)

def _manage_trades():
    from risk_manager import manage_open_trades
    from trader import get_open_trades
    from data import get_price
    import config
    open_trades = get_open_trades()
    price_map = {}
    for pair in config.PAIRS:
        try:
            price_map[pair] = get_price(pair)["mid"]
        except Exception:
            pass
    manage_open_trades(price_map, open_trades)  # should not raise

check("get_account_equity returns positive float", _equity)
check("dynamic_units(EUR_USD) within MIN/MAX bounds", _dynamic_units)
check("dynamic_units(USD_JPY) within MIN/MAX bounds (USD-base pip)", _dynamic_units_jpy)
check("check_kill_switch returns (bool, str)", _kill_switch)
check("manage_open_trades runs without error", _manage_trades)


# ─── 9. TRADER ────────────────────────────────────────────────────────────────
print("\n[9] Trader")

def _open_trades():
    from trader import get_open_trades
    trades = get_open_trades()
    assert isinstance(trades, list)

def _has_position():
    from trader import has_open_position, get_open_trades
    trades = get_open_trades()
    result = has_open_position("EUR_USD", open_trades=trades)
    assert isinstance(result, bool)

def _update_sl_signature():
    from trader import update_trade_sl
    import inspect
    sig = inspect.signature(update_trade_sl)
    params = list(sig.parameters)
    assert "trade_id" in params and "new_sl" in params

check("get_open_trades returns list", _open_trades)
check("has_open_position returns bool", _has_position)
check("update_trade_sl has correct signature", _update_sl_signature)


# ─── 10. ML MODEL ─────────────────────────────────────────────────────────────
print("\n[10] ML Model")

def _build_features():
    from data import get_candles
    from strategy import add_indicators
    from ml_model import build_features
    df = add_indicators(get_candles("EUR_USD", count=200))
    feat = build_features(df)
    assert not feat.empty
    expected = ["adx", "rsi", "breakout_strength", "atr_pct", "vol_ratio"]
    for col in expected:
        assert col in feat.columns, f"missing feature: {col}"

def _ml_filter():
    from data import get_candles
    from strategy import add_indicators
    from ml_model import ml_filter_passes
    df = add_indicators(get_candles("EUR_USD", count=200))
    passed, confidence = ml_filter_passes(df, "buy")
    assert isinstance(passed, bool)
    assert 0.0 <= confidence <= 1.0, f"confidence={confidence} out of [0,1]"

check("build_features returns non-empty DataFrame with expected cols", _build_features)
check("ml_filter_passes returns (bool, float in [0,1])", _ml_filter)


# ─── 11. TRADE HISTORY ────────────────────────────────────────────────────────
print("\n[11] Trade History")

def _trade_history():
    from trade_history import get_history, record_trade_result
    before = len(get_history())
    record_trade_result("win")
    after = get_history()
    assert len(after) == before + 1
    assert after[-1] == "win"
    # undo the test record
    from trade_history import _history, _save
    _history.pop()
    _save(_history)

check("record_trade_result persists and get_history retrieves", _trade_history)


# ─── 12. BOT LOG ─────────────────────────────────────────────────────────────
print("\n[12] Bot Log")

def _bot_log():
    import bot_log, json, os
    bot_log.info("verify.py: test entry")
    assert os.path.exists(bot_log.LOG_PATH)
    with open(bot_log.LOG_PATH) as f:
        entries = json.load(f)
    assert any(e.get("message", "").startswith("verify.py") for e in entries)

check("bot_log.info writes entry to bot_log.json", _bot_log)


# ─── 13. BOT CONTROL ─────────────────────────────────────────────────────────
print("\n[13] Bot Control")

def _bot_control():
    import bot_control
    original = bot_control.is_paused()
    bot_control.pause("verify test")
    assert bot_control.is_paused() is True
    assert bot_control.status()["reason"] == "verify test"
    bot_control.resume()
    assert bot_control.is_paused() is False
    # restore original state
    if original:
        bot_control.pause("restored")

check("pause/resume/is_paused round-trip works", _bot_control)


# ─── 14. ALERTS ──────────────────────────────────────────────────────────────
print("\n[14] Alerts")

def _alerts_no_token():
    import config, alerts
    # If no token, should print to stdout without raising
    original_token = config.TELEGRAM_BOT_TOKEN
    config.TELEGRAM_BOT_TOKEN = ""
    try:
        alerts.send("verify.py: test alert (no token)")
    finally:
        config.TELEGRAM_BOT_TOKEN = original_token

def _alert_functions_exist():
    from alerts import send, trade_opened, price_alert, error_alert
    assert callable(send)
    assert callable(trade_opened)
    assert callable(price_alert)
    assert callable(error_alert)

check("alerts.send with no token falls back to print without raising", _alerts_no_token)
check("all alert functions are callable", _alert_functions_exist)


# ─── 15. MONITOR ─────────────────────────────────────────────────────────────
print("\n[15] Monitor")

def _monitor():
    from monitor import check_price_moves
    check_price_moves()   # should not raise

check("check_price_moves runs without error", _monitor)


# ─── 16. JOURNAL ─────────────────────────────────────────────────────────────
print("\n[16] Journal (Claude API)")

def _journal():
    import journal, json, os
    before = 0
    if os.path.exists(journal._JOURNAL_FILE):
        with open(journal._JOURNAL_FILE) as f:
            before = len(json.load(f))

    journal.analyse_trade({
        "instrument":    "EUR_USD",
        "direction":     "BUY",
        "entry":         1.10000,
        "exit_price":    1.10270,
        "pl_pips":       27.0,
        "rsi_at_entry":  58.0,
        "atr_ratio":     1.1,
        "ml_confidence": "n/a",
        "sentiment_bias":"neutral",
        "regime":        "trending_up",
        "news_events":   [],
    })

    with open(journal._JOURNAL_FILE) as f:
        after = json.load(f)
    assert len(after) == before + 1
    last = after[-1]
    assert "analysis" in last
    assert "score" in last["analysis"]

check("journal.analyse_trade writes scored entry to trade_journal.json", _journal)


# ─── 17. WATCHDOG ────────────────────────────────────────────────────────────
print("\n[17] Watchdog")

def _watchdog_import():
    import watchdog
    assert callable(watchdog.run)

check("watchdog imports and run() is callable", _watchdog_import)


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
passed  = [r for r in results if r[1] is True]
failed  = [r for r in results if r[1] is False]
skipped = [r for r in results if r[1] is None]

print(f"\n{'─'*60}")
print(f"Results: {len(passed)} passed  |  {len(failed)} failed  |  {len(skipped)} skipped")

if failed:
    print("\nFailed checks:")
    for name, _, msg in failed:
        print(f"  {FAIL}  {name}")
        print(f"         {msg}")

sys.exit(0 if not failed else 1)
