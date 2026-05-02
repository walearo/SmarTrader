"""
SmarTrader — functional and regression test suite.

Covers:
  - Price formatting (trader._fmt_price / _price_dp)
  - Config constants (INSTRUMENT_PRICE_DP)
  - Alert settings  (bot_control get/set/toggle)
  - Alert gating    (alerts.py per-type suppression)
  - Bot settings    (save / apply / reset)
  - Strategy SL/TP  (per-instrument rounding → _fmt_price pipeline)
  - Dashboard input validation logic (mirrors POST /api/settings rules)
  - Regression: pause/resume preserves alerts and bot settings
  - Regression: partial save preserves unrelated keys

Run:  py test_suite.py
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(close: float = 1.1, atr: float = 0.001, n: int = 60):
    """Minimal OHLCV DataFrame suitable for add_indicators()."""
    import numpy as np
    import pandas as pd

    rng    = np.random.default_rng(42)
    noise  = rng.normal(0, atr * 0.01, n)
    closes = np.maximum(close + noise, 0.0001)
    df = pd.DataFrame({
        "time":   pd.date_range("2024-01-01", periods=n, freq="h"),
        "open":   closes,
        "high":   closes * 1.001,
        "low":    closes * 0.999,
        "close":  closes,
        "volume": np.full(n, 10_000.0),
    })
    return df


# ─── 1. PRICE FORMATTING ──────────────────────────────────────────────────────

class TestPriceFormatting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import trader
        cls.t = trader

    def test_xau_usd_rounds_to_2dp(self):
        self.assertEqual(self.t._fmt_price("XAU_USD", 3326.4567), "3326.46")

    def test_xau_usd_pads_trailing_zero(self):
        self.assertEqual(self.t._fmt_price("XAU_USD", 3326.4), "3326.40")

    def test_wtico_usd_rounds_to_3dp(self):
        self.assertEqual(self.t._fmt_price("WTICO_USD", 79.4561), "79.456")

    def test_usd_jpy_rounds_to_3dp(self):
        self.assertEqual(self.t._fmt_price("USD_JPY", 150.1239), "150.124")

    def test_eur_usd_rounds_to_5dp(self):
        self.assertEqual(self.t._fmt_price("EUR_USD", 1.123456789), "1.12346")

    def test_unknown_pair_defaults_to_5dp(self):
        self.assertEqual(self.t._fmt_price("", 1.123456789), "1.12346")

    def test_gbp_usd_5dp(self):
        result = self.t._fmt_price("GBP_USD", 1.2699999)
        self.assertRegex(result, r"^\d+\.\d{5}$")

    def test_price_dp_xau(self):
        self.assertEqual(self.t._price_dp("XAU_USD"), 2)

    def test_price_dp_wtico(self):
        self.assertEqual(self.t._price_dp("WTICO_USD"), 3)

    def test_price_dp_jpy(self):
        self.assertEqual(self.t._price_dp("USD_JPY"), 3)

    def test_price_dp_default(self):
        self.assertEqual(self.t._price_dp("EUR_USD"), 5)
        self.assertEqual(self.t._price_dp("UNKNOWN"), 5)

    def test_output_is_string(self):
        result = self.t._fmt_price("XAU_USD", 3326.45)
        self.assertIsInstance(result, str)


# ─── 2. CONFIG CONSTANTS ──────────────────────────────────────────────────────

class TestConfigConstants(unittest.TestCase):

    def setUp(self):
        import config
        self.cfg = config

    def test_instrument_price_dp_dict_exists(self):
        self.assertTrue(hasattr(self.cfg, "INSTRUMENT_PRICE_DP"))
        self.assertIsInstance(self.cfg.INSTRUMENT_PRICE_DP, dict)

    def test_instrument_price_dp_default_exists(self):
        self.assertTrue(hasattr(self.cfg, "INSTRUMENT_PRICE_DP_DEFAULT"))
        self.assertEqual(self.cfg.INSTRUMENT_PRICE_DP_DEFAULT, 5)

    def test_xau_usd_dp_is_2(self):
        self.assertEqual(self.cfg.INSTRUMENT_PRICE_DP["XAU_USD"], 2)

    def test_wtico_usd_dp_is_3(self):
        self.assertEqual(self.cfg.INSTRUMENT_PRICE_DP["WTICO_USD"], 3)

    def test_usd_jpy_dp_is_3(self):
        self.assertEqual(self.cfg.INSTRUMENT_PRICE_DP["USD_JPY"], 3)

    def test_all_pairs_constant_matches_config(self):
        import bot_control
        # ALL_PAIRS and config.PAIRS should contain exactly the same instruments
        self.assertEqual(set(bot_control.ALL_PAIRS), set(self.cfg.PAIRS))


# ─── 3. ALERT SETTINGS — bot_control ─────────────────────────────────────────

class TestAlertSettings(unittest.TestCase):
    """Each test gets a fresh temp file for bot_control.json."""

    def setUp(self):
        import bot_control
        self.bc = bot_control
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp = tmp.name
        self._orig = bot_control.CONTROL_PATH
        bot_control.CONTROL_PATH = self._tmp

    def tearDown(self):
        self.bc.CONTROL_PATH = self._orig
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_all_alert_types_present(self):
        settings = self.bc.get_alert_settings()
        self.assertEqual(set(settings.keys()), set(self.bc.ALERT_TYPES))

    def test_defaults_are_all_true(self):
        for k in self.bc.ALERT_TYPES:
            self.assertTrue(self.bc.is_alert_enabled(k), f"{k} should default to True")

    def test_disable_price_alert(self):
        self.bc.set_alert("price_alert", False)
        self.assertFalse(self.bc.is_alert_enabled("price_alert"))

    def test_re_enable_after_disable(self):
        self.bc.set_alert("price_alert", False)
        self.bc.set_alert("price_alert", True)
        self.assertTrue(self.bc.is_alert_enabled("price_alert"))

    def test_disable_one_does_not_affect_others(self):
        self.bc.set_alert("price_alert", False)
        for k in self.bc.ALERT_TYPES:
            if k != "price_alert":
                self.assertTrue(self.bc.is_alert_enabled(k), f"{k} should still be True")

    def test_unknown_type_is_noop(self):
        self.bc.set_alert("NONEXISTENT_ALERT", False)
        # existing defaults should be unchanged
        for k in self.bc.ALERT_TYPES:
            self.assertTrue(self.bc.is_alert_enabled(k))

    def test_all_types_can_be_disabled(self):
        for k in self.bc.ALERT_TYPES:
            self.bc.set_alert(k, False)
        for k in self.bc.ALERT_TYPES:
            self.assertFalse(self.bc.is_alert_enabled(k))

    def test_persistence_across_reads(self):
        self.bc.set_alert("kill_switch", False)
        # re-read from disk
        settings = self.bc.get_alert_settings()
        self.assertFalse(settings["kill_switch"])


# ─── 4. REGRESSION: PAUSE/RESUME PRESERVES SETTINGS ─────────────────────────

class TestPausePreservesData(unittest.TestCase):

    def setUp(self):
        import bot_control
        self.bc = bot_control
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp = tmp.name
        self._orig = bot_control.CONTROL_PATH
        bot_control.CONTROL_PATH = self._tmp

    def tearDown(self):
        self.bc.CONTROL_PATH = self._orig
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_pause_preserves_alert_flags(self):
        self.bc.set_alert("price_alert", False)
        self.bc.pause("test")
        self.assertFalse(self.bc.is_alert_enabled("price_alert"))
        self.assertTrue(self.bc.is_paused())

    def test_resume_preserves_alert_flags(self):
        self.bc.set_alert("trade_open", False)
        self.bc.pause("test")
        self.bc.resume()
        self.assertFalse(self.bc.is_alert_enabled("trade_open"))
        self.assertFalse(self.bc.is_paused())

    def test_pause_then_resume_cycle(self):
        self.bc.set_alert("error_alert", False)
        self.bc.set_alert("kill_switch", False)
        self.bc.pause("risk limit")
        self.bc.resume()
        self.assertFalse(self.bc.is_alert_enabled("error_alert"))
        self.assertFalse(self.bc.is_alert_enabled("kill_switch"))
        self.assertTrue(self.bc.is_alert_enabled("price_alert"))


# ─── 5. BOT SETTINGS ─────────────────────────────────────────────────────────

class TestBotSettings(unittest.TestCase):

    def setUp(self):
        import bot_control, config
        self.bc  = bot_control
        self.cfg = config
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp      = tmp.name
        self._orig_path = bot_control.SETTINGS_PATH
        bot_control.SETTINGS_PATH = self._tmp
        # snapshot config values we might mutate
        self._orig_max_trades = config.MAX_CONCURRENT_TRADES
        self._orig_units      = config.UNITS
        self._orig_pairs      = list(config.PAIRS)
        self._orig_sessions   = list(config.SESSIONS)

    def tearDown(self):
        self.bc.SETTINGS_PATH = self._orig_path
        # restore any config mutations
        self.cfg.MAX_CONCURRENT_TRADES = self._orig_max_trades
        self.cfg.UNITS                 = self._orig_units
        self.cfg.PAIRS                 = self._orig_pairs
        self.cfg.SESSIONS              = self._orig_sessions
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    # defaults

    def test_defaults_match_config(self):
        s = self.bc.get_bot_settings()
        self.assertEqual(s["MAX_CONCURRENT_TRADES"], self.cfg.MAX_CONCURRENT_TRADES)
        self.assertEqual(s["RISK_PCT_PER_TRADE"],    self.cfg.RISK_PCT_PER_TRADE)
        self.assertEqual(s["PAIRS"],                 list(self.cfg.PAIRS))

    def test_defaults_contain_all_keys(self):
        s = self.bc.get_bot_settings()
        for k in self.bc.SETTINGS_KEYS:
            self.assertIn(k, s, f"Missing key: {k}")

    # save & read

    def test_save_integer_setting(self):
        self.bc.save_bot_settings({"MAX_CONCURRENT_TRADES": 5})
        self.assertEqual(self.bc.get_bot_settings()["MAX_CONCURRENT_TRADES"], 5)

    def test_save_float_setting(self):
        self.bc.save_bot_settings({"RISK_PCT_PER_TRADE": 2.5})
        self.assertAlmostEqual(self.bc.get_bot_settings()["RISK_PCT_PER_TRADE"], 2.5)

    def test_save_pairs(self):
        self.bc.save_bot_settings({"PAIRS": ["EUR_USD", "GBP_USD"]})
        self.assertEqual(self.bc.get_bot_settings()["PAIRS"], ["EUR_USD", "GBP_USD"])

    def test_save_sessions(self):
        self.bc.save_bot_settings({"SESSIONS": [[8, 17], [14, 22]]})
        self.assertEqual(self.bc.get_bot_settings()["SESSIONS"], [[8, 17], [14, 22]])

    def test_partial_save_preserves_other_keys(self):
        self.bc.save_bot_settings({"MAX_CONCURRENT_TRADES": 3})
        self.bc.save_bot_settings({"UNITS": 2000})
        s = self.bc.get_bot_settings()
        self.assertEqual(s["MAX_CONCURRENT_TRADES"], 3)
        self.assertEqual(s["UNITS"], 2000)

    # apply

    def test_apply_patches_config_integer(self):
        self.bc.save_bot_settings({"MAX_CONCURRENT_TRADES": 7})
        self.bc.apply_bot_settings()
        self.assertEqual(self.cfg.MAX_CONCURRENT_TRADES, 7)

    def test_apply_patches_config_pairs(self):
        new_pairs = ["EUR_USD", "USD_JPY"]
        self.bc.save_bot_settings({"PAIRS": new_pairs})
        self.bc.apply_bot_settings()
        self.assertEqual(list(self.cfg.PAIRS), new_pairs)

    def test_apply_sessions_are_tuples(self):
        self.bc.save_bot_settings({"SESSIONS": [[9, 17], [14, 21]]})
        self.bc.apply_bot_settings()
        for s in self.cfg.SESSIONS:
            self.assertIsInstance(s, tuple, "SESSIONS entries should be tuples after apply")

    def test_apply_noop_on_empty_file(self):
        original = self.cfg.MAX_CONCURRENT_TRADES
        self.bc.apply_bot_settings()   # file is empty {}
        self.assertEqual(self.cfg.MAX_CONCURRENT_TRADES, original)

    # reset

    def test_reset_removes_file(self):
        self.bc.save_bot_settings({"MAX_CONCURRENT_TRADES": 9})
        self.bc.reset_bot_settings()
        self.assertFalse(os.path.exists(self._tmp))

    def test_after_reset_defaults_return(self):
        self.bc.save_bot_settings({"MAX_CONCURRENT_TRADES": 9})
        self.bc.reset_bot_settings()
        s = self.bc.get_bot_settings()
        self.assertEqual(s["MAX_CONCURRENT_TRADES"], self._orig_max_trades)


# ─── 6. ALERT GATING (alerts.py) ─────────────────────────────────────────────

class TestAlertGating(unittest.TestCase):
    """Verify each alert function respects its enabled/disabled flag."""

    def setUp(self):
        import bot_control
        self.bc = bot_control
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp  = tmp.name
        self._orig = bot_control.CONTROL_PATH
        bot_control.CONTROL_PATH = self._tmp

    def tearDown(self):
        self.bc.CONTROL_PATH = self._orig
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def _call_with_send_spy(self, fn, *args, **kwargs):
        """Return (called: bool) — True if requests.post was invoked."""
        import alerts
        with patch("alerts.requests.post") as mock_post, \
             patch("alerts.config.TELEGRAM_BOT_TOKEN", "tok"), \
             patch("alerts.config.TELEGRAM_CHAT_ID",   "cid"):
            fn(*args, **kwargs)
            return mock_post.called

    def test_price_alert_fires_when_enabled(self):
        self.bc.set_alert("price_alert", True)
        import alerts
        self.assertTrue(self._call_with_send_spy(alerts.price_alert, "EUR_USD", 1.1, 25.0))

    def test_price_alert_suppressed_when_disabled(self):
        self.bc.set_alert("price_alert", False)
        import alerts
        self.assertFalse(self._call_with_send_spy(alerts.price_alert, "EUR_USD", 1.1, 25.0))

    def test_trade_opened_fires_when_enabled(self):
        self.bc.set_alert("trade_open", True)
        import alerts
        self.assertTrue(self._call_with_send_spy(alerts.trade_opened, "EUR_USD", "buy", 1.1, 1.09, 1.12))

    def test_trade_opened_suppressed_when_disabled(self):
        self.bc.set_alert("trade_open", False)
        import alerts
        self.assertFalse(self._call_with_send_spy(alerts.trade_opened, "EUR_USD", "buy", 1.1, 1.09, 1.12))

    def test_trade_closed_fires_when_enabled(self):
        self.bc.set_alert("trade_close", True)
        import alerts
        self.assertTrue(self._call_with_send_spy(alerts.trade_closed, "EUR_USD", "win", 20.5))

    def test_trade_closed_suppressed_when_disabled(self):
        self.bc.set_alert("trade_close", False)
        import alerts
        self.assertFalse(self._call_with_send_spy(alerts.trade_closed, "EUR_USD", "loss", -10.0))

    def test_kill_switch_fires_when_enabled(self):
        self.bc.set_alert("kill_switch", True)
        import alerts
        self.assertTrue(self._call_with_send_spy(alerts.kill_switch_alert, "Daily loss"))

    def test_kill_switch_suppressed_when_disabled(self):
        self.bc.set_alert("kill_switch", False)
        import alerts
        self.assertFalse(self._call_with_send_spy(alerts.kill_switch_alert, "Daily loss"))

    def test_error_alert_fires_when_enabled(self):
        self.bc.set_alert("error_alert", True)
        import alerts
        self.assertTrue(self._call_with_send_spy(alerts.error_alert, "ctx", Exception("e")))

    def test_error_alert_suppressed_when_disabled(self):
        self.bc.set_alert("error_alert", False)
        import alerts
        self.assertFalse(self._call_with_send_spy(alerts.error_alert, "ctx", Exception("e")))

    def test_disabling_one_alert_does_not_suppress_another(self):
        self.bc.set_alert("price_alert", False)
        import alerts
        # trade_closed should still fire
        self.assertTrue(self._call_with_send_spy(alerts.trade_closed, "EUR_USD", "win", 10.0))


# ─── 7. STRATEGY SL/TP → PRICE FORMAT PIPELINE ───────────────────────────────

class TestSlTpPrecisionPipeline(unittest.TestCase):
    """
    Verify that get_sl_tp() + _fmt_price() together produce strings with
    the correct decimal precision for each instrument.
    """

    def _sltp_formatted(self, pair: str, close: float, atr: float, signal: str):
        from strategy import get_sl_tp
        from trader import _fmt_price
        df = _make_ohlcv_df(close=close, atr=atr)
        sl, tp = get_sl_tp(df, signal, entry_price=close, pair=pair)
        return _fmt_price(pair, sl), _fmt_price(pair, tp)

    def test_xau_usd_buy_produces_2dp_strings(self):
        sl_s, tp_s = self._sltp_formatted("XAU_USD", 3326.0, 5.0, "buy")
        self.assertRegex(sl_s, r"^\d+\.\d{2}$", f"XAU SL: {sl_s}")
        self.assertRegex(tp_s, r"^\d+\.\d{2}$", f"XAU TP: {tp_s}")

    def test_xau_usd_sell_produces_2dp_strings(self):
        sl_s, tp_s = self._sltp_formatted("XAU_USD", 3326.0, 5.0, "sell")
        self.assertRegex(sl_s, r"^\d+\.\d{2}$")
        self.assertRegex(tp_s, r"^\d+\.\d{2}$")

    def test_wtico_usd_produces_3dp_strings(self):
        sl_s, tp_s = self._sltp_formatted("WTICO_USD", 79.0, 0.5, "buy")
        self.assertRegex(sl_s, r"^\d+\.\d{3}$", f"WTICO SL: {sl_s}")
        self.assertRegex(tp_s, r"^\d+\.\d{3}$", f"WTICO TP: {tp_s}")

    def test_usd_jpy_produces_3dp_strings(self):
        sl_s, tp_s = self._sltp_formatted("USD_JPY", 150.0, 0.3, "sell")
        self.assertRegex(sl_s, r"^\d+\.\d{3}$")
        self.assertRegex(tp_s, r"^\d+\.\d{3}$")

    def test_eur_usd_produces_5dp_strings(self):
        sl_s, tp_s = self._sltp_formatted("EUR_USD", 1.10, 0.001, "buy")
        self.assertRegex(sl_s, r"^\d+\.\d{5}$")
        self.assertRegex(tp_s, r"^\d+\.\d{5}$")

    def test_sl_below_entry_for_buy(self):
        from strategy import get_sl_tp
        df = _make_ohlcv_df(close=3326.0, atr=5.0)
        sl, tp = get_sl_tp(df, "buy", entry_price=3326.0, pair="XAU_USD")
        self.assertLess(sl, 3326.0)
        self.assertGreater(tp, 3326.0)

    def test_sl_above_entry_for_sell(self):
        from strategy import get_sl_tp
        df = _make_ohlcv_df(close=3326.0, atr=5.0)
        sl, tp = get_sl_tp(df, "sell", entry_price=3326.0, pair="XAU_USD")
        self.assertGreater(sl, 3326.0)
        self.assertLess(tp, 3326.0)

    def test_no_pair_arg_still_works(self):
        from strategy import get_sl_tp
        from trader import _fmt_price
        df = _make_ohlcv_df(close=1.1, atr=0.001)
        sl, tp = get_sl_tp(df, "buy", entry_price=1.1)   # no pair arg
        # should not raise; default 5 dp
        sl_s = _fmt_price("", sl)
        self.assertRegex(sl_s, r"^\d+\.\d{5}$")


# ─── 8. SETTINGS VALIDATION LOGIC ────────────────────────────────────────────

class TestSettingsValidation(unittest.TestCase):
    """
    Mirror the POST /api/settings validation logic and test boundary conditions.
    """

    @staticmethod
    def _validate(body: dict) -> list[str]:
        import bot_control
        errors = []
        _int_ranges   = [("MAX_CONCURRENT_TRADES", 1, 10), ("UNITS", 100, 100_000),
                         ("ALERT_PRICE_MOVE_PIPS", 1, 500), ("NEWS_BLACKOUT_MINUTES", 0, 180),
                         ("MAX_TRADE_HOURS", 0, 168), ("MAX_DAILY_TRADES", 0, 50),
                         ("MAX_CONSECUTIVE_LOSSES", 0, 20)]
        _float_ranges = [("RISK_PCT_PER_TRADE", 0.1, 5.0),
                         ("MAX_DAILY_LOSS_PCT", 0.5, 20.0), ("MAX_WEEKLY_LOSS_PCT", 1.0, 50.0),
                         ("SPREAD_MAX_PIPS", 0.5, 200.0)]

        for key, lo, hi in _int_ranges:
            if key in body:
                v = body[key]
                if not isinstance(v, int) or not (lo <= v <= hi):
                    errors.append(key)

        for key, lo, hi in _float_ranges:
            if key in body:
                v = body[key]
                if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                    errors.append(key)

        if "PAIRS" in body:
            v = body["PAIRS"]
            valid = set(bot_control.ALL_PAIRS)
            if not isinstance(v, list) or not v or not all(p in valid for p in v):
                errors.append("PAIRS")

        if "SESSIONS" in body:
            v = body["SESSIONS"]
            if not isinstance(v, list) or not v:
                errors.append("SESSIONS")
            else:
                for s in v:
                    if not isinstance(s, list) or len(s) != 2:
                        errors.append("SESSIONS"); break
                    start, end = s
                    if not isinstance(start, int) or not isinstance(end, int) \
                            or not (0 <= start < end <= 24):
                        errors.append("SESSIONS"); break

        return errors

    # happy path
    def test_valid_full_body_passes(self):
        body = {
            "MAX_CONCURRENT_TRADES": 3, "UNITS": 2000, "RISK_PCT_PER_TRADE": 1.5,
            "MAX_DAILY_LOSS_PCT": 3.0,  "MAX_WEEKLY_LOSS_PCT": 8.0,
            "ALERT_PRICE_MOVE_PIPS": 20, "NEWS_BLACKOUT_MINUTES": 30,
            "PAIRS": ["EUR_USD", "GBP_USD"],
            "SESSIONS": [[7, 16], [13, 22]],
        }
        self.assertEqual(self._validate(body), [])

    def test_empty_body_passes(self):
        self.assertEqual(self._validate({}), [])

    # MAX_CONCURRENT_TRADES
    def test_max_trades_zero_fails(self):
        self.assertIn("MAX_CONCURRENT_TRADES", self._validate({"MAX_CONCURRENT_TRADES": 0}))

    def test_max_trades_eleven_fails(self):
        self.assertIn("MAX_CONCURRENT_TRADES", self._validate({"MAX_CONCURRENT_TRADES": 11}))

    def test_max_trades_boundary_1_passes(self):
        self.assertNotIn("MAX_CONCURRENT_TRADES", self._validate({"MAX_CONCURRENT_TRADES": 1}))

    def test_max_trades_boundary_10_passes(self):
        self.assertNotIn("MAX_CONCURRENT_TRADES", self._validate({"MAX_CONCURRENT_TRADES": 10}))

    def test_max_trades_float_fails(self):
        self.assertIn("MAX_CONCURRENT_TRADES", self._validate({"MAX_CONCURRENT_TRADES": 2.5}))

    # UNITS
    def test_units_too_low_fails(self):
        self.assertIn("UNITS", self._validate({"UNITS": 99}))

    def test_units_too_high_fails(self):
        self.assertIn("UNITS", self._validate({"UNITS": 100_001}))

    def test_units_boundary_passes(self):
        self.assertNotIn("UNITS", self._validate({"UNITS": 100}))
        self.assertNotIn("UNITS", self._validate({"UNITS": 100_000}))

    # RISK_PCT_PER_TRADE
    def test_risk_too_high_fails(self):
        self.assertIn("RISK_PCT_PER_TRADE", self._validate({"RISK_PCT_PER_TRADE": 5.1}))

    def test_risk_too_low_fails(self):
        self.assertIn("RISK_PCT_PER_TRADE", self._validate({"RISK_PCT_PER_TRADE": 0.0}))

    def test_risk_boundary_passes(self):
        self.assertNotIn("RISK_PCT_PER_TRADE", self._validate({"RISK_PCT_PER_TRADE": 0.1}))
        self.assertNotIn("RISK_PCT_PER_TRADE", self._validate({"RISK_PCT_PER_TRADE": 5.0}))

    # MAX_DAILY / MAX_WEEKLY
    def test_daily_loss_zero_fails(self):
        self.assertIn("MAX_DAILY_LOSS_PCT", self._validate({"MAX_DAILY_LOSS_PCT": 0.0}))

    def test_weekly_loss_too_low_fails(self):
        self.assertIn("MAX_WEEKLY_LOSS_PCT", self._validate({"MAX_WEEKLY_LOSS_PCT": 0.5}))

    # NEWS_BLACKOUT_MINUTES
    def test_news_blackout_zero_passes(self):
        self.assertNotIn("NEWS_BLACKOUT_MINUTES", self._validate({"NEWS_BLACKOUT_MINUTES": 0}))

    def test_news_blackout_too_high_fails(self):
        self.assertIn("NEWS_BLACKOUT_MINUTES", self._validate({"NEWS_BLACKOUT_MINUTES": 181}))

    # PAIRS
    def test_empty_pairs_fails(self):
        self.assertIn("PAIRS", self._validate({"PAIRS": []}))

    def test_invalid_pair_name_fails(self):
        self.assertIn("PAIRS", self._validate({"PAIRS": ["FAKE_USD"]}))

    def test_single_valid_pair_passes(self):
        self.assertNotIn("PAIRS", self._validate({"PAIRS": ["EUR_USD"]}))

    def test_all_valid_pairs_passes(self):
        import bot_control
        self.assertNotIn("PAIRS", self._validate({"PAIRS": bot_control.ALL_PAIRS}))

    def test_mixed_valid_invalid_fails(self):
        self.assertIn("PAIRS", self._validate({"PAIRS": ["EUR_USD", "BAD_USD"]}))

    # SESSIONS
    def test_empty_sessions_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": []}))

    def test_session_start_after_end_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[17, 7]]}))

    def test_session_equal_start_end_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[8, 8]]}))

    def test_session_end_exceeds_24_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[0, 25]]}))

    def test_session_negative_start_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[-1, 8]]}))

    def test_valid_single_session_passes(self):
        self.assertNotIn("SESSIONS", self._validate({"SESSIONS": [[7, 16]]}))

    def test_session_midnight_span_passes(self):
        self.assertNotIn("SESSIONS", self._validate({"SESSIONS": [[0, 24]]}))

    def test_session_wrong_length_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[7, 16, 0]]}))

    def test_session_float_hours_fails(self):
        self.assertIn("SESSIONS", self._validate({"SESSIONS": [[7.5, 16]]}))


# ─── entry point ──────────────────────────────────────────────────────────────

# ─── 9. REGRESSION: DRAWDOWN LIMITS REFLECT SAVED SETTINGS ──────────────────

class TestDrawdownLimitsFollowSettings(unittest.TestCase):
    """
    Regression for the dashboard drawdown display showing stale limits.

    The dashboard is a separate process — it never runs the bot's cycle, so
    apply_bot_settings() must be called inside api_data() to patch the
    dashboard's own config module before get_drawdown_status() reads the limits.
    """

    def setUp(self):
        import bot_control, config
        self.bc  = bot_control
        self.cfg = config
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp           = tmp.name
        self._orig_path     = bot_control.SETTINGS_PATH
        self._orig_daily    = config.MAX_DAILY_LOSS_PCT
        self._orig_weekly   = config.MAX_WEEKLY_LOSS_PCT
        bot_control.SETTINGS_PATH = self._tmp

    def tearDown(self):
        self.bc.SETTINGS_PATH    = self._orig_path
        self.cfg.MAX_DAILY_LOSS_PCT  = self._orig_daily
        self.cfg.MAX_WEEKLY_LOSS_PCT = self._orig_weekly
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_apply_bot_settings_updates_daily_limit(self):
        self.bc.save_bot_settings({"MAX_DAILY_LOSS_PCT": 5.0})
        self.bc.apply_bot_settings()
        self.assertAlmostEqual(self.cfg.MAX_DAILY_LOSS_PCT, 5.0)

    def test_apply_bot_settings_updates_weekly_limit(self):
        self.bc.save_bot_settings({"MAX_WEEKLY_LOSS_PCT": 12.0})
        self.bc.apply_bot_settings()
        self.assertAlmostEqual(self.cfg.MAX_WEEKLY_LOSS_PCT, 12.0)

    def test_get_drawdown_status_uses_updated_limits(self):
        from risk_manager import get_drawdown_status
        self.bc.save_bot_settings({"MAX_DAILY_LOSS_PCT": 7.0, "MAX_WEEKLY_LOSS_PCT": 15.0})
        self.bc.apply_bot_settings()
        dd = get_drawdown_status()
        self.assertAlmostEqual(dd["daily_limit"],  7.0)
        self.assertAlmostEqual(dd["weekly_limit"], 15.0)

    def test_stale_config_shows_wrong_limit_without_apply(self):
        """Demonstrates the original bug: without apply, limits are stale."""
        from risk_manager import get_drawdown_status
        # Save new limits but do NOT call apply_bot_settings
        self.bc.save_bot_settings({"MAX_DAILY_LOSS_PCT": 9.0})
        dd = get_drawdown_status()
        # Limits should still be the original config value (bug scenario)
        self.assertAlmostEqual(dd["daily_limit"], self._orig_daily)

    def test_limits_restored_on_reset(self):
        from risk_manager import get_drawdown_status
        self.bc.save_bot_settings({"MAX_DAILY_LOSS_PCT": 7.0})
        self.bc.apply_bot_settings()
        self.bc.reset_bot_settings()
        # reset removes file; manually restore config to simulate bot behaviour
        self.cfg.MAX_DAILY_LOSS_PCT = self._orig_daily
        dd = get_drawdown_status()
        self.assertAlmostEqual(dd["daily_limit"], self._orig_daily)


# ─── 10. NEW SETTINGS VALIDATION BOUNDARIES ──────────────────────────────────

class TestNewSettingsValidation(unittest.TestCase):
    """Boundary tests for the 4 new settings fields added in the last session."""

    _validate = staticmethod(TestSettingsValidation._validate)   # reuse the updated method

    # SPREAD_MAX_PIPS (float 0.5–200.0)
    def test_spread_min_boundary_passes(self):
        self.assertNotIn("SPREAD_MAX_PIPS", self._validate({"SPREAD_MAX_PIPS": 0.5}))

    def test_spread_max_boundary_passes(self):
        self.assertNotIn("SPREAD_MAX_PIPS", self._validate({"SPREAD_MAX_PIPS": 200.0}))

    def test_spread_below_min_fails(self):
        self.assertIn("SPREAD_MAX_PIPS", self._validate({"SPREAD_MAX_PIPS": 0.4}))

    def test_spread_above_max_fails(self):
        self.assertIn("SPREAD_MAX_PIPS", self._validate({"SPREAD_MAX_PIPS": 200.1}))

    def test_spread_typical_value_passes(self):
        self.assertNotIn("SPREAD_MAX_PIPS", self._validate({"SPREAD_MAX_PIPS": 3.0}))

    # MAX_TRADE_HOURS (int 0–168)
    def test_trade_hours_zero_passes(self):
        self.assertNotIn("MAX_TRADE_HOURS", self._validate({"MAX_TRADE_HOURS": 0}))

    def test_trade_hours_max_passes(self):
        self.assertNotIn("MAX_TRADE_HOURS", self._validate({"MAX_TRADE_HOURS": 168}))

    def test_trade_hours_too_high_fails(self):
        self.assertIn("MAX_TRADE_HOURS", self._validate({"MAX_TRADE_HOURS": 169}))

    def test_trade_hours_negative_fails(self):
        self.assertIn("MAX_TRADE_HOURS", self._validate({"MAX_TRADE_HOURS": -1}))

    def test_trade_hours_float_fails(self):
        self.assertIn("MAX_TRADE_HOURS", self._validate({"MAX_TRADE_HOURS": 24.5}))

    # MAX_DAILY_TRADES (int 0–50)
    def test_daily_trades_zero_passes(self):
        self.assertNotIn("MAX_DAILY_TRADES", self._validate({"MAX_DAILY_TRADES": 0}))

    def test_daily_trades_max_passes(self):
        self.assertNotIn("MAX_DAILY_TRADES", self._validate({"MAX_DAILY_TRADES": 50}))

    def test_daily_trades_too_high_fails(self):
        self.assertIn("MAX_DAILY_TRADES", self._validate({"MAX_DAILY_TRADES": 51}))

    def test_daily_trades_negative_fails(self):
        self.assertIn("MAX_DAILY_TRADES", self._validate({"MAX_DAILY_TRADES": -1}))

    # MAX_CONSECUTIVE_LOSSES (int 0–20)
    def test_consec_losses_zero_passes(self):
        self.assertNotIn("MAX_CONSECUTIVE_LOSSES", self._validate({"MAX_CONSECUTIVE_LOSSES": 0}))

    def test_consec_losses_max_passes(self):
        self.assertNotIn("MAX_CONSECUTIVE_LOSSES", self._validate({"MAX_CONSECUTIVE_LOSSES": 20}))

    def test_consec_losses_too_high_fails(self):
        self.assertIn("MAX_CONSECUTIVE_LOSSES", self._validate({"MAX_CONSECUTIVE_LOSSES": 21}))

    def test_consec_losses_negative_fails(self):
        self.assertIn("MAX_CONSECUTIVE_LOSSES", self._validate({"MAX_CONSECUTIVE_LOSSES": -1}))


# ─── 11. NEW SETTINGS DEFAULTS & APPLY ───────────────────────────────────────

class TestNewSettingsDefaults(unittest.TestCase):
    """Verify the 4 new settings keys are present in SETTINGS_KEYS and get_bot_settings()."""

    def setUp(self):
        import bot_control, config
        self.bc  = bot_control
        self.cfg = config
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{}")
        tmp.close()
        self._tmp      = tmp.name
        self._orig_path = bot_control.SETTINGS_PATH
        bot_control.SETTINGS_PATH = self._tmp
        # snapshot originals
        self._orig_spread  = config.SPREAD_MAX_PIPS
        self._orig_hours   = config.MAX_TRADE_HOURS
        self._orig_daily   = config.MAX_DAILY_TRADES
        self._orig_consec  = config.MAX_CONSECUTIVE_LOSSES

    def tearDown(self):
        self.bc.SETTINGS_PATH   = self._orig_path
        self.cfg.SPREAD_MAX_PIPS       = self._orig_spread
        self.cfg.MAX_TRADE_HOURS       = self._orig_hours
        self.cfg.MAX_DAILY_TRADES      = self._orig_daily
        self.cfg.MAX_CONSECUTIVE_LOSSES = self._orig_consec
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_new_keys_in_settings_keys(self):
        new_keys = {"SPREAD_MAX_PIPS", "MAX_TRADE_HOURS", "MAX_DAILY_TRADES", "MAX_CONSECUTIVE_LOSSES"}
        self.assertTrue(new_keys.issubset(set(self.bc.SETTINGS_KEYS)))

    def test_spread_max_pips_default_matches_config(self):
        self.assertAlmostEqual(self.bc.get_bot_settings()["SPREAD_MAX_PIPS"], self.cfg.SPREAD_MAX_PIPS)

    def test_max_trade_hours_default_matches_config(self):
        self.assertEqual(self.bc.get_bot_settings()["MAX_TRADE_HOURS"], self.cfg.MAX_TRADE_HOURS)

    def test_max_daily_trades_default_matches_config(self):
        self.assertEqual(self.bc.get_bot_settings()["MAX_DAILY_TRADES"], self.cfg.MAX_DAILY_TRADES)

    def test_max_consecutive_losses_default_matches_config(self):
        self.assertEqual(self.bc.get_bot_settings()["MAX_CONSECUTIVE_LOSSES"], self.cfg.MAX_CONSECUTIVE_LOSSES)

    def test_new_settings_saved_and_read(self):
        self.bc.save_bot_settings({"SPREAD_MAX_PIPS": 5.0, "MAX_TRADE_HOURS": 48,
                                   "MAX_DAILY_TRADES": 10, "MAX_CONSECUTIVE_LOSSES": 3})
        s = self.bc.get_bot_settings()
        self.assertAlmostEqual(s["SPREAD_MAX_PIPS"], 5.0)
        self.assertEqual(s["MAX_TRADE_HOURS"], 48)
        self.assertEqual(s["MAX_DAILY_TRADES"], 10)
        self.assertEqual(s["MAX_CONSECUTIVE_LOSSES"], 3)

    def test_new_settings_applied_to_config(self):
        self.bc.save_bot_settings({"SPREAD_MAX_PIPS": 7.0, "MAX_TRADE_HOURS": 12,
                                   "MAX_DAILY_TRADES": 8,  "MAX_CONSECUTIVE_LOSSES": 5})
        self.bc.apply_bot_settings()
        self.assertAlmostEqual(self.cfg.SPREAD_MAX_PIPS, 7.0)
        self.assertEqual(self.cfg.MAX_TRADE_HOURS, 12)
        self.assertEqual(self.cfg.MAX_DAILY_TRADES, 8)
        self.assertEqual(self.cfg.MAX_CONSECUTIVE_LOSSES, 5)


# ─── 12. CONSECUTIVE LOSSES ───────────────────────────────────────────────────

class TestConsecutiveLosses(unittest.TestCase):
    """Unit tests for trade_history.consecutive_losses()."""

    def _run(self, history: list) -> int:
        import trade_history
        with patch("trade_history.db.history_recent", return_value=history):
            return trade_history.consecutive_losses()

    def test_empty_history_returns_zero(self):
        self.assertEqual(self._run([]), 0)

    def test_single_win_returns_zero(self):
        self.assertEqual(self._run(["win"]), 0)

    def test_single_loss_returns_one(self):
        self.assertEqual(self._run(["loss"]), 1)

    def test_three_consecutive_losses(self):
        self.assertEqual(self._run(["win", "loss", "loss", "loss"]), 3)

    def test_win_breaks_streak(self):
        self.assertEqual(self._run(["loss", "loss", "win", "loss"]), 1)

    def test_all_losses_counted(self):
        self.assertEqual(self._run(["loss"] * 5), 5)

    def test_win_at_end_resets_count(self):
        self.assertEqual(self._run(["loss", "loss", "win"]), 0)

    def test_alternating_returns_one(self):
        self.assertEqual(self._run(["win", "loss", "win", "loss"]), 1)


# ─── 13. SPREAD FILTER LOGIC ─────────────────────────────────────────────────

class TestSpreadFilterLogic(unittest.TestCase):
    """Verify the spread pip calculation and per-instrument override logic."""

    def _spread_pips(self, pair: str, bid: float, ask: float) -> float:
        import config
        pip = config.INSTRUMENT_PIP.get(pair, config.INSTRUMENT_PIP_DEFAULT)
        return (ask - bid) / pip

    def _spread_limit(self, pair: str) -> float:
        import config
        return config.INSTRUMENT_SPREAD_MAX_PIPS.get(pair, config.SPREAD_MAX_PIPS)

    def _is_blocked(self, pair: str, bid: float, ask: float) -> bool:
        return self._spread_pips(pair, bid, ask) > self._spread_limit(pair)

    def test_eur_usd_tight_spread_passes(self):
        # 0.0002 / 0.0001 = 2 pips, limit 3.0 → pass
        self.assertFalse(self._is_blocked("EUR_USD", 1.10000, 1.10020))

    def test_eur_usd_wide_spread_blocked(self):
        # 0.00040 / 0.0001 = 4 pips, limit 3.0 → block
        self.assertTrue(self._is_blocked("EUR_USD", 1.10000, 1.10040))

    def test_eur_usd_exact_limit_passes(self):
        # exactly 3.0 pips (not strictly greater) → pass
        self.assertFalse(self._is_blocked("EUR_USD", 1.10000, 1.10030))

    def test_xau_usd_uses_per_instrument_limit(self):
        import config
        limit = config.INSTRUMENT_SPREAD_MAX_PIPS["XAU_USD"]
        self.assertEqual(limit, 80.0)

    def test_xau_usd_normal_spread_passes(self):
        # 0.30 / 0.01 = 30 pips, limit 80.0 → pass
        self.assertFalse(self._is_blocked("XAU_USD", 3326.00, 3326.30))

    def test_xau_usd_excessive_spread_blocked(self):
        # 0.90 / 0.01 = 90 pips, limit 80.0 → block
        self.assertTrue(self._is_blocked("XAU_USD", 3326.00, 3326.90))

    def test_wtico_usd_normal_spread_passes(self):
        # 0.030 / 0.001 = 30 pips, limit 50.0 → pass
        self.assertFalse(self._is_blocked("WTICO_USD", 79.000, 79.030))

    def test_wtico_usd_excessive_spread_blocked(self):
        # 0.060 / 0.001 = 60 pips, limit 50.0 → block
        self.assertTrue(self._is_blocked("WTICO_USD", 79.000, 79.060))

    def test_usd_jpy_pip_size_is_0_01(self):
        import config
        self.assertEqual(config.INSTRUMENT_PIP["USD_JPY"], 0.01)

    def test_unknown_pair_uses_global_limit(self):
        import config
        self.assertNotIn("GBP_USD", config.INSTRUMENT_SPREAD_MAX_PIPS)
        self.assertEqual(self._spread_limit("GBP_USD"), config.SPREAD_MAX_PIPS)


# ─── 14. DB: PAIR STATS ───────────────────────────────────────────────────────

class TestPairStats(unittest.TestCase):
    """Tests for db.pair_stats() against an isolated temp database."""

    def setUp(self):
        import db as _db
        self._db = _db
        self._orig_path = _db.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self._tmp_name = tmp.name
        _db.DB_PATH = self._tmp_name
        # create tables directly — avoids _migrate_json() which imports live JSON files
        conn = _db._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bot_log (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL, result TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        self._db.DB_PATH = self._orig_path
        if os.path.exists(self._tmp_name):
            os.unlink(self._tmp_name)

    def _insert(self, pair: str, result: str, pnl_pips: float) -> None:
        self._db.log_append({"type": "trade_close", "pair": pair,
                             "result": result, "pnl_pips": pnl_pips, "time": "2024-01-01"})

    def test_empty_db_returns_empty(self):
        self.assertEqual(self._db.pair_stats(), [])

    def test_single_win(self):
        self._insert("EUR_USD", "win", 25.0)
        stats = self._db.pair_stats()
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertEqual(s["pair"], "EUR_USD")
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["losses"], 0)
        self.assertEqual(s["trades"], 1)
        self.assertAlmostEqual(s["pnl_pips"], 25.0)
        self.assertAlmostEqual(s["win_rate"], 100.0)

    def test_single_loss(self):
        self._insert("GBP_USD", "loss", -15.0)
        s = self._db.pair_stats()[0]
        self.assertEqual(s["losses"], 1)
        self.assertEqual(s["wins"], 0)
        self.assertAlmostEqual(s["win_rate"], 0.0)

    def test_win_rate_calculation(self):
        self._insert("EUR_USD", "win", 25.0)
        self._insert("EUR_USD", "loss", -15.0)
        self._insert("EUR_USD", "win", 20.0)
        s = self._db.pair_stats()[0]
        self.assertEqual(s["trades"], 3)
        self.assertEqual(s["wins"], 2)
        self.assertAlmostEqual(s["win_rate"], 66.7, places=1)

    def test_multiple_pairs_sorted_alphabetically(self):
        self._insert("GBP_USD", "win", 10.0)
        self._insert("EUR_USD", "win", 20.0)
        stats = self._db.pair_stats()
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["pair"], "EUR_USD")
        self.assertEqual(stats[1]["pair"], "GBP_USD")

    def test_pnl_accumulates_across_trades(self):
        self._insert("EUR_USD", "win", 25.0)
        self._insert("EUR_USD", "loss", -10.0)
        s = self._db.pair_stats()[0]
        self.assertAlmostEqual(s["pnl_pips"], 15.0)

    def test_non_trade_close_entries_ignored(self):
        self._db.log_append({"type": "signal", "pair": "EUR_USD", "time": "2024-01-01"})
        self.assertEqual(self._db.pair_stats(), [])


# ─── 15. DB: TRADE PNL SERIES ─────────────────────────────────────────────────

class TestTradePnlSeries(unittest.TestCase):
    """Tests for db.trade_pnl_series() against an isolated temp database."""

    def setUp(self):
        import db as _db
        self._db = _db
        self._orig_path = _db.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self._tmp_name = tmp.name
        _db.DB_PATH = self._tmp_name
        conn = _db._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bot_log (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        self._db.DB_PATH = self._orig_path
        if os.path.exists(self._tmp_name):
            os.unlink(self._tmp_name)

    def _insert(self, pnl_pips: float) -> None:
        result = "win" if pnl_pips > 0 else "loss"
        self._db.log_append({"type": "trade_close", "pair": "EUR_USD",
                             "result": result, "pnl_pips": pnl_pips, "time": "2024-01-01"})

    def test_empty_db_returns_empty(self):
        self.assertEqual(self._db.trade_pnl_series(), [])

    def test_single_entry(self):
        self._insert(25.0)
        series = self._db.trade_pnl_series()
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0], 25.0)

    def test_ordered_oldest_first(self):
        for pnl in [10.0, -5.0, 20.0]:
            self._insert(pnl)
        series = self._db.trade_pnl_series()
        self.assertAlmostEqual(series[0], 10.0)
        self.assertAlmostEqual(series[1], -5.0)
        self.assertAlmostEqual(series[2], 20.0)

    def test_non_trade_close_excluded(self):
        self._db.log_append({"type": "signal", "pair": "EUR_USD", "time": "2024-01-01"})
        self._insert(15.0)
        series = self._db.trade_pnl_series()
        self.assertEqual(len(series), 1)

    def test_values_are_floats(self):
        self._insert(10.0)
        series = self._db.trade_pnl_series()
        self.assertIsInstance(series[0], float)

    def test_negative_pnl_included(self):
        self._insert(-12.5)
        series = self._db.trade_pnl_series()
        self.assertAlmostEqual(series[0], -12.5)


# ─── 16. NEWS POST-EVENT COOL-DOWN ───────────────────────────────────────────

class TestNewsPostEventCoolDown(unittest.TestCase):
    """
    Unit tests for the asymmetric news_blackout_active() logic.

    Mocks _fetch_calendar() to inject synthetic events so we can control
    timing precisely without hitting the network.
    """

    def _run(self, pair: str, now_dt, events: list) -> tuple[bool, str]:
        import news
        with patch("news._fetch_calendar", return_value=events), \
             patch("news.datetime") as mock_dt:
            mock_dt.now.return_value = now_dt
            mock_dt.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
            return news.news_blackout_active(pair)

    def _make_event(self, currency: str, iso_time: str, impact: str = "High") -> dict:
        return {"currency": currency, "title": "Fed Rate Decision", "date": iso_time, "impact": impact}

    def _dt(self, iso: str):
        from datetime import datetime, timezone
        return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)

    def test_pre_event_window_blocks(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T13:45:00")   # 15 min before → within 30-min pre-window
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, reason = news.news_blackout_active("EUR_USD")
        self.assertTrue(blocked)
        self.assertIn("pre-event", reason)

    def test_outside_pre_window_passes(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T13:00:00")   # 60 min before → outside 30-min pre-window
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, _ = news.news_blackout_active("EUR_USD")
        self.assertFalse(blocked)

    def test_post_event_window_blocks(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T15:30:00")   # 90 min after → within 3-hour post-window
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, reason = news.news_blackout_active("EUR_USD")
        self.assertTrue(blocked)
        self.assertIn("post-event", reason)

    def test_post_event_reason_shows_elapsed_and_remaining(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T15:00:00")   # 60 min after
        import news, config
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, reason = news.news_blackout_active("EUR_USD")
        self.assertTrue(blocked)
        self.assertIn("60min ago", reason)
        # remaining = 3h - 1h = 2h = 120 min
        self.assertIn("120min remaining", reason)

    def test_after_post_event_window_passes(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T17:30:00")   # 3.5 hours after → outside 3-hour post-window
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, _ = news.news_blackout_active("EUR_USD")
        self.assertFalse(blocked)

    def test_non_high_impact_event_ignored(self):
        event = self._make_event("USD", "2024-01-15T14:00:00Z", impact="Medium")
        now   = self._dt("2024-01-15T14:30:00")   # 30 min after
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, _ = news.news_blackout_active("EUR_USD")
        self.assertFalse(blocked)

    def test_unrelated_currency_ignored(self):
        event = self._make_event("JPY", "2024-01-15T14:00:00Z")
        now   = self._dt("2024-01-15T14:30:00")
        import news
        with patch("news._fetch_calendar", return_value=[event]):
            with patch("news.datetime") as m:
                m.now.return_value = now
                m.fromisoformat.side_effect = lambda s: __import__("datetime").datetime.fromisoformat(s)
                blocked, _ = news.news_blackout_active("EUR_USD")
        self.assertFalse(blocked)


# ─── 17. REGIME CACHE VOLATILE TTL ───────────────────────────────────────────

class TestRegimeCacheBehaviour(unittest.TestCase):
    """
    Tests for the regime cache invalidation logic:
      - Volatile regime uses _VOLATILE_CACHE_TTL (60s) not _CACHE_TTL (300s)
      - ATR spike bypasses the cache entirely
    """

    def _make_df(self, atr_ratio: float = 1.0, n: int = 120):
        """Build a minimal DataFrame with ATR scaled to the requested ratio."""
        import numpy as np
        import pandas as pd
        base_atr = 0.001
        avg_atr  = base_atr
        cur_atr  = base_atr * atr_ratio
        atrs = np.full(n, avg_atr)
        atrs[-1] = cur_atr
        closes = np.full(n, 1.1)
        df = pd.DataFrame({
            "time":    pd.date_range("2024-01-01", periods=n, freq="h"),
            "open":    closes, "high": closes * 1.001,
            "low":     closes * 0.999, "close": closes,
            "volume":  np.full(n, 10_000.0),
            "atr":     atrs,
            "rsi":     np.full(n, 55.0),
            "ema_fast": np.full(n, 1.101),
            "ema_slow": np.full(n, 1.099),
            "adx":     np.full(n, 30.0),
            "adx_pos": np.full(n, 25.0),
            "adx_neg": np.full(n, 15.0),
        })
        return df

    def test_volatile_regime_uses_short_ttl(self):
        import regime
        # Plant a cached volatile result that's 90 seconds old
        regime._cache["TEST_V"] = (
            __import__("time").time() - 90,
            "volatile", "test"
        )
        # With _VOLATILE_CACHE_TTL=60, a 90-second-old volatile entry should be expired
        with patch("regime._client") as mock_client:
            mock_resp = mock_client.return_value.messages.create.return_value
            mock_resp.content = [type("C", (), {"text": '{"regime":"volatile","reason":"still hot"}'})()]
            df = self._make_df(atr_ratio=1.0)   # no spike, so spike-bypass won't fire
            regime.get_market_regime("TEST_V", df)
        # Claude was called because the 90s-old entry exceeded the 60s volatile TTL
        mock_client.return_value.messages.create.assert_called_once()
        regime._cache.pop("TEST_V", None)

    def test_normal_regime_uses_long_ttl(self):
        import regime
        # Plant a cached trending result that's 90 seconds old (within 300s TTL)
        regime._cache["TEST_N"] = (
            __import__("time").time() - 90,
            "trending_up", "test"
        )
        with patch("regime._client") as mock_client:
            df = self._make_df(atr_ratio=1.0)   # no ATR spike
            result_regime, _ = regime.get_market_regime("TEST_N", df)
        # Cache hit — Claude was NOT called
        mock_client.return_value.messages.create.assert_not_called()
        self.assertEqual(result_regime, "trending_up")
        regime._cache.pop("TEST_N", None)

    def test_atr_spike_bypasses_cache(self):
        import regime
        # Plant a fresh cached trending_up result (only 10 seconds old)
        regime._cache["TEST_S"] = (
            __import__("time").time() - 10,
            "trending_up", "strong momentum"
        )
        with patch("regime._client") as mock_client:
            mock_resp = mock_client.return_value.messages.create.return_value
            mock_resp.content = [type("C", (), {"text": '{"regime":"volatile","reason":"spike"}'})()]
            # ATR ratio > 1.8 → spike detected → cache bypassed
            df = self._make_df(atr_ratio=2.0)
            result_regime, _ = regime.get_market_regime("TEST_S", df)
        mock_client.return_value.messages.create.assert_called_once()
        self.assertEqual(result_regime, "volatile")
        regime._cache.pop("TEST_S", None)

    def test_no_spike_uses_cache(self):
        import regime
        regime._cache["TEST_NS"] = (
            __import__("time").time() - 10,
            "trending_down", "bearish"
        )
        with patch("regime._client") as mock_client:
            df = self._make_df(atr_ratio=1.2)   # below 1.8 threshold
            result_regime, _ = regime.get_market_regime("TEST_NS", df)
        mock_client.return_value.messages.create.assert_not_called()
        self.assertEqual(result_regime, "trending_down")
        regime._cache.pop("TEST_NS", None)


# ─── 18. MARGIN PRE-CHECK ─────────────────────────────────────────────────────

class TestMarginPreCheck(unittest.TestCase):
    """
    Tests for the margin pre-check guard in _evaluate_pair() and
    get_margin_available() in trader.py.

    The guard skips a new order if free margin (marginAvailable / NAV) falls
    below config.MARGIN_MIN_FREE_PCT (default 20%).  On API error, it logs a
    warning and proceeds so transient network issues don't block trading.
    """

    def setUp(self):
        import main
        self._orig_daily_trades = main._daily_trades

    def tearDown(self):
        import main
        main._daily_trades = self._orig_daily_trades

    def _idf(self):
        """Indicator-augmented DataFrame for use in _evaluate_pair() tests."""
        import numpy as np
        df = _make_ohlcv_df()
        n = len(df)
        df["atr"]      = np.full(n, 0.001)
        df["rsi"]      = np.full(n, 55.0)
        df["ema_fast"] = np.full(n, 1.101)
        df["ema_slow"] = np.full(n, 1.099)
        return df

    def _run_evaluate(self, margin_avail, nav, *, raises=False):
        """
        Run _evaluate_pair("EUR_USD") with all upstream filters mocked to pass
        and the margin check returning (margin_avail, nav).  Returns
        (mock_place_order, mock_filter_blocked).
        """
        import contextlib
        import main
        idf = self._idf()

        filter_patches = [
            patch("main.news_blackout_active",                  return_value=(False, "")),
            patch("main.in_session",                            return_value=True),
            patch("main.has_open_position",                     return_value=False),
            patch("main.get_candles",                           return_value=_make_ohlcv_df()),
            patch("main.add_indicators",                        return_value=idf),
            patch("main.get_signal",                            return_value="buy"),
            patch("main.upcoming_events",                       return_value=[]),
            patch("main.sentiment_mod.sentiment_blocks_signal", return_value=(False, "")),
            patch("main.sentiment_mod.get_news_sentiment",      return_value={"bias": "neutral"}),
            patch("main.regime_mod.regime_blocks_signal",       return_value=(False, "")),
            patch("main.regime_mod.get_market_regime",          return_value=("trending_up", "")),
            patch("main.all_filters_pass",                      return_value=(True, "")),
            patch("main.ml_filter_passes",                      return_value=(True, 0.75)),
            patch("main.correlation_ok",                        return_value=(True, "")),
            patch("main.get_price",  return_value={"ask": 1.1001, "bid": 1.0999, "mid": 1.1000}),
            patch("main.get_sl_tp",  return_value=(1.095, 1.115)),
            patch("main.dynamic_units", return_value=1000),
        ]
        margin_patch = (
            patch("main.get_margin_available", side_effect=Exception("API err"))
            if raises else
            patch("main.get_margin_available", return_value=(margin_avail, nav))
        )

        mock_place = mock_blocked = None
        with contextlib.ExitStack() as stack:
            for p in filter_patches:
                stack.enter_context(p)
            stack.enter_context(margin_patch)
            mock_place   = stack.enter_context(patch("main.place_order", return_value={}))
            mock_blocked = stack.enter_context(patch("main.bot_log.filter_blocked"))
            stack.enter_context(patch("main.bot_log.trade_open"))
            stack.enter_context(patch("main.alerts.trade_opened"))
            stack.enter_context(patch("main.alerts.send"))
            main._evaluate_pair("EUR_USD", [])

        return mock_place, mock_blocked

    # ── config checks ───────────────────────────────────────────────────────────

    def test_config_margin_min_free_pct_default(self):
        import config
        self.assertAlmostEqual(config.MARGIN_MIN_FREE_PCT, 20.0)

    def test_xau_usd_max_units_capped(self):
        import config
        self.assertLessEqual(config.INSTRUMENT_MAX_UNITS["XAU_USD"], 10)

    def test_wtico_usd_max_units_capped(self):
        import config
        self.assertLessEqual(config.INSTRUMENT_MAX_UNITS["WTICO_USD"], 100)

    # ── get_margin_available() ──────────────────────────────────────────────────

    def test_get_margin_available_returns_correct_values(self):
        import trader
        fake_acct = {"marginAvailable": "8000.00", "NAV": "10000.00"}
        with patch("trader.accounts_ep.AccountSummary") as mock_ep, \
             patch("trader._retry_request"):
            mock_ep.return_value.response = {"account": fake_acct}
            margin, nav = trader.get_margin_available()
        self.assertAlmostEqual(margin, 8000.0)
        self.assertAlmostEqual(nav, 10000.0)

    # ── _evaluate_pair() margin guard ───────────────────────────────────────────

    def test_margin_check_blocks_below_threshold(self):
        """15% free margin is below the 20% threshold — order must be skipped."""
        mock_place, mock_blocked = self._run_evaluate(1500.0, 10000.0)
        mock_place.assert_not_called()
        mock_blocked.assert_called_once()
        self.assertIn("margin", mock_blocked.call_args[0][1])

    def test_margin_check_passes_above_threshold(self):
        """80% free margin is well above the 20% threshold — order is placed."""
        mock_place, _ = self._run_evaluate(8000.0, 10000.0)
        mock_place.assert_called_once()

    def test_margin_exactly_at_threshold_passes(self):
        """20.0% equals the threshold — check is strict (<), so order is placed."""
        mock_place, _ = self._run_evaluate(2000.0, 10000.0)
        mock_place.assert_called_once()

    def test_margin_check_proceeds_on_api_error(self):
        """If get_margin_available() raises, order still proceeds (safe fallback)."""
        mock_place, _ = self._run_evaluate(0, 0, raises=True)
        mock_place.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
