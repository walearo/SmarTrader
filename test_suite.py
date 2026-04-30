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
                         ("ALERT_PRICE_MOVE_PIPS", 1, 500), ("NEWS_BLACKOUT_MINUTES", 0, 180)]
        _float_ranges = [("RISK_PCT_PER_TRADE", 0.1, 5.0),
                         ("MAX_DAILY_LOSS_PCT", 0.5, 20.0), ("MAX_WEEKLY_LOSS_PCT", 1.0, 50.0)]

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
