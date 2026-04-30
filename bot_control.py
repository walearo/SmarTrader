"""
Shared bot control file — dashboard writes, bot reads.
Uses bot_control.json as the communication channel between processes.

Cross-process safety: filelock.FileLock provides OS-level file locking that
works correctly across separate Python processes (threading.Lock does not).
"""

import json
import os
from datetime import datetime, timezone

try:
    from filelock import FileLock
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

CONTROL_PATH = "bot_control.json"
_LOCK_PATH   = "bot_control.json.lock"

ALERT_TYPES = ["price_alert", "trade_open", "trade_close", "kill_switch", "error_alert"]


def _lock():
    """Return an OS-level file lock context manager, or a no-op if filelock is unavailable."""
    if _FILELOCK_AVAILABLE:
        return FileLock(_LOCK_PATH, timeout=5)
    # graceful fallback — no cross-process protection but won't crash
    import contextlib
    return contextlib.nullcontext()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read() -> dict:
    try:
        with _lock():
            if not os.path.exists(CONTROL_PATH):
                return {"paused": False, "reason": None, "at": None}
            with open(CONTROL_PATH) as f:
                return json.load(f)
    except Exception:
        return {"paused": False, "reason": None, "at": None}


def _write(data: dict) -> None:
    with _lock():
        with open(CONTROL_PATH, "w") as f:
            json.dump(data, f, indent=2)


def is_paused() -> bool:
    return _read().get("paused", False)


def pause(reason: str = "dashboard") -> None:
    data = _read()
    data.update({"paused": True, "reason": reason, "at": _now()})
    _write(data)


def resume() -> None:
    data = _read()
    data.update({"paused": False, "reason": None, "at": _now()})
    _write(data)


def status() -> dict:
    return _read()


def get_alert_settings() -> dict:
    alerts = _read().get("alerts", {})
    return {k: bool(alerts.get(k, True)) for k in ALERT_TYPES}


def is_alert_enabled(alert_type: str) -> bool:
    return _read().get("alerts", {}).get(alert_type, True)


def set_alert(alert_type: str, enabled: bool) -> None:
    if alert_type not in ALERT_TYPES:
        return
    data = _read()
    alerts = data.get("alerts", {})
    alerts[alert_type] = enabled
    data["alerts"] = alerts
    _write(data)


# ─── BOT SETTINGS ─────────────────────────────────────────────────────────────

SETTINGS_PATH       = "bot_settings.json"
_SETTINGS_LOCK_PATH = "bot_settings.json.lock"

SETTINGS_KEYS = [
    "MAX_CONCURRENT_TRADES",
    "UNITS",
    "RISK_PCT_PER_TRADE",
    "MAX_DAILY_LOSS_PCT",
    "MAX_WEEKLY_LOSS_PCT",
    "ALERT_PRICE_MOVE_PIPS",
    "NEWS_BLACKOUT_MINUTES",
    "PAIRS",
    "SESSIONS",
]

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CHF", "USD_CAD", "NZD_USD", "XAU_USD", "WTICO_USD",
]


def _settings_lock():
    if _FILELOCK_AVAILABLE:
        return FileLock(_SETTINGS_LOCK_PATH, timeout=5)
    import contextlib
    return contextlib.nullcontext()


def _read_settings() -> dict:
    try:
        with _settings_lock():
            if not os.path.exists(SETTINGS_PATH):
                return {}
            with open(SETTINGS_PATH) as f:
                return json.load(f)
    except Exception:
        return {}


def _write_settings(data: dict) -> None:
    with _settings_lock():
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)


def get_bot_settings() -> dict:
    import config as _cfg
    defaults = {
        "MAX_CONCURRENT_TRADES": _cfg.MAX_CONCURRENT_TRADES,
        "UNITS":                 _cfg.UNITS,
        "RISK_PCT_PER_TRADE":    _cfg.RISK_PCT_PER_TRADE,
        "MAX_DAILY_LOSS_PCT":    _cfg.MAX_DAILY_LOSS_PCT,
        "MAX_WEEKLY_LOSS_PCT":   _cfg.MAX_WEEKLY_LOSS_PCT,
        "ALERT_PRICE_MOVE_PIPS": _cfg.ALERT_PRICE_MOVE_PIPS,
        "NEWS_BLACKOUT_MINUTES": _cfg.NEWS_BLACKOUT_MINUTES,
        "PAIRS":                 list(_cfg.PAIRS),
        "SESSIONS":              [list(s) for s in _cfg.SESSIONS],
    }
    saved = _read_settings()
    result = dict(defaults)
    for k in SETTINGS_KEYS:
        if k in saved:
            result[k] = saved[k]
    return result


def save_bot_settings(updates: dict) -> None:
    data = _read_settings()
    for k in SETTINGS_KEYS:
        if k in updates:
            data[k] = updates[k]
    _write_settings(data)


def reset_bot_settings() -> None:
    if os.path.exists(SETTINGS_PATH):
        os.remove(SETTINGS_PATH)


def apply_bot_settings() -> None:
    """Apply saved settings overrides to the config module (called each cycle)."""
    saved = _read_settings()
    if not saved:
        return
    import config as _cfg
    for key in SETTINGS_KEYS:
        if key not in saved:
            continue
        value = saved[key]
        if key == "SESSIONS":
            value = [tuple(s) for s in value]
        setattr(_cfg, key, value)
