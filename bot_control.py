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
    _write({"paused": True, "reason": reason, "at": _now()})


def resume() -> None:
    _write({"paused": False, "reason": None, "at": _now()})


def status() -> dict:
    return _read()
