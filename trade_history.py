"""Persistent rolling trade history used by risk_manager for dynamic sizing."""

import logging
import db

log = logging.getLogger(__name__)

_KEEP = 20


def record_trade_result(result: str) -> None:
    """Append 'win' or 'loss' to persistent storage."""
    try:
        db.history_append(result)
    except Exception as e:
        log.warning(f"Could not save trade history: {e}")


def get_history() -> list[str]:
    """Return the last 20 trade results (oldest first)."""
    try:
        return db.history_recent(_KEEP)
    except Exception:
        return []


def consecutive_losses() -> int:
    """Return the current streak of consecutive losses (0 if last result was a win or no history)."""
    history = get_history()
    count = 0
    for result in reversed(history):
        if result == "loss":
            count += 1
        else:
            break
    return count
