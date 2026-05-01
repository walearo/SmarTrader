"""Structured event logger — writes to SQLite so the dashboard can read it."""

from datetime import datetime, timezone
import db


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def signal(pair: str, signal: str, confidence: float = None) -> None:
    db.log_append({
        "time": _now(), "type": "signal",
        "pair": pair, "signal": signal,
        "confidence": f"{confidence:.1%}" if confidence is not None else "n/a",
    })


def trade_open(pair: str, direction: str, entry: float, sl: float, tp: float, units: int) -> None:
    db.log_append({
        "time": _now(), "type": "trade_open",
        "pair": pair, "direction": direction,
        "entry": entry, "sl": sl, "tp": tp, "units": units,
    })


def trade_close(pair: str, result: str, pnl_pips: float) -> None:
    db.log_append({
        "time": _now(), "type": "trade_close",
        "pair": pair, "result": result, "pnl_pips": round(pnl_pips, 1),
    })


def filter_blocked(pair: str, reason: str) -> None:
    db.log_append({"time": _now(), "type": "filter", "pair": pair, "reason": reason})


def kill_switch(reason: str) -> None:
    db.log_append({"time": _now(), "type": "kill_switch", "reason": reason})


def info(message: str) -> None:
    db.log_append({"time": _now(), "type": "info", "message": message})


def error(context: str, err: Exception) -> None:
    db.log_append({"time": _now(), "type": "error", "context": context, "message": str(err)})
