"""
SQLite persistence layer — replaces bot_log.json, trade_journal.json, trade_history.json.

WAL mode allows the dashboard process to read concurrently while the bot writes.
A threading.Lock within each process serialises writes safely.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

log = logging.getLogger(__name__)

DB_PATH = "fx_bot.db"
_lock   = threading.Lock()


# ── Connection ─────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────────────

def init() -> None:
    """Create tables (idempotent). Called on first import and at bot/dashboard startup."""
    with _lock:
        conn = _connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bot_log (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT    NOT NULL,
                type TEXT    NOT NULL,
                data TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_journal (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                time     TEXT NOT NULL,
                trade    TEXT NOT NULL,
                analysis TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_history (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                time   TEXT NOT NULL,
                result TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    _migrate_json()


# ── One-time JSON migration ────────────────────────────────────────────────────

def _migrate_json() -> None:
    _migrate_bot_log()
    _migrate_journal()
    _migrate_history()


def _migrate_bot_log() -> None:
    path = "bot_log.json"
    if not os.path.exists(path):
        return
    with _lock:
        conn = _connect()
        try:
            if conn.execute("SELECT COUNT(*) FROM bot_log").fetchone()[0] > 0:
                return
            with open(path) as f:
                entries = json.load(f)
            for e in entries:
                conn.execute(
                    "INSERT INTO bot_log (time, type, data) VALUES (?,?,?)",
                    (e.get("time", ""), e.get("type", ""), json.dumps(e))
                )
            conn.commit()
            log.info(f"Migrated {len(entries)} bot_log entries to SQLite.")
        except Exception as e:
            log.warning(f"bot_log migration failed: {e}")
        finally:
            conn.close()


def _migrate_journal() -> None:
    path = "trade_journal.json"
    if not os.path.exists(path):
        return
    with _lock:
        conn = _connect()
        try:
            if conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0] > 0:
                return
            with open(path) as f:
                entries = json.load(f)
            for e in entries:
                conn.execute(
                    "INSERT INTO trade_journal (time, trade, analysis) VALUES (?,?,?)",
                    (e.get("time", ""), json.dumps(e.get("trade", {})), json.dumps(e.get("analysis", {})))
                )
            conn.commit()
            log.info(f"Migrated {len(entries)} journal entries to SQLite.")
        except Exception as e:
            log.warning(f"Journal migration failed: {e}")
        finally:
            conn.close()


def _migrate_history() -> None:
    path = "trade_history.json"
    if not os.path.exists(path):
        return
    with _lock:
        conn = _connect()
        try:
            if conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0] > 0:
                return
            with open(path) as f:
                results = json.load(f)
            now = datetime.now(timezone.utc).isoformat()
            for r in results:
                conn.execute(
                    "INSERT INTO trade_history (time, result) VALUES (?,?)",
                    (now, r)
                )
            conn.commit()
            log.info(f"Migrated {len(results)} trade history entries to SQLite.")
        except Exception as e:
            log.warning(f"Trade history migration failed: {e}")
        finally:
            conn.close()


# ── bot_log ────────────────────────────────────────────────────────────────────

def log_append(entry: dict) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO bot_log (time, type, data) VALUES (?,?,?)",
            (entry.get("time", ""), entry.get("type", ""), json.dumps(entry))
        )
        conn.commit()
        conn.close()


def prune_bot_log(max_rows: int = 10_000) -> None:
    """Delete oldest bot_log rows beyond max_rows. Called daily to cap table size."""
    with _lock:
        conn = _connect()
        conn.execute(
            "DELETE FROM bot_log WHERE id NOT IN "
            "(SELECT id FROM bot_log ORDER BY id DESC LIMIT ?)",
            (max_rows,)
        )
        conn.commit()
        conn.close()


def log_recent(count: int = 40) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT data FROM bot_log ORDER BY id DESC LIMIT ?", (count,)
    ).fetchall()
    conn.close()
    return [json.loads(r["data"]) for r in rows]


# ── trade_journal ──────────────────────────────────────────────────────────────

def journal_append(entry: dict) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO trade_journal (time, trade, analysis) VALUES (?,?,?)",
            (entry.get("time", ""), json.dumps(entry.get("trade", {})), json.dumps(entry.get("analysis", {})))
        )
        conn.commit()
        conn.close()


def journal_all() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT time, trade, analysis FROM trade_journal ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {
            "time":     r["time"],
            "trade":    json.loads(r["trade"]),
            "analysis": json.loads(r["analysis"]),
        }
        for r in rows
    ]


# ── trade_history ──────────────────────────────────────────────────────────────

def history_append(result: str) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO trade_history (time, result) VALUES (?,?)",
            (datetime.now(timezone.utc).isoformat(), result)
        )
        conn.commit()
        conn.close()


def history_recent(count: int = 20) -> list[str]:
    conn = _connect()
    rows = conn.execute(
        "SELECT result FROM trade_history ORDER BY id DESC LIMIT ?", (count,)
    ).fetchall()
    conn.close()
    return [r["result"] for r in reversed(rows)]


# ── state ──────────────────────────────────────────────────────────────────────

def state_set(key: str, value) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?,?)",
            (key, json.dumps(value))
        )
        conn.commit()
        conn.close()


def state_get(key: str, default=None):
    conn = _connect()
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    conn.close()
    return json.loads(row["value"]) if row else default


# Initialise on first import so any module that imports db gets a ready database.
init()
