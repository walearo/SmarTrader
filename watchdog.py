"""
Process watchdog for main.py.

Starts main.py as a subprocess and restarts it if it crashes.
Uses exponential backoff (up to 5 min) to avoid restart storms.
After MAX_RESTARTS consecutive rapid crashes it gives up and alerts.
"""

import subprocess
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
import alerts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PYTHON        = sys.executable
BOT_SCRIPT    = Path(__file__).parent / "main.py"
MAX_RESTARTS  = 10          # give up after this many consecutive rapid crashes
RAPID_CRASH_S = 60          # a crash within this many seconds counts as "rapid"
BASE_DELAY_S  = 5           # initial restart delay
MAX_DELAY_S   = 300         # cap backoff at 5 minutes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run() -> None:
    consecutive_rapid = 0
    delay = BASE_DELAY_S

    log.info(f"Watchdog started — monitoring {BOT_SCRIPT.name}")
    alerts.send(f"*Watchdog started* — monitoring `{BOT_SCRIPT.name}`")

    while True:
        start = time.monotonic()
        log.info(f"Launching {BOT_SCRIPT.name} ...")

        proc = subprocess.Popen(
            [PYTHON, str(BOT_SCRIPT)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        exit_code = proc.wait()
        elapsed   = time.monotonic() - start

        if exit_code == 0:
            log.info("Bot exited cleanly (code 0). Watchdog stopping.")
            alerts.send("*Bot exited cleanly.* Watchdog stopped.")
            break

        log.warning(f"Bot crashed (exit code {exit_code}) after {elapsed:.0f}s")

        if elapsed < RAPID_CRASH_S:
            consecutive_rapid += 1
        else:
            consecutive_rapid = 0
            delay = BASE_DELAY_S   # reset backoff after a stable run

        if consecutive_rapid >= MAX_RESTARTS:
            msg = (
                f"*Watchdog giving up* — {MAX_RESTARTS} rapid crashes in a row.\n"
                f"Last exit code: `{exit_code}` at {_now()}\n"
                "Please check the logs and restart manually."
            )
            log.error(msg)
            alerts.send(msg)
            break

        alerts.send(
            f"*Bot crashed* (exit `{exit_code}`) at {_now()}\n"
            f"Restarting in {delay}s "
            f"(attempt {consecutive_rapid}/{MAX_RESTARTS})"
        )
        log.info(f"Restarting in {delay}s ...")
        time.sleep(delay)
        delay = min(delay * 2, MAX_DELAY_S)


if __name__ == "__main__":
    run()
