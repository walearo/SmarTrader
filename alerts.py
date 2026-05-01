"""Send Telegram alerts."""

import requests
import config
import bot_control
import bot_log


def send(message: str) -> None:
    """Send a message to the configured Telegram chat."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[ALERT] {message}")
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    config.TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ALERT ERROR] {e}")


def trade_opened(pair: str, signal: str, price: float, sl: float, tp: float) -> None:
    if not bot_control.is_alert_enabled("trade_open"):
        return
    direction = "LONG" if signal == "buy" else "SHORT"
    send(
        f"*Trade Opened* \n"
        f"Pair: `{pair}`\n"
        f"Direction: `{direction}`\n"
        f"Entry: `{price}`\n"
        f"SL: `{sl}`\n"
        f"TP: `{tp}`"
    )


def price_alert(pair: str, price: float, move_pips: float) -> None:
    if not bot_control.is_alert_enabled("price_alert"):
        return
    send(
        f"*Price Alert* \n"
        f"Pair: `{pair}`\n"
        f"Current Price: `{price}`\n"
        f"Move: `{move_pips:.1f} pips`"
    )


def trade_closed(pair: str, result: str, pl_pips: float) -> None:
    if not bot_control.is_alert_enabled("trade_close"):
        return
    emoji  = "✅" if result == "win" else "❌"
    sign   = "+" if pl_pips >= 0 else ""
    send(
        f"{emoji} *Trade Closed*\n"
        f"Pair: `{pair}`\n"
        f"Result: `{result.upper()}`\n"
        f"P&L: `{sign}{pl_pips:.1f} pips`"
    )


def kill_switch_alert(reason: str) -> None:
    if not bot_control.is_alert_enabled("kill_switch"):
        return
    send(f"🚨 *Kill Switch Activated*\n{reason}\nBot paused for the rest of the day.")


def error_alert(context: str, err: Exception) -> None:
    bot_log.error(context, err)
    if not bot_control.is_alert_enabled("error_alert"):
        return
    send(f"*Bot Error* \n`{context}`: {err}")
