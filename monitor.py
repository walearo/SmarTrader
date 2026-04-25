"""Monitor price movements and fire alerts."""

import config
import alerts
from data import get_price

# Track last known prices to detect big moves
_last_prices: dict[str, float] = {}


def _pip_value(pair: str) -> float:
    """Return the pip size for a pair."""
    return 0.0001 if "JPY" not in pair else 0.01


def check_price_moves() -> None:
    """Check each pair for significant price moves and alert if triggered."""
    for pair in config.PAIRS:
        try:
            current = get_price(pair)["mid"]
            pip = _pip_value(pair)

            if pair in _last_prices:
                move_pips = abs(current - _last_prices[pair]) / pip
                if move_pips >= config.ALERT_PRICE_MOVE_PIPS:
                    alerts.price_alert(pair, current, move_pips)
                    _last_prices[pair] = current  # reset baseline
            else:
                _last_prices[pair] = current

        except Exception as e:
            alerts.error_alert(f"monitor/{pair}", e)
