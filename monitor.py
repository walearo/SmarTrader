"""Monitor price movements and fire alerts."""

import config
import alerts
from data import get_prices

# Track last known prices to detect big moves
_last_prices: dict[str, float] = {}


def _pip_value(pair: str) -> float:
    return config.INSTRUMENT_PIP.get(pair, config.INSTRUMENT_PIP_DEFAULT)


def check_price_moves(price_map: dict = None) -> None:
    """Check each pair for significant price moves and alert if triggered.

    Accepts a pre-fetched price_map to avoid a redundant API call when the
    caller has already fetched prices this cycle. Falls back to a single
    batched fetch if not provided.
    """
    if price_map is None:
        try:
            price_map = get_prices(config.PAIRS)
        except Exception as e:
            alerts.error_alert("monitor/batch_price_fetch", e)
            return

    for pair in config.PAIRS:
        try:
            entry = price_map.get(pair)
            if entry is None:
                continue
            current = entry["mid"]
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
