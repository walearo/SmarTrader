"""Place and manage orders on OANDA."""

import time as _time

import oandapyV20
import oandapyV20.endpoints.accounts as accounts_ep
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades_ep
import oandapyV20.endpoints.positions as positions
from oandapyV20.exceptions import V20Error
import requests.exceptions as _req_exc

import config


def _client():
    return oandapyV20.API(
        access_token=config.OANDA_API_KEY,
        environment=config.OANDA_ENVIRONMENT,
    )


def _retry_request(client, ep, max_attempts: int = 3) -> None:
    """Execute an OANDA request, retrying only on transient network errors.

    V20Error (API-level rejections like 400/404/401) are never retried since
    they indicate a logic or auth problem, not a transient failure.
    Order-mutating calls (place_order, partial_close_trade) must NOT use this
    to avoid double-fills on ambiguous timeouts.
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            client.request(ep)
            return
        except V20Error:
            raise
        except (_req_exc.ConnectionError, _req_exc.Timeout) as exc:
            if attempt == max_attempts - 1:
                raise
            _time.sleep(delay)
            delay *= 2


def place_order(pair: str, signal: str, sl: float, tp: float, units: int = None) -> dict:
    """Place a market order with SL and TP."""
    size  = units if units is not None else config.UNITS
    units = str(size) if signal == "buy" else str(-size)

    body = {
        "order": {
            "type":        "MARKET",
            "instrument":  pair,
            "units":       units,
            "stopLossOnFill":   {"price": str(sl)},
            "takeProfitOnFill": {"price": str(tp)},
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }

    client = _client()
    r = orders.OrderCreate(accountID=config.OANDA_ACCOUNT_ID, data=body)
    client.request(r)
    return r.response


def get_open_trades() -> list:
    """Return list of open trades."""
    client = _client()
    r = trades_ep.OpenTrades(accountID=config.OANDA_ACCOUNT_ID)
    _retry_request(client, r)
    return r.response.get("trades", [])


def has_open_position(pair: str, open_trades: list = None) -> bool:
    """Return True if there is already an open trade for this pair.

    Pass open_trades to reuse an already-fetched list and avoid an extra API call.
    """
    trades = open_trades if open_trades is not None else get_open_trades()
    return any(t["instrument"] == pair for t in trades)


def get_closed_trade(trade_id: str) -> dict | None:
    """Fetch full details of a specific closed trade by ID."""
    client = _client()
    try:
        r = trades_ep.TradeDetails(accountID=config.OANDA_ACCOUNT_ID, tradeID=trade_id)
        _retry_request(client, r)
        return r.response.get("trade")
    except Exception:
        return None


def close_position(pair: str) -> dict:
    """Close all units for a pair."""
    client = _client()
    body = {"longUnits": "ALL", "shortUnits": "ALL"}
    r = positions.PositionClose(
        accountID=config.OANDA_ACCOUNT_ID,
        instrument=pair,
        data=body,
    )
    client.request(r)
    return r.response


def get_account_equity() -> float:
    """Return current account NAV from OANDA."""
    client = _client()
    r = accounts_ep.AccountSummary(accountID=config.OANDA_ACCOUNT_ID)
    _retry_request(client, r)
    return float(r.response["account"]["NAV"])


def update_trade_sl(trade_id: str, new_sl: float) -> None:
    """Update the stop-loss price on an open trade."""
    client = _client()
    body = {"stopLoss": {"price": f"{new_sl:.5f}", "timeInForce": "GTC"}}
    r = trades_ep.TradeCRCDO(
        accountID=config.OANDA_ACCOUNT_ID,
        tradeID=trade_id,
        data=body,
    )
    client.request(r)


def partial_close_trade(trade_id: str, units: int) -> dict:
    """Partially close a trade by reducing it by `units` (always a positive integer)."""
    client = _client()
    body = {"units": str(units)}
    r = trades_ep.TradeClose(
        accountID=config.OANDA_ACCOUNT_ID,
        tradeID=trade_id,
        data=body,
    )
    client.request(r)
    return r.response
