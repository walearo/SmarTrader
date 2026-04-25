"""Fetch candle + price data from OANDA."""

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing

import config


def _client() -> oandapyV20.API:
    env = "practice" if config.OANDA_ENVIRONMENT == "practice" else "live"
    return oandapyV20.API(access_token=config.OANDA_API_KEY, environment=env)


def get_candles(pair: str, count: int = 200, timeframe: str = None) -> pd.DataFrame:
    """Return a DataFrame with OHLC + volume for the last `count` candles."""
    client = _client()
    params = {"count": count, "granularity": timeframe or config.TIMEFRAME, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    rows = []
    for candle in r.response["candles"]:
        if not candle["complete"]:
            continue
        mid = candle["mid"]
        rows.append({
            "time":   pd.to_datetime(candle["time"]),
            "open":   float(mid["o"]),
            "high":   float(mid["h"]),
            "low":    float(mid["l"]),
            "close":  float(mid["c"]),
            "volume": int(candle["volume"]),
        })

    df = pd.DataFrame(rows).set_index("time")
    return df


def get_price(pair: str) -> dict:
    """Return current bid/ask for a pair."""
    client = _client()
    params = {"instruments": pair}
    r = pricing.PricingInfo(accountID=config.OANDA_ACCOUNT_ID, params=params)
    client.request(r)
    price = r.response["prices"][0]
    return {
        "bid": float(price["bids"][0]["price"]),
        "ask": float(price["asks"][0]["price"]),
        "mid": (float(price["bids"][0]["price"]) + float(price["asks"][0]["price"])) / 2,
    }
