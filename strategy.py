"""
Donchian channel breakout + volume surge + RSI momentum + ADX trend filter.

Signal rules:
  BUY  — close breaks above 20-bar Donchian high by ≥ 0.3 ATR
          AND RSI > 55  (clear upward momentum, not just midpoint)
          AND volume > 1.5x 20-bar average  (institutional conviction)
          AND ADX > 25  (confirmed trending market)
          AND candle closes in top 40% of its bar range  (no pinbar wicks)
  SELL — mirror of the above downward

Donchian bands use .shift(1) so bar-i close is compared to the channel
built from bars [i-DONCHIAN_PERIOD … i-1] — no lookahead.
ADX confirms the move is a genuine trend breakout, not a range-market spike.
Candle close quality filters out breakouts where price reversed intra-bar.
"""

import pandas as pd
import ta

import config


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "dc_high" in df.columns:
        return df   # already computed — skip recomputation
    df = df.copy()

    # ── trend / momentum ──────────────────────────────────────────────────────
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=config.EMA_FAST).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=config.EMA_SLOW).ema_indicator()
    df["rsi"]      = ta.momentum.RSIIndicator(df["close"], window=config.RSI_PERIOD).rsi()

    # ── trend strength (ADX) ──────────────────────────────────────────────────
    adx_ind        = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=config.ADX_PERIOD)
    df["adx"]      = adx_ind.adx()
    df["adx_pos"]  = adx_ind.adx_pos()   # +DI
    df["adx_neg"]  = adx_ind.adx_neg()   # -DI

    # ── volatility ────────────────────────────────────────────────────────────
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=config.ATR_PERIOD
    ).average_true_range()

    # ── Donchian channel (shift(1) prevents lookahead) ────────────────────────
    df["dc_high"] = df["high"].rolling(config.DONCHIAN_PERIOD).max().shift(1)
    df["dc_low"]  = df["low"].rolling(config.DONCHIAN_PERIOD).min().shift(1)

    # ── volume baseline ───────────────────────────────────────────────────────
    df["vol_ma"] = df["volume"].rolling(20).mean()

    return df


def get_signal(df: pd.DataFrame) -> str:
    """
    Returns 'buy', 'sell', or 'flat'.

    BUY  — confirmed Donchian breakout upward with volume + ADX trend filter
    SELL — confirmed Donchian breakout downward with volume + ADX trend filter
    """
    df   = add_indicators(df)
    curr = df.iloc[-1]

    if pd.isna(curr["dc_high"]) or pd.isna(curr["vol_ma"]) or pd.isna(curr["adx"]):
        return "flat"

    vol_surge = curr["volume"] > config.VOLUME_SURGE_FACTOR * curr["vol_ma"]
    adx_ok    = curr["adx"] > config.ADX_MIN_TREND
    buf       = config.BREAKOUT_ATR_BUFFER * curr["atr"]

    candle_range = curr["high"] - curr["low"]
    close_pct    = (curr["close"] - curr["low"]) / candle_range if candle_range > 0 else 0.5

    if (curr["close"] > curr["dc_high"] + buf
            and curr["rsi"] > config.RSI_BUY_MIN
            and vol_surge
            and adx_ok
            and curr["adx_pos"] > curr["adx_neg"]
            and close_pct >= config.BREAKOUT_CANDLE_CLOSE_PCT):   # close in top 40% of bar
        return "buy"

    if (curr["close"] < curr["dc_low"] - buf
            and curr["rsi"] < config.RSI_SELL_MAX
            and vol_surge
            and adx_ok
            and curr["adx_neg"] > curr["adx_pos"]
            and close_pct <= (1.0 - config.BREAKOUT_CANDLE_CLOSE_PCT)):   # close in bottom 40% of bar
        return "sell"

    return "flat"


def get_sl_tp(df: pd.DataFrame, signal: str, entry_price: float = None) -> tuple[float, float]:
    """Calculate stop loss and take profit based on ATR.

    entry_price: actual fill price (ask for buy, bid for sell).
                 Falls back to last close if not provided.
    """
    df    = add_indicators(df)
    atr   = df["atr"].iloc[-1]
    price = entry_price if entry_price is not None else df["close"].iloc[-1]

    sl_distance = atr * config.ATR_SL_MULTIPLIER
    tp_distance = atr * config.ATR_TP_MULTIPLIER

    if signal == "buy":
        sl = round(price - sl_distance, 5)
        tp = round(price + tp_distance, 5)
    else:  # sell
        sl = round(price + sl_distance, 5)
        tp = round(price - tp_distance, 5)

    return sl, tp
