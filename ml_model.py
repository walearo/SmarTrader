"""
ML signal filter — Voting Ensemble.
Loads a trained model and returns the win probability for a given signal.
Only signals with probability >= ML_MIN_CONFIDENCE pass through.

Feature design philosophy:
  Features answer one question: "is this a QUALITY breakout or a false breakout?"
  - ADX measures trend strength — strong trends validate breakouts
  - Breakout strength (how far price moved through the channel) separates conviction moves
  - Volume confirms institutional participation
  - Donchian position gives context (near top/bottom of range = risky vs midpoint)
  - RSI momentum confirms direction
  Noisy features removed: raw return lags, hour/dow (forex patterns aren't stable by time-of-day)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

import config
from strategy import add_indicators

log = logging.getLogger(__name__)

_model = None


_MODEL_STALE_DAYS = 30   # warn if model file is older than this


def check_model_staleness() -> None:
    """Log a warning if the model file is older than _MODEL_STALE_DAYS days.
    Called once daily from _daily_reset(). Retrain with: python train_ml.py --days 730
    """
    if not os.path.exists(config.ML_MODEL_PATH):
        return
    import time as _time
    age_days = (_time.time() - os.path.getmtime(config.ML_MODEL_PATH)) / 86400
    if age_days >= _MODEL_STALE_DAYS:
        log.warning(
            f"ML model is {age_days:.0f} days old (limit: {_MODEL_STALE_DAYS}). "
            "Retrain with: python train_ml.py --days 730"
        )


def _load_model():
    global _model
    if _model is None:
        if os.path.exists(config.ML_MODEL_PATH):
            with open(config.ML_MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            log.info("ML model loaded.")
        else:
            log.warning("No ML model found — skipping ML filter.")
    return _model


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from a candle DataFrame.

    Features are grouped by what they measure:
      trend strength       → adx, adx_slope, adx_pos, adx_neg, di_diff, ema_diff, ema_diff_lag1
      momentum             → rsi, rsi_slope
      breakout quality     → breakout_strength, breakout_close_pct, prior_false_breaks,
                             prior_consolidation_bars, dc_position, dc_dist_high, dc_dist_low,
                             dc_width_atr, range_tightness
      weekly context       → weekly_range_pct
      volatility           → atr_pct, atr_ratio
      volume               → vol_ratio, vol_ratio_lag1
      candle structure     → body, upper_wick, lower_wick

    Key additions vs previous version:
      range_tightness      — std(closes in prior Donchian window) / ATR. Tight consolidation
                             before a break is the classic compression-expansion setup.
      prior_false_breaks   — how many times price touched the channel boundary but reversed
                             in the last 2×DONCHIAN_PERIOD bars. More = level well-defended.
      breakout_close_pct   — where the breakout candle closes in its own range (0=low, 1=high).
                             Close near high for buy breakout = conviction; near middle = indecision.
      weekly_range_pct     — price position within the last 5 trading days (120 H1 bars).
                             Breakout at 95% of weekly range = extending; at 50% = new leg.
    """
    df   = add_indicators(df).copy()
    atr  = df["atr"].replace(0, np.nan)
    feat = pd.DataFrame(index=df.index)

    # ── trend strength ────────────────────────────────────────────────────────
    feat["adx"]           = df["adx"] / 100.0
    feat["adx_slope"]     = df["adx"].diff(3) / 3 / 100.0   # rising=positive, falling=negative
    feat["adx_pos"]       = df["adx_pos"] / 100.0
    feat["adx_neg"]       = df["adx_neg"] / 100.0
    feat["di_diff"]       = (df["adx_pos"] - df["adx_neg"]) / 100.0

    feat["ema_diff"]      = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    feat["ema_diff_lag1"] = feat["ema_diff"].shift(1)

    # ── momentum ──────────────────────────────────────────────────────────────
    feat["rsi"]           = df["rsi"] / 100.0
    feat["rsi_slope"]     = df["rsi"].diff(3) / 3

    # ── breakout quality ──────────────────────────────────────────────────────
    dc_range = (df["dc_high"] - df["dc_low"]).replace(0, np.nan)

    feat["breakout_strength"] = (
        (df["close"] - df["dc_high"]).clip(lower=0) / atr
        - (df["dc_low"] - df["close"]).clip(lower=0) / atr
    )

    # where the breakout candle closes within its own high-low range (0=low end, 1=high end)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    feat["breakout_close_pct"] = (df["close"] - df["low"]) / candle_range

    # how many times price touched the Donchian boundary and reversed (false break)
    # within the last 2×DONCHIAN_PERIOD bars — well-defended levels break less reliably
    above_channel = (df["close"] > df["dc_high"]).astype(int)
    below_channel = (df["close"] < df["dc_low"]).astype(int)
    any_break     = (above_channel | below_channel).astype(int)
    # a false break: price touched channel but returned inside within 2 bars
    any_break_shifted = any_break.shift(1).fillna(0) + any_break.shift(2).fillna(0)
    false_break = ((any_break_shifted > 0) & (any_break == 0)).astype(int)
    feat["prior_false_breaks"] = (
        false_break.rolling(2 * config.DONCHIAN_PERIOD).sum()
        .clip(upper=10) / 10.0
    )

    # bars since last breakout in either direction (consolidation duration)
    group = any_break.cumsum()
    feat["prior_consolidation_bars"] = (
        group.groupby(group).cumcount()
        .clip(upper=100) / 100.0
    )

    # range tightness: std of closes during the prior Donchian window / ATR
    # tight consolidation (low std/ATR) before a breakout = compression-expansion setup
    feat["range_tightness"] = (
        df["close"].rolling(config.DONCHIAN_PERIOD).std().shift(1) / (atr + 1e-9)
    ).clip(upper=5) / 5.0   # normalise; cap outliers

    feat["dc_position"]   = (df["close"] - df["dc_low"]) / dc_range
    feat["dc_dist_high"]  = (df["dc_high"] - df["close"]) / atr
    feat["dc_dist_low"]   = (df["close"]   - df["dc_low"]) / atr
    feat["dc_width_atr"]  = dc_range / atr

    # weekly range context: price position within the last 5 trading days (≈120 H1 bars)
    weekly_high = df["high"].rolling(120).max()
    weekly_low  = df["low"].rolling(120).min()
    weekly_range = (weekly_high - weekly_low).replace(0, np.nan)
    feat["weekly_range_pct"] = (df["close"] - weekly_low) / weekly_range

    # ── volatility ────────────────────────────────────────────────────────────
    feat["atr_pct"]       = atr / df["close"]
    feat["atr_ratio"]     = atr / atr.rolling(20).mean()

    # ── volume ────────────────────────────────────────────────────────────────
    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    feat["vol_ratio"]     = df["volume"] / vol_ma
    feat["vol_ratio_lag1"] = feat["vol_ratio"].shift(1)

    # ── candle structure ──────────────────────────────────────────────────────
    feat["body"]          = abs(df["close"] - df["open"]) / (atr + 1e-9)
    feat["upper_wick"]    = (df["high"] - df[["open", "close"]].max(axis=1)) / (atr + 1e-9)
    feat["lower_wick"]    = (df[["open", "close"]].min(axis=1) - df["low"])   / (atr + 1e-9)

    return feat.dropna()


def predict_win_probability(df: pd.DataFrame, signal: str) -> float:
    """
    Return predicted probability (0–1) that the trade will be a winner.
    Returns 1.0 (bypass) if no model is loaded.
    """
    model = _load_model()
    if model is None:
        return 1.0

    features = build_features(df)
    if features.empty:
        return 1.0

    row = features.iloc[[-1]].copy()
    row["signal"] = 1 if signal == "buy" else 0

    try:
        prob = model.predict_proba(row)[0][1]
        return float(prob)
    except Exception as e:
        log.error(f"ML prediction error: {e}")
        return 1.0


def ml_filter_passes(df: pd.DataFrame, signal: str) -> tuple[bool, float]:
    """Return (passes: bool, confidence: float)."""
    prob = predict_win_probability(df, signal)
    return prob >= config.ML_MIN_CONFIDENCE, prob
