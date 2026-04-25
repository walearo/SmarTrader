"""
Claude-powered market regime detector.

Classifies the current market as one of:
  trending_up   — clear upward momentum, favour BUY signals
  trending_down — clear downward momentum, favour SELL signals
  ranging       — sideways, EMAs compressed, skip new signals
  volatile      — abnormally large moves, skip until stable

Result is cached per pair for 5 minutes so we don't hit the API
on every 60-second bot cycle.
"""

import json
import logging
import re
import time

import anthropic
import pandas as pd

import config


def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from a model response.
    Handles markdown code fences, leading/trailing prose, and partial wrapping.
    Raises json.JSONDecodeError if no valid object is found.
    """
    # 1. strip markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?", "", text).strip()
    # 2. find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group())

log = logging.getLogger(__name__)

_CACHE_TTL = 300   # 5 minutes

# ─── CLIENT ───────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    key = config.ANTHROPIC_API_KEY or None
    return anthropic.Anthropic(api_key=key)


# ─── CACHE ────────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, str, str]] = {}   # pair -> (ts, regime, reason)


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a quantitative forex analyst specialising in technical market structure.
Classify the current market regime from indicator data alone.
Always respond with ONLY valid JSON — no explanation outside the JSON object."""


# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────

def get_market_regime(pair: str, df: pd.DataFrame) -> tuple[str, str]:
    """
    Classify the current market regime for `pair` using Claude.

    Returns (regime, reason) where regime is one of:
      'trending_up', 'trending_down', 'ranging', 'volatile'

    Falls back to 'ranging' on any error (safe default — avoids bad entries).
    """
    now = time.time()
    if pair in _cache:
        ts, regime, reason = _cache[pair]
        if now - ts < _CACHE_TTL:
            log.debug(f"{pair}: regime cache hit → {regime}")
            return regime, reason

    try:
        last       = df.iloc[-1]
        atr_avg    = float(df["atr"].tail(100).mean())
        atr_now    = float(last["atr"])
        close_list = [round(float(x), 5) for x in df["close"].tail(100).tolist()]

        context = {
            "pair":              pair,
            "closes_last_100":   close_list,
            "rsi":               round(float(last["rsi"]), 1),
            "ema_fast":          round(float(last["ema_fast"]), 5),
            "ema_slow":          round(float(last["ema_slow"]), 5),
            "ema_diff_pct":      round((float(last["ema_fast"]) - float(last["ema_slow"]))
                                       / float(last["close"]) * 100, 4),
            "atr_now":           round(atr_now, 5),
            "atr_100_avg":       round(atr_avg, 5),
            "atr_ratio":         round(atr_now / atr_avg, 2) if atr_avg else 1.0,
        }

        prompt = f"""\
Classify the current market regime for {pair} using this data:

{json.dumps(context, indent=2)}

The closes_last_100 field gives you 100 hourly bars of context — use it to
assess the broader trend structure (higher highs/lows, lower highs/lows,
compression) before deciding on regime.

Choose exactly ONE regime:
- trending_up:   EMA fast clearly above slow, RSI > 50, price making higher highs over the 100-bar window
- trending_down: EMA fast clearly below slow, RSI < 50, price making lower lows over the 100-bar window
- ranging:       EMAs close together (ema_diff_pct near 0), price oscillating within a range
- volatile:      atr_ratio > 1.8 — abnormally large candles, erratic price swings

Respond ONLY with this JSON (no markdown, no extra text):
{{"regime": "trending_up", "reason": "one sentence"}}"""

        client = _client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=[{
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text)
        regime = result.get("regime", "ranging")
        reason = result.get("reason", "")

        if regime not in ("trending_up", "trending_down", "ranging", "volatile"):
            regime = "ranging"

        _cache[pair] = (now, regime, reason)
        log.info(f"{pair} regime → {regime} | {reason}")
        return regime, reason

    except Exception as e:
        log.warning(f"{pair}: regime API error — {e}")
        return "ranging", "API error"


def regime_blocks_signal(pair: str, signal: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Return (blocked, reason).

    Blocks if:
      - regime is 'ranging'  (no directional edge)
      - regime is 'volatile' (too dangerous)
      - regime direction contradicts signal
        e.g. regime=trending_down but signal=buy → block
    """
    regime, reason = get_market_regime(pair, df)

    if regime == "ranging":
        return True, f"Claude regime: ranging market — {reason}"
    if regime == "volatile":
        return True, f"Claude regime: volatile market — {reason}"
    if regime == "trending_up" and signal == "sell":
        return True, f"Claude regime: uptrend conflicts with SELL — {reason}"
    if regime == "trending_down" and signal == "buy":
        return True, f"Claude regime: downtrend conflicts with BUY — {reason}"

    return False, ""
