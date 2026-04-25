"""
Claude-powered news sentiment filter.

Replaces the binary 30-minute blackout with a directional assessment.
Instead of blocking both buy AND sell around news, the bot only skips
trades that conflict with Claude's directional bias.

Cache: results are stored per event set so we don't re-query the same
       news events on every 60-second cycle.
"""

import json
import logging
import re
import time

import anthropic

import config


def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from a model response.
    Handles markdown code fences, leading/trailing prose, and partial wrapping.
    Raises json.JSONDecodeError if no valid object is found.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group())

log = logging.getLogger(__name__)

_CACHE_TTL = 6 * 3600   # 6 hours — same news events are reassessed after this

# ─── CLIENT ───────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    key = config.ANTHROPIC_API_KEY or None
    return anthropic.Anthropic(api_key=key)


# ─── CACHE ────────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, dict]] = {}   # cache_key -> (timestamp, result)


def _cache_key(pair: str, events: list) -> str:
    event_ids = "|".join(
        f"{e.get('currency')}:{e.get('title')}:{e.get('time')}"
        for e in events
    )
    return f"{pair}:{event_ids}"


# ─── SYSTEM PROMPT (cached by Anthropic between calls) ────────────────────────

_SYSTEM = """\
You are a professional forex market analyst specialising in fundamental event analysis.
Your job is to assess the directional impact of upcoming high-impact economic events on currency pairs.
Always respond with ONLY valid JSON — no explanation outside the JSON object."""


# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────

def get_news_sentiment(pair: str, events: list) -> dict:
    """
    Ask Claude whether upcoming high-impact news is bullish, bearish, or neutral
    for the given pair, and whether to avoid trading at all.

    Returns:
        {
            "bias":       "bullish" | "bearish" | "neutral",
            "confidence": "low" | "medium" | "high",
            "avoid":      bool,
            "reason":     str
        }

    If no events or API unavailable, returns a safe neutral default.
    """
    if not events:
        return {"bias": "neutral", "confidence": "low", "avoid": False, "reason": "no events"}

    key = _cache_key(pair, events)
    if key in _cache:
        ts, cached_result = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            log.debug(f"{pair}: sentiment cache hit")
            return cached_result
        # TTL expired — fall through to re-query

    base, quote = pair.split("_")
    events_text = "\n".join(
        f"  - [{e.get('currency')}] {e.get('title')} at {e.get('time')}"
        for e in events
    )

    prompt = f"""\
Assess the likely directional impact on {base}/{quote} from these upcoming events:

{events_text}

Rules:
- "bullish" means {base} is expected to strengthen vs {quote}
- "bearish" means {base} is expected to weaken vs {quote}
- "neutral" means the impact is unclear or mixed
- Set "avoid": true if the event is likely to cause a sharp, unpredictable move in either direction

Respond ONLY with this JSON (no markdown, no extra text):
{{"bias": "bullish", "confidence": "medium", "avoid": false, "reason": "one sentence"}}"""

    try:
        client = _client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=[{
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},   # cache system prompt
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text)
        # normalise fields
        result.setdefault("bias", "neutral")
        result.setdefault("confidence", "low")
        result.setdefault("avoid", False)
        result.setdefault("reason", "")
        _cache[key] = (time.time(), result)
        log.info(
            f"{pair} sentiment → {result['bias']} ({result['confidence']}) | "
            f"avoid={result['avoid']} | {result['reason']}"
        )
        return result
    except Exception as e:
        log.warning(f"{pair}: sentiment API error — {e}")
        return {"bias": "neutral", "confidence": "low", "avoid": False, "reason": "API error"}


def sentiment_blocks_signal(pair: str, signal: str, events: list) -> tuple[bool, str]:
    """
    Return (blocked, reason).

    Blocks if:
      - avoid=True  (too unpredictable)
      - bias contradicts signal with medium/high confidence
        e.g. signal=buy but sentiment=bearish/high → block
    """
    if not events:
        return False, ""

    s = get_news_sentiment(pair, events)

    if s["avoid"]:
        return True, f"Claude: avoid trading — {s['reason']}"

    if s["confidence"] in ("medium", "high"):
        if signal == "buy" and s["bias"] == "bearish":
            return True, f"Claude: bearish sentiment ({s['confidence']}) conflicts with BUY — {s['reason']}"
        if signal == "sell" and s["bias"] == "bullish":
            return True, f"Claude: bullish sentiment ({s['confidence']}) conflicts with SELL — {s['reason']}"

    return False, ""
