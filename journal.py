"""
Claude-powered trade journal.

After every trade closes, Claude analyses what happened and extracts
structured lessons: why it won/lost, what was done well, one improvement,
and a quality score (1–10) independent of the outcome.

Entries are appended to trade_journal.json so you build up a searchable
record of every trade over time.
"""

import json
import logging
import re
from datetime import datetime, timezone

import anthropic

import config
import db

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group())


# ─── CLIENT ───────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    key = config.ANTHROPIC_API_KEY or None
    return anthropic.Anthropic(api_key=key)


# ─── PERSISTENCE ──────────────────────────────────────────────────────────────

def _append(entry: dict) -> None:
    try:
        db.journal_append(entry)
    except Exception as e:
        log.warning(f"Could not save journal entry: {e}")


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert forex trading coach who reviews completed trades objectively.
Your feedback is concise, specific, and actionable.
Always respond with ONLY valid JSON — no explanation outside the JSON object."""


# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────

def analyse_trade(trade: dict) -> None:
    """
    Ask Claude to analyse a completed trade and save the result to trade_journal.json.

    Expected keys in `trade`:
        instrument      e.g. "EUR_USD"
        direction       "BUY" or "SELL"
        entry           float
        exit_price      float
        pl_pips         float
        rsi_at_entry    float  (optional)
        atr_ratio       float  (optional, current ATR / 20-period avg)
        ml_confidence   float  (optional, 0–1)
        sentiment_bias  str    (optional, "bullish"/"bearish"/"neutral")
        regime          str    (optional, "trending_up" etc.)
        news_events     list   (optional, list of event title strings)
    """
    result_word = "WON" if trade.get("pl_pips", 0) > 0 else "LOST"
    pips        = trade.get("pl_pips", 0)

    news_str = ", ".join(trade.get("news_events") or []) or "none"

    prompt = f"""\
Review this completed forex trade and provide coaching feedback.

Trade:
  Pair:           {trade.get('instrument')}
  Direction:      {trade.get('direction')}
  Entry:          {trade.get('entry')}
  Exit:           {trade.get('exit_price')}
  P&L:            {pips:+.1f} pips  ({result_word})
  RSI at entry:   {trade.get('rsi_at_entry', 'N/A')}
  ATR ratio:      {trade.get('atr_ratio', 'N/A')}  (1.0 = average volatility)
  ML confidence:  {trade.get('ml_confidence', 'N/A')}
  Sentiment bias: {trade.get('sentiment_bias', 'N/A')}
  Market regime:  {trade.get('regime', 'N/A')}
  Nearby news:    {news_str}

Provide:
1. why        — primary reason this trade {result_word.lower()} (be specific, reference the data)
2. done_well  — one thing executed correctly
3. improve    — one concrete improvement for next time
4. score      — trade quality 1–10 judged on process, NOT outcome

Respond ONLY with this JSON (no markdown, no extra text):
{{"why": "...", "done_well": "...", "improve": "...", "score": 7}}"""

    try:
        client = _client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=[{
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = _extract_json(response.content[0].text)

        entry = {
            "time":     datetime.now(timezone.utc).isoformat(),
            "trade":    trade,
            "analysis": analysis,
        }
        _append(entry)

        log.info(
            f"Journal [{trade.get('instrument')} {result_word} {pips:+.1f}p] "
            f"score={analysis.get('score')}/10 — {analysis.get('why')}"
        )
    except Exception as e:
        log.warning(f"Journal API error: {e}")
