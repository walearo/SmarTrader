"""
Level 3 — News Blackout
Fetches the weekly economic calendar from ForexFactory's public JSON feed
and blocks trading around high-impact events.
"""

from datetime import datetime, timedelta, timezone
import requests
import logging

import config

log = logging.getLogger(__name__)

_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_cache: list = []
_cache_fetched_at: datetime = None


def _fetch_calendar() -> list:
    """Fetch (and cache) this week's economic events."""
    global _cache, _cache_fetched_at

    now = datetime.now(timezone.utc)
    # refresh cache every 6 hours
    if _cache_fetched_at and (now - _cache_fetched_at).total_seconds() < 21600:
        return _cache

    try:
        resp = requests.get(_CALENDAR_URL, timeout=10)
        resp.raise_for_status()
        _cache = resp.json()
        _cache_fetched_at = now
        log.info(f"News calendar refreshed — {len(_cache)} events loaded.")
    except Exception as e:
        log.warning(f"Could not fetch news calendar: {e}")

    return _cache


def _parse_event_time(event: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    except Exception:
        return None


def news_blackout_active(pair: str) -> tuple[bool, str]:
    """
    Return (True, reason) if trading should be blocked around a high-impact event.

    Two-phase block:
      Pre-event:  NEWS_BLACKOUT_MINUTES before the event fires.
      Post-event: NEWS_POST_EVENT_HOURS after the event fires (market settling period).

    The post-event window is intentionally longer — central bank decisions and
    NFP prints often cause volatility that takes hours to resolve into a clean trend.
    """
    currencies = set(pair.split("_"))
    watched    = currencies & set(config.NEWS_CURRENCIES)
    if not watched:
        return False, ""

    events      = _fetch_calendar()
    now         = datetime.now(timezone.utc)
    pre_window  = timedelta(minutes=config.NEWS_BLACKOUT_MINUTES)
    post_window = timedelta(hours=config.NEWS_POST_EVENT_HOURS)

    for event in events:
        if event.get("impact", "").lower() != "high":
            continue
        if event.get("currency") not in watched:
            continue

        event_time = _parse_event_time(event)
        if event_time is None:
            continue

        title = event.get("title", "Unknown event")
        t_str = event_time.strftime("%H:%M UTC")

        if event_time - pre_window <= now < event_time:
            return True, f"{event['currency']} pre-event blackout: '{title}' at {t_str}"

        if event_time <= now < event_time + post_window:
            elapsed   = int((now - event_time).total_seconds() / 60)
            remaining = int((event_time + post_window - now).total_seconds() / 60)
            return True, (
                f"{event['currency']} post-event cool-down: '{title}' was {elapsed}min ago "
                f"({remaining}min remaining)"
            )

    return False, ""


def upcoming_events(hours: int = 4) -> list[dict]:
    """Return high-impact events in the next `hours` hours (for alerts)."""
    events  = _fetch_calendar()
    now     = datetime.now(timezone.utc)
    cutoff  = now + timedelta(hours=hours)
    results = []

    for event in events:
        if event.get("impact", "").lower() != "high":
            continue
        event_time = _parse_event_time(event)
        if event_time and now <= event_time <= cutoff:
            results.append({
                "currency": event.get("currency") or "",
                "title":    event.get("title") or "",
                "time":     event_time.strftime("%H:%M UTC"),
            })

    return results
