"""Local, pseudonymous usage analytics for StorySprout."""

import json
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")
ANALYTICS_FILE = os.path.join(ANALYTICS_DIR, "events.json")
_LOCK = threading.RLock()
_ACTION_TYPES = {"story_created", "word_mastered", "coaching_query", "game_played"}
_RECENT_ACTIVITY_LIMIT = 50

os.makedirs(ANALYTICS_DIR, exist_ok=True)


def _clean_required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _clean_optional(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    return _clean_required(value, name)


def _load_events() -> List[Dict[str, Any]]:
    if not os.path.exists(ANALYTICS_FILE):
        return []
    with open(ANALYTICS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError(f"Invalid analytics data in {ANALYTICS_FILE!r}: expected an events list.")
    return data["events"]


def _save_events(events: List[Dict[str, Any]]) -> None:
    temporary_file = ANALYTICS_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump({"events": events}, file, indent=2, ensure_ascii=False)
    os.replace(temporary_file, ANALYTICS_FILE)


def _record(event: Dict[str, Any]) -> None:
    with _LOCK:
        events = _load_events()
        events.append(event)
        _save_events(events)


def record_page_view(
    page: str,
    session_id: str,
    language: Optional[str] = None,
    level: Optional[str] = None,
) -> None:
    """Record a page view, using the app session ID only to count unique visitors."""
    event = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "type": "page_view",
        "page": _clean_required(page, "page"),
        "session_id": _clean_required(session_id, "session_id"),
        "language": _clean_optional(language, "language"),
        "level": _clean_optional(level, "level"),
    }
    _record(event)


def record_action(action: str, language: Optional[str] = None) -> None:
    """Record a learner action supported by the creator dashboard."""
    action = _clean_required(action, "action")
    if action not in _ACTION_TYPES:
        raise ValueError(f"Unknown analytics action: {action!r}.")
    _record({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "type": action,
        "language": _clean_optional(language, "language"),
    })


def load_usage_metrics() -> Dict[str, Any]:
    """Return all-time usage metrics in the shape consumed by creator_app.py."""
    with _LOCK:
        events = _load_events()

    summary = {
        "total_unique_visitors": 0,
        "total_page_views": 0,
        "stories_created": 0,
        "words_mastered": 0,
        "coaching_queries": 0,
        "games_played": 0,
    }
    unique_visitors = set()
    language_popularity = Counter()
    level_breakdown = Counter()
    daily_visitors = defaultdict(set)
    daily_page_views = Counter()
    daily_stories = Counter()
    recent_activity = []

    action_labels = {
        "page_view": "Page view",
        "story_created": "Story created",
        "word_mastered": "Word mastered",
        "coaching_query": "Tutor query",
        "game_played": "Game played",
    }
    summary_keys = {
        "story_created": "stories_created",
        "word_mastered": "words_mastered",
        "coaching_query": "coaching_queries",
        "game_played": "games_played",
    }

    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("timestamp"), str):
            raise ValueError(f"Invalid event in {ANALYTICS_FILE!r}: expected an event with a timestamp.")
        event_type = event.get("type")
        timestamp = datetime.fromisoformat(event["timestamp"])
        day = timestamp.date().isoformat()
        language = event.get("language")
        level = event.get("level")

        if event_type == "page_view":
            summary["total_page_views"] += 1
            session_id = event.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"Invalid page-view event in {ANALYTICS_FILE!r}: missing session_id.")
            unique_visitors.add(session_id)
            daily_visitors[day].add(session_id)
            daily_page_views[day] += 1
            if language:
                language_popularity[language] += 1
            if level:
                level_breakdown[level] += 1
        elif event_type in summary_keys:
            summary[summary_keys[event_type]] += 1
            if event_type == "story_created":
                daily_stories[day] += 1
            if language:
                language_popularity[language] += 1
        else:
            raise ValueError(f"Unknown analytics event type in {ANALYTICS_FILE!r}: {event_type!r}.")

        recent_activity.append({
            "time": event["timestamp"],
            "activity": action_labels[event_type],
            "language": language or "",
            "level": level or "",
            "page": event.get("page", ""),
        })

    summary["total_unique_visitors"] = len(unique_visitors)
    today = datetime.now().date().isoformat()
    daily_traffic = {
        day: {
            "visitors": len(daily_visitors[day]),
            "page_views": daily_page_views[day],
            "stories": daily_stories[day],
        }
        for day in set(daily_visitors) | set(daily_page_views) | set(daily_stories) | {today}
    }

    return {
        "summary": summary,
        "daily_traffic": daily_traffic,
        "language_popularity": dict(language_popularity),
        "level_breakdown": dict(level_breakdown),
        "recent_activity_log": list(reversed(recent_activity[-_RECENT_ACTIVITY_LIMIT:])),
    }
