"""
StorySprout - Learner Progress (per user, per language)

Each language has its own journey, so switching language never mixes progress:

    A1 Game -> A1 Story -> A2 Game -> A2 Story -> B1 Game -> ... -> C1 Story -> Complete
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .curriculum import LEVELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_DIR = os.path.join(BASE_DIR, "data", "users")
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "progress.json")
os.makedirs(PROGRESS_DIR, exist_ok=True)


def _load() -> Dict[str, Any]:
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: Dict[str, Any]) -> None:
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)


def _profile(data: Dict[str, Any], username: str) -> Dict[str, Any]:
    return data.setdefault(username, {"xp": 0, "languages": {}, "stories": [], "last_language": None})


def journey_for(start_level: str = "A1") -> List[Tuple[str, str]]:
    """Ordered stages from the starting level up to C1: [("game","A1"), ("story","A1"), ...]."""
    start = LEVELS.index(start_level) if start_level in LEVELS else 0
    stages: List[Tuple[str, str]] = []
    for lvl in LEVELS[start:]:
        stages.append(("game", lvl))
        stages.append(("story", lvl))
    return stages


def get_profile(username: str) -> Dict[str, Any]:
    return _profile(_load(), username)


def get_language_state(username: str, language: str):
    """Return the saved state for a language, or None if the learner never started it."""
    return _profile(_load(), username)["languages"].get(language)


def start_language(username: str, language: str, start_level: str = "A1", restart: bool = False) -> Dict[str, Any]:
    """Select a language. Existing progress is kept unless restart=True."""
    data = _load()
    prof = _profile(data, username)
    if restart or language not in prof["languages"]:
        prof["languages"][language] = {
            "start_level": start_level,
            "stage": 0,
            "words_mastered": [],
            "started": datetime.now().isoformat(timespec="seconds"),
        }
    prof["last_language"] = language
    _save(data)
    return prof["languages"][language]


def current_stage(state: Dict[str, Any]):
    """Return (kind, level) for the learner's current stage, or ("done", "") when finished."""
    stages = journey_for(state["start_level"])
    if state["stage"] >= len(stages):
        return "done", ""
    return stages[state["stage"]]


def advance(username: str, language: str) -> None:
    data = _load()
    prof = _profile(data, username)
    if language in prof["languages"]:
        prof["languages"][language]["stage"] += 1
        _save(data)


def add_xp(username: str, amount: int) -> int:
    data = _load()
    prof = _profile(data, username)
    prof["xp"] = prof.get("xp", 0) + int(amount)
    _save(data)
    return prof["xp"]


def add_words(username: str, language: str, words: List[str]) -> None:
    data = _load()
    prof = _profile(data, username)
    state = prof["languages"].get(language)
    if state is not None:
        known = set(state.get("words_mastered", []))
        known.update(words)
        state["words_mastered"] = sorted(known)
        _save(data)


def add_story(username: str, language: str, level: str, text: str) -> None:
    data = _load()
    prof = _profile(data, username)
    prof["stories"].append({
        "language": language,
        "level": level,
        "text": text,
        "saved": datetime.now().isoformat(timespec="seconds"),
    })
    _save(data)


def all_profiles() -> Dict[str, Any]:
    return _load()
