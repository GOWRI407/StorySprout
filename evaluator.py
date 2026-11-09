"""
StorySprout - Story Difficulty Evaluator

Provides estimate_story_difficulty(text, level) which returns a dict with:
  - level_fit: human-readable label such as "Good fit for A1"
  - metrics: {"word_count": int, "avg_word_length": float, "long_word_ratio": float,
               "sentence_count": int, "avg_sentence_length": float}
"""

from __future__ import annotations

import re
import unicodedata

# CEFR levels in ascending order
LEVELS = ["A1", "A2", "B1", "B2", "C1"]

# Rough expected word-count ranges per level (minimum story length already
# enforced by curriculum; we use these to gauge difficulty relative to level).
_LEVEL_INDEX = {lvl: i for i, lvl in enumerate(LEVELS)}

# Average word-length thresholds that mark a text as "harder than level X"
# These are heuristics; they don't need to be linguistically perfect.
_AVG_LEN_HARD_THRESHOLD = {
    "A1": 5.0,
    "A2": 5.5,
    "B1": 6.0,
    "B2": 6.5,
    "C1": 7.5,
}

# Long-word ratio thresholds (fraction of words with >= 7 characters)
_LONG_RATIO_HARD_THRESHOLD = {
    "A1": 0.10,
    "A2": 0.15,
    "B1": 0.22,
    "B2": 0.30,
    "C1": 0.40,
}


def _is_cjk(ch: str) -> bool:
    """Return True for CJK ideographs (each counts as one 'word unit')."""
    try:
        name = unicodedata.name(ch, "")
    except (TypeError, ValueError):
        return False
    return name.startswith("CJK")


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens, handling CJK characters individually."""
    tokens: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            tokens.append(ch)
    # Also grab regular words (letters/apostrophes)
    tokens += re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\u00C0-\u024F\u0400-\u04FF\u0370-\u03FF\u0600-\u06FF]+", text)
    return tokens


def _sentence_count(text: str) -> int:
    """Rough sentence count based on sentence-ending punctuation."""
    sentences = re.split(r"[.!?。！？]+", text.strip())
    return max(1, len([s for s in sentences if s.strip()]))


def _compute_metrics(text: str) -> dict:
    words = _tokenize(text)
    word_count = len(words)
    if word_count == 0:
        return {
            "word_count": 0,
            "avg_word_length": 0.0,
            "long_word_ratio": 0.0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
        }

    avg_word_length = sum(len(w) for w in words) / word_count
    long_word_ratio = sum(1 for w in words if len(w) >= 7) / word_count
    sentence_count = _sentence_count(text)
    avg_sentence_length = word_count / sentence_count

    return {
        "word_count": word_count,
        "avg_word_length": round(avg_word_length, 2),
        "long_word_ratio": round(long_word_ratio, 3),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
    }


def _infer_level(metrics: dict) -> str:
    """Map metrics to the closest CEFR level label."""
    avg_len = metrics["avg_word_length"]
    long_ratio = metrics["long_word_ratio"]

    # Walk from easiest level upward; stop when we're below the 'hard' threshold.
    inferred = "A1"
    for lvl in LEVELS:
        if avg_len >= _AVG_LEN_HARD_THRESHOLD[lvl] or long_ratio >= _LONG_RATIO_HARD_THRESHOLD[lvl]:
            inferred = lvl
        else:
            break
    return inferred


def _fit_label(inferred: str, target: str) -> str:
    """Return a human-readable fit description."""
    i_idx = _LEVEL_INDEX.get(inferred, 0)
    t_idx = _LEVEL_INDEX.get(target, 0)
    diff = i_idx - t_idx
    if diff == 0:
        return f"Good fit for {target}"
    elif diff == -1:
        return f"Slightly easy for {target} (reads like {inferred})"
    elif diff <= -2:
        return f"Too easy for {target} (reads like {inferred})"
    elif diff == 1:
        return f"Slightly challenging for {target} (reads like {inferred})"
    else:
        return f"Too advanced for {target} (reads like {inferred})"


def estimate_story_difficulty(text: str, level: str) -> dict:
    """
    Estimate the CEFR difficulty of *text* and compare it to *level*.

    Returns:
        {
            "level_fit": str,          # e.g. "Good fit for A1"
            "inferred_level": str,     # e.g. "A1"
            "metrics": {
                "word_count": int,
                "avg_word_length": float,
                "long_word_ratio": float,
                "sentence_count": int,
                "avg_sentence_length": float,
            }
        }
    """
    metrics = _compute_metrics(text)
    if metrics["word_count"] == 0:
        return {
            "level_fit": "No text to evaluate",
            "inferred_level": level,
            "metrics": metrics,
        }

    inferred = _infer_level(metrics)
    level_fit = _fit_label(inferred, level)

    return {
        "level_fit": level_fit,
        "inferred_level": inferred,
        "metrics": metrics,
    }

