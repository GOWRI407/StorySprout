"""
StorySprout - Language-Aware Curriculum
Single source of truth for what a learner sees in the selected language.

RULE: content is ALWAYS in the language the learner selected. If a language/level has no
vocabulary yet, we say so (or generate it with Gemini) - we never silently fall back to Spanish.

Vocabulary sources, highest priority first:
  1. Creator overrides   (data/content/custom_vocab.json, edited in creator_app.py)
  2. Built-in starter    (languages.STARTER_VOCABULARY)
  3. Extra A1 starters   (EXTRA_VOCAB below)
  4. Gemini-generated    (cached in data/content/ai_vocab_cache.json)
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

from importlib import import_module

_languages = import_module(f"{__package__}.languages" if __package__ else "languages")
STARTER_VOCABULARY = _languages.STARTER_VOCABULARY
GLOBAL_LANGUAGES = _languages.GLOBAL_LANGUAGES

try:
    import_module("dotenv").load_dotenv()
except Exception:
    pass

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
LEVEL_NAMES = {"A1": "Beginner", "A2": "Elementary", "B1": "Intermediate", "B2": "Upper-Intermediate", "C1": "Advanced"}

ROUND_SIZE = 5          # questions per game round
PASS_MARK = 3           # correct answers needed to unlock the story stage
MIN_WORDS_USED = 2      # target words the learner must use in their story
MIN_STORY_UNITS = {"A1": 12, "A2": 20, "B1": 30, "B2": 45, "C1": 60}  # words (or CJK chars / 2)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT_DIR = os.path.join(BASE_DIR, "data", "content")
CUSTOM_FILE = os.path.join(CONTENT_DIR, "custom_vocab.json")
CACHE_FILE = os.path.join(CONTENT_DIR, "ai_vocab_cache.json")
STORIES_FILE = os.path.join(CONTENT_DIR, "creator_stories.json")
os.makedirs(CONTENT_DIR, exist_ok=True)

EXTRA_VOCAB: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "Italian": {"A1": [
        {"word": "ciao", "translation": "hello / bye", "pos": "interjection", "phonetic": "CHOW"},
        {"word": "grazie", "translation": "thank you", "pos": "interjection", "phonetic": "GRAH-tsyeh"},
        {"word": "amico", "translation": "friend", "pos": "noun", "phonetic": "ah-MEE-koh"},
        {"word": "casa", "translation": "house / home", "pos": "noun", "phonetic": "KAH-zah"},
        {"word": "acqua", "translation": "water", "pos": "noun", "phonetic": "AHK-kwah"},
        {"word": "pane", "translation": "bread", "pos": "noun", "phonetic": "PAH-neh"}]},
    "Portuguese": {"A1": [
        {"word": "olá", "translation": "hello", "pos": "interjection", "phonetic": "oh-LAH"},
        {"word": "obrigado", "translation": "thank you", "pos": "interjection", "phonetic": "oh-bree-GAH-doo"},
        {"word": "amigo", "translation": "friend", "pos": "noun", "phonetic": "ah-MEE-goo"},
        {"word": "casa", "translation": "house / home", "pos": "noun", "phonetic": "KAH-zah"},
        {"word": "água", "translation": "water", "pos": "noun", "phonetic": "AH-gwah"},
        {"word": "pão", "translation": "bread", "pos": "noun", "phonetic": "pown"}]},
    "Korean": {"A1": [
        {"word": "안녕하세요", "translation": "hello", "pos": "greeting", "phonetic": "an-nyeong-ha-se-yo"},
        {"word": "감사합니다", "translation": "thank you", "pos": "phrase", "phonetic": "gam-sa-ham-ni-da"},
        {"word": "친구", "translation": "friend", "pos": "noun", "phonetic": "chin-goo"},
        {"word": "물", "translation": "water", "pos": "noun", "phonetic": "mool"},
        {"word": "집", "translation": "house / home", "pos": "noun", "phonetic": "jip"}]},
    "Russian": {"A1": [
        {"word": "привет", "translation": "hello", "pos": "interjection", "phonetic": "pree-VYET"},
        {"word": "спасибо", "translation": "thank you", "pos": "interjection", "phonetic": "spa-SEE-ba"},
        {"word": "друг", "translation": "friend", "pos": "noun", "phonetic": "droog"},
        {"word": "вода", "translation": "water", "pos": "noun", "phonetic": "va-DAH"},
        {"word": "дом", "translation": "house / home", "pos": "noun", "phonetic": "dom"}]},
}

# Wrong-answer padding (English meanings). Filtered so they never overlap the right answer.
GENERIC_MEANINGS = ["water", "house", "sun", "book", "journey", "peaceful", "knowledge",
                    "freedom", "tree", "music", "market", "river", "letter", "mountain"]


# ----------------------------------------------------------------------------- storage
def _read(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_custom_vocab() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    return _read(CUSTOM_FILE, {})


def save_custom_vocab_level(language: str, level: str, items: List[Dict[str, str]]) -> None:
    data = load_custom_vocab()
    if items:
        data.setdefault(language, {})[level] = items
    else:
        data.get(language, {}).pop(level, None)
        if not data.get(language):
            data.pop(language, None)
    _write(CUSTOM_FILE, data)


def clean_items(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Keep only complete vocab rows with the expected keys."""
    out = []
    for r in rows:
        word = str(r.get("word") or "").strip()
        trans = str(r.get("translation") or "").strip()
        if word and trans:
            out.append({
                "word": word,
                "translation": trans,
                "pos": str(r.get("pos") or "word").strip(),
                "phonetic": str(r.get("phonetic") or "").strip(),
            })
    return out


# ----------------------------------------------------------------------------- vocabulary
def ai_available() -> bool:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    return bool(key) and key != "your_gemini_api_key_here"


def generate_vocab_with_ai(language: str, level: str, n: int = 8) -> List[Dict[str, str]]:
    """Ask Gemini for level-appropriate vocabulary. Returns [] if unavailable or on error."""
    if not ai_available():
        return []
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = (
            f"List {n} common {language} vocabulary items suitable for CEFR {level} learners. "
            "Return ONLY a JSON array. Each element must have the keys: "
            '"word" (in the native script of the language), "translation" (English), '
            '"pos" (part of speech), "phonetic" (simple English-style respelling). '
            "Every item must be genuinely in " + language + "."
        )
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        return clean_items(json.loads(resp.text))
    except Exception as e:
        print(f"[Curriculum] AI vocab generation failed: {e}")
        return []


def get_level_vocab(language: str, level: str, generate: bool = False) -> Tuple[List[Dict[str, str]], str]:
    """Vocabulary for exactly this language + level. Returns (items, source)."""
    custom = load_custom_vocab().get(language, {}).get(level)
    if custom:
        return custom, "creator"
    starter = STARTER_VOCABULARY.get(language, {}).get(level)
    if starter:
        return starter, "built-in"
    extra = EXTRA_VOCAB.get(language, {}).get(level)
    if extra:
        return extra, "built-in"
    cache = _read(CACHE_FILE, {})
    cached = cache.get(language, {}).get(level)
    if cached:
        return cached, "ai"
    if generate:
        items = generate_vocab_with_ai(language, level)
        if items:
            cache.setdefault(language, {})[level] = items
            _write(CACHE_FILE, cache)
            return items, "ai"
    return [], "none"


def all_vocab_for_language(language: str, generate: bool = False) -> Dict[str, List[Dict[str, str]]]:
    out = {}
    for lvl in LEVELS:
        items, _ = get_level_vocab(language, lvl, generate=generate)
        if items:
            out[lvl] = items
    return out


def get_round_pool(language: str, level: str, generate: bool = True) -> Tuple[List[Dict[str, str]], str]:
    """Words for a game round in this language. Tops up with other levels of the SAME language."""
    items, source = get_level_vocab(language, level, generate=generate)
    pool = list(items)
    note = ""
    if len(pool) < ROUND_SIZE:
        others = sorted((l for l in LEVELS if l != level), key=lambda l: abs(LEVELS.index(l) - LEVELS.index(level)))
        for lvl in others:
            extra, _ = get_level_vocab(language, lvl, generate=False)
            for it in extra:
                if it not in pool:
                    pool.append(it)
            if len(pool) >= ROUND_SIZE:
                break
        if pool and len(items) < len(pool):
            note = f"Only {len(items)} {language} words exist for {level} so far - mixing in nearby {language} levels."
    if not pool:
        note = (f"No {language} vocabulary for {level} yet. Ask a creator to add it, "
                f"or set GEMINI_API_KEY so it can be generated automatically.")
    return pool, note


# ----------------------------------------------------------------------------- game round
def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


def build_round(language: str, level: str, seed: Any = None, n: int = ROUND_SIZE) -> Tuple[List[Dict[str, Any]], str]:
    """Build n multiple-choice questions, all in the selected language."""
    pool, note = get_round_pool(language, level)
    if not pool:
        return [], note

    rng = random.Random(seed)
    order = pool[:]
    rng.shuffle(order)

    every = [it for items in all_vocab_for_language(language).values() for it in items]
    foreign = sorted({it["word"] for it in every})
    meanings = sorted({it["translation"] for it in every})

    questions = []
    for k in range(n):
        t = order[k % len(order)]
        other_words = [w for w in foreign if w != t["word"]]
        kind = "word" if (k % 2 == 1 and len(other_words) >= 3) else "meaning"

        if kind == "meaning":
            bank = [m for m in meanings + GENERIC_MEANINGS if not (_tokens(m) & _tokens(t["translation"]))]
            correct = t["translation"]
            prompt = "What does this word mean?"
        else:
            bank = other_words
            correct = t["word"]
            prompt = f"Which {language} word means \u201c{t['translation']}\u201d?"

        bank = sorted(set(bank))
        options = rng.sample(bank, min(3, len(bank))) + [correct]
        rng.shuffle(options)
        questions.append({
            "kind": kind,
            "prompt": prompt,
            "word": t["word"],
            "translation": t["translation"],
            "phonetic": t.get("phonetic", ""),
            "pos": t.get("pos", "word"),
            "options": options,
            "answer": correct,
        })
    return questions, note


# ----------------------------------------------------------------------------- story check
def word_forms(entry_word: str) -> List[str]:
    """'こんにちは (Konnichiwa)' -> ['こんにちは', 'konnichiwa']; 'casa' -> ['casa']."""
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", entry_word.strip())
    forms = [m.group(1), m.group(2)] if m else [entry_word]
    return [f.strip().lower() for f in forms if f.strip()]


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def measure_length(text: str) -> int:
    """Words for spaced scripts; characters / 2 for Chinese, Japanese and Korean text."""
    tokens = text.split()
    cjk = len(_CJK.findall(text))
    return cjk // 2 if cjk > len(tokens) else len(tokens)


def check_story(text: str, level: str, vocab: List[Dict[str, str]]) -> Dict[str, Any]:
    lowered = text.lower()
    used = [v["word"] for v in vocab if any(f in lowered for f in word_forms(v["word"]))]
    needed_len = MIN_STORY_UNITS.get(level, 20)
    needed_words = min(MIN_WORDS_USED, len(vocab))
    length = measure_length(text)
    return {
        "length": length,
        "needed_length": needed_len,
        "used_words": used,
        "needed_words": needed_words,
        "length_ok": length >= needed_len,
        "words_ok": len(used) >= needed_words,
        "passed": length >= needed_len and len(used) >= needed_words,
    }


# ----------------------------------------------------------------------------- creator stories
def load_creator_stories() -> List[Dict[str, Any]]:
    return _read(STORIES_FILE, [])


def save_creator_story(entry: Dict[str, Any]) -> None:
    stories = load_creator_stories()
    stories.append(entry)
    _write(STORIES_FILE, stories)


def delete_creator_story(index: int) -> None:
    stories = load_creator_stories()
    if 0 <= index < len(stories):
        stories.pop(index)
        _write(STORIES_FILE, stories)


def stories_for(language: str, level: str) -> List[Dict[str, Any]]:
    return [s for s in load_creator_stories() if s.get("language") == language and s.get("level") == level]


# ----------------------------------------------------------------------------- language helpers
def language_choices() -> Dict[str, str]:
    """{'🇪🇸 Spanish (Español)': 'Spanish', ...} - exact mapping, no string parsing."""
    return {f"{i['flag']} {name} ({i['native']})": name for name, i in GLOBAL_LANGUAGES.items()}


def content_status(language: str) -> Dict[str, str]:
    """Which levels have vocabulary and where it comes from (for the creator dashboard)."""
    return {lvl: get_level_vocab(language, lvl)[1] for lvl in LEVELS}
