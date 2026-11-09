"""
StorySprout Capstone - Interactive Vocabulary Arcade & Level Games
Provides gamified language practice:
1. Vocab Flashcard Quiz with audio pronunciation and instant scoring
2. Speed Word-Meaning Matcher
3. CEFR Level Convert Game (transform sentences from A1 to B1/B2)
4. XP, Streak, and Mastery Badges with automated Word Bank synchronization
"""

import random
from typing import List, Dict, Any, Optional
from .languages import STARTER_VOCABULARY, GLOBAL_LANGUAGES


def get_game_vocabulary(language: str, level: str) -> List[Dict[str, str]]:
    """Vocabulary for the requested language (creator overrides + built-in + AI cache).
    Never falls back to another language; returns [] when the language has no words yet."""
    from .curriculum import get_round_pool
    pool, _ = get_round_pool(language, level, generate=False)
    return pool


def generate_vocab_quiz_question(language: str, level: str, current_question_index: int = 0) -> Dict[str, Any]:
    """Generate a 4-option multiple-choice vocabulary quiz question."""
    vocab_pool = get_game_vocabulary(language, level)
    if not vocab_pool:
        return {"word": "", "pos": "", "phonetic": "", "correct_answer": "", "options": [],
                "correct_index": -1, "level": level, "language": language,
                "error": f"No {language} vocabulary for {level} yet."}

    # Select target item
    item_idx = current_question_index % len(vocab_pool)
    target = vocab_pool[item_idx]

    # Create distractors
    all_translations = [v["translation"] for v in vocab_pool if v["translation"] != target["translation"]]
    distractors_pool = all_translations + ["water", "house", "sun", "book", "journey", "peaceful", "knowledge", "freedom"]
    random.seed(42 + current_question_index)
    distractors = random.sample([d for d in distractors_pool if d != target["translation"]], min(3, len(distractors_pool)))

    options = distractors + [target["translation"]]
    random.shuffle(options)

    return {
        "word": target["word"],
        "pos": target.get("pos", "word"),
        "phonetic": target.get("phonetic", ""),
        "correct_answer": target["translation"],
        "options": options,
        "correct_index": options.index(target["translation"]),
        "level": level,
        "language": language
    }


# Pre-authored CEFR Level Conversion Challenge Exercises
LEVEL_CONVERT_EXERCISES = [
    {
        "id": 1,
        "language": "Spanish",
        "base_level": "A1",
        "target_level": "B1",
        "base_sentence": "Yo voy al parque todos los días.",
        "instruction": "Upgrade this sentence to CEFR B1 by using a habitual imperfect phrase and a narrative connector.",
        "options": [
            "Ayer fui al parque una sola vez.",
            "Solía pasear por el parque con frecuencia, ya que disfrutaba de la tranquilidad.",
            "Yo caminar parque hoy.",
            "Voy a ir al parque mañana por la tarde."
        ],
        "correct_index": 1,
        "explanation": "Option 2 uses the nuanced habitual imperfect verb 'Solía pasear' and causal connector 'ya que', meeting B1 criteria."
    },
    {
        "id": 2,
        "language": "French",
        "base_level": "A1",
        "target_level": "B1",
        "base_sentence": "Il fait froid dehors aujourd'hui.",
        "instruction": "Convert this elementary statement into B1 expressing personal impression and condition.",
        "options": [
            "Il fait chaud dehors.",
            "Bien qu'il fasse un froid glacial aujourd'hui, j'ai décidé de sortir.",
            "Froid est aujourd'hui dehors.",
            "Demain il fera froid."
        ],
        "correct_index": 1,
        "explanation": "Option 2 incorporates the concessive conjunction 'Bien que' with the subjunctive 'fasse' and descriptive adjective 'glacial'."
    },
    {
        "id": 3,
        "language": "German",
        "base_level": "A1",
        "target_level": "B1",
        "base_sentence": "Ich bleibe zu Hause. Ich bin krank.",
        "instruction": "Combine and convert into a B1 sentence using the subordinate conjunction 'weil' with proper verb-final word order.",
        "options": [
            "Ich bleibe zu Hause weil ich bin krank.",
            "Ich bleibe zu Hause, weil ich krank bin.",
            "Weil krank ich bin, bleibe Hause.",
            "Ich bin krank und bleibe Hause."
        ],
        "correct_index": 1,
        "explanation": "In German B1 subordinate clauses with 'weil', the conjugated verb 'bin' is kicked to the very end."
    }
]


def calculate_badge(xp: int) -> Dict[str, str]:
    """Return user rank and achievement badge based on accumulated XP points."""
    if xp >= 500:
        return {"title": "🌳 Forest Fluent", "icon": "🌳", "color": "#2E7D32", "desc": "Advanced linguistic scholar"}
    elif xp >= 250:
        return {"title": "🌿 Branch Explorer", "icon": "🌿", "color": "#1976D2", "desc": "Solid intermediate comprehension"}
    elif xp >= 100:
        return {"title": "🌱 Sprout Starter", "icon": "🌱", "color": "#43A047", "desc": "Active daily learner"}
    else:
        return {"title": "🌰 Seedling Novice", "icon": "🌰", "color": "#8D6E63", "desc": "Beginning the language journey"}

