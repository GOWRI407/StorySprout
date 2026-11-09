"""Language metadata and starter vocabulary used by StorySprout."""

from typing import Dict, List

GLOBAL_LANGUAGES: Dict[str, Dict[str, str]] = {
    "Chinese": {"flag": "🇨🇳", "native": "中文", "speech_tag": "zh-CN"},
    "French": {"flag": "🇫🇷", "native": "Français", "speech_tag": "fr-FR"},
    "German": {"flag": "🇩🇪", "native": "Deutsch", "speech_tag": "de-DE"},
    "Hindi": {"flag": "🇮🇳", "native": "हिन्दी", "speech_tag": "hi-IN"},
    "Italian": {"flag": "🇮🇹", "native": "Italiano", "speech_tag": "it-IT"},
    "Japanese": {"flag": "🇯🇵", "native": "日本語", "speech_tag": "ja-JP"},
    "Korean": {"flag": "🇰🇷", "native": "한국어", "speech_tag": "ko-KR"},
    "Portuguese": {"flag": "🇵🇹", "native": "Português", "speech_tag": "pt-PT"},
    "Russian": {"flag": "🇷🇺", "native": "Русский", "speech_tag": "ru-RU"},
    "Spanish": {"flag": "🇪🇸", "native": "Español", "speech_tag": "es-ES"},
    "Tamil": {"flag": "🇮🇳", "native": "தமிழ்", "speech_tag": "ta-IN"},
}

STARTER_VOCABULARY: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "Chinese": {
        "A1": [
            {"word": "你好 (nǐ hǎo)", "translation": "hello", "pos": "greeting", "phonetic": "nee how"},
            {"word": "谢谢 (xièxie)", "translation": "thank you", "pos": "phrase", "phonetic": "shyeh-shyeh"},
            {"word": "朋友 (péngyou)", "translation": "friend", "pos": "noun", "phonetic": "pung-yo"},
            {"word": "家 (jiā)", "translation": "home", "pos": "noun", "phonetic": "jyah"},
            {"word": "水 (shuǐ)", "translation": "water", "pos": "noun", "phonetic": "shway"},
            {"word": "书 (shū)", "translation": "book", "pos": "noun", "phonetic": "shoo"},
        ],
    },
    "French": {
        "A1": [
            {"word": "bonjour", "translation": "hello", "pos": "greeting", "phonetic": "bohn-zhoor"},
            {"word": "merci", "translation": "thank you", "pos": "phrase", "phonetic": "mehr-see"},
            {"word": "ami", "translation": "friend", "pos": "noun", "phonetic": "ah-mee"},
            {"word": "maison", "translation": "house", "pos": "noun", "phonetic": "meh-zohn"},
            {"word": "eau", "translation": "water", "pos": "noun", "phonetic": "oh"},
            {"word": "livre", "translation": "book", "pos": "noun", "phonetic": "leev-ruh"},
        ],
    },
    "German": {
        "A1": [
            {"word": "hallo", "translation": "hello", "pos": "greeting", "phonetic": "hah-loh"},
            {"word": "danke", "translation": "thank you", "pos": "phrase", "phonetic": "dahn-kuh"},
            {"word": "Freund", "translation": "friend", "pos": "noun", "phonetic": "froynt"},
            {"word": "Haus", "translation": "house", "pos": "noun", "phonetic": "hows"},
            {"word": "Wasser", "translation": "water", "pos": "noun", "phonetic": "vah-ser"},
            {"word": "Buch", "translation": "book", "pos": "noun", "phonetic": "bookh"},
        ],
    },
    "Hindi": {
        "A1": [
            {"word": "नमस्ते (namaste)", "translation": "hello", "pos": "greeting", "phonetic": "nuh-muh-stay"},
            {"word": "धन्यवाद (dhanyavaad)", "translation": "thank you", "pos": "phrase", "phonetic": "dhun-yuh-vaad"},
            {"word": "दोस्त (dost)", "translation": "friend", "pos": "noun", "phonetic": "dost"},
            {"word": "घर (ghar)", "translation": "house", "pos": "noun", "phonetic": "ghar"},
            {"word": "पानी (paani)", "translation": "water", "pos": "noun", "phonetic": "paa-nee"},
            {"word": "किताब (kitaab)", "translation": "book", "pos": "noun", "phonetic": "ki-taab"},
        ],
    },
    "Japanese": {
        "A1": [
            {"word": "こんにちは (konnichiwa)", "translation": "hello", "pos": "greeting", "phonetic": "kon-nee-chee-wah"},
            {"word": "ありがとう (arigatō)", "translation": "thank you", "pos": "phrase", "phonetic": "ah-ree-gah-toh"},
            {"word": "友達 (tomodachi)", "translation": "friend", "pos": "noun", "phonetic": "toh-moh-dah-chee"},
            {"word": "家 (ie)", "translation": "house", "pos": "noun", "phonetic": "ee-eh"},
            {"word": "水 (mizu)", "translation": "water", "pos": "noun", "phonetic": "mee-zoo"},
            {"word": "本 (hon)", "translation": "book", "pos": "noun", "phonetic": "hohn"},
        ],
    },
    "Spanish": {
        "A1": [
            {"word": "hola", "translation": "hello", "pos": "greeting", "phonetic": "OH-lah"},
            {"word": "gracias", "translation": "thank you", "pos": "phrase", "phonetic": "GRAH-see-ahs"},
            {"word": "amigo", "translation": "friend", "pos": "noun", "phonetic": "ah-MEE-goh"},
            {"word": "casa", "translation": "house", "pos": "noun", "phonetic": "KAH-sah"},
            {"word": "agua", "translation": "water", "pos": "noun", "phonetic": "AH-gwah"},
            {"word": "libro", "translation": "book", "pos": "noun", "phonetic": "LEE-broh"},
        ],
    },
    "Tamil": {
        "A1": [
            {"word": "வணக்கம் (vanakkam)", "translation": "hello", "pos": "greeting", "phonetic": "va-na-kkam"},
            {"word": "நன்றி (nandri)", "translation": "thank you", "pos": "phrase", "phonetic": "nun-dree"},
            {"word": "நண்பர் (nanbar)", "translation": "friend", "pos": "noun", "phonetic": "nun-bar"},
            {"word": "வீடு (veedu)", "translation": "house", "pos": "noun", "phonetic": "vee-doo"},
            {"word": "தண்ணீர் (thanneer)", "translation": "water", "pos": "noun", "phonetic": "thun-neer"},
            {"word": "புத்தகம் (puthagam)", "translation": "book", "pos": "noun", "phonetic": "pu-tha-gam"},
        ],
    },
}


def get_speech_tag(language: str) -> str:
    """Return a browser Web Speech API locale for a language."""
    return GLOBAL_LANGUAGES.get(language, {}).get("speech_tag", "en-US")
