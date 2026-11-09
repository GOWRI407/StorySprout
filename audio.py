"""
StorySprout Capstone - Native Browser Audio & Speech Synthesis
Renders zero-latency, cross-browser Web Speech API audio players for language learners.
Requires NO external API keys and generates native pronunciation in 60+ global accents.
"""

from typing import List, Dict
import html


def render_pronunciation_button(word: str, speech_code: str = "es-ES", button_label: str = "🔊 Pronounce", height: int = 50) -> str:
    """Generate HTML/JavaScript snippet for a clickable speech synthesis button."""
    safe_word = html.escape(word).replace("'", "\\'")
    safe_code = html.escape(speech_code)
    safe_label = html.escape(button_label)

    return f"""
    <div style="display: inline-block; margin: 4px 0;">
        <button onclick="speak_{id(word)}()" style="
            background: linear-gradient(135deg, #2E86AB, #1E3D59);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 7px 16px;
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.12);
            transition: all 0.2s ease;
        " onmouseover="this.style.transform='scale(1.04)'" onmouseout="this.style.transform='scale(1.0)'">
            {safe_label}
        </button>
        <script>
        function speak_{id(word)}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var u = new SpeechSynthesisUtterance('{safe_word}');
                u.lang = '{safe_code}';
                u.rate = 0.88;
                u.pitch = 1.0;
                window.speechSynthesis.speak(u);
            }} else {{
                alert('Audio speech synthesis is not supported on this browser.');
            }}
        }}
        </script>
    </div>
    """


def render_vocab_card_with_audio(word: str, translation: str, phonetic: str, speech_code: str, pos: str = "noun") -> str:
    """Generate an interactive vocabulary flashcard with 1-click native pronunciation."""
    safe_word = html.escape(word).replace("'", "\\'")
    safe_code = html.escape(speech_code)
    safe_trans = html.escape(translation)
    safe_phon = html.escape(phonetic)
    safe_pos = html.escape(pos)
    card_id = f"c_{abs(hash(word)) % 1000000}"

    return f"""
    <div style="
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s;
    " onmouseover="this.style.borderColor='#38A169'" onmouseout="this.style.borderColor='#E2E8F0'">
        <div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1A202C;">
                {html.escape(word)}
                <span style="font-size: 0.78rem; font-weight: 600; color: #718096; background: #EDF2F7; padding: 2px 8px; border-radius: 10px; margin-left: 6px;">
                    {safe_pos}
                </span>
            </div>
            <div style="font-size: 0.95rem; color: #4A5568; margin-top: 4px;">
                <strong>Meaning:</strong> {safe_trans}
            </div>
            <div style="font-size: 0.85rem; color: #718096; font-style: italic; margin-top: 2px;">
                Phonetic: /{safe_phon}/
            </div>
        </div>
        <div>
            <button onclick="play_{card_id}()" style="
                background: #F0FFF4;
                border: 2px solid #38A169;
                color: #276749;
                border-radius: 50%;
                width: 44px;
                height: 44px;
                font-size: 1.2rem;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
            " title="Click to hear native pronunciation" onmouseover="this.style.background='#C6F6D5'" onmouseout="this.style.background='#F0FFF4'">
                🔊
            </button>
        </div>
        <script>
        function play_{card_id}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var u = new SpeechSynthesisUtterance('{safe_word}');
                u.lang = '{safe_code}';
                u.rate = 0.85;
                window.speechSynthesis.speak(u);
            }}
        }}
        </script>
    </div>
    """



def render_safe_speech_button(word: str, speech_code: str = "es-ES", button_label: str = "🔊 Hear it") -> str:
    """Pronunciation button that is safe for any word (apostrophes, quotes, scripts) and any language.

    The word is passed to JavaScript through json.dumps, so characters like ' " \\ are spoken correctly
    instead of being HTML-escaped (the older buttons read "s'il vous plaît" as "s&#x27;il ...").
    """
    import json

    js_word = json.dumps(word).replace("</", "<\\/").replace("<!--", "<\\!--")
    js_lang = json.dumps(speech_code)
    label = html.escape(button_label)
    return f"""
    <button id="b" style="background:linear-gradient(135deg,#2E86AB,#1E3D59);color:#fff;border:none;
        border-radius:20px;padding:7px 16px;font-size:.92rem;font-weight:600;cursor:pointer;
        box-shadow:0 2px 6px rgba(0,0,0,.12);">{label}</button>
    <script>
    document.getElementById('b').onclick = function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var u = new SpeechSynthesisUtterance({js_word});
            u.lang = {js_lang}; u.rate = 0.88;
            window.speechSynthesis.speak(u);
        }} else {{ alert('Speech is not supported in this browser.'); }}
    }};
    </script>
    """
