"""
StorySprout - AI Language Tutor

Provides AILanguageTutor with a single public method:
    clear_doubt(question, language, level) -> str (markdown answer)

If GEMINI_API_KEY is not set (or the SDK is not installed), tutor.client
is None and the caller should show a "no key" warning instead.
"""

from __future__ import annotations

import os

try:
    from importlib import import_module as _imp
    _dotenv = _imp("dotenv")
    _dotenv.load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Gemini client (same lazy-load pattern as generator.py)
# ---------------------------------------------------------------------------

def _build_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        pass
    try:
        from google import genai as genai2  # type: ignore[import-not-found]
        client = genai2.Client(api_key=api_key)
        return _GenAI2Wrapper(client)
    except Exception:
        return None


class _GenAI2Wrapper:
    def __init__(self, client):
        self._client = client

    def generate_content(self, prompt: str):
        response = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return _FakeResponse(response.text)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a friendly, encouraging language tutor helping a learner study {language} \
at CEFR level {level}.

Answer the learner's question clearly and concisely. Use simple English.
Where helpful, give a short example in {language} with an English translation.
Keep your answer under 200 words.

Learner's question: {question}"""


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AILanguageTutor:
    """AI-powered language tutor backed by Gemini."""

    def __init__(self):
        self.client = _build_client()

    def clear_doubt(self, question: str, language: str, level: str = "A1") -> str:
        """
        Answer a learner's language question.

        Returns a markdown-formatted string.
        Raises RuntimeError if the client is not configured.
        """
        if self.client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to enable the AI tutor."
            )

        prompt = _SYSTEM_PROMPT.format(
            language=language,
            level=level,
            question=question.strip(),
        )
        response = self.client.generate_content(prompt)
        return response.text.strip()

