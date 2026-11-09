"""
StorySprout - AI Story Generator

Wraps the Gemini generative AI API to produce CEFR-levelled example stories.
If no GEMINI_API_KEY is set (or the google-genai package is not installed),
the generator degrades gracefully: gen.client is None and callers should
display a "no key" message instead of calling generate_story().

generate_story() returns a dict:
    {
        "story": {
            "target_language": str,
            "cefr_level":      str,
            "title":           str,
            "story_text":      str,       # in the target language
            "full_translation": str,      # English translation
        }
    }
"""

from __future__ import annotations

import json
import os
import re

try:
    from importlib import import_module as _imp

    _dotenv = _imp("dotenv")
    _dotenv.load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Lazy Gemini client setup
# ---------------------------------------------------------------------------

def _build_client():
    """Return a Gemini GenerativeModel client, or None if unavailable."""
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
    """Thin adapter so both google-generativeai and google-genai SDKs share one interface."""

    def __init__(self, client):
        self._client = client

    def generate_content(self, prompt: str) -> "_FakeResponse":
        response = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return _FakeResponse(response.text)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_LEVEL_GUIDANCE = {
    "A1": "very simple sentences, common everyday words, present tense, 60-80 words",
    "A2": "simple sentences, common vocabulary, some past tense, 80-120 words",
    "B1": "clear paragraphs, varied vocabulary, past and future tenses, 120-180 words",
    "B2": "varied sentence structure, idiomatic phrases, 180-250 words",
    "C1": "complex structures, nuanced vocabulary, 250-350 words",
}

_PROMPT_TEMPLATE = """Write a short language-learning story in {language} for a CEFR {level} learner.

Guidelines:
- {guidance}
- The story must be entirely in {language} (no English in the story itself).
- Choose a simple, engaging topic related to: {topic}

Respond ONLY with valid JSON, no markdown fences, in this exact shape:
{{
  "target_language": "{language}",
  "cefr_level": "{level}",
  "title": "<title in {language}>",
  "story_text": "<the story in {language}>",
  "full_translation": "<faithful English translation of the story>"
}}"""


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a (possibly fenced) string."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in generator response:\n{raw[:300]}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class StoryGenerator:
    """Generate CEFR-levelled stories via Gemini."""

    def __init__(self):
        self.client = _build_client()

    def generate_story(
        self,
        target_language: str = "Spanish",
        cefr_level: str = "A1",
        topic: str = "daily life",
    ) -> dict:
        """
        Generate a story and return it as:
            {"story": {"target_language": ..., "cefr_level": ..., "title": ...,
                       "story_text": ..., "full_translation": ...}}

        Raises RuntimeError if the client is not configured.
        """
        if self.client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to enable story generation."
            )

        guidance = _LEVEL_GUIDANCE.get(cefr_level, _LEVEL_GUIDANCE["A1"])
        prompt = _PROMPT_TEMPLATE.format(
            language=target_language,
            level=cefr_level,
            guidance=guidance,
            topic=topic,
        )

        response = self.client.generate_content(prompt)
        story_dict = _extract_json(response.text)

        # Ensure required keys are present with sane defaults
        story_dict.setdefault("target_language", target_language)
        story_dict.setdefault("cefr_level", cefr_level)
        story_dict.setdefault("title", "Story")
        story_dict.setdefault("story_text", "")
        story_dict.setdefault("full_translation", "")

        return {"story": story_dict}

