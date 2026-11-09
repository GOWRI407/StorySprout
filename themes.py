"""Theme palettes and typography for the StorySprout Streamlit apps."""

from typing import Dict

THEMES: Dict[str, Dict[str, str]] = {
    "🌿 Emerald Sprout (Light)": {
        "background": "#f4fbf6",
        "surface": "#ffffff",
        "text": "#183b2b",
        "muted": "#527060",
        "primary": "#238653",
        "border": "#d8eadf",
    },
    "🌙 Midnight Cyber (Dark)": {
        "background": "#101827",
        "surface": "#1b2638",
        "text": "#edf5ff",
        "muted": "#adbed2",
        "primary": "#55d6be",
        "border": "#34445a",
    },
    "🌊 Ocean Breeze (Light)": {
        "background": "#f1f8fc",
        "surface": "#ffffff",
        "text": "#17354a",
        "muted": "#526e80",
        "primary": "#147d9b",
        "border": "#d3e6ee",
    },
    "🌅 Sunset Glow (Light)": {
        "background": "#fff8f2",
        "surface": "#ffffff",
        "text": "#492d25",
        "muted": "#806358",
        "primary": "#c45b37",
        "border": "#f1dfd4",
    },
}

_FONT_SIZES = {
    "Standard": "1rem",
    "Large (Comfort)": "1.125rem",
    "Extra Large": "1.25rem",
}


def get_theme_css(theme: str, font: str = "Standard") -> str:
    """Return Streamlit CSS for a named palette and accessible text size."""
    palette = THEMES.get(theme, THEMES["🌿 Emerald Sprout (Light)"])
    font_size = _FONT_SIZES.get(font, _FONT_SIZES["Standard"])
    return f"""
    <style>
    :root {{
        --story-bg: {palette["background"]};
        --story-surface: {palette["surface"]};
        --story-text: {palette["text"]};
        --story-muted: {palette["muted"]};
        --story-primary: {palette["primary"]};
        --story-border: {palette["border"]};
    }}
    html, body, [class*="stApp"] {{
        font-size: {font_size};
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: var(--story-bg);
        color: var(--story-text);
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--story-surface);
        border-right: 1px solid var(--story-border);
    }}
    [data-testid="stMarkdownContainer"] {{
        color: var(--story-text);
    }}
    [data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stForm"] {{
        background-color: var(--story-surface);
        border-color: var(--story-border);
    }}
    button[kind="primary"] {{
        background-color: var(--story-primary);
        border-color: var(--story-primary);
    }}
    </style>
    """
