"""StorySprout - CREATOR APP (admin / content author)
Flow:  Login (creator account)  ->  Dashboard | Content Manager | Story Studio | Learners

What creators control:
  * Vocabulary for every language + CEFR level (this is what learners' games and stories use)
  * Example stories shown to learners in the story stage
  * Audience analytics and learner progress

Run:  streamlit run creator_app.py """

import importlib
import os
import random
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _find_pkg() -> str:
    if os.getenv("STORYSPROUT_PKG"):
        return os.environ["STORYSPROUT_PKG"]
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name)
        has_legacy_layout = (
            os.path.isfile(os.path.join(p, "__init__.py"))
            and os.path.isfile(os.path.join(p, "generator.py"))
        )
        has_storysprout_core = all(
            os.path.isfile(os.path.join(p, module))
            for module in ("auth.py", "progress.py", "curriculum.py")
        )
        if os.path.isdir(p) and (has_legacy_layout or has_storysprout_core):
            return name
    raise ImportError(
        "Could not find the StorySprout package folder next to creator_app.py. "
        "Expected a package with generator.py or a core folder with auth.py, "
        "progress.py, and curriculum.py."
    )




def _mod(name: str):
    return importlib.import_module(f"{_find_pkg()}.{name}")


auth = _mod("auth")
progress = _mod("progress")
cur = _mod("curriculum")
audio = _mod("audio")
languages = _mod("languages")
themes = _mod("themes")
tracker = _mod("analytics_tracker")
evaluator = _mod("evaluator")

st.set_page_config(page_title="StorySprout - Creator Studio", page_icon="🎨", layout="wide")
ss = st.session_state
for key, default in {
    "creator": None,
    "theme": "🌙 Midnight Cyber (Dark)",
    "font": "Standard",
    "preview": None,
    "gen_story": None,
}.items():
    ss.setdefault(key, default)

st.markdown(themes.get_theme_css(ss.theme, ss.font), unsafe_allow_html=True)


def play_html(markup: str, height: int = 56):
    """Render a small HTML/JS snippet (works on old and new Streamlit versions)."""
    if hasattr(st, "iframe"):
        st.iframe(markup, height=height)
    else:
        components.html(markup, height=height)


@st.cache_resource
def get_generator():
    return _mod("generator").StoryGenerator()


# ============================================================================ LOGIN
def view_login():
    st.markdown('<div class="main-header">🎨 StorySprout Creator Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sign in to manage languages, levels, stories and analytics.</div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Log in", "Create creator account"])
        with tab_in:
            with st.form("c_login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", width="stretch"):
                    acct = auth.authenticate(u, p)
                    if acct and acct["role"] == "creator":
                        ss.creator = acct["username"]
                        st.rerun()
                    elif acct:
                        st.error("This is a learner account. Please use the learner app (user_app.py).")
                    else:
                        st.error("Wrong username or password.")
        with tab_up:
            if not auth.creator_exists():
                st.info("First-time setup: the first creator account needs no access code.")
            with st.form("c_signup"):
                u = st.text_input("Choose a username")
                p = st.text_input("Choose a password (6+ characters)", type="password")
                code = st.text_input(
                    "Creator access code",
                    type="password",
                    help="Required once a creator already exists. Set CREATOR_ACCESS_CODE in .env",
                )
                if st.form_submit_button("Create creator account", width="stretch"):
                    ok, msg = auth.register(u, p, role="creator", access_code=code)
                    (st.success if ok else st.error)(msg)


# ============================================================================ DASHBOARD
def page_dashboard():
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)
    m = tracker.load_usage_metrics()
    s = m["summary"]
    cols = st.columns(6)
    for col, (label, key) in zip(
        cols,
        [
            ("Visitors", "total_unique_visitors"),
            ("Page views", "total_page_views"),
            ("Stories", "stories_created"),
            ("Words mastered", "words_mastered"),
            ("Tutor queries", "coaching_queries"),
            ("Games played", "games_played"),
        ],
    ):
        col.metric(label, f"{s.get(key, 0):,}")

    st.subheader("Daily traffic")
    daily = pd.DataFrame(m["daily_traffic"]).T.sort_index()
    st.line_chart(daily[["visitors", "page_views", "stories"]])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Language popularity")
        lp = pd.Series(m["language_popularity"]).sort_values(ascending=False)
        st.bar_chart(lp)
    with c2:
        st.subheader("Level breakdown")
        lb = pd.Series(m["level_breakdown"]).reindex(cur.LEVELS).fillna(0)
        st.bar_chart(lb)

    st.subheader("Recent activity")
    st.dataframe(
        pd.DataFrame(m.get("recent_activity_log", [])),
        width="stretch",
        hide_index=True,
    )


# ============================================================================ CONTENT MANAGER
def page_content():
    st.markdown('<div class="main-header">📚 Content Manager</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Vocabulary here powers the learners\' games and story words, '
        "in exactly the language you choose.</div>",
        unsafe_allow_html=True,
    )

    choices = cur.language_choices()
    c1, c2 = st.columns([3, 1])
    lang = choices[c1.selectbox("Language", list(choices))]
    level = c2.selectbox(
        "Level",
        cur.LEVELS,
        format_func=lambda l: f"{l} - {cur.LEVEL_NAMES[l]}",
    )

    status = cur.content_status(lang)
    st.markdown(
        "**Coverage for " + lang + ":**  "
        + "   ".join(
            f"{l} {'✅ ' + src if src != 'none' else '⬜ empty'}"
            for l, src in status.items()
        )
    )

    items, source = cur.get_level_vocab(lang, level)
    st.caption(f"Source for {lang} {level}: **{source}**. Saving below creates a creator override.")

    df = pd.DataFrame(
        items or [{"word": "", "translation": "", "pos": "noun", "phonetic": ""}],
        columns=["word", "translation", "pos", "phonetic"],
    )
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key=f"ed_{lang}_{level}",
    )

    b1, b2, b3 = st.columns(3)
    if b1.button("💾 Save vocabulary", type="primary"):
        rows = cur.clean_items(edited.to_dict("records"))
        cur.save_custom_vocab_level(lang, level, rows)
        st.success(f"Saved {len(rows)} {lang} words for {level}. Learners see them immediately.")
        st.rerun()
    if b2.button("✨ Generate with Gemini"):
        if not cur.ai_available():
            st.warning("Add GEMINI_API_KEY to your .env file to enable generation.")
        else:
            with st.spinner("Generating..."):
                new = cur.generate_vocab_with_ai(lang, level)
            if new:
                cur.save_custom_vocab_level(lang, level, new)
                st.success(f"Generated and saved {len(new)} words.")
                st.rerun()
            else:
                st.error("Generation failed. Check your API key and try again.")
    if b3.button("🗑️ Remove my override"):
        cur.save_custom_vocab_level(lang, level, [])
        st.rerun()

    st.divider()
    st.subheader("Preview the learner game")
    if st.button("🎮 Build a 5-question round"):
        ss.preview = cur.build_round(lang, level, seed=random.randrange(10**9))
    if ss.preview:
        qs, note = ss.preview
        if note:
            st.caption("ℹ️ " + note)
        if not qs:
            st.warning("Nothing to preview - this language/level has no vocabulary yet.")
        for i, q in enumerate(qs, 1):
            st.markdown(
                f"**{i}. {q['prompt']}**"
                + (f"  →  `{q['word']}`" if q["kind"] == "meaning" else "")
            )
            st.write(
                "  |  ".join(
                    ("✅ " if o == q["answer"] else "") + o for o in q["options"]
                )
            )
            if q["kind"] == "meaning" and i == 1:
                play_html(
                    audio.render_safe_speech_button(
                        q["word"], languages.get_speech_tag(lang)
                    )
                )


# ============================================================================ STORY STUDIO
def page_stories():
    st.markdown('<div class="main-header">✍️ Story Studio</div>', unsafe_allow_html=True)
    choices = cur.language_choices()
    c1, c2, c3 = st.columns([3, 1, 2])
    lang = choices[c1.selectbox("Language", list(choices), key="st_lang")]
    level = c2.selectbox("Level", cur.LEVELS, key="st_level")
    topic = c3.text_input("Topic", "A day at the market")
    gen = get_generator()

    if not gen.client:
        st.warning(
            f"No Gemini key found, so the generator can't write new {lang} stories. "
            f"You can still paste your own {lang} story below."
        )
    elif st.button("✨ Generate story", type="primary"):
        with st.spinner("Writing..."):
            result = gen.generate_story(
                target_language=lang, cefr_level=level, topic=topic
            )
            # Normalize: some generators return {"story": {...}}, others return the story dict directly
            if isinstance(result, dict) and "story" in result and isinstance(result["story"], dict):
                ss.gen_story = result["story"]
            else:
                ss.gen_story = result

    # Only reuse a previously generated story when it matches the current language
    # (or when the generator did not tag a language)
    story = None
    if isinstance(ss.gen_story, dict):
        story_lang = ss.gen_story.get("target_language")
        if story_lang is None or story_lang == lang:
            story = ss.gen_story

    st.subheader("Story to publish")
    title = st.text_input(
        "Title",
        value=(story.get("title") or "") if story else "",
    )
    text = st.text_area(
        f"Story text ({lang})",
        value=(story.get("story_text") or "") if story else "",
        height=200,
    )
    translation = st.text_area(
        "English translation",
        value=(story.get("full_translation") or "") if story else "",
        height=120,
    )

    if text.strip():
        d = evaluator.estimate_story_difficulty(text, level)
        st.caption(
            f"Difficulty check: {d['level_fit']}  •  {d['metrics']['word_count']} words"
        )

    if st.button(
        "📤 Publish to learners",
        disabled=not (title.strip() and text.strip()),
    ):
        cur.save_creator_story(
            {
                "language": lang,
                "level": level,
                "title": title.strip(),
                "story_text": text.strip(),
                "full_translation": translation.strip(),
            }
        )
        st.success(f"Published. Learners studying {lang} at {level} can now read it.")

    st.divider()
    st.subheader("Published stories")
    stories = cur.load_creator_stories()
    if not stories:
        st.caption("Nothing published yet.")
    for i, s in enumerate(stories):
        with st.expander(f"{s['language']} {s['level']} - {s['title']}"):
            st.write(s["story_text"])
            if s.get("full_translation"):
                st.caption(s["full_translation"])
            if st.button("Delete", key=f"del_{i}"):
                cur.delete_creator_story(i)
                st.rerun()


# ============================================================================ LEARNERS
def page_learners():
    st.markdown('<div class="main-header">👥 Learners</div>', unsafe_allow_html=True)
    accounts = auth.list_accounts()
    profiles = progress.all_profiles()
    rows = []
    for name, acct in accounts.items():
        if acct.get("role") != "learner":
            continue
        profile = profiles.get(name, {})
        langs = profile.get("languages", {})
        rows.append(
            {
                "username": name,
                "xp": profile.get("xp", 0),
                "languages": ", ".join(langs) or "-",
                "current": profile.get("last_language") or "-",
                "words mastered": sum(
                    len(state.get("words_mastered", []))
                    for state in langs.values()
                ),
                "stories written": len(profile.get("stories", [])),
                "joined": (acct.get("created") or "")[:10] or "-",
            }
        )
    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values("xp", ascending=False),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No learners have signed up yet.")


# ============================================================================ ROUTER
if not ss.creator:
    view_login()
else:
    with st.sidebar:
        st.markdown(f"### 🎨 {ss.creator}")
        page = st.radio(
            "Go to",
            ["📊 Dashboard", "📚 Content Manager", "✍️ Story Studio", "👥 Learners"],
        )
        st.divider()

        theme_options = list(themes.THEMES)
        try:
            theme_index = theme_options.index(ss.theme)
        except ValueError:
            theme_index = 0
        ss.theme = st.selectbox("Theme", theme_options, index=theme_index)

        if st.button("Log out", width="stretch"):
            ss.creator = None
            ss.preview = None
            ss.gen_story = None
            st.rerun()

    page_routes = {
        "📊 Dashboard": page_dashboard,
        "📚 Content Manager": page_content,
        "✍️ Story Studio": page_stories,
        "👥 Learners": page_learners,
    }
    page_routes[page]()
