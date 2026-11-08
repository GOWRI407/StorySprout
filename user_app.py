"""
StorySprout - LEARNER APP
Flow:  1) Login  ->  2) Choose language  ->  3) Game round (5 questions)
       ->  4) Write a story  ->  5) Next level's game  ->  ...  ->  C1 complete

Run:  streamlit run user_app.py
"""

import importlib
import os
import random
import sys
import uuid

import streamlit as st  # type: ignore[import-not-found]
import streamlit.components.v1 as components  # type: ignore[import-not-found]

# ---- locate the core package next to this file -----------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _find_pkg() -> str:
    def is_storysprout_package(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_legacy_layout = (
            os.path.isfile(os.path.join(path, "__init__.py"))
            and os.path.isfile(os.path.join(path, "generator.py"))
        )
        has_storysprout_core = all(
            os.path.isfile(os.path.join(path, module))
            for module in ("auth.py", "progress.py", "curriculum.py")
        )
        return has_legacy_layout or has_storysprout_core

    configured = os.getenv("STORYSPROUT_PKG", "").strip()
    if configured:
        package_path = os.path.join(ROOT, *configured.split("."))
        if is_storysprout_package(package_path):
            return configured
        raise ImportError(
            f"STORYSPROUT_PKG={configured!r} does not identify a StorySprout "
            "package next to user_app.py."
        )

    try:
        names = sorted(os.listdir(ROOT))
    except OSError as exc:
        raise ImportError(f"Could not inspect the app directory {ROOT!r}.") from exc
    for name in names:
        if is_storysprout_package(os.path.join(ROOT, name)):
            return name
    raise ImportError(
        "Could not find the StorySprout package folder next to user_app.py. "
        "Expected a package with generator.py or a core folder with auth.py, "
        "progress.py, and curriculum.py."
    )


PKG = _find_pkg()


def _mod(name: str):
    return importlib.import_module(f"{PKG}.{name}")


auth = _mod("auth")
progress = _mod("progress")
cur = _mod("curriculum")
audio = _mod("audio")
languages = _mod("languages")
themes = _mod("themes")
tracker = _mod("analytics_tracker")
games = _mod("games")
evaluator = _mod("evaluator")

st.set_page_config(page_title="StorySprout - Learn", page_icon="🌱", layout="wide")
ss = st.session_state

for key, default in {
    "user": None, "language": None, "game": None, "story_result": None,
    "sid": uuid.uuid4().hex, "viewed": False,
    "theme": "🌿 Emerald Sprout (Light)", "font": "Standard", "model_story": None,
}.items():
    ss.setdefault(key, default)

st.markdown(themes.get_theme_css(ss.theme, ss.font), unsafe_allow_html=True)


@st.cache_resource
def get_tutor():
    return _mod("tutor").AILanguageTutor()


@st.cache_resource
def get_generator():
    return _mod("generator").StoryGenerator()


def play_html(markup: str, height: int = 56):
    """Render a small HTML/JS snippet (works on old and new Streamlit versions)."""
    if hasattr(st, "iframe"):
        st.iframe(markup, height=height)
    else:
        components.html(markup, height=height)


def speak(word: str, language: str, label: str = "🔊 Hear it"):
    """Pronunciation button. Must run inside an iframe component or the JS will not execute."""
    play_html(audio.render_safe_speech_button(word, languages.get_speech_tag(language), label))


# ============================================================================ 1) LOGIN
def view_login():
    st.markdown('<div class="main-header">🌱 StorySprout</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Learn any language through games and stories.</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Log in", "Create account"])
        with tab_in:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", width="stretch"):
                    acct = auth.authenticate(u, p)
                    if acct:
                        ss.user = acct["username"]
                        ss.language = None
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")
        with tab_up:
            with st.form("signup_form"):
                u = st.text_input("Choose a username")
                p = st.text_input("Choose a password (6+ characters)", type="password")
                if st.form_submit_button("Create account", width="stretch"):
                    ok, msg = auth.register(u, p, role="learner")
                    (st.success if ok else st.error)(msg + (" You can log in now." if ok else ""))


# ============================================================================ 2) LANGUAGE
def view_language_select():
    st.markdown(f'<div class="main-header">Welcome, {ss.user}! 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pick the language you want to learn. Everything - games, words, '
                'stories and audio - will be in that language.</div>', unsafe_allow_html=True)

    prof = progress.get_profile(ss.user)
    started = prof.get("languages", {})
    if started:
        st.subheader("Continue learning")
        cols = st.columns(min(4, len(started)))
        for i, (lang, state) in enumerate(started.items()):
            kind, lvl = progress.current_stage(state)
            info = languages.GLOBAL_LANGUAGES.get(lang, {})
            label = f"{info.get('flag', '🌍')} {lang}\n" + ("Complete 🎓" if kind == "done" else f"{lvl} {kind}")
            if cols[i % len(cols)].button(label, key=f"cont_{lang}", width="stretch"):
                select_language(lang, state["start_level"])
        st.divider()

    st.subheader("Start a new language")
    choices = cur.language_choices()
    q = st.text_input("🔎 Search languages", placeholder="e.g. Tamil, Japanese, French")
    shown = [d for d in choices if q.strip().lower() in d.lower()] if q.strip() else list(choices)
    if not shown:
        st.warning("No language matches your search.")
        return
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Language", shown)
    level = c2.selectbox("Start at level", cur.LEVELS, format_func=lambda l: f"{l} - {cur.LEVEL_NAMES[l]}")
    lang = choices[choice]

    existing = started.get(lang)
    if existing:
        st.info(f"You already started {lang}. Choosing it again continues where you left off.")
        if st.button(f"Continue {lang}", type="primary"):
            select_language(lang, existing["start_level"])
        if st.button(f"Restart {lang} from {level}"):
            select_language(lang, level, restart=True)
    elif st.button(f"Start learning {lang}", type="primary"):
        select_language(lang, level)


def select_language(lang: str, level: str, restart: bool = False):
    progress.start_language(ss.user, lang, level, restart=restart)
    ss.language = lang
    ss.game = None
    ss.story_result = None
    ss.model_story = None
    tracker.record_page_view("journey", ss.sid, language=lang, level=level)
    st.rerun()


# ============================================================================ 3-5) JOURNEY
def journey_map(state, current_idx):
    stages = progress.journey_for(state["start_level"])
    levels = []
    for _, lvl in stages:
        if lvl not in levels:
            levels.append(lvl)
    cols = st.columns(len(levels))
    for col, lvl in zip(cols, levels):
        marks = []
        for kind, icon in (("game", "🎮"), ("story", "✍️")):
            idx = stages.index((kind, lvl))
            status = "✅" if idx < current_idx else ("▶️" if idx == current_idx else "🔒")
            marks.append(f"{status} {icon}")
        col.markdown(f"**{lvl}**  \n{'  '.join(marks)}")
    st.progress(min(1.0, current_idx / len(stages)))


def view_journey():
    lang = ss.language
    info = languages.GLOBAL_LANGUAGES.get(lang, {})
    state = progress.get_language_state(ss.user, lang)
    if state is None:  # safety: language chosen but no saved state
        state = progress.start_language(ss.user, lang, "A1")
    kind, level = progress.current_stage(state)

    st.markdown(f'<div class="main-header">{info.get("flag", "🌍")} Learning {lang} '
                f'<span style="font-size:1.2rem">({info.get("native", "")})</span></div>', unsafe_allow_html=True)
    journey_map(state, state["stage"])
    st.divider()

    if kind == "game":
        view_game(lang, level, state)
    elif kind == "story":
        view_story(lang, level, state)
    else:
        view_complete(lang)

    with st.expander("💬 Ask the AI tutor about " + lang):
        tutor = get_tutor()
        question = st.text_input("Your question", key="tutor_q", placeholder=f"e.g. How do I greet someone in {lang}?")
        if st.button("Ask", key="tutor_btn") and question.strip():
            if tutor.client:
                with st.spinner("Thinking..."):
                    st.markdown(tutor.clear_doubt(question, lang, level or "A1"))
                tracker.record_action("coaching_query", lang)
            else:
                st.warning("The AI tutor needs a GEMINI_API_KEY in your .env file. Games and stories still work without it.")


# ---- game ---------------------------------------------------------------------------------
def new_game(lang, level):
    qs, note = cur.build_round(lang, level, seed=random.randrange(10 ** 9))
    ss.game = {"language": lang, "level": level, "questions": qs, "note": note, "idx": 0, "score": 0,
               "answered": False, "correct": False, "awarded": False, "correct_words": [],
               "rid": uuid.uuid4().hex[:8]}


def view_game(lang, level, state):
    g = ss.game
    if g is None or g["language"] != lang or g["level"] != level:
        new_game(lang, level)
        g = ss.game

    st.subheader(f"🎮 {level} Word Game - {cur.LEVEL_NAMES[level]}")
    if g["note"]:
        st.caption("ℹ️ " + g["note"])
    if not g["questions"]:
        st.warning(g["note"])
        if st.button("Check again"):
            ss.game = None
            st.rerun()
        return

    qs = g["questions"]
    total = len(qs)

    if g["idx"] >= total:  # ---- results
        passed = g["score"] >= cur.PASS_MARK
        st.progress(1.0, text="Round finished")
        st.metric("Your score", f"{g['score']} / {total}")
        if passed:
            if not g["awarded"]:
                g["awarded"] = True
                progress.add_xp(ss.user, 10 * g["score"] + 30)
                progress.add_words(ss.user, lang, g["correct_words"])
                tracker.record_action("game_played", lang)
                for _ in g["correct_words"]:
                    tracker.record_action("word_mastered", lang)
            st.success(f"Great job! You unlocked the {level} story. +{10 * g['score'] + 30} XP")
            st.balloons() if g["score"] == total else None
            if st.button("Continue ➜ Write your story", type="primary"):
                progress.advance(ss.user, lang)
                ss.game = None
                ss.story_result = None
                st.rerun()
        else:
            if not g["awarded"]:
                g["awarded"] = True
                tracker.record_action("game_played", lang)
            st.warning(f"You need {cur.PASS_MARK} correct answers to move on. Try again!")
            if st.button("Play again", type="primary"):
                ss.game = None
                st.rerun()
        return

    q = qs[g["idx"]]
    st.progress(g["idx"] / total, text=f"Question {g['idx'] + 1} of {total}   •   Score {g['score']}")

    if q["kind"] == "meaning":
        st.markdown(f"<div style='font-size:2.2rem;font-weight:800'>{q['word']}</div>"
                    f"<div style='opacity:.7'>{q['pos']}  /{q['phonetic']}/</div>", unsafe_allow_html=True)
        speak(q["word"], lang)

    pick = st.radio(q["prompt"], q["options"], index=None, key=f"r_{g['rid']}_{g['idx']}", disabled=g["answered"])

    if not g["answered"]:
        if st.button("Check answer", disabled=pick is None, type="primary"):
            g["answered"] = True
            g["correct"] = pick == q["answer"]
            if g["correct"]:
                g["score"] += 1
                g["correct_words"].append(q["word"])
            st.rerun()
    else:
        if g["correct"]:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Not quite. The answer is **{q['answer']}**.")
        if q["kind"] == "word":
            st.markdown(f"**{q['word']}**  /{q['phonetic']}/  = {q['translation']}")
            speak(q["word"], lang)
        last = g["idx"] + 1 >= total
        if st.button("See results" if last else "Next ➜", type="primary"):
            g["idx"] += 1
            g["answered"] = False
            st.rerun()


# ---- story --------------------------------------------------------------------------------
def view_story(lang, level, state):
    vocab, _ = cur.get_level_vocab(lang, level, generate=True)
    if not vocab:  # same fallback pool the game used
        vocab, _ = cur.get_round_pool(lang, level)
    need_len = cur.MIN_STORY_UNITS[level]
    need_words = min(cur.MIN_WORDS_USED, len(vocab))

    st.subheader(f"✍️ {level} Story Time - write in {lang}")
    st.markdown(f"Write a short story or paragraph **in {lang}** (at least **{need_len}** words). "
                f"Use at least **{need_words}** of these words:")
    cols = st.columns(min(4, max(1, len(vocab))))
    for i, v in enumerate(vocab):
        with cols[i % len(cols)]:
            st.markdown(f"**{v['word']}**  \n{v['translation']}")
            speak(v["word"], lang, "🔊")

    models = cur.stories_for(lang, level)
    gen = get_generator()
    with st.expander("📖 See an example story first"):
        for s in models:
            st.markdown(f"**{s.get('title', '')}**\n\n{s.get('story_text', '')}")
            if s.get("full_translation"):
                st.caption(s["full_translation"])
        if not models:
            if gen.client:
                if st.button("Generate an example story"):
                    with st.spinner("Writing..."):
                        res = gen.generate_story(target_language=lang, cefr_level=level)
                    ss.model_story = res["story"]
                if ss.model_story and ss.model_story.get("target_language") == lang:
                    st.markdown(f"**{ss.model_story['title']}**\n\n{ss.model_story['story_text']}")
                    st.caption(ss.model_story["full_translation"])
            else:
                st.info(f"No example {lang} story for {level} yet. A creator can add one, "
                        "or add GEMINI_API_KEY to generate them.")

    text = st.text_area("Your story", height=200, key=f"story_{lang}_{level}",
                        placeholder=f"Write in {lang}...")
    if st.button("Check my story", type="primary"):
        ss.story_result = {"check": cur.check_story(text, level, vocab),
                           "difficulty": evaluator.estimate_story_difficulty(text, level), "text": text,
                           "awarded": False}

    res = ss.story_result
    if res:
        c = res["check"]
        st.write(f"Length: **{c['length']}** / {c['needed_length']} words  "
                 f"{'✅' if c['length_ok'] else '❌'}")
        st.write(f"Target words used: **{len(c['used_words'])}** / {c['needed_words']}  "
                 f"{'✅' if c['words_ok'] else '❌'}  {', '.join(c['used_words'])}")
        if c["length"] > 0:
            st.caption(f"Estimated difficulty: {res['difficulty']['level_fit']}")
        if c["passed"]:
            if not res["awarded"]:
                res["awarded"] = True
                progress.add_xp(ss.user, 40)
                progress.add_story(ss.user, lang, level, res["text"])
                tracker.record_action("story_created", lang)
            st.success("🎉 Story accepted! +40 XP")
            nxt = "Finish the course 🎓" if level == cur.LEVELS[-1] else "Next level game ➜"
            if st.button(nxt, type="primary"):
                progress.advance(ss.user, lang)
                ss.game = None
                ss.story_result = None
                st.rerun()
        else:
            st.info("Almost there - adjust your story and check again.")


def view_complete(lang):
    st.success(f"🎓 You finished the whole {lang} journey!")
    st.balloons()
    if st.button("Learn another language", type="primary"):
        ss.language = None
        ss.game = None
        st.rerun()


# ============================================================================ SIDEBAR + ROUTER
def sidebar():
    with st.sidebar:
        st.markdown(f"### 🌱 {ss.user}")
        xp = progress.get_profile(ss.user).get("xp", 0)
        badge = games.calculate_badge(xp)
        st.markdown(f"**{badge['title']}**  \n{xp} XP - {badge['desc']}")
        if ss.language:
            info = languages.GLOBAL_LANGUAGES.get(ss.language, {})
            st.markdown(f"**Language:** {info.get('flag', '')} {ss.language}")
            if st.button("🔄 Change language", width="stretch"):
                ss.language = None
                ss.game = None
                ss.story_result = None
                st.rerun()
        st.divider()
        ss.theme = st.selectbox("Theme", list(themes.THEMES), index=list(themes.THEMES).index(ss.theme))
        ss.font = st.selectbox("Text size", ["Standard", "Large (Comfort)", "Extra Large"],
                               index=["Standard", "Large (Comfort)", "Extra Large"].index(ss.font))
        if st.button("Log out", width="stretch"):
            for k in ("user", "language", "game", "story_result", "model_story"):
                ss[k] = None
            st.rerun()


if not ss.user:
    view_login()
else:
    sidebar()
    if not ss.language:
        view_language_select()
    else:
        view_journey()
