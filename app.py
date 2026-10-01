"""Streamlit UI for the Notion AI Assistant (Step 12).

This file is UI ONLY. All chatbot logic (retrieval, prompting, Gemini,
follow-ups, error handling) lives in the chatbot package:

    app.py -> chatbot.chatbot.answer_question() -> retriever -> prompts -> llm

Run with:  streamlit run app.py
"""

import re

import streamlit as st

from chatbot import config, prompts
from chatbot.chatbot import answer_question

# --------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Notion AI Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling: small, restrained, Notion-inspired (light, subtle borders)
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.stApp { background-color: #ffffff; }
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
/* Streamlit's fixed top toolbar is ~3.75rem tall; keep content clear of it. */
.block-container { max-width: 860px; padding-top: 4.5rem; padding-bottom: 6rem; }

.app-header { padding: 0.25rem 0 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid #e9e9e7; }
.app-title { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.06em; line-height: 1.3;
             color: #37352f; margin: 0; padding: 0; }
.app-subtitle { font-size: 1rem; line-height: 1.5; color: #787774; margin: 0.35rem 0 0 0; }

section[data-testid="stSidebar"] { background-color: #f7f7f5; border-right: 1px solid #e9e9e7; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left; background: #ffffff; color: #37352f;
    border: 1px solid #e3e2e0; border-radius: 8px; font-size: 0.88rem;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color: #2383e2; color: #2383e2; }

[data-testid="stChatMessage"] {
    border: 1px solid #ececea; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.6rem;
    background-color: #ffffff;
}
.sources { margin-top: 0.5rem; font-size: 0.85rem; color: #787774; }
.sources a { color: #2383e2; text-decoration: none; }
.sources a:hover { text-decoration: underline; }

@media (max-width: 640px) {
    .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
    .app-title { font-size: 1.3rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
#   messages      : [{"role": ..., "content": ...}]  (exactly what the backend expects)
#   sources       : {message_index: [{"title":..., "url":...}]}  (kept separately so
#                   messages stay in the plain role/content format)
#   messages_used : how many user questions have been sent this session
#   pending_question : set when a suggested-question button is clicked
# --------------------------------------------------------------------------
def init_chat() -> None:
    """Reset chat to the initial greeting (no Gemini call needed)."""
    st.session_state.messages = [
        {"role": "assistant", "content": prompts.greeting_message()}
    ]
    st.session_state.sources = {}
    st.session_state.messages_used = 0
    st.session_state.pending_question = None


if "messages" not in st.session_state:
    init_chat()


def queue_question(question: str) -> None:
    """Callback for suggested-question buttons."""
    st.session_state.pending_question = question


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_INLINE_CODE = re.compile(r"`+([^`\n]+)`+")
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def format_for_display(text: str) -> str:
    """Display-only cleanup of assistant text (the stored answer is not changed).

    - Removes inline-code backticks so prices/plan names render as normal text.
    - Escapes "$" so Streamlit doesn't treat "$0 ... $10" as LaTeX math.
    Paragraphs, bullets and **bold** are untouched; fenced code blocks are kept.
    """
    parts = _FENCE.split(text)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        part = _INLINE_CODE.sub(r"\1", part)
        parts[i] = _UNESCAPED_DOLLAR.sub(r"\\$", part)
    return "".join(parts)


def render_sources(sources) -> None:
    """Show clickable links for sources returned by ChatResult (never invented)."""
    if not sources:
        return
    seen = set()
    links = []
    for src in sources:
        url = src.get("url")
        title = src.get("title") or url
        if not url or url in seen:
            continue
        seen.add(url)
        links.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>')
    if links:
        st.markdown(
            '<div class="sources">Sources: ' + " &nbsp;·&nbsp; ".join(links) + "</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "An AI assistant that answers questions about Notion using a verified "
        "knowledge base built from official Notion pages. Answers include their sources."
    )

    st.markdown("### What I can help with")
    st.markdown(
        "- Notion overview\n"
        "- Products and features\n"
        "- Notion AI\n"
        "- Plans and pricing\n"
        "- Common FAQs\n"
        "- Use cases"
    )

    st.markdown("### Suggested questions")
    for i, question in enumerate(prompts.SUGGESTED_QUESTIONS):
        st.button(question, key=f"suggested_{i}", on_click=queue_question, args=(question,))

    st.divider()
    st.caption(
        f"Messages used: {st.session_state.messages_used} / {config.MAX_MESSAGES_PER_SESSION}"
    )
    st.button("🗑️ Clear chat", key="clear_chat", on_click=init_chat)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">NOTION AI ASSISTANT</p>
        <p class="app-subtitle">Ask questions about Notion's products, features, AI, plans and pricing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Render existing chat history
# --------------------------------------------------------------------------
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(format_for_display(msg["content"]))
            render_sources(st.session_state.sources.get(index))
        else:
            st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Handle a new question (typed, or clicked from the sidebar)
# --------------------------------------------------------------------------
typed = st.chat_input("Ask about Notion...")
user_text = typed or st.session_state.pending_question
st.session_state.pending_question = None

if user_text:
    # Show the user's message immediately.
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The current message is passed separately: history must NOT contain it.
                result = answer_question(
                    user_text,
                    history=list(st.session_state.messages),
                    messages_used=st.session_state.messages_used,
                )
                answer, sources = result.answer, result.sources
            except Exception:
                # Last-resort guard so users never see a stack trace.
                answer = "Sorry, something went wrong. Please try again in a moment."
                sources = []
        st.markdown(format_for_display(answer))
        render_sources(sources)

    # Save both turns (plain role/content dicts) and the sources for this reply.
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources[len(st.session_state.messages) - 1] = sources
    st.session_state.messages_used += 1

    # Refresh so the sidebar message counter updates.
    st.rerun()