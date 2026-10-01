"""
chatbot.py - the orchestration layer (the "conductor").

It does not search, build prompts or talk to Gemini itself. It calls the
modules that do, in the right order, and returns one simple ChatResult
that the web page (app.py) can display:

    check input -> small talk -> load KB -> retrieve -> build prompt
    -> ask Gemini -> attach sources from the KB records

Rules this file follows:
  * No UI code and no Streamlit.
  * Source links come from knowledge-base records, never from Gemini.
  * Raw exceptions never reach the visitor.
  * This file never touches the Gemini key; only llm.py does.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from chatbot import config, knowledge_base, llm, prompts, retriever

logger = logging.getLogger(__name__)

# ---- What kind of answer was it? (ChatResult.kind) -------------------------
KIND_GREETING = "greeting"   # small talk, Gemini not called
KIND_THANKS = "thanks"       # small talk, Gemini not called
KIND_GOODBYE = "goodbye"     # small talk, Gemini not called
KIND_FALLBACK = "fallback"   # nothing relevant found, Gemini not called
KIND_LLM = "llm"             # normal answer written by Gemini
KIND_DEGRADED = "degraded"   # Gemini failed, answered from the KB instead
KIND_ERROR = "error"         # a friendly error message


@dataclass
class ChatResult:
    """Everything app.py needs to show one reply."""

    answer: str
    sources: list = field(default_factory=list)  # [{"title": ..., "url": ...}]
    kind: str = KIND_LLM
    error_kind: Optional[str] = None  # e.g. "rate_limit", "too_long"; None if fine


# ---- Small talk ------------------------------------------------------------
_FILLERS = {"ok", "okay", "great", "cool", "awesome", "perfect", "nice", "alright"}

_GREETING_OPENERS = (
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
)
_GREETING_SUFFIXES = ("", "there", "everyone", "team", "notion")

_THANKS = {
    "thanks", "thank you", "thanks a lot", "thanks so much", "thank you so much",
    "thank you very much", "many thanks", "thx", "ty",
}
_GOODBYES = {
    "bye", "bye bye", "goodbye", "good bye", "see you", "see you later",
    "see ya", "cya", "take care", "thats all", "that is all", "im done",
    "thanks bye", "thank you bye",
}


def _normalize(text):
    """Lowercase, drop apostrophes and punctuation, collapse spaces."""
    text = str(text).lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _is_greeting(normalized):
    for opener in _GREETING_OPENERS:
        if normalized == opener:
            return True
        if normalized.startswith(opener + " "):
            if normalized[len(opener) + 1:] in _GREETING_SUFFIXES:
                return True
    return False


def _strip_filler(normalized):
    """'ok thanks' -> 'thanks' (only removes one leading filler word)."""
    words = normalized.split()
    if len(words) > 1 and words[0] in _FILLERS:
        words = words[1:]
    return " ".join(words)


def _small_talk_kind(normalized):
    """Return a KIND_* for greetings/thanks/goodbye, otherwise None."""
    if _is_greeting(normalized):
        return KIND_GREETING
    core = _strip_filler(normalized)
    if core in _THANKS:
        return KIND_THANKS
    if core in _GOODBYES:
        return KIND_GOODBYE
    return None


# ---- Errors ----------------------------------------------------------------
# llm.py kind -> key used in prompts.ERROR_MESSAGES. Anything not listed
# (server, model_unavailable, ...) falls back to the generic message.
_ERROR_KEY = {"missing_key": "no_api_key"}

# Temporary Gemini problems: we can still answer from the knowledge base.
# Not included: missing_key / auth (a setup problem you should notice) and
# blocked (a safety block should not be bypassed).
_DEGRADED_KINDS = {
    llm.RATE_LIMIT, llm.TIMEOUT, llm.CONNECTION, llm.SERVER,
    llm.EMPTY, llm.MODEL_UNAVAILABLE, llm.UNKNOWN,
}


def _error_result(error_kind):
    key = _ERROR_KEY.get(error_kind, error_kind)
    return ChatResult(
        answer=prompts.get_error_message(key),
        sources=[],
        kind=KIND_ERROR,
        error_kind=error_kind,
    )


# ---- Helpers ---------------------------------------------------------------
def _get_previous_question(history):
    """
    The visitor's most recent earlier question, looking only at the last
    HISTORY_MESSAGES messages. `history` is a list like
    [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
    and must NOT include the message being answered right now.
    """
    recent = (history or [])[-config.HISTORY_MESSAGES:]
    for message in reversed(recent):
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return None


def _build_sources(matches):
    """Unique source links from the retrieved records, best match first."""
    sources = []
    seen = set()
    for match in matches:
        url = str(match.record.get("source_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"title": match.record.get("topic") or url, "url": url})
        if len(sources) >= config.MAX_SOURCES:
            break
    return sources


def _question_for_llm(question, previous_question, used_followup):
    """For a follow-up, tell Gemini what it is following up on."""
    if used_followup and previous_question:
        return (
            f"{question}\n"
            f'(This is a follow-up to the visitor\'s previous question: "{previous_question}")'
        )
    return question


# ---- The main entry point --------------------------------------------------
def answer_question(message, history=None, messages_used=0, records=None):
    """
    Answer one visitor message and return a ChatResult.

    message       - what the visitor just typed
    history       - earlier messages (see _get_previous_question); optional
    messages_used - how many messages the visitor has ALREADY sent in this
                    session, not counting this one (app.py keeps this count)
    records       - KB records; loaded from the knowledge base if None
                    (tests pass a small fake list)
    """
    text = str(message or "").strip()

    # 1. Basic input checks
    if not text:
        return ChatResult(prompts.NO_INFO_MESSAGE, [], KIND_FALLBACK)
    if messages_used >= config.MAX_MESSAGES_PER_SESSION:
        return _error_result("session_limit")
    if len(text) > config.MAX_MESSAGE_CHARS:
        return _error_result("too_long")

    # 2. Small talk (no knowledge base and no Gemini needed)
    small_talk = _small_talk_kind(_normalize(text))
    if small_talk == KIND_GREETING:
        return ChatResult(prompts.greeting_message(text), [], KIND_GREETING)
    if small_talk == KIND_THANKS:
        return ChatResult(prompts.THANKS_REPLY, [], KIND_THANKS)
    if small_talk == KIND_GOODBYE:
        return ChatResult(prompts.GOODBYE_REPLY, [], KIND_GOODBYE)

    # 3. Load verified knowledge. Without it we never call Gemini.
    if records is None:
        try:
            records = knowledge_base.get_records()
        except knowledge_base.KnowledgeBaseError:
            logger.warning("Knowledge base could not be loaded")
            return _error_result("kb_unavailable")
    if not records:
        return _error_result("kb_unavailable")

    # 4. Retrieve relevant records and build the grounded prompt
    previous_question = _get_previous_question(history)
    try:
        result = retriever.retrieve(
            text, records=records, previous_question=previous_question
        )
        if not result.found or not result.matches:
            return ChatResult(prompts.NO_INFO_MESSAGE, [], KIND_FALLBACK)
        sources = _build_sources(result.matches)
        user_message = prompts.build_user_message(
            _question_for_llm(text, previous_question, result.used_followup),
            [match.record for match in result.matches],
        )
    except Exception as exc:  # noqa: BLE001 - never show raw errors
        logger.warning("Retrieval/prompt step failed: %s", type(exc).__name__)
        return _error_result(llm.UNKNOWN)

    # 5. Ask Gemini
    try:
        answer = llm.generate_answer(user_message)
        return ChatResult(answer, sources, KIND_LLM)
    except llm.LLMError as exc:
        error_kind = exc.kind
    except Exception as exc:  # noqa: BLE001 - llm.py should not leak, but be safe
        logger.warning("Unexpected LLM failure: %s", type(exc).__name__)
        error_kind = llm.UNKNOWN

    # 6. Gemini failed. Fall back to the best verified KB answer if appropriate.
    top_record = result.matches[0].record
    top_answer = str(top_record.get("answer", "")).strip()
    if error_kind in _DEGRADED_KINDS and top_answer:
        return ChatResult(
            answer=f"{prompts.DEGRADED_PREFIX}\n\n{top_answer}",
            sources=_build_sources(result.matches[:1]),
            kind=KIND_DEGRADED,
            error_kind=error_kind,
        )
    return _error_result(error_kind)


if __name__ == "__main__":
    # Manual check:  python -m chatbot.chatbot "How much does the Plus plan cost?"
    # (A normal question makes one real Gemini call using the key from .env.)
    import sys

    demo_question = " ".join(sys.argv[1:]) or "What is Notion?"
    demo = answer_question(demo_question)
    print(f"[{demo.kind}] {demo.answer}")
    for source in demo.sources:
        print(f"  source: {source['title']} - {source['url']}")
    if demo.error_kind:
        print(f"  error_kind: {demo.error_kind}")