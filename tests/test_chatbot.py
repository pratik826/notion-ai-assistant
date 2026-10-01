"""Tests for chatbot.py. No real Gemini calls: llm.generate_answer is mocked."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chatbot import chatbot as bot
from chatbot import config, knowledge_base, llm, prompts, retriever

FAKE_KEY = "FAKE-SECRET-KEY-123"
PLUS_Q = "How much does the Plus plan cost?"
BUSINESS_Q = "How much does the Business plan cost?"
HISTORY = [
    {"role": "user", "content": PLUS_Q},
    {"role": "assistant", "content": "Notion's pricing page currently displays Plus at ..."},
]
SOURCE_CODE = Path(bot.__file__).read_text(encoding="utf-8")


# ---------- helpers and fixtures ----------

@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Every test gets a fake Gemini. Real calls are impossible."""
    mock = MagicMock(return_value="Mocked Gemini answer")
    monkeypatch.setattr(bot.llm, "generate_answer", mock)
    return mock


@pytest.fixture
def spy_retrieve(monkeypatch):
    """Runs the real retriever but records how it was called."""
    calls = []
    real = retriever.retrieve

    def spy(question, **kwargs):
        calls.append({"question": question, **kwargs})
        return real(question, **kwargs)

    monkeypatch.setattr(bot.retriever, "retrieve", spy)
    return calls


def fake_record(number, url, answer="Verified answer"):
    return {
        "id": f"rec-{number}", "category": "faq", "topic": f"Topic {number}",
        "question": f"Question {number}?", "answer": answer,
        "keywords": ["x"], "source_url": url,
    }


def fake_result(records):
    matches = [retriever.Match(record=r, score=5.0, strong=True) for r in records]
    return retriever.RetrievalResult(matches=matches, found=True, top_score=5.0)


# ---------- small talk: Gemini must not be called ----------

def test_greeting_does_not_call_gemini(mock_llm):
    result = bot.answer_question("hello")
    assert result.kind == bot.KIND_GREETING
    assert result.sources == []
    mock_llm.assert_not_called()


@pytest.mark.parametrize("text", [
    "hi", "Hello!", "hey", "Good morning", "good afternoon", "Good evening!", "hi there",
])
def test_greeting_variants(mock_llm, text):
    result = bot.answer_question(text)
    assert result.kind == bot.KIND_GREETING
    assert result.answer == prompts.greeting_message(text)
    assert "Notion" in result.answer  # introduces Notion, not just "Hello"
    mock_llm.assert_not_called()


@pytest.mark.parametrize("text", ["thanks", "Thank you!", "ok thanks", "thanks a lot"])
def test_thanks_does_not_call_gemini(mock_llm, text):
    result = bot.answer_question(text)
    assert result.kind == bot.KIND_THANKS
    assert result.answer == prompts.THANKS_REPLY
    mock_llm.assert_not_called()


@pytest.mark.parametrize("text", ["bye", "Goodbye!", "see you later", "that's all"])
def test_goodbye_does_not_call_gemini(mock_llm, text):
    result = bot.answer_question(text)
    assert result.kind == bot.KIND_GOODBYE
    assert result.answer == prompts.GOODBYE_REPLY
    mock_llm.assert_not_called()


def test_greeting_followed_by_real_question_is_not_small_talk(mock_llm):
    result = bot.answer_question("hi, how much does the Plus plan cost?")
    assert result.kind == bot.KIND_LLM
    mock_llm.assert_called_once()


# ---------- normal questions ----------

def test_normal_question_calls_retriever_and_llm(spy_retrieve, mock_llm):
    result = bot.answer_question(PLUS_Q)
    assert len(spy_retrieve) == 1
    assert spy_retrieve[0]["question"] == PLUS_Q
    mock_llm.assert_called_once()
    assert result.kind == bot.KIND_LLM
    assert result.answer == "Mocked Gemini answer"
    assert result.error_kind is None


def test_gemini_message_is_grounded_and_has_no_urls(mock_llm):
    bot.answer_question(PLUS_Q)
    sent = mock_llm.call_args.args[0]
    assert "KNOWLEDGE BASE EXCERPTS" in sent
    assert PLUS_Q in sent
    for record in knowledge_base.get_records():
        assert record["source_url"] not in sent


def test_sources_come_from_kb_records(mock_llm):
    result = bot.answer_question(PLUS_Q)
    kb_urls = {r["source_url"] for r in knowledge_base.get_records()}
    assert result.sources
    assert len(result.sources) <= config.MAX_SOURCES
    for source in result.sources:
        assert set(source) == {"title", "url"}
        assert source["url"] in kb_urls


def test_sources_are_deduplicated_and_limited(monkeypatch):
    urls = ["https://n.com/a", "https://n.com/a", "https://n.com/b",
            "https://n.com/c", "https://n.com/d"]
    recs = [fake_record(i, url) for i, url in enumerate(urls)]
    monkeypatch.setattr(bot.retriever, "retrieve", lambda question, **kw: fake_result(recs))
    result = bot.answer_question("What are the plans?", records=recs)
    got = [s["url"] for s in result.sources]
    assert len(got) == len(set(got))
    assert got == list(dict.fromkeys(urls))[:config.MAX_SOURCES]


def test_records_without_url_are_skipped(monkeypatch):
    recs = [fake_record(1, ""), fake_record(2, "https://n.com/ok")]
    monkeypatch.setattr(bot.retriever, "retrieve", lambda question, **kw: fake_result(recs))
    result = bot.answer_question("What are the plans?", records=recs)
    assert [s["url"] for s in result.sources] == ["https://n.com/ok"]


# ---------- unrelated questions ----------

@pytest.mark.parametrize("question", [
    "What is the weather in Paris?",
    "Tell me a joke",
    "How much does a pizza cost?",
])
def test_unrelated_question_returns_no_info_without_gemini(mock_llm, question):
    result = bot.answer_question(question)
    assert result.kind == bot.KIND_FALLBACK
    assert result.answer == prompts.NO_INFO_MESSAGE
    assert result.sources == []
    mock_llm.assert_not_called()


# ---------- Gemini failures ----------

@pytest.mark.parametrize("kind, message_key", [
    ("missing_key", "no_api_key"),
    ("auth", "auth"),
    ("blocked", "blocked"),
])
def test_setup_and_blocked_errors_show_friendly_message(mock_llm, kind, message_key):
    mock_llm.side_effect = llm.LLMError(kind)
    result = bot.answer_question(PLUS_Q)
    assert result.kind == bot.KIND_ERROR
    assert result.error_kind == kind
    assert result.answer == prompts.get_error_message(message_key)
    assert result.sources == []


@pytest.mark.parametrize("kind", [
    "rate_limit", "timeout", "connection", "server", "empty", "model_unavailable", "unknown",
])
def test_degraded_answer_when_gemini_fails(mock_llm, kind):
    mock_llm.side_effect = llm.LLMError(kind)
    top = retriever.retrieve(PLUS_Q).matches[0].record
    result = bot.answer_question(PLUS_Q)
    assert result.kind == bot.KIND_DEGRADED
    assert result.error_kind == kind
    assert result.answer.startswith(prompts.DEGRADED_PREFIX)
    assert top["answer"] in result.answer
    assert [s["url"] for s in result.sources] == [top["source_url"]]


def test_unexpected_exception_is_hidden(mock_llm, caplog):
    mock_llm.side_effect = RuntimeError(f"boom {FAKE_KEY}")
    with caplog.at_level(logging.DEBUG):
        result = bot.answer_question(PLUS_Q)
    assert result.kind == bot.KIND_DEGRADED
    assert result.error_kind == llm.UNKNOWN
    assert FAKE_KEY not in result.answer
    assert FAKE_KEY not in repr(result)
    assert FAKE_KEY not in caplog.text


def test_chatbot_never_touches_api_key():
    assert "get_api_key" not in SOURCE_CODE
    assert "GEMINI_API_KEY" not in SOURCE_CODE


def test_chatbot_has_no_streamlit_code():
    lowered = SOURCE_CODE.lower()
    assert "import streamlit" not in lowered
    assert "from streamlit" not in lowered
    assert "session_state" not in lowered


# ---------- session limit, message length, empty input ----------

def test_session_limit_reached(mock_llm):
    result = bot.answer_question("hello", messages_used=config.MAX_MESSAGES_PER_SESSION)
    assert result.kind == bot.KIND_ERROR
    assert result.error_kind == "session_limit"
    assert result.answer == prompts.get_error_message("session_limit")
    mock_llm.assert_not_called()


def test_below_session_limit_still_works():
    result = bot.answer_question("hello", messages_used=config.MAX_MESSAGES_PER_SESSION - 1)
    assert result.kind == bot.KIND_GREETING


def test_too_long_message_rejected(mock_llm):
    result = bot.answer_question("a" * (config.MAX_MESSAGE_CHARS + 1))
    assert result.kind == bot.KIND_ERROR
    assert result.error_kind == "too_long"
    assert result.answer == prompts.get_error_message("too_long")
    mock_llm.assert_not_called()


def test_message_exactly_at_limit_is_accepted():
    result = bot.answer_question("a" * config.MAX_MESSAGE_CHARS)
    assert result.error_kind is None


def test_empty_message_returns_fallback_without_gemini(mock_llm):
    result = bot.answer_question("   ")
    assert result.kind == bot.KIND_FALLBACK
    assert result.answer == prompts.NO_INFO_MESSAGE
    mock_llm.assert_not_called()


# ---------- missing knowledge base ----------

def _break_kb(monkeypatch):
    def broken(*args, **kwargs):
        raise knowledge_base.KnowledgeBaseError("missing")
    monkeypatch.setattr(bot.knowledge_base, "get_records", broken)


def test_missing_knowledge_base(monkeypatch, mock_llm):
    _break_kb(monkeypatch)
    result = bot.answer_question(PLUS_Q)
    assert result.kind == bot.KIND_ERROR
    assert result.error_kind == "kb_unavailable"
    assert result.answer == prompts.get_error_message("kb_unavailable")
    mock_llm.assert_not_called()


def test_small_talk_works_without_knowledge_base(monkeypatch):
    _break_kb(monkeypatch)
    assert bot.answer_question("hello").kind == bot.KIND_GREETING


# ---------- conversation history ----------

def test_followup_passes_previous_question_to_retriever(spy_retrieve):
    bot.answer_question("What about Business?", history=HISTORY)
    assert spy_retrieve[0]["previous_question"] == PLUS_Q


def test_no_history_means_no_previous_question(spy_retrieve):
    bot.answer_question(PLUS_Q)
    assert spy_retrieve[0]["previous_question"] is None


def test_only_recent_history_is_used(spy_retrieve):
    history = [{"role": "user", "content": "Old question"}]
    history += [{"role": "assistant", "content": "filler"}] * config.HISTORY_MESSAGES
    bot.answer_question(PLUS_Q, history=history)
    assert spy_retrieve[0]["previous_question"] is None


def test_previous_question_is_the_last_user_message(spy_retrieve):
    history = [
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "second q"},
        {"role": "assistant", "content": "a2"},
    ]
    bot.answer_question(PLUS_Q, history=history)
    assert spy_retrieve[0]["previous_question"] == "second q"


def test_followup_context_reaches_gemini(mock_llm):
    result = bot.answer_question("What about Business?", history=HISTORY)
    assert result.kind == bot.KIND_LLM
    sent = mock_llm.call_args.args[0]
    assert PLUS_Q in sent
    assert "Business" in sent


def test_context_is_rebuilt_fresh_for_each_question(mock_llm):
    bot.answer_question(PLUS_Q)
    bot.answer_question(BUSINESS_Q)
    first, second = [call.args[0] for call in mock_llm.call_args_list]
    assert PLUS_Q in first
    assert BUSINESS_Q in second
    assert PLUS_Q not in second  # nothing carried over from the previous question


# ---------- result structure ----------

def test_result_structure():
    result = bot.answer_question(PLUS_Q)
    assert isinstance(result, bot.ChatResult)
    assert isinstance(result.answer, str) and result.answer
    assert isinstance(result.sources, list)
    assert result.kind in {
        bot.KIND_GREETING, bot.KIND_THANKS, bot.KIND_GOODBYE, bot.KIND_FALLBACK,
        bot.KIND_LLM, bot.KIND_DEGRADED, bot.KIND_ERROR,
    }
    assert result.error_kind is None
    hello = bot.answer_question("hello")
    assert hello.sources == [] and hello.error_kind is None