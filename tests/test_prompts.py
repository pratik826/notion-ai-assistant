"""
Tests for chatbot/prompts.py

Run from the project folder (the one that contains app.py):
    python -m pytest tests/test_prompts.py -v

No API key and no internet needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from chatbot import prompts
from chatbot.knowledge_base import get_records
from chatbot.retriever import retrieve


@pytest.fixture(scope="module")
def records():
    return get_records()


def record_by_id(records, record_id):
    return next(r for r in records if r["id"] == record_id)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
class TestSystemPrompt:
    def test_contains_the_required_pricing_wording(self):
        assert (
            "Notion's pricing page currently displays Plus at $10 per member/month. "
            "For the exact current monthly or yearly billing price, "
            "please check Notion's official pricing page."
        ) in prompts.SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "rule",
        [
            "ONLY the facts",                 # grounding
            "don't have verified information",  # unavailable info
            "Never reveal",                   # no prompt / key leaks
            "Do not calculate",               # no price maths
            "Keep plans and products separate",
            "DISPLAYED ON PAGE",
            "VERIFIED",
            "Enterprise has custom pricing",
            "not about Notion",               # off-topic handling
            "Do not write web addresses",     # sources come from the website
        ],
    )
    def test_key_rules_are_present(self, rule):
        assert rule in prompts.SYSTEM_PROMPT

    def test_prompt_does_not_name_an_ai_provider(self):
        text = prompts.SYSTEM_PROMPT.lower()
        for name in ["gemini", "openai", "gpt", "google"]:
            assert name not in text


# ---------------------------------------------------------------------------
# Building the context
# ---------------------------------------------------------------------------
class TestContext:
    def test_pricing_record_shows_status_and_price_fields(self, records):
        block = prompts.format_record(record_by_id(records, "pricing-plus"), 1)
        assert "[Excerpt 1]" in block
        assert "Plan: Plus" in block
        assert "Displayed price: $10" in block
        assert "Billing unit: per member/month" in block
        assert "DISPLAYED ON PAGE" in block
        assert "VERIFIED" not in block

    def test_verified_pricing_record_is_labelled_verified(self, records):
        block = prompts.format_record(record_by_id(records, "pricing-custom-domains"), 2)
        assert "VERIFIED" in block
        assert "DISPLAYED ON PAGE" not in block

    def test_non_pricing_record_has_no_pricing_lines(self, records):
        block = prompts.format_record(record_by_id(records, "company-overview"), 1)
        for label in ["Plan:", "Displayed price:", "Billing unit:", "Pricing status:"]:
            assert label not in block

    def test_unknown_pricing_status_is_treated_cautiously(self):
        fake = {"topic": "t", "question": "q", "answer": "a", "pricing_status": "something-new",
                "plan": "Plus", "displayed_price": "$1", "billing_unit": "x"}
        assert "UNVERIFIED" in prompts.format_record(fake, 1)

    def test_context_never_contains_urls_ids_or_keywords(self, records):
        # The AI must not see URLs, so it cannot invent or repeat them.
        context = prompts.format_context(records)  # all 67 records
        assert "http" not in context
        for record in records:
            assert record["id"] not in context
            assert record["source_url"] not in context

    def test_every_real_record_can_be_formatted(self, records):
        for number, record in enumerate(records, start=1):
            block = prompts.format_record(record, number)
            assert record["answer"] in block

    def test_user_message_contains_question_and_excerpts(self, records):
        question = "How much does the Plus plan cost?"
        message = prompts.build_user_message(question, retrieve(question, records).records)
        assert "KNOWLEDGE BASE EXCERPTS" in message
        assert "VISITOR QUESTION:\n" + question in message
        assert "Plus plan price" in message

    def test_empty_records_are_handled(self):
        message = prompts.build_user_message("Anything?", [])
        assert "No excerpts were found" in message


# ---------------------------------------------------------------------------
# Greeting and small talk
# ---------------------------------------------------------------------------
class TestGreeting:
    def test_greeting_is_more_than_just_hello(self):
        text = prompts.greeting_message("Hi")
        assert text.startswith("Hi there!")
        assert "Welcome to Notion" in text
        assert "pricing" in text and "Notion AI" in text
        assert len(text) > 150

    @pytest.mark.parametrize(
        "typed, expected_start",
        [
            ("hello", "Hello!"),
            ("Good morning", "Good morning!"),
            ("good afternoon!", "Good afternoon!"),
            ("Good evening", "Good evening!"),
            ("hey", "Hi there!"),
            ("", "Hi there!"),
        ],
    )
    def test_greeting_mirrors_the_visitor(self, typed, expected_start):
        assert prompts.greeting_message(typed).startswith(expected_start)

    def test_greeting_facts_come_from_the_knowledge_base(self, records):
        overview = record_by_id(records, "company-overview")["answer"]
        for product in ["Notion AI", "Docs", "Projects", "Wikis", "Notion Calendar"]:
            assert product in prompts.GREETING_BODY
            assert product in overview
        assert "all-in-one workspace" in overview
        assert "all-in-one workspace" in prompts.GREETING_BODY


# ---------------------------------------------------------------------------
# Fallback, errors and suggestions
# ---------------------------------------------------------------------------
class TestMessages:
    def test_no_info_message_does_not_invent_anything(self):
        assert "don't have verified information" in prompts.NO_INFO_MESSAGE
        assert "http" not in prompts.NO_INFO_MESSAGE

    @pytest.mark.parametrize("kind", list(prompts.ERROR_MESSAGES))
    def test_error_messages_are_friendly_and_leak_nothing(self, kind):
        message = prompts.get_error_message(kind)
        assert message.strip()
        for forbidden in ["api key", "api_key", "gemini", "openai", "traceback", "exception",
                          "429", "401", "env", "secret", "token"]:
            assert forbidden not in message.lower()

    def test_unknown_error_kind_falls_back_to_generic_message(self):
        assert prompts.get_error_message("something-else") == prompts.ERROR_MESSAGES["unknown"]

    def test_all_the_error_kinds_we_plan_to_use_exist(self):
        for kind in ["no_api_key", "auth", "rate_limit", "timeout", "connection", "empty",
                     "blocked", "kb_unavailable", "too_long", "session_limit", "unknown"]:
            assert kind in prompts.ERROR_MESSAGES


class TestSuggestedQuestions:
    def test_there_are_a_few_suggestions(self):
        assert 4 <= len(prompts.SUGGESTED_QUESTIONS) <= 8

    @pytest.mark.parametrize("question", prompts.SUGGESTED_QUESTIONS)
    def test_retriever_can_answer_every_suggestion(self, question, records):
        result = retrieve(question, records)
        assert result.found, f"Suggested question finds nothing: {question}"

    def test_suggestions_have_no_duplicates(self):
        assert len(set(prompts.SUGGESTED_QUESTIONS)) == len(prompts.SUGGESTED_QUESTIONS)