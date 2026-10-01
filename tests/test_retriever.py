"""
Tests for chatbot/retriever.py

Run from the project folder (the one that contains app.py):
    python -m pytest tests/test_retriever.py -v

These tests need NO API key and NO internet. They use the real
data/notion_kb.json, so they also prove that retrieval works on your data.
"""

import sys
from pathlib import Path

# Make "import chatbot" work no matter how pytest is started.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from chatbot import retriever
from chatbot.knowledge_base import get_records
from chatbot.retriever import content_words, is_pricing_question, retrieve, tokenize


@pytest.fixture(scope="module")
def records():
    return get_records()


def top_id(question, records, previous=None):
    """ID of the best record for a question (None if nothing relevant)."""
    result = retrieve(question, records, previous_question=previous)
    return result.matches[0].record["id"] if result.found else None


def ids(question, records, previous=None):
    result = retrieve(question, records, previous_question=previous)
    return [m.record["id"] for m in result.matches]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
class TestTextCleaning:
    def test_lowercase_and_punctuation(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_plurals_become_singular(self):
        assert tokenize("plans teamspaces companies") == ["plan", "teamspace", "company"]

    def test_words_ending_in_ss_us_is_are_not_broken(self):
        assert tokenize("business plus analysis") == ["business", "plus", "analysis"]

    def test_apostrophes(self):
        assert tokenize("Notion's mission") == ["notion", "mission"]
        assert tokenize("can't add") == ["cant", "add"]

    def test_stop_words_removed(self):
        words = content_words(tokenize("What is the price of the Plus plan?"))
        assert words == ["price", "plus", "plan"]


# ---------------------------------------------------------------------------
# Normal questions
# ---------------------------------------------------------------------------
class TestNormalQuestions:
    @pytest.mark.parametrize(
        "question, expected",
        [
            ("What is Notion?", "company-overview"),
            ("Tell me about Notion", "company-overview"),
            ("What is Notion's mission?", "company-mission"),
            ("Who is the CEO of Notion?", "company-leadership"),
            ("What products does Notion offer?", "products-overview"),
            ("What is Notion Calendar?", "products-calendar"),
            ("Can I use Notion as a company wiki?", "products-wikis"),
            ("Does Notion have an API?", "features-developer-tools"),
            ("Can I publish a Notion page as a website?", "features-sites-publishing"),
            ("Which apps does Notion integrate with?", "features-connections"),
            ("What can Notion AI do?", "ai-overview"),
            ("What are Custom Agents?", "ai-custom-agents"),
            ("Is my data used to train AI models?", "ai-data-privacy"),
            ("What is a teamspace?", "collab-teamspaces"),
            ("Can I use Notion for personal projects?", "use-case-personal"),
            ("Does Notion have templates?", "use-case-templates"),
            ("How do I cancel my subscription?", "faq-cancel"),
            ("Does Notion offer a student discount?", "faq-student-discount"),
            ("Can I get a refund?", "faq-refunds"),
            ("What is a block?", "faq-what-is-block"),
            ("When was Notion founded?", "limit-founding-funding"),
        ],
    )
    def test_best_record(self, question, expected, records):
        assert top_id(question, records) == expected

    def test_typing_style_does_not_matter(self, records):
        assert top_id("WHO IS THE CEO???", records) == "company-leadership"
        assert top_id("who is the ceo", records) == "company-leadership"

    def test_plural_and_singular_find_the_same_thing(self, records):
        assert top_id("Tell me about teamspace", records) == top_id("Tell me about teamspaces", records)

    def test_does_notion_support_x_is_not_a_support_question(self, records):
        # "support" is a verb here, so the Contact Support FAQ must not win.
        assert top_id("Does Notion support SSO?", records) != "faq-contact-support"
        assert top_id("How do I contact Notion support?", records) == "faq-contact-support"

    def test_normal_question_returns_at_most_default_top_k(self, records):
        result = retrieve("What can Notion AI do?", records)
        assert 1 <= len(result.matches) <= retriever.TOP_K_DEFAULT


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
class TestPricing:
    def test_pricing_intent_detection(self):
        assert is_pricing_question("How much does Notion cost?")
        assert is_pricing_question("What is the price of Plus?")
        assert is_pricing_question("Is Notion free?")
        assert is_pricing_question("Can I pay monthly?")
        assert not is_pricing_question("What is Notion?")
        assert not is_pricing_question("Who is the CEO?")

    def test_general_pricing_question(self, records):
        result = retrieve("How much does Notion cost?", records)
        assert result.found and result.pricing_intent
        assert "pricing-overview" in ids("How much does Notion cost?", records)

    def test_is_notion_free(self, records):
        assert top_id("Is Notion free?", records) == "pricing-free"

    def test_pricing_questions_may_use_more_records(self, records):
        result = retrieve("How much does Notion cost?", records)
        assert len(result.matches) <= retriever.TOP_K_PRICING

    def test_custom_domain_price(self, records):
        assert top_id("How much does it cost to use my own domain?", records) == "pricing-custom-domains"

    def test_custom_agents_price(self, records):
        assert top_id("How much do Custom Agents cost?", records) == "pricing-custom-agent-credits"

    def test_currency_question(self, records):
        assert top_id("Can I see the price in INR?", records) == "pricing-currency"

    def test_yearly_billing(self, records):
        assert "pricing-billing-options" in ids("Is it cheaper to pay yearly?", records)

    def test_pricing_boost_only_applies_to_pricing_categories(self, records):
        result = retrieve("How much does Notion cost?", records)
        for match in result.matches:
            if match.boost_score > 0:
                assert match.record["category"] in retriever.PRICING_CATEGORIES

    def test_pricing_words_alone_do_not_make_an_off_topic_question_relevant(self, records):
        assert top_id("How much does a pizza cost?", records) is None


# ---------------------------------------------------------------------------
# Plan matching
# ---------------------------------------------------------------------------
class TestPlanMatching:
    @pytest.mark.parametrize(
        "question, expected",
        [
            ("How much does the Plus plan cost?", "pricing-plus"),
            ("How much is the Business plan?", "pricing-business"),
            ("Enterprise price", "pricing-enterprise"),
            ("How much does the Free plan cost?", "pricing-free"),
        ],
    )
    def test_plan_price_record_is_first(self, question, expected, records):
        assert top_id(question, records) == expected

    def test_plan_names_are_detected(self):
        result = retrieve("How much do Plus and Business cost?")
        assert set(result.plans) == {"plus", "business"}

    def test_plan_price_record_is_always_included(self, records):
        assert "pricing-plus" in ids("How much does the Plus plan cost?", records)

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("Free vs Plus", "plan-diff-free-vs-plus"),
            ("What is the difference between Free and Plus?", "plan-diff-free-vs-plus"),
            ("Plus vs Business", "plan-diff-plus-vs-business"),
            ("Compare Plus and Business", "plan-diff-plus-vs-business"),
            ("Business versus Enterprise", "plan-diff-business-vs-enterprise"),
        ],
    )
    def test_plan_comparisons(self, question, expected, records):
        assert expected in ids(question, records)[:2]

    def test_comparison_questions_are_detected(self):
        assert retrieve("Free vs Plus").comparison_intent
        assert not retrieve("What is Notion?").comparison_intent

    def test_plan_matching_uses_contains(self):
        """A record whose plan is "Plus and Business" must match a question about "Plus"."""
        fake = [
            {"id": "x-both", "category": "pricing", "topic": "Shared price", "question": "Shared price?",
             "answer": "Plus and Business are priced per member.", "keywords": ["price", "pricing"],
             "source_url": "https://www.notion.com/pricing", "plan": "Plus and Business",
             "displayed_price": "$1", "billing_unit": "per member", "pricing_status": "displayed_on_page"},
            {"id": "x-enterprise", "category": "pricing", "topic": "Other price", "question": "Other price?",
             "answer": "Enterprise is custom.", "keywords": ["price", "pricing"],
             "source_url": "https://www.notion.com/pricing", "plan": "Enterprise",
             "displayed_price": "custom", "billing_unit": "n/a", "pricing_status": "verified"},
        ]
        result = retrieve("What is the price of Plus?", fake)
        scores = {m.record["id"]: m for m in result.matches}
        assert scores["x-both"].boost_score >= retriever.WEIGHT_PLAN_BOOST
        assert scores["x-enterprise"].boost_score < retriever.WEIGHT_PLAN_BOOST
        assert result.matches[0].record["id"] == "x-both"


# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------
class TestFollowUps:
    def test_what_about_business(self, records):
        previous = "How much does the Plus plan cost?"
        result = retrieve("What about Business?", records, previous_question=previous)
        assert result.used_followup
        assert result.pricing_intent          # inherited from the previous question
        assert result.matches[0].record["id"] == "pricing-business"

    def test_and_enterprise(self, records):
        assert top_id("And Enterprise?", records, previous="How much is Plus?") == "pricing-enterprise"

    def test_pronoun_followup_inherits_pricing_intent(self, records):
        result = retrieve("How much does it cost?", records, previous_question="What can Notion AI do?")
        assert result.used_followup and result.pricing_intent and result.found

    def test_followup_without_previous_question_is_a_normal_search(self, records):
        result = retrieve("What about Business?", records)
        assert not result.used_followup
        assert result.found

    def test_new_topic_ignores_previous_question(self, records):
        result = retrieve("What is a teamspace?", records, previous_question="How much does the Plus plan cost?")
        assert not result.used_followup
        assert result.matches[0].record["id"] == "collab-teamspaces"

    def test_long_question_is_not_treated_as_followup(self, records):
        long_question = "Can you explain how Notion handles permissions and access control for teams?"
        result = retrieve(long_question, records, previous_question="How much is Plus?")
        assert not result.used_followup

    def test_followup_uses_current_plan_not_previous_plan(self, records):
        result = retrieve("What about Business?", records, previous_question="How much does Plus cost?")
        assert result.plans == ["business"]


# ---------------------------------------------------------------------------
# Unrelated / unanswerable questions
# ---------------------------------------------------------------------------
class TestUnrelatedQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "What's the weather in Paris?",
            "Who won the world cup?",
            "What is the capital of France?",
            "Tell me a joke",
            "How do I bake a cake?",
            "How much does a pizza cost?",
            "Write python code to sort a list",
            "asdfgh",
            "",
            "   ",
            "?!?!",
            "Hi",
        ],
    )
    def test_nothing_relevant_found(self, question, records):
        result = retrieve(question, records)
        assert result.found is False
        assert result.matches == []
        assert result.records == []

    @pytest.mark.parametrize(
        "question", ["Does Notion work offline?", "Does Notion have a mobile app?"]
    )
    def test_notion_questions_the_kb_does_not_cover(self, question, records):
        # They mention Notion but no record really answers them,
        # so the chatbot should say "I don't have that information".
        assert retrieve(question, records).found is False


# ---------------------------------------------------------------------------
# Settings, structure and safety
# ---------------------------------------------------------------------------
class TestSettingsAndStructure:
    def test_results_are_sorted_best_first(self, records):
        scores = [m.score for m in retrieve("How much does Notion cost?", records).matches]
        assert scores == sorted(scores, reverse=True)

    def test_same_question_gives_same_result(self, records):
        assert ids("What can Notion AI do?", records) == ids("What can Notion AI do?", records)

    def test_top_k_can_be_overridden(self, records):
        assert len(retrieve("How much does Notion cost?", records, top_k=2).matches) <= 2

    def test_minimum_score_is_configurable(self, records, monkeypatch):
        assert retrieve("What is Notion?", records).found
        monkeypatch.setattr(retriever, "MIN_SCORE", 1000)
        assert retrieve("What is Notion?", records).found is False

    def test_relative_cutoff_removes_weak_records(self, records, monkeypatch):
        monkeypatch.setattr(retriever, "RELATIVE_SCORE_CUTOFF", 0.99)
        result = retrieve("How much does Notion cost?", records)
        assert len(result.matches) >= 1
        assert all(m.score >= result.top_score * 0.99 or m.record["category"] == "pricing"
                   for m in result.matches)

    def test_retriever_does_not_change_the_records(self, records):
        import copy
        before = copy.deepcopy(records)
        retrieve("How much does the Plus plan cost?", records)
        retrieve("What about Business?", records, previous_question="How much is Plus?")
        assert records == before

    def test_every_match_has_a_source_url(self, records):
        for match in retrieve("How much does Notion cost?", records).matches:
            assert match.record["source_url"].startswith("https://www.notion.com/")