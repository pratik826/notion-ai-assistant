"""
retriever.py - finds the knowledge-base records that best match a question.

Pure Python: no Gemini, no OpenAI, no Streamlit, no extra packages.
Same question in -> same records out (it is deterministic), so it is easy
to test and to explain.

How it works, in one paragraph:
  1. Clean the question into a list of simple words (lowercase, no stop
     words, plurals made singular).
  2. Give every record a score: matches in its "keywords" count most,
     then "topic"/"question", then "answer".
  3. Add boosts for pricing questions and for plan names (Free, Plus ...).
  4. Keep the best few records. If even the best score is too low, say
     "nothing relevant found" so the chatbot can answer honestly.
"""

import re
from dataclasses import dataclass, field

from chatbot import config
from chatbot.knowledge_base import get_records


# ---------------------------------------------------------------------------
# Settings (read from config.py; the defaults are used if a name is missing)
# ---------------------------------------------------------------------------
def _setting(name, default):
    return getattr(config, name, default)


TOP_K_DEFAULT = _setting("TOP_K_DEFAULT", 4)
TOP_K_PRICING = _setting("TOP_K_PRICING", 6)
MIN_SCORE = _setting("MIN_SCORE", 3.0)
RELATIVE_SCORE_CUTOFF = _setting("RELATIVE_SCORE_CUTOFF", 0.4)
WEIGHT_KEYWORD = _setting("WEIGHT_KEYWORD", 3.0)
WEIGHT_TOPIC = _setting("WEIGHT_TOPIC", 2.0)
WEIGHT_ANSWER = _setting("WEIGHT_ANSWER", 0.5)
WEIGHT_PHRASE_BONUS = _setting("WEIGHT_PHRASE_BONUS", 2.0)
WEIGHT_PRICING_BOOST = _setting("WEIGHT_PRICING_BOOST", 3.0)
WEIGHT_PLAN_BOOST = _setting("WEIGHT_PLAN_BOOST", 4.0)
FOLLOWUP_WEIGHT = _setting("FOLLOWUP_WEIGHT", 0.5)  # how much the previous question counts

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------
STOP_WORDS_RAW = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall", "may", "might",
    "i", "me", "my", "you", "your", "we", "our", "us", "he", "she", "they", "them", "their",
    "it", "its", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "into", "about",
    "and", "or", "but", "if", "so", "then", "than",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "there", "any", "have", "has", "had", "get", "got", "tell", "please", "just", "also",
    "s", "t", "im", "ive", "some", "more", "very", "really", "like", "want", "need",
    "between", "give", "show", "know", "vs", "versus",
}

# Words that mean "the visitor is asking about money".
PRICING_WORDS = {
    "price", "pricing", "cost", "pay", "paid", "billing", "billed", "charge", "charged",
    "fee", "subscription", "cheap", "cheaper", "expensive", "afford", "affordable",
    "discount", "monthly", "yearly", "annual", "annually", "free", "refund",
}
PRICING_PHRASES = ["how much", "per month", "per member", "per year", "per user"]

# Words that mean "the visitor is comparing things" (we then allow more records).
COMPARISON_WORDS = {"vs", "versus", "compare", "comparison", "difference", "differ", "better", "cheaper"}

# Used to spot "does Notion support ...?" (support = verb, not customer support).
VERB_HELPERS = {"doe", "do", "can", "could", "will", "would"}
VERB_SUBJECTS = {"notion", "it", "you", "they"}

# Plan names we look for in the question and in a record's "plan" field.
PLAN_NAMES = ["free", "plus", "business", "enterprise"]

# Categories that get the pricing boost.
PRICING_CATEGORIES = {"pricing", "plan_differences"}

# Words so common in this knowledge base that they should count very little.
GENERIC_WORDS = {"notion"}
GENERIC_MULTIPLIER = 0.25

# Phrases that usually start a short follow-up message ("What about Business?").
FOLLOWUP_OPENERS = ("and ", "what about", "how about", "what if", "also ", "ok ", "okay ", "then ", "so ")
FOLLOWUP_PRONOUNS = {"it", "its", "that", "this", "they", "them", "those", "these"}
FOLLOWUP_MAX_WORDS = 5  # follow-ups are short; longer messages are treated as new questions


# ---------------------------------------------------------------------------
# Step 1: cleaning text
# ---------------------------------------------------------------------------
def _singular(word):
    """Very simple plural -> singular: 'plans' -> 'plan', 'companies' -> 'company'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text):
    """Turn text into a list of lowercase, singular words (stop words kept)."""
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"'s\b", "", text)   # notion's -> notion
    text = text.replace("'", "")       # can't -> cant
    return [_singular(word) for word in re.findall(r"[a-z0-9]+", text)]


STOP_WORDS = {_singular(w) for w in STOP_WORDS_RAW}


def content_words(tokens):
    """Remove stop words, keeping only the words that carry meaning."""
    return [t for t in tokens if t not in STOP_WORDS]


def _word_weight(word):
    return GENERIC_MULTIPLIER if word in GENERIC_WORDS else 1.0


# ---------------------------------------------------------------------------
# Step 2: understanding the question
# ---------------------------------------------------------------------------
@dataclass
class ParsedQuery:
    raw: str
    tokens: list          # all words, stop words kept (used for phrase matching)
    words: list           # meaningful words only
    pricing_intent: bool
    comparison_intent: bool
    plans: list           # plan names mentioned, e.g. ["plus"]


def _drop_verb_support(tokens):
    """In "Does Notion support SSO?" the word "support" is a verb, not a request
    for customer support. Remove it in patterns like "does notion support",
    "can it support", "do you support" so it does not match the Support FAQ."""
    kept = []
    for i, token in enumerate(tokens):
        if (token == "support" and i >= 2
                and tokens[i - 2] in VERB_HELPERS and tokens[i - 1] in VERB_SUBJECTS):
            continue
        kept.append(token)
    return kept


def parse_query(text):
    tokens = tokenize(text)
    words = content_words(_drop_verb_support(tokens))
    joined = " ".join(tokens)
    pricing = bool(set(words) & {_singular(w) for w in PRICING_WORDS}) or any(
        p in joined for p in PRICING_PHRASES
    )
    comparison = bool(set(tokens) & {_singular(w) for w in COMPARISON_WORDS})
    plans = [p for p in PLAN_NAMES if p in words]
    return ParsedQuery(str(text), tokens, words, pricing, comparison, plans)


def is_pricing_question(text):
    """True if the text looks like a question about price/cost/billing."""
    return parse_query(text).pricing_intent


# ---------------------------------------------------------------------------
# Step 3: scoring one record
# ---------------------------------------------------------------------------
def _contains_phrase(tokens, phrase):
    """True if `phrase` (a list of words) appears in order, side by side, in `tokens`."""
    n = len(phrase)
    return n > 0 and any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


def _keyword_score(query, keywords):
    """Score the query against a record's keyword list.

    Returns (score, strong). `strong` is True when at least one keyword was
    matched properly (the whole phrase, a whole single word, or all the
    phrase's words). Loose partial matches alone are not "strong".
    """
    total = 0.0
    strong = False
    generic_awarded = False  # the word "notion" is in many keywords; count it once only
    query_words = set(query.words)
    query_subject = [w for w in query.words if w not in GENERIC_WORDS]

    for keyword in keywords:
        k_tokens = tokenize(keyword)
        k_words = content_words(k_tokens)
        generic_only = bool(k_words) and all(w in GENERIC_WORDS for w in k_words)

        if len(k_tokens) > 1 and _contains_phrase(query.tokens, k_tokens):
            # The whole phrase was typed, e.g. "custom domain".
            if generic_only and query_subject:
                # "what is notion" says nothing specific. If the question has
                # other topic words ("...mission?"), it counts very little.
                if not generic_awarded:
                    total += WEIGHT_KEYWORD * GENERIC_MULTIPLIER
                    generic_awarded = True
            else:
                total += WEIGHT_KEYWORD + WEIGHT_PHRASE_BONUS
                strong = True
        elif not k_words:
            continue  # keyword made only of stop words: ignore
        elif generic_only:
            if k_words[0] in query_words and not generic_awarded:
                total += WEIGHT_KEYWORD * GENERIC_MULTIPLIER
                generic_awarded = True
        elif len(k_words) == 1:
            if k_words[0] in query_words:
                total += WEIGHT_KEYWORD
                strong = True
        else:
            matched = [w for w in k_words if w in query_words and w not in GENERIC_WORDS]
            if len(matched) == len(k_words):
                # All the phrase's words are present, just not side by side.
                total += WEIGHT_KEYWORD + WEIGHT_PHRASE_BONUS / 2
                strong = True
            elif matched:
                # Only some of the words matched: small partial credit.
                total += WEIGHT_KEYWORD * 0.5 * len(matched) / len(k_words)
    return total, strong


def _text_score(query, record):
    """Score the query against one record (before pricing/plan boosts).

    Returns (keyword_score, topic_score, answer_score, strong).
    """
    keyword_score, strong = _keyword_score(query, record["keywords"])

    topic_words = set(content_words(tokenize(record["topic"] + " " + record["question"])))
    answer_words = set(content_words(tokenize(record["answer"])))

    topic_score = 0.0
    answer_score = 0.0
    for word in set(query.words):
        weight = _word_weight(word)
        if word in topic_words:
            topic_score += WEIGHT_TOPIC * weight
        if word in answer_words:
            answer_score += WEIGHT_ANSWER * weight
    return keyword_score, topic_score, answer_score, strong


# ---------------------------------------------------------------------------
# Step 4: the result object
# ---------------------------------------------------------------------------
@dataclass
class Match:
    record: dict
    score: float
    keyword_score: float = 0.0
    topic_score: float = 0.0
    answer_score: float = 0.0
    boost_score: float = 0.0
    strong: bool = False     # True if a keyword was matched properly


@dataclass
class RetrievalResult:
    matches: list = field(default_factory=list)  # best first
    found: bool = False            # False = nothing relevant (use the fallback reply)
    top_score: float = 0.0
    pricing_intent: bool = False
    comparison_intent: bool = False
    plans: list = field(default_factory=list)
    used_followup: bool = False

    @property
    def records(self):
        return [m.record for m in self.matches]


def _looks_like_followup(query, previous):
    """Is this a short follow-up that depends on the previous question?"""
    if previous is None or len(query.words) > FOLLOWUP_MAX_WORDS:
        return False
    text = query.raw.strip().lower()
    if text.startswith(FOLLOWUP_OPENERS):
        return True
    return bool(set(query.tokens) & FOLLOWUP_PRONOUNS)


def _knowledge_base_vocabulary(records):
    """Every meaningful word that appears anywhere in the knowledge base."""
    vocabulary = set()
    for record in records:
        text = " ".join([record["topic"], record["question"], record["answer"]] + record["keywords"])
        vocabulary.update(content_words(tokenize(text)))
    return vocabulary


def _is_off_topic(query, records):
    """True if the question never mentions Notion AND none of its subject words
    exist anywhere in the knowledge base (e.g. "How much does a pizza cost?").

    Pricing words like "cost" are ignored here, because they would otherwise
    make every "how much is ...?" question look like a Notion question.
    """
    if "notion" in query.words:
        return False
    ignore = {_singular(w) for w in PRICING_WORDS} | GENERIC_WORDS | {"much", "many"}
    subject = [w for w in query.words if w not in ignore]
    if not subject:
        return False  # e.g. "How much does it cost?" -> let it through
    return not (set(subject) & _knowledge_base_vocabulary(records))


# ---------------------------------------------------------------------------
# Step 5: the main function
# ---------------------------------------------------------------------------
def retrieve(question, records=None, previous_question=None, top_k=None):
    """Find the records that best answer `question`.

    question          - what the visitor just typed
    records           - list of KB records (loaded from notion_kb.json if None)
    previous_question - the visitor's previous question (used for short follow-ups)
    top_k             - override how many records to return

    Returns a RetrievalResult. If result.found is False, nothing relevant
    was found and the chatbot should use its "I don't have that" reply.
    """
    if records is None:
        records = get_records()

    query = parse_query(question)
    previous = parse_query(previous_question) if previous_question else None
    use_followup = _looks_like_followup(query, previous)

    # Combine what we learned from this question (and the previous one, if follow-up).
    pricing_intent = query.pricing_intent or (use_followup and previous.pricing_intent)
    comparison_intent = query.comparison_intent or (use_followup and previous.comparison_intent)
    plans = query.plans or (previous.plans if use_followup else [])

    if not query.words and not use_followup:
        return RetrievalResult(pricing_intent=pricing_intent)
    if not use_followup and _is_off_topic(query, records):
        return RetrievalResult(pricing_intent=pricing_intent)

    # Score every record.
    scored = []
    for record in records:
        k, t, a, strong = _text_score(query, record)
        if use_followup:
            pk, pt, pa, pstrong = _text_score(previous, record)
            k, t, a = k + FOLLOWUP_WEIGHT * pk, t + FOLLOWUP_WEIGHT * pt, a + FOLLOWUP_WEIGHT * pa
            strong = strong or pstrong
        base = k + t + a
        if base <= 0:
            continue  # never boost a record that has no connection to the question

        boost = 0.0
        if pricing_intent and record["category"] in PRICING_CATEGORIES:
            boost += WEIGHT_PRICING_BOOST
        if pricing_intent and plans:
            record_plan = str(record.get("plan", "")).lower()
            if any(p in record_plan for p in plans):  # "contains" matching
                boost += WEIGHT_PLAN_BOOST

        scored.append(Match(record, base + boost, k, t, a, boost, strong))

    # Best first (ties keep the original file order).
    scored.sort(key=lambda m: m.score, reverse=True)

    result = RetrievalResult(
        pricing_intent=pricing_intent,
        comparison_intent=comparison_intent,
        plans=plans,
        used_followup=use_followup,
    )
    if not scored:
        return result
    result.top_score = scored[0].score
    # Relevant only if some record matched a keyword properly AND scored high enough.
    best_strong = max((m.score for m in scored if m.strong), default=0.0)
    if best_strong < MIN_SCORE:
        return result  # nothing relevant enough

    # Decide how many records to keep.
    if top_k is None:
        top_k = TOP_K_PRICING if (pricing_intent or comparison_intent) else TOP_K_DEFAULT
    cutoff = result.top_score * RELATIVE_SCORE_CUTOFF
    selected = [m for m in scored if m.score >= cutoff][:top_k]

    # For "How much is Plus?" always include the Plus price record itself.
    if pricing_intent and plans:
        for match in scored:
            record = match.record
            is_plan_record = record["category"] == "pricing" and str(record.get("plan", "")).lower() in plans
            if is_plan_record and match not in selected:
                selected.append(match)
        while len(selected) > top_k:
            removable = [m for m in selected if not (
                m.record["category"] == "pricing" and str(m.record.get("plan", "")).lower() in plans)]
            if not removable:
                break
            selected.remove(min(removable, key=lambda m: m.score))
        selected.sort(key=lambda m: m.score, reverse=True)

    result.matches = selected
    result.found = True
    return result


# ---------------------------------------------------------------------------
# Try it from the terminal:
#   python -m chatbot.retriever "How much is the Plus plan?"
#   python -m chatbot.retriever "What about Business?" "How much is Plus?"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m chatbot.retriever "your question" ["previous question"]')
        sys.exit(1)

    res = retrieve(sys.argv[1], previous_question=sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"Question: {sys.argv[1]}")
    print(f"Pricing intent: {res.pricing_intent} | Plans: {res.plans} | Follow-up: {res.used_followup}")
    print(f"Found: {res.found} | Top score: {res.top_score:.2f}")
    for m in res.matches:
        print(f"  {m.score:6.2f}  {m.record['id']:<32} "
              f"(kw {m.keyword_score:.1f}, topic {m.topic_score:.1f}, "
              f"ans {m.answer_score:.1f}, boost {m.boost_score:.1f})")