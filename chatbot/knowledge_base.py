"""
knowledge_base.py - loads and validates data/notion_kb.json.

This file only knows about the JSON file and its structure.
It does not search, build prompts, or call Gemini.
"""

import json
from functools import lru_cache
from pathlib import Path

from chatbot import config


# Every record must contain these fields.
REQUIRED_FIELDS = [
    "id",
    "category",
    "topic",
    "question",
    "answer",
    "keywords",
    "source_url",
]

# Pricing records need these additional fields.
PRICING_FIELDS = [
    "plan",
    "displayed_price",
    "billing_unit",
    "pricing_status",
]

# These fields must contain non-empty text.
TEXT_FIELDS = [
    "id",
    "category",
    "topic",
    "question",
    "answer",
    "source_url",
]


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base is missing or invalid."""


def _check_records(records):
    """
    Validate every record in the knowledge base.

    Returns:
        A list of problems.
        An empty list means everything is valid.
    """

    problems = []
    seen_ids = set()

    for position, record in enumerate(records, start=1):

        # Record must be a dictionary.
        if not isinstance(record, dict):
            problems.append(
                f"Record #{position} is not a JSON object."
            )
            continue

        label = (
            f"Record #{position} "
            f"(id: {record.get('id', 'unknown')})"
        )

        # Check required fields.
        missing = [
            field
            for field in REQUIRED_FIELDS
            if field not in record
        ]

        if missing:
            problems.append(
                f"{label} is missing: {', '.join(missing)}"
            )
            continue

        # Check text fields.
        for field in TEXT_FIELDS:
            value = record[field]

            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f"{label}: '{field}' must be non-empty text."
                )

        # Check keywords.
        keywords = record["keywords"]

        if (
            not isinstance(keywords, list)
            or not keywords
            or not all(isinstance(k, str) for k in keywords)
        ):
            problems.append(
                f"{label}: 'keywords' must be a "
                "non-empty list of text."
            )

        # Check unique IDs.
        if record["id"] in seen_ids:
            problems.append(
                f"{label}: duplicate id."
            )

        seen_ids.add(record["id"])

        # Pricing records need extra fields.
        if record["category"] == "pricing":

            missing_pricing = [
                field
                for field in PRICING_FIELDS
                if field not in record
            ]

            if missing_pricing:
                problems.append(
                    f"{label} is missing pricing fields: "
                    f"{', '.join(missing_pricing)}"
                )

    return problems


@lru_cache(maxsize=None)
def _load_from_disk(path_text):
    """
    Read and validate the knowledge base.

    The result is cached so the JSON file is not repeatedly
    read from disk during the application session.
    """

    path = Path(path_text)

    # Check that the file exists.
    if not path.is_file():
        raise KnowledgeBaseError(
            f"Knowledge base file not found: {path}\n"
            "Make sure 'notion_kb.json' is inside the 'data' folder."
        )

    # Load JSON.
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise KnowledgeBaseError(
            "The knowledge base is not valid JSON "
            f"(problem near line {error.lineno}, "
            f"column {error.colno}): {error.msg}"
        ) from error

    except OSError as error:
        raise KnowledgeBaseError(
            f"Could not read the knowledge base file: {error}"
        ) from error

    # Check top-level structure.
    if not isinstance(data, dict):
        raise KnowledgeBaseError(
            "The knowledge base must be a JSON object "
            "with 'meta' and 'records'."
        )

    if "records" not in data:
        raise KnowledgeBaseError(
            "The knowledge base has no 'records' key."
        )

    if not isinstance(data["records"], list):
        raise KnowledgeBaseError(
            "'records' must be a list."
        )

    if len(data["records"]) == 0:
        raise KnowledgeBaseError(
            "'records' is empty - there is nothing "
            "for the chatbot to use."
        )

    # Validate every record.
    problems = _check_records(data["records"])

    if problems:
        shown = "\n - ".join(problems[:10])

        if len(problems) > 10:
            extra = f"\n ...and {len(problems) - 10} more."
        else:
            extra = ""

        raise KnowledgeBaseError(
            f"The knowledge base has {len(problems)} problem(s):\n"
            f" - {shown}{extra}"
        )

    # Read metadata if available.
    meta = data.get("meta", {})

    if not isinstance(meta, dict):
        meta = {}

    return {
        "meta": meta,
        "records": data["records"],
    }


def load_knowledge_base(path=None):
    """
    Load the knowledge base.

    Returns:
        {
            "meta": {...},
            "records": [...]
        }

    Raises:
        KnowledgeBaseError if the file is invalid.
    """

    return _load_from_disk(
        str(path or config.KB_PATH)
    )


def get_records(path=None):
    """Return only the list of knowledge-base records."""

    return load_knowledge_base(path)["records"]


def clear_cache():
    """Clear the cached knowledge base."""

    _load_from_disk.cache_clear()


# Allows us to test the knowledge base directly with:
# python -m chatbot.knowledge_base

if __name__ == "__main__":

    try:
        kb = load_knowledge_base()

    except KnowledgeBaseError as error:

        print("KNOWLEDGE BASE ERROR:")
        print(error)

    else:

        records = kb["records"]

        counts = {}

        for record in records:
            category = record["category"]
            counts[category] = counts.get(category, 0) + 1

        print(
            f"Loaded {len(records)} records "
            f"from {config.KB_PATH}"
        )

        for category, count in sorted(counts.items()):
            print(f"  {category}: {count}")

        print("Meta:", kb["meta"])

        print(
            "Cached (same object on 2nd load):",
            load_knowledge_base() is kb,
        )