"""
config.py - all settings for the chatbot in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "notion_kb.json"

# Load local .env file
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Gemini settings
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.1-flash-lite"

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 350

REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

TOP_K_DEFAULT = 4
TOP_K_PRICING = 6

MIN_SCORE = 3.0
RELATIVE_SCORE_CUTOFF = 0.4

WEIGHT_KEYWORD = 3.0
WEIGHT_TOPIC = 2.0
WEIGHT_ANSWER = 0.5
WEIGHT_PHRASE_BONUS = 2.0

WEIGHT_PRICING_BOOST = 3.0
WEIGHT_PLAN_BOOST = 4.0


# ---------------------------------------------------------------------------
# Conversation and display settings
# ---------------------------------------------------------------------------

HISTORY_MESSAGES = 6
MAX_SOURCES = 3
MAX_MESSAGES_PER_SESSION = 30
MAX_MESSAGE_CHARS = 500


# ---------------------------------------------------------------------------
# Gemini API key
# ---------------------------------------------------------------------------

def get_api_key():
    """Return the Gemini API key, or None if it is missing."""
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key or key == "your_gemini_api_key_here":
        return None

    return key


def has_api_key():
    """Return True if a usable Gemini API key is configured."""
    return get_api_key() is not None