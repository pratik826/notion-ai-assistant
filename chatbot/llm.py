"""
llm.py - the ONLY file that talks to the Gemini API.

Flow: chatbot.py builds the user message (via prompts.py) and calls
generate_answer(). This file sends it to Gemini and returns plain text.

If anything goes wrong it raises LLMError with a short, safe `kind`
(for example "rate_limit"). chatbot.py turns that kind into a
friendly message for the visitor.

This file has no retrieval, no UI and no business logic.
"""

import logging
import time

import httpx
from google import genai
from google.genai import errors, types

from chatbot import config
from chatbot.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Error kinds that chatbot.py can map to visitor-friendly messages.
MISSING_KEY = "missing_key"
AUTH = "auth"
RATE_LIMIT = "rate_limit"
TIMEOUT = "timeout"
CONNECTION = "connection"
BLOCKED = "blocked"
EMPTY = "empty"
MODEL_UNAVAILABLE = "model_unavailable"
SERVER = "server"
UNKNOWN = "unknown"

# Only temporary problems are worth retrying.
RETRYABLE_KINDS = {RATE_LIMIT, TIMEOUT, CONNECTION, SERVER}

# Safe, generic text. We NEVER put the raw exception text in here,
# because it could contain sensitive details.
_SAFE_MESSAGES = {
    MISSING_KEY: "The Gemini API key is not configured.",
    AUTH: "The Gemini API key was rejected.",
    RATE_LIMIT: "The Gemini API rate limit was reached.",
    TIMEOUT: "The Gemini API request timed out.",
    CONNECTION: "Could not connect to the Gemini API.",
    BLOCKED: "Gemini blocked this response.",
    EMPTY: "Gemini returned an empty response.",
    MODEL_UNAVAILABLE: "The configured Gemini model is not available.",
    SERVER: "The Gemini service had a temporary problem.",
    UNKNOWN: "An unexpected error occurred while contacting Gemini.",
}


class LLMError(Exception):
    """Raised for any Gemini problem. `kind` says what went wrong."""

    def __init__(self, kind):
        self.kind = kind
        super().__init__(_SAFE_MESSAGES.get(kind, _SAFE_MESSAGES[UNKNOWN]))


def _thinking_config():
    """
    max_output_tokens also counts "thinking" tokens. With a small limit
    (350) the model could spend it all thinking and return nothing,
    so we switch thinking off (2.5) or to the lowest level (3.x).
    """
    if "2.5" in config.GEMINI_MODEL:
        return types.ThinkingConfig(thinking_budget=0)
    return types.ThinkingConfig(thinking_level="minimal")


def _create_client(api_key):
    """Create the Gemini client. The timeout is in MILLISECONDS."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_SECONDS * 1000),
    )


def _classify_error(exc):
    """Turn any exception into one of our error kinds."""
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or "").lower()
        if code in (401, 403) or (code == 400 and "api key" in message):
            return AUTH
        if code == 429:
            return RATE_LIMIT
        if code in (408, 504):
            return TIMEOUT
        if code == 404:
            return MODEL_UNAVAILABLE
        if isinstance(code, int) and code >= 500:
            return SERVER
        return UNKNOWN
    # httpx.TimeoutException must be checked before TransportError
    # because it is a more specific kind of TransportError.
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return TIMEOUT
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return CONNECTION
    return UNKNOWN


def _extract_text(response):
    """Return the answer text, or raise LLMError if blocked or empty."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMError(BLOCKED)

    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        reason = str(getattr(candidates[0], "finish_reason", "")).upper()
        if any(word in reason for word in ("SAFETY", "BLOCK", "PROHIBITED", "RECITATION")):
            raise LLMError(BLOCKED)
    raise LLMError(EMPTY)


def generate_answer(user_message):
    """
    Send the user message (from prompts.build_user_message) to Gemini
    and return the answer text. Raises LLMError on failure.
    """
    api_key = config.get_api_key()
    if not api_key:
        raise LLMError(MISSING_KEY)

    client = _create_client(api_key)
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        thinking_config=_thinking_config(),
    )

    total_attempts = config.MAX_RETRIES + 1  # first try + retries
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_message,
                config=gen_config,
            )
            return _extract_text(response)
        except LLMError:
            raise  # blocked/empty: retrying will not help
        except Exception as exc:  # noqa: BLE001 - we classify everything
            kind = _classify_error(exc)
            # Log only the exception type and code, never the message.
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s, code=%s, kind=%s",
                attempt, total_attempts, type(exc).__name__,
                getattr(exc, "code", None), kind,
            )
            if kind in RETRYABLE_KINDS and attempt < total_attempts:
                time.sleep(2 ** (attempt - 1))  # wait 1s, then 2s
                continue
            raise LLMError(kind) from None