"""Tests for llm.py. No real Gemini calls: the client is mocked."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors

from chatbot import llm
from chatbot import config

FAKE_KEY = "FAKE-SECRET-KEY-123"


def make_response(text="Hello from Gemini"):
    return SimpleNamespace(text=text, prompt_feedback=None, candidates=[])


def make_api_error(error_class, code, message="something went wrong"):
    return error_class(code, {"error": {"code": code, "message": message}})


@pytest.fixture
def fake_client(monkeypatch):
    """Fake API key + fake Gemini client + no real sleeping."""
    monkeypatch.setattr(llm.config, "get_api_key", lambda: FAKE_KEY)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    client = MagicMock()
    client.models.generate_content.return_value = make_response()
    create = MagicMock(return_value=client)
    monkeypatch.setattr(llm.genai, "Client", create)
    client.create = create  # handy for checking constructor arguments
    return client


def test_successful_response(fake_client):
    assert llm.generate_answer("Hi") == "Hello from Gemini"
    assert fake_client.models.generate_content.call_count == 1


def test_settings_passed_correctly(fake_client):
    llm.generate_answer("Hi")
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == config.GEMINI_MODEL
    assert kwargs["config"].temperature == config.TEMPERATURE
    assert kwargs["config"].max_output_tokens == config.MAX_OUTPUT_TOKENS


def test_timeout_converted_to_milliseconds(fake_client):
    llm.generate_answer("Hi")
    client_kwargs = fake_client.create.call_args.kwargs
    assert client_kwargs["api_key"] == FAKE_KEY
    assert client_kwargs["http_options"].timeout == config.REQUEST_TIMEOUT_SECONDS * 1000


def test_system_prompt_passed_correctly(fake_client):
    llm.generate_answer("Hi")
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["config"].system_instruction == llm.SYSTEM_PROMPT


def test_user_message_passed_correctly(fake_client):
    llm.generate_answer("What does Plus cost?")
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "What does Plus cost?"


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(llm.config, "get_api_key", lambda: None)
    create = MagicMock()
    monkeypatch.setattr(llm.genai, "Client", create)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.MISSING_KEY
    create.assert_not_called()  # no client is created without a key


def test_empty_response(fake_client):
    fake_client.models.generate_content.return_value = make_response(text=None)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.EMPTY
    assert fake_client.models.generate_content.call_count == 1  # not retried


def test_blocked_response(fake_client):
    blocked = SimpleNamespace(
        text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[]
    )
    fake_client.models.generate_content.return_value = blocked
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.BLOCKED


def test_authentication_error_not_retried(fake_client):
    fake_client.models.generate_content.side_effect = make_api_error(errors.ClientError, 401)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.AUTH
    assert fake_client.models.generate_content.call_count == 1


def test_invalid_key_reported_as_400_is_auth_error(fake_client):
    fake_client.models.generate_content.side_effect = make_api_error(
        errors.ClientError, 400, "API key not valid. Please pass a valid API key."
    )
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.AUTH


def test_rate_limit_retries_then_fails(fake_client):
    fake_client.models.generate_content.side_effect = make_api_error(errors.ClientError, 429)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.RATE_LIMIT
    assert fake_client.models.generate_content.call_count == config.MAX_RETRIES + 1


def test_rate_limit_then_success(fake_client):
    fake_client.models.generate_content.side_effect = [
        make_api_error(errors.ClientError, 429),
        make_response("Recovered"),
    ]
    assert llm.generate_answer("Hi") == "Recovered"
    assert fake_client.models.generate_content.call_count == 2


def test_timeout_error(fake_client):
    fake_client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.TIMEOUT
    assert fake_client.models.generate_content.call_count == config.MAX_RETRIES + 1


def test_connection_error(fake_client):
    fake_client.models.generate_content.side_effect = httpx.ConnectError("no network")
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.CONNECTION


def test_server_error_is_retried(fake_client):
    fake_client.models.generate_content.side_effect = make_api_error(errors.ServerError, 503)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.SERVER
    assert fake_client.models.generate_content.call_count == config.MAX_RETRIES + 1


def test_model_not_found(fake_client):
    fake_client.models.generate_content.side_effect = make_api_error(errors.ClientError, 404)
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.MODEL_UNAVAILABLE
    assert fake_client.models.generate_content.call_count == 1


def test_unexpected_error_not_retried(fake_client):
    fake_client.models.generate_content.side_effect = ValueError("boom")
    with pytest.raises(llm.LLMError) as info:
        llm.generate_answer("Hi")
    assert info.value.kind == llm.UNKNOWN
    assert fake_client.models.generate_content.call_count == 1


def test_api_key_not_exposed(fake_client, caplog):
    """Even if Gemini's error text contains the key, it must never leak."""
    fake_client.models.generate_content.side_effect = make_api_error(
        errors.ClientError, 401, f"Bad key {FAKE_KEY}"
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(llm.LLMError) as info:
            llm.generate_answer("Hi")
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in repr(info.value)
    assert FAKE_KEY not in caplog.text
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True