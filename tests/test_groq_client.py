"""Alternative provider (Groq via the OpenAI SDK): config, request, extraction, errors, workflow.

Real openai SDK + real parser and validator; only Groq's server is faked.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest

from src import api_client
from src.api_client import MissingAPIKeyError, load_settings
from src.classifier import MAX_INPUT_CHARS, SYSTEM_PROMPT
from src.groq_client import (
    DEFAULT_GROQ_MODEL,
    GROQ_BASE_URL,
    JSON_OBJECT_RESPONSE_FORMAT,
    JSON_SCHEMA_RESPONSE_FORMAT,
    GroqAPIError,
    GroqAuthenticationError,
    GroqConnectionError,
    GroqRateLimitError,
    GroqSettings,
    GroqUnexpectedResponseError,
    MissingGroqAPIKeyError,
    classify_text_with_groq,
    extract_groq_content,
    load_groq_settings,
    send_groq_chat,
)
from src.response_parser import (
    CLASSIFICATION_JSON_SCHEMA,
    EmptyResponseError,
    MalformedJSONError,
    SchemaValidationError,
)
from tests.fakes import (
    FAKE_GROQ_API_KEY,
    FakeGroqAPI,
    chat_completion_body,
    groq_error_body,
    groq_respond,
)

GOOD_REPLY = json.dumps({"category": "billing", "confidence": 0.9, "explanation": "The user reports a double charge."})
USER_MESSAGES = [{"role": "user", "content": "hi"}]


def reply_with(content: str | None, **kwargs) -> FakeGroqAPI:
    return FakeGroqAPI(groq_respond(200, chat_completion_body(content, **kwargs)))


# --- configuration -----------------------------------------------------------


def test_load_groq_settings_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_API_KEY)

    settings = load_groq_settings()

    assert settings.api_key == FAKE_GROQ_API_KEY
    assert settings.model == DEFAULT_GROQ_MODEL


def test_load_groq_settings_reads_key_and_model_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"GROQ_API_KEY={FAKE_GROQ_API_KEY}\nGROQ_MODEL=openai/gpt-oss-120b\n")

    settings = load_groq_settings(env_file)

    assert settings.api_key == FAKE_GROQ_API_KEY
    assert settings.model == "openai/gpt-oss-120b"
    assert "GROQ_API_KEY" not in os.environ


def test_groq_environment_variable_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=gsk_from_file\n")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_API_KEY)

    assert load_groq_settings(env_file).api_key == FAKE_GROQ_API_KEY


def test_groq_uses_the_shared_default_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "shared.env"
    env_file.write_text(f"GROQ_API_KEY={FAKE_GROQ_API_KEY}\n")
    monkeypatch.setattr(api_client, "DEFAULT_ENV_FILE", env_file)

    assert load_groq_settings().api_key == FAKE_GROQ_API_KEY


def test_missing_groq_key_raises_clear_error():
    with pytest.raises(MissingGroqAPIKeyError, match="GROQ_API_KEY is not configured"):
        load_groq_settings()


@pytest.mark.parametrize("placeholder", ["", "   ", "your-groq-api-key-here"])
def test_empty_or_placeholder_groq_key_counts_as_missing(tmp_path, placeholder):
    env_file = tmp_path / ".env"
    env_file.write_text(f"GROQ_API_KEY={placeholder}\n")

    with pytest.raises(MissingGroqAPIKeyError):
        load_groq_settings(env_file)


def test_groq_key_does_not_satisfy_the_claude_key(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"GROQ_API_KEY={FAKE_GROQ_API_KEY}\n")

    assert load_groq_settings(env_file).api_key == FAKE_GROQ_API_KEY
    with pytest.raises(MissingAPIKeyError):
        load_settings(env_file)


def test_groq_settings_repr_never_shows_the_key():
    settings = GroqSettings(api_key=FAKE_GROQ_API_KEY)

    assert FAKE_GROQ_API_KEY not in repr(settings)


# --- the request -------------------------------------------------------------


def test_send_groq_chat_builds_expected_request(groq_settings):
    api = reply_with(GOOD_REPLY)

    send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES, response_format=JSON_SCHEMA_RESPONSE_FORMAT)

    request = api.last_request
    # Literal URL on purpose: comparing with GROQ_BASE_URL would also pass if the constant were wrong.
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert request.headers["authorization"] == f"Bearer {FAKE_GROQ_API_KEY}"
    body = api.last_json
    assert body["model"] == DEFAULT_GROQ_MODEL
    assert body["messages"] == USER_MESSAGES
    assert body["max_completion_tokens"] == 1024
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "text_classification", "strict": True, "schema": CLASSIFICATION_JSON_SCHEMA},
    }
    # Fields Groq rejects with HTTP 400 must never be sent.
    for unsupported in ("logprobs", "logit_bias", "top_logprobs", "n"):
        assert unsupported not in body


def test_reasoning_effort_only_sent_to_gpt_oss_models():
    settings = GroqSettings(api_key=FAKE_GROQ_API_KEY, model="qwen/qwen3.8-27b")
    api = reply_with(GOOD_REPLY)

    send_groq_chat(api.client(settings), settings, USER_MESSAGES)

    assert api.last_json["model"] == "qwen/qwen3.8-27b"
    assert "reasoning_effort" not in api.last_json


def test_send_groq_chat_rejects_empty_messages(groq_settings):
    api = reply_with(GOOD_REPLY)

    with pytest.raises(ValueError, match="at least one message"):
        send_groq_chat(api.client(groq_settings), groq_settings, [])
    assert api.requests == []


# --- extracting the reply ----------------------------------------------------


def test_extract_returns_content_and_ignores_reasoning(groq_settings):
    api = reply_with("plain answer")

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    assert extract_groq_content(completion) == "plain answer"


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},  # choices missing
        {"choices": None},
    ],
)
def test_extract_rejects_missing_choices(groq_settings, body):
    api = FakeGroqAPI(groq_respond(200, body))

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError, match="choices to be a list"):
        extract_groq_content(completion)


def test_extract_rejects_empty_choices(groq_settings):
    api = FakeGroqAPI(groq_respond(200, {**chat_completion_body("x"), "choices": []}))

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError, match="no choices"):
        extract_groq_content(completion)


def test_extract_rejects_non_json_success_body(groq_settings):
    api = FakeGroqAPI(lambda request: httpx.Response(200, text="<html>gateway page</html>"))

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError):
        extract_groq_content(completion)


def test_extract_rejects_null_content(groq_settings):
    api = reply_with(None, finish_reason="tool_calls")

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError, match="no text content.*tool_calls"):
        extract_groq_content(completion)


@pytest.mark.parametrize(
    ("finish_reason", "message"),
    [("length", "max_completion_tokens"), ("content_filter", "content filter")],
)
def test_extract_rejects_truncated_or_filtered_reply(groq_settings, finish_reason, message):
    api = reply_with('{"category": "bil', finish_reason=finish_reason)

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError, match=message):
        extract_groq_content(completion)


def test_extract_reports_refusal(groq_settings):
    api = reply_with(None, refusal="I can't help with that.")

    completion = send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    with pytest.raises(GroqUnexpectedResponseError, match="refused"):
        extract_groq_content(completion)


def test_extract_handles_objects_without_attributes():
    with pytest.raises(GroqUnexpectedResponseError):
        extract_groq_content(SimpleNamespace())


# --- API errors --------------------------------------------------------------


def test_groq_authentication_error_does_not_leak_key(groq_settings):
    api = FakeGroqAPI(groq_respond(401, groq_error_body("Invalid API Key", code="invalid_api_key")))

    with pytest.raises(GroqAuthenticationError, match="401") as caught:
        send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    assert FAKE_GROQ_API_KEY not in str(caught.value)
    assert type(caught.value.__cause__).__name__ == "AuthenticationError"


def test_groq_permission_denied_is_an_authentication_error(groq_settings):
    api = FakeGroqAPI(groq_respond(403, groq_error_body("Organization restricted", error_type="permission_error")))

    with pytest.raises(GroqAuthenticationError, match="403.*Organization restricted"):
        send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)


def test_groq_rate_limit_keeps_retry_after(groq_settings):
    api = FakeGroqAPI(
        groq_respond(429, groq_error_body("Rate limit reached", error_type="tokens", code="rate_limit_exceeded"), headers={"retry-after": "7"})
    )

    with pytest.raises(GroqRateLimitError, match="429.*7 seconds") as caught:
        send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    assert caught.value.retry_after == "7"


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (400, "json_validate_failed", "Generated JSON does not match the expected schema."),
        (404, "model_not_found", "The model `gone` does not exist"),
        (500, None, "Internal Server Error"),
        (503, None, "Service Unavailable"),
    ],
)
def test_other_groq_http_errors_become_api_errors(groq_settings, status, code, message):
    api = FakeGroqAPI(groq_respond(status, groq_error_body(message, code=code)))

    with pytest.raises(GroqAPIError, match=f"HTTP {status}") as caught:
        send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)

    assert caught.value.status_code == status
    assert caught.value.code == code
    assert message in str(caught.value)


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_groq_network_failures_become_connection_errors(groq_settings, failure):
    def fail(request):
        raise failure("simulated network failure", request=request)

    api = FakeGroqAPI(fail)

    with pytest.raises(GroqConnectionError, match="Could not reach the Groq API"):
        send_groq_chat(api.client(groq_settings), groq_settings, USER_MESSAGES)


# --- classification workflow -------------------------------------------------


def test_classify_with_groq_returns_validated_dict(groq_settings):
    api = reply_with(GOOD_REPLY)

    result = classify_text_with_groq("I was charged twice.", settings=groq_settings, client=api.client(groq_settings))

    assert result == {"category": "billing", "confidence": 0.9, "explanation": "The user reports a double charge."}
    messages = api.last_json["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "<text>\nI was charged twice.\n</text>" in messages[1]["content"]
    assert api.last_json["response_format"] == JSON_SCHEMA_RESPONSE_FORMAT


def test_json_object_mode_still_validates_locally(groq_settings):
    api = reply_with('{"category": "billing", "confidence": 0.9}')  # valid JSON, wrong schema

    with pytest.raises(SchemaValidationError, match="explanation"):
        classify_text_with_groq("I was charged twice.", settings=groq_settings, client=api.client(groq_settings), use_json_schema=False)

    assert api.last_json["response_format"] == JSON_OBJECT_RESPONSE_FORMAT


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("The category is billing.", MalformedJSONError),
        ('{"category": "billing",}', MalformedJSONError),
        ("", EmptyResponseError),
        ('{"category": "refunds", "confidence": 0.9, "explanation": "x"}', SchemaValidationError),
    ],
)
def test_classify_with_groq_rejects_bad_replies(groq_settings, content, error):
    api = reply_with(content)

    with pytest.raises(error):
        classify_text_with_groq("I was charged twice.", settings=groq_settings, client=api.client(groq_settings))


def test_classify_with_groq_missing_key_raises_before_any_request():
    with pytest.raises(MissingGroqAPIKeyError):
        classify_text_with_groq("I was charged twice.")


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_INPUT_CHARS + 1)])
def test_classify_with_groq_rejects_bad_input_before_any_request(groq_settings, text):
    api = reply_with(GOOD_REPLY)

    with pytest.raises(ValueError):
        classify_text_with_groq(text, settings=groq_settings, client=api.client(groq_settings))
    assert api.requests == []
