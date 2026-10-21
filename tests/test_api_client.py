"""Monday: configuration loading, the Messages API request, text extraction and error handling."""

from __future__ import annotations

import os
from types import SimpleNamespace

import httpx2
import pytest

from src.api_client import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeAPIError,
    ClaudeAuthenticationError,
    ClaudeConfigError,
    ClaudeConnectionError,
    ClaudeRateLimitError,
    ClaudeSettings,
    MissingAPIKeyError,
    UnexpectedResponseError,
    ask_claude,
    extract_text,
    load_settings,
    send_message,
)
from tests.fakes import (
    FAKE_API_KEY,
    FakeAnthropicAPI,
    error_body,
    message_body,
    respond,
    text_block,
    thinking_block,
)

# --- configuration -----------------------------------------------------------


def test_load_settings_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    settings = load_settings()

    assert settings.api_key == FAKE_API_KEY
    assert settings.model == DEFAULT_MODEL
    assert settings.use_fallbacks is True


def test_load_settings_reads_key_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={FAKE_API_KEY}\nCLAUDE_MODEL=claude-sonnet-5-5\n")

    settings = load_settings(env_file)

    assert settings.api_key == FAKE_API_KEY
    assert settings.model == "claude-sonnet-5-5"


def test_environment_variable_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    assert load_settings(env_file).api_key == FAKE_API_KEY


def test_loading_env_file_does_not_modify_os_environ(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={FAKE_API_KEY}\n")

    load_settings(env_file)

    assert "ANTHROPIC_API_KEY" not in os.environ


def test_missing_api_key_raises_clear_error():
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY is not configured"):
        load_settings()


@pytest.mark.parametrize("placeholder", ["", "   ", "your-api-key-here"])
def test_placeholder_api_key_counts_as_missing(tmp_path, placeholder):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={placeholder}\n")

    with pytest.raises(MissingAPIKeyError):
        load_settings(env_file)


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("TRUE", True), ("yes", True)])
def test_fallback_setting_parses_booleans(monkeypatch, raw, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("CLAUDE_USE_FALLBACKS", raw)

    assert load_settings().use_fallbacks is expected


def test_invalid_fallback_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("CLAUDE_USE_FALLBACKS", "maybe")

    with pytest.raises(ClaudeConfigError, match="CLAUDE_USE_FALLBACKS"):
        load_settings()


def test_settings_repr_never_shows_the_key():
    settings = ClaudeSettings(api_key=FAKE_API_KEY)

    assert FAKE_API_KEY not in repr(settings)
    assert FAKE_API_KEY not in str(settings)


# --- the request -------------------------------------------------------------


def test_send_message_builds_expected_request(settings):
    api = FakeAnthropicAPI(respond(200, message_body(text_block("Hello!"))))

    send_message(api.client(settings), settings, "Say hello", system="Be brief.")

    request = api.last_request
    assert request.method == "POST"
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == FAKE_API_KEY
    assert request.headers["anthropic-beta"] == FALLBACK_BETA
    body = api.last_json
    assert body["model"] == DEFAULT_MODEL
    assert body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["system"] == "Be brief."
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    # Removed/unsupported parameters on this model must never be sent.
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in body


def test_send_message_without_fallbacks_uses_standard_endpoint():
    settings = ClaudeSettings(api_key=FAKE_API_KEY, use_fallbacks=False)
    api = FakeAnthropicAPI(respond(200, message_body(text_block("Hello!"))))

    send_message(api.client(settings), settings, "Say hello")

    assert "anthropic-beta" not in api.last_request.headers
    assert "fallbacks" not in api.last_json


def test_send_message_rejects_empty_prompt_without_calling_api(settings):
    api = FakeAnthropicAPI(respond(200, message_body(text_block("unused"))))

    with pytest.raises(ValueError, match="non-empty"):
        send_message(api.client(settings), settings, "   ")
    assert api.requests == []


def test_ask_claude_returns_reply_text(settings):
    api = FakeAnthropicAPI(respond(200, message_body(thinking_block(), text_block("API test call worked."))))

    assert ask_claude("ping", settings=settings, client=api.client(settings)) == "API test call worked."


# --- text extraction ---------------------------------------------------------


def test_extract_text_skips_thinking_and_joins_text_blocks(settings):
    api = FakeAnthropicAPI(respond(200, message_body(thinking_block(), text_block("Hello, "), text_block("world."))))

    response = send_message(api.client(settings), settings, "hi")

    assert extract_text(response) == "Hello, world."


def test_extract_text_rejects_response_without_text_blocks(settings):
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    api = FakeAnthropicAPI(respond(200, message_body(thinking_block(), tool_use)))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError, match="no text blocks"):
        extract_text(response)


def test_extract_text_rejects_empty_content_list(settings):
    api = FakeAnthropicAPI(respond(200, message_body()))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError, match="no text blocks"):
        extract_text(response)


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},  # content missing entirely
        {"type": "message", "content": None},
        {"type": "message", "content": "not a list"},
    ],
)
def test_extract_text_rejects_unexpected_response_structure(settings, body):
    api = FakeAnthropicAPI(respond(200, body))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError, match="response.content to be a list"):
        extract_text(response)


def test_extract_text_rejects_non_json_success_body(settings):
    api = FakeAnthropicAPI(lambda request: httpx2.Response(200, text="<html>gateway page</html>"))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError):
        extract_text(response)


def test_extract_text_reports_refusal(settings):
    body = message_body(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None})
    api = FakeAnthropicAPI(respond(200, body))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError, match="declined.*cyber"):
        extract_text(response)


def test_extract_text_rejects_truncated_reply(settings):
    api = FakeAnthropicAPI(respond(200, message_body(text_block('{"category": "bug_'), stop_reason="max_tokens")))

    response = send_message(api.client(settings), settings, "hi")

    with pytest.raises(UnexpectedResponseError, match="max_tokens"):
        extract_text(response)


def test_extract_text_handles_objects_without_attributes():
    with pytest.raises(UnexpectedResponseError):
        extract_text(SimpleNamespace())


# --- error handling ----------------------------------------------------------


def test_authentication_error_is_translated_without_leaking_key(settings):
    api = FakeAnthropicAPI(respond(401, error_body("authentication_error", "invalid x-api-key")))

    with pytest.raises(ClaudeAuthenticationError, match="401") as caught:
        send_message(api.client(settings), settings, "hi")

    assert FAKE_API_KEY not in str(caught.value)
    assert type(caught.value.__cause__).__name__ == "AuthenticationError"


def test_permission_denied_is_an_authentication_error(settings):
    api = FakeAnthropicAPI(respond(403, error_body("permission_error", "workspace not allowed")))

    with pytest.raises(ClaudeAuthenticationError, match="403.*workspace not allowed"):
        send_message(api.client(settings), settings, "hi")


def test_rate_limit_error_keeps_retry_after(settings):
    api = FakeAnthropicAPI(respond(429, error_body("rate_limit_error", "slow down"), headers={"retry-after": "30"}))

    with pytest.raises(ClaudeRateLimitError, match="429.*30 seconds") as caught:
        send_message(api.client(settings), settings, "hi")

    assert caught.value.retry_after == "30"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(400, "invalid_request_error"), (404, "not_found_error"), (500, "api_error"), (529, "overloaded_error")],
)
def test_other_http_errors_become_api_errors(settings, status, error_type):
    api = FakeAnthropicAPI(respond(status, error_body(error_type, "details from server")))

    with pytest.raises(ClaudeAPIError, match=f"HTTP {status}: {error_type}: details from server") as caught:
        send_message(api.client(settings), settings, "hi")

    assert caught.value.status_code == status


@pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ReadTimeout])
def test_network_failures_become_connection_errors(settings, failure):
    def fail(request):
        raise failure("simulated network failure", request=request)

    api = FakeAnthropicAPI(fail)

    with pytest.raises(ClaudeConnectionError, match="Could not reach the Anthropic API"):
        send_message(api.client(settings), settings, "hi")
