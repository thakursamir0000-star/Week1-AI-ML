"""Tuesday workflow end to end: real SDK + real parser, fake HTTP server."""

from __future__ import annotations

import json

import pytest

from src.api_client import ClaudeRateLimitError, MissingAPIKeyError, UnexpectedResponseError
from src.classifier import MAX_INPUT_CHARS, SYSTEM_PROMPT, classify_text
from src.response_parser import (
    CLASSIFICATION_JSON_SCHEMA,
    EmptyResponseError,
    MalformedJSONError,
    SchemaValidationError,
)
from tests.fakes import FakeAnthropicAPI, error_body, message_body, respond, text_block, thinking_block

GOOD_REPLY = json.dumps(
    {"category": "bug_report", "confidence": 0.93, "explanation": "The app crashes when exporting a PDF."}
)


def reply_with(text: str, **kwargs) -> FakeAnthropicAPI:
    return FakeAnthropicAPI(respond(200, message_body(thinking_block(), text_block(text), **kwargs)))


def test_classify_text_returns_validated_dict(settings):
    api = reply_with(GOOD_REPLY)

    result = classify_text("The app crashes when I export a PDF.", settings=settings, client=api.client(settings))

    assert result == {
        "category": "bug_report",
        "confidence": 0.93,
        "explanation": "The app crashes when exporting a PDF.",
    }


def test_classify_text_sends_structured_output_request(settings):
    api = reply_with(GOOD_REPLY)

    classify_text("  The app crashes.  ", settings=settings, client=api.client(settings))

    body = api.last_json
    assert body["system"] == SYSTEM_PROMPT
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": CLASSIFICATION_JSON_SCHEMA},
    }
    user_content = body["messages"][0]["content"]
    assert "<text>\nThe app crashes.\n</text>" in user_content


def test_prompt_only_mode_omits_format_and_still_validates(settings):
    api = reply_with(f"```json\n{GOOD_REPLY}\n```")

    result = classify_text("The app crashes.", settings=settings, client=api.client(settings), use_structured_output=False)

    assert "format" not in api.last_json["output_config"]
    assert result["category"] == "bug_report"


def test_malformed_json_from_model_raises(settings):
    api = reply_with("Sure! The category is bug_report.")

    with pytest.raises(MalformedJSONError):
        classify_text("The app crashes.", settings=settings, client=api.client(settings))


def test_missing_field_from_model_raises(settings):
    api = reply_with('{"category": "bug_report", "confidence": 0.9}')

    with pytest.raises(SchemaValidationError, match="explanation"):
        classify_text("The app crashes.", settings=settings, client=api.client(settings))


def test_empty_model_reply_raises(settings):
    api = reply_with("")

    with pytest.raises(EmptyResponseError):
        classify_text("The app crashes.", settings=settings, client=api.client(settings))


def test_truncated_model_reply_raises(settings):
    api = reply_with('{"category": "bug_report", "confid', stop_reason="max_tokens")

    with pytest.raises(UnexpectedResponseError, match="max_tokens"):
        classify_text("The app crashes.", settings=settings, client=api.client(settings))


def test_api_errors_propagate_as_project_exceptions(settings):
    api = FakeAnthropicAPI(respond(429, error_body("rate_limit_error", "slow down"), headers={"retry-after": "5"}))

    with pytest.raises(ClaudeRateLimitError):
        classify_text("The app crashes.", settings=settings, client=api.client(settings))


def test_missing_api_key_raises_before_any_request():
    with pytest.raises(MissingAPIKeyError):
        classify_text("The app crashes.")


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_is_rejected_before_any_request(settings, text):
    api = reply_with(GOOD_REPLY)

    with pytest.raises(ValueError, match="must not be empty"):
        classify_text(text, settings=settings, client=api.client(settings))
    assert api.requests == []


def test_overlong_input_is_rejected_not_truncated(settings):
    api = reply_with(GOOD_REPLY)

    with pytest.raises(ValueError, match=str(MAX_INPUT_CHARS)):
        classify_text("x" * (MAX_INPUT_CHARS + 1), settings=settings, client=api.client(settings))
    assert api.requests == []
