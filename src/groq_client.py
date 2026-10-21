"""Alternative provider: Groq, called through the official OpenAI Python SDK.

Groq offers an OpenAI-compatible API, so the standard ``openai`` package works
once its ``base_url`` points at Groq. This module is separate from the Claude
client in ``api_client.py`` and does not change it. It reuses the same
``.env`` file, the same classification prompt (``classifier.py``) and the same
JSON parsing and schema validation (``response_parser.py``).

A successful Groq call does NOT satisfy the separate requirement to make a
working call to the Claude API.

Nothing in this module prints or logs the API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from dotenv import dotenv_values

from src import api_client
from src.classifier import SYSTEM_PROMPT, build_user_prompt, clean_input_text
from src.response_parser import CLASSIFICATION_JSON_SCHEMA, ClassificationResult, parse_classification_response

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# A Groq production model (not a preview model) that supports strict
# JSON-schema structured outputs. Override with GROQ_MODEL.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# GPT-OSS models reason before answering; leave room for that plus the JSON.
GROQ_MAX_COMPLETION_TOKENS = 1024
GROQ_TIMEOUT_SECONDS = 60.0

# Strict mode uses constrained decoding, so the reply should always match the
# schema. It is still parsed and validated locally like any other reply.
JSON_SCHEMA_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "text_classification", "strict": True, "schema": CLASSIFICATION_JSON_SCHEMA},
}
# JSON mode only guarantees syntactically valid JSON, not the schema.
JSON_OBJECT_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}

# Values meaning "the key was never filled in", e.g. an empty GROQ_API_KEY= line.
_PLACEHOLDER_KEYS = frozenset({"", "your-groq-api-key-here"})


class GroqClientError(Exception):
    """Base class for every error raised by this module."""


class GroqConfigError(GroqClientError):
    """Configuration is missing or invalid. No request was sent."""


class MissingGroqAPIKeyError(GroqConfigError):
    """GROQ_API_KEY is not set in the environment or in ``.env``."""


class GroqAuthenticationError(GroqClientError):
    """Groq rejected the key (HTTP 401) or denied access (HTTP 403)."""


class GroqRateLimitError(GroqClientError):
    """Groq still returned HTTP 429 after the SDK's automatic retries."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GroqConnectionError(GroqClientError):
    """Groq could not be reached (DNS failure, refused connection, timeout)."""


class GroqAPIError(GroqClientError):
    """Any other error response from Groq (400, 404, 500, 503, ...)."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class GroqUnexpectedResponseError(GroqClientError):
    """Groq answered, but not with usable text (truncated, filtered, odd shape)."""


@dataclass(frozen=True)
class GroqSettings:
    """Groq configuration. ``repr=False`` keeps the key out of logs and tracebacks."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_GROQ_MODEL


def load_groq_settings(env_file: Path | None = None) -> GroqSettings:
    """Build Groq settings the same way ``api_client.load_settings()`` does.

    A non-empty environment variable wins over the shared ``.env`` file, which
    is read with ``dotenv_values`` so ``os.environ`` is never modified.

    Raises:
        MissingGroqAPIKeyError: GROQ_API_KEY is unset, empty or a placeholder.
    """
    path = env_file if env_file is not None else api_client.DEFAULT_ENV_FILE
    file_values = dotenv_values(path) if path.is_file() else {}

    def lookup(name: str) -> str:
        value = os.environ.get(name) or file_values.get(name) or ""
        return value.strip()

    api_key = lookup("GROQ_API_KEY")
    if api_key in _PLACEHOLDER_KEYS:
        raise MissingGroqAPIKeyError(
            "GROQ_API_KEY is not configured. Add your key after GROQ_API_KEY= in .env, "
            "or export GROQ_API_KEY in your shell."
        )
    return GroqSettings(api_key=api_key, model=lookup("GROQ_MODEL") or DEFAULT_GROQ_MODEL)


def create_groq_client(
    settings: GroqSettings,
    *,
    max_retries: int = 2,
    timeout: float = GROQ_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
) -> openai.OpenAI:
    """Create an OpenAI SDK client that talks to Groq instead of OpenAI.

    The SDK retries connection errors, 408, 409, 429 and 5xx responses itself
    (``max_retries`` times). ``http_client`` lets tests swap in a fake network.
    """
    return openai.OpenAI(
        api_key=settings.api_key,
        base_url=GROQ_BASE_URL,
        max_retries=max_retries,
        timeout=timeout,
        http_client=http_client,
    )


def send_groq_chat(
    client: openai.OpenAI,
    settings: GroqSettings,
    messages: list[dict[str, str]],
    *,
    response_format: dict[str, Any] | None = None,
    max_completion_tokens: int = GROQ_MAX_COMPLETION_TOKENS,
) -> Any:
    """Send a chat completion request to Groq and return the SDK's response object.

    Raises:
        ValueError: ``messages`` is empty (no request is sent).
        GroqAuthenticationError, GroqRateLimitError, GroqConnectionError,
        GroqAPIError: the request failed; the SDK error is chained as ``__cause__``.
    """
    if not messages:
        raise ValueError("messages must contain at least one message.")

    request: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format is not None:
        request["response_format"] = response_format
    if settings.model.startswith("openai/gpt-oss"):
        # GPT-OSS models accept low/medium/high; "low" keeps short tasks fast.
        request["reasoning_effort"] = "low"

    # Most specific SDK exceptions first, as in api_client.send_message().
    try:
        return client.chat.completions.create(**request)
    except openai.AuthenticationError as exc:
        raise GroqAuthenticationError(
            "Groq rejected GROQ_API_KEY (HTTP 401). Check that the key is copied "
            "correctly and has not been revoked."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise GroqAuthenticationError(
            f"The Groq key is not allowed to make this request (HTTP 403): {_groq_error_detail(exc)}"
        ) from exc
    except openai.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        hint = f" Retry after {retry_after} seconds." if retry_after else " Wait a moment and retry."
        raise GroqRateLimitError(f"Rate limited by Groq (HTTP 429).{hint}", retry_after=retry_after) from exc
    except openai.APIStatusError as exc:
        raise GroqAPIError(
            f"Groq returned HTTP {exc.status_code}: {_groq_error_detail(exc)}",
            status_code=exc.status_code,
            code=exc.code,
        ) from exc
    except openai.APIConnectionError as exc:  # also covers APITimeoutError
        raise GroqConnectionError(f"Could not reach the Groq API ({exc}). Check your internet connection.") from exc
    except openai.APIError as exc:
        raise GroqAPIError(f"Unexpected error from the OpenAI SDK: {exc}") from exc


def _groq_error_detail(exc: openai.APIStatusError) -> str:
    """Return "<code or type>: <message>" from a Groq error body, if it has one."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            label = error.get("code") or error.get("type") or "error"
            return f"{label}: {error['message']}"
    return exc.message


def extract_groq_content(completion: Any) -> str:
    """Return the assistant's text from a chat completion.

    Only ``message.content`` is used; GPT-OSS reasoning arrives in a separate
    ``reasoning`` field and is ignored.

    Raises:
        GroqUnexpectedResponseError: no choices, the reply was cut off or
            filtered, the model refused, or there is no text content.
    """
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list):
        raise GroqUnexpectedResponseError(f"Expected response.choices to be a list, got {type(choices).__name__}.")
    if not choices:
        raise GroqUnexpectedResponseError("The response contained no choices.")

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise GroqUnexpectedResponseError(
            "The reply was cut off because it reached max_completion_tokens; a partial "
            "reply is not returned. Increase max_completion_tokens and try again."
        )
    if finish_reason == "content_filter":
        raise GroqUnexpectedResponseError("The reply was withheld by a content filter (finish_reason='content_filter').")

    message = getattr(choice, "message", None)
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise GroqUnexpectedResponseError(f"The model refused the request: {refusal}")

    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise GroqUnexpectedResponseError(f"The reply has no text content (finish_reason={finish_reason!r}).")
    return content


def classify_text_with_groq(
    text: str,
    *,
    settings: GroqSettings | None = None,
    client: openai.OpenAI | None = None,
    use_json_schema: bool = True,
) -> ClassificationResult:
    """Classify ``text`` with Groq and return ``{"category", "confidence", "explanation"}``.

    Same task, prompt and validation as ``classifier.classify_text()``.
    ``use_json_schema=False`` switches from strict JSON-schema mode to JSON
    object mode, which checks JSON syntax only, so local validation does the rest.

    Raises:
        ValueError: ``text`` is empty or too long (no request sent).
        GroqConfigError: GROQ_API_KEY is missing (no request sent).
        GroqClientError: the API call failed or the response had no usable text.
        ResponseParseError: the reply was empty, not valid JSON, or failed
            schema validation (EmptyResponseError, MalformedJSONError, SchemaValidationError).
    """
    cleaned = clean_input_text(text)

    if settings is None:
        settings = load_groq_settings()
    if client is None:
        client = create_groq_client(settings)

    completion = send_groq_chat(
        client,
        settings,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(cleaned)},
        ],
        response_format=JSON_SCHEMA_RESPONSE_FORMAT if use_json_schema else JSON_OBJECT_RESPONSE_FORMAT,
    )
    return parse_classification_response(extract_groq_content(completion))
