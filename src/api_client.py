"""Monday task: a small, safe client for the Claude Messages API.

This module wraps the official ``anthropic`` Python SDK and adds the three
things an application needs around every call:

* loading configuration from environment variables or a local ``.env`` file,
* translating SDK exceptions into a few project exceptions with clear,
  actionable messages, and
* extracting the text content from a response defensively.

Nothing in this module prints or logs the API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import httpx2
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "claude-opus-5-5"
# Thinking is always on for claude-opus-5-5 and counts toward max_tokens, so
# leave room for it even though the visible replies here are short.
DEFAULT_MAX_TOKENS = 1024
# Effort controls how much the model thinks (and so latency and cost).
# "low" is enough for a greeting or a short classification.
DEFAULT_EFFORT = "low"
DEFAULT_TIMEOUT_SECONDS = 60.0

# Opt-in server-side refusal fallback: if the model's safety classifiers
# decline a request, the API re-runs it on Anthropic's recommended fallback
# model inside the same call. It is a beta feature, so it needs this header.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Values meaning "the key was never filled in", e.g. .env copied from .env.example.
_PLACEHOLDER_KEYS = frozenset({"", "your-api-key-here"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ClaudeClientError(Exception):
    """Base class for every error raised by this module."""


class ClaudeConfigError(ClaudeClientError):
    """Configuration is missing or invalid. No request was sent."""


class MissingAPIKeyError(ClaudeConfigError):
    """ANTHROPIC_API_KEY is not set in the environment or in ``.env``."""


class ClaudeAuthenticationError(ClaudeClientError):
    """The API rejected the key (HTTP 401) or denied access (HTTP 403)."""


class ClaudeRateLimitError(ClaudeClientError):
    """The API still returned HTTP 429 after the SDK's automatic retries."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ClaudeConnectionError(ClaudeClientError):
    """The API could not be reached (DNS failure, refused connection, timeout)."""


class ClaudeAPIError(ClaudeClientError):
    """Any other error response from the API (400, 404, 500, 529, ...)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnexpectedResponseError(ClaudeClientError):
    """The API answered, but not with usable text (refusal, truncation, odd shape)."""


@dataclass(frozen=True)
class ClaudeSettings:
    """Runtime configuration. ``repr=False`` keeps the key out of logs and tracebacks."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    use_fallbacks: bool = True


def load_settings(env_file: Path | None = None) -> ClaudeSettings:
    """Build settings from environment variables, falling back to a ``.env`` file.

    A non-empty environment variable wins over the file. The file is read with
    ``dotenv_values`` so loading it never modifies ``os.environ``.

    Raises:
        MissingAPIKeyError: ANTHROPIC_API_KEY is unset, empty or a placeholder.
        ClaudeConfigError: an optional setting has an invalid value.
    """
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    file_values = dotenv_values(path) if path.is_file() else {}

    def lookup(name: str) -> str:
        value = os.environ.get(name) or file_values.get(name) or ""
        return value.strip()

    api_key = lookup("ANTHROPIC_API_KEY")
    if api_key in _PLACEHOLDER_KEYS:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not configured. Copy .env.example to .env and set "
            "the key from your supervisor, or export ANTHROPIC_API_KEY in your shell."
        )

    return ClaudeSettings(
        api_key=api_key,
        model=lookup("CLAUDE_MODEL") or DEFAULT_MODEL,
        use_fallbacks=_parse_bool("CLAUDE_USE_FALLBACKS", lookup("CLAUDE_USE_FALLBACKS"), default=True),
    )


def _parse_bool(name: str, raw: str, *, default: bool) -> bool:
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ClaudeConfigError(f"{name} must be true or false, got {raw!r}.")


def create_client(
    settings: ClaudeSettings,
    *,
    max_retries: int = 2,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    http_client: httpx2.Client | None = None,
) -> anthropic.Anthropic:
    """Create an SDK client for ``settings``.

    The SDK already retries connection errors, 408, 409, 429 and 5xx responses
    with exponential backoff (``max_retries`` times), so callers only see an
    error once those retries are used up. ``http_client`` lets tests swap the
    network layer for a fake one.
    """
    return anthropic.Anthropic(
        api_key=settings.api_key,
        max_retries=max_retries,
        timeout=timeout,
        http_client=http_client,
    )


def send_message(
    client: anthropic.Anthropic,
    settings: ClaudeSettings,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
    output_format: dict[str, Any] | None = None,
) -> Any:
    """Send one user message to Claude and return the SDK's response object.

    ``output_format`` is passed as ``output_config.format`` (structured outputs).

    Raises:
        ValueError: ``prompt`` is empty (no request is sent).
        ClaudeAuthenticationError, ClaudeRateLimitError, ClaudeConnectionError,
        ClaudeAPIError: the request failed; the SDK error is chained as ``__cause__``.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    output_config: dict[str, Any] = {"effort": effort}
    if output_format is not None:
        output_config["format"] = output_format

    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": output_config,
    }
    if system:
        request["system"] = system

    # Most specific SDK exceptions first: retryable (429, network) and
    # non-retryable (401/403/4xx) failures need different advice.
    try:
        if settings.use_fallbacks:
            return client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        return client.messages.create(**request)
    except anthropic.AuthenticationError as exc:
        raise ClaudeAuthenticationError(
            "The API rejected ANTHROPIC_API_KEY (HTTP 401). Check that the key is "
            "copied correctly and has not been revoked."
        ) from exc
    except anthropic.PermissionDeniedError as exc:
        raise ClaudeAuthenticationError(
            f"The API key is not allowed to make this request (HTTP 403): {_api_error_detail(exc)}"
        ) from exc
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        hint = f" Retry after {retry_after} seconds." if retry_after else " Wait a moment and retry."
        raise ClaudeRateLimitError(f"Rate limited by the API (HTTP 429).{hint}", retry_after=retry_after) from exc
    except anthropic.APIStatusError as exc:
        raise ClaudeAPIError(
            f"The API returned HTTP {exc.status_code}: {_api_error_detail(exc)}",
            status_code=exc.status_code,
        ) from exc
    except anthropic.APIConnectionError as exc:  # also covers APITimeoutError
        raise ClaudeConnectionError(
            f"Could not reach the Anthropic API ({exc}). Check your internet connection."
        ) from exc
    except anthropic.APIError as exc:
        raise ClaudeAPIError(f"Unexpected error from the Anthropic SDK: {exc}") from exc


def _api_error_detail(exc: anthropic.APIStatusError) -> str:
    """Return "<error type>: <message>" from an API error body, if it has one."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return f"{error.get('type', 'error')}: {error['message']}"
    return exc.message


def extract_text(response: Any) -> str:
    """Return the text of a Messages API response.

    A response can hold several block types (``thinking``, ``text``,
    ``tool_use``, ...), so text blocks are selected by ``type``, never by
    position, and joined in order.

    Raises:
        UnexpectedResponseError: Claude declined the request, the reply was cut
            off at ``max_tokens``, or the response contains no text blocks.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise UnexpectedResponseError(f"Claude declined the request (stop_reason='refusal', category={category!r}).")
    if stop_reason == "max_tokens":
        raise UnexpectedResponseError(
            "The reply was cut off because it reached max_tokens; a partial reply is not "
            "returned. Increase max_tokens and try again."
        )

    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise UnexpectedResponseError(f"Expected response.content to be a list, got {type(content).__name__}.")

    texts = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    if not texts:
        block_types = [getattr(block, "type", type(block).__name__) for block in content]
        raise UnexpectedResponseError(f"The response contained no text blocks (block types: {block_types}).")
    return "".join(texts)


def ask_claude(
    prompt: str,
    *,
    settings: ClaudeSettings | None = None,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Convenience wrapper: load settings, send ``prompt``, return Claude's text reply."""
    if settings is None:
        settings = load_settings()
    if client is None:
        client = create_client(settings)
    return extract_text(send_message(client, settings, prompt))
