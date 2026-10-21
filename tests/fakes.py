"""Test doubles for the Anthropic and Groq APIs.

The fakes sit at the HTTP layer (``httpx2.MockTransport`` for the anthropic
SDK, ``httpx.MockTransport`` for the openai SDK): the real SDK builds the
request, parses the response and raises its own typed exceptions, while
nothing ever leaves the machine. Only the remote server is faked; our parsing
and validation code runs for real.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import httpx2
import openai

from src.api_client import ClaudeSettings, create_client
from src.groq_client import GroqSettings, create_groq_client

FAKE_API_KEY = "sk-ant-test-0000-not-a-real-key"
FAKE_GROQ_API_KEY = "gsk_test_0000_not_a_real_key"


class _RecordingServer:
    """Calls ``handler`` for each request and keeps every request it receives."""

    def __init__(self, handler: Callable[[Any], Any]) -> None:
        self._handler = handler
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return self._handler(request)

    @property
    def last_request(self) -> Any:
        assert self.requests, "no request was sent"
        return self.requests[-1]

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.last_request.content)


# --- Anthropic ---------------------------------------------------------------


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def thinking_block() -> dict[str, Any]:
    # claude-opus-5-5 always thinks; by default the thinking text comes back empty.
    return {"type": "thinking", "thinking": "", "signature": "test-signature"}


def message_body(*blocks: dict[str, Any], stop_reason: str = "end_turn", **extra: Any) -> dict[str, Any]:
    """A Messages API success body shaped like the real one."""
    body: dict[str, Any] = {
        "id": "msg_test_0001",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": list(blocks),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 17},
    }
    body.update(extra)
    return body


def error_body(error_type: str, message: str) -> dict[str, Any]:
    """A Messages API error body shaped like the real one."""
    return {"type": "error", "error": {"type": error_type, "message": message}, "request_id": "req_test_0001"}


def respond(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> Callable[[httpx2.Request], httpx2.Response]:
    """Handler that always answers with the same status and JSON body."""
    return lambda request: httpx2.Response(status, json=body, headers=headers)


class FakeAnthropicAPI(_RecordingServer):
    """Stands in for api.anthropic.com and records every request it receives."""

    def client(self, settings: ClaudeSettings) -> anthropic.Anthropic:
        # max_retries=0: SDK retries would only repeat the same canned answer.
        transport = httpx2.MockTransport(self)
        return create_client(settings, max_retries=0, http_client=httpx2.Client(transport=transport))


# --- Groq (OpenAI-compatible) ------------------------------------------------


def chat_completion_body(
    content: str | None,
    *,
    finish_reason: str = "stop",
    **message_extra: Any,
) -> dict[str, Any]:
    """A Groq chat completion body shaped like the real one (GPT-OSS adds ``reasoning``)."""
    message = {"role": "assistant", "content": content, "reasoning": "Short reasoning text.", **message_extra}
    return {
        "id": "chatcmpl-test-0001",
        "object": "chat.completion",
        "created": 1760000000,
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


def groq_error_body(message: str, error_type: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    """A Groq error body shaped like the real one."""
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return {"error": error}


def groq_respond(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """Handler that always answers with the same status and JSON body."""
    return lambda request: httpx.Response(status, json=body, headers=headers)


class FakeGroqAPI(_RecordingServer):
    """Stands in for api.groq.com and records every request it receives."""

    def client(self, settings: GroqSettings) -> openai.OpenAI:
        transport = httpx.MockTransport(self)
        return create_groq_client(settings, max_retries=0, http_client=httpx.Client(transport=transport))
