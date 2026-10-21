"""Tuesday task, part 1: turn Claude's reply text into a validated dictionary.

This module is pure Python with no network access, so every rule can be
unit-tested directly. It never assumes the reply is valid JSON just because
the prompt asked for JSON: the text is parsed with the standard ``json``
module and then checked field by field against the expected schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

CATEGORIES = (
    "bug_report",
    "feature_request",
    "billing",
    "account_access",
    "general_question",
    "other",
)
REQUIRED_FIELDS = ("category", "confidence", "explanation")
MAX_EXPLANATION_CHARS = 500

# The same contract as a JSON Schema, sent to the API as a structured-output
# format. Built from the constants above so the two cannot drift apart. The
# 0-1 range on "confidence" is checked locally in validate_classification().
CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": list(REQUIRED_FIELDS),
    "additionalProperties": False,
}

# Matches a reply that is *entirely* one Markdown code fence, e.g.
# ```json\n{...}\n```. Text before or after the fence does not match, so this
# never guesses where JSON starts inside surrounding prose.
_WHOLE_REPLY_CODE_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(?P<body>.*)\n```\Z", re.DOTALL | re.IGNORECASE)


class ClassificationResult(TypedDict):
    category: str
    confidence: float
    explanation: str


class ResponseParseError(Exception):
    """Base class: Claude's reply could not be turned into a ClassificationResult."""


class EmptyResponseError(ResponseParseError):
    """The reply text was empty or whitespace only."""


class MalformedJSONError(ResponseParseError):
    """The reply text is not valid JSON."""


class SchemaValidationError(ResponseParseError):
    """The JSON is valid but does not match the expected schema."""


def parse_json_response(text: str | None) -> Any:
    """Parse ``text`` as JSON with the standard library and return the value.

    Raises:
        EmptyResponseError: ``text`` is None, empty or whitespace only.
        MalformedJSONError: ``text`` is not valid JSON, or uses NaN/Infinity,
            which Python accepts by default but the JSON standard does not.
    """
    if text is None or not text.strip():
        raise EmptyResponseError("Claude returned an empty reply; expected a JSON object.")

    candidate = text.strip()
    fenced = _WHOLE_REPLY_CODE_FENCE.match(candidate)
    if fenced:
        candidate = fenced.group("body").strip()

    try:
        return json.loads(candidate, parse_constant=_reject_non_standard_constant)
    except json.JSONDecodeError as exc:
        raise MalformedJSONError(
            f"Reply is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno}). "
            f"Reply starts with: {_preview(text)}"
        ) from exc
    except ValueError as exc:  # raised by _reject_non_standard_constant
        raise MalformedJSONError(f"Reply is not valid JSON ({exc}).") from exc


def _reject_non_standard_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a valid JSON value")


def _preview(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return repr(flat if len(flat) <= limit else flat[:limit] + "...")


def validate_classification(data: Any) -> ClassificationResult:
    """Check parsed JSON against the classification schema and return a clean copy.

    Raises:
        SchemaValidationError: with a message naming the first problem found.
    """
    if not isinstance(data, dict):
        raise SchemaValidationError(f"Expected a JSON object, got {type(data).__name__}.")

    missing = [name for name in REQUIRED_FIELDS if name not in data]
    if missing:
        raise SchemaValidationError(f"Missing required field(s): {', '.join(missing)}.")

    unexpected = sorted(set(data) - set(REQUIRED_FIELDS))
    if unexpected:
        raise SchemaValidationError(f"Unexpected field(s): {', '.join(unexpected)}.")

    category = data["category"]
    if not isinstance(category, str) or category not in CATEGORIES:
        raise SchemaValidationError(f"'category' must be one of {', '.join(CATEGORIES)}; got {category!r}.")

    confidence = data["confidence"]
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise SchemaValidationError(f"'confidence' must be a number; got {type(confidence).__name__}.")
    if not 0.0 <= confidence <= 1.0:
        raise SchemaValidationError(f"'confidence' must be between 0 and 1; got {confidence!r}.")

    explanation = data["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise SchemaValidationError("'explanation' must be a non-empty string.")
    if len(explanation) > MAX_EXPLANATION_CHARS:
        raise SchemaValidationError(f"'explanation' is longer than {MAX_EXPLANATION_CHARS} characters.")

    return {
        "category": category,
        "confidence": float(confidence),
        "explanation": explanation.strip(),
    }


def parse_classification_response(text: str | None) -> ClassificationResult:
    """Parse and validate Claude's reply in one step.

    Raises:
        EmptyResponseError, MalformedJSONError, SchemaValidationError
        (all subclasses of ResponseParseError).
    """
    return validate_classification(parse_json_response(text))
