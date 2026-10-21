"""Tuesday: JSON parsing and schema validation. Pure functions, so no mocks at all."""

from __future__ import annotations

import pytest

from src.response_parser import (
    CATEGORIES,
    CLASSIFICATION_JSON_SCHEMA,
    REQUIRED_FIELDS,
    EmptyResponseError,
    MalformedJSONError,
    ResponseParseError,
    SchemaValidationError,
    parse_classification_response,
    parse_json_response,
    validate_classification,
)

VALID = {"category": "bug_report", "confidence": 0.92, "explanation": "The user reports a crash."}

# --- parse_json_response -----------------------------------------------------


def test_parse_valid_json_object():
    assert parse_json_response('{"category": "billing", "confidence": 1, "explanation": "x"}') == {
        "category": "billing",
        "confidence": 1,
        "explanation": "x",
    }


def test_parse_tolerates_surrounding_whitespace():
    assert parse_json_response('\n  {"a": 1}  \n') == {"a": 1}


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_parse_accepts_reply_that_is_one_code_fence(fence):
    assert parse_json_response(f'{fence}\n{{"a": 1}}\n```') == {"a": 1}


@pytest.mark.parametrize("text", [None, "", "   ", "\n\t"])
def test_parse_empty_reply_raises_empty_response_error(text):
    with pytest.raises(EmptyResponseError):
        parse_json_response(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"category": "billing",}',  # trailing comma
        "{'category': 'billing'}",  # single quotes
        '{"category": "billing"',  # truncated
        'Sure! Here is the JSON: {"category": "billing"}',  # prose before JSON
        '{"category": "billing"} Hope that helps!',  # prose after JSON
        "Here you go:\n```json\n{\"a\": 1}\n```",  # fence inside prose is not unwrapped
        "category: billing",
    ],
)
def test_parse_malformed_json_raises(text):
    with pytest.raises(MalformedJSONError, match="not valid JSON"):
        parse_json_response(text)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parse_rejects_non_standard_number_constants(constant):
    with pytest.raises(MalformedJSONError, match=constant):
        parse_json_response(f'{{"confidence": {constant}}}')


def test_malformed_json_error_keeps_the_original_decode_error():
    with pytest.raises(MalformedJSONError) as caught:
        parse_json_response("{oops}")

    assert caught.value.__cause__ is not None


# --- validate_classification -------------------------------------------------


def test_validate_returns_clean_result():
    result = validate_classification({"category": "billing", "confidence": 1, "explanation": "  Refund request.  "})

    assert result == {"category": "billing", "confidence": 1.0, "explanation": "Refund request."}
    assert isinstance(result["confidence"], float)


@pytest.mark.parametrize("category", CATEGORIES)
def test_validate_accepts_every_category(category):
    assert validate_classification({**VALID, "category": category})["category"] == category


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_validate_reports_missing_required_field(field):
    data = {key: value for key, value in VALID.items() if key != field}

    with pytest.raises(SchemaValidationError, match=f"Missing required field.*{field}"):
        validate_classification(data)


def test_validate_reports_all_missing_fields():
    with pytest.raises(SchemaValidationError, match="category, confidence, explanation"):
        validate_classification({})


def test_validate_rejects_unexpected_fields():
    with pytest.raises(SchemaValidationError, match="Unexpected field.*sentiment"):
        validate_classification({**VALID, "sentiment": "negative"})


@pytest.mark.parametrize("data", [[VALID], "bug_report", 42, None, True])
def test_validate_rejects_non_object_json(data):
    with pytest.raises(SchemaValidationError, match="Expected a JSON object"):
        validate_classification(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("category", "Bug Report", "must be one of"),
        ("category", "refund", "must be one of"),
        ("category", 3, "must be one of"),
        ("confidence", "0.9", "must be a number"),
        ("confidence", True, "must be a number"),
        ("confidence", None, "must be a number"),
        ("confidence", 1.5, "between 0 and 1"),
        ("confidence", -0.1, "between 0 and 1"),
        ("confidence", float("nan"), "between 0 and 1"),
        ("explanation", "", "non-empty string"),
        ("explanation", "   ", "non-empty string"),
        ("explanation", ["a", "b"], "non-empty string"),
        ("explanation", "x" * 501, "longer than 500"),
    ],
)
def test_validate_rejects_wrong_types_and_values(field, value, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_classification({**VALID, field: value})


# --- end to end and schema consistency ---------------------------------------


def test_parse_classification_response_success():
    text = '{"category": "feature_request", "confidence": 0.8, "explanation": "Asks for dark mode."}'

    assert parse_classification_response(text) == {
        "category": "feature_request",
        "confidence": 0.8,
        "explanation": "Asks for dark mode.",
    }


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", EmptyResponseError),
        ("not json", MalformedJSONError),
        ('{"category": "billing"}', SchemaValidationError),
    ],
)
def test_parse_classification_response_errors_share_a_base_class(text, error):
    with pytest.raises(error) as caught:
        parse_classification_response(text)

    assert isinstance(caught.value, ResponseParseError)


def test_json_schema_matches_local_validation_rules():
    assert CLASSIFICATION_JSON_SCHEMA["required"] == list(REQUIRED_FIELDS)
    assert set(CLASSIFICATION_JSON_SCHEMA["properties"]) == set(REQUIRED_FIELDS)
    assert CLASSIFICATION_JSON_SCHEMA["properties"]["category"]["enum"] == list(CATEGORIES)
    assert CLASSIFICATION_JSON_SCHEMA["additionalProperties"] is False
