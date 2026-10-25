"""Command-line behaviour: output, exit codes and secret hygiene."""

from __future__ import annotations

import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from src import main as cli
from tests.fakes import (
    FAKE_API_KEY,
    FAKE_GROQ_API_KEY,
    FakeAnthropicAPI,
    FakeGroqAPI,
    chat_completion_body,
    error_body,
    groq_error_body,
    groq_respond,
    message_body,
    respond,
    text_block,
)

GOOD_REPLY = '{"category": "billing", "confidence": 0.88, "explanation": "Asks about a double charge."}'


@pytest.fixture
def fake_api(monkeypatch):
    """Configure a fake key and route the CLI's client through a fake API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    def install(handler) -> FakeAnthropicAPI:
        api = FakeAnthropicAPI(handler)
        monkeypatch.setattr(cli, "create_client", lambda settings: api.client(settings))
        return api

    return install


def test_classify_prints_json_and_exits_zero(fake_api, capsys):
    fake_api(respond(200, message_body(text_block(GOOD_REPLY))))

    exit_code = cli.main(["classify", "I", "was", "charged", "twice"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_OK
    assert json.loads(captured.out) == {
        "category": "billing",
        "confidence": 0.88,
        "explanation": "Asks about a double charge.",
    }


def test_classify_prompts_for_text_when_no_argument(fake_api, monkeypatch, capsys):
    api = fake_api(respond(200, message_body(text_block(GOOD_REPLY))))
    monkeypatch.setattr("builtins.input", lambda prompt: "I was charged twice")

    assert cli.main(["classify"]) == cli.EXIT_OK
    assert "I was charged twice" in api.last_json["messages"][0]["content"]


def test_classify_without_key_exits_with_config_error(capsys):
    exit_code = cli.main(["classify", "hello"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_CONFIG
    assert "ANTHROPIC_API_KEY is not configured" in captured.err
    assert "No request was sent" in captured.err
    assert captured.out == ""


def test_classify_reports_invalid_model_json(fake_api, capsys):
    fake_api(respond(200, message_body(text_block("billing"))))

    exit_code = cli.main(["classify", "I was charged twice"])

    assert exit_code == cli.EXIT_FAILURE
    assert "could not be used" in capsys.readouterr().err


def test_auth_failure_exits_one_without_printing_key(fake_api, capsys):
    fake_api(respond(401, error_body("authentication_error", "invalid x-api-key")))

    exit_code = cli.main(["ping"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "HTTP 401" in captured.err
    assert FAKE_API_KEY not in captured.out + captured.err


def test_ping_prints_reply_and_usage(fake_api, capsys):
    fake_api(respond(200, message_body(text_block("The API test call worked."))))

    exit_code = cli.main(["ping"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "Reply : The API test call worked." in out
    assert "Tokens: 42 input / 17 output" in out


def test_check_env_works_without_credentials(capsys):
    exit_code = cli.main(["check-env"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "Python version" in out
    assert "anthropic" in out
    assert "ANTHROPIC_API_KEY : not configured" in out


def test_check_env_hides_configured_key(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    cli.main(["check-env"])

    out = capsys.readouterr().out
    assert "configured (value hidden)" in out
    assert FAKE_API_KEY not in out


@pytest.fixture
def fake_packages(monkeypatch):
    """Make check-env see every package as installed (version 9.9.9) except the ones passed to the returned function."""

    def install(*missing: str) -> None:
        def version(package: str) -> str:
            if package in missing:
                raise metadata.PackageNotFoundError(package)
            return "9.9.9"

        fake = SimpleNamespace(version=version, PackageNotFoundError=metadata.PackageNotFoundError)
        monkeypatch.setattr(cli, "metadata", fake)

    return install


def test_check_env_lists_the_day3_and_day4_packages(fake_packages, capsys):
    fake_packages()

    exit_code = cli.main(["check-env"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    for package in ("numpy", "sentence-transformers", "torch"):
        assert f"{package:<18}: 9.9.9\n" in out
    assert "Missing required" not in out


@pytest.mark.parametrize("missing", [("numpy",), ("sentence-transformers",), ("torch",), ("numpy", "torch")])
def test_check_env_fails_and_names_missing_day3_and_day4_packages(fake_packages, capsys, missing):
    fake_packages(*missing)

    exit_code = cli.main(["check-env"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_FAILURE
    for package in missing:
        assert f"{package:<18}: NOT INSTALLED\n" in out
    assert f"  ! Missing required package(s): {', '.join(missing)}. See README.md, section 'Setup'.\n" in out


def test_check_env_does_not_require_pytest(fake_packages, capsys):
    fake_packages("pytest")

    exit_code = cli.main(["check-env"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "pytest            : NOT INSTALLED\n" in out
    assert "Missing required" not in out


# --- optional Groq provider --------------------------------------------------


@pytest.fixture
def fake_groq_api(monkeypatch):
    """Configure a fake Groq key and route the CLI's Groq client through a fake API."""
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_API_KEY)

    def install(handler) -> FakeGroqAPI:
        api = FakeGroqAPI(handler)
        monkeypatch.setattr(cli, "create_groq_client", lambda settings: api.client(settings))
        return api

    return install


def test_groq_classify_prints_json_and_exits_zero(fake_groq_api, capsys):
    api = fake_groq_api(groq_respond(200, chat_completion_body(GOOD_REPLY)))

    exit_code = cli.main(["groq-classify", "I", "was", "charged", "twice"])

    assert exit_code == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["category"] == "billing"
    assert api.last_json["response_format"]["type"] == "json_schema"


def test_groq_classify_json_object_flag(fake_groq_api, capsys):
    api = fake_groq_api(groq_respond(200, chat_completion_body(GOOD_REPLY)))

    assert cli.main(["groq-classify", "--json-object", "I was charged twice"]) == cli.EXIT_OK
    assert api.last_json["response_format"] == {"type": "json_object"}


def test_groq_classify_without_key_exits_with_config_error(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)  # a Claude key must not be used for Groq

    exit_code = cli.main(["groq-classify", "hello"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_CONFIG
    assert "GROQ_API_KEY is not configured" in captured.err
    assert "No request was sent" in captured.err


def test_groq_key_alone_does_not_enable_claude_commands(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_API_KEY)

    exit_code = cli.main(["ping"])

    assert exit_code == cli.EXIT_CONFIG
    assert "ANTHROPIC_API_KEY is not configured" in capsys.readouterr().err


def test_groq_auth_failure_exits_one_without_printing_key(fake_groq_api, capsys):
    fake_groq_api(groq_respond(401, groq_error_body("Invalid API Key", code="invalid_api_key")))

    exit_code = cli.main(["groq-classify", "hello"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "HTTP 401" in captured.err
    assert FAKE_GROQ_API_KEY not in captured.out + captured.err


def test_groq_classify_reports_malformed_json(fake_groq_api, capsys):
    fake_groq_api(groq_respond(200, chat_completion_body("billing")))

    assert cli.main(["groq-classify", "hello"]) == cli.EXIT_FAILURE
    assert "could not be used" in capsys.readouterr().err


def test_check_env_reports_groq_key_hidden(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_API_KEY)

    cli.main(["check-env"])

    out = capsys.readouterr().out
    assert "GROQ_API_KEY      : configured (value hidden)" in out
    assert "ANTHROPIC_API_KEY : not configured" in out
    assert FAKE_GROQ_API_KEY not in out
