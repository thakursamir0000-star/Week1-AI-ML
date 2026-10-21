"""Command-line entry point. Run from the project root:

    python -m src.main check-env                 # verify interpreter, packages, credentials (offline)
    python -m src.main ping                      # Monday: one live test call to Claude
    python -m src.main classify "text to label"  # Tuesday: classify text with Anthropic, print validated JSON
    python -m src.main groq-classify "text"      # classify text with Groq, print validated JSON

Exit codes: 0 success, 1 API or response failure, 2 missing/invalid
configuration or input (no request was sent).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata

from src.api_client import (
    ClaudeClientError,
    ClaudeConfigError,
    MissingAPIKeyError,
    create_client,
    extract_text,
    load_settings,
    send_message,
)
from src.classifier import classify_text
from src.groq_client import (
    GroqClientError,
    GroqConfigError,
    MissingGroqAPIKeyError,
    classify_text_with_groq,
    create_groq_client,
    load_groq_settings,
)
from src.response_parser import ResponseParseError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2

REQUIRED_PACKAGES = ("anthropic", "httpx2", "python-dotenv", "openai", "httpx")
TEST_PACKAGES = ("pytest",)
DEFAULT_PING_PROMPT = "Reply with one short sentence confirming that this API test call worked."


def cmd_check_env(args: argparse.Namespace) -> int:
    ok = True
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    if sys.version_info < (3, 10):
        print("  ! anthropic 1.x requires Python 3.10 or newer")
        ok = False

    for package in REQUIRED_PACKAGES + TEST_PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
            ok = ok and package not in REQUIRED_PACKAGES
        print(f"{package:<18}: {version}")

    try:
        settings = load_settings()
    except MissingAPIKeyError:
        print("ANTHROPIC_API_KEY : not configured - Anthropic commands (ping, classify) unavailable")
    except ClaudeConfigError as exc:
        print(f"Configuration     : INVALID - {exc}")
        ok = False
    else:
        print("ANTHROPIC_API_KEY : configured (value hidden)")
        print(f"Model             : {settings.model}")
        print(f"Refusal fallbacks : {'on' if settings.use_fallbacks else 'off'}")

    try:
        groq_settings = load_groq_settings()
    except MissingGroqAPIKeyError:
        print("GROQ_API_KEY      : not configured - groq-classify unavailable")
    else:
        print("GROQ_API_KEY      : configured (value hidden)")
        print(f"Groq model        : {groq_settings.model}")
    return EXIT_OK if ok else EXIT_FAILURE


def cmd_ping(args: argparse.Namespace) -> int:
    settings = load_settings()
    client = create_client(settings)
    response = send_message(client, settings, args.prompt)
    reply = extract_text(response)

    print(f"Model : {getattr(response, 'model', settings.model)}")
    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"Tokens: {usage.input_tokens} input / {usage.output_tokens} output")
    print(f"Reply : {reply}")
    return EXIT_OK


def _read_text(words: list[str]) -> str:
    """Join command-line words, or prompt for the text when none were given."""
    if words:
        return " ".join(words)
    try:
        return input("Text to classify: ")
    except EOFError:
        return ""


def cmd_classify(args: argparse.Namespace) -> int:
    text = _read_text(args.text)
    settings = load_settings()
    client = create_client(settings)
    result = classify_text(text, settings=settings, client=client, use_structured_output=not args.prompt_only)
    print(json.dumps(result, indent=2))
    return EXIT_OK


def cmd_groq_classify(args: argparse.Namespace) -> int:
    text = _read_text(args.text)
    settings = load_groq_settings()
    client = create_groq_client(settings)
    result = classify_text_with_groq(text, settings=settings, client=client, use_json_schema=not args.json_object)
    print(json.dumps(result, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Week 1 LLM API tools: groq-classify (Groq); ping and classify (Anthropic).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check_env = commands.add_parser("check-env", help="verify interpreter, packages and credential status (offline)")
    check_env.set_defaults(handler=cmd_check_env)

    ping = commands.add_parser("ping", help="Monday: send one test message to Claude and print the reply")
    ping.add_argument("--prompt", default=DEFAULT_PING_PROMPT, help="message to send (default: a short test prompt)")
    ping.set_defaults(handler=cmd_ping)

    classify = commands.add_parser("classify", help="Tuesday: classify a short text with Anthropic (Claude) and print validated JSON")
    classify.add_argument("text", nargs="*", help="text to classify; if omitted you are prompted for it")
    classify.add_argument(
        "--prompt-only",
        action="store_true",
        help="skip structured outputs and rely on the JSON-only prompt plus local validation",
    )
    classify.set_defaults(handler=cmd_classify)

    groq = commands.add_parser(
        "groq-classify",
        help="classify a short text with Groq and print validated JSON",
    )
    groq.add_argument("text", nargs="*", help="text to classify; if omitted you are prompted for it")
    groq.add_argument(
        "--json-object",
        action="store_true",
        help="use JSON object mode (valid JSON only) instead of strict JSON-schema mode; still validated locally",
    )
    groq.set_defaults(handler=cmd_groq_classify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (MissingAPIKeyError, MissingGroqAPIKeyError) as exc:
        print(f"Error: {exc}\nNo request was sent. See README.md, section 'Configuration'.", file=sys.stderr)
        return EXIT_CONFIG
    except (ClaudeConfigError, GroqConfigError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ResponseParseError as exc:
        print(f"Error: the model's reply could not be used: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except (ClaudeClientError, GroqClientError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
