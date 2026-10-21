# Week1-AI-ML

Week 1 work from my AI/ML Engineer internship: an LLM API client with robust
error handling, and a script that asks an LLM to classify text and validates
the JSON it returns.

Two providers are implemented behind the same task, prompt and validation:

| Provider | SDK | Default model | Commands | Live-tested |
|----------|-----|---------------|----------|-------------|
| **Groq** | official `openai` SDK, pointed at Groq's OpenAI-compatible endpoint `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` (OpenAI's open-weight model, hosted by Groq) | `groq-classify` | Yes (2026-10-21) |
| **Anthropic** | official `anthropic` SDK (Messages API) | `claude-opus-5-5` | `ping`, `classify` | No, offline tests only |

Groq is the provider used for live API testing.

## Week 1 progress

| Day | Focus | Status |
|-----|-------|--------|
| Day 1 (Mon) | Environment setup and API integration | Done. The live API test passed with Groq on 2026-10-21. The Anthropic client is covered by offline tests only. |
| Day 2 (Tue) | JSON parsing, response validation, classification and offline testing | Done |
| Day 3 (Wed) | Embeddings | Not yet completed |
| Day 4 (Thu) | Semantic search | Not yet completed |
| Day 5 (Fri) | Review | Not yet completed |

### Day 1: environment setup and API integration

- Git repository, a Python virtual environment, and dependency version ranges in
  `requirements.txt`.
- `python -m src.main check-env` reports the interpreter, package versions and
  which API keys are configured, without showing their values.
- Groq client (`src/groq_client.py`): the official `openai` SDK pointed at
  Groq's OpenAI-compatible endpoint.
- Anthropic client (`src/api_client.py`): the official `anthropic` SDK.
- Both clients read keys from environment variables or `.env`. They raise typed
  errors for missing keys, authentication failures, rate limits, other HTTP
  errors, network failures and timeouts, and unusable responses.
- Live API test: `python -m src.main groq-classify "I was charged twice this month."`
  succeeded on 2026-10-21. One request was sent, it exited with code 0, and the
  reply passed schema validation (output under [Usage](#usage)). No live request
  has been made with the Anthropic client.

### Day 2: JSON parsing, response validation and classification

- `src/classifier.py` (Anthropic) and `classify_text_with_groq()` in
  `src/groq_client.py` ask the model to classify a short text into one of six
  categories and to reply with JSON.
- `src/response_parser.py` parses the reply with `json.loads` and validates
  every field against the [response schema](#response-schema). It raises
  `EmptyResponseError`, `MalformedJSONError` or `SchemaValidationError`.
- The CLI commands `groq-classify` (Groq) and `classify` (Anthropic) print the
  validated JSON.
- Offline tests cover parsing, validation and both classification workflows
  using fake HTTP servers. No API key or network is needed (see [Tests](#tests)).

## Project structure

```
Week1-AI-ML/
├── src/
│   ├── api_client.py       # Anthropic client: settings, send_message(), extract_text(), errors
│   ├── groq_client.py      # Groq client (OpenAI SDK): settings, send_groq_chat(), errors, classify_text_with_groq()
│   ├── classifier.py       # shared classification prompt + Anthropic classify_text() workflow
│   ├── response_parser.py  # JSON parsing and schema validation (pure Python, no network)
│   └── main.py             # CLI: check-env, ping, classify, groq-classify
├── tests/                  # 162 offline tests (fake HTTP servers, no keys, no network)
├── .env.example            # configuration template (placeholders only)
├── requirements.txt
└── pytest.ini
```

`response_parser.py` holds no network code, so its rules are tested
directly. Both providers reuse the same prompt, input limits and validation.

## Setup

Requires Python 3.10 or newer. Tested with Python 3.12 on Ubuntu (WSL2).

```bash
git clone https://github.com/thakursamir0000-star/Week1-AI-ML.git
cd Week1-AI-ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main check-env      # offline: shows Python, package versions, which keys are set
```

On Ubuntu, `python3 -m venv` may first need `sudo apt install python3-venv`.
Any existing virtual environment with the packages from `requirements.txt`
works too. `.venv/` is excluded from Git.

## Configuration

Create `.env` from the template **only if it does not exist yet**, then add
your keys in an editor:

```bash
test -f .env || cp .env.example .env
nano .env
```

| Variable | Needed for | Default | Notes |
|----------|------------|---------|-------|
| `GROQ_API_KEY` | `groq-classify` | none | From https://console.groq.com/keys |
| `GROQ_MODEL` | | `openai/gpt-oss-20b` | Any Groq chat model with JSON-schema support |
| `ANTHROPIC_API_KEY` | `ping`, `classify` | none | From the Anthropic Console |
| `CLAUDE_MODEL` | | `claude-opus-5-5` | |
| `CLAUDE_USE_FALLBACKS` | | `true` | Server-side refusal fallback; see [Design notes](#design-notes) |

- A non-empty environment variable takes precedence over `.env`.
- An empty value or the template placeholder counts as missing, and no request is sent.
- Keys are never printed: `check-env` only reports `configured (value hidden)`.
- `.env` is listed in `.gitignore`. Never force-add it, and never overwrite an
  existing `.env` with the template.

## Usage

### Classify text with Groq (live-tested)

```bash
python -m src.main groq-classify "I was charged twice this month."
python -m src.main groq-classify --json-object "Please add a dark mode."
python -m src.main groq-classify                 # prompts: Text to classify:
```

Live result recorded on 2026-10-21 (one request, succeeded on the first
attempt, exit code 0, passed schema validation):

```json
{
  "category": "billing",
  "confidence": 0.99,
  "explanation": "The user reports being charged twice, which is a billing issue."
}
```

Each live run sends one real request to Groq and uses API quota. Input is
capped at 4,000 characters and output at 1,024 tokens.

### Anthropic commands

```bash
python -m src.main ping                          # one test message, prints reply and token usage
python -m src.main classify "The app crashes every time I export a PDF."
python -m src.main classify --prompt-only "I was charged twice this month."
```

These need `ANTHROPIC_API_KEY`. Each call is capped at `max_tokens=1024`
(thinking included), so it costs at most about US$0.03 at Claude Opus 5.5
list prices ($4 / $20 per million input / output tokens).

### From Python

```python
from src.groq_client import classify_text_with_groq
from src.classifier import classify_text        # Anthropic
from src.api_client import ask_claude           # Anthropic, plain text reply

result = classify_text_with_groq("Please add a dark mode.")
# e.g. {"category": "feature_request", "confidence": ..., "explanation": "..."}
```

## Response schema

Both providers return the same dictionary. No other fields are allowed.

| Field | Type | Rule |
|-------|------|------|
| `category` | string | One of `bug_report`, `feature_request`, `billing`, `account_access`, `general_question`, `other` |
| `confidence` | number | 0 to 1 inclusive (booleans and strings are rejected) |
| `explanation` | string | Non-empty, at most 500 characters |

### How a reply is handled

1. **Input check.** The text must be non-empty and at most 4,000 characters.
   Longer input is rejected, never silently truncated.
2. **Prompt.** A JSON-only system prompt is sent. The text to classify is
   wrapped in `<text>` tags and treated as data.
3. **Structured output.** It is on by default, in each provider's own format:
   - Groq: `response_format` `json_schema` with `"strict": true`. With
     `--json-object`, it switches to `json_object`, which guarantees valid
     JSON syntax only.
   - Anthropic: `output_config.format` with a JSON Schema. With
     `--prompt-only`, the prompt alone is relied on.
4. **The reply is never trusted.**
   - Only the text content is read: Groq's `reasoning` field and Claude's
     `thinking` blocks are ignored.
   - Truncated, filtered and refused replies are rejected.
   - The text is parsed with `json.loads`, and `NaN`/`Infinity` are rejected.
   - Every field is validated. The code never slices out "the part that looks
     like JSON". The only leniency: a reply that is entirely one Markdown code
     fence is unwrapped.

## Errors

Every failure raises a documented exception. The CLI prints a one-line
message to stderr (never a key) and exits with the code shown.

| Failure | Groq (`groq_client`) | Anthropic (`api_client`) | Exit |
|---------|----------------------|--------------------------|------|
| Key missing, empty or placeholder (no request sent) | `MissingGroqAPIKeyError` | `MissingAPIKeyError` | 2 |
| Invalid setting, e.g. `CLAUDE_USE_FALLBACKS=maybe` | | `ClaudeConfigError` | 2 |
| Empty or too-long input (no request sent) | `ValueError` | `ValueError` | 2 |
| HTTP 401 / 403 | `GroqAuthenticationError` | `ClaudeAuthenticationError` | 1 |
| HTTP 429 after the SDK's retries (`.retry_after` kept) | `GroqRateLimitError` | `ClaudeRateLimitError` | 1 |
| Other HTTP errors (`.status_code` kept; Groq also keeps `.code`) | `GroqAPIError` | `ClaudeAPIError` | 1 |
| Network failure or timeout | `GroqConnectionError` | `ClaudeConnectionError` | 1 |
| Refusal, truncation, filtering, or no text content | `GroqUnexpectedResponseError` | `UnexpectedResponseError` | 1 |
| Empty reply / invalid JSON / schema violation | `EmptyResponseError` / `MalformedJSONError` / `SchemaValidationError` (shared) | same | 1 |

The base classes are `GroqClientError`, `ClaudeClientError` and
`ResponseParseError`. Each SDK's original exception is kept as `__cause__`.

## Tests

```bash
python -m pytest -q      # 162 tests, about 1 second
```

The tests need **no API key and no network**:
- `conftest.py` hides real environment variables and `.env`, and blocks
  socket connections.
- Each API is faked at the HTTP layer: Groq with `httpx.MockTransport`,
  Anthropic with `httpx2.MockTransport`.
- The real SDKs still build each request, parse each response and raise their
  own typed exceptions. The parsing and validation code runs unmocked.

| Test file | Covers |
|-----------|--------|
| `test_groq_client.py` | config from env/`.env`; request shape (Groq endpoint, Bearer auth, strict schema vs JSON mode, `reasoning_effort` for GPT-OSS models only); empty/missing choices, `null` content, `length`/`content_filter`, refusal; HTTP 401/403/429/400/404/500/503 and network errors; end-to-end classification |
| `test_api_client.py` | Anthropic config, request shape (effort, fallback beta header, no unsupported parameters), text extraction, HTTP 401/403/429/400/404/500/529 and network errors |
| `test_classifier.py` | Anthropic classification workflow end to end |
| `test_response_parser.py` | JSON parsing (valid, fenced, empty, malformed, NaN) and schema validation (missing/extra fields, wrong types and ranges) |
| `test_main.py` | CLI output and exit codes, and that keys never appear in output |

## Design notes

- **Groq.**
  - The client is the official OpenAI SDK with
    `base_url="https://api.groq.com/openai/v1"`.
  - The default model, `openai/gpt-oss-20b`, was a production (non-preview)
    model with strict JSON-schema support in Groq's documentation as of
    2026-10-21. `openai/gpt-oss-120b` also qualifies.
  - GPT-OSS models get `reasoning_effort="low"`.
  - Groq retires models regularly; check its deprecations page if you get an
    HTTP 404.
- **Anthropic.**
  - Thinking is always on for `claude-opus-5-5`, and `output_config.effort`
    (`"low"` here) controls depth and cost.
  - `temperature`/`top_p`/`top_k` are never sent.
  - Requests opt into the server-side refusal fallback (`fallbacks: "default"`,
    beta header `server-side-fallback-2026-07-01`). If your organization isn't
    enabled for this beta, set `CLAUDE_USE_FALLBACKS=false`.
- **Secrets.** Each key goes only to its own SDK. The settings classes hide
  keys from `repr`, and tests check that no key appears in errors or CLI
  output.
- **Retries.** Both SDKs retry connection errors, 408, 409, 429 and 5xx
  responses up to twice with backoff. There is no extra retry loop.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `... API_KEY is not configured` | Add the key to `.env` (see [Configuration](#configuration)). |
| HTTP 401 | The key is mistyped or revoked. |
| HTTP 429 | Rate limited. Wait for the `retry-after` time. |
| Groq HTTP 404 `model_not_found` | The model was retired or mistyped. Set `GROQ_MODEL` to a current model. |
| Groq HTTP 400 `json_validate_failed` | Groq could not produce schema-valid JSON. Retry or rephrase. |
| Anthropic HTTP 400 mentioning `anthropic-beta` | Set `CLAUDE_USE_FALLBACKS=false`. |
| Anthropic HTTP 400 mentioning credit balance | API billing is not set up for that Anthropic organization. |
| `Could not reach the ... API` | Check the internet or proxy settings. |
