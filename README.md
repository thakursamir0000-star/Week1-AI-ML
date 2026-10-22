# Week1-AI-ML

Week 1 work from my AI/ML Engineer internship: an LLM API client with robust
error handling, a script that asks an LLM to classify text and validates
the JSON it returns, and a local sentence-embedding demo that compares texts
with cosine similarity.

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
| Day 3 (Wed) | Embeddings | Done. Five example texts compared with a local embedding model, and all tests pass (2026-10-22). This is a demonstration, not an evaluation: see [the similarity experiment](#similarity-experiment-day-3). |
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

### Day 3: embeddings and cosine similarity

- `src/embeddings.py` turns five example texts into embeddings with the
  local model
  [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (384 numbers per text, CPU only, no API key).
- It computes the cosine similarity between every pair of texts, prints a
  5 x 5 similarity matrix, and ranks the 10 pairs from most to least similar.
- The cosine-similarity maths is pure NumPy, separate from model execution,
  so it is tested without a model. `sentence_transformers` and PyTorch are
  imported only when the model is loaded.
- `python -m src.embeddings` runs the demo (see [Usage](#compare-five-texts-with-embeddings-day-3)).
- `tests/test_embeddings.py` adds 108 tests. 105 tests run without the model
  (pure-maths tests or tests with fake encoders) and never load PyTorch. 3
  load the real model from the local cache with the network blocked, and are
  skipped if it is not cached.
- Results and their limits are in
  [the similarity experiment](#similarity-experiment-day-3). The existing API
  clients, classifier, CLI and response parser are unchanged.

## Project structure

```
Week1-AI-ML/
├── src/
│   ├── api_client.py       # Anthropic client: settings, send_message(), extract_text(), errors
│   ├── groq_client.py      # Groq client (OpenAI SDK): settings, send_groq_chat(), errors, classify_text_with_groq()
│   ├── classifier.py       # shared classification prompt + Anthropic classify_text() workflow
│   ├── response_parser.py  # JSON parsing and schema validation (pure Python, no network)
│   ├── embeddings.py       # Day 3: cosine-similarity maths (NumPy only), local embedding model, demo CLI
│   └── main.py             # CLI: check-env, ping, classify, groq-classify
├── tests/                  # 270 offline tests (no keys, no network; 3 load the cached embedding model)
├── .env.example            # configuration template (placeholders only)
├── requirements.txt
└── pytest.ini              # also registers the `model` test marker
```

`response_parser.py` holds no network code, so its rules are tested
directly. Both providers reuse the same prompt, input limits and validation.
In the same way, the similarity functions in `embeddings.py` need only NumPy.

## Setup

Requires Python 3.10 or newer. Tested with Python 3.12 on Ubuntu (WSL2).

```bash
git clone https://github.com/thakursamir0000-star/Week1-AI-ML.git
cd Week1-AI-ML
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"   # CPU-only PyTorch, for Day 3
pip install -r requirements.txt
python -m src.main check-env      # offline: shows Python, package versions, which keys are set
```

On Ubuntu, `python3 -m venv` may first need `sudo apt install python3-venv`.
Any existing virtual environment with the packages from `requirements.txt`
works too. `.venv/` is excluded from Git.

Install PyTorch from the CPU-only index **before** `requirements.txt`.
Otherwise `sentence-transformers` pulls PyTorch from PyPI, including several GB
of CUDA packages.

**Embedding model download.** The first run of `python -m src.embeddings`
downloads `all-MiniLM-L6-v2` (about 90 MB) from the Hugging Face Hub into
`~/.cache/huggingface`. Set `HF_HOME` to use another folder. After that, it
loads from the cache and makes no network requests. `--offline` enforces this.
No API key or Hugging Face account is needed, and the tests never download the
model.

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

### Compare five texts with embeddings (Day 3)

```bash
python -m src.embeddings             # download the model on the first run, then compare the texts
python -m src.embeddings --offline   # local cache only: fail instead of downloading
python -m src.embeddings --top-k 5   # list the 5 most similar pairs instead of 3
```

No API key is used. Real output from 2026-10-22 (WSL2 Ubuntu, CPU, about 6
seconds including the PyTorch import):

```
Loading sentence-transformers/all-MiniLM-L6-v2 ...
Embedded 5 texts; each embedding has 384 dimensions.

Texts:
  T1  I was charged twice for my subscription this month.
  T2  My card was billed two times for the same monthly plan.
  T3  The app crashes whenever I export a report as a PDF.
  T4  Exporting a PDF makes the application close unexpectedly.
  T5  I charge my phone twice a day because the battery drains fast.

Cosine similarity (1 = same direction, 0 = orthogonal, -1 = opposite):
          T1      T2      T3      T4      T5
  T1   1.000   0.739   0.136   0.055   0.387
  T2   0.739   1.000   0.131   0.059   0.374
  T3   0.136   0.131   1.000   0.738   0.151
  T4   0.055   0.059   0.738   1.000   0.063
  T5   0.387   0.374   0.151   0.063   1.000

Most similar pairs:
  1. T1 & T2  0.739
  2. T3 & T4  0.738
  3. T1 & T5  0.387

Least similar pair: T1 & T4  0.055
```

While the model loads, `transformers` also prints a `Loading weights` progress
bar to stderr.

| Exit code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | The model could not be loaded: not cached and no download possible, or `sentence-transformers` is not installed |
| 2 | Invalid arguments, e.g. `--top-k 0` |

From Python, invalid input raises `ValueError` with a message naming the
problem. That covers an empty or blank text, a single string instead of a
list, vectors of different lengths, a zero vector, NaN or infinity, and
non-numeric values.

### From Python

```python
from src.groq_client import classify_text_with_groq
from src.classifier import classify_text        # Anthropic
from src.api_client import ask_claude           # Anthropic, plain text reply

result = classify_text_with_groq("Please add a dark mode.")
# e.g. {"category": "feature_request", "confidence": ..., "explanation": "..."}

from src.embeddings import compare_texts, cosine_similarity, load_model

cosine_similarity([1, 0], [1, 1])               # 0.7071... (pure maths, no model)
report = compare_texts(load_model(), ["The app crashes.", "The program keeps closing."])
report.similarities                             # 2 x 2 NumPy array
report.pairs[0]                                 # SimilarPair(first=0, second=1, score=...)
```

## Similarity experiment (Day 3)

**Question.** Does the embedding model score two texts with the same meaning
higher than texts that are about different things, even when the wording
differs or the words overlap?

**Set-up.** The five texts in `EXAMPLE_TEXTS` were written for this demo:

| Text | Role |
|------|------|
| T1, T2 | The same billing problem, with almost no content words in common ("charged twice" vs "billed two times") |
| T3, T4 | The same crash, worded differently |
| T5 | Shares "charge" and "twice" with T1, but is about a phone battery |

Each text is embedded once. The cosine similarity
`cos(a, b) = (a · b) / (|a| |b|)` is then computed for all 10 pairs. The
environment was Python 3.12.15, sentence-transformers 6.1.0, transformers
5.20.0 and torch 2.14.1+cpu. Scores can differ slightly with other library
versions or hardware.

**Observed results** (the matrix above):

- The two paraphrase pairs have the two highest scores: T1-T2 0.739 and T3-T4
  0.738. T1 and T2 score high despite sharing almost no content words.
- The two pairs differ by only 0.001, so their order is not meaningful.
- Pairs across the two topics (billing vs crash) score 0.055 to 0.136.
- T5, the word-overlap sentence, scores 0.387 with T1 and 0.374 with T2. That
  is well below the paraphrase scores, but clearly above the cross-topic
  pairs. Shared words, and a loosely shared "charging" theme, do raise the
  score. The model did not treat T5 as unrelated.
- No pair scores near 0 or below. These scores are only meaningful compared
  with each other. They are not probabilities, and 0.5 is not a natural
  cut-off for "similar".

**What this does and does not show.**

- It is a **successful demonstration**. The pipeline works end to end
  (text, embedding, cosine similarity, ranking), and on these five sentences
  the ranking matches what a person would expect.
- It is **not an evaluation** of semantic accuracy. The five sentences
  (10 pairs) were written by hand to illustrate the idea, in one domain, in
  English.
- The three real-model tests were written after looking at these scores.
  They catch a change in behaviour, for example after a library upgrade, but
  they are not independent evidence that the model is accurate.
- A rigorous evaluation would need:
  - a labelled dataset that was not used to pick the examples, for example
    the STS Benchmark (human similarity ratings, scored with Spearman
    correlation), or a retrieval set with relevance labels (recall@k, MRR);
  - many more and more varied pairs, including hard negatives;
  - comparison with a keyword baseline such as TF-IDF, and with other models;
  - a similarity threshold chosen on held-out data, if one is needed.
- **Known limits.**
  - The model reads at most 256 word pieces per text and silently truncates
    longer input. `embeddings.py` does not check for this.
  - The model is trained mainly on English.

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
python -m pytest -q                  # all 270 tests, about 10 seconds (3 load the cached embedding model)
python -m pytest -q -m "not model"   # 267 tests, about 2-3 seconds, never imports PyTorch
```

Results recorded on 2026-10-22 (WSL2 Ubuntu, Python 3.12.15, pytest 9.1.1):

| Command | Result |
|---------|--------|
| `python -m pytest -q` | 270 passed in 7.97 s |
| `python -m pytest -q -m "not model"` | 267 passed, 3 deselected in 2.80 s |
| `python -m pytest -q --ignore=tests/test_embeddings.py` (the Day 1-2 suite) | 162 passed in 2.43 s, unchanged from before Day 3 |
| `python -m pytest -q tests/test_embeddings.py` | 108 passed in 7.02 s |
| `HF_HOME=<empty folder> python -m pytest -q -m model` (model not cached) | 3 skipped, 267 deselected; nothing was downloaded |

The tests need **no API key and no network**:
- `conftest.py` hides real environment variables and `.env`, and blocks
  socket connections.
- Each API is faked at the HTTP layer: Groq with `httpx.MockTransport`,
  Anthropic with `httpx2.MockTransport`.
- The real SDKs still build each request, parse each response and raise their
  own typed exceptions. The parsing and validation code runs unmocked.
- The embedding tests use small fake encoders and a stand-in
  `sentence_transformers` package, so they never load PyTorch or a model.
  The cosine-similarity maths is checked against hand-computed values.
- The 3 tests marked `model` load the real `all-MiniLM-L6-v2` with
  `local_files_only=True` while sockets are blocked. If the model is not
  cached, they are skipped. They never download it.

| Test file | Covers |
|-----------|--------|
| `test_groq_client.py` | config from env/`.env`; request shape (Groq endpoint, Bearer auth, strict schema vs JSON mode, `reasoning_effort` for GPT-OSS models only); empty/missing choices, `null` content, `length`/`content_filter`, refusal; HTTP 401/403/429/400/404/500/503 and network errors; end-to-end classification |
| `test_api_client.py` | Anthropic config, request shape (effort, fallback beta header, no unsupported parameters), text extraction, HTTP 401/403/429/400/404/500/529 and network errors |
| `test_classifier.py` | Anthropic classification workflow end to end |
| `test_response_parser.py` | JSON parsing (valid, fenced, empty, malformed, NaN) and schema validation (missing/extra fields, wrong types and ranges) |
| `test_main.py` | CLI output and exit codes, and that keys never appear in output |
| `test_embeddings.py` | cosine similarity: known values, symmetry, scale invariance, clipping to [-1, 1], extreme magnitudes (1e-200 to 1e200), invalid vectors; the similarity matrix, including extreme magnitudes; pair ranking, ties and `top_k`; `top_k` validation (0, -1, 2.5, booleans) in `rank_pairs()` and `format_report()`; embedding shape for 1 to 768 dimensions; invalid texts and unusable model output; loading from the cache first and downloading only when allowed; CLI output and exit codes; that importing the module does not load PyTorch. With the cached model: 384 dimensions, and the expected ranking of the five example texts |

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
- **Embeddings.**
  - Model: `all-MiniLM-L6-v2`, a small (about 90 MB, 384-dimension),
    Apache-2.0 model. It is a common baseline for English sentence
    similarity and runs on a CPU.
  - Cosine similarity is computed in NumPy rather than with
    `sentence_transformers.util.cos_sim`, so it can be tested without
    PyTorch.
  - Results are clipped to [-1, 1], because rounding often produces
    1.0000000000000002.
  - A zero vector raises `ValueError`, because its cosine similarity is
    undefined. It is not silently scored as 0.
  - The matrix is made exactly symmetric, with 1.0 on the diagonal.
  - `embed_texts()` accepts any object with an `encode()` method, so the
    tests can pass fakes.
  - `embeddings.py` does not import the API clients and needs no key. Its
    only network access is the one-time model download.

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
| `Error: could not load sentence-transformers/all-MiniLM-L6-v2` | The model is not cached (or `--offline` was used) and could not be downloaded. Connect once and run `python -m src.embeddings`. |
| `OSError: [WinError 4551]` when importing torch | Windows Smart App Control blocks PyTorch's DLLs under Windows Python. Run under WSL2 Ubuntu. |
| `pip install` starts downloading `nvidia-*` packages | PyTorch is coming from PyPI with CUDA. Stop it and install the CPU build first (see [Setup](#setup)). |
| `3 skipped` for tests marked `model` | The embedding model is not in the local cache. Run `python -m src.embeddings` once. |
