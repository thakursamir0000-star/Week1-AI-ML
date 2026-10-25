# Week1-AI-ML

Week 1 work from my AI/ML Engineer internship: an LLM API client with robust
error handling, a script that asks an LLM to classify text and validates
the JSON it returns, a local sentence-embedding demo that compares texts
with cosine similarity, and semantic search over 20 documents, evaluated
against a keyword baseline on labelled queries.

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
| Day 4 (Thu) | Semantic search | Done. Search over 20 documents, evaluated on 23 labelled queries against a keyword baseline, and all tests pass (2026-10-22). The set is small and easy, so the scores are not a general accuracy figure: see [the search evaluation](#search-evaluation-day-4). |
| Day 5 (Fri) | Review | In progress. Done: an AI-assisted code review of Days 1-4, and the accuracy and cost note in `docs/day5_accuracy_cost.md`. The two confirmed review findings are fixed (`check-env` now checks the Day 3-4 packages, and the tie discussion covers `q09` and `q16`), with 6 new tests, and all 469 tests pass (2026-10-25). Pending: peer review by a human mentor, not yet confirmed. |

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

### Day 4: semantic search and top-3 evaluation

- **Dataset.** `data/documents.json` holds 20 short documents on six
  subjects, each with a stable ID such as `db-02`. `data/queries.json` holds
  23 search queries, each listing the IDs of the documents relevant to it
  (27 labels in all). Both are plain JSON, one entry per line (see
  [Dataset](#dataset)).
- **Semantic search.** `SemanticSearcher` in `src/semantic_search.py`
  embeds the 20 documents once with Day 3's `embed_texts()` and keeps the
  vectors. Each query is then embedded on its own and compared with every
  document using Day 3's `cosine_similarity()`. The top 3 come back with
  their rank, ID, text and score. No embedding or similarity code is
  duplicated, and there is no framework or vector database.
- **Keyword baseline.** `KeywordSearcher` ranks documents by the rare words
  they share with the query (IDF-weighted word overlap, no model). Both
  methods are scored on the same queries and labels.
- **Evaluation.** `evaluate()` reports, for every query, its relevant IDs,
  its retrieved IDs, whether it is a hit, its recall and precision, and the
  rank of its first relevant document. It also reports four averages,
  defined under [Metrics](#metrics): the **top-3 hit rate**, **Recall@3**,
  **Precision@3** and **MRR@3**.
- **Validation.** It rejects:
  - empty or non-string queries;
  - a `top_k` that is not a whole number of at least 1 (0, -1, 2.5, `True`);
  - duplicate document or query IDs, and blank IDs or texts;
  - JSON entries with missing or extra fields, or a repeated key;
  - queries without relevant IDs, and relevant IDs that are not in the
    collection.
- **Ties** are broken by document ID, so results do not depend on the order
  of the documents in the file.
- **CLI.** `python -m src.semantic_search search "..."` and
  `python -m src.semantic_search evaluate` (see [Usage](#semantic-search-day-4)).
- **Tests.** `tests/test_semantic_search.py` adds 193 tests. 189 use fake
  encoders or fixed vectors and never load PyTorch. 4 load the real model
  from the local cache: 3 check that the pipeline works, and 1 is a
  regression test that fails if the measured results under
  [Results](#results) change (a lost hit, or a relevant document moved
  down).
- The Day 1-3 code is unchanged; Day 4 only adds files.

## Project structure

```
Week1-AI-ML/
├── src/
│   ├── api_client.py       # Anthropic client: settings, send_message(), extract_text(), errors
│   ├── groq_client.py      # Groq client (OpenAI SDK): settings, send_groq_chat(), errors, classify_text_with_groq()
│   ├── classifier.py       # shared classification prompt + Anthropic classify_text() workflow
│   ├── response_parser.py  # JSON parsing and schema validation (pure Python, no network)
│   ├── embeddings.py       # Day 3: cosine-similarity maths (NumPy only), local embedding model, demo CLI
│   ├── semantic_search.py  # Day 4: dataset loading, semantic + keyword search, hit rate / Recall@k / Precision@k / MRR@k, CLI
│   └── main.py             # CLI: check-env, ping, classify, groq-classify
├── data/
│   ├── documents.json      # Day 4: the 20 documents (id, text)
│   └── queries.json        # Day 4: the 23 labelled queries (id, text, relevant_ids)
├── tests/                  # 469 offline tests (no keys, no network; 7 load the cached embedding model)
├── .env.example            # configuration template (placeholders only)
├── requirements.txt
└── pytest.ini              # also registers the `model` test marker
```

`response_parser.py` holds no network code, so its rules are tested
directly. Both providers reuse the same prompt, input limits and validation.
In the same way, the similarity functions in `embeddings.py` need only NumPy,
and the ranking, metrics and keyword baseline in `semantic_search.py` need
no model.

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

**Embedding model download.** The first run of `python -m src.embeddings`,
or of `python -m src.semantic_search` with semantic search, downloads
`all-MiniLM-L6-v2` (about 90 MB) from the Hugging Face Hub into
`~/.cache/huggingface`. Set `HF_HOME` to use another folder. After that, it
loads from the cache and makes no network requests. `--offline` enforces this.
No API key or Hugging Face account is needed, and the tests never download the
model. Both commands share the same cached model.

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

### Semantic search (Day 4)

```bash
python -m src.semantic_search search "How do I keep two Python projects from needing different library versions?"
python -m src.semantic_search search --method keyword "python virtual environment"   # keyword baseline, no model
python -m src.semantic_search search --top-k 5 "database speed"                      # 5 results instead of 3
python -m src.semantic_search evaluate             # both methods on the 23 labelled queries
python -m src.semantic_search evaluate --offline   # local cache only: fail instead of downloading
```

The first semantic run downloads the model if it is not cached yet (see
[Setup](#setup)). `--method keyword` never loads it. Real output from
2026-10-22 (WSL2 Ubuntu, CPU, about 12 seconds including the PyTorch import):

```
Loading sentence-transformers/all-MiniLM-L6-v2 ...
Query: How do I keep two Python projects from needing different library versions?
Semantic search (sentence-transformers/all-MiniLM-L6-v2, cosine similarity): top 3 of 20 documents

  Rank  ID         Score  Text
  1     py-02      0.696  A Python virtual environment keeps the packages installed for one project separate from other projects, so each project can use its own library versions.
  2     py-03      0.229  When an error occurs at runtime, Python raises an exception. A try/except block catches the exception so that the program can recover instead of stopping.
  3     cloud-03   0.188  Containers package an application together with its dependencies. Kubernetes schedules and scales containers across a cluster of machines.
```

The scores are cosine similarities. They only mean something compared with
each other, and they are not probabilities. The keyword baseline only lists
documents that share at least one informative word with the query, so it can
return fewer than 3 results, or none:

```
Query: Make sure a money transfer is never half-applied if the system fails midway
Keyword baseline (IDF-weighted word overlap): top 3 of 20 documents

  Rank  ID       Score  Text
  1     fin-03   2.996  Banks use machine learning models to flag unusual card transactions that may be fraudulent and to block them before money is lost.

  Only 1 document(s) share a keyword with the query.
```

`evaluate` prints two lines per query and method (outcome, recall,
precision, the rank of the first relevant document, relevant IDs, retrieved
IDs, then the query text), for example:

```
  q10  hit   recall@3 1/1  precision@3 1/3  first relevant: 3  relevant: db-03                   retrieved: fin-03, py-03, db-03
       Make sure a money transfer is never half-applied if the system fails midway
```

It ends with this summary. Real output from 2026-10-23 (the full run is
described in [the search evaluation](#search-evaluation-day-4)):

```
Summary (23 queries)
  Method    Top-3 hit rate     Recall@3  Precision@3  MRR@3
  semantic  100.0% (23/23)     1.000     0.391        0.971
  keyword   87.0% (20/23)      0.819     0.304        0.819

Top-3 hit rate: the share of queries with at least one relevant document in the top 3.
Recall@3: for each query, (relevant documents in the top 3) / (all of its relevant documents),
  averaged over the queries. The two are equal for a query with exactly one relevant document.
Precision@3: for each query, (relevant documents in the top 3) / 3, averaged over the queries.
  A query with fewer than 3 relevant documents cannot reach 1; an empty slot counts as a miss.
MRR@3: for each query, 1 / (rank of the first relevant document in the top 3), or 0 if there is none,
  averaged over the queries.
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | The model could not be loaded, or `data/documents.json` / `data/queries.json` is missing or invalid (the message names the file and the problem) |
| 2 | Invalid arguments (e.g. `--top-k 0`) or an empty query |

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

from src.semantic_search import KeywordSearcher, SemanticSearcher, evaluate, load_documents, load_queries

documents = load_documents()                    # data/documents.json, validated
searcher = SemanticSearcher(load_model(), documents)   # embeds the 20 documents once
searcher.search("Speed up lookups in a big table")     # [SearchResult(rank=1, id=..., text=..., score=...), ...]
report = evaluate(searcher, load_queries(documents))
report.hit_rate, report.mean_recall             # top-3 hit rate and Recall@3, each from 0 to 1
report.mean_precision, report.mrr               # Precision@3 (at most report.max_mean_precision) and MRR@3
evaluate(KeywordSearcher(documents), load_queries(documents)).hit_rate   # the baseline, no model
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

## Search evaluation (Day 4)

**Question.** On a small, labelled collection, does semantic search put a
relevant document in the top 3 more often than a simple keyword search?

### Dataset

`data/documents.json` holds 20 documents, written for this task. Each is
one or two sentences:

| Subject | IDs | Documents |
|---------|-----|-----------|
| AI and machine learning | `ml-01` to `ml-04` | 4: supervised learning, overfitting, neural networks, large language models |
| Python programming | `py-01` to `py-03` | 3: list comprehensions, virtual environments, exceptions |
| Databases | `db-01` to `db-03` | 3: relational tables and SQL, indexes, transactions |
| Cloud computing | `cloud-01` to `cloud-03` | 3: on-demand virtual machines, serverless functions, containers and Kubernetes |
| Cybersecurity | `sec-01` to `sec-04` | 4: phishing, SQL injection, multi-factor authentication, ransomware |
| Finance | `fin-01` to `fin-03` | 3: compound interest, diversification, card-fraud detection |

Two documents deliberately cross subjects: `sec-02` (SQL injection:
security and databases) and `fin-03` (fraud detection with machine learning).

`data/queries.json` holds 23 queries:
- **Labels.** They were written after the documents and labelled from the
  document texts before any search was run. No label was changed after
  seeing results.
- **One relevant document.** 20 queries have exactly one, and each document
  is the answer to exactly one of them.
- **Several.** 3 queries have more: `q21` "common types of cyber attack"
  (`sec-01`, `sec-02`, `sec-04`), `q22` cloud services that scale
  automatically (`cloud-02`, `cloud-03`), and `q23` saving and investing
  (`fin-01`, `fin-02`).
- **Word overlap varies, and is often high.** After the keyword baseline's
  tokenising (see below), 15 of the 23 queries share at least two words with
  a relevant document. `q02`, `q06`, `q15` and `q18` share four, and `q18`
  nearly restates `fin-01` ("grow faster the longer"). Only `q10` and `q19`
  share no word with any relevant document, for example `q19` "Don't put
  all your eggs in one basket when investing" for diversification. This
  overlap is why the keyword baseline finds a relevant document for 20 of
  the 23 queries (see [Results](#results)).

To add a document or query, add one line to the JSON file. Loading checks
every entry.

### Metrics

For each query q, R(q) is its set of relevant documents and T(q) the top 3
retrieved documents. Q is the set of 23 queries.

- **Top-3 hit rate** = (number of queries with at least one relevant
  document in T(q)) / |Q|. A query counts once, however many relevant
  documents it finds.
- **Recall@3** = the mean over all queries of |R(q) ∩ T(q)| / |R(q)|, the
  share of each query's relevant documents found in its top 3.
- **How they differ.** For a query with one relevant document, the two
  values are the same (0 or 1). They differ only on `q21` to `q23`: finding
  1 of 3 relevant documents is a full hit but a recall of 1/3. A query with
  more than 3 relevant documents could never reach a recall of 1 (none
  here).
- **Precision@3** = the mean over all queries of |R(q) ∩ T(q)| / 3, the
  share of each top 3 that is relevant. An empty slot (the keyword baseline
  can return fewer than 3 documents) counts as a miss. A query with one
  relevant document scores at most 1/3, so with these labels the best
  possible value is 27 / 69 = 0.391, not 1. It shows how much of each top 3
  is noise, which the hit rate and recall ignore.
- **MRR@3** (mean reciprocal rank) = the mean over all queries of
  1 / (rank of the first relevant document in T(q)), or 0 if there is none.
  A relevant document in first place scores 1, in third place 1/3, so unlike
  the hit rate it rewards ranking the answer higher.
- None of the metrics is called "accuracy".

**Keyword baseline.**
- Words are lower-cased, with common English stop words and one-letter words
  dropped, and a final plural "s" removed.
- A document's score is the sum of idf(w) = ln(N / df(w)) over the distinct
  words w it shares with the query. N = 20 is the number of documents and
  df(w) the number that contain w.
- Only documents with a score above 0 are returned, so a query with no shared
  word gets no results instead of three arbitrary ones.
- The stop-word list and rules were fixed before the evaluation was run, and
  were not tuned on the queries.

**Ties.** Equal scores are ordered by document ID in both methods. This is
deterministic, but arbitrary. In the keyword top 3 it decided two queries,
both described below: the hit for `q09` and the miss for `q16`.

### Results

Measured on 2026-10-22 with `python -m src.semantic_search evaluate --offline`.
The run took about 15 seconds on WSL2 Ubuntu (CPU), with the same library
versions as [Day 3](#similarity-experiment-day-3). Precision@3 and MRR@3
were added on 2026-10-23 and measured by re-running the same command; the
hit rates and recall did not change. No relevance label was changed.

| Method | Top-3 hit rate | Recall@3 | Precision@3 | MRR@3 |
|--------|----------------|----------|-------------|-------|
| Semantic search (all-MiniLM-L6-v2, cosine similarity) | **100.0%** (23 of 23) | **1.000** | **0.391** | **0.971** |
| Keyword baseline (IDF-weighted word overlap) | 87.0% (20 of 23) | 0.819 | 0.304 | 0.819 |

- **Precision@3 of 0.391 is the best these labels allow** (see
  [Metrics](#metrics)): every one of the 27 labelled documents is in its
  query's top 3. The other 42 of the 69 top-3 slots hold unlabelled
  documents, which count as misses. A manual check on 2026-10-23 (one
  reviewer, labels unchanged) judged 37 of those 42 unrelated to their
  query, 3 partly relevant and 2 uncertain. For example, `fin-03` (card
  fraud) is ranked first for `q10` (money transfers).
- **MRR@3 is below 1 only because of `q10`**, whose relevant document is
  third: (22 + 1/3) / 23 = 0.971.

The same run with `--top-k 1` (only the first result counts) separates the
methods more, because the top 3 leaves little room for error. At top 1,
Precision@1 and MRR@1 equal the top-1 hit rate:

| Method | Top-1 hit rate | Recall@1 |
|--------|----------------|----------|
| Semantic search | 95.7% (22 of 23) | 0.884 |
| Keyword baseline | 78.3% (18 of 23) | 0.710 |

**Per-query observations** (the full per-query output is printed by `evaluate`):
- **Keyword misses.**
  - `q10` (money transfer): the query shares no word with `db-03`
    (transactions). Its one shared word, "money", retrieves `fin-03`.
  - `q16` ("keep my account safe ... password"): a four-way tie at 2.303
    between `db-03`, `py-02`, `sec-01` and `sec-03`, each sharing one word
    ("keep" or "password"). ID order put `sec-03`, the relevant document,
    fourth. So the tie-breaking rule decided this miss.
  - `q19` ("eggs in one basket"): no shared word, no results.
- **Keyword hit decided by a tie.** `q09` ("make finding one row in a huge
  table much faster"): `db-01` is first (5.991, sharing "row" and
  "table"). Then `db-02`, `fin-01` and `ml-04` tie at 2.996 for the last
  two places, each sharing one word ("much", "faster" or "huge"). ID order
  kept `db-02`, the relevant document, in second place. So the
  tie-breaking rule decided this hit.
- **Keyword partial recall.** For `q21`, only `sec-02` contains the word
  "attack" (1 of 3). For `q23`, it found `fin-01` but not `fin-02` (1 of 2).
- **Semantic near-misses.**
  - `q10`: `db-03` was third (0.201), behind `fin-03` (0.276) and `py-03`
    (0.217). It is semantic search's only top-1 miss.
  - `q22`: `cloud-02` was third, behind `cloud-03` and `cloud-01`.
  - The top-3 scores are close together, so a slightly different query could
    turn these hits into misses.

### What this does and does not show

- **The pipeline works.** Semantic search ran end to end on every labelled
  query: documents embedded once, queries embedded, cosine similarity,
  ranking, metrics. It found a relevant document in the top 3 for every
  query, and in first place for 22 of 23.
- **100% is a ceiling effect, not a general accuracy figure.** The
  collection is small (20 documents), its six subjects are far apart, and
  the documents are short and clean. An easy set like this cannot tell a
  good model from a better one.
- **The sample is too small to call the gap significant.**
  - Each query is worth 4.3 percentage points.
  - 95% Wilson intervals: semantic 85.7-100%, keyword 67.9-95.5%. They
    overlap.
  - Semantic search won all 3 queries on which the methods disagreed at
    top 3 (4 at top 1), and the keyword baseline won none. An exact McNemar
    test on those pairs gives p = 0.25 (top 3) and p = 0.125 (top 1), which
    is not significant.
- **The queries may favour semantic search.** One author wrote the
  documents, the queries and the labels, knowing that semantic search would
  be compared with keyword search. Some queries are paraphrases with no
  shared word (`q10`, `q19`), which is where keyword overlap is weakest. But
  most share two or more words with their answer, so the set is not a hard
  test of either method.
- **The labels are one person's binary judgements.** There is no graded
  relevance and no second annotator. Some calls are debatable: for
  example, `cloud-01` (on-demand virtual machines) is not labelled relevant
  to `q22` (automatic scaling).
- **The keyword baseline is deliberately simple.** It has no stemming beyond
  a plural "s" ("pretending" does not match "pretend"), no synonyms and no
  length normalisation. A stronger lexical baseline such as BM25 may score
  differently.
- **Known limits.**
  - The model reads at most 256 word pieces per text and truncates longer
    input.
  - It is trained mainly on English.
  - Search compares the query with every document, which is right for 20
    documents. Thousands would need a single matrix product or a vector
    index.
- **A rigorous evaluation would need:**
  - an independent, public retrieval benchmark (for example, a BEIR
    dataset) with labels written by other people;
  - hundreds of queries, including hard negatives and near-duplicate
    documents;
  - several annotators and graded relevance, scored with nDCG as well;
  - BM25 and other embedding models as baselines;
  - significance tests on the paired results.

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
python -m pytest -q                  # all 469 tests, about 10-20 seconds (7 load the cached embedding model)
python -m pytest -q -m "not model"   # 462 tests, about 3 seconds, never imports PyTorch in the test process
```

Results recorded on 2026-10-25, after Day 5 added 6 `check-env` tests to
`tests/test_main.py` (WSL2 Ubuntu, Python 3.12.15, pytest 9.1.1). The
virtual environment was on the Linux file system and the repository on
`/mnt/d`:

| Command | Result |
|---------|--------|
| `python -m pytest -q` | 469 passed in 16.14 s |
| `python -m pytest -q -m "not model"` | 462 passed, 7 deselected in 3.28 s |
| `python -m pytest -q --ignore=tests/test_embeddings.py --ignore=tests/test_semantic_search.py` (the Day 1-2 suite) | 168 passed in 1.83 s: the 162 earlier tests and the 6 new `check-env` tests |
| `python -m pytest -q tests/test_embeddings.py` (Day 3) | 108 passed in 6.85 s, unchanged |
| `python -m pytest -q tests/test_semantic_search.py` (Day 4) | 193 passed in 7.82 s, unchanged |
| `HF_HOME=<empty folder> python -m pytest -q -m model` (model not cached) | 7 skipped, 462 deselected; nothing was downloaded |

Earlier results, recorded on 2026-10-23 after Precision@3, MRR@3 and the
real-model regression test were added (WSL2 Ubuntu, Python 3.12.15,
pytest 9.1.1). That run used a virtual environment on the Windows drive
(`/mnt/d`), where importing PyTorch is much slower, so the times are longer
than on 2026-10-22 (16.82 s for the full suite then):

| Command | Result |
|---------|--------|
| `python -m pytest -q` | 463 passed in 259.00 s |
| `python -m pytest -q -m "not model"` | 456 passed, 7 deselected in 20.82 s |
| `python -m pytest -q --ignore=tests/test_embeddings.py --ignore=tests/test_semantic_search.py` (the Day 1-2 suite) | 162 passed in 12.97 s, unchanged |
| `python -m pytest -q tests/test_embeddings.py` (Day 3) | 108 passed in 259.49 s, unchanged |
| `python -m pytest -q tests/test_semantic_search.py` (Day 4) | 193 passed in 179.94 s |
| `HF_HOME=<empty folder> python -m pytest -q -m model` (model not cached) | 7 skipped, 456 deselected; nothing was downloaded |

Before Day 4, the suite was 270 tests. Day 4 brought it to 463, and Day 5 to
469 (the 6 `check-env` tests in `tests/test_main.py`). All of them still pass.

Passing tests show that the code behaves as specified. They do not show that
the model is accurate. The measured retrieval quality is under
[the search evaluation](#search-evaluation-day-4).

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
- The semantic-search tests use fake encoders that return fixed vectors, so
  every expected ranking and score can be worked out by hand. The metric
  tests use hand-calculated examples.
- The 7 tests marked `model` (3 for Day 3, 4 for Day 4) load the real
  `all-MiniLM-L6-v2` with `local_files_only=True` while sockets are blocked.
  If the model is not cached, they are skipped. They never download it.

| Test file | Covers |
|-----------|--------|
| `test_groq_client.py` | config from env/`.env`; request shape (Groq endpoint, Bearer auth, strict schema vs JSON mode, `reasoning_effort` for GPT-OSS models only); empty/missing choices, `null` content, `length`/`content_filter`, refusal; HTTP 401/403/429/400/404/500/503 and network errors; end-to-end classification |
| `test_api_client.py` | Anthropic config, request shape (effort, fallback beta header, no unsupported parameters), text extraction, HTTP 401/403/429/400/404/500/529 and network errors |
| `test_classifier.py` | Anthropic classification workflow end to end |
| `test_response_parser.py` | JSON parsing (valid, fenced, empty, malformed, NaN) and schema validation (missing/extra fields, wrong types and ranges) |
| `test_main.py` | CLI output and exit codes; `check-env` package checks (numpy, sentence-transformers and torch are required and each missing one is named and fails the check; pytest is optional); and that keys never appear in output |
| `test_embeddings.py` | cosine similarity: known values, symmetry, scale invariance, clipping to [-1, 1], extreme magnitudes (1e-200 to 1e200), invalid vectors; the similarity matrix, including extreme magnitudes; pair ranking, ties and `top_k`; `top_k` validation (0, -1, 2.5, booleans) in `rank_pairs()` and `format_report()`; embedding shape for 1 to 768 dimensions; invalid texts and unusable model output; loading from the cache first and downloading only when allowed; CLI output and exit codes; that importing the module does not load PyTorch. With the cached model: 384 dimensions, and the expected ranking of the five example texts |
| `test_semantic_search.py` | the dataset (20 documents, unique IDs, six subjects; every label points at a real document); malformed documents and JSON entries, duplicate document and query IDs, repeated JSON keys; ranking order, top 3 and `top_k` larger than the collection; invalid queries and `top_k` (0, -1, 2.5, booleans, strings, `None`) rejected before the model is called; ties broken by ID whatever the document order; document embeddings computed once, reused and read-only; Day 3 numerical behaviour (1e-200 to 1e200, clipping to [-1, 1], zero vectors); keyword tokenising, IDF scores and ties; top-3 hit rate, Recall@3, Precision@3 (with empty slots counted as misses) and MRR@3 against hand-calculated examples; empty and invalid labels; CLI output, `--method keyword`, `--offline`, `--top-k` and exit codes; that importing the module does not load PyTorch. With the cached model: 384 dimensions, 3 results best first, every labelled query evaluated, and the measured results (no miss, `q10` the only query whose relevant document is not first, Recall@3 1.000, Precision@3 27/69, MRR@3 0.971) |

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
- **Semantic search.**
  - `semantic_search.py` imports the Day 3 functions (`embed_texts`,
    `cosine_similarity`, `load_model`) and its input checks, so the two
    modules accept and reject the same values.
  - A `SemanticSearcher` embeds its documents once, in one batch, and keeps
    the vectors read-only. It holds its own copy of the collection, so the
    cached vectors always match it. Searching other documents means creating
    a new searcher.
  - Ranking, metrics and the keyword baseline take plain lists and need no
    model.
  - The keyword baseline sums its idf weights with `math.fsum`. That sum is
    exactly rounded, so ties are exact and do not depend on Python's
    hash-seeded set order.
  - The dataset is JSON rather than Python code. It is data, and it can be
    edited and checked without touching the code.

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
| `7 skipped` for tests marked `model` | The embedding model is not in the local cache. Run `python -m src.embeddings` once. |
| `Error: could not load .../data/documents.json: ...` | The dataset file is missing or invalid. The message names the entry and the problem, for example a duplicate ID or a missing field. |
