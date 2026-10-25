# Day 5: accuracy and cost note

AI/ML Engineering, Week 1, Friday. Written on 2026-10-25 against commit
`786b978`.

> **AI-assisted.** This note was drafted with an AI coding assistant (Claude
> Code) from the code, the README and commands re-run on 2026-10-25. No human
> mentor has reviewed or validated it yet.

**Sources.** `README.md`, `src/semantic_search.py`, `src/embeddings.py`,
`src/api_client.py`, `src/groq_client.py`, `src/classifier.py`,
`data/documents.json` and `data/queries.json`, plus these commands, run on
2026-10-25 under WSL2 Ubuntu in the existing virtual environment (Python
3.12.15, torch 2.14.1+cpu, sentence-transformers 6.1.0, transformers 5.20.0,
pytest 9.1.1):

```bash
python -m pytest -q --durations=10
python -m src.semantic_search evaluate --offline
python -m src.semantic_search evaluate --offline --top-k 1
```

A one-off timing script and a tie analysis of the keyword baseline were also
run. Both only call the project's own functions and are not part of the
repository.

**Status labels.** *Re-verified*: reproduced on 2026-10-25. *Reported*: taken
from the README, not re-run. *Unverified*: no measurement exists.

## 1. Accuracy

### Semantic search vs keyword baseline (top 3)

23 labelled queries, 20 documents, 27 relevant-document labels.
*Re-verified*: identical to the README's results from 2026-10-22/23.

| Method | Top-3 hit rate | Recall@3 | Precision@3 | MRR@3 |
|--------|----------------|----------|-------------|-------|
| Semantic search (all-MiniLM-L6-v2, cosine similarity) | 100.0% (23/23) | 1.000 | 0.391 | 0.971 |
| Keyword baseline (IDF-weighted word overlap) | 87.0% (20/23) | 0.819 | 0.304 | 0.819 |

- 0.391 (27/69) is the highest Precision@3 these labels allow, so semantic
  search is at the ceiling on every metric except MRR@3.
- MRR@3 is below 1 only because of `q10`, whose relevant document is third.
- The keyword baseline misses `q10`, `q16` and `q19`.
- The real-model regression test passed, so the semantic numbers are also
  pinned by a test. The keyword numbers are not.

### Top 1

*Re-verified* with `--top-k 1`:

| Method | Top-1 hit rate | Recall@1 | Precision@1 | MRR@1 |
|--------|----------------|----------|-------------|-------|
| Semantic search | 95.7% (22/23) | 0.884 | 0.957 | 0.957 |
| Keyword baseline | 78.3% (18/23) | 0.710 | 0.783 | 0.783 |

### Sensitivity to tie-breaking (keyword baseline)

Both methods order exactly equal scores by document ID. Keyword scores are
sums of a few IDF values, so exact ties are common. Only the keyword baseline
was checked. If tied documents were ordered at random instead, its expected
scores would be (computed exactly from the tie groups on 2026-10-25):

| Keyword baseline | Ties ordered by ID (reported) | Expected with random tie order |
|------------------|-------------------------------|--------------------------------|
| Top-3 hit rate | 87.0% | 88.8% (20.42/23) |
| Recall@3 | 0.819 | 0.837 |
| Top-1 hit rate | 78.3% | 77.2% (17.75/23) |
| Recall@1 | 0.710 | 0.710 |

- At top 3, ties decide two queries: `q16` (a miss by ID order, a 3/4 chance
  of a hit at random) and `q09` (a hit by ID order, a 2/3 chance at random).
  The README mentions only `q16`.
- The gap between the two methods does not depend on the tie order.

### Classification (Days 1-2)

*Unverified.* There is no labelled classification set, so classification
accuracy has not been measured for either provider. The one live Groq request
(2026-10-21) returned a schema-valid `billing` label for one hand-written
input. That shows the pipeline works, not how accurate it is.

## 2. Evaluation limitations

- **Small sample.** The set has 23 queries and 20 documents, so each query is
  worth 4.3 percentage points of hit rate.
  - 95% Wilson intervals for the top-3 hit rate: semantic 85.7-100%, keyword
    67.9-95.5%. They overlap.
  - Exact McNemar test on the queries where the methods disagree: p = 0.25 at
    top 3 (3 queries) and p = 0.125 at top 1 (4 queries). Neither is
    significant.
  - These figures are *reported* in the README and were recomputed by hand
    during this review, with the same results.
- **Saturated, easy set.**
  - The six subjects are far apart and the documents are short and clean.
  - 20 of the 23 queries have exactly one relevant document, and each document
    is the answer to exactly one of them.
  - Semantic search reaches the ceiling, so the set cannot tell a good model
    from a better one. 100% is not a general accuracy figure.
- **Label quality drives every metric.**
  - One person wrote the documents, the queries and the binary labels, knowing
    the two methods would be compared. There is no second annotator and no
    graded relevance.
  - Debatable labels change the numbers directly. `cloud-01` is not labelled
    relevant to `q22`. In the README's one-reviewer check of the 42
    unlabelled top-3 slots, 3 were judged partly relevant and 2 uncertain.
  - Each extra relevant label inside a top 3 would raise Precision@3 by 1/69
    (0.014). A wrong label could turn a hit into a miss, or the other way
    round, which moves the hit rate by 4.3 points.
- **Word overlap.** 15 of the 23 queries share at least two words with a
  relevant document, which helps the keyword baseline. 2 share none (`q10`,
  `q19`), which hurts it (*reported*).
- **No independent test set.** The regression test was written after the
  results were known. It catches drift, but it is not evidence of accuracy.
- **Simple baseline.**
  - There is no stemming beyond a final "s", no synonyms and no length
    normalisation, and BM25 has not been tried.
  - The plural rule turns "queries" into "querie" and "indexes" into
    "indexe", which then do not match "query" and "index".
- **Not validated by a person.** No mentor or second annotator has checked
  the labels or these results.

## 3. Embedding cost (local all-MiniLM-L6-v2)

There is **no per-request fee**. The model runs locally on the CPU and needs
no API key. The only network access is the one-time download of the model
(about 90 MB, *reported*) into the Hugging Face cache. After that, `--offline`
runs make no network requests. The real costs are compute, memory, disk and
start-up time.

| Item | Value | Status |
|------|-------|--------|
| Parameters | 22,713,216 | Re-verified (counted from the loaded model) |
| Weights in memory (float32) | about 91 MB | Computed: parameters × 4 bytes |
| Embedding size | 384 numbers per text | Re-verified |
| Longest input | 256 word pieces; longer text is truncated silently | Re-verified (`max_seq_length`) |
| Stored vectors for the collection | 20 × 384 float64 = 61,440 bytes | Computed (`embed_texts()` returns float64) |
| Peak memory of one evaluation process | about 526 MB resident (PyTorch, model and data) | Measured once on 2026-10-25 |
| Model download and disk space | about 90 MB | Reported |
| Hardware | CPU only (torch 2.14.1+cpu), 6 PyTorch threads on 12 logical CPUs | Re-verified |

These things would raise the cost:
- **More documents.** Indexing takes one forward pass per document, and each
  query takes one forward pass plus a comparison with every document.
- **Longer documents.** Input is truncated at 256 word pieces, so compute
  stops growing at that length, but accuracy may fall.
- **Deployment.** On a server, every running process holds its own copy of
  the model in RAM.

## 4. LLM and API cost

The embeddings and semantic search (Days 3-4) use **no** LLM API. API costs
apply only to the classification commands from Days 1-2.

### Groq (`groq-classify`, live-tested)

- **One live request** was sent on 2026-10-21, with `openai/gpt-oss-20b`. Its
  token usage was **not recorded**, because `groq-classify` does not print
  `usage`. Its cost is therefore *unverified*.
- **Prices and limits.** Groq's per-token prices and rate limits are not in
  the repository and were not checked for this note (*unverified*). Take them
  from Groq's pricing page and record the date.
- **Limits set in the code** (`src/groq_client.py`, `src/classifier.py`):
  - input of at most 4,000 characters (longer input is rejected, never
    truncated);
  - `max_completion_tokens=1024`, which must also hold the GPT-OSS reasoning;
  - `reasoning_effort="low"`;
  - a 60-second timeout, with up to 2 SDK retries.

### Anthropic (`ping`, `classify`, not live-tested)

- **No live request has been made.** All Anthropic behaviour is covered by
  offline tests against a fake HTTP server. No cost or latency has been
  measured.
- **Settings in the code:**
  - model `claude-opus-5-5`;
  - `max_tokens=1024`. Thinking is always on for this model and counts
    toward that limit;
  - `effort="low"`, with structured JSON output;
  - a 60-second timeout and up to 2 SDK retries.
- **Price.** The README gives US$4 / US$20 per million input / output tokens
  for Claude Opus 5.5. That matches the Claude API reference available to the
  assistant (cached 2026-10-06), but it was not checked against Anthropic's
  live pricing page. Thinking tokens are billed as output tokens.
- **Upper bound per call.** Output is capped at 1,024 tokens, which costs at
  most 1,024 × $20 / 1M = US$0.020. The README's "at most about US$0.03" also
  needs the input to stay under about 2,400 tokens. With the 4,000-character
  cap that is likely, but it is *unverified*. Anthropic's token-counting
  endpoint would confirm it.
- **What can push a call above that bound:**
  - **Refusal fallback.** It is on by default (`CLAUDE_USE_FALLBACKS=true`).
    According to Anthropic's documentation, if a request is refused part-way
    through, the declined attempt is billed, and so is the fallback run, at
    the fallback model's rates.
  - **Retries.** Retries after timeouts or 5xx errors can repeat work that
    was already billed.
  - **Rejected replies.** A reply cut off at `max_tokens`, or one that fails
    local validation, is paid for and then rejected. There is no automatic
    retry.
- **Usage reporting.** `ping` prints input and output tokens; `classify` does
  not.

## 5. Latency and performance

Measured once on 2026-10-25 under WSL2 Ubuntu (12 logical CPUs, 7.6 GiB RAM),
with the virtual environment on the Linux file system and the repository on
`/mnt/d`. The file cache was warm, because a test run had just finished. Treat
these numbers as one sample, not a benchmark.

| Step | Time |
|------|------|
| `import torch` | 1.40 s |
| Load all-MiniLM-L6-v2 from the cache | 4.73 s |
| Embed the 20 documents (one batch) | 0.154 s |
| Evaluate 23 queries (one encode call each) | 0.291 s, about 12.7 ms per query |
| Keyword baseline, 23 queries | 0.002 s |
| `python -m src.semantic_search evaluate --offline`, end to end | 7.0 s wall clock |

- **Start-up dominates.** Importing PyTorch and loading the model take about
  6 s, while searching takes about 13 ms per query. In a long-running process
  that loads the model once, queries are cheap.
- **Test timings.**
  - Today: `463 passed in 15.48s` (24.0 s wall clock, including interpreter
    start-up).
  - The slowest step is the set-up of the Day 3 real-model fixture, at
    10.81 s, which includes importing PyTorch and loading the model. Every
    other test takes under 1 s.
  - *Reported* in the README: 16.82 s on 2026-10-22, and 259.00 s on
    2026-10-23 with the virtual environment on the Windows drive (`/mnt/d`),
    where importing PyTorch is much slower. That is about 17 times today's
    time. The README attributes the difference to where the virtual
    environment is stored, not to the code.
- **Scaling.** `SemanticSearcher.scores()` compares the query with each
  document in a Python loop and re-normalises every document vector for every
  query, and `evaluate()` encodes the queries one at a time. That is fine for
  20 documents. Thousands would need a pre-normalised matrix, a single matrix
  product and batched query encoding.
- **API latency.** Groq and Anthropic latency is *unverified*: no timing was
  recorded for the one Groq call, and no Anthropic call has been made.

## 6. Recommendations

**Cost control**
1. Record `usage` (input, output and reasoning tokens), the model and the
   latency for every LLM call. Print them in `classify` and `groq-classify`,
   as `ping` already does. The first real cost figures will come from this.
2. Before the first Anthropic call:
   - count the tokens in a 4,000-character input;
   - decide whether the refusal fallback should be on by default;
   - set a spending limit in the Anthropic Console.
3. Measure how often replies are cut off at 1,024 tokens or fail validation,
   and size `max_tokens` from that data.
4. Once quality is measured on a labelled set, compare cheaper model tiers or
   providers for classification. For bulk jobs that are not urgent, consider
   Anthropic's Message Batches API, which is documented at a lower price than
   standard requests.
5. Keep the embeddings local. Load the model once per process, batch the query
   encoding in `evaluate()`, and keep the virtual environment on the Linux
   file system.
6. Record Groq's prices and rate limits, with the date they were checked.

**Evaluation reliability**
1. Ask a second person, such as the mentor, to label the 23 queries
   independently. Measure agreement and resolve disagreements before quoting
   any numbers.
2. Add queries written by someone else that have not been used for tuning:
   hard negatives, near-duplicate documents, and queries with more than one
   relevant document. Aim for hundreds of queries, or use part of a public
   benchmark such as a BEIR dataset.
3. Report tie-aware keyword metrics, or list the ties in the output. Add BM25
   as a second baseline.
4. Add tests that pin the keyword baseline's results and the top-1 results.
   Both are deterministic, and the keyword test needs no model.
5. Have `evaluate` print confidence intervals, so they are no longer computed
   by hand.
6. Build a small labelled set for classification, and report accuracy and cost
   for each provider.
7. Before longer documents are added, warn when a text is longer than the
   model's 256 word pieces.
