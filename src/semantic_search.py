"""Thursday task: semantic search over 20 short documents, with a keyword baseline and a top-3 evaluation.

The module has four parts, kept apart so that most of it is tested without a model:

- **Data** (``Document``, ``LabelledQuery``, ``load_documents``, ``load_queries``):
  the documents and the labelled queries live in ``data/documents.json`` and
  ``data/queries.json`` and are validated when they are loaded.
- **Pure ranking and evaluation** (``rank_by_score``, ``hit_at_k``,
  ``recall_at_k``, ``precision_at_k``, ``reciprocal_rank``,
  ``evaluate_rankings``): plain Python and NumPy, no model.
- **Searchers**: ``SemanticSearcher`` embeds the documents once with the Day 3
  ``embed_texts()`` and scores each query with ``cosine_similarity()``.
  ``KeywordSearcher`` is the baseline: IDF-weighted word overlap, no model.
- **Command line**: ``search`` and ``evaluate``.

Run from the project root:

    python -m src.semantic_search search "How do I keep two Python projects' libraries apart?"
    python -m src.semantic_search search --method keyword "python virtual environment"
    python -m src.semantic_search evaluate             # both methods on the labelled queries
    python -m src.semantic_search evaluate --offline   # use the local cache only; fail instead of downloading

As in ``src.embeddings``, ``sentence_transformers`` (and so PyTorch) is
imported only when the model is loaded.

Exit codes: 0 success, 1 the model or the dataset could not be loaded,
2 invalid arguments or an empty query.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# Day 3 code, reused rather than reimplemented. The private validation helpers
# are reused too, so both modules accept and reject exactly the same values.
from src.embeddings import (
    MODEL_NAME,
    Encoder,
    _as_float_array,
    _check_top_k,
    _positive_int,
    cosine_similarity,
    embed_texts,
    load_model,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
QUERIES_PATH = DATA_DIR / "queries.json"
DEFAULT_TOP_K = 3
METHODS = ("semantic", "keyword")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


# --- Data -----------------------------------------------------------------------


def _check_text(value: Any, name: str) -> None:
    """Raise ValueError unless ``value`` is a string with at least one non-space character."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string; got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} is empty.")


def _check_id(value: Any, name: str) -> None:
    """Like ``_check_text()``, but also reject surrounding spaces, which would make "a" and "a " two IDs."""
    _check_text(value, name)
    if value != value.strip():
        raise ValueError(f"{name} {value!r} has leading or trailing whitespace.")


def _duplicates(values: Iterable[str]) -> list[str]:
    """The values that occur more than once, sorted."""
    return sorted(value for value, count in Counter(values).items() if count > 1)


@dataclass(frozen=True)
class Document:
    """One searchable document: a stable, unique ``id`` and its ``text``."""

    id: str
    text: str

    def __post_init__(self) -> None:
        _check_id(self.id, "Document ID")
        _check_text(self.text, f"Text of document {self.id!r}")


@dataclass(frozen=True)
class LabelledQuery:
    """A test query and the IDs of the documents a person judged relevant to it."""

    id: str
    text: str
    relevant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_id(self.id, "Query ID")
        _check_text(self.text, f"Text of query {self.id!r}")
        if not isinstance(self.relevant_ids, (list, tuple)):
            raise ValueError(
                f"relevant_ids of query {self.id!r} must be a list of document IDs; "
                f"got {type(self.relevant_ids).__name__}."
            )
        if not self.relevant_ids:
            raise ValueError(f"Query {self.id!r} has no relevant document IDs; Recall@k is undefined without them.")
        for document_id in self.relevant_ids:
            _check_id(document_id, f"A relevant ID of query {self.id!r}")
        repeated = _duplicates(self.relevant_ids)
        if repeated:
            raise ValueError(f"Query {self.id!r} lists relevant ID(s) more than once: {', '.join(repeated)}.")
        object.__setattr__(self, "relevant_ids", tuple(self.relevant_ids))  # a JSON list becomes a tuple


def _as_tuple(values: Any, kind: str) -> tuple[Any, ...]:
    """Return ``values`` as a non-empty tuple; a single string, mapping or object is rejected, not iterated."""
    if isinstance(values, (str, Mapping)):
        raise ValueError(f"The {kind} must be a list; got {type(values).__name__}.")
    try:
        items = tuple(values)
    except TypeError:
        raise ValueError(f"The {kind} must be a list; got {type(values).__name__}.") from None
    if not items:
        raise ValueError(f"The list of {kind} is empty.")
    return items


def check_documents(documents: Any) -> tuple[Document, ...]:
    """Return ``documents`` as a tuple of Documents with unique IDs.

    Raises:
        ValueError: ``documents`` is empty or not a list, an item is not a
            ``Document``, or two documents share an ID.
    """
    items = _as_tuple(documents, "documents")
    for index, item in enumerate(items):
        if not isinstance(item, Document):
            raise ValueError(f"Item {index} of the documents must be a Document; got {type(item).__name__}.")
    repeated = _duplicates(document.id for document in items)
    if repeated:
        raise ValueError(f"Duplicate document ID(s): {', '.join(repeated)}.")
    return items


def check_queries(queries: Any) -> tuple[LabelledQuery, ...]:
    """Return ``queries`` as a tuple of LabelledQuery objects with unique IDs.

    Raises:
        ValueError: ``queries`` is empty or not a list, an item is not a
            ``LabelledQuery``, or two queries share an ID.
    """
    items = _as_tuple(queries, "labelled queries")
    for index, item in enumerate(items):
        if not isinstance(item, LabelledQuery):
            raise ValueError(f"Item {index} of the labelled queries must be a LabelledQuery; got {type(item).__name__}.")
    repeated = _duplicates(query.id for query in items)
    if repeated:
        raise ValueError(f"Duplicate query ID(s): {', '.join(repeated)}.")
    return items


def check_labels(queries: Sequence[LabelledQuery], documents: Sequence[Document]) -> None:
    """Raise ValueError if a query lists a relevant document ID that is not in ``documents``."""
    known = {document.id for document in documents}
    for query in queries:
        unknown = [document_id for document_id in query.relevant_ids if document_id not in known]
        if unknown:
            raise ValueError(f"Query {query.id!r} lists unknown document ID(s): {', '.join(unknown)}.")


def _records(data: Any, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Check that parsed JSON is a non-empty list of objects that have exactly ``fields``."""
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list; got {type(data).__name__}.")
    if not data:
        raise ValueError("The JSON list is empty.")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Entry {index} must be a JSON object; got {type(record).__name__}.")
        missing = [field for field in fields if field not in record]
        unexpected = sorted(set(record) - set(fields))
        if missing or unexpected:
            problems = [f"missing {', '.join(missing)}"] if missing else []
            problems += [f"unexpected {', '.join(unexpected)}"] if unexpected else []
            raise ValueError(f"Entry {index} must have exactly the fields {', '.join(fields)} ({'; '.join(problems)}).")
    return data


def parse_documents(data: Any) -> tuple[Document, ...]:
    """Turn the parsed JSON of a documents file into validated Documents.

    Raises:
        ValueError: the data is not a non-empty list of ``{"id", "text"}``
            objects, a value is invalid (the message names the entry), or an
            ID is repeated.
    """
    documents = []
    for index, record in enumerate(_records(data, ("id", "text"))):
        try:
            documents.append(Document(record["id"], record["text"]))
        except ValueError as exc:
            raise ValueError(f"Entry {index}: {exc}") from None
    return check_documents(documents)


def parse_queries(data: Any, documents: Sequence[Document]) -> tuple[LabelledQuery, ...]:
    """Turn the parsed JSON of a queries file into validated LabelledQuery objects.

    Raises:
        ValueError: the data is not a non-empty list of
            ``{"id", "text", "relevant_ids"}`` objects, a value is invalid
            (the message names the entry), a query ID is repeated, or a
            relevant ID is not one of ``documents``.
    """
    queries = []
    for index, record in enumerate(_records(data, ("id", "text", "relevant_ids"))):
        try:
            queries.append(LabelledQuery(record["id"], record["text"], record["relevant_ids"]))
        except ValueError as exc:
            raise ValueError(f"Entry {index}: {exc}") from None
    items = check_queries(queries)
    check_labels(items, documents)
    return items


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads`` hook: reject an object that repeats a key, which would otherwise keep only the last value."""
    repeated = _duplicates(key for key, _ in pairs)
    if repeated:
        raise ValueError(f"A JSON object repeats the key(s): {', '.join(repeated)}.")
    return dict(pairs)


def _read_json(path: Path | str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}).") from None


def load_documents(path: Path | str = DOCUMENTS_PATH) -> tuple[Document, ...]:
    """Read and validate a documents file: a JSON list of ``{"id": ..., "text": ...}`` objects.

    Raises:
        OSError: the file cannot be read.
        ValueError: anything listed under ``parse_documents()``, or the file is not valid JSON.
    """
    return parse_documents(_read_json(path))


def load_queries(documents: Sequence[Document], path: Path | str = QUERIES_PATH) -> tuple[LabelledQuery, ...]:
    """Read and validate a queries file: a JSON list of ``{"id", "text", "relevant_ids"}`` objects.

    Raises:
        OSError: the file cannot be read.
        ValueError: anything listed under ``parse_queries()``, or the file is not valid JSON.
    """
    return parse_queries(_read_json(path), documents)


# --- Pure ranking and evaluation (no model) -------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One retrieved document: its 1-based ``rank``, ``id``, ``text`` and ``score``."""

    rank: int
    id: str
    text: str
    score: float


def rank_by_score(ids: Sequence[str], scores: Any, top_k: int = DEFAULT_TOP_K) -> list[int]:
    """Return the positions of the ``top_k`` highest scores, highest first.

    Equal scores are ordered by ID, ascending, so the result does not depend on
    the order of the documents in the collection. Only exactly equal scores are
    ties; scores are not rounded first.

    Raises:
        ValueError: ``top_k`` is not a whole number of at least 1, an ID is
            repeated, or ``scores`` is not a non-empty 1-D list of finite
            numbers with one score per ID.
    """
    _check_top_k(top_k)
    values = _as_float_array(scores, "scores", ndim=1)
    if len(ids) != values.size:
        raise ValueError(f"Got {len(ids)} IDs but {values.size} scores.")
    repeated = _duplicates(ids)
    if repeated:
        raise ValueError(f"Duplicate ID(s): {', '.join(repeated)}.")
    order = sorted(range(values.size), key=lambda index: (-values[index], ids[index]))
    return order[:top_k]


def _top_results(documents: Sequence[Document], scores: Sequence[float], top_k: int) -> list[SearchResult]:
    order = rank_by_score([document.id for document in documents], scores, top_k)
    return [
        SearchResult(rank, documents[index].id, documents[index].text, float(scores[index]))
        for rank, index in enumerate(order, start=1)
    ]


def _relevant_and_top(relevant_ids: Any, retrieved_ids: Any, top_k: Any) -> tuple[set[str], set[str]]:
    """Validate one ranking; return the relevant IDs and the first ``top_k`` retrieved IDs as sets."""
    _check_top_k(top_k)
    for name, ids in (("relevant_ids", relevant_ids), ("retrieved_ids", retrieved_ids)):
        if isinstance(ids, str) or not isinstance(ids, Sequence):
            raise ValueError(f"{name} must be a list of document IDs; got {type(ids).__name__}.")
        if not all(isinstance(document_id, str) for document_id in ids):
            raise ValueError(f"{name} must contain only string IDs.")
        repeated = _duplicates(ids)
        if repeated:
            raise ValueError(f"{name} lists ID(s) more than once: {', '.join(repeated)}.")
    if not relevant_ids:
        raise ValueError("relevant_ids is empty; the hit rate and Recall@k need at least one relevant document.")
    return set(relevant_ids), set(retrieved_ids[:top_k])


def hit_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = DEFAULT_TOP_K) -> bool:
    """True if at least one relevant document is among the first ``top_k`` retrieved documents.

    Raises:
        ValueError: ``relevant_ids`` is empty; either list is not a list of
            unique string IDs; or ``top_k`` is not a whole number of at least 1.
    """
    relevant, top = _relevant_and_top(relevant_ids, retrieved_ids, top_k)
    return not relevant.isdisjoint(top)


def recall_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = DEFAULT_TOP_K) -> float:
    """Recall@k = (relevant documents among the first ``top_k`` retrieved) / (all relevant documents).

    The result is from 0 to 1. A query with more than ``top_k`` relevant
    documents cannot reach 1.

    Raises:
        ValueError: as for ``hit_at_k()``.
    """
    relevant, top = _relevant_and_top(relevant_ids, retrieved_ids, top_k)
    return len(relevant & top) / len(relevant)


def precision_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = DEFAULT_TOP_K) -> float:
    """Precision@k = (relevant documents among the first ``top_k`` retrieved) / ``top_k``.

    The result is from 0 to 1. The divisor is always ``top_k``, so a slot left
    empty (the keyword baseline can return fewer than ``top_k`` documents)
    counts as a miss. A query with fewer relevant documents than ``top_k``
    cannot reach 1: with one relevant document, Precision@3 is at most 1/3.

    Raises:
        ValueError: as for ``hit_at_k()``.
    """
    relevant, top = _relevant_and_top(relevant_ids, retrieved_ids, top_k)
    return len(relevant & top) / top_k


def reciprocal_rank(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = DEFAULT_TOP_K) -> float:
    """1 / (rank of the first relevant document among the first ``top_k`` retrieved), or 0 if there is none.

    Averaged over the queries, this is MRR@k: 1 if every query has a relevant
    document in first place, 1/3 if it is always third.

    Raises:
        ValueError: as for ``hit_at_k()``.
    """
    relevant, _ = _relevant_and_top(relevant_ids, retrieved_ids, top_k)
    for rank, document_id in enumerate(retrieved_ids[:top_k], start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


@dataclass(frozen=True)
class QueryEvaluation:
    """One labelled query: its first ``top_k`` retrieved IDs, whether any is relevant, and its per-query metrics."""

    query: LabelledQuery
    retrieved_ids: tuple[str, ...]
    hit: bool
    recall: float
    precision: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    """The result for every labelled query, and the averages over all of them."""

    top_k: int
    results: tuple[QueryEvaluation, ...]

    @property
    def hits(self) -> int:
        """Number of queries with at least one relevant document in the top ``top_k``."""
        return sum(result.hit for result in self.results)

    @property
    def hit_rate(self) -> float:
        """Top-k hit rate: hits / number of queries, from 0 to 1."""
        return self.hits / len(self.results)

    @property
    def mean_recall(self) -> float:
        """Recall@k: the per-query Recall@k averaged over the queries, from 0 to 1."""
        return math.fsum(result.recall for result in self.results) / len(self.results)

    @property
    def mean_precision(self) -> float:
        """Precision@k: the per-query Precision@k averaged over the queries, from 0 to 1."""
        return math.fsum(result.precision for result in self.results) / len(self.results)

    @property
    def max_mean_precision(self) -> float:
        """The highest Precision@k these labels allow: a perfect ranking scores min(relevant, k) / k per query."""
        best = sum(min(len(result.query.relevant_ids), self.top_k) for result in self.results)
        return best / (self.top_k * len(self.results))

    @property
    def mrr(self) -> float:
        """MRR@k: the per-query reciprocal rank averaged over the queries, from 0 to 1."""
        return math.fsum(result.reciprocal_rank for result in self.results) / len(self.results)


def evaluate_rankings(
    queries: Sequence[LabelledQuery], rankings: Sequence[Sequence[str]], top_k: int = DEFAULT_TOP_K
) -> EvaluationReport:
    """Score one ranked list of document IDs per labelled query, in the same order as ``queries``.

    Raises:
        ValueError: ``top_k`` is invalid, the queries fail ``check_queries()``,
            there is not one ranking per query, or a ranking is not a list of
            unique string IDs.
    """
    _check_top_k(top_k)
    items = check_queries(queries)
    ranked = list(rankings)
    if len(ranked) != len(items):
        raise ValueError(f"Got {len(ranked)} rankings for {len(items)} queries.")
    results = []
    for query, retrieved_ids in zip(items, ranked):
        try:
            hit = hit_at_k(query.relevant_ids, retrieved_ids, top_k)
            recall = recall_at_k(query.relevant_ids, retrieved_ids, top_k)
            precision = precision_at_k(query.relevant_ids, retrieved_ids, top_k)
            rank_score = reciprocal_rank(query.relevant_ids, retrieved_ids, top_k)
        except ValueError as exc:
            raise ValueError(f"Query {query.id!r}: {exc}") from None
        results.append(QueryEvaluation(query, tuple(retrieved_ids[:top_k]), hit, recall, precision, rank_score))
    return EvaluationReport(top_k, tuple(results))


# --- Searchers ------------------------------------------------------------------


class Searcher(Protocol):
    """What ``evaluate()`` needs: the document collection and a ``search()`` method."""

    @property
    def documents(self) -> tuple[Document, ...]: ...

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]: ...


class SemanticSearcher:
    """Semantic search: cosine similarity between the query's embedding and each document's.

    The documents are embedded once, when the searcher is created, and those
    embeddings are reused for every query. The searcher keeps its own copy of
    the collection, so the collection cannot change under it; to search other
    documents, create a new searcher.
    """

    def __init__(self, encoder: Encoder, documents: Sequence[Document]) -> None:
        """Embed ``documents`` in one batch.

        Raises:
            ValueError: the documents fail ``check_documents()`` (the model is
                not called), or the model output is unusable.
        """
        self._encoder = encoder
        self._documents = check_documents(documents)
        embeddings = embed_texts(encoder, [document.text for document in self._documents])
        embeddings.setflags(write=False)  # callers cannot change the cached vectors by accident
        self._embeddings = embeddings

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._documents

    @property
    def document_embeddings(self) -> np.ndarray:
        """One read-only row per document, in collection order, computed once in ``__init__``."""
        return self._embeddings

    def scores(self, query: str) -> list[float]:
        """Return the cosine similarity between ``query`` and each document, in collection order.

        Raises:
            ValueError: the query is not a non-blank string (the model is not
                called), or the model output is unusable.
        """
        _check_text(query, "The query")
        query_vector = embed_texts(self._encoder, [query])[0]
        return [cosine_similarity(query_vector, row) for row in self._embeddings]

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Return the ``top_k`` documents most similar to ``query``, most similar first; ties by ID.

        Raises:
            ValueError: ``top_k`` is not a whole number of at least 1, or any
                problem listed under ``scores()``. Neither calls the model.
        """
        _check_top_k(top_k)
        return _top_results(self._documents, self.scores(query), top_k)


_WORD = re.compile(r"[a-z0-9]+")

# Common English function words. They are fixed in advance, not tuned on the labelled queries.
STOP_WORDS = frozenset(
    """
    a about after all also an and any are as at be been before both but by can could did do does don
    each even every for from had has have how i if in into is it its just me more most my no not of on
    one only or other our out over own same should so some such than that the their them then there
    these they this those through to too under until up very was we were what when whenever where which
    while who why will with without would you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Return the lower-cased words of ``text``, without stop words, one-letter words and a final plural "s".

    ``"Databases"`` becomes ``"database"``; words ending in "ss" keep it. There
    is no other stemming, so ``"pretending"`` does not match ``"pretend"``.
    """
    words = []
    for word in _WORD.findall(text.lower()):
        if len(word) < 2 or word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return words


class KeywordSearcher:
    """Keyword baseline: scores each document by the rare words it shares with the query.

    ``score(q, d)`` is the sum of ``idf(w) = ln(N / df(w))`` over the distinct
    words ``w`` found in both the query and the document, where ``N`` is the
    number of documents and ``df(w)`` the number that contain ``w``. A word in
    every document scores 0. Only documents with a score above 0 are returned,
    so a query that shares no informative word with any document gets no
    results (or fewer than ``top_k``) rather than arbitrary ones.
    """

    def __init__(self, documents: Sequence[Document]) -> None:
        """Index ``documents``. Raises ValueError if they fail ``check_documents()``."""
        self._documents = check_documents(documents)
        self._document_words = [frozenset(tokenize(document.text)) for document in self._documents]
        counts = Counter(word for words in self._document_words for word in words)
        total = len(self._documents)
        self._idf = {word: math.log(total / count) for word, count in counts.items()}

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._documents

    def scores(self, query: str) -> list[float]:
        """Return the keyword score of each document for ``query``, in collection order.

        Raises:
            ValueError: the query is not a non-blank string.
        """
        _check_text(query, "The query")
        query_words = set(tokenize(query))
        # fsum is exactly rounded, so the score does not depend on the (hash-seeded) order of the set.
        return [math.fsum(self._idf[word] for word in query_words & words) for words in self._document_words]

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Return up to ``top_k`` documents that share a word with ``query``, best first; ties by ID.

        Raises:
            ValueError: ``top_k`` is not a whole number of at least 1, or the
                query is not a non-blank string.
        """
        _check_top_k(top_k)
        scores = self.scores(query)
        matching = [index for index, score in enumerate(scores) if score > 0]
        if not matching:
            return []
        return _top_results([self._documents[i] for i in matching], [scores[i] for i in matching], top_k)


def evaluate(searcher: Searcher, queries: Sequence[LabelledQuery], top_k: int = DEFAULT_TOP_K) -> EvaluationReport:
    """Run every labelled query through ``searcher`` and score its first ``top_k`` results.

    Raises:
        ValueError: ``top_k`` is invalid, the queries fail ``check_queries()``,
            or a query lists a relevant ID that is not in the searcher's
            collection. These are checked before anything is searched.
    """
    _check_top_k(top_k)
    items = check_queries(queries)
    check_labels(items, searcher.documents)
    rankings = [[result.id for result in searcher.search(query.text, top_k=top_k)] for query in items]
    return evaluate_rankings(items, rankings, top_k)


# --- Output ---------------------------------------------------------------------

METHOD_TITLES = {
    "semantic": f"Semantic search ({MODEL_NAME}, cosine similarity)",
    "keyword": "Keyword baseline (IDF-weighted word overlap)",
}


def format_results(query: str, results: Sequence[SearchResult], method: str, top_k: int, document_count: int) -> str:
    """Format search results as a table: rank, ID, score and the full text."""
    lines = [f"Query: {query}", f"{METHOD_TITLES[method]}: top {top_k} of {document_count} documents", ""]
    if not results:
        return "\n".join(lines + ["  No document shares a keyword with the query."])
    id_width = max(2, *(len(result.id) for result in results))
    lines.append(f"  Rank  {'ID':<{id_width}}  {'Score':>6}  Text")
    lines += [f"  {result.rank:<4}  {result.id:<{id_width}}  {result.score:6.3f}  {result.text}" for result in results]
    if len(results) < top_k:
        lines += ["", f"  Only {len(results)} document(s) share a keyword with the query."]
    return "\n".join(lines)


def _format_query_results(report: EvaluationReport) -> list[str]:
    """Two lines per query: the outcome and IDs, then the query text."""
    relevant_lists = [", ".join(result.query.relevant_ids) for result in report.results]
    relevant_width = max(len(text) for text in relevant_lists)
    id_width = max(len(result.query.id) for result in report.results)
    k = report.top_k
    lines = []
    for result, relevant in zip(report.results, relevant_lists):
        found = len(set(result.query.relevant_ids) & set(result.retrieved_ids))
        outcome = "hit " if result.hit else "MISS"
        first = str(round(1 / result.reciprocal_rank)) if result.reciprocal_rank else "-"
        retrieved = ", ".join(result.retrieved_ids) or "(none)"
        lines.append(
            f"  {result.query.id:<{id_width}}  {outcome}  recall@{k} {found}/{len(result.query.relevant_ids)}"
            f"  precision@{k} {found}/{k}  first relevant: {first:<{len(str(k))}}"
            f"  relevant: {relevant:<{relevant_width}}  retrieved: {retrieved}"
        )
        lines.append(f"  {'':<{id_width}}  {result.query.text}")
    return lines


def format_evaluation(reports: Mapping[str, EvaluationReport], document_count: int) -> str:
    """Format per-query results for each method, a summary table and the metric definitions."""
    first = next(iter(reports.values()))
    k = first.top_k
    query_count = len(first.results)
    label_count = sum(len(result.query.relevant_ids) for result in first.results)
    lines = [
        f"Evaluated {query_count} labelled queries ({label_count} relevant-document labels) "
        f"against {document_count} documents, top {k}.",
        "",
    ]
    for method, report in reports.items():
        lines += [METHOD_TITLES[method], *_format_query_results(report), ""]
        lines += [
            f"  Top-{k} hit rate: {report.hits} of {query_count} queries = {report.hit_rate:.1%}",
            f"  Recall@{k}:       {report.mean_recall:.3f}",
            f"  Precision@{k}:    {report.mean_precision:.3f} (best possible on these labels: {report.max_mean_precision:.3f})",
            f"  MRR@{k}:          {report.mrr:.3f}",
            "",
        ]
    hit_heading = f"Top-{k} hit rate"
    lines += [
        f"Summary ({query_count} queries)",
        f"  {'Method':<9} {hit_heading:<18} {f'Recall@{k}':<9} {f'Precision@{k}':<12} MRR@{k}",
    ]
    for method, report in reports.items():
        hit_rate = f"{report.hit_rate:.1%} ({report.hits}/{query_count})"
        lines.append(
            f"  {method:<9} {hit_rate:<18} {report.mean_recall:<9.3f} {report.mean_precision:<12.3f} {report.mrr:.3f}"
        )
    lines += [
        "",
        f"Top-{k} hit rate: the share of queries with at least one relevant document in the top {k}.",
        f"Recall@{k}: for each query, (relevant documents in the top {k}) / (all of its relevant documents),",
        "  averaged over the queries. The two are equal for a query with exactly one relevant document.",
        f"Precision@{k}: for each query, (relevant documents in the top {k}) / {k}, averaged over the queries.",
        f"  A query with fewer than {k} relevant documents cannot reach 1; an empty slot counts as a miss.",
        f"MRR@{k}: for each query, 1 / (rank of the first relevant document in the top {k}), or 0 if there is none,",
        "  averaged over the queries.",
    ]
    return "\n".join(lines)


# --- Command line ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.semantic_search",
        description=(
            "Day 4: semantic search over 20 short documents with a local Sentence Transformers model, "
            "compared with a keyword baseline. No API key is needed."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="{search,evaluate}")
    search = commands.add_parser("search", help="show the best matches for one query")
    search.add_argument("query", help="the text to search for")
    search.add_argument(
        "--method",
        choices=METHODS,
        default="semantic",
        help="semantic (default) or keyword, the baseline, which does not load the model",
    )
    evaluate_command = commands.add_parser(
        "evaluate", help="score semantic search and the keyword baseline on the labelled queries"
    )
    for command in (search, evaluate_command):
        command.add_argument(
            "--top-k",
            type=_positive_int,
            default=DEFAULT_TOP_K,
            help=f"number of documents to retrieve per query (default: {DEFAULT_TOP_K})",
        )
        command.add_argument(
            "--offline",
            action="store_true",
            help="load the model from the local cache only; fail instead of downloading it",
        )
    return parser


def _load_model_or_report(offline: bool) -> Encoder | None:
    """Load the model as ``python -m src.embeddings`` does; on failure print why and return None."""
    print(f"Loading {MODEL_NAME} ...")
    try:
        return load_model(local_files_only=offline)
    except ImportError as exc:
        print(f"Error: sentence-transformers is not installed ({exc}). Run: pip install -r requirements.txt", file=sys.stderr)
    except OSError as exc:
        hint = (
            "It is not in the local cache; run once without --offline to download it."
            if offline
            else "The first run needs internet access to download it (about 90 MB)."
        )
        print(f"Error: could not load {MODEL_NAME}: {exc}\n{hint}", file=sys.stderr)
    return None


def _load_dataset(with_queries: bool) -> tuple[tuple[Document, ...], tuple[LabelledQuery, ...]] | None:
    """Load the documents (and the labelled queries if asked); on failure print which file failed and return None."""
    path = DOCUMENTS_PATH
    try:
        documents = load_documents(path)
        queries: tuple[LabelledQuery, ...] = ()
        if with_queries:
            path = QUERIES_PATH
            queries = load_queries(documents, path)
    except (OSError, ValueError) as exc:
        print(f"Error: could not load {path}: {exc}", file=sys.stderr)
        return None
    return documents, queries


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "search":
        try:
            _check_text(args.query, "The query")
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_USAGE

    dataset = _load_dataset(with_queries=args.command == "evaluate")
    if dataset is None:
        return EXIT_FAILURE
    documents, queries = dataset

    model = None
    if args.command == "evaluate" or args.method == "semantic":
        model = _load_model_or_report(args.offline)
        if model is None:
            return EXIT_FAILURE

    if args.command == "search":
        searcher = SemanticSearcher(model, documents) if args.method == "semantic" else KeywordSearcher(documents)
        results = searcher.search(args.query, top_k=args.top_k)
        print(format_results(args.query, results, args.method, args.top_k, len(documents)))
        return EXIT_OK

    reports = {
        "semantic": evaluate(SemanticSearcher(model, documents), queries, top_k=args.top_k),
        "keyword": evaluate(KeywordSearcher(documents), queries, top_k=args.top_k),
    }
    print(format_evaluation(reports, len(documents)))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
