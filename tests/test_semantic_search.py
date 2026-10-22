"""Thursday: semantic search, the keyword baseline and the top-3 evaluation.

Ranking, the metrics, validation and the keyword baseline are tested directly
with hand-worked values. Semantic search uses fake encoders that return fixed
vectors, so none of those tests imports PyTorch or touches a model. Tests
marked ``model`` use the real all-MiniLM-L6-v2 from the local Hugging Face
cache only (network blocked), are skipped when it is not cached, and check
that the pipeline works and still gives the results reported in the README.

Run everything except the real-model tests with:  python -m pytest -m "not model"
"""

from __future__ import annotations

import json
import math
import socket
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import semantic_search
from src.embeddings import MODEL_DIMENSIONS, MODEL_NAME, load_model
from src.semantic_search import (
    DEFAULT_TOP_K,
    Document,
    KeywordSearcher,
    LabelledQuery,
    SearchResult,
    SemanticSearcher,
    check_documents,
    check_labels,
    check_queries,
    evaluate,
    evaluate_rankings,
    format_evaluation,
    hit_at_k,
    load_documents,
    load_queries,
    main,
    parse_documents,
    parse_queries,
    precision_at_k,
    rank_by_score,
    recall_at_k,
    reciprocal_rank,
    tokenize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Not whole numbers of at least 1. True and False are rejected even though bool is a subclass of int.
INVALID_TOP_K = [0, -1, 2.5, True, False, "3", None]


class FakeEncoder:
    """Stands in for SentenceTransformer: looks up a fixed vector for each text and records every batch."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, sentences, /, **kwargs):
        self.calls.append(list(sentences))
        return np.array([self.vectors[text] for text in sentences], dtype=np.float64)


class LetterEncoder:
    """Embeds any text as its 26 letter counts: meaningless but deterministic, for the CLI on the real dataset."""

    def encode(self, sentences, /, **kwargs):
        return np.array(
            [[text.lower().count(letter) for letter in string.ascii_lowercase] for text in sentences], dtype=np.float64
        )


# Unit vectors whose cosine similarity with the query [1, 0, 0] is simply their first component.
VECTORS = {
    "query": [1.0, 0.0, 0.0],
    "text a": [0.6, 0.8, 0.0],  # 0.6
    "text b": [1.0, 0.0, 0.0],  # 1.0
    "text c": [0.0, 0.0, 1.0],  # 0.0
    "text d": [0.8, 0.0, 0.6],  # 0.8
    "text e": [-1.0, 0.0, 0.0],  # -1.0
}
DOCUMENTS = tuple(Document(f"doc-{letter}", f"text {letter}") for letter in "abcde")
DOCUMENT_TEXTS = [document.text for document in DOCUMENTS]


def ids(results: list[SearchResult]) -> list[str]:
    return [result.id for result in results]


def semantic_searcher(documents=DOCUMENTS, vectors=VECTORS) -> SemanticSearcher:
    return SemanticSearcher(FakeEncoder(vectors), documents)


# --- The dataset ------------------------------------------------------------------


def test_the_dataset_has_20_documents_with_unique_ids_across_six_subjects():
    documents = load_documents()
    assert len(documents) == 20
    assert len({document.id for document in documents}) == 20
    assert {document.id.split("-")[0] for document in documents} == {"ml", "py", "db", "cloud", "sec", "fin"}


def test_every_labelled_query_points_at_existing_documents_and_every_document_is_labelled():
    documents = load_documents()
    queries = load_queries(documents)
    assert len(queries) == 23
    check_labels(queries, documents)
    labelled = {document_id for query in queries for document_id in query.relevant_ids}
    assert labelled == {document.id for document in documents}


# --- Documents: validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("document_id", "text", "message"),
    [
        ("", "some text", "Document ID is empty"),
        ("   ", "some text", "Document ID is empty"),
        (" doc-a", "some text", "leading or trailing whitespace"),
        (7, "some text", "Document ID must be a string; got int"),
        (None, "some text", "Document ID must be a string"),
        ("doc-a", "", "Text of document 'doc-a' is empty"),
        ("doc-a", " \n\t", "Text of document 'doc-a' is empty"),
        ("doc-a", None, "Text of document 'doc-a' must be a string"),
        ("doc-a", ["some text"], "must be a string; got list"),
    ],
)
def test_malformed_documents_are_rejected(document_id, text, message):
    with pytest.raises(ValueError, match=message):
        Document(document_id, text)


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ([], "The list of documents is empty"),
        ("a single string", "must be a list; got str"),
        (Document("doc-a", "text a"), "must be a list; got Document"),
        ({"id": "doc-a", "text": "text a"}, "must be a list; got dict"),
        (None, "must be a list; got NoneType"),
        ([Document("doc-a", "text a"), "text b"], "Item 1 of the documents must be a Document; got str"),
        ([{"id": "doc-a", "text": "text a"}], "Item 0 of the documents must be a Document; got dict"),
    ],
)
def test_check_documents_rejects_bad_collections(documents, message):
    with pytest.raises(ValueError, match=message):
        check_documents(documents)


def test_duplicate_document_ids_are_rejected_and_all_are_named():
    documents = [Document("b", "one"), Document("a", "two"), Document("b", "three"), Document("a", "four")]
    with pytest.raises(ValueError, match=r"Duplicate document ID\(s\): a, b\."):
        check_documents(documents)


@pytest.mark.parametrize("make_searcher", [lambda docs: SemanticSearcher(FakeEncoder(VECTORS), docs), KeywordSearcher])
def test_searchers_reject_duplicate_ids(make_searcher):
    with pytest.raises(ValueError, match="Duplicate document ID"):
        make_searcher([Document("doc-a", "text a"), Document("doc-a", "text b")])


def test_semantic_searcher_rejects_bad_documents_without_calling_the_model():
    encoder = FakeEncoder(VECTORS)
    with pytest.raises(ValueError, match="Duplicate document ID"):
        SemanticSearcher(encoder, [Document("doc-a", "text a"), Document("doc-a", "text b")])
    with pytest.raises(ValueError, match="empty"):
        SemanticSearcher(encoder, [])
    assert encoder.calls == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"id": "doc-a", "text": "text a"}, "Expected a JSON list; got dict"),
        ([], "The JSON list is empty"),
        (["text a"], "Entry 0 must be a JSON object; got str"),
        ([{"id": "doc-a"}], r"Entry 0 must have exactly the fields id, text \(missing text\)"),
        ([{"id": "doc-a", "text": "t", "topic": "x"}], r"\(unexpected topic\)"),
        ([{"id": "doc-a", "txt": "t"}], r"\(missing text; unexpected txt\)"),
        ([{"id": "doc-a", "text": "t"}, {"id": 2, "text": "t"}], "Entry 1: Document ID must be a string; got int"),
        ([{"id": "doc-a", "text": "   "}], "Entry 0: Text of document 'doc-a' is empty"),
        ([{"id": "doc-a", "text": "t"}, {"id": "doc-a", "text": "u"}], "Duplicate document ID"),
    ],
)
def test_parse_documents_rejects_malformed_json(data, message):
    with pytest.raises(ValueError, match=message):
        parse_documents(data)


def test_load_documents_reads_a_valid_file(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{"id": "doc-a", "text": "text a"}, {"id": "doc-b", "text": "text b"}]), encoding="utf-8")
    assert load_documents(path) == (Document("doc-a", "text a"), Document("doc-b", "text b"))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('[{"id": "doc-a", "text": "text a"},]', "is not valid JSON"),
        ('[{"id": "doc-a", "id": "doc-b", "text": "text a"}]', r"repeats the key\(s\): id"),
    ],
)
def test_load_documents_rejects_broken_files(tmp_path, content, message):
    path = tmp_path / "documents.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_documents(path)


def test_load_documents_reports_a_missing_file(tmp_path):
    with pytest.raises(OSError):
        load_documents(tmp_path / "no-such-file.json")


# --- Ranking (pure) -------------------------------------------------------------------


def test_rank_by_score_orders_from_highest_to_lowest():
    assert rank_by_score(["a", "b", "c", "d"], [0.2, 0.9, -0.5, 0.4], top_k=4) == [1, 3, 0, 2]
    assert rank_by_score(["a", "b", "c", "d"], [0.2, 0.9, -0.5, 0.4]) == [1, 3, 0]  # default top 3


def test_rank_by_score_breaks_ties_by_id_not_by_position():
    assert rank_by_score(["c", "a", "b"], [0.5, 0.5, 0.5]) == [1, 2, 0]
    # A tie across the top-k boundary keeps the smallest IDs.
    assert rank_by_score(["d", "c", "b", "a", "z"], [0.5, 0.5, 0.5, 0.5, 0.9], top_k=3) == [4, 3, 2]


@pytest.mark.parametrize(("top_k", "expected"), [(1, [1]), (2, [1, 0]), (3, [1, 0]), (100, [1, 0])])
def test_rank_by_score_top_k_larger_than_the_collection_returns_everything(top_k, expected):
    assert rank_by_score(["a", "b"], [0.1, 0.2], top_k=top_k) == expected


@pytest.mark.parametrize(
    ("ids_", "scores", "message"),
    [
        (["a", "b"], [0.1], "Got 2 IDs but 1 scores"),
        (["a", "a"], [0.1, 0.2], r"Duplicate ID\(s\): a"),
        (["a"], [float("nan")], "NaN or infinity"),
        (["a"], [float("inf")], "NaN or infinity"),
        (["a"], ["0.5"], "only numbers"),
        ([], [], "empty"),
    ],
)
def test_rank_by_score_rejects_invalid_input(ids_, scores, message):
    with pytest.raises(ValueError, match=message):
        rank_by_score(ids_, scores)


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_rank_by_score_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a whole number of at least 1"):
        rank_by_score(["a"], [0.5], top_k=top_k)


# --- Semantic search (fake encoder, no model) -------------------------------------------


def test_semantic_search_returns_the_top_three_by_cosine_similarity():
    results = semantic_searcher().search("query")
    assert results == [
        SearchResult(1, "doc-b", "text b", pytest.approx(1.0, abs=1e-12)),
        SearchResult(2, "doc-d", "text d", pytest.approx(0.8, abs=1e-12)),
        SearchResult(3, "doc-a", "text a", pytest.approx(0.6, abs=1e-12)),
    ]
    assert DEFAULT_TOP_K == 3


def test_semantic_scores_are_the_cosine_similarities_in_collection_order():
    assert semantic_searcher().scores("query") == pytest.approx([0.6, 1.0, 0.0, 0.8, -1.0], abs=1e-12)


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [
        (1, ["doc-b"]),
        (5, ["doc-b", "doc-d", "doc-a", "doc-c", "doc-e"]),
        (50, ["doc-b", "doc-d", "doc-a", "doc-c", "doc-e"]),  # more than there are documents
    ],
)
def test_semantic_search_top_k(top_k, expected):
    results = semantic_searcher().search("query", top_k=top_k)
    assert ids(results) == expected
    assert [result.rank for result in results] == list(range(1, len(expected) + 1))


def test_semantic_ties_are_broken_by_id_whatever_the_collection_order():
    vectors = {"query": [1.0, 0.0], "best": [1.0, 0.0], "up": [0.6, 0.8], "down": [0.6, -0.8], "up again": [0.6, 0.8]}
    documents = [Document("tie-c", "up"), Document("tie-a", "down"), Document("top", "best"), Document("tie-b", "up again")]
    for order in (documents, documents[::-1], [documents[2], documents[3], documents[0], documents[1]]):
        results = semantic_searcher(order, vectors).search("query", top_k=4)
        assert ids(results) == ["top", "tie-a", "tie-b", "tie-c"]
        assert results[1].score == results[2].score == results[3].score  # exact ties, not just close
        assert semantic_searcher(order, vectors).search("query", top_k=2) == results[:2]


def test_semantic_search_is_repeatable():
    searcher = semantic_searcher()
    assert searcher.search("query") == searcher.search("query") == semantic_searcher().search("query")


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "The query is empty"),
        ("   \n\t", "The query is empty"),
        (None, "The query must be a string; got NoneType"),
        (42, "The query must be a string; got int"),
        (["query"], "The query must be a string; got list"),
    ],
)
def test_semantic_search_rejects_invalid_queries_without_calling_the_model(query, message):
    searcher = semantic_searcher()
    with pytest.raises(ValueError, match=message):
        searcher.search(query)
    assert searcher._encoder.calls == [DOCUMENT_TEXTS]  # only the documents, when the searcher was built


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_semantic_search_rejects_invalid_top_k_without_calling_the_model(top_k):
    searcher = semantic_searcher()
    with pytest.raises(ValueError, match="top_k must be a whole number of at least 1"):
        searcher.search("query", top_k=top_k)
    assert searcher._encoder.calls == [DOCUMENT_TEXTS]


def test_semantic_search_rejects_a_query_vector_of_the_wrong_length():
    vectors = {**VECTORS, "short query": [1.0, 0.0]}
    with pytest.raises(ValueError, match="same length"):
        semantic_searcher(vectors=vectors).search("short query")


def test_semantic_search_rejects_a_zero_query_vector():
    vectors = {**VECTORS, "zero": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="zero vector"):
        semantic_searcher(vectors=vectors).search("zero")


# --- Caching: document embeddings are computed once --------------------------------------


def test_document_embeddings_are_computed_once_and_reused_for_every_query():
    searcher = semantic_searcher()
    encoder = searcher._encoder
    assert encoder.calls == [DOCUMENT_TEXTS]  # one batch for all the documents
    cached = searcher.document_embeddings
    np.testing.assert_array_equal(cached, [VECTORS[text] for text in DOCUMENT_TEXTS])
    for _ in range(3):
        searcher.search("query")
    searcher.scores("query")
    assert encoder.calls == [DOCUMENT_TEXTS] + [["query"]] * 4  # then only the query, once per search
    assert searcher.document_embeddings is cached


def test_cached_document_embeddings_are_read_only():
    searcher = semantic_searcher()
    with pytest.raises(ValueError, match="read-only"):
        searcher.document_embeddings[0, 0] = 99.0
    assert ids(searcher.search("query")) == ["doc-b", "doc-d", "doc-a"]


def test_the_searcher_keeps_its_own_copy_of_the_collection():
    documents = list(DOCUMENTS)
    searcher = semantic_searcher(documents)
    documents.append(Document("doc-new", "query"))  # changing the caller's list afterwards...
    documents.pop(1)
    assert searcher.documents == DOCUMENTS  # ...does not change the searcher
    assert ids(searcher.search("query")) == ["doc-b", "doc-d", "doc-a"]
    assert searcher._encoder.calls == [DOCUMENT_TEXTS, ["query"]]


def test_a_new_collection_needs_a_new_searcher_which_embeds_it():
    encoder = FakeEncoder(VECTORS)
    first = SemanticSearcher(encoder, DOCUMENTS)
    second = SemanticSearcher(encoder, DOCUMENTS[:2])
    assert encoder.calls == [DOCUMENT_TEXTS, DOCUMENT_TEXTS[:2]]
    assert ids(second.search("query")) == ["doc-b", "doc-a"]
    assert first.document_embeddings.shape == (5, 3)


def test_evaluate_embeds_the_documents_once_and_each_query_once():
    encoder = FakeEncoder(VECTORS)
    searcher = SemanticSearcher(encoder, DOCUMENTS)
    queries = [LabelledQuery("q1", "query", ["doc-b"]), LabelledQuery("q2", "text c", ["doc-c"])]
    evaluate(searcher, queries)
    assert encoder.calls == [DOCUMENT_TEXTS, ["query"], ["text c"]]


# --- Day 3 numerical behaviour carried through ------------------------------------------


def test_semantic_scores_keep_the_day3_handling_of_extreme_magnitudes():
    # Squaring these components would overflow (1e200) or underflow (1e-200) a naive norm.
    vectors = {
        "query": [1e200, 1e200],
        "same, huge": [1e200, 1e200],
        "same, tiny": [1e-200, 1e-200],
        "orthogonal": [1e200, -1e200],
        "opposite": [-1e-200, -1e-200],
    }
    documents = [Document("huge", "same, huge"), Document("tiny", "same, tiny"), Document("orth", "orthogonal"), Document("opp", "opposite")]
    results = semantic_searcher(documents, vectors).search("query", top_k=4)
    assert ids(results) == ["huge", "tiny", "orth", "opp"]
    assert [result.score for result in results] == pytest.approx([1.0, 1.0, 0.0, -1.0], abs=1e-12)
    assert all(-1.0 <= result.score <= 1.0 for result in results)  # still clipped to the documented range


def test_self_similarity_never_exceeds_one():
    # Without the Day 3 clipping, some of these round to 1.0000000000000002.
    rng = np.random.default_rng(0)
    texts = [f"text {index}" for index in range(30)]
    vectors = {text: rng.normal(size=MODEL_DIMENSIONS).tolist() for text in texts}
    searcher = semantic_searcher([Document(f"doc-{index:02d}", text) for index, text in enumerate(texts)], vectors)
    for text in texts:
        assert searcher.search(text, top_k=1)[0].score <= 1.0


def test_a_zero_document_vector_is_reported_not_scored_as_zero():
    vectors = {**VECTORS, "text c": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="zero vector"):
        semantic_searcher(vectors=vectors).search("query")


# --- Keyword baseline ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cats chase the mice; dogs don't.", ["cat", "chase", "mice", "dog"]),
        ("SQL Databases and their INDEXES", ["sql", "database", "indexe"]),
        ("a class is not a bus", ["class", "bus"]),  # "ss" and short words keep their final "s"
        ("Python 3 and x * x", ["python"]),  # one-character words are dropped
        ("the and of to", []),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


ANIMALS = (
    Document("a", "Cats chase mice"),
    Document("b", "Dogs chase cats"),
    Document("c", "Birds sing songs"),
)


def test_keyword_scores_are_idf_weighted_word_overlap():
    # N = 3. df: cat 2, chase 2, mice 1, dog 1, bird 1, sing 1, song 1. idf(w) = ln(3 / df(w)).
    scores = KeywordSearcher(ANIMALS).scores("mice and cats")
    assert scores == pytest.approx([math.log(3) + math.log(3 / 2), math.log(3 / 2), 0.0], abs=1e-12)


def test_keyword_search_returns_only_documents_that_share_a_word():
    results = KeywordSearcher(ANIMALS).search("mice and cats")
    assert ids(results) == ["a", "b"]  # "c" scores 0 and is left out, although top_k is 3
    assert results[0].score == pytest.approx(math.log(4.5), abs=1e-12)
    assert KeywordSearcher(ANIMALS).search("quantum physics") == []
    assert KeywordSearcher(ANIMALS).search("the and of") == []  # only stop words: a valid query, no matches


def test_a_word_in_every_document_carries_no_weight():
    searcher = KeywordSearcher([Document("a", "data about cats"), Document("b", "data about dogs")])
    assert searcher.scores("data") == [0.0, 0.0]
    assert searcher.search("data") == []


def test_keyword_ties_are_broken_by_id_whatever_the_collection_order():
    documents = [Document("z", "red apple"), Document("m", "red apple pie"), Document("a", "red apple"), Document("x", "green")]
    for order in (documents, documents[::-1]):
        results = KeywordSearcher(order).search("red apple")
        assert ids(results) == ["a", "m", "z"]
        assert results[0].score == results[1].score == results[2].score


@pytest.mark.parametrize(("query", "message"), [("", "empty"), ("  ", "empty"), (None, "must be a string")])
def test_keyword_search_rejects_invalid_queries(query, message):
    with pytest.raises(ValueError, match=message):
        KeywordSearcher(ANIMALS).search(query)


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_keyword_search_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a whole number of at least 1"):
        KeywordSearcher(ANIMALS).search("cats", top_k=top_k)


def test_keyword_baseline_evaluation_worked_by_hand():
    queries = [
        LabelledQuery("k1", "mice", ["a"]),  # retrieves [a]: hit, recall 1/1
        LabelledQuery("k2", "pet cats", ["a", "b"]),  # "cat" ties a and b: [a, b]: hit, recall 2/2
        LabelledQuery("k3", "animals that chase", ["a", "b", "c"]),  # only "chase" matches: [a, b]: hit, recall 2/3
        LabelledQuery("k4", "puppies", ["b"]),  # no shared word: []: miss, recall 0/1
    ]
    report = evaluate(KeywordSearcher(ANIMALS), queries)
    assert [result.retrieved_ids for result in report.results] == [("a",), ("a", "b"), ("a", "b"), ()]
    assert [result.hit for result in report.results] == [True, True, True, False]
    assert [result.recall for result in report.results] == pytest.approx([1.0, 1.0, 2 / 3, 0.0])
    assert report.hits == 3
    assert report.hit_rate == pytest.approx(3 / 4)
    assert report.mean_recall == pytest.approx((1 + 1 + 2 / 3 + 0) / 4)  # 0.667, below the hit rate
    # The divisor is always 3, so the empty slots of the short result lists count as misses.
    assert [result.precision for result in report.results] == pytest.approx([1 / 3, 2 / 3, 2 / 3, 0.0])
    assert report.mean_precision == pytest.approx(5 / 12)
    assert report.mrr == pytest.approx(3 / 4)


def test_semantic_and_keyword_are_scored_on_the_same_queries_and_labels():
    vectors = {**VECTORS, "text f": [0.0, 1.0, 0.0]}
    documents = (*DOCUMENTS, Document("doc-f", "text f"))
    queries = [LabelledQuery("q1", "query", ["doc-b", "doc-c"]), LabelledQuery("q2", "text f", ["doc-f"])]
    semantic = evaluate(semantic_searcher(documents, vectors), queries)
    keyword = evaluate(KeywordSearcher(documents), queries)
    # Semantic: q1 -> [b, d, a] finds b only (1/2); q2 -> [f, a, b] finds f (1/1).
    assert (semantic.hit_rate, semantic.mean_recall) == (1.0, pytest.approx(0.75))
    # Keyword: one-letter words are dropped, so every document is just "text", whose idf is ln(6/6) = 0.
    # Neither query retrieves anything: two misses.
    assert (keyword.hit_rate, keyword.mean_recall) == (0.0, 0.0)
    assert [r.query for r in semantic.results] == [r.query for r in keyword.results]


# --- Metrics (pure), worked by hand ---------------------------------------------------------

# Four queries, top 3:
#   m1  relevant {a}           retrieved [a, b, c]  -> hit, recall 1/1 = 1,   precision 1/3, first relevant 1st
#   m2  relevant {b, d}        retrieved [a, d, c]  -> hit, recall 1/2 = 0.5, precision 1/3, first relevant 2nd
#   m3  relevant {e}           retrieved [a, b, c]  -> miss, recall 0/1 = 0,  precision 0/3, none
#   m4  relevant {a, b, c, d}  retrieved [a, b, e]  -> hit, recall 2/4 = 0.5, precision 2/3, first relevant 1st
# Top-3 hit rate = 3/4 = 0.75.  Recall@3 = (1 + 0.5 + 0 + 0.5) / 4 = 0.5.
# Precision@3 = (1/3 + 1/3 + 0 + 2/3) / 4 = 1/3; at best (1 + 2 + 1 + 3) / 12 = 7/12.
# MRR@3 = (1 + 1/2 + 0 + 1) / 4 = 0.625.
WORKED_QUERIES = [
    LabelledQuery("m1", "first", ["a"]),
    LabelledQuery("m2", "second", ["b", "d"]),
    LabelledQuery("m3", "third", ["e"]),
    LabelledQuery("m4", "fourth", ["a", "b", "c", "d"]),
]
WORKED_RANKINGS = [["a", "b", "c"], ["a", "d", "c"], ["a", "b", "c"], ["a", "b", "e"]]


def test_hit_rate_and_recall_worked_by_hand():
    report = evaluate_rankings(WORKED_QUERIES, WORKED_RANKINGS)
    assert [result.hit for result in report.results] == [True, True, False, True]
    assert [result.recall for result in report.results] == [1.0, 0.5, 0.0, 0.5]
    assert report.hits == 3
    assert report.hit_rate == 0.75
    assert report.mean_recall == 0.5
    assert report.top_k == 3


def test_precision_and_mrr_worked_by_hand():
    report = evaluate_rankings(WORKED_QUERIES, WORKED_RANKINGS)
    assert [result.precision for result in report.results] == pytest.approx([1 / 3, 1 / 3, 0.0, 2 / 3])
    assert [result.reciprocal_rank for result in report.results] == [1.0, 0.5, 0.0, 1.0]
    assert report.mean_precision == pytest.approx(1 / 3)
    assert report.max_mean_precision == pytest.approx(7 / 12)
    assert report.mrr == 0.625


@pytest.mark.parametrize(
    ("relevant", "retrieved", "top_k", "precision", "rank_score"),
    [
        (["a"], ["a", "b", "c"], 3, 1 / 3, 1.0),  # one relevant document: 1/3 is the most Precision@3 can be
        (["c"], ["a", "b", "c"], 3, 1 / 3, 1 / 3),  # rank 3
        (["b", "c"], ["a", "b", "c"], 3, 2 / 3, 1 / 2),  # only the first relevant document sets the reciprocal rank
        (["d"], ["a", "b", "c", "d"], 3, 0.0, 0.0),  # rank 4 does not count in the top 3...
        (["d"], ["a", "b", "c", "d"], 4, 1 / 4, 1 / 4),  # ...but does in the top 4
        (["a", "b"], ["a"], 3, 1 / 3, 1.0),  # fewer results than top_k: the empty slots count as misses
        (["a"], [], 3, 0.0, 0.0),  # no results at all
        (["a", "b", "c", "d"], ["a", "b", "c"], 3, 1.0, 1.0),
    ],
)
def test_precision_at_k_and_reciprocal_rank(relevant, retrieved, top_k, precision, rank_score):
    assert precision_at_k(relevant, retrieved, top_k) == precision
    assert reciprocal_rank(relevant, retrieved, top_k) == rank_score


def test_evaluation_output_shows_precision_and_the_first_relevant_rank_of_each_query():
    out = format_evaluation({"semantic": evaluate_rankings(WORKED_QUERIES, WORKED_RANKINGS)}, document_count=5)
    assert "  m2  hit   recall@3 1/2  precision@3 1/3  first relevant: 2  relevant: b, d " in out
    assert "  m3  MISS  recall@3 0/1  precision@3 0/3  first relevant: -  relevant: e " in out
    assert "  Precision@3:    0.333 (best possible on these labels: 0.583)\n" in out
    assert "  MRR@3:          0.625\n" in out
    assert "  Method    Top-3 hit rate     Recall@3  Precision@3  MRR@3\n" in out
    assert "  semantic  75.0% (3/4)        0.500     0.333        0.625\n" in out


@pytest.mark.parametrize(
    ("relevant", "retrieved", "top_k", "hit", "recall"),
    [
        (["a"], ["a", "b", "c"], 3, True, 1.0),
        (["c"], ["a", "b", "c"], 3, True, 1.0),  # rank 3 still counts
        (["d"], ["a", "b", "c", "d"], 3, False, 0.0),  # rank 4 does not count in the top 3...
        (["d"], ["a", "b", "c", "d"], 4, True, 1.0),  # ...but does in the top 4
        (["a", "b"], ["a"], 3, True, 0.5),  # fewer results than top_k (keyword baseline)
        (["a"], [], 3, False, 0.0),  # no results at all
        (["a", "b", "c", "d"], ["a", "b", "c"], 3, True, 0.75),  # 4 relevant: 1.0 is out of reach in the top 3
        (["x", "y"], ["a", "b", "c"], 3, False, 0.0),
    ],
)
def test_hit_at_k_and_recall_at_k(relevant, retrieved, top_k, hit, recall):
    assert hit_at_k(relevant, retrieved, top_k) is hit
    assert recall_at_k(relevant, retrieved, top_k) == recall


def test_hit_rate_and_recall_coincide_when_every_query_has_one_relevant_document():
    queries = [LabelledQuery(f"s{i}", "text", [relevant]) for i, relevant in enumerate("abcd")]
    report = evaluate_rankings(queries, [["a", "x", "y"], ["x", "b", "y"], ["x", "y", "z"], ["x", "y", "d"]])
    assert report.hit_rate == report.mean_recall == 0.75


def test_evaluate_rankings_keeps_only_the_top_k_retrieved_ids():
    report = evaluate_rankings([LabelledQuery("q1", "text", ["d"])], [["a", "b", "c", "d"]])
    assert report.results[0].retrieved_ids == ("a", "b", "c")
    assert not report.results[0].hit


# --- Labels: empty or invalid -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("relevant_ids", "message"),
    [
        ([], "has no relevant document IDs"),
        ((), "has no relevant document IDs"),
        ("doc-a", "must be a list of document IDs; got str"),
        (None, "must be a list of document IDs; got NoneType"),
        ({"doc-a"}, "must be a list of document IDs; got set"),
        ([""], "A relevant ID of query 'q1' is empty"),
        (["doc-a", 3], "A relevant ID of query 'q1' must be a string; got int"),
        (["doc-a ", "doc-b"], "leading or trailing whitespace"),
        (["doc-a", "doc-b", "doc-a"], r"more than once: doc-a"),
    ],
)
def test_invalid_relevance_labels_are_rejected(relevant_ids, message):
    with pytest.raises(ValueError, match=message):
        LabelledQuery("q1", "some query", relevant_ids)


@pytest.mark.parametrize(
    ("query_id", "text", "message"),
    [("", "text", "Query ID is empty"), ("q1", "  ", "Text of query 'q1' is empty"), (1, "text", "Query ID must be a string")],
)
def test_invalid_labelled_queries_are_rejected(query_id, text, message):
    with pytest.raises(ValueError, match=message):
        LabelledQuery(query_id, text, ["doc-a"])


def test_a_json_list_of_relevant_ids_becomes_a_tuple():
    assert LabelledQuery("q1", "text", ["doc-a", "doc-b"]).relevant_ids == ("doc-a", "doc-b")


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        ([], "The list of labelled queries is empty"),
        (None, "must be a list"),
        ([LabelledQuery("q1", "text", ["a"]), LabelledQuery("q1", "other", ["b"])], r"Duplicate query ID\(s\): q1"),
        ([LabelledQuery("q1", "text", ["a"]), {"id": "q2"}], "Item 1 of the labelled queries must be a LabelledQuery"),
    ],
)
def test_check_queries_rejects_bad_query_lists(queries, message):
    with pytest.raises(ValueError, match=message):
        check_queries(queries)


def test_labels_must_name_documents_in_the_collection():
    with pytest.raises(ValueError, match=r"Query 'q1' lists unknown document ID\(s\): doc-z"):
        check_labels([LabelledQuery("q1", "text", ["doc-a", "doc-z"])], DOCUMENTS)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "The JSON list is empty"),
        ([{"id": "q1", "text": "text"}], r"missing relevant_ids"),
        ([{"id": "q1", "text": "text", "relevant_ids": []}], "Entry 0: Query 'q1' has no relevant document IDs"),
        ([{"id": "q1", "text": "text", "relevant_ids": "doc-a"}], "Entry 0: relevant_ids of query 'q1' must be a list"),
        ([{"id": "q1", "text": "text", "relevant_ids": ["doc-z"]}], "unknown document ID"),
    ],
)
def test_parse_queries_rejects_malformed_labels(data, message):
    with pytest.raises(ValueError, match=message):
        parse_queries(data, DOCUMENTS)


class RecordingSearcher:
    """A searcher that returns fixed IDs and records every query it is asked."""

    def __init__(self, documents, rankings: dict[str, list[str]]) -> None:
        self.documents = tuple(documents)
        self.rankings = rankings
        self.asked: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        self.asked.append((query, top_k))
        return [SearchResult(rank, doc_id, "", 0.0) for rank, doc_id in enumerate(self.rankings[query][:top_k], 1)]


def test_evaluate_checks_every_label_before_searching_anything():
    searcher = RecordingSearcher(DOCUMENTS, {"good": ["doc-a"], "bad": ["doc-a"]})
    queries = [LabelledQuery("q1", "good", ["doc-a"]), LabelledQuery("q2", "bad", ["doc-missing"])]
    with pytest.raises(ValueError, match="unknown document ID"):
        evaluate(searcher, queries)
    assert searcher.asked == []


def test_evaluate_passes_top_k_to_the_searcher():
    documents = [Document(doc_id, doc_id) for doc_id in "abcde"]
    searcher = RecordingSearcher(documents, {"first": ["a", "b", "c", "d", "e"]})
    report = evaluate(searcher, [LabelledQuery("q1", "first", ["d"])], top_k=4)
    assert searcher.asked == [("first", 4)]
    assert report.results[0].retrieved_ids == ("a", "b", "c", "d")
    assert report.hit_rate == 1.0


@pytest.mark.parametrize(
    ("relevant", "retrieved", "message"),
    [
        ([], ["a"], "relevant_ids is empty"),
        ("a", ["a"], "relevant_ids must be a list of document IDs; got str"),
        (["a"], "abc", "retrieved_ids must be a list of document IDs; got str"),
        (["a"], None, "retrieved_ids must be a list"),
        (["a"], ["a", "a"], "retrieved_ids lists ID"),
        (["a", "a"], ["a"], "relevant_ids lists ID"),
        (["a"], [1, 2], "only string IDs"),
    ],
)
def test_metrics_reject_invalid_input(relevant, retrieved, message):
    for metric in (hit_at_k, recall_at_k, precision_at_k, reciprocal_rank):
        with pytest.raises(ValueError, match=message):
            metric(relevant, retrieved)


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_metrics_and_evaluation_reject_invalid_top_k(top_k):
    message = "top_k must be a whole number of at least 1"
    for call in (
        lambda: hit_at_k(["a"], ["a"], top_k),
        lambda: recall_at_k(["a"], ["a"], top_k),
        lambda: precision_at_k(["a"], ["a"], top_k),
        lambda: reciprocal_rank(["a"], ["a"], top_k),
        lambda: evaluate_rankings(WORKED_QUERIES, WORKED_RANKINGS, top_k),
        lambda: evaluate(KeywordSearcher(ANIMALS), [LabelledQuery("q1", "cats", ["a"])], top_k),
    ):
        with pytest.raises(ValueError, match=message):
            call()


def test_evaluate_rankings_needs_one_ranking_per_query():
    with pytest.raises(ValueError, match="Got 3 rankings for 4 queries"):
        evaluate_rankings(WORKED_QUERIES, WORKED_RANKINGS[:3])


def test_evaluate_rankings_names_the_query_with_a_bad_ranking():
    with pytest.raises(ValueError, match="Query 'm2': retrieved_ids lists ID"):
        evaluate_rankings(WORKED_QUERIES, [["a"], ["b", "b"], ["c"], ["d"]])


# --- Module import --------------------------------------------------------------------------


def test_importing_the_module_does_not_load_pytorch():
    # A fresh interpreter: other tests in this process may already have imported the model libraries.
    code = (
        "import sys; import src.semantic_search as s; "
        "assert len(s.load_documents()) == 20; "
        "print(sorted(m for m in ('torch', 'transformers', 'sentence_transformers') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# --- Command line ---------------------------------------------------------------------------


@pytest.fixture
def fake_model_loads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Make main() use LetterEncoder instead of the real model; return the keyword arguments of each load."""
    loads: list[dict[str, Any]] = []

    def fake_load_model(**kwargs: Any) -> LetterEncoder:
        loads.append(kwargs)
        return LetterEncoder()

    monkeypatch.setattr(semantic_search, "load_model", fake_load_model)
    return loads


def result_lines(results: list[SearchResult]) -> str:
    """The table rows main() should print for ``results``: rank, ID padded to the longest ID, score, text."""
    width = max(len(result.id) for result in results)
    return "".join(f"  {r.rank:<4}  {r.id:<{width}}  {r.score:6.3f}  {r.text}\n" for r in results)


def test_search_command_prints_the_top_three_matches(fake_model_loads, capsys):
    query = "Keep the libraries of two Python projects apart"
    assert main(["search", query]) == 0
    out = capsys.readouterr().out
    expected = SemanticSearcher(LetterEncoder(), load_documents()).search(query)
    assert f"Query: {query}\n" in out
    assert f"Semantic search ({MODEL_NAME}, cosine similarity): top 3 of 20 documents" in out
    assert result_lines(expected) in out
    assert "\n  4 " not in out
    assert fake_model_loads == [{"local_files_only": False}]


def test_search_command_top_k(fake_model_loads, capsys):
    assert main(["search", "--top-k", "5", "databases"]) == 0
    out = capsys.readouterr().out
    assert "top 5 of 20 documents" in out
    assert "\n  5     " in out and "\n  6     " not in out


def test_keyword_search_command_does_not_load_the_model(fake_model_loads, capsys):
    assert main(["search", "--method", "keyword", "python virtual environment"]) == 0
    out = capsys.readouterr().out
    expected = KeywordSearcher(load_documents()).search("python virtual environment")
    assert "Keyword baseline (IDF-weighted word overlap): top 3 of 20 documents" in out
    assert result_lines(expected) in out
    assert "Loading" not in out
    assert fake_model_loads == []


def test_keyword_search_command_with_no_matching_word(fake_model_loads, capsys):
    assert main(["search", "--method", "keyword", "the and of"]) == 0
    assert "No document shares a keyword with the query." in capsys.readouterr().out


def test_evaluate_command_prints_both_methods_and_the_metric_definitions(fake_model_loads, capsys):
    assert main(["evaluate"]) == 0
    out = capsys.readouterr().out
    documents = load_documents()
    queries = load_queries(documents)
    semantic = evaluate(SemanticSearcher(LetterEncoder(), documents), queries)
    keyword = evaluate(KeywordSearcher(documents), queries)
    assert "Evaluated 23 labelled queries (27 relevant-document labels) against 20 documents, top 3." in out
    assert f"Semantic search ({MODEL_NAME}, cosine similarity)" in out
    assert "Keyword baseline (IDF-weighted word overlap)" in out
    for query in queries:
        assert f"\n       {query.text}\n" in out
    for name, report in (("semantic", semantic), ("keyword", keyword)):
        assert f"  Top-3 hit rate: {report.hits} of 23 queries = {report.hit_rate:.1%}\n" in out
        assert f"  Recall@3:       {report.mean_recall:.3f}\n" in out
        assert f"  Precision@3:    {report.mean_precision:.3f} (best possible on these labels: 0.391)\n" in out
        assert f"  MRR@3:          {report.mrr:.3f}\n" in out
        summary = f"{report.hit_rate:.1%} ({report.hits}/23)"
        assert (
            f"  {name:<9} {summary:<18} {report.mean_recall:<9.3f} {report.mean_precision:<12.3f} {report.mrr:.3f}\n"
            in out
        )
    assert "Top-3 hit rate: the share of queries with at least one relevant document in the top 3." in out
    assert "Recall@3: for each query, (relevant documents in the top 3) / (all of its relevant documents)," in out
    assert "Precision@3: for each query, (relevant documents in the top 3) / 3, averaged over the queries." in out
    assert "MRR@3: for each query, 1 / (rank of the first relevant document in the top 3), or 0 if there is none," in out
    assert fake_model_loads == [{"local_files_only": False}]


def test_evaluate_command_top_k_renames_the_metrics(fake_model_loads, capsys):
    assert main(["evaluate", "--top-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "Top-5 hit rate" in out and "Recall@5" in out and "Precision@5" in out and "MRR@5" in out
    assert "Top-3" not in out


@pytest.mark.parametrize("argv", [["search", "--offline", "a query"], ["evaluate", "--offline"]])
def test_offline_never_allows_a_download(fake_model_loads, argv):
    assert main(argv) == 0
    assert fake_model_loads == [{"local_files_only": True}]


@pytest.mark.parametrize("query", ["", "   "])
def test_search_command_rejects_an_empty_query_before_loading_the_model(fake_model_loads, capsys, query):
    assert main(["search", query]) == 2
    assert "Error: The query is empty." in capsys.readouterr().err
    assert fake_model_loads == []


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "three"])
@pytest.mark.parametrize("command", [["search", "a query"], ["evaluate"]])
def test_cli_rejects_an_invalid_top_k(command, value):
    with pytest.raises(SystemExit) as caught:
        main([*command, "--top-k", value])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("argv", "hint"),
    [(["evaluate"], "needs internet access"), (["search", "--offline", "a query"], "without --offline")],
)
def test_cli_reports_a_model_that_cannot_be_loaded(monkeypatch, capsys, argv, hint):
    def fail(**kwargs: Any) -> None:
        raise OSError("couldn't find the model in the cached files")

    monkeypatch.setattr(semantic_search, "load_model", fail)
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert f"Error: could not load {MODEL_NAME}" in captured.err
    assert hint in captured.err
    assert "hit rate" not in captured.out


def test_cli_reports_a_missing_sentence_transformers_package(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import inside load_model fail
    assert main(["search", "a query"]) == 1
    assert "pip install -r requirements.txt" in capsys.readouterr().err


def test_cli_reports_a_broken_dataset(monkeypatch, tmp_path, fake_model_loads, capsys):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{"id": "doc-a", "text": "one"}, {"id": "doc-a", "text": "two"}]), encoding="utf-8")
    monkeypatch.setattr(semantic_search, "DOCUMENTS_PATH", path)
    assert main(["search", "a query"]) == 1
    assert "Duplicate document ID(s): doc-a." in capsys.readouterr().err
    assert fake_model_loads == []


# --- Real model (all-MiniLM-L6-v2 from the local cache only) -----------------------------------


def _refuse_network(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Tests must not open network connections.")


@pytest.fixture(scope="module")
def real_searcher():
    """A SemanticSearcher over the real documents, with the model loaded from the cache and the network blocked.

    Module-scoped so the model loads once; it therefore blocks sockets itself
    instead of relying on the function-scoped ``block_network`` fixture.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _refuse_network)
        patch.setattr(socket, "create_connection", _refuse_network)
        try:
            model = load_model(local_files_only=True)
        except (ImportError, OSError) as exc:  # OSError also covers PyTorch DLLs blocked on Windows
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            pytest.skip(f"{MODEL_NAME} could not be loaded from the local cache ({first_line}); run `python -m src.embeddings` once")
        return SemanticSearcher(model, load_documents())


@pytest.mark.model
def test_real_model_embeds_each_document_as_384_numbers(real_searcher):
    assert real_searcher.document_embeddings.shape == (20, MODEL_DIMENSIONS)
    assert np.isfinite(real_searcher.document_embeddings).all()


@pytest.mark.model
def test_real_model_search_returns_three_documents_best_first(real_searcher):
    results = real_searcher.search("How do I keep two Python projects from needing different library versions?")
    assert [result.rank for result in results] == [1, 2, 3]
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)
    assert len({result.id for result in results}) == 3


@pytest.mark.model
def test_real_model_evaluation_covers_every_labelled_query(real_searcher):
    # Checks the pipeline; the regression test below pins the measured numbers.
    report = evaluate(real_searcher, load_queries(real_searcher.documents))
    assert len(report.results) == 23
    assert all(len(result.retrieved_ids) == 3 for result in report.results)
    assert 0.0 <= report.mean_recall <= report.hit_rate <= 1.0


@pytest.mark.model
def test_real_model_still_gives_the_results_reported_in_the_readme(real_searcher):
    # A regression test, not a benchmark: a change to the model, the documents or the ranking code that loses a
    # hit or moves a relevant document down fails here. Fix the cause, or re-measure and update the README;
    # never edit the relevance labels to make this pass.
    report = evaluate(real_searcher, load_queries(real_searcher.documents))
    assert [result.query.id for result in report.results if not result.hit] == []
    # q10 is the only query whose first relevant document is not ranked first. Its db-03 is third, about 0.02
    # above the fourth document, so it is the hit most likely to be lost.
    not_first = {
        result.query.id: round(1 / result.reciprocal_rank) for result in report.results if result.reciprocal_rank < 1
    }
    assert not_first == {"q10": 3}
    assert report.mean_recall == 1.0
    assert report.mean_precision == pytest.approx(27 / 69)  # every one of the 27 labels is in its query's top 3
    assert report.mean_precision == pytest.approx(report.max_mean_precision)
    assert report.mrr == pytest.approx((22 + 1 / 3) / 23)
