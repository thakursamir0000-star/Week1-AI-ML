"""Wednesday: embeddings and cosine similarity.

The maths is tested directly with hand-computed values. Embedding, loading and
the CLI use small fake encoders, so none of those tests imports PyTorch or
touches a model. Tests marked ``model`` use the real all-MiniLM-L6-v2 from the
local Hugging Face cache only (network blocked); they are skipped when it is
not cached, and never download it.

Run everything except the real-model tests with:  python -m pytest -m "not model"
"""

from __future__ import annotations

import math
import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import embeddings
from src.embeddings import (
    EXAMPLE_TEXTS,
    MODEL_DIMENSIONS,
    MODEL_NAME,
    SimilarPair,
    compare_texts,
    cosine_similarity,
    cosine_similarity_matrix,
    embed_texts,
    format_report,
    format_similarity_matrix,
    load_model,
    main,
    rank_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    """Stands in for SentenceTransformer: looks up a fixed vector for each text and records every call."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def encode(self, sentences, /, **kwargs):
        self.calls.append((sentences, kwargs))
        return np.array([self.vectors[text] for text in sentences], dtype=np.float64)


class SizedEncoder:
    """Returns a reproducible random vector of ``dimensions`` numbers for each text."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def encode(self, sentences, /, **kwargs):
        rng = np.random.default_rng(len(sentences))
        return rng.normal(size=(len(sentences), self.dimensions)).astype(np.float32)


class CannedEncoder:
    """Returns the same, possibly broken, output whatever it is asked to encode."""

    def __init__(self, output: Any) -> None:
        self.output = output

    def encode(self, sentences, /, **kwargs):
        return self.output


# Unit vectors chosen so every cosine similarity is easy to work out by hand:
# T1.T2 = 0.8, T3.T4 = 0.96, T1.T5 = 0.6, T2.T5 = 0.48, T2.T4 = 0.168,
# T1.T3 = T1.T4 = T2.T3 = 0, T4.T5 = -0.768, T3.T5 = -0.8.
EXAMPLE_VECTORS = dict(
    zip(
        EXAMPLE_TEXTS,
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.28, 0.96],
            [0.6, 0.0, -0.8],
        ],
    )
)
EXPECTED_PAIR_ORDER = [(2, 3), (0, 1), (0, 4), (1, 4), (1, 3), (0, 2), (0, 3), (1, 2), (3, 4), (2, 4)]

# Not whole numbers of at least 1. True and False are rejected even though bool is a subclass of int.
INVALID_TOP_K = [0, -1, 2.5, True, False]


def as_index_pairs(pairs: list[SimilarPair] | tuple[SimilarPair, ...]) -> list[tuple[int, int]]:
    return [(pair.first, pair.second) for pair in pairs]


# --- Example texts ------------------------------------------------------------


def test_there_are_five_distinct_non_empty_example_texts():
    assert len(EXAMPLE_TEXTS) == 5
    assert len(set(EXAMPLE_TEXTS)) == 5
    assert all(text.strip() for text in EXAMPLE_TEXTS)


# --- cosine_similarity ----------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),  # identical
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1.0),  # same direction, different length
        ([1.0, 0.0], [0.0, 1.0], 0.0),  # orthogonal
        ([1.0, 2.0], [-1.0, -2.0], -1.0),  # opposite
        ([1.0, 0.0], [1.0, 1.0], 1 / math.sqrt(2)),  # 45 degrees
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32 / math.sqrt(14 * 77)),  # (a.b) / (|a||b|) by hand
        ([3], [-7], -1.0),  # one dimension, integers
    ],
)
def test_cosine_similarity_known_values(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-12)


def test_cosine_similarity_returns_a_python_float():
    assert type(cosine_similarity(np.array([1, 2]), np.array([3, 4]))) is float


def test_cosine_similarity_is_symmetric_and_ignores_vector_length():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=384), rng.normal(size=384)
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-15)
    assert cosine_similarity(5.0 * a, 0.01 * b) == pytest.approx(cosine_similarity(a, b), abs=1e-12)


def test_cosine_similarity_never_leaves_the_range_minus_one_to_one():
    # Without clipping, about 15% of these self-similarities round to 1.0000000000000002.
    rng = np.random.default_rng(0)
    for _ in range(500):
        vector = rng.normal(size=384)
        assert cosine_similarity(vector, vector) <= 1.0
        assert cosine_similarity(vector, -vector) >= -1.0


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1e200, 1e200], [1e200, 1e200], 1.0),  # squaring would overflow to infinity
        ([1e-200, 1e-200], [1e-200, 1e-200], 1.0),  # squaring would underflow to zero
        ([1e200, -1e200], [-1e200, 1e200], -1.0),  # opposite, large
        ([1e-200, -1e-200], [-1e-200, 1e-200], -1.0),  # opposite, small
        ([1e200, 0.0], [0.0, 1e-200], 0.0),  # orthogonal, large and small
        ([1e200, 1e200], [1e-200, 1e-200], 1.0),  # same direction, 400 orders of magnitude apart
        ([1e200, 1e-200], [1.0, 0.0], 1.0),  # a tiny component next to a huge one
        ([5e-324, 0.0], [1.0, 0.0], 1.0),  # the smallest positive float is not a zero vector
    ],
)
def test_cosine_similarity_handles_extreme_magnitudes(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0], "same length"),
        ([], [], "empty"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "1 dimension"),
        (1.0, 2.0, "1 dimension"),
        ([0.0, 0.0], [1.0, 1.0], "zero vector"),
        ([1.0, 1.0], [0.0, 0.0], "zero vector"),
        ([1.0, float("nan")], [1.0, 1.0], "NaN or infinity"),
        ([1.0, 1.0], [float("inf"), 1.0], "NaN or infinity"),
        (["1", "2"], [1.0, 2.0], "only numbers"),
        ([1.0, None], [1.0, 2.0], "only numbers"),
        (None, [1.0], "only numbers"),
        ([1 + 2j, 1], [1.0, 2.0], "only numbers"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(a, b, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(a, b)


# --- cosine_similarity_matrix ---------------------------------------------------


def test_matrix_known_values():
    rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-2.0, 0.0]]
    half = 1 / math.sqrt(2)
    expected = [
        [1.0, 0.0, half, -1.0],
        [0.0, 1.0, half, 0.0],
        [half, half, 1.0, -half],
        [-1.0, 0.0, -half, 1.0],
    ]
    np.testing.assert_allclose(cosine_similarity_matrix(rows), expected, atol=1e-12)


def test_matrix_entries_match_the_pairwise_function():
    vectors = np.random.default_rng(3).normal(size=(5, MODEL_DIMENSIONS))
    matrix = cosine_similarity_matrix(vectors)
    assert matrix.shape == (5, 5)
    for i in range(5):
        for j in range(5):
            assert matrix[i, j] == pytest.approx(cosine_similarity(vectors[i], vectors[j]), abs=1e-12)


def test_matrix_is_exactly_symmetric_with_ones_on_the_diagonal():
    matrix = cosine_similarity_matrix(np.random.default_rng(4).normal(size=(6, 50)))
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.ones(6))
    assert matrix.min() >= -1.0 and matrix.max() <= 1.0


def test_matrix_ignores_row_lengths():
    vectors = np.random.default_rng(5).normal(size=(4, 10))
    scaled = vectors * np.array([[0.001], [2.0], [50.0], [7.5]])
    np.testing.assert_allclose(cosine_similarity_matrix(scaled), cosine_similarity_matrix(vectors), atol=1e-12)


def test_matrix_of_a_single_row_is_one_by_one():
    assert cosine_similarity_matrix([[0.3, -0.4]]).tolist() == [[1.0]]


def test_matrix_handles_extreme_magnitudes():
    rows = [
        [1e200, 1e200],
        [1e-200, 1e-200],  # same direction as row 0
        [1e200, -1e200],  # orthogonal to rows 0 and 1
        [-1e-200, -1e-200],  # opposite to rows 0 and 1
    ]
    expected = [
        [1.0, 1.0, 0.0, -1.0],
        [1.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0, 1.0],
    ]
    matrix = cosine_similarity_matrix(rows)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    for i in range(4):
        for j in range(4):
            assert matrix[i, j] == pytest.approx(cosine_similarity(rows[i], rows[j]), abs=1e-12)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([1.0, 2.0], "2 dimension"),
        ([[[1.0]]], "2 dimension"),
        (np.empty((0, 3)), "empty"),
        (np.empty((3, 0)), "empty"),
        ([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]], r"Row\(s\) 1, 3 are zero vectors"),
        ([[1.0, float("nan")], [1.0, 2.0]], "NaN or infinity"),
        ([["a", "b"], ["c", "d"]], "only numbers"),
        ([[1.0, 2.0], [3.0]], "rectangular"),
    ],
)
def test_matrix_rejects_invalid_embeddings(rows, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity_matrix(rows)


# --- rank_pairs -------------------------------------------------------------------


def test_rank_pairs_orders_from_most_to_least_similar():
    matrix = [
        [1.0, 0.2, 0.9],
        [0.2, 1.0, -0.5],
        [0.9, -0.5, 1.0],
    ]
    assert rank_pairs(matrix) == [SimilarPair(0, 2, 0.9), SimilarPair(0, 1, 0.2), SimilarPair(1, 2, -0.5)]


def test_rank_pairs_lists_every_pair_once():
    matrix = cosine_similarity_matrix(np.random.default_rng(6).normal(size=(5, 8)))
    pairs = as_index_pairs(rank_pairs(matrix))
    assert len(pairs) == 10  # 5 choose 2
    assert sorted(pairs) == [(i, j) for i in range(5) for j in range(i + 1, 5)]
    scores = [pair.score for pair in rank_pairs(matrix)]
    assert scores == sorted(scores, reverse=True)


def test_rank_pairs_breaks_ties_by_index():
    matrix = np.full((4, 4), 0.5)
    np.fill_diagonal(matrix, 1.0)
    assert as_index_pairs(rank_pairs(matrix)) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@pytest.mark.parametrize(("top_k", "expected_count"), [(1, 1), (3, 3), (10, 10), (100, 10)])
def test_rank_pairs_top_k(top_k, expected_count):
    matrix = cosine_similarity_matrix(np.random.default_rng(8).normal(size=(5, 8)))
    ranked = rank_pairs(matrix, top_k=top_k)
    assert len(ranked) == expected_count
    assert ranked == rank_pairs(matrix)[:expected_count]


@pytest.mark.parametrize(
    ("matrix", "top_k", "message"),
    [
        ([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1]], None, "square"),
        ([[1.0]], None, "At least two texts"),
        ([[1.0, 0.5], [0.4, 1.0]], None, "symmetric"),
        ([[14.0, 32.0], [32.0, 77.0]], None, "between -1 and 1"),  # raw dot products, not cosines
        ([[1.0, float("nan")], [float("nan"), 1.0]], None, "NaN or infinity"),
        ([[1.0, 0.5], [0.5, 1.0]], 0, "top_k"),
        ([[1.0, 0.5], [0.5, 1.0]], -2, "top_k"),
    ],
)
def test_rank_pairs_rejects_invalid_input(matrix, top_k, message):
    with pytest.raises(ValueError, match=message):
        rank_pairs(matrix, top_k=top_k)


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_rank_pairs_rejects_a_top_k_that_is_not_a_positive_whole_number(top_k):
    with pytest.raises(ValueError, match="top_k must be a whole number of at least 1"):
        rank_pairs([[1.0, 0.5], [0.5, 1.0]], top_k=top_k)


# --- embed_texts (fake encoders, no model) ----------------------------------------


@pytest.mark.parametrize("dimensions", [1, 8, MODEL_DIMENSIONS, 768])
def test_embed_texts_returns_one_row_per_text(dimensions):
    result = embed_texts(SizedEncoder(dimensions), EXAMPLE_TEXTS)
    assert result.shape == (len(EXAMPLE_TEXTS), dimensions)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


def test_embed_texts_encodes_all_texts_in_one_quiet_batch():
    encoder = FakeEncoder(EXAMPLE_VECTORS)
    embed_texts(encoder, EXAMPLE_TEXTS)
    assert encoder.calls == [(list(EXAMPLE_TEXTS), {"show_progress_bar": False})]


@pytest.mark.parametrize(
    ("texts", "message"),
    [
        ("I was charged twice.", "not a single string"),
        ([], "At least one text"),
        (["fine", ""], "Text 1 is empty"),
        (["fine", "  \n\t"], "Text 1 is empty"),
        (["fine", None], "Text 1 must be a string"),
        ([42], "Text 0 must be a string"),
        (None, "list of strings"),
        (7, "list of strings"),
    ],
)
def test_embed_texts_rejects_invalid_texts_without_calling_the_model(texts, message):
    encoder = FakeEncoder({})
    with pytest.raises(ValueError, match=message):
        embed_texts(encoder, texts)
    assert encoder.calls == []


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones(3), "2 dimension"),  # one flat vector instead of one row per text
        (np.ones((3, 4)), "Expected 2 embeddings"),
        (np.empty((2, 0)), "empty"),
        (np.array([[1.0, np.nan], [1.0, 1.0]]), "NaN or infinity"),
        (None, "only numbers"),
    ],
)
def test_embed_texts_rejects_unusable_model_output(output, message):
    with pytest.raises(ValueError, match=message):
        embed_texts(CannedEncoder(output), ["first text", "second text"])


# --- compare_texts ------------------------------------------------------------------


def test_compare_texts_ranks_pairs_from_the_embeddings():
    report = compare_texts(FakeEncoder(EXAMPLE_VECTORS), EXAMPLE_TEXTS)
    assert report.texts == EXAMPLE_TEXTS
    assert report.dimensions == 3
    assert report.similarities.shape == (5, 5)
    assert as_index_pairs(report.pairs) == EXPECTED_PAIR_ORDER
    expected_scores = [0.96, 0.8, 0.6, 0.48, 0.168, 0.0, 0.0, 0.0, -0.768, -0.8]
    assert [pair.score for pair in report.pairs] == pytest.approx(expected_scores, abs=1e-12)


@pytest.mark.parametrize("texts", [["only one text"], []])
def test_compare_texts_needs_at_least_two_texts(texts):
    with pytest.raises(ValueError, match="At least"):
        compare_texts(FakeEncoder({}), texts)


# --- Output ---------------------------------------------------------------------


def test_format_similarity_matrix_layout():
    table = format_similarity_matrix(np.array([[1.0, -0.25], [-0.25, 1.0]]), ["T1", "T2"])
    assert table.splitlines() == [
        "        T1      T2",
        "T1   1.000  -0.250",
        "T2  -0.250   1.000",
    ]


def test_format_similarity_matrix_rejects_wrong_label_count():
    with pytest.raises(ValueError, match="labels"):
        format_similarity_matrix(np.eye(3), ["T1", "T2"])


@pytest.mark.parametrize("top_k", INVALID_TOP_K)
def test_format_report_rejects_a_top_k_that_is_not_a_positive_whole_number(top_k):
    report = compare_texts(FakeEncoder(EXAMPLE_VECTORS), EXAMPLE_TEXTS)
    with pytest.raises(ValueError, match="top_k must be a whole number of at least 1"):
        format_report(report, top_k=top_k)


# --- load_model (stand-in sentence_transformers package, no PyTorch) ----------------------


def install_fake_library(monkeypatch: pytest.MonkeyPatch, *, cached: bool) -> list[dict[str, Any]]:
    """Replace the sentence_transformers package with a stand-in; return the list of model loads it saw."""
    loads: list[dict[str, Any]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            loads.append({"model_name": model_name, **kwargs})
            if kwargs.get("local_files_only") and not cached:
                raise OSError(f"{model_name} is not in the local cache")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return loads


def test_load_model_uses_the_local_cache_first(monkeypatch, capsys):
    loads = install_fake_library(monkeypatch, cached=True)
    load_model()
    assert loads == [{"model_name": MODEL_NAME, "local_files_only": True}]
    assert capsys.readouterr().err == ""


def test_load_model_downloads_only_when_the_model_is_not_cached(monkeypatch, capsys):
    loads = install_fake_library(monkeypatch, cached=False)
    load_model()
    assert loads == [{"model_name": MODEL_NAME, "local_files_only": True}, {"model_name": MODEL_NAME}]
    assert "downloading" in capsys.readouterr().err


def test_load_model_never_downloads_with_local_files_only(monkeypatch):
    loads = install_fake_library(monkeypatch, cached=False)
    with pytest.raises(OSError, match="not in the local cache"):
        load_model(local_files_only=True)
    assert loads == [{"model_name": MODEL_NAME, "local_files_only": True}]


def test_importing_the_module_does_not_load_pytorch():
    # A fresh interpreter: other tests in this process may already have imported the model libraries.
    code = (
        "import sys; import src.embeddings as e; "
        "assert e.cosine_similarity([1, 0], [1, 1]) > 0.7; "
        "print(sorted(m for m in ('torch', 'transformers', 'sentence_transformers') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# --- Command line -------------------------------------------------------------------


@pytest.fixture
def fake_model_loads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Make main() use FakeEncoder instead of the real model; return the keyword arguments of each load."""
    loads: list[dict[str, Any]] = []

    def fake_load_model(**kwargs: Any) -> FakeEncoder:
        loads.append(kwargs)
        return FakeEncoder(EXAMPLE_VECTORS)

    monkeypatch.setattr(embeddings, "load_model", fake_load_model)
    return loads


def test_main_prints_the_texts_the_matrix_and_the_most_similar_pairs(fake_model_loads, capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Embedded 5 texts; each embedding has 3 dimensions." in out
    for label, text in zip(["T1", "T2", "T3", "T4", "T5"], EXAMPLE_TEXTS):
        assert f"  {label}  {text}" in out
    assert "          T1      T2      T3      T4      T5" in out
    assert "  T3   0.000   0.000   1.000   0.960  -0.800" in out
    assert "Most similar pairs:\n  1. T3 & T4  0.960\n  2. T1 & T2  0.800\n  3. T1 & T5  0.600\n" in out
    assert "\n  4. " not in out
    assert "Least similar pair: T3 & T5  -0.800" in out
    assert fake_model_loads == [{"local_files_only": False}]


def test_main_top_k_lists_more_pairs(fake_model_loads, capsys):
    assert main(["--top-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "  5. T2 & T4  0.168" in out
    assert "\n  6. " not in out


def test_main_offline_never_allows_a_download(fake_model_loads):
    assert main(["--offline"]) == 0
    assert fake_model_loads == [{"local_files_only": True}]


@pytest.mark.parametrize(("argv", "hint"), [([], "needs internet access"), (["--offline"], "without --offline")])
def test_main_reports_a_model_that_cannot_be_loaded(monkeypatch, capsys, argv, hint):
    def fail(**kwargs: Any) -> None:
        raise OSError("couldn't find the model in the cached files")

    monkeypatch.setattr(embeddings, "load_model", fail)
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert f"Error: could not load {MODEL_NAME}" in captured.err
    assert hint in captured.err
    assert "Most similar" not in captured.out


def test_main_reports_a_missing_sentence_transformers_package(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import inside load_model fail
    assert main([]) == 1
    assert "pip install -r requirements.txt" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "three"])
def test_main_rejects_an_invalid_top_k(value, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--top-k", value])
    assert caught.value.code == 2


# --- Real model (all-MiniLM-L6-v2 from the local cache only) ------------------------


def _refuse_network(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Tests must not open network connections.")


@pytest.fixture(scope="module")
def real_report():
    """Compare the five example texts with the real model, loaded from the cache with the network blocked.

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
        return compare_texts(model, EXAMPLE_TEXTS)


@pytest.mark.model
def test_real_model_embeds_each_text_as_384_numbers(real_report):
    assert real_report.embeddings.shape == (len(EXAMPLE_TEXTS), MODEL_DIMENSIONS)
    assert np.isfinite(real_report.embeddings).all()


@pytest.mark.model
def test_real_model_ranks_the_two_paraphrase_pairs_highest(real_report):
    assert set(as_index_pairs(real_report.pairs[:2])) == {(0, 1), (2, 3)}


@pytest.mark.model
def test_real_model_finds_each_paraphrase_partner_as_the_nearest_text(real_report):
    for text, partner in [(0, 1), (1, 0), (2, 3), (3, 2)]:
        others = real_report.similarities[text].copy()
        others[text] = -np.inf  # a text is always most similar to itself
        assert int(np.argmax(others)) == partner
