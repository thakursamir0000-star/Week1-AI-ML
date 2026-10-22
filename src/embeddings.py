"""Wednesday task: sentence embeddings and cosine similarity.

The module has two parts, kept apart on purpose:

- **Pure maths** (``cosine_similarity``, ``cosine_similarity_matrix``,
  ``rank_pairs``): NumPy only, with no model, no PyTorch and no network, so
  every rule is unit-tested directly.
- **Model execution** (``load_model``, ``embed_texts``, ``compare_texts``):
  runs a Sentence Transformers model locally on the CPU. ``sentence_transformers``
  is imported only inside ``load_model()``, so importing this module stays fast
  and never loads PyTorch.

Run the five-text demo from the project root:

    python -m src.embeddings             # load the model (download it on the first run) and compare the texts
    python -m src.embeddings --offline   # use the local cache only; fail instead of downloading
    python -m src.embeddings --top-k 5   # list more of the most similar pairs

Exit codes: 0 success, 1 the model could not be loaded, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSIONS = 384  # length of each all-MiniLM-L6-v2 embedding
DEFAULT_TOP_K = 3

EXIT_OK = 0
EXIT_FAILURE = 1

# Two paraphrase pairs, plus one sentence that shares words with the first text
# but not its meaning. Texts from different pairs are about unrelated problems.
EXAMPLE_TEXTS: tuple[str, ...] = (
    # T1 and T2: the same billing problem, worded differently.
    "I was charged twice for my subscription this month.",
    "My card was billed two times for the same monthly plan.",
    # T3 and T4: the same crash, worded differently.
    "The app crashes whenever I export a report as a PDF.",
    "Exporting a PDF makes the application close unexpectedly.",
    # T5: shares "charge" and "twice" with T1, but is about a phone battery.
    "I charge my phone twice a day because the battery drains fast.",
)


class Encoder(Protocol):
    """Anything with a SentenceTransformer-style ``encode`` method (the real model or a test fake)."""

    def encode(self, sentences: list[str], /, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SimilarPair:
    """Two different texts, by 0-based index (``first < second``), and their cosine similarity."""

    first: int
    second: int
    score: float


@dataclass(frozen=True, eq=False)
class SimilarityReport:
    """Everything ``compare_texts()`` computed: one embedding row per text, the matrix and the ranked pairs."""

    texts: tuple[str, ...]
    embeddings: np.ndarray
    similarities: np.ndarray
    pairs: tuple[SimilarPair, ...]  # every pair, most similar first

    @property
    def dimensions(self) -> int:
        return int(self.embeddings.shape[1])


# --- Pure maths (NumPy only) --------------------------------------------------


def _as_float_array(values: Any, name: str, ndim: int) -> np.ndarray:
    """Return ``values`` as a float64 array with ``ndim`` dimensions, or raise ValueError."""
    try:
        array = np.asarray(values)
    except ValueError as exc:  # e.g. ragged nested lists
        raise ValueError(f"{name} must be a rectangular array of numbers ({exc}).") from exc
    # Accept booleans, integers and floats only: strings such as "1.5" would
    # otherwise be converted silently, and objects such as None not at all.
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain only numbers; got values of type {array.dtype}.")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s); got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} is empty; got shape {array.shape}.")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row of a 2-D array to length 1. No row may be all zeros.

    Each row is first divided by its largest absolute value, so squaring the
    components for the norm can neither overflow (values near 1e200) nor
    underflow to zero (values near 1e-200).
    """
    scaled = vectors / np.abs(vectors).max(axis=1, keepdims=True)
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def _check_top_k(top_k: Any) -> None:
    """Raise ValueError unless ``top_k`` is a whole number of at least 1."""
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a whole number of at least 1; got {top_k!r}.")


def cosine_similarity(a: Any, b: Any) -> float:
    """Return cos(a, b) = (a . b) / (|a| |b|), a number from -1 to 1.

    1 means the vectors point the same way, 0 that they are orthogonal, and -1
    that they point in opposite directions. The length of a vector does not
    matter, only its direction.

    Raises:
        ValueError: either vector is not 1-D, is empty, contains a non-number,
            NaN or infinity, or is all zeros; or the lengths differ.
    """
    first = _as_float_array(a, "first vector", ndim=1)
    second = _as_float_array(b, "second vector", ndim=1)
    if first.shape != second.shape:
        raise ValueError(f"Vectors must have the same length; got {first.size} and {second.size}.")
    if not first.any() or not second.any():
        raise ValueError("Cosine similarity is undefined for a zero vector (it has no direction).")
    first_unit, second_unit = _unit_rows(np.stack([first, second]))
    # Rounding can push the result a hair past +-1; clip so the documented range holds.
    return float(np.clip(first_unit @ second_unit, -1.0, 1.0))


def cosine_similarity_matrix(embeddings: Any) -> np.ndarray:
    """Return the n x n matrix of cosine similarities between the n rows of ``embeddings``.

    Entry ``[i, j]`` equals ``cosine_similarity(embeddings[i], embeddings[j])``.
    The matrix is exactly symmetric, with 1.0 on the diagonal.

    Raises:
        ValueError: ``embeddings`` is not a non-empty 2-D array of finite
            numbers, or a row is all zeros.
    """
    matrix = _as_float_array(embeddings, "embeddings", ndim=2)
    zero_rows = np.flatnonzero(~matrix.any(axis=1))
    if zero_rows.size:
        rows = ", ".join(str(index) for index in zero_rows)
        raise ValueError(f"Row(s) {rows} are zero vectors; cosine similarity is undefined for them.")
    unit_rows = _unit_rows(matrix)
    similarities = unit_rows @ unit_rows.T
    # Matrix products can round [i, j] and [j, i] differently in the last bit.
    similarities = np.clip((similarities + similarities.T) / 2, -1.0, 1.0)
    np.fill_diagonal(similarities, 1.0)
    return similarities


def rank_pairs(similarities: Any, top_k: int | None = None) -> list[SimilarPair]:
    """Return every pair of different texts, most similar first (only the best ``top_k`` if given).

    Each pair appears once, as ``(i, j)`` with ``i < j``. Equal scores keep
    index order, so the ranking is deterministic.

    Raises:
        ValueError: the matrix is not square, symmetric and within [-1, 1];
            it has fewer than two rows; or ``top_k`` is not a whole number of
            at least 1.
    """
    matrix = _as_float_array(similarities, "similarity matrix", ndim=2)
    count = matrix.shape[0]
    if matrix.shape != (count, count):
        raise ValueError(f"The similarity matrix must be square; got shape {matrix.shape}.")
    if count < 2:
        raise ValueError("At least two texts are needed to form a pair.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("The similarity matrix must be symmetric.")
    # Catches raw dot products of unnormalised vectors passed in by mistake.
    if np.abs(matrix).max() > 1.0 + 1e-9:
        raise ValueError("Cosine similarities must be between -1 and 1.")
    if top_k is not None:
        _check_top_k(top_k)

    firsts, seconds = np.triu_indices(count, k=1)  # row-major order: (0, 1), (0, 2), ...
    scores = matrix[firsts, seconds]
    best_first = np.argsort(-scores, kind="stable")
    pairs = [SimilarPair(int(firsts[i]), int(seconds[i]), float(scores[i])) for i in best_first]
    return pairs if top_k is None else pairs[:top_k]


# --- Model execution ------------------------------------------------------------


def load_model(model_name: str = MODEL_NAME, *, local_files_only: bool = False) -> SentenceTransformer:
    """Load a Sentence Transformers model, preferring the local Hugging Face cache.

    The first run downloads the model (about 90 MB for all-MiniLM-L6-v2) into
    the Hugging Face cache, ``~/.cache/huggingface`` by default. Later runs load
    it from there and make no network requests.

    Raises:
        ImportError: sentence-transformers is not installed.
        OSError: the model is not cached and ``local_files_only`` is True, or
            the download failed.
    """
    # Imported here so the maths above, and its tests, never pay the
    # multi-second cost of importing PyTorch.
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        if local_files_only:
            raise
        print(f"{model_name} is not cached yet; downloading it from the Hugging Face Hub...", file=sys.stderr)
        return SentenceTransformer(model_name)


def _validate_texts(texts: Any) -> list[str]:
    """Return ``texts`` as a list of non-blank strings, or raise ValueError."""
    if isinstance(texts, str):
        # A string is itself a sequence, so it would otherwise be embedded one character at a time.
        raise ValueError("Pass a list of texts, not a single string.")
    try:
        items = list(texts)
    except TypeError:
        raise ValueError(f"Texts must be a list of strings; got {type(texts).__name__}.") from None
    if not items:
        raise ValueError("At least one text is needed.")
    for index, text in enumerate(items):
        if not isinstance(text, str):
            raise ValueError(f"Text {index} must be a string; got {type(text).__name__}.")
        if not text.strip():
            raise ValueError(f"Text {index} is empty.")
    return items


def embed_texts(model: Encoder, texts: Sequence[str]) -> np.ndarray:
    """Encode ``texts`` in one batch and return a float64 array of shape ``(len(texts), dimensions)``.

    Raises:
        ValueError: ``texts`` is a single string, empty, or contains a blank or
            non-string item (the model is not called); or the model returned
            something other than one finite row per text.
    """
    items = _validate_texts(texts)
    output = model.encode(items, show_progress_bar=False)
    embeddings = _as_float_array(output, "model output", ndim=2)
    if embeddings.shape[0] != len(items):
        raise ValueError(f"Expected {len(items)} embeddings from the model; got {embeddings.shape[0]}.")
    return embeddings


def compare_texts(model: Encoder, texts: Sequence[str]) -> SimilarityReport:
    """Embed ``texts`` and compare each one with every other one.

    Raises:
        ValueError: fewer than two texts, or any problem listed under
            ``embed_texts()``.
    """
    items = _validate_texts(texts)
    if len(items) < 2:
        raise ValueError("At least two texts are needed to compare them.")
    embeddings = embed_texts(model, items)
    similarities = cosine_similarity_matrix(embeddings)
    return SimilarityReport(tuple(items), embeddings, similarities, tuple(rank_pairs(similarities)))


# --- Output ---------------------------------------------------------------------


def text_labels(count: int) -> list[str]:
    """Short labels T1, T2, ... used in the matrix and the pair list."""
    return [f"T{number}" for number in range(1, count + 1)]


def format_similarity_matrix(similarities: np.ndarray, labels: Sequence[str]) -> str:
    """Format the matrix as a table with ``labels`` as row and column headings (3 decimals)."""
    if len(labels) != len(similarities):
        raise ValueError(f"Got {len(labels)} labels for a {len(similarities)} x {len(similarities)} matrix.")
    label_width = max(len(label) for label in labels)
    column_width = max(8, label_width + 2)
    lines = [" " * label_width + "".join(f"{label:>{column_width}}" for label in labels)]
    for label, row in zip(labels, similarities):
        lines.append(f"{label:<{label_width}}" + "".join(f"{value:>{column_width}.3f}" for value in row))
    return "\n".join(lines)


def format_report(report: SimilarityReport, top_k: int = DEFAULT_TOP_K) -> str:
    """Format the texts, the similarity matrix and the most similar pairs for the terminal.

    Raises:
        ValueError: ``top_k`` is not a whole number of at least 1.
    """
    _check_top_k(top_k)
    labels = text_labels(len(report.texts))
    lines = [f"Embedded {len(report.texts)} texts; each embedding has {report.dimensions} dimensions.", "", "Texts:"]
    lines += [f"  {label}  {text}" for label, text in zip(labels, report.texts)]
    lines += ["", "Cosine similarity (1 = same direction, 0 = orthogonal, -1 = opposite):"]
    lines += ["  " + line for line in format_similarity_matrix(report.similarities, labels).splitlines()]

    def describe(pair: SimilarPair) -> str:
        return f"{labels[pair.first]} & {labels[pair.second]}  {pair.score:.3f}"

    lines += ["", "Most similar pairs:"]
    lines += [f"  {rank}. {describe(pair)}" for rank, pair in enumerate(report.pairs[:top_k], start=1)]
    lines += ["", f"Least similar pair: {describe(report.pairs[-1])}"]
    return "\n".join(lines)


# --- Command line -----------------------------------------------------------------


def _positive_int(value: str) -> int:
    """argparse type for --top-k: a whole number of at least 1."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.embeddings",
        description=(
            "Day 3: embed five example texts with a local Sentence Transformers model "
            "and compare them with cosine similarity. No API key is needed."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help=f"number of most similar pairs to list (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load the model from the local cache only; fail instead of downloading it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Loading {MODEL_NAME} ...")
    try:
        model = load_model(local_files_only=args.offline)
    except ImportError as exc:
        print(f"Error: sentence-transformers is not installed ({exc}). Run: pip install -r requirements.txt", file=sys.stderr)
        return EXIT_FAILURE
    except OSError as exc:
        hint = (
            "It is not in the local cache; run once without --offline to download it."
            if args.offline
            else "The first run needs internet access to download it (about 90 MB)."
        )
        print(f"Error: could not load {MODEL_NAME}: {exc}\n{hint}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_report(compare_texts(model, EXAMPLE_TEXTS), top_k=args.top_k))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
