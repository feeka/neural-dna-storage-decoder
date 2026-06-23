from __future__ import annotations

"""String reconstruction metrics for DNA-storage decoding.

Pure-stdlib edit-distance utilities used to score predicted strands against
ground-truth originals. No numpy / torch dependency.
"""

from typing import Dict, List


def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Classic dynamic-programming algorithm using two rolling rows, giving
    O(len(a) * len(b)) time and O(min(len)) extra space. Insertions,
    deletions, and substitutions each cost 1.

    Parameters
    ----------
    a, b : str
        Input strings.

    Returns
    -------
    int
        Minimum number of single-character edits to turn ``a`` into ``b``.
    """
    # Make ``b`` the shorter string so the rolling rows use minimal memory.
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    # previous[j] = edit distance between a[:i] and b[:j]
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost,  # substitution / match
            )
        previous = current
    return previous[len(b)]


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer string length.

    Returns ``levenshtein(a, b) / max(len(a), len(b), 1)``, a value in [0, 1].
    Two empty strings yield 0.0.
    """
    return levenshtein(a, b) / max(len(a), len(b), 1)


def symbol_error_rate(pred: str, target: str) -> float:
    """Symbol error rate of ``pred`` relative to ``target``.

    Defined as ``levenshtein(pred, target) / max(len(target), 1)``. Note this
    is normalized by the *target* length, so it can exceed 1.0 when the
    prediction is much longer than the target.
    """
    return levenshtein(pred, target) / max(len(target), 1)


def exact_match(pred: str, target: str) -> bool:
    """Return True iff ``pred`` and ``target`` are identical strings."""
    return pred == target


def aggregate(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """Aggregate reconstruction metrics over paired predictions / targets.

    Parameters
    ----------
    preds, targets : list[str]
        Equal-length lists of predicted and ground-truth strings.

    Returns
    -------
    dict
        Keys ``"mean_edit_distance"``, ``"mean_ser"``, ``"exact_match_rate"``,
        and ``"n"`` (the number of pairs). For an empty input all means and
        rates are 0.0 and ``n`` is 0.

    Raises
    ------
    ValueError
        If ``preds`` and ``targets`` differ in length.
    """
    if len(preds) != len(targets):
        raise ValueError(
            f"preds and targets must have equal length, got {len(preds)} and {len(targets)}"
        )
    n = len(preds)
    if n == 0:
        return {"mean_edit_distance": 0.0, "mean_ser": 0.0, "exact_match_rate": 0.0, "n": 0}

    total_edit = 0
    total_ser = 0.0
    total_exact = 0
    for pred, target in zip(preds, targets):
        total_edit += levenshtein(pred, target)
        total_ser += symbol_error_rate(pred, target)
        total_exact += 1 if exact_match(pred, target) else 0

    return {
        "mean_edit_distance": total_edit / n,
        "mean_ser": total_ser / n,
        "exact_match_rate": total_exact / n,
        "n": n,
    }
