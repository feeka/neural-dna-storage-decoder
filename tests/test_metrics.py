from __future__ import annotations

"""Tests for dnadecoder.metrics (stdlib only, runnable as a script)."""

from dnadecoder.metrics import (
    aggregate,
    exact_match,
    levenshtein,
    normalized_edit_distance,
    symbol_error_rate,
)


def test_levenshtein_known() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("ACGT", "ACGT") == 0
    assert levenshtein("", "ACGT") == 4
    assert levenshtein("ACGT", "") == 4
    assert levenshtein("", "") == 0


def test_levenshtein_indels() -> None:
    # pure deletion
    assert levenshtein("ACGT", "AGT") == 1
    # pure insertion
    assert levenshtein("AGT", "ACGT") == 1
    # single substitution
    assert levenshtein("ACGT", "ACCT") == 1
    # symmetry
    assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw")


def test_normalized_edit_distance() -> None:
    assert normalized_edit_distance("ACGT", "ACGT") == 0.0
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("", "ACGT") == 1.0
    # kitten/sitting: 3 / 7
    assert abs(normalized_edit_distance("kitten", "sitting") - 3 / 7) < 1e-12
    # always within [0, 1]
    val = normalized_edit_distance("ACGTACGT", "TTTT")
    assert 0.0 <= val <= 1.0


def test_symbol_error_rate_bounds() -> None:
    assert symbol_error_rate("ACGT", "ACGT") == 0.0
    # normalized by target length (4)
    assert symbol_error_rate("ACCT", "ACGT") == 0.25
    # empty target -> divide by max(0,1)=1
    assert symbol_error_rate("AC", "") == 2.0
    assert symbol_error_rate("", "") == 0.0
    # can exceed 1 when pred much longer than target
    assert symbol_error_rate("ACGTACGT", "A") == 7.0


def test_exact_match() -> None:
    assert exact_match("ACGT", "ACGT") is True
    assert exact_match("ACGT", "ACGA") is False
    assert exact_match("", "") is True


def test_aggregate_handcrafted() -> None:
    preds = ["ACGT", "ACCT", "AAAA"]
    targets = ["ACGT", "ACGT", "ACGT"]
    # edit distances: 0, 1, 3 -> mean 4/3
    # sers (target len 4): 0, 0.25, 0.75 -> mean 1.0/3
    # exact: 1/3
    res = aggregate(preds, targets)
    assert res["n"] == 3
    assert abs(res["mean_edit_distance"] - 4 / 3) < 1e-12
    assert abs(res["mean_ser"] - (0.0 + 0.25 + 0.75) / 3) < 1e-12
    assert abs(res["exact_match_rate"] - 1 / 3) < 1e-12


def test_aggregate_empty() -> None:
    res = aggregate([], [])
    assert res == {"mean_edit_distance": 0.0, "mean_ser": 0.0, "exact_match_rate": 0.0, "n": 0}


def test_aggregate_length_mismatch() -> None:
    try:
        aggregate(["A"], ["A", "B"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
