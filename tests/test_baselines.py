"""Tests for :mod:`dnadecoder.baselines` (stdlib only, no metrics import)."""
from __future__ import annotations

import unittest

from dnadecoder.baselines import (
    bma_reconstruct,
    consensus,
    majority_vote,
)


class TestMajorityVote(unittest.TestCase):
    def test_identical_traces_recover_original(self) -> None:
        original = "ACGTACGTAC"
        traces = [original] * 5
        self.assertEqual(majority_vote(traces), original)

    def test_fixes_single_substitution_with_odd_clean_copies(self) -> None:
        original = "ACGTACGT"
        # Two clean copies + one with a single substitution at position 3.
        noisy = "ACGAACGT"  # T -> A at index 3
        traces = [original, original, noisy]
        self.assertEqual(majority_vote(traces), original)

    def test_empty_traces(self) -> None:
        self.assertEqual(majority_vote([]), "")

    def test_deterministic_tie_break_alphabetical(self) -> None:
        # Two traces, position 0 differs: A vs C -> tie -> alphabetical 'A'.
        traces = ["AG", "CG"]
        self.assertEqual(majority_vote(traces), "AG")


class TestBMAReconstruct(unittest.TestCase):
    def test_identical_clean_traces_recover_original(self) -> None:
        original = "ACGTACGTAC"
        traces = [original] * 4
        self.assertEqual(bma_reconstruct(traces), original)

    def test_empty_traces(self) -> None:
        self.assertEqual(bma_reconstruct([]), "")

    def test_improves_over_single_noisy_trace_on_indel_example(self) -> None:
        # Handcrafted: an early deletion in one trace shifts everything after it.
        original = "ACGTACGT"
        # trace0/trace1: clean. trace2: deletion of the first 'A'.
        t_del = "CGTACGT"
        traces = [original, original, t_del]

        recon = bma_reconstruct(traces, length=len(original))

        def edit_distance(a: str, b: str) -> int:
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
                prev = cur
            return prev[-1]

        # BMA should beat the lone noisy (deleted) trace against the original.
        self.assertLess(edit_distance(recon, original), edit_distance(t_del, original))
        # And here it should recover the original exactly.
        self.assertEqual(recon, original)

    def test_respects_explicit_length(self) -> None:
        original = "ACGTACGT"
        traces = [original] * 3
        self.assertEqual(bma_reconstruct(traces, length=4), "ACGT")

    def test_deterministic(self) -> None:
        traces = ["ACGTAC", "ACGTAC", "ACGTAC"]
        first = bma_reconstruct(traces)
        for _ in range(5):
            self.assertEqual(bma_reconstruct(traces), first)


class TestConsensusDispatch(unittest.TestCase):
    def test_bma_default(self) -> None:
        traces = ["ACGT", "ACGT", "ACGT"]
        self.assertEqual(consensus(traces), bma_reconstruct(traces))

    def test_majority_dispatch(self) -> None:
        traces = ["ACGT", "ACGT", "ACGT"]
        self.assertEqual(consensus(traces, method="majority"), majority_vote(traces))

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            consensus(["ACGT"], method="nope")


if __name__ == "__main__":
    unittest.main()
