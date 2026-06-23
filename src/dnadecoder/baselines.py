"""Classical trace-reconstruction baselines for DNA-storage decoding.

These string-only baselines reconstruct a single original strand from a set of
noisy reads ("traces") that have passed through an insertion/deletion/
substitution (IDS) channel. They serve as the classical reference points the
neural denoiser is benchmarked against.

Two algorithms are provided:

``majority_vote``
    Naive positional consensus. Robust to substitutions, fragile to indels
    because a single early insertion/deletion shifts all downstream positions.

``bma_reconstruct``
    Symbolwise Majority Alignment (the 4-ary generalisation of the *Bitwise*
    Majority Alignment of Batu, Kannan, Khanna and McGregor, "Reconstructing
    strings from random traces", SODA 2004). Per-trace pointers let traces fall
    out of and back into agreement, which absorbs indels far better than naive
    positional voting.

Only the Python standard library is used; this module deliberately does not
depend on :mod:`dnadecoder.metrics`.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .tokens import BASES


def _median_length(traces: List[str]) -> int:
    """Return ``round(median(len(t) for t in traces))`` as an int.

    Uses the standard "average the two middle values" definition of the median
    for an even count. Banker's rounding (Python's ``round``) is acceptable here
    because trace lengths are integers and exact ties are rare; the result is
    deterministic in all cases.
    """
    if not traces:
        return 0
    lengths = sorted(len(t) for t in traces)
    n = len(lengths)
    mid = n // 2
    if n % 2 == 1:
        med = float(lengths[mid])
    else:
        med = (lengths[mid - 1] + lengths[mid]) / 2.0
    return int(round(med))


def _majority_symbol(symbols: List[str]) -> str:
    """Most common symbol in ``symbols`` with an alphabetical tie-break.

    Ties are broken by the canonical :data:`dnadecoder.tokens.BASES` ordering
    (``"ACGT"``) so the output is fully deterministic.
    """
    counts = Counter(symbols)
    best_count = max(counts.values())
    # Among all symbols achieving the max count, pick the alphabetically first.
    candidates = [s for s, c in counts.items() if c == best_count]
    return min(candidates, key=lambda s: (BASES.index(s) if s in BASES else len(BASES), s))


def majority_vote(traces: List[str]) -> str:
    """Reconstruct a strand by naive positional majority vote.

    The target length ``L`` is the rounded median of the trace lengths. For each
    position ``i`` in ``[0, L)`` the most common base among the traces that have
    an ``i``-th character is emitted, with an alphabetical tie-break.

    Parameters
    ----------
    traces:
        Noisy reads of a single original strand (``ACGT`` strings).

    Returns
    -------
    str
        The reconstructed strand of length ``L`` (empty if ``traces`` is empty).

    Notes
    -----
    Strong when substitutions dominate; weak under indels because positional
    alignment drifts after the first insertion or deletion.
    """
    if not traces:
        return ""
    length = _median_length(traces)
    out: List[str] = []
    for i in range(length):
        column = [t[i] for t in traces if i < len(t)]
        if not column:
            break  # no trace is long enough to vote on this position
        out.append(_majority_symbol(column))
    return "".join(out)


def bma_reconstruct(traces: List[str], length: Optional[int] = None) -> str:
    """Reconstruct a strand via Symbolwise Majority Alignment (BMA).

    Generalisation of the Bitwise Majority Alignment algorithm of Batu, Kannan,
    Khanna and McGregor ("Reconstructing strings from random traces", SODA
    2004) to the 4-ary DNA alphabet.

    Algorithm
    ---------
    Keep one read pointer per trace, all initialised to ``0``. Then repeat:

    1. Look at the current symbol of every trace whose pointer is still in
       range.
    2. Emit the *majority* of those current symbols (alphabetical tie-break).
    3. Advance the pointer of **every** trace whose current symbol equals the
       emitted one; leave the others untouched. Leaving the disagreeing traces
       in place is what absorbs insertions/deletions: a trace that is "off by
       one" simply waits until the consensus catches up to it.

    Stop when ``length`` symbols have been emitted (if ``length`` is given) or,
    otherwise, when fewer than half of the traces still have symbols remaining.

    Parameters
    ----------
    traces:
        Noisy reads of a single original strand (``ACGT`` strings).
    length:
        Desired output length. Defaults to the rounded median trace length.

    Returns
    -------
    str
        The reconstructed strand (empty if ``traces`` is empty).
    """
    if not traces:
        return ""

    target_len = _median_length(traces) if length is None else length
    if target_len <= 0:
        return ""

    n = len(traces)
    pointers = [0] * n
    out: List[str] = []

    while len(out) < target_len:
        # Gather the current symbol of every trace still in range.
        active = [j for j in range(n) if pointers[j] < len(traces[j])]
        # Stop early if too few traces remain to form a meaningful majority.
        if len(active) * 2 < n:
            break
        if not active:
            break

        current = [traces[j][pointers[j]] for j in active]
        emit = _majority_symbol(current)
        out.append(emit)

        # Advance only the traces that agreed with the emitted symbol.
        for j in active:
            if traces[j][pointers[j]] == emit:
                pointers[j] += 1

    return "".join(out)


def consensus(traces: List[str], method: str = "bma", length: Optional[int] = None) -> str:
    """Dispatch to a named reconstruction baseline.

    Parameters
    ----------
    traces:
        Noisy reads of a single original strand.
    method:
        Either ``"bma"`` (Symbolwise Majority Alignment, the default) or
        ``"majority"`` (naive positional vote).
    length:
        Optional target length forwarded to the chosen method (ignored by
        ``majority_vote``, which always uses the median length).

    Returns
    -------
    str
        The reconstructed strand.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported names.
    """
    if method == "bma":
        return bma_reconstruct(traces, length=length)
    if method == "majority":
        return majority_vote(traces)
    raise ValueError(f"unknown consensus method: {method!r} (expected 'bma' or 'majority')")
