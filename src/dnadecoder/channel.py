"""Synthetic DNA-storage source models and read channels.

Two source models:

* ``uniform`` — i.i.d. uniform bases (no structure).
* ``markov``  — an order-1 Markov chain over ``ACGT`` with tunable self-transition
  probability. This mimics the local correlations of real encoded/biological DNA
  and gives a *prior* that a learned decoder can exploit but that a symbol-wise
  consensus cannot.

Two channels:

* The substitution channel (``p_sub`` only) keeps every read the same length as
  the original, so reads stay position-aligned — the regime the neural denoiser
  targets.
* The general IDS channel additionally applies insertions/deletions, used by the
  simulator and the indel-aware baseline.

All randomness flows through ``numpy.random.Generator`` instances seeded from a
master ``SeedSequence`` so datasets are reproducible from ``(seed, sizes)`` alone.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from dnadecoder.config import ChannelConfig
from dnadecoder.tokens import BASES

__all__ = [
    "random_strand",
    "markov_strand",
    "make_strand",
    "corrupt",
    "substitute",
    "make_traces",
    "generate_dataset",
    "generate_from_config",
    "generate_mixed_dataset",
]

_BASES = tuple(BASES)
_NUM_BASES = len(_BASES)


# --------------------------------------------------------------------------- #
# source models
# --------------------------------------------------------------------------- #
def random_strand(length: int, rng: np.random.Generator) -> str:
    """Generate a uniformly random DNA strand of ``length`` bases."""
    if length < 0:
        raise ValueError("length must be non-negative")
    idx = rng.integers(0, _NUM_BASES, size=length)
    return "".join(_BASES[i] for i in idx)


def markov_strand(
    length: int, rng: np.random.Generator, stay: float = 0.75
) -> str:
    """Generate a strand from an order-1 Markov chain over ``ACGT``.

    With probability ``stay`` the next base repeats the current one; otherwise it
    is one of the other three bases, chosen uniformly. Larger ``stay`` means more
    correlated (lower-entropy) strands.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0.0 <= stay <= 1.0:
        raise ValueError("stay must be in [0, 1]")
    if length == 0:
        return ""
    other = (1.0 - stay) / (_NUM_BASES - 1)
    cur = int(rng.integers(0, _NUM_BASES))
    out = [_BASES[cur]]
    probs = np.full(_NUM_BASES, other)
    for _ in range(length - 1):
        probs[:] = other
        probs[cur] = stay
        cur = int(rng.choice(_NUM_BASES, p=probs))
        out.append(_BASES[cur])
    return "".join(out)


def make_strand(
    length: int,
    rng: np.random.Generator,
    source: str = "markov",
    markov_stay: float = 0.75,
) -> str:
    """Dispatch to the requested source model."""
    if source == "uniform":
        return random_strand(length, rng)
    if source == "markov":
        return markov_strand(length, rng, stay=markov_stay)
    raise ValueError(f"unknown source: {source!r}")


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #
def _random_base(rng: np.random.Generator) -> str:
    return _BASES[int(rng.integers(0, _NUM_BASES))]


def _random_base_excluding(base: str, rng: np.random.Generator) -> str:
    """Draw a uniformly random base different from ``base`` (a substitution)."""
    cur = _BASES.index(base)
    off = int(rng.integers(1, _NUM_BASES))  # 1..3, never 0
    return _BASES[(cur + off) % _NUM_BASES]


def substitute(strand: str, p_sub: float, rng: np.random.Generator) -> str:
    """Substitution-only channel: each base is flipped to a different base with
    probability ``p_sub``. The read stays aligned (same length) with the input."""
    out = []
    for base in strand:
        if rng.random() < p_sub:
            out.append(_random_base_excluding(base, rng))
        else:
            out.append(base)
    return "".join(out)


def corrupt(
    strand: str,
    p_sub: float,
    p_del: float,
    p_ins: float,
    rng: np.random.Generator,
) -> str:
    """General memoryless IDS channel producing one noisy read.

    Each base is processed left to right: an insertion may fire before it, then
    the base is deleted, substituted, or copied. A final trailing insertion check
    follows the last base. Reads can change length and the result may rarely be
    empty.
    """
    out: List[str] = []
    for base in strand:
        if p_ins and rng.random() < p_ins:
            out.append(_random_base(rng))
        if p_del and rng.random() < p_del:
            continue
        if p_sub and rng.random() < p_sub:
            out.append(_random_base_excluding(base, rng))
        else:
            out.append(base)
    if p_ins and rng.random() < p_ins:
        out.append(_random_base(rng))
    return "".join(out)


def make_traces(
    strand: str,
    num_traces: int,
    p_sub: float,
    p_del: float,
    p_ins: float,
    rng: np.random.Generator,
) -> List[str]:
    """Produce ``num_traces`` independent reads through the IDS channel.

    When ``p_del == p_ins == 0`` this is the substitution channel and all reads
    keep the original length.
    """
    if p_del == 0.0 and p_ins == 0.0:
        return [substitute(strand, p_sub, rng) for _ in range(num_traces)]
    return [corrupt(strand, p_sub, p_del, p_ins, rng) for _ in range(num_traces)]


# --------------------------------------------------------------------------- #
# dataset generation
# --------------------------------------------------------------------------- #
def _record(rng, length, num_traces, p_sub, p_del, p_ins, source, markov_stay):
    original = make_strand(length, rng, source=source, markov_stay=markov_stay)
    traces = make_traces(original, num_traces, p_sub, p_del, p_ins, rng)
    return {"original": original, "traces": traces}


def generate_dataset(
    num_strands: int,
    length: int,
    num_traces: int,
    p_sub: float,
    p_del: float = 0.0,
    p_ins: float = 0.0,
    seed: int = 0,
    source: str = "markov",
    markov_stay: float = 0.75,
) -> List[dict]:
    """Generate a reproducible dataset of ``{"original", "traces"}`` records.

    Determinism comes from spawning one independent child generator per strand
    from the master ``SeedSequence``, so output depends only on ``seed`` and the
    requested sizes.
    """
    child_seeds = np.random.SeedSequence(seed).spawn(num_strands)
    return [
        _record(
            np.random.default_rng(ss), length, num_traces,
            p_sub, p_del, p_ins, source, markov_stay,
        )
        for ss in child_seeds
    ]


def generate_from_config(
    cfg: ChannelConfig,
    num_strands: int,
    seed: Optional[int] = None,
) -> List[dict]:
    """Generate a dataset using parameters from a :class:`ChannelConfig`."""
    eff_seed = cfg.seed if seed is None else seed
    return generate_dataset(
        num_strands=num_strands,
        length=cfg.length,
        num_traces=cfg.num_traces,
        p_sub=cfg.p_sub,
        p_del=cfg.p_del,
        p_ins=cfg.p_ins,
        seed=eff_seed,
        source=cfg.source,
        markov_stay=cfg.markov_stay,
    )


def generate_mixed_dataset(
    num_strands: int,
    length: int,
    num_traces: int,
    p_sub_levels: Sequence[float],
    seed: int = 0,
    p_del: float = 0.0,
    p_ins: float = 0.0,
    source: str = "markov",
    markov_stay: float = 0.75,
) -> List[dict]:
    """Generate a dataset spanning several substitution rates.

    Each strand is corrupted at one rate drawn round-robin from ``p_sub_levels``;
    training on the mixture yields a single decoder robust across the whole range.
    """
    levels = [float(x) for x in p_sub_levels]
    if not levels:
        raise ValueError("p_sub_levels must be non-empty")
    child_seeds = np.random.SeedSequence(seed).spawn(num_strands)
    records: List[dict] = []
    for i, ss in enumerate(child_seeds):
        rng = np.random.default_rng(ss)
        records.append(
            _record(
                rng, length, num_traces, levels[i % len(levels)],
                p_del, p_ins, source, markov_stay,
            )
        )
    return records
