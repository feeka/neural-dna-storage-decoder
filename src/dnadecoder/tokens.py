"""Canonical 4-symbol DNA alphabet and index mapping.

The neural denoiser operates directly on base indices ``0..3``; keeping the
mapping in one module guarantees the channel, dataset, model and evaluation code
all agree on which index means which base.

    A=0  C=1  G=2  T=3
"""
from __future__ import annotations

from typing import Iterable, List

BASES = "ACGT"
NUM_BASES = len(BASES)

BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for i, b in enumerate(BASES)}

# Index used to pad ragged read grids (e.g. under indels). Distinct from any
# base so the model can learn to ignore it. Embeddings size = NUM_BASES + 1.
PAD_IDX = NUM_BASES  # 4
INPUT_VOCAB = NUM_BASES + 1  # 5 (4 bases + pad)


def bases_to_indices(seq: str) -> List[int]:
    """Map an ``ACGT`` string to a list of base indices."""
    return [BASE_TO_IDX[b] for b in seq]


def indices_to_bases(idxs: Iterable[int]) -> str:
    """Map base indices back to an ``ACGT`` string (pad indices are dropped)."""
    return "".join(IDX_TO_BASE[int(i)] for i in idxs if int(i) in IDX_TO_BASE)
