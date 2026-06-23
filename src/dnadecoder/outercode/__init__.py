"""Classical outer-code layer for DNA storage.

A finite-field (Galois field) implementation and a Reed-Solomon codec built on
top of it. In a real DNA-storage system the Reed-Solomon code protects the data
*across* strands; here it is provided as the algebraic counterpart to the neural
inner decoder and to demonstrate finite-field error correction end to end.
"""
from __future__ import annotations

from .galois import GF256
from .reed_solomon import ReedSolomon

__all__ = ["GF256", "ReedSolomon"]
