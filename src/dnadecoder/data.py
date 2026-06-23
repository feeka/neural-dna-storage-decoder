"""Data pipeline for the non-autoregressive denoiser.

A channel record ``{"original": str, "traces": list[str]}`` is turned into:

* ``reads``  : a ``[K, L]`` grid of base indices (one row per read), padded or
  truncated to the original length ``L`` with :data:`~dnadecoder.tokens.PAD_IDX`,
* ``target`` : an ``[L]`` vector of the original base indices.

Under the substitution channel reads already have length ``L`` so no padding
occurs; padding only matters if the general IDS channel is fed in.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from dnadecoder.tokens import PAD_IDX, bases_to_indices


def _read_row(trace: str, length: int) -> List[int]:
    """Indices for one read, truncated/padded to ``length``."""
    idx = bases_to_indices(trace)[:length]
    if len(idx) < length:
        idx = idx + [PAD_IDX] * (length - len(idx))
    return idx


def encode_record(record: dict) -> Dict[str, torch.Tensor]:
    """Encode one record into ``reads`` ``[K, L]`` and ``target`` ``[L]`` tensors."""
    original = record["original"]
    L = len(original)
    reads = [_read_row(t, L) for t in record["traces"]]
    return {
        "reads": torch.tensor(reads, dtype=torch.long),       # [K, L]
        "target": torch.tensor(bases_to_indices(original), dtype=torch.long),  # [L]
    }


class TraceDataset(Dataset):
    """Wraps channel records as ``(reads, target, original)`` items.

    All records must share the same strand length ``L`` and read count ``K`` (true
    for a fixed :class:`~dnadecoder.config.ChannelConfig`), so the default
    collation stacks them into dense batches.
    """

    def __init__(self, records: List[dict]) -> None:
        self.records = records
        self.encoded = [encode_record(r) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict:
        item = self.encoded[i]
        return {
            "reads": item["reads"],
            "target": item["target"],
            "original": self.records[i]["original"],
        }


def collate_fn(batch: List[dict]) -> Dict:
    """Stack a batch into ``reads`` ``[B, K, L]``, ``target`` ``[B, L]`` tensors."""
    reads = torch.stack([b["reads"] for b in batch], dim=0)     # [B, K, L]
    target = torch.stack([b["target"] for b in batch], dim=0)   # [B, L]
    originals = [b["original"] for b in batch]
    return {"reads": reads, "target": target, "originals": originals}


def make_dataloader(
    records: List[dict], batch_size: int = 128, shuffle: bool = True
) -> DataLoader:
    """Build a ``DataLoader`` over channel records."""
    return DataLoader(
        TraceDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
