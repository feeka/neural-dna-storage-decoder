"""Tests for the denoiser data pipeline."""
import torch

from dnadecoder.data import (
    TraceDataset,
    collate_fn,
    encode_record,
    make_dataloader,
)
from dnadecoder.tokens import PAD_IDX, bases_to_indices


def test_encode_record_shapes_and_values():
    rec = {"original": "ACGT", "traces": ["ACGT", "ACGA", "TCGT"]}
    enc = encode_record(rec)
    assert enc["reads"].shape == (3, 4)         # [K, L]
    assert enc["target"].shape == (4,)          # [L]
    assert enc["target"].tolist() == bases_to_indices("ACGT")
    assert enc["reads"][0].tolist() == bases_to_indices("ACGT")


def test_encode_record_pads_short_reads():
    rec = {"original": "ACGT", "traces": ["AC"]}   # short read
    enc = encode_record(rec)
    row = enc["reads"][0].tolist()
    assert row[:2] == bases_to_indices("AC")
    assert row[2:] == [PAD_IDX, PAD_IDX]


def test_dataset_getitem():
    recs = [{"original": "ACGT", "traces": ["ACGT", "ACGA"]}]
    ds = TraceDataset(recs)
    item = ds[0]
    assert item["reads"].shape == (2, 4)
    assert item["target"].shape == (4,)
    assert item["original"] == "ACGT"


def test_collate_batches():
    recs = [
        {"original": "ACGT", "traces": ["ACGT", "ACGA"]},
        {"original": "TTGC", "traces": ["TTGC", "TAGC"]},
    ]
    batch = collate_fn([TraceDataset(recs)[i] for i in range(2)])
    assert batch["reads"].shape == (2, 2, 4)    # [B, K, L]
    assert batch["target"].shape == (2, 4)      # [B, L]
    assert batch["originals"] == ["ACGT", "TTGC"]


def test_dataloader_yields_batch():
    recs = [{"original": "ACGT", "traces": ["ACGT", "ACGA"]}] * 5
    loader = make_dataloader(recs, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["reads"].dim() == 3
    assert isinstance(batch["target"], torch.Tensor)
