"""Evaluation: neural denoiser vs classical consensus baselines.

The neural model predicts each original base in one forward pass; baselines
reconstruct via symbol-wise majority voting (or indel-aware BMA). All methods are
scored with edit distance, symbol-error-rate, and exact-match rate, and rendered
as a markdown comparison table.
"""
from __future__ import annotations

from typing import Iterable

import torch

from dnadecoder import baselines, metrics
from dnadecoder.data import TraceDataset, collate_fn
from dnadecoder.tokens import indices_to_bases


def evaluate_model(
    model, records: list[dict], device: str = "cpu", batch_size: int = 256
) -> dict:
    """Evaluate the neural denoiser on ``records`` and return aggregate metrics."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        TraceDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    preds: list[str] = []
    originals: list[str] = []
    for batch in loader:
        pred_idx = model.predict(batch["reads"], device=device)  # list[list[int]]
        preds.extend(indices_to_bases(row) for row in pred_idx)
        originals.extend(batch["originals"])
    return metrics.aggregate(preds, originals)


def evaluate_baseline(records: list[dict], method: str = "majority") -> dict:
    """Evaluate a classical consensus baseline on ``records``."""
    preds = [baselines.consensus(r["traces"], method=method) for r in records]
    originals = [r["original"] for r in records]
    return metrics.aggregate(preds, originals)


def compare(
    model,
    records: list[dict],
    device: str = "cpu",
    methods: Iterable[str] = ("majority", "bma"),
) -> dict:
    """Compare the neural model against one or more baseline methods."""
    results: dict = {"neural": evaluate_model(model, records, device=device)}
    for m in methods:
        results[m] = evaluate_baseline(records, method=m)
    return results


def format_comparison_table(results: dict) -> str:
    """Render a comparison ``results`` dict as a markdown table (neural first)."""
    header = (
        "| Method | Mean edit dist | Symbol error rate | Exact match rate |\n"
        "| --- | ---: | ---: | ---: |"
    )
    keys = list(results.keys())
    ordered = (["neural"] if "neural" in keys else []) + [
        k for k in keys if k != "neural"
    ]
    rows = []
    for key in ordered:
        m = results[key]
        edit = m.get("mean_edit_distance", float("nan"))
        ser = m.get("mean_ser", float("nan"))
        emr = m.get("exact_match_rate", float("nan"))
        rows.append(f"| {key} | {edit:.4f} | {ser * 100:.2f}% | {emr * 100:.2f}% |")
    return "\n".join([header, *rows])
