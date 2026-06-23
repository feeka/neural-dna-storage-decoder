"""End-to-end experiment: train one denoiser, sweep the substitution rate.

A single decoder is trained on a *mixture* of substitution rates (so it is robust
across the whole range) and then benchmarked against the classical consensus
baselines at each rate. Results are written as a markdown table and a
symbol-error-rate-versus-noise plot.

Uses a non-interactive matplotlib backend so it runs headless (CI, remote boxes).
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import ModelConfig, TrainConfig
from . import channel as channel_mod
from . import evaluate as evaluate_mod
from . import train as train_mod

_SERIES = ("neural", "majority", "bma")


def _quick_overrides() -> Dict[str, Any]:
    """Tiny settings for sub-minute smoke runs."""
    return {
        "length": 12, "num_traces": 3, "markov_stay": 0.75,
        "num_train": 600, "num_val": 100, "epochs": 2, "batch_size": 64,
        "d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 64,
        "lr": 2e-3, "warmup_steps": 50,
        "noise_levels": [0.15], "n_eval": 60,
    }


def _standard_overrides() -> Dict[str, Any]:
    """Defaults for a full run (~minutes on CPU)."""
    return {
        "length": 24, "num_traces": 3, "markov_stay": 0.75,
        "num_train": 10000, "num_val": 1000, "epochs": 12, "batch_size": 128,
        "d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256,
        "lr": 2e-3, "warmup_steps": 200,
        "noise_levels": [0.05, 0.15, 0.25, 0.35], "n_eval": 300,
    }


def run_experiment(
    quick: bool = False,
    out_dir: str = "results",
    noise_levels: Optional[List[float]] = None,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    source: str = "markov",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train one denoiser and benchmark it against baselines across noise levels.

    Returns ``{"results": <p_sub -> compare() dict>, "history": ..., "out_dir": ...}``.
    """
    o = _quick_overrides() if quick else _standard_overrides()

    if noise_levels is None:
        noise_levels = o["noise_levels"]
    noise_levels = sorted(float(x) for x in noise_levels)
    n_eval = o["n_eval"]
    L, K, stay = o["length"], o["num_traces"], o["markov_stay"]

    if model_cfg is None:
        model_cfg = ModelConfig(
            d_model=o["d_model"], nhead=o["nhead"],
            num_layers=o["num_layers"], dim_feedforward=o["dim_feedforward"],
        )
    if train_cfg is None:
        train_cfg = TrainConfig(
            epochs=o["epochs"], batch_size=o["batch_size"], lr=o["lr"],
            warmup_steps=o["warmup_steps"], num_train=o["num_train"],
            num_val=o["num_val"], device=device,
        )
    else:
        train_cfg = replace(train_cfg, device=device)

    os.makedirs(out_dir, exist_ok=True)

    # ---- train on the MIXTURE of substitution rates --------------------------
    train_records = channel_mod.generate_mixed_dataset(
        train_cfg.num_train, L, K, noise_levels,
        seed=train_cfg.seed, source=source, markov_stay=stay,
    )
    val_records = channel_mod.generate_mixed_dataset(
        train_cfg.num_val, L, K, noise_levels,
        seed=train_cfg.seed + 1, source=source, markov_stay=stay,
    )
    model, history = train_mod.train_on_records(
        train_records, val_records, model_cfg, train_cfg
    )

    # ---- evaluate at each substitution rate ----------------------------------
    results: Dict[float, Any] = {}
    for i, p in enumerate(noise_levels):
        records = channel_mod.generate_dataset(
            n_eval, L, K, p_sub=p, seed=10_000 + i,
            source=source, markov_stay=stay,
        )
        results[p] = evaluate_mod.compare(model, records, device=device)

    # ---- markdown summary ----------------------------------------------------
    md = ["# DNA decoder experiment results", ""]
    md.append(
        f"Source: {source} (stay={stay}); strand length {L}; {K} reads/strand; "
        f"substitution channel. Model trained on mixed rates {noise_levels}."
    )
    md.append("")
    for p in noise_levels:
        md.append(f"## Substitution rate p = {p:g}")
        md.append("")
        md.append(evaluate_mod.format_comparison_table(results[p]))
        md.append("")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    _plot_ser(results, noise_levels, os.path.join(out_dir, "ser_vs_noise.png"))
    return {"results": results, "history": history, "out_dir": out_dir}


def _plot_ser(results, noise_levels, png_path) -> None:
    """Plot mean SER vs substitution rate for each method."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    labels = {"neural": "neural (this work)", "majority": "majority vote", "bma": "BMA"}
    for series in _SERIES:
        ys = [results[p].get(series, {}).get("mean_ser", float("nan")) * 100
              for p in noise_levels]
        ax.plot(noise_levels, ys, marker="o", label=labels.get(series, series))
    ax.set_xlabel("per-base substitution rate")
    ax.set_ylabel("mean symbol-error-rate (%)")
    ax.set_title("Reconstruction error vs noise: neural decoder vs baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
