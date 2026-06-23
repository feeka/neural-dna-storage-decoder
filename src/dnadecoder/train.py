"""Training loop and checkpointing for the denoising Transformer.

Teacher-free, single-pass training: the model predicts every original base from
the read grid and is optimised with per-position cross-entropy. An optional
learning-rate warmup (linear ramp then inverse-sqrt decay) lets the pre-LayerNorm
Transformer converge quickly at a high peak learning rate.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from .channel import generate_from_config
from .config import ChannelConfig, ModelConfig, TrainConfig
from .data import make_dataloader
from .model import DenoiserTransformer


def _infer_shape(records: list[dict]) -> tuple[int, int]:
    """Return ``(length, num_traces)`` inferred from the first record."""
    if not records:
        raise ValueError("no records to train on")
    first = records[0]
    return len(first["original"]), len(first["traces"])


def _run_epoch(
    model: DenoiserTransformer,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    log_every: int = 0,
    epoch: int = 0,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> float:
    """Run one epoch; train if ``optimizer`` is given, else evaluate."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader, start=1):
            reads = batch["reads"].to(device)      # [B, K, L]
            target = batch["target"].to(device)    # [B, L]

            logits = model(reads)                  # [B, L, 4]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1)
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if log_every and step % log_every == 0:
                    print(
                        f"  epoch {epoch} | step {step}/{len(loader)} "
                        f"| loss {loss.item():.4f}"
                    )

            total_loss += float(loss.item())
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: DenoiserTransformer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: TrainConfig,
) -> dict[str, list[float]]:
    """Train ``model`` and return the per-epoch loss history."""
    device = cfg.device
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scheduler = None
    warmup = getattr(cfg, "warmup_steps", 0)
    if warmup and warmup > 0:
        def _lr_lambda(step: int) -> float:
            step = max(step, 1)
            return min(step / warmup, (warmup / step) ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=cfg.grad_clip,
            log_every=cfg.log_every, epoch=epoch, scheduler=scheduler,
        )
        val_loss = _run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch {epoch}/{cfg.epochs} | train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f}"
        )

    meta: dict[str, Any] = {
        "model_config": model.cfg.as_dict(),
        "length": model.length,
        "num_traces": model.num_traces,
        "train_config": cfg.as_dict(),
        "history": history,
    }
    save_checkpoint(model, cfg.ckpt_path, meta)
    print(f"saved checkpoint -> {cfg.ckpt_path}")
    return history


def train_on_records(
    train_records: list[dict],
    val_records: list[dict],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[DenoiserTransformer, dict[str, list[float]]]:
    """Build a model from pre-generated records and train it."""
    torch.manual_seed(train_cfg.seed)
    length, num_traces = _infer_shape(train_records)

    train_loader = make_dataloader(
        train_records, batch_size=train_cfg.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        val_records, batch_size=train_cfg.batch_size, shuffle=False
    )
    model = DenoiserTransformer(model_cfg, length=length, num_traces=num_traces)
    history = train(model, train_loader, val_loader, train_cfg)
    return model, history


def build_and_train(
    channel_cfg: ChannelConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[DenoiserTransformer, dict[str, list[float]]]:
    """Generate data from a channel config, build a model, and train it."""
    train_records = generate_from_config(
        channel_cfg, train_cfg.num_train, seed=train_cfg.seed
    )
    val_records = generate_from_config(
        channel_cfg, train_cfg.num_val, seed=train_cfg.seed + 1
    )
    return train_on_records(train_records, val_records, model_cfg, train_cfg)


def save_checkpoint(model: DenoiserTransformer, path: str, meta: dict) -> None:
    """Save model weights and metadata (incl. shape) to ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: str, map_location: str = "cpu"
) -> tuple[DenoiserTransformer, dict]:
    """Load a checkpoint and reconstruct the model."""
    ckpt = torch.load(path, map_location=map_location)
    meta = ckpt["meta"]
    model_cfg = ModelConfig(**meta["model_config"])
    model = DenoiserTransformer(
        model_cfg, length=meta["length"], num_traces=meta["num_traces"]
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, meta
