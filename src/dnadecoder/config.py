"""Configuration dataclasses shared across the pipeline.

Every stage (data generation, model construction, training, evaluation) reads
its parameters from one of these dataclasses, so an experiment is fully described
by a small, serialisable set of values.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ChannelConfig:
    """Parameters of the synthetic DNA-storage read channel.

    The neural decoder is benchmarked on the *substitution* channel (``p_sub``),
    where reads stay aligned to the original strand. The insertion/deletion rates
    (``p_del``/``p_ins``) drive the more general IDS channel used by the simulator
    and the indel-aware baselines.
    """

    length: int = 24            # bases per original strand
    num_traces: int = 3         # noisy reads observed per strand
    p_sub: float = 0.10         # per-base substitution probability
    p_del: float = 0.0          # per-base deletion probability (IDS channel)
    p_ins: float = 0.0          # per-base insertion probability (IDS channel)
    source: str = "markov"      # "markov" (correlated) or "uniform" source
    markov_stay: float = 0.75   # P(next base == current) for the Markov source
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Architecture of the non-autoregressive denoising Transformer."""

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    norm_first: bool = True     # pre-LayerNorm: trains stably without warmup

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Optimisation / dataset-size knobs for a training run."""

    epochs: int = 12
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 0.0
    warmup_steps: int = 200     # >0 enables linear warmup + inverse-sqrt decay
    num_train: int = 10000
    num_val: int = 1000
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 0
    log_every: int = 50
    ckpt_path: str = "checkpoints/model.pt"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
