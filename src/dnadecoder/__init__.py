"""neural-dna-decoder: a non-autoregressive Transformer that denoises DNA-storage
reads, performing learned MAP decoding that beats classical consensus baselines.
"""
from __future__ import annotations

from .config import ChannelConfig, ModelConfig, TrainConfig
from .model import DenoiserTransformer
from . import tokens

__all__ = [
    "ChannelConfig",
    "ModelConfig",
    "TrainConfig",
    "DenoiserTransformer",
    "tokens",
    "__version__",
]

__version__ = "0.1.0"
