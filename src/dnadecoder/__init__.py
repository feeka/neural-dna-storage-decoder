"""neural-dna-decoder: neural consensus for noisy DNA sequencing reads.

A non-autoregressive Transformer that learns the source sequence structure and
performs MAP-style decoding over several error-prone reads, beating classical
majority-vote consensus — especially as the error rate rises.
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
