"""Non-autoregressive denoising Transformer for DNA reads.

Given ``K`` position-aligned noisy reads of one strand, the model predicts the
original base at every position in a single forward pass (no autoregressive
decoding, hence no exposure bias).

The reads are embedded on a ``[K, L]`` grid with separate learned position
(``L``) and read-index (``K``) encodings, flattened to ``K*L`` tokens, and passed
through a Transformer encoder. Self-attention lets each position attend both to
the other reads at the same position (the *evidence*, enabling a learned majority
vote) and to neighbouring positions (the *source prior*). The per-position read
representations are mean-pooled over the ``K`` reads and classified into the four
bases.

Because the model receives the full set of reads at each position, it can always
reproduce symbol-wise majority voting — and it improves on it by additionally
exploiting the source's correlation structure, especially at high error rates.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .tokens import INPUT_VOCAB, NUM_BASES, PAD_IDX


class DenoiserTransformer(nn.Module):
    """Encoder-only Transformer mapping a read grid to per-position base logits.

    Parameters
    ----------
    cfg:
        Architecture hyper-parameters.
    length:
        Strand length ``L`` (number of positions).
    num_traces:
        Number of reads ``K`` per strand.
    """

    def __init__(self, cfg: ModelConfig, length: int, num_traces: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.length = length
        self.num_traces = num_traces
        d = cfg.d_model

        # Input symbols: 4 bases + pad. Output classes: 4 bases.
        self.embedding = nn.Embedding(INPUT_VOCAB, d, padding_idx=PAD_IDX)
        # Learned position (over L) and read-index (over K) encodings.
        self.pos_l = nn.Parameter(torch.randn(1, 1, length, d) * 0.02)
        self.pos_k = nn.Parameter(torch.randn(1, num_traces, 1, d) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=getattr(cfg, "norm_first", True),
        )
        # enable_nested_tensor=False avoids a benign prototype warning with
        # pre-LayerNorm layers and keeps behaviour identical.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, cfg.num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d, NUM_BASES)

    def forward(self, reads: Tensor) -> Tensor:
        """Map reads ``[B, K, L]`` of base indices to logits ``[B, L, 4]``."""
        B, K, L = reads.shape
        d = self.cfg.d_model

        h = self.embedding(reads) + self.pos_l + self.pos_k   # [B, K, L, d]
        h = h.reshape(B, K * L, d)
        h = self.encoder(h)                                   # [B, K*L, d]
        h = h.reshape(B, K, L, d).mean(dim=1)                 # [B, L, d] pool reads
        return self.head(h)                                   # [B, L, 4]

    @torch.no_grad()
    def predict(self, reads: Tensor, device: Optional[torch.device] = None) -> List[List[int]]:
        """Return per-example predicted base-index sequences (greedy argmax)."""
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        logits = self.forward(reads.to(device))               # [B, L, 4]
        preds = logits.argmax(dim=-1).cpu().tolist()           # [B, L]
        if was_training:
            self.train()
        return preds
