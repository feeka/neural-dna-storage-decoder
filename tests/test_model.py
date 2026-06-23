"""Tests for the non-autoregressive DenoiserTransformer."""
import torch

from dnadecoder.config import ModelConfig
from dnadecoder.model import DenoiserTransformer
from dnadecoder.tokens import NUM_BASES


def _tiny_model(L=6, K=3):
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    return DenoiserTransformer(cfg, length=L, num_traces=K), L, K


def test_forward_shape_and_finite():
    model, L, K = _tiny_model()
    reads = torch.randint(0, NUM_BASES, (4, K, L))
    logits = model(reads)
    assert logits.shape == (4, L, NUM_BASES)
    assert torch.isfinite(logits).all()


def test_predict_returns_index_lists():
    model, L, K = _tiny_model()
    reads = torch.randint(0, NUM_BASES, (5, K, L))
    preds = model.predict(reads)
    assert len(preds) == 5
    assert all(len(p) == L for p in preds)
    assert all(0 <= v < NUM_BASES for p in preds for v in p)


def test_one_training_step_runs():
    model, L, K = _tiny_model()
    reads = torch.randint(0, NUM_BASES, (8, K, L))
    target = torch.randint(0, NUM_BASES, (8, L))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(reads)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, NUM_BASES), target.reshape(-1)
    )
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
