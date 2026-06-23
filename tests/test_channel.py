"""Tests for the source models and read channels."""
import numpy as np
import pytest

from dnadecoder import channel
from dnadecoder.config import ChannelConfig
from dnadecoder.tokens import BASES


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_random_strand_length_and_alphabet():
    s = channel.random_strand(50, _rng())
    assert len(s) == 50
    assert set(s) <= set(BASES)


def test_markov_strand_length_and_alphabet():
    s = channel.markov_strand(40, _rng(), stay=0.8)
    assert len(s) == 40
    assert set(s) <= set(BASES)


def test_markov_stay_one_is_constant():
    s = channel.markov_strand(30, _rng(), stay=1.0)
    assert len(set(s)) == 1  # never transitions


def test_substitute_zero_noise_is_identity():
    s = channel.random_strand(40, _rng(1))
    assert channel.substitute(s, 0.0, _rng(2)) == s


def test_substitute_preserves_length_and_changes_at_full_noise():
    s = channel.random_strand(40, _rng(1))
    out = channel.substitute(s, 1.0, _rng(3))
    assert len(out) == len(s)
    assert all(a != b for a, b in zip(s, out))


def test_substitution_traces_stay_aligned():
    s = "ACGTACGTAC"
    traces = channel.make_traces(s, 4, p_sub=0.3, p_del=0.0, p_ins=0.0, rng=_rng(5))
    assert len(traces) == 4
    assert all(len(t) == len(s) for t in traces)


def test_ids_channel_can_change_length():
    s = channel.random_strand(60, _rng(1))
    lens = {len(channel.corrupt(s, 0.05, 0.1, 0.1, _rng(i))) for i in range(20)}
    assert any(L != 60 for L in lens)


def test_generate_dataset_determinism_and_shape():
    a = channel.generate_dataset(8, 20, 3, p_sub=0.1, seed=7)
    b = channel.generate_dataset(8, 20, 3, p_sub=0.1, seed=7)
    assert a == b
    assert len(a) == 8
    for rec in a:
        assert set(rec) == {"original", "traces"}
        assert len(rec["original"]) == 20
        assert len(rec["traces"]) == 3
        assert all(len(t) == 20 for t in rec["traces"])  # substitution-only


def test_generate_from_config_uses_cfg():
    cfg = ChannelConfig(length=16, num_traces=5, p_sub=0.0, source="markov")
    recs = channel.generate_from_config(cfg, 4, seed=1)
    assert all(all(t == r["original"] for t in r["traces"]) for r in recs)


def test_generate_mixed_dataset_spans_levels():
    recs = channel.generate_mixed_dataset(9, 16, 3, [0.0, 0.5], seed=2)
    assert len(recs) == 9
    assert all(t == recs[0]["original"] for t in recs[0]["traces"])


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        channel.make_strand(10, _rng(), source="banana")
