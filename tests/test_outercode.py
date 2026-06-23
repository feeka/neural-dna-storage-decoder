"""Tests for the classical finite-field outer code (GF256 + Reed-Solomon)."""

from __future__ import annotations

import numpy as np
import pytest

from dnadecoder.outercode.galois import EXP, LOG, GF256
from dnadecoder.outercode.reed_solomon import ReedSolomon, ReedSolomonError


# --------------------------------------------------------------------------- #
# GF256 tests.
# --------------------------------------------------------------------------- #
def test_add_sub_are_xor() -> None:
    for a in range(0, 256, 7):
        for b in range(0, 256, 11):
            assert GF256.add(a, b) == a ^ b
            assert GF256.sub(a, b) == a ^ b
            # a + a == 0 in characteristic 2.
            assert GF256.add(a, a) == 0


def test_inverse_sanity() -> None:
    for a in range(1, 256):
        assert GF256.mul(a, GF256.inv(a)) == 1


def test_inv_and_div_by_zero_raise() -> None:
    with pytest.raises(ZeroDivisionError):
        GF256.inv(0)
    with pytest.raises(ZeroDivisionError):
        GF256.div(5, 0)


def test_mul_by_zero() -> None:
    for a in range(256):
        assert GF256.mul(a, 0) == 0
        assert GF256.mul(0, a) == 0


def test_mul_div_round_trip() -> None:
    for a in range(0, 256, 5):
        for b in range(1, 256, 5):  # b != 0
            assert GF256.div(GF256.mul(a, b), b) == a


def test_exp_log_consistency() -> None:
    # log(exp(i)) == i for the multiplicative group, and exp/log invert.
    for i in range(255):
        assert LOG[EXP[i]] == i
    for a in range(1, 256):
        assert EXP[LOG[a]] == a
    # Doubled EXP table wraps with period 255.
    for i in range(255):
        assert EXP[i] == EXP[i + 255]


def test_distributivity() -> None:
    rng = np.random.default_rng(0)
    vals = [int(x) for x in rng.integers(0, 256, size=20)]
    for a in vals:
        for b in vals:
            for c in vals:
                left = GF256.mul(a, GF256.add(b, c))
                right = GF256.add(GF256.mul(a, b), GF256.mul(a, c))
                assert left == right


def test_pow() -> None:
    for a in range(1, 256, 9):
        acc = 1
        for n in range(6):
            assert GF256.pow(a, n) == acc
            acc = GF256.mul(acc, a)


# --------------------------------------------------------------------------- #
# Reed-Solomon tests.
# --------------------------------------------------------------------------- #
RS_PARAMS = [(20, 12), (255, 223), (15, 11)]


@pytest.mark.parametrize("n,k", RS_PARAMS)
def test_encode_length_and_systematic(n: int, k: int) -> None:
    rs = ReedSolomon(n, k)
    rng = np.random.default_rng(1)
    msg = [int(x) for x in rng.integers(0, 256, size=k)]
    cw = rs.encode(msg)
    assert len(cw) == n
    # Systematic: the codeword starts with the message verbatim.
    assert cw[:k] == msg


@pytest.mark.parametrize("n,k", RS_PARAMS)
def test_round_trip_no_errors(n: int, k: int) -> None:
    rs = ReedSolomon(n, k)
    rng = np.random.default_rng(2)
    for _ in range(20):
        msg = [int(x) for x in rng.integers(0, 256, size=k)]
        cw = rs.encode(msg)
        decoded, nerr = rs.decode(cw)
        assert decoded == msg
        assert nerr == 0


@pytest.mark.parametrize("n,k", RS_PARAMS)
def test_correct_up_to_t_errors(n: int, k: int) -> None:
    rs = ReedSolomon(n, k)
    t = (n - k) // 2
    rng = np.random.default_rng(123)
    for _ in range(100):
        msg = [int(x) for x in rng.integers(0, 256, size=k)]
        cw = rs.encode(msg)
        num_err = int(rng.integers(0, t + 1))
        positions = (
            rng.choice(n, size=num_err, replace=False) if num_err else np.array([], dtype=int)
        )
        corrupted = list(cw)
        for p in positions:
            err = int(rng.integers(1, 256))  # nonzero error magnitude
            corrupted[int(p)] ^= err
        decoded, nerr = rs.decode(corrupted)
        assert decoded == msg
        assert nerr == num_err


@pytest.mark.parametrize("n,k", RS_PARAMS)
def test_more_than_t_errors_not_silently_wrong(n: int, k: int) -> None:
    """Beyond t errors decoding must either raise or be detected as wrong.

    We do NOT assert correct recovery beyond t; we only assert that the decoder
    does not silently return the original message while claiming success in a way
    that would mask the failure. Either it raises ReedSolomonError, or it returns
    a result that differs from the true message (mis-decode / detected).
    """
    rs = ReedSolomon(n, k)
    t = (n - k) // 2
    if t + 1 > n:
        pytest.skip("not enough room for t+1 errors")
    rng = np.random.default_rng(999)
    saw_raise = False
    saw_wrong = False
    for _ in range(50):
        msg = [int(x) for x in rng.integers(0, 256, size=k)]
        cw = rs.encode(msg)
        num_err = t + 1
        positions = rng.choice(n, size=num_err, replace=False)
        corrupted = list(cw)
        for p in positions:
            err = int(rng.integers(1, 256))
            corrupted[int(p)] ^= err
        try:
            decoded, _ = rs.decode(corrupted)
        except ReedSolomonError:
            saw_raise = True
            continue
        if decoded != msg:
            saw_wrong = True
    # Over many trials, uncorrectable patterns must manifest as raises and/or
    # mis-decodes -- never uniformly silent correct recovery.
    assert saw_raise or saw_wrong


def test_uncorrectable_raises_concrete() -> None:
    # A heavily corrupted small codeword should be flagged uncorrectable.
    rs = ReedSolomon(15, 11)  # t = 2
    cw = rs.encode(list(range(11)))
    # Inject 5 errors (>> t).
    for i in (0, 2, 4, 6, 8):
        cw[i] ^= 0x5A
    # Either it raises, or it returns something that is not the message.
    try:
        decoded, _ = rs.decode(cw)
        assert decoded != list(range(11))
    except ReedSolomonError:
        pass


def test_correct_message_helper() -> None:
    rs = ReedSolomon(20, 12)
    msg = list(range(12))
    cw = rs.encode(msg)
    cw[3] ^= 0x11
    assert rs.correct_message(cw) == msg
