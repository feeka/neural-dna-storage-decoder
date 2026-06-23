"""Finite-field arithmetic over GF(2^8).

This module implements the Galois field GF(256) used by the Reed-Solomon outer
code. Arithmetic uses the standard primitive polynomial ``0x11d`` and a
generator ``alpha = 0x02``. Exponential (``EXP``) and logarithm (``LOG``)
lookup tables are precomputed at import time so that multiplication and division
reduce to integer additions/subtractions of discrete logarithms.

All field elements are integers in ``[0, 255]``.
"""

from __future__ import annotations

# Primitive (irreducible) polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1.
PRIM: int = 0x11D
FIELD_SIZE: int = 256
GENERATOR: int = 0x02


def _build_tables() -> tuple[list[int], list[int]]:
    """Build the EXP and LOG tables for GF(256).

    Returns
    -------
    exp : list[int]
        Table of length 512 where ``exp[i] == alpha**i``. The table is doubled
        (``exp[i] == exp[i + 255]``) so that ``mul`` can index without an
        explicit modulo of the summed logarithms.
    log : list[int]
        Table of length 256 where ``log[exp[i]] == i`` for ``i`` in 0..254.
        ``log[0]`` is undefined (left as 0) and must never be used.
    """
    exp = [0] * 512
    log = [0] * FIELD_SIZE

    x = 1
    for i in range(FIELD_SIZE - 1):  # 0 .. 254, the multiplicative group order
        exp[i] = x
        log[x] = i
        # Multiply x by the generator (0x02) in GF(2^8): shift then reduce.
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    # Duplicate the table so exp[i + 255] == exp[i]; allows skipping the modulo.
    for i in range(FIELD_SIZE - 1, 512):
        exp[i] = exp[i - (FIELD_SIZE - 1)]
    return exp, log


# Precompute lookup tables at import time.
EXP, LOG = _build_tables()


class GF256:
    """Arithmetic over the Galois field GF(2^8) with primitive poly ``0x11d``.

    All methods are static/class methods operating on plain ``int`` field
    elements in ``[0, 255]``. Addition and subtraction are both XOR.
    """

    EXP = EXP
    LOG = LOG

    @staticmethod
    def add(a: int, b: int) -> int:
        """Return ``a + b`` in GF(256) (bitwise XOR)."""
        return a ^ b

    @staticmethod
    def sub(a: int, b: int) -> int:
        """Return ``a - b`` in GF(256) (bitwise XOR, identical to add)."""
        return a ^ b

    @staticmethod
    def mul(a: int, b: int) -> int:
        """Return ``a * b`` in GF(256). Multiplying by zero yields zero."""
        if a == 0 or b == 0:
            return 0
        return EXP[LOG[a] + LOG[b]]

    @staticmethod
    def div(a: int, b: int) -> int:
        """Return ``a / b`` in GF(256).

        Raises
        ------
        ZeroDivisionError
            If ``b`` is zero.
        """
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(256)")
        if a == 0:
            return 0
        # log(a/b) = log(a) - log(b); add 255 to keep the index non-negative
        # (the doubled EXP table makes any index in [0, 510] valid).
        return EXP[LOG[a] + 255 - LOG[b]]

    @staticmethod
    def pow(a: int, n: int) -> int:
        """Return ``a ** n`` in GF(256) for any integer exponent ``n``."""
        if a == 0:
            # 0**0 is conventionally 1; 0**(positive) is 0.
            return 1 if n == 0 else 0
        return EXP[(LOG[a] * n) % 255]

    @staticmethod
    def inv(a: int) -> int:
        """Return the multiplicative inverse of ``a`` in GF(256).

        Raises
        ------
        ZeroDivisionError
            If ``a`` is zero (zero has no inverse).
        """
        if a == 0:
            raise ZeroDivisionError("zero has no inverse in GF(256)")
        return EXP[255 - LOG[a]]
