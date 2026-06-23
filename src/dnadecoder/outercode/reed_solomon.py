"""Systematic Reed-Solomon codec over GF(256).

Implements an ``(n, k)`` Reed-Solomon code with ``nsym = n - k`` parity symbols,
following the canonical "Reed-Solomon codes for coders" reference algorithm:

* generator polynomial with roots ``alpha^0 .. alpha^(nsym-1)``,
* systematic encoding (message symbols followed by parity symbols),
* decoding via syndrome computation, Berlekamp-Massey for the error-locator
  polynomial, a Chien search for the error positions, and Forney's algorithm for
  the error magnitudes.

It can correct up to ``t = nsym // 2`` symbol errors. All polynomial arithmetic
is carried out over GF(256) using :class:`dnadecoder.outercode.galois.GF256`.

Polynomials are represented as ``list[int]`` in descending-degree order, i.e.
``poly[0]`` is the coefficient of the highest power of ``x``.
"""

from __future__ import annotations

from .galois import GF256


class ReedSolomonError(Exception):
    """Raised when a codeword has more errors than the code can correct."""


# --------------------------------------------------------------------------- #
# Polynomial helpers over GF(256).
# --------------------------------------------------------------------------- #
def _poly_scale(p: list[int], x: int) -> list[int]:
    """Multiply every coefficient of polynomial ``p`` by scalar ``x``."""
    return [GF256.mul(coef, x) for coef in p]


def _poly_add(p: list[int], q: list[int]) -> list[int]:
    """Add two polynomials in GF(256) (descending-degree, XOR per coefficient)."""
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def _poly_mul(p: list[int], q: list[int]) -> list[int]:
    """Multiply two polynomials in GF(256)."""
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            r[i + j] ^= GF256.mul(a, b)
    return r


def _poly_eval(p: list[int], x: int) -> int:
    """Evaluate polynomial ``p`` at ``x`` in GF(256) using Horner's method."""
    y = p[0]
    for coef in p[1:]:
        y = GF256.add(GF256.mul(y, x), coef)
    return y


class ReedSolomon:
    """Systematic ``(n, k)`` Reed-Solomon codec over GF(256).

    Parameters
    ----------
    n : int
        Codeword length, ``n <= 255``.
    k : int
        Message length; ``nsym = n - k`` parity symbols are appended.
    """

    def __init__(self, n: int, k: int) -> None:
        if not (0 < k < n <= 255):
            raise ValueError("require 0 < k < n <= 255")
        self.n = n
        self.k = k
        self.nsym = n - k
        self.gen = self._generator_poly(self.nsym)

    # ------------------------------------------------------------------ #
    # Setup.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _generator_poly(nsym: int) -> list[int]:
        """Build the generator polynomial with roots ``alpha^0 .. alpha^(nsym-1)``."""
        g = [1]
        for i in range(nsym):
            g = _poly_mul(g, [1, GF256.pow(2, i)])
        return g

    # ------------------------------------------------------------------ #
    # Encoding.
    # ------------------------------------------------------------------ #
    def encode(self, message: list[int]) -> list[int]:
        """Systematically encode ``message`` into a length-``n`` codeword.

        Parameters
        ----------
        message : list[int]
            ``k`` message symbols in ``[0, 255]``.

        Returns
        -------
        list[int]
            Codeword of length ``n``: the message followed by ``nsym`` parity
            symbols.
        """
        if len(message) != self.k:
            raise ValueError(f"message must have length k={self.k}")

        # Polynomial division of message * x^nsym by the generator; the
        # remainder is the parity. We compute it via synthetic division.
        remainder = [0] * self.nsym
        for symbol in message:
            factor = symbol ^ remainder[0]
            remainder = remainder[1:] + [0]
            if factor != 0:
                for j in range(self.nsym):
                    remainder[j] ^= GF256.mul(self.gen[j + 1], factor)
        return list(message) + remainder

    # ------------------------------------------------------------------ #
    # Decoding.
    # ------------------------------------------------------------------ #
    def _calc_syndromes(self, codeword: list[int]) -> list[int]:
        """Evaluate the received polynomial at ``alpha^0 .. alpha^(nsym-1)``."""
        return [_poly_eval(codeword, GF256.pow(2, i)) for i in range(self.nsym)]

    @staticmethod
    def _berlekamp_massey(synd: list[int]) -> list[int]:
        """Find the error-locator polynomial via Berlekamp-Massey."""
        err_loc = [1]  # sigma(x)
        old_loc = [1]
        for i in range(len(synd)):
            old_loc = old_loc + [0]
            # Discrepancy delta.
            delta = synd[i]
            for j in range(1, len(err_loc)):
                delta ^= GF256.mul(err_loc[-(j + 1)], synd[i - j])
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = _poly_scale(old_loc, delta)
                    old_loc = _poly_scale(err_loc, GF256.inv(delta))
                    err_loc = new_loc
                err_loc = _poly_add(err_loc, _poly_scale(old_loc, delta))
        # Strip leading zeros.
        while len(err_loc) > 1 and err_loc[0] == 0:
            err_loc = err_loc[1:]
        return err_loc

    def _find_error_positions(self, err_loc: list[int], msg_len: int) -> list[int]:
        """Chien search: return error positions (indices into the codeword)."""
        errs = len(err_loc) - 1  # number of errors
        positions: list[int] = []
        for i in range(msg_len):
            # The locator's roots are alpha^(-pos) where pos is counted from the
            # right end of the codeword. Position i from the left corresponds to
            # pos = msg_len - 1 - i from the right, so we evaluate at
            # alpha^-(msg_len-1-i).
            pos_from_right = msg_len - 1 - i
            root = GF256.inv(GF256.pow(2, pos_from_right))
            if _poly_eval(err_loc, root) == 0:
                positions.append(i)
        if len(positions) != errs:
            raise ReedSolomonError("could not locate all errors (uncorrectable)")
        return positions

    def _forney(
        self, synd: list[int], err_loc: list[int], positions: list[int], msg_len: int
    ) -> list[int]:
        """Compute error magnitudes via Forney's algorithm.

        Parameters
        ----------
        positions : list[int]
            Error positions as left-indices into the codeword.

        Returns
        -------
        list[int]
            Length-``msg_len`` correction polynomial to XOR onto the received
            codeword.
        """
        # Error locators X_i = alpha^(position counted from the right end).
        coef_pos = [msg_len - 1 - p for p in positions]
        x_locs = [GF256.pow(2, cp) for cp in coef_pos]

        # Error-evaluator polynomial omega(x) = (S(x) * sigma(x)) mod x^nsym,
        # with S(x) = sum_j synd[j] x^j (ascending). Work in ascending order.
        synd_asc = list(synd)
        err_loc_asc = err_loc[::-1]
        omega = _poly_mul(synd_asc, err_loc_asc)[: self.nsym]  # truncate mod x^nsym

        # Formal derivative of sigma in ascending order: drop even-power terms
        # (they vanish in characteristic 2) and shift down by one degree.
        sigma_deriv = [err_loc_asc[j] if j % 2 == 1 else 0 for j in range(len(err_loc_asc))]
        sigma_deriv = sigma_deriv[1:]  # divide by x (shift), even slots are zero

        magnitude = [0] * msg_len
        for i, xi in enumerate(x_locs):
            xi_inv = GF256.inv(xi)
            # Evaluate omega(X_i^-1) and sigma'(X_i^-1) (ascending Horner).
            num = self._eval_asc(omega, xi_inv)
            den = self._eval_asc(sigma_deriv, xi_inv)
            if den == 0:
                raise ReedSolomonError("Forney denominator is zero (uncorrectable)")
            # e_i = X_i * omega(X_i^-1) / sigma'(X_i^-1).
            magnitude[positions[i]] = GF256.mul(xi, GF256.div(num, den))
        return magnitude

    @staticmethod
    def _eval_asc(poly_asc: list[int], x: int) -> int:
        """Evaluate an ascending-order polynomial at ``x`` in GF(256)."""
        result = 0
        xp = 1
        for coef in poly_asc:
            result = GF256.add(result, GF256.mul(coef, xp))
            xp = GF256.mul(xp, x)
        return result

    def decode(self, codeword: list[int]) -> tuple[list[int], int]:
        """Decode a (possibly corrupted) codeword.

        Parameters
        ----------
        codeword : list[int]
            Received codeword of length ``n``.

        Returns
        -------
        tuple[list[int], int]
            The ``k`` recovered message symbols and the number of corrected
            errors.

        Raises
        ------
        ReedSolomonError
            If the codeword has more errors than the code can correct.
        """
        if len(codeword) != self.n:
            raise ValueError(f"codeword must have length n={self.n}")

        r = list(codeword)
        synd = self._calc_syndromes(r)

        # All-zero syndrome means no detectable error.
        if max(synd) == 0:
            return r[: self.k], 0

        err_loc = self._berlekamp_massey(synd)
        num_errors = len(err_loc) - 1
        if num_errors > self.nsym // 2:
            raise ReedSolomonError("too many errors to correct")

        positions = self._find_error_positions(err_loc, self.n)
        correction = self._forney(synd, err_loc, positions, self.n)

        corrected = [GF256.add(r[i], correction[i]) for i in range(self.n)]

        # Re-check syndromes: if any nonzero remains, decoding failed.
        if max(self._calc_syndromes(corrected)) != 0:
            raise ReedSolomonError("decoding failed verification (uncorrectable)")

        return corrected[: self.k], len(positions)

    def correct_message(self, codeword: list[int]) -> list[int]:
        """Convenience helper: decode and return only the message symbols."""
        message, _ = self.decode(codeword)
        return message
