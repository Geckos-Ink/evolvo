"""Vectorized GFSL operation semantics for batched execution.

``GFSLExecutor.execute_batch`` runs one genome over ``n`` input rows at once. Each built-in
operation below reproduces, element by element, what ``GFSLExecutor._execute_instruction`` does
for one row (the scalar path is the reference; ``tests/test_gfsl_core.py`` checks equality on
random genomes and special values):

* ``_safe_float``: booleans -> 0/1, complex -> modulus, ``None`` -> 0.0.
* ``DIV`` / ``MOD``: 0.0 when the divisor is 0; Python's sign convention for ``%``.
* ``POW``: 0.0 for a negative base with a non-integer (or non-finite) exponent, for ``0 ** negative``
  (``ZeroDivisionError`` in Python) and for finite operands whose power overflows
  (``OverflowError``); otherwise ``base ** exp``.
* ``SQRT``: 0.0 for negative or NaN input. ``LOG``: -100.0 unless input > 1e-9.
* ``EQ``: ``|a - b| < 1e-9``. ``AND/OR/NOT``: truthiness (NaN is true).
* ``GTE``, ``LTE``, ``NEQ`` and tensor/control operations assign nothing (as in the scalar path).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .enums import Operation

LIST_OPS = frozenset(
    {
        Operation.PREPEND,
        Operation.APPEND,
        Operation.CLONE,
        Operation.FIFO,
        Operation.FILO,
        Operation.LISTCOUNT,
        Operation.LISTHASITEMS,
    }
)
# Operations whose results depend on per-row Python objects (lists, function frames): a genome
# using any of them in its main body is executed row by row on the scalar path.
SCALAR_ONLY_OPS = LIST_OPS | {Operation.CALL}


def as_float(value: Any, n: int) -> np.ndarray:
    """Vector version of ``GFSLExecutor._safe_float``, broadcast to length ``n``."""
    if value is None:
        return np.zeros(n)
    arr = np.asarray(value)
    if arr.dtype == bool:
        arr = arr.astype(np.float64)
    elif arr.dtype.kind == "c":
        arr = np.abs(arr)
    elif arr.dtype.kind in "iuf":
        arr = arr.astype(np.float64, copy=False)
    else:
        flat = [_safe_float_scalar(v) for v in np.ravel(arr)]
        arr = np.asarray(flat, dtype=np.float64).reshape(arr.shape)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


def truthy(value: Any, n: int) -> np.ndarray:
    """Vector version of ``bool(value)`` (NaN is true, 0.0 is false)."""
    if value is None:
        return np.zeros(n, dtype=bool)
    arr = np.asarray(value)
    if arr.dtype == bool:
        return np.broadcast_to(arr, (n,)) if arr.ndim == 0 else arr
    return as_float(arr, n) != 0


def _safe_float_scalar(val: Any) -> float:
    try:
        if isinstance(val, complex):
            return abs(val)
        return float(val)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def apply_builtin(op: Operation, s1: Any, s2: Any, n: int) -> Optional[np.ndarray]:
    """Result of a built-in operation over ``n`` rows, or ``None`` when it assigns nothing."""
    with np.errstate(all="ignore"):
        if op in (Operation.GT, Operation.LT, Operation.EQ):
            a, b = as_float(s1, n), as_float(s2, n)
            if op == Operation.GT:
                return a > b
            if op == Operation.LT:
                return a < b
            return np.abs(a - b) < 1e-9
        if op == Operation.AND:
            return truthy(s1, n) & truthy(s2, n)
        if op == Operation.OR:
            return truthy(s1, n) | truthy(s2, n)
        if op == Operation.NOT:
            return ~truthy(s1, n)

        if op == Operation.ADD:
            return as_float(s1, n) + as_float(s2, n)
        if op == Operation.SUB:
            return as_float(s1, n) - as_float(s2, n)
        if op == Operation.MUL:
            return as_float(s1, n) * as_float(s2, n)
        if op == Operation.DIV:
            a, b = as_float(s1, n), as_float(s2, n)
            nz = b != 0
            return np.where(nz, a / np.where(nz, b, 1.0), 0.0)
        if op == Operation.POW:
            a, b = as_float(s1, n), as_float(s2, n)
            finite_b = np.isfinite(b)
            negative_non_integer = (a < 0) & ~(finite_b & (b == np.floor(np.where(finite_b, b, 0.0))))
            r = np.power(a, b)
            zero_division = (a == 0) & (b < 0) & finite_b
            overflow = np.isfinite(a) & finite_b & np.isinf(r)
            return np.where(negative_non_integer | zero_division | overflow, 0.0, r)
        if op == Operation.SQRT:
            a = as_float(s1, n)
            ok = a >= 0
            return np.where(ok, np.power(np.where(ok, a, 0.0), 0.5), 0.0)
        if op == Operation.ABS:
            return np.abs(as_float(s1, n))
        if op == Operation.SIN:
            return np.sin(as_float(s1, n))
        if op == Operation.COS:
            return np.cos(as_float(s1, n))
        if op == Operation.EXP:
            return np.exp(as_float(s1, n))
        if op == Operation.LOG:
            a = as_float(s1, n)
            ok = a > 1e-9
            return np.where(ok, np.log(np.where(ok, a, 1.0)), -100.0)
        if op == Operation.MOD:
            a, b = as_float(s1, n), as_float(s2, n)
            nz = b != 0
            return np.where(nz, np.mod(a, np.where(nz, b, 1.0)), 0.0)
    return None


__all__ = ["LIST_OPS", "SCALAR_ONLY_OPS", "apply_builtin", "as_float", "truthy"]
