"""Value normalisation shared by every comparison mode.

Reference rows come from the database driver (``Decimal``, ``datetime``, ``bool``);
agent rows often come through JSON (strings, floats). Normalising both sides the
same way, under the tolerance policy, is what makes them comparable.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections.abc import Hashable, Sequence
from decimal import Decimal
from typing import Any

from mindzeval.schema.case import Tolerance


def as_number(value: Any) -> float | None:
    """The value as a float, if it is numeric (or a numeric string). ``None`` otherwise."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def numbers_equal(a: float, b: float, tolerance: Tolerance) -> bool:
    return math.isclose(a, b, rel_tol=tolerance.relative, abs_tol=tolerance.absolute)


def normalise(value: Any, tolerance: Tolerance) -> Hashable:
    """A hashable, comparable form of one cell value.

    Numbers are rounded to the precision the tolerance allows, so equal-within-
    tolerance values hash the same in set and row comparisons.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        if isinstance(value, Decimal) and value == value.to_integral_value():
            return int(value)
        number = float(value)
        if not math.isfinite(number):
            return str(number)
        if number.is_integer() and abs(number) < 2**53:
            return int(number)
        return _round(number, tolerance)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (uuid.UUID, bytes, bytearray, memoryview)):
        value = value.hex() if isinstance(value, (bytes, bytearray, memoryview)) else str(value)
    if isinstance(value, (list, tuple)):
        return tuple(normalise(v, tolerance) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), normalise(v, tolerance)) for k, v in value.items()))
    text = str(value)
    if tolerance.strip_whitespace:
        text = text.strip()
    if not tolerance.case_sensitive:
        text = text.casefold()
    if tolerance.null_equals_empty and text == "":
        return None
    return text


def normalise_key(values: Sequence[Any], tolerance: Tolerance) -> str:
    """Identity of a record from its key column values, as a stable string.

    Keys compare as text so that ``1``, ``1.0`` and ``"1"`` from JSON match the
    integer the database returned.
    """
    parts = []
    for value in values:
        number = as_number(value) if not isinstance(value, str) else None
        if number is not None and number.is_integer():
            parts.append(str(int(number)))
        else:
            norm = normalise(value, tolerance)
            parts.append("" if norm is None else str(norm))
    return "\x1f".join(parts)


def _round(number: float, tolerance: Tolerance) -> float:
    if abs(number) <= tolerance.absolute:
        return 0.0
    if tolerance.relative <= 0:
        return number
    digits = max(1, min(15, round(-math.log10(tolerance.relative))))
    return float(f"{number:.{digits}g}")
