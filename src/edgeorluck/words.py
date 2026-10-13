"""Numbers in plain words: short, signed, in the trader's own unit."""
from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd


def fmt(value: Optional[float], unit: str, signed: bool = True, ref: Optional[float] = None) -> str:
    """+12.30 USDT, -0.42%, +1,234; "n/a" when missing. ``ref`` sets the
    precision from a related number (a range is shown at its own scale)."""
    if value is None or not isinstance(value, (int, float, np.floating, np.integer)) or not math.isfinite(value):
        return "n/a"
    v = float(value)
    a = abs(v)
    if ref is not None and math.isfinite(ref) and abs(ref) > a:
        scale = abs(ref)
        digits = 0 if scale >= 100 else 2 if scale >= 1 else max(2, min(6, 2 - int(math.floor(math.log10(scale)))))
        body = f"{a:,.{digits}f}"
        if body.strip("0.,") == "" and a > 0:
            return fmt(value, unit, signed)          # don't round a sign away: show it at its own scale
        sign = ("+" if v > 0 else "-" if v < 0 else "") if signed else ("-" if v < 0 else "")
        text = sign + body
        return text + "%" if unit == "%" else text if unit in ("money", "", None) else f"{text} {unit}"
    if a >= 100:
        body = f"{a:,.0f}"
    elif a >= 1 or a == 0:
        body = f"{a:,.2f}"
    else:
        digits = max(2, min(6, 2 - int(math.floor(math.log10(a)))))
        body = f"{a:.{digits}f}"
    sign = ("+" if v > 0 else "-" if v < 0 else "") if signed else ("-" if v < 0 else "")
    if body.strip("0.,") == "":
        sign = ""
    text = sign + body
    if unit == "%":
        return text + "%"
    if unit in ("money", "", None):
        return text
    return f"{text} {unit}"


def unit_words(unit: str) -> str:
    if unit == "%":
        return "percent per trade"
    if unit == "money":
        return "money (currency not given)"
    return unit


def share(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100 * value:.0f}%"


def count(n: int, word: str, plural: Optional[str] = None) -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def one_in(p: Optional[float], draws: int) -> str:
    """'about 1 in 12' from a probability; 'fewer than 1 in 20,000' at zero."""
    if p is None:
        return "n/a"
    if p <= 0 or p * draws < 1:
        return f"fewer than 1 in {draws:,}"
    n = 1.0 / p
    if n < 1.5:
        return "almost always"
    rounded = int(round(n)) if n < 20 else int(round(n, -1)) if n < 200 else int(round(n, -2)) if n < 2000 \
        else int(round(n, -3))
    return f"about 1 in {rounded:,}"


def day(t: Optional[pd.Timestamp]) -> Optional[str]:
    if t is None or pd.isna(t):
        return None
    return f"{t:%b} {t.day}, {t.year}"


def iso_day(t: Optional[pd.Timestamp]) -> Optional[str]:
    if t is None or pd.isna(t):
        return None
    return t.date().isoformat()


def join_and(parts: Sequence[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def rnd(value: Any, digits: int = 4) -> Any:
    """Round for stable, compact JSON: floats to ``digits`` significant
    digits, nan/inf to None, numpy scalars to Python."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return None
        if v == 0:
            return 0.0
        return float(f"{v:.{digits}g}")
    if isinstance(value, (list, tuple)):
        return [rnd(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: rnd(v, digits) for k, v in value.items()}
    return value


def round_list(values: np.ndarray) -> List[float]:
    return [float(f"{v:.4g}") for v in values]
