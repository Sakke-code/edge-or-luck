"""Backtest vs live: are the live trades behaving like the backtest promised?

The backtest is a finished sample; live trades keep coming and the trader
checks often. So the live average gets an anytime-valid range (one side at
1.25%), the backtest average a fixed-sample range (one side at 1.25%), and
the difference range is their union: valid at 95% however often the live
side is checked. "Worse than the backtest" means the whole difference range
is below zero, which is the same as the live e-value against the backtest's
pessimistic average reaching 80.

Next to it, plain odds (one look): if live trades were drawn from the
backtest's own trades, how often would a live record this weak happen?
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional

import numpy as np

from .evidence import half_width, log_e_one_sided, mixing_rho2, serial_factor
from .readers import ReadResult, Trades
from .words import count, fmt, one_in, rnd

MIN_LIVE = 10
BAR = 80.0                      # one side at 1.25% for the live side
Z_BACKTEST = 2.2414             # one side at 1.25% for the backtest side
DRAW_CELLS = 2_000_000


def _sd(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    return max(sd, 1e-12) * math.sqrt(serial_factor(x))


def _pick(read: ReadResult, name: Optional[str] = None) -> Optional[Trades]:
    if not read.strategies:
        return None
    if name:
        for st in read.strategies:
            if st.name == name:
                return st
    return read.strategies[0]


def _series(st: Trades, unit: str) -> Optional[np.ndarray]:
    if st.unit == unit:
        return st.pnl
    if unit == "%" and st.pct is not None and np.isfinite(st.pct).all():
        return st.pct
    return None


def compare_runs(live_read: ReadResult, backtest_read: ReadResult) -> Dict[str, Any]:
    why = ("A backtest is a promise. Live trades show whether the promise holds, and the sooner you know, the less "
           "it costs.")
    live = _pick(live_read)
    bt = _pick(backtest_read, live.name if live else None)
    if live is None or bt is None:
        missing = "live trades" if live is None else "backtest trades"
        return {"code": "need_input", "label": "Can't compare", "tone": "muted",
                "headline": f"Can't compare: no {missing} found.",
                "plain": (backtest_read.need or live_read.need or {}).get("problem", ""), "why": why,
                "next_step": {"text": "Send both files: the backtest's list of trades and the live trades.",
                              "why": "The comparison needs one result per trade on both sides."}}
    unit = live.unit if live.unit == bt.unit else "%"
    x, y = _series(live, unit), _series(bt, unit)
    if x is None or y is None:
        return {"code": "units_differ", "label": "Can't compare", "tone": "muted",
                "headline": f"Can't compare: the live trades are in {live.unit} and the backtest in {bt.unit}.",
                "plain": "Export both in the same unit, or with a percent column, and check again.", "why": why,
                "next_step": {"text": "Export both lists with a percent-per-trade column, then compare again.",
                              "why": "Money per trade depends on position size; percent compares like with like."}}
    n, nb = len(x), len(y)
    mean_l, mean_b = float(np.mean(x)), float(np.mean(y)) if nb else float("nan")
    base = {"live": {"name": live.name, "trades": n, "mean": rnd(mean_l), "unit": unit},
            "backtest": {"name": bt.name, "trades": nb, "mean": rnd(mean_b), "unit": unit},
            "unit_compared": unit, "why": why}
    if nb < 30:
        return dict(base, code="too_early", label="Too early", tone="muted",
                    headline=f"The backtest has only {count(nb, 'trade')}; it can't set a standard yet.",
                    plain="A backtest needs at least 30 trades to say what to expect.",
                    next_step={"text": "Run the backtest over a longer period, then compare again.",
                               "why": "A short backtest promises very little."})
    sd_l, sd_b = _sd(x), _sd(y)
    h_live = half_width(n, sd_l, BAR) if n >= 2 else math.inf
    h_bt = Z_BACKTEST * sd_b / math.sqrt(nb)
    diff = mean_l - mean_b
    lo, hi = diff - h_live - h_bt, diff + h_live + h_bt
    floor = mean_b - h_bt
    log_e = log_e_one_sided(n, float(np.sum(x - floor)), sd_l, mixing_rho2(BAR), -1) if n >= 2 else 0.0
    p = _odds(x, y)
    odds = one_in(p, _draws(n))
    base.update(difference={"value": rnd(diff), "range_95": [rnd(lo), rnd(hi)]}, odds=rnd(p), odds_text=odds,
                e_value_worse=rnd(math.exp(min(log_e, 700))), bar=BAR)
    if n < MIN_LIVE:
        return dict(base, code="too_early", label="Too early", tone="muted",
                    headline=f"Too early to compare: {count(n, 'live trade')} so far.",
                    plain=f"Live averages {fmt(mean_l, unit)} per trade and the backtest {fmt(mean_b, unit)}. With "
                          f"{count(n, 'trade')}, luck decides almost everything.",
                    next_step={"text": f"Collect at least {MIN_LIVE - n} more live trades, then compare again.",
                               "why": "A handful of trades can't tell a broken strategy from a bad week."})
    if hi < 0:
        return dict(base, code="worse_than_backtest", label="Worse than the backtest", tone="bad",
                    headline=f"Live is worse than the backtest promised: {fmt(mean_l, unit)} per trade live, "
                             f"{fmt(mean_b, unit)} in the backtest.",
                    plain=f"Over {count(n, 'live trade')} the gap is {fmt(diff, unit)} per trade (likely "
                          f"{fmt(lo, unit)} to {fmt(hi, unit)}). If live trades were like the backtest's, a live "
                          f"record this weak would happen {odds} times. The backtest was likely too kind: fees, "
                          "fills, or rules tuned to the past.",
                    next_step={"text": "Compare the live fills with the backtest's: fees, slippage and the exact "
                                       "entry prices. Ask your AI to look for lookahead in the backtest.",
                               "why": "The gap between backtest and live almost always comes from costs, fills or "
                                      "a backtest that saw the future."})
    gap_text = (f" If live trades were like the backtest's, a live record this weak would happen {odds} times."
                if diff < 0 else "")
    return dict(base, code="consistent", label="In line with the backtest", tone="good" if diff >= 0 else "warn",
                headline=f"Live is in line with the backtest so far: {fmt(mean_l, unit)} per trade live, "
                         f"{fmt(mean_b, unit)} in the backtest.",
                plain=f"Over {count(n, 'live trade')} the gap is {fmt(diff, unit)} per trade (likely {fmt(lo, unit)} "
                      f"to {fmt(hi, unit)}), and that includes zero.{gap_text} This does not prove the edge; it only "
                      "says live has not broken the promise yet.",
                next_step={"text": "Keep comparing every 20 to 30 new live trades.",
                           "why": "The comparison stays valid however often you check, so checking often is free."})


def _draws(n: int) -> int:
    return int(max(2000, min(20000, DRAW_CELLS // max(n, 1))))


def _odds(x: np.ndarray, y: np.ndarray) -> float:
    """P(a live record of len(x) drawn from the backtest's trades sums this low)."""
    n = len(x)
    draws = _draws(n)
    digest = hashlib.sha256(np.round(np.concatenate((x, y)), 8).tobytes() + b"compare").digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    obs = float(x.sum())
    low = 0
    for start in range(0, draws, 1000):
        rows = min(1000, draws - start)
        sums = y[rng.integers(0, len(y), size=(rows, n))].sum(axis=1)
        low += int((sums <= obs + 1e-12).sum())
    return (low + 1) / (draws + 1)
