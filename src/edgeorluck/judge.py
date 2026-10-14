"""The judge: is a trading record a real edge, or luck?

``judge(data, ...)`` reads whatever it is given (see readers.py), and for each
strategy says in plain words whether its record beats luck and how sure that
is, what would change the verdict, what to check, what to ask the trader, and
one next step. The same input always gives the same output.

The verdict per strategy:

* too_early  fewer than MIN_TRADES closed trades;
* faded      it had a proven edge, and the latest trades are both negative and
             too bad for that edge to explain (a strict, two-window test);
* real_edge  e-BH selects "the average is above zero" across every strategy
             and version tried, and (when we can measure it) the edge survives
             taking out what the market itself did during the trades;
* losing     e-BH selects "the average is below zero";
* cant_tell  the average is positive but not proven;
* no_edge    the average is zero or negative, not steadily enough to be "losing".

All maths is in evidence.py; this module turns it into words.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evidence import (ALPHA, BASE_BAR, Q, T_STAR, ebh, log_e_record, max_versions_passing, mixing_rho2,
                       normal_tail, range_of, record_of, skew_shift, trades_needed, wilson_bounds_count, z_equivalent)
from .compare import compare_runs
from .readers import ReadResult, Trades, number, read_any
from .words import count, day, fmt, iso_day, join_and, one_in, rnd, round_list, share

ENGINE_VERSION = "2.0.0"
SCHEMA_VERSION = "1.0"
CHANGELOG = [
    {"version": "2.0.0", "date": "2026-10-13",
     "change": "Standalone judge for traders and their AI assistants: versions tried raise the bar (e-BH with the "
               "unseen versions as e = 0), a rough check against buying and holding, money first, plain 'why' on "
               "everything, red flags for records that look too good, a strict 'faded' test, 'broken or bad "
               "streak' odds, backtest-vs-live, and a serial-correlation widening of the spread."},
    {"version": "1.0", "date": "2026-10-12",
     "change": "First judge: time-uniform range, e-BH across a trader's strategies, coin-flip copies, drawdowns "
               "against the trader's own history, strategies that are one bet, and a graded forward call."},
]

MIN_TRADES = 30
MIN_LOSSES = 5            # "real edge" needs this many losing trades: an unseen loss has an unknown size
FEW_TRADES = 100
RECENT = 30
FADED_WINDOWS = (30, 60)
FADED_P = 0.0025          # each window; two windows -> about 0.5% for a false "faded"
STREAK_DRAWS = 20_000
COIN_FLIPS = 1000
FAN_PATHS = 40
CURVE_POINTS = 160
BOOT_CELLS = 3_000_000
SIM_CELLS = 20_000_000    # above this, coin flips are counted from the normal approximation
MIN_PAIR_DAYS = 20
SAME_BET = 0.6

VERDICTS = {
    "real_edge": ("Real edge", "good"),
    "cant_tell": ("Can't tell yet", "warn"),
    "no_edge": ("No edge", "muted"),
    "losing": ("Losing", "bad"),
    "faded": ("Faded", "faded"),
    "too_early": ("Too early", "muted"),
}

VERDICT_WHY = {
    "real_edge": "A real edge means the profit is too steady to be luck. Only then is it worth more testing.",
    "cant_tell": "A good-looking record can come from luck. If it is luck, it stops working when real money goes in.",
    "no_edge": "With no edge, each trade pays fees and nothing comes back on average.",
    "losing": "Steady losses mean something in the rules works against you. Bad luck does not look this steady.",
    "faded": "Markets change. A strategy that worked before can stop working, and old profits do not protect new "
             "money.",
    "too_early": "With few trades, luck decides almost everything. Any verdict now would be a guess.",
}

SOURCES = {"backtest": "backtest", "back test": "backtest", "bt": "backtest", "simulation": "backtest",
           "paper": "paper", "demo": "paper", "forward": "paper", "forward test": "paper", "live": "live",
           "real": "live", "real money": "live", "real-money": "live"}


# ── settings ──────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    variants_tried: Optional[int] = None
    benchmark_return_pct: Optional[float] = None
    fees_included: Optional[bool] = None
    source: Optional[str] = None
    unit: Optional[str] = None
    kind: Optional[str] = None
    include_chart: bool = False
    notes: Tuple[str, ...] = ()


def make_settings(variants_tried: Any = None, benchmark_return_pct: Any = None, fees_included: Any = None,
                  source: Any = None, unit: Any = None, kind: Any = None, include_chart: bool = False) -> Settings:
    """Settings from whatever the caller passed: "about 40", "34.5%", "yes"."""
    notes: List[str] = []
    versions = None
    if variants_tried is not None and not isinstance(variants_tried, bool):
        value = number(variants_tried)
        if math.isfinite(value) and value >= 1:
            versions = int(round(value))
        elif str(variants_tried).strip():
            notes.append(f"Ignored variants_tried={variants_tried!r}: it needs a whole number of 1 or more.")
    bench = None
    if benchmark_return_pct is not None and not isinstance(benchmark_return_pct, bool):
        value = number(benchmark_return_pct)
        if math.isfinite(value) and value > -100:
            bench = value
        elif str(benchmark_return_pct).strip():
            notes.append(f"Ignored benchmark_return_pct={benchmark_return_pct!r}: give a percent like 34.5.")
    fees = None
    if isinstance(fees_included, bool):
        fees = fees_included
    elif fees_included is not None:
        text = str(fees_included).strip().lower()
        fees = True if text in ("yes", "y", "true", "1", "included") else False if text in (
            "no", "n", "false", "0", "excluded", "not included") else None
    src = None
    if source is not None:
        src = SOURCES.get(str(source).strip().lower())
        if src is None and str(source).strip():
            notes.append(f"Ignored source={source!r}: use backtest, paper or live.")
    return Settings(versions, bench, fees, src, unit, kind, bool(include_chart), tuple(notes))


# ── small statistics ──────────────────────────────────────────────────────────

def _drawdown(x: np.ndarray) -> Tuple[float, float, int]:
    """(worst drop from a peak, current drop, trades since the last peak)."""
    curve = np.concatenate(([0.0], np.cumsum(x)))
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    # "At the peak" up to rounding, measured against the size of the curve. An absolute 1e-12 misread large
    # amounts (628040.6 - 72356.44 + 72356.44 is not exactly 628040.6 in floating point), and the card then said
    # "0.000000 USDT below your peak, 2 trades ago" (found by tests/test_bridge_counting.py).
    tol = 1e-10 * max(1.0, float(np.max(np.abs(curve))))
    dd = np.where(dd <= tol, 0.0, dd)
    at_peak = np.nonzero(dd == 0.0)[0]
    return float(dd.max()), float(dd[-1]), int(len(curve) - 1 - at_peak[-1])


def _rng(x: np.ndarray, salt: str) -> np.random.Generator:
    """Seeded by the trades themselves: the same trades, the same numbers."""
    digest = hashlib.sha256(np.round(np.asarray(x, dtype=float), 8).tobytes() + salt.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _drawdown_odds(x: np.ndarray, depth: float) -> Optional[float]:
    """How often histories resampled from these trades dip at least this deep."""
    if depth <= 0:
        return None
    n = len(x)
    paths = int(max(200, min(2000, BOOT_CELLS // max(n, 1))))
    rng = _rng(x, "drawdown")
    deeper = 0
    for start in range(0, paths, 250):
        rows = min(250, paths - start)
        curve = np.cumsum(x[rng.integers(0, n, size=(rows, n))], axis=1)
        curve = np.concatenate((np.zeros((rows, 1)), curve), axis=1)
        worst = (np.maximum.accumulate(curve, axis=1) - curve).max(axis=1)
        deeper += int((worst >= depth - 1e-12).sum())
    return 100.0 * deeper / paths


def _coin_flips(x: np.ndarray, fan: bool) -> Tuple[int, List[np.ndarray]]:
    """How many of COIN_FLIPS sign-randomised copies of these trades ended
    higher (same trades, same sizes, signs from a coin), and FAN_PATHS curves."""
    n = len(x)
    rng = _rng(x, "coin")
    total = float(x.sum())
    curves: List[np.ndarray] = []
    if fan:
        signs = rng.integers(0, 2, size=(FAN_PATHS, n), dtype=np.int8) * 2 - 1
        for row in signs * x:
            curves.append(np.concatenate(([0.0], np.cumsum(row))))
    if COIN_FLIPS * n > SIM_CELLS:
        z = total / math.sqrt(max(float(np.dot(x, x)), 1e-300))
        return int(round(COIN_FLIPS * normal_tail(z))), curves
    above = 0
    for start in range(0, COIN_FLIPS, 200):
        rows = min(200, COIN_FLIPS - start)
        signs = rng.integers(0, 2, size=(rows, n), dtype=np.int8) * 2 - 1
        above += int(((signs * x).sum(axis=1) > total + 1e-12).sum())
    return above, curves


def _downsample(path: np.ndarray, points: int = CURVE_POINTS) -> np.ndarray:
    if len(path) <= points:
        return path
    return path[np.linspace(0, len(path) - 1, points).round().astype(int)]


def _stretch(earlier: np.ndarray, recent: np.ndarray, salt: str) -> Dict[str, float]:
    """Odds of a stretch like ``recent`` if the trades kept coming from
    ``earlier``: its total that low (or high) and a drop inside it that deep."""
    w = len(recent)
    rng = _rng(np.concatenate((earlier, recent)), salt)
    idx = rng.integers(0, len(earlier), size=(STREAK_DRAWS, w))
    paths = earlier[idx]
    sums = paths.sum(axis=1)
    curve = np.concatenate((np.zeros((STREAK_DRAWS, 1)), np.cumsum(paths, axis=1)), axis=1)
    falls = (np.maximum.accumulate(curve, axis=1) - curve).max(axis=1)
    obs_sum = float(recent.sum())
    rc = np.concatenate(([0.0], np.cumsum(recent)))
    obs_fall = float((np.maximum.accumulate(rc) - rc).max())
    d = STREAK_DRAWS + 1
    return {"sum": obs_sum, "fall": obs_fall,
            "p_low": (int((sums <= obs_sum + 1e-12).sum()) + 1) / d,
            "p_high": (int((sums >= obs_sum - 1e-12).sum()) + 1) / d,
            "p_fall": (int((falls >= obs_fall - 1e-12).sum()) + 1) / d if obs_fall > 0 else 1.0}


def _max_losing_streak(x: np.ndarray) -> int:
    best = run = 0
    for v in x:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


def _p_streak_at_most(n: int, p_loss: float, longest: int) -> float:
    """P(the longest losing streak in n independent trades is <= longest)."""
    if longest >= n:
        return 1.0
    state = [0.0] * (longest + 1)
    state[0] = 1.0
    for _ in range(n):
        nxt = [0.0] * (longest + 1)
        stay = sum(state)
        nxt[0] = stay * (1.0 - p_loss)
        for k in range(longest):
            nxt[k + 1] = state[k] * p_loss
        state = nxt
    return float(sum(state))


def _overlap_share(entries: List[pd.Timestamp], exits: List[pd.Timestamp]) -> Optional[float]:
    """Share of trades that were open at the same time as another trade."""
    if not entries or not exits or len(entries) != len(exits):
        return None
    spans = sorted((e.value, x.value, i) for i, (e, x) in enumerate(zip(entries, exits)) if x.value >= e.value)
    if len(spans) < 2:
        return None
    overlapping = set()
    latest_end, latest_i = spans[0][1], spans[0][2]
    for start, end, i in spans[1:]:
        if start < latest_end:
            overlapping.add(i)
            overlapping.add(latest_i)
        if end > latest_end:
            latest_end, latest_i = end, i
    return len(overlapping) / len(spans)


def _exposure(entries: List[pd.Timestamp], exits: List[pd.Timestamp]) -> Optional[float]:
    """Share of the time from the first entry to the last exit spent in a trade."""
    spans = sorted((e.value, x.value) for e, x in zip(entries, exits) if x.value >= e.value)
    if not spans:
        return None
    start, end = spans[0][0], max(x for _, x in spans)
    if end <= start:
        return None
    covered, cur_s, cur_e = 0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > cur_e:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += cur_e - cur_s
    return covered / (end - start)


# ── measuring one strategy ────────────────────────────────────────────────────

def _measure(st: Trades, family_bar: float) -> Dict[str, Any]:
    x = st.pnl
    n = len(x)
    m: Dict[str, Any] = {"n": n}
    if n == 0:
        return m
    rec = record_of(x)
    rho_family = mixing_rho2(family_bar)
    m.update(mean=rec.mean, total=rec.total, sd_eff=rec.sd, serial_factor=rec.serial, skew=rec.skew,
             record=rec, rho_family=rho_family, family_bar=family_bar)
    # No range before MIN_TRADES: the spread estimated from a handful of trades is too unsteady for the 95%
    # promise (at 2 trades the range misses the true mean 9% of the time, and a trader checking after every
    # trade from 2 on is misled about 13% of the time). From 30 trades, where verdicts start, it holds.
    if n >= MIN_TRADES and math.isfinite(rec.sd):
        m["lower"], m["upper"] = range_of(rec, BASE_BAR)
        m["adj_lower"], m["adj_upper"] = range_of(rec, family_bar, rho_family)
    ok = n >= MIN_TRADES and math.isfinite(rec.sd)
    m["log_e_up"] = log_e_record(rec, family_bar, +1, rho_family) if ok else 0.0
    m["log_e_down"] = log_e_record(rec, family_bar, -1, rho_family) if ok else 0.0
    return m


def _passes_alone(x: np.ndarray, bar: float) -> bool:
    if len(x) < MIN_TRADES:
        return False
    return log_e_record(record_of(x), bar, +1) >= math.log(bar)


def _market(st: Trades, settings: Settings, single_symbol: bool) -> Optional[Dict[str, Any]]:
    """What simply holding did over the same dates, and, when every trade's
    dates and size are known, what the market alone gave each trade (rough:
    a steady rise at the hold's average rate)."""
    hold, source = None, None
    if settings.benchmark_return_pct is not None:
        hold, source = settings.benchmark_return_pct, "given"
    elif st.benchmark_pct is not None:
        hold, source = st.benchmark_pct, st.benchmark_source or "from the input"
    elif (single_symbol and st.entry_price is not None and st.exit_price is not None
          and np.isfinite(st.entry_price[0]) and np.isfinite(st.exit_price).any() and st.entry_price[0] > 0):
        last = st.exit_price[np.isfinite(st.exit_price)][-1]
        first_i = 0
        if st.entry_times:
            first_i = int(np.argmin([t.value for t in st.entry_times]))
        first = st.entry_price[first_i]
        if math.isfinite(first) and first > 0:
            hold, source = 100.0 * (last / first - 1.0), "derived: first entry price to last exit price"
    out: Dict[str, Any] = {"hold_pct": hold, "hold_source": source}
    strategy_pct, strategy_source = None, None
    if st.strategy_return_pct is not None:
        strategy_pct, strategy_source = st.strategy_return_pct, "from the input"
    else:
        pct = st.pnl if st.unit == "%" else st.pct
        if pct is not None and np.isfinite(pct).all() and (pct > -100).all():
            strategy_pct = 100.0 * (float(np.prod(1.0 + pct / 100.0)) - 1.0)
            strategy_source = "rough: every trade's percent compounded, one trade at a time"
    out.update(strategy_pct=strategy_pct, strategy_source=strategy_source)
    entries, exits = st.entry_times, st.times
    if entries and exits:
        out["time_in_market"] = _exposure(entries, exits)
        span = (max(t.value for t in exits) - min(t.value for t in entries)) / 86400e9
        out["span_days"] = span
    if hold is None:
        return None
    g_total = hold / 100.0
    if entries and exits and out.get("span_days", 0) > 0 and g_total > -1:
        span_ns = max(t.value for t in exits) - min(t.value for t in entries)
        held = np.array([max(0, x.value - e.value) / span_ns for e, x in zip(entries, exits)])
        drift = (1.0 + g_total) ** held - 1.0
        side = st.side if st.side is not None else np.ones(len(st.pnl))
        side = np.where(np.isfinite(side), side, 1.0)
        size = None
        if st.unit == "%":
            size = np.full(len(st.pnl), 100.0)
        elif st.size is not None and np.isfinite(st.size).all():
            size = st.size
        elif st.pct is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                implied = np.where(np.abs(st.pct) > 1e-9, 100.0 * st.pnl / st.pct, np.nan)
            if np.isfinite(implied).sum() >= len(implied) / 2:
                size = np.where(np.isfinite(implied), implied, np.nanmedian(implied))
        if size is not None:
            out["market_part"] = side * size * drift
    return out


# ── red flags: records that look too good ─────────────────────────────────────

AUDIT_ITEMS = {
    "lookahead": "Lookahead: does it decide on a bar's close and then fill at that same close or earlier? A signal "
                 "only exists after its bar closes, so the trade must happen on the next bar.",
    "peeking": "Indicators that peek at the future: repainting, request.security with lookahead on, pivots or "
               "zigzags that confirm bars later, centered averages, shift(-1), or scaling with numbers from the "
               "whole data set.",
    "fees": "Fees and slippage: are trading fees charged on both the buy and the sell, plus a little slippage?",
    "survivorship": "Survivorship: was the coin or stock list picked today (ones that still exist, or went up) "
                    "and then tested on the past?",
    "compounding": "Compounding size: does the position size grow with the account, so a few late trades decide "
                   "the whole result?",
    "averaging_down": "Averaging down or martingale: does it buy more when a position is losing, or raise the size "
                      "after a loss? Is there a hard stop that always fires?",
}


def _red_flags(st: Trades, m: Dict[str, Any]) -> List[Dict[str, Any]]:
    x, n = st.pnl, len(st.pnl)
    flags: List[Dict[str, Any]] = []
    if n < MIN_TRADES:
        return flags
    wins, losses = x[x > 0], x[x < 0]
    win_rate = len(wins) / n
    pf = float(wins.sum() / -losses.sum()) if len(losses) and len(wins) else None
    if win_rate >= 0.85 and n >= 50:
        flags.append({"code": "high_win_rate", "items": ["lookahead", "peeking", "averaging_down"],
                      "text": f"It wins {share(win_rate)} of {n} trades. Real strategies rarely win that often.",
                      "why": "A very high win rate is the most common sign of a backtest bug, like trading on a "
                             "price the bot could not have known yet."})
    if len(losses) >= 5 and n >= 50:
        longest = _max_losing_streak(x)
        p = _p_streak_at_most(n, len(losses) / n, longest)
        if p < 0.01:
            flags.append({"code": "no_losing_streaks", "items": ["lookahead", "peeking"],
                          "text": f"It never loses more than {longest} trade{'s' if longest != 1 else ''} in a row. "
                                  f"With this many losses, luck would give longer losing runs ({one_in(p, 10**6)}).",
                          "why": "Losing trades come in clumps in real markets. Records without clumps often use "
                                 "information from the future."})
    if pf is not None and pf >= 4 and n >= 50:
        flags.append({"code": "huge_profit_factor", "items": ["lookahead", "peeking", "fees"],
                      "text": f"It wins {pf:.1f} times what it loses. Above about 2 is rare outside backtests.",
                      "why": "Profit this lopsided usually means the test is too kind: fees, fills or future "
                             "prices."})
    worst_dd = m.get("max_dd", 0.0)
    if m.get("total", 0) > 0 and len(losses) >= 5 and n >= 50:
        typical_loss = float(np.median(-losses))
        if worst_dd <= 2.0 * typical_loss:
            flags.append({"code": "tiny_drawdown", "items": ["lookahead", "peeking"],
                          "text": "Its worst drop is only about two losing trades deep, over the whole record.",
                          "why": "Every real strategy has bad weeks. A curve with no dips is a warning sign."})
    if len(wins) >= 5 and len(losses) >= 1 and win_rate >= 0.7:
        avg_win, avg_loss = float(wins.mean()), float(-losses.mean())
        worst = float(-losses.min())
        if avg_loss >= 3 * avg_win or worst >= 10 * float(np.median(wins)):
            flags.append({"code": "small_wins_big_losses", "items": ["averaging_down", "fees"],
                          "text": f"Many small wins (about {fmt(avg_win, st.unit, False)}) and rare big losses "
                                  f"(up to {fmt(worst, st.unit, False)}). This is the shape of grid, martingale "
                                  "and averaging-down bots.",
                          "why": "These records look great until one bad move. Then one loss can take months of "
                                 "wins, or the whole account."})
    if st.buys_per_trade is not None and len(st.buys_per_trade) == n:
        multi = float(np.mean(st.buys_per_trade >= 3))
        if multi >= 0.3:
            flags.append({"code": "averaging_down", "items": ["averaging_down"],
                          "text": f"{share(multi)} of positions were built with 3 or more buys.",
                          "why": "Buying more as the price falls hides losses until the price never comes back."})
    if st.unit == "%" or st.pct is not None:
        pass
    if st.size is not None and np.isfinite(st.size).all() and n >= 50:
        first, last = float(np.median(st.size[: n // 5])), float(np.median(st.size[-(n // 5):]))
        if first > 0 and last / first >= 5:
            flags.append({"code": "growing_size", "items": ["compounding"],
                          "text": f"Positions grew about {last / first:.0f} times bigger from the first trades to "
                                  "the last.",
                          "why": "When size grows with the account, the last few trades decide the total, and a "
                                 "small edge looks big."})
    return flags


def _audit_prompt(flags: List[Dict[str, Any]], st: Trades, positive_backtest: bool,
                  losing: bool = False) -> Optional[str]:
    if not flags and not positive_backtest:
        return None
    keys: List[str] = []
    for f in flags:
        for item in f["items"]:
            if item not in keys:
                keys.append(item)
    for item in ("lookahead", "peeking", "fees", "survivorship", "compounding", "averaging_down"):
        if item not in keys:
            keys.append(item)
    reasons = "; ".join(f["text"] for f in flags) if flags else \
        "the record passed, and I want to be sure the test itself is honest"
    intro = ("I am not a programmer. Please audit my trading bot's code: its results look better than real "
             "trading usually is, and I want to know if a bug causes that.")
    if losing:
        intro = ("I am not a programmer. Please audit my trading bot's code: many small wins are being erased by a "
                 "few big losses, and I want to know which rules cause that.")
    lines = [
        intro,
        f"Why I am asking: an independent check of my trades said: {reasons}",
        "Check each of these, in this order:",
    ]
    lines += [f"{i}. {AUDIT_ITEMS[k]}" for i, k in enumerate(keys, start=1)]
    lines += [
        "For each one: show me the exact line of code, explain the problem in simple words, and fix it.",
        "Then run the backtest again with the fixes and give me the new list of trades as a CSV file, so I can "
        "check it again.",
    ]
    return "\n".join(lines)


# ── the cards ─────────────────────────────────────────────────────────────────

def _check(code: str, status: str, text: str, why: str, fix: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "status": status, "text": text, "fix": fix, "why": why}


def _question(qid: str, field: Optional[str], question: str, why: str, answer_with: str,
              impact: str = "high") -> Dict[str, Any]:
    return {"id": qid, "field": field, "question": question, "why": why, "answer_with": answer_with,
            "impact": impact}


def _verdict(code: str, headline: str, plain: str, why: Optional[str] = None) -> Dict[str, Any]:
    label, tone = VERDICTS[code]
    return {"code": code, "label": label, "tone": tone, "headline": headline, "plain": plain,
            "why": why or VERDICT_WHY[code], "sentence": headline}


def _fingerprint_of(st: Trades) -> str:
    body: Dict[str, Any] = {"unit": st.unit, "pnl": [round(float(v), 8) for v in st.pnl]}
    if st.times:
        body["times"] = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in st.times]
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _card(st: Trades, m: Dict[str, Any], *, up: bool, down: bool, settings: Settings, read: ReadResult,
          strategies: int, details: bool) -> Dict[str, Any]:
    x, unit, n = st.pnl, st.unit, m["n"]
    source = settings.source or read.source_hint
    versions = settings.variants_tried
    family = max(versions or 1, strategies)
    first, last = (st.times[0], st.times[-1]) if st.times else (None, None)
    card: Dict[str, Any] = {"name": st.name, "unit": unit, "n": n, "first": iso_day(first), "last": iso_day(last),
                            "fingerprint": _fingerprint_of(st), "warnings": [], "chart": None}
    checks: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    if n == 0:
        card.update(verdict=_verdict("too_early", "No trades to judge.", "We found no closed trades."),
                    checks=[], ask_the_user=[], what_would_change_it=[], next_step=None, numbers=None)
        return card
    mean, total, sd_eff = m["mean"], m["total"], m["sd_eff"]
    wins, losses = x[x > 0], x[x < 0]
    win_rate = len(wins) / n
    worst_dd, current_dd, since_peak = _drawdown(x)
    m["max_dd"] = worst_dd
    pf = float(wins.sum() / -losses.sum()) if len(losses) and len(wins) else None
    bar = m["family_bar"]
    passes_up = up and mean > 0
    e_up = math.exp(min(m["log_e_up"], 700))
    single_symbol = (st.symbols is None and read.format in ("tradingview", "backtesting.py", "vectorbt")) or (
        st.symbols is not None and len(set(st.symbols)) == 1)
    market = _market(st, settings, single_symbol)
    market_adjusted = None
    if market and market.get("market_part") is not None and n >= MIN_TRADES:
        excess = x - market["market_part"]
        mm = _measure(Trades(st.name, unit, excess), bar)
        market_adjusted = {"mean": mm["mean"], "range": [mm.get("adj_lower"), mm.get("adj_upper")],
                           "market_part_per_trade": float(np.mean(market["market_part"])),
                           "passes": mm["log_e_up"] >= math.log(bar) and mm["mean"] > 0,
                           "e_value": math.exp(min(mm["log_e_up"], 700))}

    # recent trades: broken, or a bad streak?
    recent = None
    stretch = None
    if n >= 2 * RECENT and details:
        earlier, latest = x[:-RECENT], x[-RECENT:]
        stretch = _stretch(earlier, latest, "stretch30")
        recent = _recent_card(stretch, earlier, latest, unit, st)
    faded, faded_window, faded_p = False, None, None
    if n >= 2 * MIN_TRADES and (passes_up or _passes_alone(x[:-RECENT], bar)):
        for w in FADED_WINDOWS:
            if n - w < max(MIN_TRADES, w):
                continue
            latest = x[-w:]
            if latest.mean() > 0 or not _passes_alone(x[:-w], bar) and not passes_up:
                continue
            s = stretch if (w == RECENT and stretch is not None) else _stretch(x[:-w], latest, f"stretch{w}")
            if s["p_low"] < FADED_P and (faded_p is None or s["p_low"] < faded_p):
                faded, faded_window, faded_p = True, w, s["p_low"]

    # the verdict
    downgraded = few_losses = False
    if n < MIN_TRADES:
        code = "too_early"
    elif faded:
        code = "faded"
    elif passes_up and len(losses) < MIN_LOSSES:
        code, few_losses = "cant_tell", True
    elif passes_up:
        code = "real_edge"
        if market_adjusted is not None and not market_adjusted["passes"]:
            code, downgraded = ("cant_tell" if market_adjusted["mean"] > 0 else "no_edge"), True
    elif down and mean < 0:
        code = "losing"
    elif mean > 0:
        code = "cant_tell"
        if market_adjusted is not None and market_adjusted["mean"] <= 0:
            code, downgraded = "no_edge", True
        elif len(losses) < MIN_LOSSES:
            few_losses = True
    else:
        code = "no_edge"
    needed = None
    if code == "cant_tell" and not downgraded and not few_losses:
        needed = trades_needed(m["record"], bar, m["rho_family"])
        if needed is not None and needed > max(5000, 50 * n):
            needed = None                       # out of reach: say so instead of a huge number
        elif needed is not None and needed >= 100:
            needed = int(float(f"{needed:.2g}"))  # an estimate: two significant figures
    survives = max_versions_passing(m["record"], strategies=strategies) if code == "real_edge" else None

    # money first, percent second
    pct = st.pct if unit != "%" else None
    shadow = getattr(st, "money_shadow", None)
    money_card = _money_card(st, x, unit, pct, shadow)

    # the verdict in words
    scale = max(abs(m.get("adj_lower") or 0.0), abs(m.get("adj_upper") or 0.0), abs(mean))
    rng_text = f"{fmt(m.get('adj_lower'), unit, ref=scale)} to {fmt(m.get('adj_upper'), unit, ref=scale)}"
    per = f"{fmt(mean, unit)} per trade"
    judged_all = strategies == len(read.strategies)
    tried = (f" when you tried {versions} versions" if versions and versions > 1 else
             (f" across your {strategies} strategies" if judged_all else
              f" across your {strategies} strategies with {MIN_TRADES} or more trades") if strategies > 1 else "")
    only_version = code == "real_edge" and versions is None and strategies == 1
    split_day = day(st.times[-faded_window]) if (faded and st.times) else None
    if code == "too_early":
        verdict = _verdict(code, f"Too early to judge: {count(n, 'trade')}. A verdict needs at least {MIN_TRADES}.",
                           f"So far: {per} over {count(n, 'trade')}. With so few trades, luck decides almost "
                           f"everything. Come back after {MIN_TRADES - n} more.")
    elif code == "faded":
        when = f"until {split_day}" if split_day else f"for its first {n - faded_window} trades"
        latest_mean = float(np.mean(x[-faded_window:]))
        verdict = _verdict(code, f"It worked {when}, but the edge is gone: the last {faded_window} trades average "
                                 f"{fmt(latest_mean, unit)}.",
                           f"Earlier it made {fmt(float(np.mean(x[:-faded_window])), unit)} per trade, and that was "
                           f"more than luck. The last {faded_window} trades average {fmt(latest_mean, unit)}. If "
                           f"nothing had changed, a stretch this bad would happen {one_in(faded_p, STREAK_DRAWS)} "
                           "times. So something likely changed.")
    elif code == "real_edge":
        head = f"This looks like a real edge: {per}, more than luck explains"
        head += " (if this is the only version you tried)." if only_version else f"{tried}." if tried else "."
        plain = (f"Over {count(n, 'trade')} it made {per} on average. Luck alone rarely makes a record this strong"
                 f"{tried}. The likely range is {rng_text} per trade, all above zero.")
        if only_version:
            plain += " This assumes it is the only version you tried."
        plain += " Past results can still stop working, so keep checking new trades."
        verdict = _verdict(code, head, plain)
    elif code == "cant_tell":
        if few_losses:
            lost = len(losses)
            few = "no losing trades" if lost == 0 else f"only {count(lost, 'losing trade')}"
            head = f"Can't tell yet: {per}, but {few}, so nobody can tell how big a loss gets."
            plain = (f"It won {len(wins)} of {count(n, 'trade')}, {per} on average. Strategies that win small and "
                     "often can hide one huge loss, and with so few losses its size is unknown. A verdict needs "
                     f"at least {MIN_LOSSES} losing trades.")
        elif downgraded and market is not None:
            head = (f"Can't tell yet: most of the {per} looks like the market rising "
                    f"({fmt(market['hold_pct'], '%')} over the same dates).")
            plain = (f"It made {per}, but the price itself rose {fmt(market['hold_pct'], '%')} over the same dates. "
                     f"Taking out what the market gave each trade leaves about "
                     f"{fmt(market_adjusted['mean'], unit)} per trade, and that is not more than luck. This is a "
                     "rough check: it assumes the price rose at a steady pace.")
        else:
            head = f"Can't tell yet: {per} so far, but luck often makes records like this{tried}."
            plain = (f"It made {per} over {count(n, 'trade')}. That looks good, but luck alone makes a record this "
                     f"good too often{tried}. The likely range is {rng_text} per trade, and that includes zero.")
            if needed is not None and needed > 0:
                plain += f" About {needed:,} more trades at this average would settle it."
            elif needed is None:
                plain += " At this average it would take many thousands of trades to tell it apart from luck."
        verdict = _verdict(code, head, plain)
    elif code == "losing":
        verdict = _verdict(code, f"Losing for real: {per}, too steady to be bad luck.",
                           f"Over {count(n, 'trade')} it lost {fmt(-mean, unit, False)} per trade on average. The "
                           f"likely range is {rng_text} per trade, all below zero. Bad luck does not look this "
                           "steady: something in the rules or the costs works against you.")
    else:
        if downgraded and market is not None:
            head = (f"No edge once the market is taken out: the price rose {fmt(market['hold_pct'], '%')} over "
                    "the same dates.")
            plain = (f"It made {per}, but holding the asset made {fmt(market['hold_pct'], '%')} over the same dates. "
                     f"Taking out what the market gave each trade leaves {fmt(market_adjusted['mean'], unit)} per "
                     "trade. This is a rough check: it assumes the price rose at a steady pace.")
        else:
            head = f"No edge: {per} on average."
            plain = (f"Over {count(n, 'trade')} it averaged {per}. That is what luck alone does. More trades at this "
                     "average will not turn it into an edge.")
        verdict = _verdict(code, head, plain)
    if downgraded:
        verdict["why"] = ("A rising market lifts every strategy that buys. Only what is left after the market is the "
                          "strategy's own edge.")
    if money_card["disagree"]:
        verdict["plain"] = money_card["sentence"] + " " + verdict["plain"]

    # numbers for the assistant
    if details:
        coin_above, fan = _coin_flips(x, fan=settings.include_chart) if n >= 2 else (0, [])
    else:                       # validation runs: the count from the normal approximation, no pictures
        z_obs = total / math.sqrt(max(float(np.dot(x, x)), 1e-300))
        coin_above, fan = int(round(COIN_FLIPS * normal_tail(z_obs))), []
    z_bar = (z_equivalent(n, bar, m["rho_family"]) + skew_shift(m["skew"], n, bar, m["rho_family"], +1)
             / math.sqrt(n)) if n >= 2 else math.inf
    needed_share = normal_tail(z_bar)
    coin = {"copies": COIN_FLIPS, "ended_higher": coin_above, "share": coin_above / COIN_FLIPS,
            "needed_at_most": int(math.floor(COIN_FLIPS * needed_share)),
            "needed_one_in": int(round(1.0 / needed_share)) if needed_share > 0 else None}
    coin["sentence"] = _coin_sentence(coin, code, mean, versions, strategies)
    if downgraded:
        coin["sentence"] += (" Coin-flip copies keep the market's rise inside every trade, so they can't tell skill "
                             "from a rising market; the market check above can.")
    top3 = None
    if n >= 4 and total > 0:
        best = np.sort(x)[-3:]
        top3 = {"share_of_profit": float(best.sum() / total), "mean_without": float((total - best.sum()) / (n - 3))}
    fees = None
    if st.fees is not None and np.isfinite(st.fees).any():
        paid = float(np.nansum(st.fees))
        gross = total + paid
        fees = {"paid": paid, "share_of_gross": (paid / gross) if gross > 0 else None}
    dd_odds = _drawdown_odds(x, current_dd) if (details and current_dd > 0 and n >= MIN_TRADES) else None
    drawdown = _drawdown_card(current_dd, since_peak, worst_dd, dd_odds, unit)
    lo_w, hi_w = wilson_bounds_count(len(wins), n)
    numbers: Dict[str, Any] = {
        "trades": n, "unit": unit, "first_trade": iso_day(first), "last_trade": iso_day(last),
        "total": total, "mean_per_trade": {"value": mean, "range_95": [m.get("lower"), m.get("upper")],
                                           "range_kind": "anytime-valid 95%"},
        "percent_per_trade": ({"mean": float(np.mean(pct))} if pct is not None else None),
        "win_rate": {"value": win_rate, "range_95": [lo_w, hi_w]},
        "profit_factor": pf, "best_trade": float(x.max()), "worst_trade": float(x.min()),
        "fees": fees, "top_3": top3,
        "drawdown": {"max": worst_dd, "current": current_dd, "trades_since_peak": since_peak,
                     "odds_pct": dd_odds},
        "coin_flip": {k: v for k, v in coin.items() if k != "sentence"},
        "evidence": {"e_value": e_up if n >= MIN_TRADES else None,
                     "e_value_below": math.exp(min(m["log_e_down"], 700)) if n >= MIN_TRADES else None,
                     "bar": bar, "passes": bool(passes_up), "passes_below": bool(down and mean < 0),
                     "versions_or_strategies": family, "serial_factor": m["serial_factor"],
                     "skew": m["skew"]},
        "adjusted": ({"versions_tried": versions, "strategies": strategies,
                      "range": [m.get("adj_lower"), m.get("adj_upper")], "bar": bar, "passes": bool(passes_up)}
                     if family > 1 else None),
        "benchmark": None, "market_adjusted": market_adjusted,
        "trades_needed": needed, "versions_it_survives": survives,
    }
    benchmark = _benchmark_card(market, market_adjusted, unit, code, downgraded) if market else None
    if benchmark is not None:
        numbers["benchmark"] = {k: benchmark[k] for k in ("hold_pct", "hold_source", "strategy_pct",
                                                           "strategy_source", "beat", "time_in_market")}

    # checks, with a plain why
    flags = _red_flags(st, m)
    checks += _checks(st, read, settings, n, code, mean, unit, fees, top3, drawdown, recent, money_card, benchmark,
                      market_adjusted, downgraded, versions, strategies, source, flags)
    # questions worth asking
    questions += _questions(st, read, settings, code, mean, n, unit, benchmark, versions, strategies, source)
    # what would change it
    changes += _changes(code, n, mean, unit, m, needed, survives, versions, settings, benchmark, faded_window,
                        market_adjusted, total, few_losses)
    positive_backtest = code == "real_edge" and source != "live"
    audit = _audit_prompt(flags, st, positive_backtest, losing=code in ("losing", "no_edge"))
    next_step = _next_step(code, n, needed, settings, source, flags, versions, strategies, fees, read, split_day,
                           few_losses, market_explains=downgraded)
    card.update(
        verdict=verdict, next_step=next_step, numbers=rnd(numbers), checks=checks, ask_the_user=questions,
        what_would_change_it=changes, ai_audit_prompt=audit, red_flags=[f["code"] for f in flags],
        money=rnd(money_card), benchmark=rnd(benchmark), compare=None,
        edge={"mean": rnd(mean), "lower": rnd(m.get("lower")), "upper": rnd(m.get("upper")),
              "adjusted_lower": rnd(m.get("adj_lower")), "adjusted_upper": rnd(m.get("adj_upper"))},
        luck={"one_in": int(min(e_up, 1e9)) if (n >= MIN_TRADES and mean > 0 and e_up >= 2) else None,
              "coin_flips": COIN_FLIPS, "coin_flips_above": coin_above, "sentence": coin["sentence"]},
        stats=rnd({"win_rate": win_rate, "profit_factor": pf, "total": total, "best": float(x.max()),
                   "worst": float(x.min()), "max_drawdown": worst_dd}),
        drawdown=rnd(drawdown), recent=rnd(recent), trades_needed=needed,
    )
    if details and settings.include_chart:
        curve = np.concatenate(([0.0], np.cumsum(x)))
        card["chart"] = {"you": round_list(_downsample(curve)), "fan": [round_list(_downsample(p)) for p in fan]}
    return card


def _recent_card(s: Dict[str, float], earlier: np.ndarray, latest: np.ndarray, unit: str,
                 st: Trades) -> Dict[str, Any]:
    w = len(latest)
    p_bad = min(1.0, 2.0 * min(s["p_low"], s["p_fall"]))
    if s["p_high"] < 0.02:
        reading = "hot"
    elif s["sum"] >= 0 and s["fall"] <= 0:
        reading = "in_line"
    elif p_bad >= 0.10:
        reading = "normal"
    elif p_bad >= 0.02:
        reading = "bad_streak"
    elif p_bad >= 0.002:
        reading = "unusual"
    else:
        reading = "likely_broken"
    head = (f"Your last {w} trades: {fmt(s['sum'], unit)} in total"
            + (f", with a drop of {fmt(s['fall'], unit, False)} inside them." if s["fall"] > 0 else "."))
    odds = one_in(s["p_fall"], STREAK_DRAWS)
    tail = {
        "hot": f"That is better than usual for this strategy ({one_in(s['p_high'], STREAK_DRAWS)}). Streaks like "
               "this usually cool off.",
        "in_line": "That is in line with your earlier trades.",
        "normal": (f"If nothing changed, a drop that deep within {w} trades happens {odds} times for this strategy. "
                   "That is normal for it." if s["fall"] > 0 else "That is normal for this strategy."),
        "bad_streak": f"If nothing changed, a drop that deep within {w} trades happens {odds} times for this "
                      "strategy. That is a bad streak, not proof that it broke.",
        "unusual": f"If nothing changed, a drop that deep within {w} trades happens only {odds} times for this "
                   "strategy. That is unusual: check whether something changed.",
        "likely_broken": f"If nothing changed, a drop that deep within {w} trades happens only {odds} times for "
                         "this strategy. Something has likely changed.",
    }[reading]
    return {"n": w, "window": w, "total": s["sum"], "mean": float(np.mean(latest)), "earlier_mean":
            float(np.mean(earlier)), "deepest_drop": s["fall"], "odds_drop": s["p_fall"], "odds_total_this_low":
            s["p_low"], "reading": reading, "sentence": f"{head} {tail}",
            "why": "A losing stretch feels like the strategy broke. Often it is just a normal streak, and the odds "
                   "tell which."}


def _drawdown_card(depth: float, since: int, worst: float, odds: Optional[float], unit: str) -> Dict[str, Any]:
    if depth <= 0:
        return {"depth": 0.0, "max": worst, "trades_since_peak": 0, "happens_pct": None, "reading": "normal",
                "sentence": "You are at a new high."}
    reading = "normal" if odds is None or odds >= 10 else "unusual" if odds >= 2 else "alarming"
    where = f"You are {fmt(depth, unit, False)} below your peak, {count(since, 'trade')} ago"
    if odds is None:
        sentence = f"{where}."
    elif odds >= 99.5:
        sentence = f"{where}. That is an everyday dip for a strategy like yours."
    else:
        meaning = {"normal": "normal", "unusual": "unusual", "alarming": "rare: check whether something changed"}
        sentence = (f"{where}. That is {meaning[reading]}: histories like yours dip this deep {odds:.0f}% of the "
                    "time.")
    return {"depth": depth, "max": worst, "trades_since_peak": since, "happens_pct": odds, "reading": reading,
            "sentence": sentence}


def _clearly_nonzero(v: Optional[np.ndarray]) -> bool:
    """An average at least one standard error from zero: worth naming when
    percent and money point different ways (two near-zero numbers don't)."""
    if v is None:
        return False
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return False
    se = float(np.std(v, ddof=1)) / math.sqrt(len(v))
    return abs(float(np.mean(v))) >= max(se, 1e-12)


def _money_card(st: Trades, x: np.ndarray, unit: str, pct: Optional[np.ndarray],
                shadow: Optional[np.ndarray]) -> Dict[str, Any]:
    total = float(x.sum())
    if unit != "%":
        pmean = float(np.mean(pct)) if pct is not None else None
        disagree = (pmean is not None and (pmean > 0) != (total > 0)
                    and _clearly_nonzero(pct) and _clearly_nonzero(x))
        if disagree and total < 0:
            sentence = (f"Your average trade is {fmt(pmean, '%')}, but you lost {fmt(-total, unit, False)} in total: "
                        "your losing trades were bigger than your winning ones.")
        elif disagree:
            sentence = (f"Your average trade is {fmt(pmean, '%')}, but you made {fmt(total, unit, False)} in total: "
                        "your winning trades were the bigger ones.")
        else:
            sentence = f"In total: {fmt(total, unit)}" + (f" (average trade {fmt(pmean, '%')})." if pmean is not None
                                                         else ".")
        return {"total": total, "unit": unit, "percent_mean": pmean, "disagree": bool(disagree),
                "sentence": sentence, "judged_in": unit,
                "why": "Money is what you keep. A good average percent can hide a loss when the losing trades are "
                       "the big ones."}
    pmean = float(np.mean(x))
    if shadow is not None and np.isfinite(shadow).any():
        mt = float(np.nansum(shadow))
        disagree = (pmean > 0) != (mt > 0) and _clearly_nonzero(x) and _clearly_nonzero(shadow)
        sentence = (f"Your average trade is {fmt(pmean, '%')}, but in money the trades add up to {fmt(mt, 'money')}: "
                    f"your {'losing' if mt < 0 else 'winning'} trades were the bigger ones." if disagree else
                    f"In total: {fmt(mt, 'money')} (average trade {fmt(pmean, '%')}).")
        return {"total": mt, "unit": "money", "percent_mean": pmean, "disagree": bool(disagree),
                "sentence": sentence, "judged_in": "%",
                "why": "Money is what you keep. A good average percent can hide a loss when the losing trades are "
                       "the big ones."}
    return {"total": None, "unit": "%", "percent_mean": pmean, "disagree": False, "judged_in": "%",
            "sentence": "Your results are in percent per trade. We can't see the money amounts.",
            "why": "Money is what you keep. Percent per trade can hide a loss when the losing trades are bigger."}


def _benchmark_card(market: Dict[str, Any], adjusted: Optional[Dict[str, Any]], unit: str, code: str,
                    downgraded: bool) -> Optional[Dict[str, Any]]:
    hold, strat = market.get("hold_pct"), market.get("strategy_pct")
    exposure = market.get("time_in_market")
    beat = None if (hold is None or strat is None) else bool(strat >= hold)
    parts = []
    if hold is not None:
        parts.append(f"Simply holding made {fmt(hold, '%')} over the same dates")
    if strat is not None:
        parts.append(f"the strategy made about {fmt(strat, '%')}")
    sentence = (", and ".join(parts) + ".") if parts else ""
    if exposure is not None:
        sentence += f" It was in the market {share(exposure)} of the time."
    if adjusted is not None:
        sentence += (f" Taking out what the market gave each trade leaves {fmt(adjusted['mean'], unit)} per trade"
                     + (", still more than luck." if adjusted["passes"] else ", not more than luck." if
                        adjusted["mean"] > 0 else "."))
    sentence = sentence.strip() + " (Rough check.)"
    return {"hold_pct": hold, "hold_source": market.get("hold_source"), "strategy_pct": strat,
            "strategy_source": market.get("strategy_source"), "beat": beat, "time_in_market": exposure,
            "market_adjusted": adjusted, "downgraded": downgraded, "sentence": sentence, "rough": True,
            "why": "A strategy that only buys looks good whenever prices rise. The question is whether it did "
                   "better than simply holding."}


def _coin_sentence(coin: Dict[str, Any], code: str, mean: float, versions: Optional[int], strategies: int) -> str:
    above, copies = coin["ended_higher"], coin["copies"]
    base = f"In {copies:,} coin-flip copies of your own trades, {above:,} ended higher than you."
    if mean <= 0:
        return base + " Luck alone did as well as this record."
    if coin.get("needed_one_in") is None:
        return base + f" With so few trades, no result could pass yet."
    reason = ("you can check again at any time" + (f" and you tried {versions} versions" if versions and versions > 1
                                                   else f" and you have {strategies} strategies" if strategies > 1
                                                   else ""))
    rare = coin["needed_one_in"]
    rare = int(float(f"{rare:.2g}")) if rare >= 100 else rare
    # The coin flips are a picture; the verdict rests on the evidence test, which also allows for checking again
    # and for the versions tried. The two can differ by a few copies (found by tests/test_bridge_counting.py: a
    # real_edge card read "3 ended higher ... a pass needs 1 or fewer ... It passes"), so the bar is given as
    # odds, never as a count of copies.
    if code in ("real_edge",):
        return base + (f" The verdict uses a stricter test than this picture: luck would need odds rarer than about "
                       f"1 in {rare:,}, because {reason}. It passes.")
    return base + (f" That alone might look convincing, but the verdict uses a stricter test: luck would need odds "
                   f"rarer than about 1 in {rare:,}, because {reason}.")


def _checks(st, read, settings, n, code, mean, unit, fees, top3, drawdown, recent, money_card, benchmark,
            market_adjusted, downgraded, versions, strategies, source, flags) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if n < MIN_TRADES:
        out.append(_check("enough_trades", "fail", f"Only {count(n, 'trade')}. A verdict needs at least {MIN_TRADES}.",
                          "With few trades, a lucky run looks the same as a real edge.",
                          f"Collect {MIN_TRADES - n} more trades (paper trading is fine), then check again."))
    elif n < FEW_TRADES:
        out.append(_check("enough_trades", "warn", f"{n} trades: enough for a verdict, but only a big edge can show.",
                          "Small edges need hundreds of trades to tell apart from luck.",
                          "Keep collecting trades and check again."))
    else:
        out.append(_check("enough_trades", "pass", f"{n:,} trades.",
                          "More trades make luck easier to tell apart from a real edge."))
    if mean > 0 and n >= MIN_TRADES:
        if versions:
            out.append(_check("versions_tried", "pass", f"Counted the {versions} versions you tried.",
                              "Trying many versions and keeping the best one makes luck look like skill. The bar is "
                              "raised to match."))
        elif strategies > 1:
            out.append(_check("versions_tried", "warn", f"Judged across your {strategies} strategies, but we don't "
                                                        "know how many versions you tried in total.",
                              "Each version you tried is another chance for luck to look good.",
                              "Tell us how many versions or settings you tried (variants_tried)."))
        else:
            out.append(_check("versions_tried", "warn", "We don't know how many versions you tried before this one.",
                              "If you tried 40 versions, one of them looks good by luck alone, so the bar for "
                              "'real' must be higher.",
                              "Tell us how many versions or settings you tried (variants_tried)."))
    fees_known = settings.fees_included if settings.fees_included is not None else st.fees_included
    if fees is not None and fees["paid"] > 0:
        share_ = fees.get("share_of_gross")
        if share_ is not None and share_ > 0.25:
            gross = fees["paid"] / share_
            text = (f"Fees ({fmt(fees['paid'], unit, False)}) were more than everything the trades made before fees "
                    f"({fmt(gross, unit, False)})." if share_ >= 1 else
                    f"Fees took {share(share_)} of what the trades made before fees.")
            out.append(_check("fees", "warn", text,
                              "When fees eat a big part of the profit, a small change in costs can wipe it out.",
                              "Look for fewer, larger trades, or lower fees (maker orders, fee discounts)."))
        else:
            out.append(_check("fees", "pass", f"Fees are in the file ({fmt(fees['paid'], unit, False)} paid).",
                              "Results without fees always look better than real trading."))
    elif fees_known is True:
        out.append(_check("fees", "pass", "Fees are included in the results.",
                          "Results without fees always look better than real trading."))
    elif fees_known is False:
        out.append(_check("fees", "fail", "These results are before fees.",
                          "Every trade pays fees twice, on the way in and out. A small edge before fees is often a "
                          "loss after them.",
                          "Re-run the backtest with fees and slippage turned on (about 0.1% per side for spot "
                          "crypto)."))
    elif n >= MIN_TRADES:
        out.append(_check("fees", "warn", "We can't see whether fees are included.",
                          "Every trade pays fees twice, on the way in and out. A small edge before fees is often a "
                          "loss after them.",
                          "Check the strategy's commission setting, or tell us (fees_included)."))
    n_losses = int((st.pnl < 0).sum())
    if n >= MIN_TRADES and mean > 0 and n_losses < MIN_LOSSES:
        out.append(_check("few_losses", "warn", (f"No losing trades in {count(n, 'trade')}." if n_losses == 0 else
                                                  f"Only {count(n_losses, 'losing trade')} in {count(n, 'trade')}."),
                          "Strategies that win small and often can hide one huge loss. Until a few losses have "
                          "happened, nobody can tell how big they get.",
                          "Keep trading it on paper until it has had a few losses, then check again."))
    if money_card["disagree"]:
        out.append(_check("percent_vs_money", "warn", money_card["sentence"], money_card["why"],
                          "Look at the size of each trade: the losses come from the big positions."))
    if top3 is not None and n >= MIN_TRADES and top3["mean_without"] <= 0:
        out.append(_check("few_big_trades", "warn",
                          f"Without your 3 best trades, the average is {fmt(top3['mean_without'], unit)} per trade.",
                          "When a few trades carry the whole result, the next year may not have trades like them.",
                          "Look at what made those 3 trades special, and whether it can repeat."))
    if benchmark is not None:
        if downgraded:
            out.append(_check("beats_the_market", "fail", benchmark["sentence"], benchmark["why"],
                              "Compare it with simply holding, and test it in a period when prices fell."))
        elif benchmark.get("beat") is False and (benchmark.get("hold_pct") or 0) > 0:
            out.append(_check("beats_the_market", "warn", benchmark["sentence"], benchmark["why"],
                              "Test it in a period when prices fell. A real edge should not need a rising market."))
        elif benchmark.get("hold_pct") is not None:
            out.append(_check("beats_the_market", "pass", benchmark["sentence"], benchmark["why"]))
    elif mean > 0 and n >= MIN_TRADES and (st.side is None or np.nanmean(st.side) > 0):
        out.append(_check("beats_the_market", "warn", "Not compared with simply holding the asset.",
                          "A strategy that only buys looks good whenever prices rise.",
                          "Tell us what holding returned over the same dates (benchmark_return_pct)."))
    if recent is not None:
        status = {"in_line": "pass", "normal": "pass", "hot": "warn", "bad_streak": "warn", "unusual": "warn",
                  "likely_broken": "fail"}[recent["reading"]]
        fix = None if status == "pass" else ("Don't expect the streak to last." if recent["reading"] == "hot" else
                                             "Check what changed: the market, the exchange, the settings or the code.")
        out.append(_check("recent_trades", status, recent["sentence"], recent["why"], fix))
    if drawdown and drawdown.get("reading") == "alarming":
        out.append(_check("drawdown_now", "fail", drawdown["sentence"],
                          "A drop this deep is rare for this strategy's own history.",
                          "Check whether something changed before running it further."))
    if st.entry_times and st.times:
        overlap = _overlap_share(st.entry_times, st.times)
        if overlap is not None and overlap > 0.5:
            out.append(_check("overlapping_trades", "warn",
                              f"{share(overlap)} of trades were open at the same time as another trade.",
                              "Trades open at the same time often win or lose together, so they count less than "
                              "separate bets. The verdict may be too kind.",
                              "Judge one position at a time, or count overlapping trades as one."))
    if st.times and n >= MIN_TRADES:
        span = (st.times[-1] - st.times[0]).total_seconds() / 86400
        if span < 30:
            out.append(_check("short_period", "warn", f"All trades are from {max(1, round(span))} day(s).",
                              "A short period is one kind of market. The strategy may only work in that kind.",
                              "Test it on a longer period, with both rising and falling prices."))
    if source == "backtest":
        out.append(_check("backtest_only", "warn", "These are backtest results.",
                          "Backtests are usually better than live trading: real fills are worse, and the rules were "
                          "tuned on this same data.",
                          "Paper-trade it and compare the new trades with the backtest."))
    elif source == "live":
        out.append(_check("live_results", "pass", "These are live results.",
                          "Live trades include real fills and fees, so they are the most honest record."))
    for flag in flags:
        out.append(_check(f"red_flag:{flag['code']}", "warn", flag["text"], flag["why"],
                          "Ask your AI to audit the code with the audit prompt below."))
    for w in read.warnings:
        out.append(_check("data", "warn", w, _data_why(w)))
    return out


def _data_why(warning: str) -> str:
    w = warning.lower()
    if "fee" in w:
        return "Fees that are not counted make the results look better than they were."
    if "no earlier buy" in w:
        return "A sell without its buy has no known cost, so it can't be scored."
    if "still open" in w:
        return "An open position has no final result yet."
    if "quoted in" in w:
        return "Results in different coins can't be added up."
    if "still open" in w or "open" in w:
        return "An open trade has no final result yet."
    return "Rows we could not read are not in the verdict."


def _questions(st, read, settings, code, mean, n, unit, benchmark, versions, strategies,
               source) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    positive = mean > 0 and n >= MIN_TRADES
    if positive and not versions:
        out.append(_question("variants_tried", "variants_tried",
                             "How many versions or settings of this strategy did you try before this one? A rough "
                             "number is fine.",
                             "If you tried 40 versions, one of them looks good by luck alone, so the bar for 'real' "
                             "must be higher.", "variants_tried=<number>",
                             "high" if code in ("real_edge", "cant_tell") else "medium"))
    fees_known = settings.fees_included if settings.fees_included is not None else st.fees_included
    if fees_known is None and (st.fees is None or not np.isfinite(st.fees).any()) and n >= MIN_TRADES:
        out.append(_question("fees_included", "fees_included", "Do these results include trading fees and slippage?",
                             "A small edge before fees is often a loss after them.", "fees_included=true|false",
                             "high" if positive else "low"))
    if positive and benchmark is None and (st.side is None or np.nanmean(st.side) > 0):
        out.append(_question("benchmark", "benchmark_return_pct",
                             "Over the same dates, what would simply buying and holding the asset have returned, "
                             "in percent?",
                             "A strategy that only buys looks good whenever prices rise. This tells us if it beat "
                             "simply holding.", "benchmark_return_pct=<percent>", "medium"))
    if source is None and n >= MIN_TRADES:
        out.append(_question("source", "source", "Are these backtest, paper-trading or live (real money) results?",
                             "Backtests are usually better than live trading, so the next step depends on it.",
                             "source=backtest|paper|live", "medium" if positive else "low"))
    if st.unit == "money" and read.format == "pnl_list":
        out.append(_question("unit", "unit", "Are these numbers money (which currency?) or percent per trade?",
                             "Money and percent can tell different stories when trade sizes differ.",
                             "unit=USD or unit=%", "low"))
    if not st.times and read.format == "pnl_list" and n >= 2 * RECENT:
        out.append(_question("order", None, "Is the first number the oldest trade?",
                             "We read the list as oldest first. The recent-trades check depends on the order.",
                             "pass the list oldest first", "low"))
    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda q: rank[q["impact"]])
    return out


def _changes(code, n, mean, unit, m, needed, survives, versions, settings, benchmark, faded_window,
             market_adjusted, total, few_losses=False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if code == "too_early":
        out.append({"code": "more_trades", "text": f"At least {MIN_TRADES - n} more trades before any verdict."})
    if code == "cant_tell" and few_losses:
        out.append({"code": "more_losses", "text": f"At least {MIN_LOSSES} losing trades. Their size decides "
                                                   "whether the wins are worth it."})
    elif code == "cant_tell":
        if needed:
            out.append({"code": "more_trades", "text": f"About {needed:,} more trades at this average "
                                                       f"({fmt(mean, unit)} each) would settle it."})
        elif market_adjusted is None:
            out.append({"code": "more_trades", "text": "At this average it would take many thousands of trades to "
                                                       "prove it: the edge is small next to the swings."})
        if versions and versions > 1:
            out.append({"code": "fewer_versions", "text": "Fresh trades the strategy was never tuned on count fully: "
                                                          "test it on a new period and judge those trades alone "
                                                          "(variants_tried=1)."})
    if code == "real_edge":
        if survives is not None:
            if versions:
                out.append({"code": "versions", "text": f"It would still pass if you tried up to {survives:,} "
                                                        "versions."})
            else:
                out.append({"code": "versions", "text": f"If you tried more than {survives:,} versions, it would no "
                                                        "longer count as real."})
        lower = m.get("adj_lower")
        if lower is not None and lower > 0:
            out.append({"code": "costs", "text": f"Extra costs above about {fmt(lower, unit, False)} per trade (fees or "
                                                 "slippage not in the results) would make it unprovable; above "
                                                 f"{fmt(mean, unit, False)} per trade they would erase the average."})
    if code in ("cant_tell",) and mean > 0:
        out.append({"code": "costs", "text": f"Costs of {fmt(mean, unit, False)} per trade would wipe out the "
                                             "average."})
    market_explains = (market_adjusted is not None and code in ("no_edge", "cant_tell")
                       and not market_adjusted["passes"] and mean > 0)
    if market_explains:
        out.append({"code": "market", "text": "Making money in a period when prices fell or moved sideways would "
                                              "show an edge the market can't explain."})
    elif code in ("no_edge", "losing"):
        out.append({"code": "change_strategy", "text": "More trades at this average won't help. Only a different "
                                                       "strategy would change it."})
    if code == "faded":
        out.append({"code": "recover", "text": f"If the next {faded_window} trades return to the earlier average, the "
                                               "bad stretch was noise."})
    if benchmark is None and mean > 0 and code in ("real_edge", "cant_tell"):
        out.append({"code": "market", "text": "If simply holding the asset made as much over the same dates, the "
                                              "'edge' may just be the market rising."})
    return out


def _next_step(code, n, needed, settings, source, flags, versions, strategies, fees, read,
               split_day, few_losses=False, market_explains=False) -> Dict[str, str]:
    fees_known = settings.fees_included
    if market_explains and not flags:
        return {"text": "Test it on a period when the price fell or moved sideways, and compare it with simply "
                        "holding.",
                "why": "A rising market lifts every strategy that buys. Only what is left after the market is the "
                       "strategy's own."}
    loss_shape = {"small_wins_big_losses", "averaging_down"}
    flag_codes = {f["code"] for f in flags}
    if flags and (code in ("losing", "no_edge") and flag_codes & loss_shape or flag_codes <= loss_shape):
        return {"text": "Ask your AI to check how the bot handles losing trades: the stop, averaging down and "
                        "position size. The audit prompt lists what to look for.",
                "why": "A few big losses are erasing many small wins. That comes from how losing trades are "
                       "handled, and it can be fixed."}
    if flags and code not in ("losing", "no_edge"):
        return {"text": "Before trusting this, ask your AI to audit the bot's code with the audit prompt, then "
                        "re-check the new trades.",
                "why": "Records this good are usually a bug in the test, not money waiting to be made."}
    if fees_known is False and code in ("real_edge", "cant_tell"):
        return {"text": "Re-run the backtest with fees and slippage, then check again.",
                "why": "A small edge before fees is often a loss after them."}
    if code == "cant_tell" and few_losses:
        return {"text": f"Keep it on paper until it has had at least {MIN_LOSSES} losing trades, then check again.",
                "why": "The size of the losses decides whether many small wins are worth it."}
    if code in ("real_edge", "cant_tell") and not versions and strategies == 1 and source != "live":
        return {"text": "Tell your AI how many versions you tried, then check again.",
                "why": "Trying many versions and keeping the best one makes luck look like skill."}
    fee_share = (fees or {}).get("share_of_gross")
    if code in ("no_edge", "losing") and fee_share is not None and fee_share >= 0.5:
        return {"text": "Look at the fees first: they took "
                        + ("more than everything the trades made" if fee_share >= 1 else "half or more of the gross")
                        + ". Check again with fewer trades, maker orders or lower fees.",
                "why": "Before fees these trades made money; the fees decide the result, and fees are the easiest "
                       "part to change."}
    if code == "too_early":
        return {"text": f"Collect at least {MIN_TRADES - n} more trades (paper trading is fine), then check again.",
                "why": "With few trades, luck decides almost everything."}
    if code == "real_edge":
        if source == "live":
            return {"text": "Keep checking every 30 new trades, and compare them with this record.",
                    "why": "Edges fade. Checking often catches it early, and the method allows checking any time."}
        return {"text": "Paper-trade it for 30 to 50 trades, then compare those trades with this backtest.",
                "why": "A backtest can be tuned to its own data. New trades it has never seen are the real test."}
    if code == "cant_tell" and few_losses:
        return {"text": f"Keep it on paper until it has had at least {MIN_LOSSES} losing trades, then check again.",
                "why": "The size of the losses decides whether many small wins are worth it."}
    if code == "cant_tell":
        if needed:
            return {"text": f"Keep it on paper until about {needed:,} more trades, then check again.",
                    "why": "Paper trades cost nothing and add the evidence this record is missing."}
        return {"text": "Test the idea on a different period or market before spending more time on it.",
                "why": "At this size the edge would take thousands of trades to prove, so more of the same data "
                       "won't decide it soon."}
    if code == "faded":
        when = f" around {split_day}" if split_day else ""
        return {"text": f"Find out what changed{when}: the market, the exchange, the settings or the code. Then "
                        "check again.",
                "why": "A strategy that stopped working usually stopped for a reason you can find."}
    if code == "losing":
        return {"text": "Find out why it loses before running it again: ask your AI to check the fees and the entry "
                        "and exit rules.",
                "why": "Steady losses come from the rules or the costs, and those can be found and fixed."}
    return {"text": "This version shows nothing worth more testing. If you change the idea, test the new version on "
                    "fresh data.",
            "why": "Tweaking settings on the same data until it looks good is how luck turns into a fake edge."}


# ── pairs: strategies that are one bet ────────────────────────────────────────

def _pairs(strategies: List[Trades]) -> List[Dict[str, Any]]:
    daily = {}
    for st in strategies:
        if st.times and len(st.pnl) >= MIN_TRADES:
            days = pd.Series(st.pnl, index=pd.DatetimeIndex(st.times).floor("D")).groupby(level=0).sum()
            daily[st.name] = days
    names = list(daily)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            joined = pd.concat([daily[a], daily[b]], axis=1, sort=True).fillna(0.0)
            active = (joined != 0).all(axis=1).sum()
            if len(joined) < MIN_PAIR_DAYS or active < 10:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if not math.isfinite(corr):
                continue
            if corr >= SAME_BET:
                out.append({"a": a, "b": b, "corr": round(corr, 2), "reading": "same_bet",
                            "sentence": f"{a} and {b} win and lose on the same days ({corr:.2f}). They are really "
                                        "one bet.",
                            "why": "Two strategies that move together don't spread your risk. When one loses, so "
                                   "does the other."})
            elif corr <= -SAME_BET:
                out.append({"a": a, "b": b, "corr": round(corr, 2), "reading": "opposite",
                            "sentence": f"{a} and {b} move against each other ({corr:.2f}). Together they cancel "
                                        "out.",
                            "why": "Running both pays fees twice for results that cancel."})
    return out


# ── the whole result ──────────────────────────────────────────────────────────

FOR_THE_ASSISTANT = [
    "Lead with verdict.headline word for word, then verdict.plain (or reply_draft) in your own short words.",
    "Never call 'cant_tell' a win, and never call 'no_edge' almost there. Do not round a verdict up.",
    "Ask the questions in ask_the_user, in order. Each says which parameter to set when you check again.",
    "Don't add your own t-test, Sharpe ratio or p-value as a second opinion: they ignore repeated checking and the "
    "versions tried, which this verdict accounts for.",
    "This is a statistical check of past trades, not financial advice. Don't turn it into buy, sell or position-size "
    "advice; next_step is about testing.",
]

LIMITATIONS = [
    "Trades are treated as independent; neighbouring trades that move together are allowed for (Newey-West), but "
    "trades driven by the same market moves still count more than they should.",
    "The ranges and e-values use a normal approximation with an estimated spread: valid as trades accumulate, not "
    "exactly at every small sample.",
    "The comparison with buying and holding is rough: it assumes the price moved at a steady pace.",
    "It judges the trades it is given. It can't see a bug in the backtest, only signs of one.",
]

HOW_WE_KNOW = [
    "The range is an anytime-valid 95% range (a confidence sequence; Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas "
    "2021; Howard, Ramdas, McAuliffe & Sekhon 2021): from 30 trades on it stays valid however often you check, "
    "unlike a normal confidence interval read every day. Below 30 trades no range is given.",
    "Each strategy gets one-sided normal-mixture e-values for 'the average is above zero' and 'below zero'. Every "
    "version tried and every strategy shown is one family; e-BH (Wang & Ramdas 2022) selects at a 5% false "
    "discovery rate. Versions tried but not shown count as e = 0, which is valid under any dependence. One strategy "
    "alone needs e >= 40; with N versions it needs e >= 40 N, exactly where its Bonferroni-adjusted 95% range clears "
    "zero.",
    "The spread is the sample standard deviation, widened when neighbouring trades move together: by the larger "
    "of a Newey-West long-run variance factor and a bias-corrected AR(1) factor (never narrowed).",
    "When losses are rarer but bigger than wins (negative skew), the bar for 'above zero' is raised by the "
    "Edgeworth term of a studentized mean (Hall 1987), and the range is widened on that side; the same for "
    "'below zero' and positive skew. Skew never lowers a bar.",
    "Coin-flip copies flip the sign of each trade at random: same trades, same sizes, zero edge by construction.",
    "'Faded' needs the earlier trades to pass on their own and the latest 30 or 60 trades to be worse than 99.75% of "
    "stretches resampled from the earlier trades.",
]


def _input_fingerprint(read: ReadResult, backtest: Optional[ReadResult]) -> str:
    """The trades as read: the same trades give the same fingerprint, however
    they were passed (path, bytes, text, table). Names count only when a file
    holds several strategies (then they come from the data itself)."""
    body = []
    for r in (read, backtest):
        if r is None:
            continue
        for st in r.strategies:
            body.append([st.name if len(r.strategies) > 1 else None, st.unit,
                         [round(float(v), 8) for v in st.pnl],
                         [t.strftime("%Y-%m-%dT%H:%M:%S") for t in st.times] if st.times else None])
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def need_input(read: ReadResult, settings: Settings, problem: Optional[str] = None) -> Dict[str, Any]:
    cols = read.columns_seen
    problem = problem or (read.need or {}).get("problem") or "We couldn't find trades to judge."
    needs = ("One row per closed trade with its result (money or %), ideally with the close date. Or one row per "
             "fill with time, pair, side (buy/sell), price, quantity and fee. Or a plain list of results per trade.")
    question = ("Can you export the closed trades? In TradingView: Strategy Tester, then 'List of trades', then "
                "export. On an exchange: spot trade history (fills), as CSV.")
    seen = f" Columns we saw: {', '.join(cols[:20])}." if cols else ""
    return {
        "schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "need_input",
        "input_fingerprint": None,
        "verdict": None, "next_step": {"text": "Send the closed trades in one of the forms below.",
                                       "why": "The judge needs one result per closed trade."},
        "numbers": None, "checks": [], "what_would_change_it": [], "ai_audit_prompt": None,
        "ask_the_user": [_question("export", None, question, "The judge needs one result per closed trade.",
                                   "data=<file path, CSV text or list of results>")],
        "need": {"problem": problem + seen, "needs": needs, "columns_seen": cols, "question_for_trader": question},
        "summary": {"headline": "We couldn't judge this yet.", "text": problem, "counts": {}, "strategies": 0,
                    "trades": 0},
        "strategies": [], "portfolio": {"pairs": []}, "compare": None,
        "input": {"format": read.format, "rows": read.rows, "columns_seen": cols, "columns_used": read.columns_used,
                  "warnings": read.warnings},
        "settings": _settings_dict(settings), "warnings": list(read.warnings) + list(settings.notes),
        "for_the_assistant": ["Tell the trader what is missing (need.problem) and relay need.question_for_trader.",
                              "Don't guess a verdict without the trades."] ,
        "reply_draft": f"I couldn't read trades from that yet: {problem} {question}",
        "how_we_know": [], "limitations": LIMITATIONS,
    }


def _settings_dict(s: Settings) -> Dict[str, Any]:
    return {"variants_tried": s.variants_tried, "benchmark_return_pct": s.benchmark_return_pct,
            "fees_included": s.fees_included, "source": s.source, "unit": s.unit, "kind": s.kind,
            "q": Q, "confidence": 1 - ALPHA, "min_trades": MIN_TRADES, "t_star": T_STAR}


def analyze(read: ReadResult, settings: Settings, *, details: bool = True,
            backtest: Optional[ReadResult] = None) -> Dict[str, Any]:
    """The result for one input (see judge())."""
    if read.need is not None or not read.strategies:
        return need_input(read, settings)
    strategies = read.strategies
    shown = sum(len(st.pnl) >= MIN_TRADES for st in strategies)
    family = max(settings.variants_tried or 1, max(shown, 1))
    bar = BASE_BAR * family
    measured = [_measure(st, bar) for st in strategies]
    log_e: Dict[str, float] = {}
    for i, mm in enumerate(measured):
        if mm["n"] >= MIN_TRADES and "log_e_up" in mm:
            log_e[f"{i}#up"], log_e[f"{i}#down"] = mm["log_e_up"], mm["log_e_down"]
    selected = ebh(log_e, Q, total=2 * family)
    cards = [_card(st, mm, up=f"{i}#up" in selected, down=f"{i}#down" in selected, settings=settings, read=read,
                   strategies=max(shown, 1), details=details)
             for i, (st, mm) in enumerate(zip(strategies, measured))]
    pairs = _pairs(strategies) if len(strategies) > 1 else []
    compare = None
    if backtest is not None:
        compare = compare_runs(read, backtest)
        if len(cards) == 1:
            cards[0]["compare"] = compare
    counts = {code: sum(c["verdict"]["code"] == code for c in cards) for code in VERDICTS}
    lead = max(range(len(cards)), key=lambda i: (cards[i]["verdict"]["code"] == "real_edge",
                                                 measured[i].get("log_e_up", 0.0)))
    single = len(cards) == 1
    lead_card = cards[lead]
    summary = _summary(cards, counts, pairs)
    if single:
        verdict = dict(lead_card["verdict"])
    else:
        verdict = dict(lead_card["verdict"], headline=summary["headline"], plain=summary["text"],
                       about=lead_card["name"])
    checks = lead_card["checks"] if single else _multi_checks(cards, pairs, read)
    questions = lead_card["ask_the_user"] if single else _merge_questions(cards)
    changes = lead_card["what_would_change_it"] if single else [
        dict(c, text=f"{card['name']}: {c['text']}") for card in cards for c in card["what_would_change_it"][:1]]
    next_step = lead_card["next_step"]
    if compare is not None and compare.get("code") == "worse_than_backtest":
        next_step = compare["next_step"]
    audit = next((c["ai_audit_prompt"] for c in cards if c.get("ai_audit_prompt")), None)
    assistant = list(FOR_THE_ASSISTANT)
    if any(c["verdict"]["code"] == "real_edge" for c in cards) and not settings.variants_tried:
        assistant.append("The verdict assumes this was the only version tried. Say so, and ask how many were tried.")
    if audit:
        assistant.append("Offer the ai_audit_prompt: suggest checking the bot's code for the listed bugs first.")
    if compare is not None:
        assistant.append("compare says whether the live trades behave like the backtest promised; mention it.")
    result = {
        "schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "ok",
        "input_fingerprint": _input_fingerprint(read, backtest),
        "verdict": verdict, "next_step": next_step,
        "numbers": lead_card.get("numbers") if single else None,
        "checks": checks, "ask_the_user": questions, "what_would_change_it": changes,
        "ai_audit_prompt": audit, "for_the_assistant": assistant,
        "reply_draft": _reply_draft(verdict, next_step, questions, compare),
        "how_we_know": HOW_WE_KNOW + _how_numbers(cards), "limitations": LIMITATIONS,
        "summary": summary, "strategies": cards, "portfolio": {"pairs": pairs}, "compare": compare,
        "input": {"format": read.format, "rows": read.rows, "read_as": read.summary,
                  "columns_seen": read.columns_seen[:60],
                  "columns_used": read.columns_used, "strategies": len(cards),
                  "trades": int(sum(c["n"] for c in cards)), "warnings": list(read.warnings)},
        "settings": _settings_dict(settings),
        "warnings": list(read.warnings) + list(settings.notes),
        "need": None,
    }
    return result


def _how_numbers(cards: List[Dict[str, Any]]) -> List[str]:
    out = []
    for c in cards:
        ev = (c.get("numbers") or {}).get("evidence") or {}
        if ev.get("e_value") is None:
            continue
        e = ev["e_value"]
        e_text = f"{e:.3g}" if e < 1e6 else f"{e:.2e}"
        sf = ev.get("serial_factor") or 1.0
        out.append(f"{c['name']}: e-value {e_text} against a bar of {ev['bar']:,.0f} "
                   f"({'passes' if ev['passes'] else 'does not pass'}); spread widened x{math.sqrt(sf):.2f} for "
                   "neighbouring trades." if sf > 1.0001 else
                   f"{c['name']}: e-value {e_text} against a bar of {ev['bar']:,.0f} "
                   f"({'passes' if ev['passes'] else 'does not pass'}).")
    return out


def _reply_draft(verdict: Dict[str, Any], next_step: Dict[str, str], questions: List[Dict[str, Any]],
                 compare: Optional[Dict[str, Any]]) -> str:
    parts = [verdict["headline"]]
    plain = verdict.get("plain") or ""
    sentences = [s.strip() for s in plain.replace("  ", " ").split(". ") if s.strip()]
    if len(sentences) > 1:
        parts.append(sentences[1].rstrip(".") + ".")
    if compare is not None and compare.get("headline"):
        parts.append(compare["headline"])
    parts.append(f"Next step: {next_step['text']}")
    if questions:
        parts.append(f"One question: {questions[0]['question']}")
    return " ".join(parts)


def _multi_checks(cards, pairs, read) -> List[Dict[str, Any]]:
    out = []
    for c in cards:
        tone = {"real_edge": "pass", "cant_tell": "warn", "no_edge": "warn", "losing": "fail", "faded": "fail",
                "too_early": "warn"}[c["verdict"]["code"]]
        out.append(_check(f"strategy:{c['name']}", tone, f"{c['name']}: {c['verdict']['headline']}",
                          c["verdict"]["why"]))
    for p in pairs:
        out.append(_check("same_bet" if p["reading"] == "same_bet" else "opposite_bets", "warn", p["sentence"],
                          p["why"]))
    for w in read.warnings:
        out.append(_check("data", "warn", w, _data_why(w)))
    return out


def _merge_questions(cards) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for c in cards:
        for q in c["ask_the_user"]:
            if q["id"] not in seen:
                seen.add(q["id"])
                out.append(q)
    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda q: rank[q["impact"]])
    return out


def _summary(cards: List[Dict[str, Any]], counts: Dict[str, int], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    k = len(cards)
    trades = sum(c["n"] for c in cards)
    real = [c["name"] for c in cards if c["verdict"]["code"] == "real_edge"]
    if k == 1:
        headline = cards[0]["verdict"]["headline"]
    elif real:
        headline = (f"{len(real)} of {k} strategies {'shows' if len(real) == 1 else 'show'} a real edge: "
                    f"{join_and(real)}.")
    else:
        headline = f"None of your {k} strategies beats luck yet."
    words = {"real_edge": "{n} {s} a real edge", "cant_tell": "{n} can't be told from luck yet",
             "no_edge": "{n} {s} no edge", "losing": "{n} {l} for real", "faded": "{n} {h} faded",
             "too_early": "{n} {i} too early to judge"}
    parts = [words[code].format(n=n, s="shows" if n == 1 else "show", l="loses" if n == 1 else "lose",
                                h="has" if n == 1 else "have", i="is" if n == 1 else "are")
             for code, n in counts.items() if n]
    text = f"{count(k, 'strategy', 'strategies')}, {trades:,} trades."
    if k > 1 and parts:
        text += " " + join_and(parts) + "."
    for pair in pairs:
        text += " " + pair["sentence"]
    return {"headline": headline, "text": text, "strategies": k, "trades": trades, "counts": counts}


# ── the public call ───────────────────────────────────────────────────────────

def judge_dict(data: Any = None, *, variants_tried: Any = None, benchmark_return_pct: Any = None,
               kind: Optional[str] = None, unit: Optional[str] = None, fees_included: Any = None,
               source: Any = None, backtest: Any = None, name: Optional[str] = None,
               include_chart: bool = False, details: bool = True) -> Dict[str, Any]:
    settings = make_settings(variants_tried, benchmark_return_pct, fees_included, source, unit, kind, include_chart)
    try:
        read = read_any(data, kind=kind, unit=unit, name=name)
    except Exception as exc:  # the readers should not raise; if one does, the trader still gets an answer
        read = ReadResult([], "unknown", need={"problem": f"Could not read the input ({type(exc).__name__}: {exc})."})
    bt = None
    if backtest is not None:
        try:
            bt = read_any(backtest, kind=None, unit=None, name="Backtest")
        except Exception as exc:
            bt = ReadResult([], "unknown", need={"problem": f"Could not read the backtest ({exc})."})
    try:
        return analyze(read, settings, details=details, backtest=bt)
    except Exception as exc:  # never crash the caller; say what failed
        out = need_input(read, settings, problem=f"The judge failed on this input ({type(exc).__name__}: {exc}).")
        out["status"] = "error"
        return out
