"""Honest backtests and honest tuning, judged by the same judge (eol is the
edgeorluck package):

    px = eol.prices("BTCUSDT", "1h", start="2024-01-01")
    bt = eol.backtest(px, {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"})
    print(bt.to_markdown())

    tuned = eol.tune(px, {"buy": "rsi(14) < {low}", "sell": "rsi(14) > {high}"},
                     {"low": [20, 25, 30], "high": [65, 70, 75]})
    print(tuned.to_markdown())

What makes them honest:

* A decision is made at a bar's close and filled at the NEXT bar's open (or
  close, when there are no opens), with a fee and slippage on every fill. A
  rule never trades at a price it couldn't have had.
* A rule written in the rule language (rulebook.py) only looks back by
  construction. A Python rule is checked: its decision at a bar must not change
  when later bars are added; if it does, the rule peeks at the future and the
  backtest stops instead of judging it.
* The delay check runs the rule one bar later. An edge that vanishes with one
  bar of delay depends on instant fills or a subtle look-ahead.
* tune() picks the settings on the first part of the data only, then judges the
  chosen ones once on the rest, which it never saw. It knows how many settings
  it tried, so the in-sample view raises the bar by exactly that many.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .market import PriceError, load_prices
from .result import Result, judge
from .rulebook import RULE_SERIES, Rule, RuleError, compile_rule, rule_from

DEFAULT_FEE_PCT = 0.1           # per fill: a typical spot taker fee
DEFAULT_SLIPPAGE_PCT = 0.05     # per fill, against you
PEEK_CHECKPOINTS = 24           # evenly spread cuts, plus the first bars and the decision changes
PEEK_CHANGES = 60               # at most this many decision changes are cut at (and the bars before them)
MAX_COMBINATIONS = 1000
MIN_TUNE_TRADES = 10

SIM_FOR_THE_ASSISTANT = [
    "Quote the verdict's headline word for word; the checks say how far the backtest itself can be trusted.",
    "A backtest is a promise, not a result: the next step is paper or small live trading, then compare live with "
    "the backtest (judge(live, backtest=...)).",
    "If you tried other rules or settings before this one, say how many in all as variants_tried.",
    "Not financial advice.",
]


# ── the price frame and the fills ─────────────────────────────────────────────

def _sim_frame(prices: Any) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame) and "close" in prices.columns:
        frame = prices.copy()
    else:
        frame = load_prices(prices)
    frame = frame[pd.to_numeric(frame["close"], errors="coerce") > 0].copy()
    notes: List[str] = list(frame.attrs.get("notes", []))
    if isinstance(frame.index, pd.DatetimeIndex):
        # Oldest first, one row per time. A newest-first table would run the backtest backwards (trades exiting
        # before they enter), and a repeated bar would fill a decision at the open it was made from.
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index(kind="stable")
            notes.append("The prices were not in time order; they were sorted oldest first.")
        if frame.index.has_duplicates:
            dupes = int(frame.index.duplicated(keep="last").sum())
            frame = frame[~frame.index.duplicated(keep="last")]
            notes.append(f"{dupes:,} repeated bar{'s' if dupes != 1 else ''} (the same time twice) were dropped.")
    if len(frame) < 30:
        raise PriceError("A backtest needs at least 30 bars of prices.")
    frame.attrs["notes"] = notes
    return frame


def _sim_view(frame: pd.DataFrame) -> pd.DataFrame:
    """What a Python rule is given: the prices only (open, high, low, close, volume),
    never other columns, so a signal computed in advance from later prices isn't there."""
    return frame[[c for c in RULE_SERIES if c in frame.columns]].copy()


def _sim_call(fn: Callable, view: pd.DataFrame) -> Any:
    try:
        return fn(view)
    except KeyError as exc:
        raise RuleError(f"your rule asked for the column {exc}: a Python rule gets only the prices (open, high, low, "
                        "close, volume). Compute its signal inside the rule from them.") from None


def _sim_fill_prices(frame: pd.DataFrame) -> Tuple[np.ndarray, str]:
    close = frame["close"].to_numpy(float)
    if "open" in frame.columns:
        opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(float)
        good = np.isfinite(opens) & (opens > 0)
        if good.mean() > 0.99:
            return np.where(good, opens, close), "the next bar's open"
    return close, "the next bar's close (these prices have no opens)"


def _sim_when(frame: pd.DataFrame, i: int) -> Any:
    idx = frame.index[i]
    return idx.isoformat() if isinstance(idx, pd.Timestamp) else int(i)


def _sim_show(value: Any) -> str:
    """A time for people: 2024-01-27 19:00 (or the date alone at midnight); a bar number as is."""
    if isinstance(value, str):
        try:
            ts = pd.Timestamp(value)
        except ValueError:
            return value
        return ts.strftime("%Y-%m-%d") if (ts.hour, ts.minute) == (0, 0) else ts.strftime("%Y-%m-%d %H:%M")
    return f"bar {value}"


# ── decisions: what the rule wants to hold after each bar's close ────────────

def _sim_rule_decisions(frame: pd.DataFrame, rule: Rule, fills: np.ndarray, delay: int) -> np.ndarray:
    """The rule language: a position chosen at each bar's close. Exits are checked
    only after the entry has been filled, against that fill's price."""
    buy = compile_rule(rule.buy)(frame).to_numpy(bool)
    sell = compile_rule(rule.sell)(frame).to_numpy(bool) if rule.sell else np.zeros(len(frame), bool)
    close = frame["close"].to_numpy(float)
    sign = -1.0 if rule.side == "short" else 1.0
    n = len(frame)
    decisions = np.zeros(n)
    holding, filled_at = False, -1
    for t in range(n):
        if holding:
            if t >= filled_at:
                move = sign * 100.0 * (close[t] / fills[filled_at] - 1.0)
                held = t - filled_at + 1
                if (sell[t] or (rule.stop_loss_pct and move <= -rule.stop_loss_pct)
                        or (rule.take_profit_pct and move >= rule.take_profit_pct)
                        or (rule.max_hold_bars and held >= rule.max_hold_bars)):
                    holding = False
        elif buy[t] and t + delay < n:
            holding, filled_at = True, t + delay
        decisions[t] = sign if holding else 0.0
    return decisions


def _sim_values(out: Any, frame: pd.DataFrame, allow_short: bool, where: str) -> np.ndarray:
    if isinstance(out, pd.DataFrame):
        raise RuleError(f"{where}: return one position per bar (a Series or a list), not a table")
    if isinstance(out, pd.Series):
        if out.index.equals(frame.index) or out.index.isin(frame.index).any():
            arr = out.reindex(frame.index).to_numpy()
        elif len(out) == len(frame):
            arr = out.to_numpy()                 # a plain 0..n-1 index: the positions in order
        else:
            raise RuleError(f"{where}: its positions are on other dates or bars than the prices it was given")
    else:
        arr = np.asarray(out)
    if arr.ndim != 1 or len(arr) != len(frame):
        raise RuleError(f"{where}: return one position per bar ({len(frame)} values, got "
                        f"{len(arr) if arr.ndim == 1 else arr.shape})")
    if arr.dtype == bool:
        return arr.astype(float)
    vals = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0.0).to_numpy(float)
    return np.sign(vals) if allow_short else (vals > 0).astype(float)


def _sim_peeks(frame: pd.DataFrame, fn: Callable, full: np.ndarray,
               allow_short: bool) -> Tuple[Optional[Dict[str, Any]], int]:
    """The look-ahead test: every decision up to bar t must be the same whether or
    not the bars after t exist. Cut at spread-out bars, the earliest bars (where
    warm-up tricks like a backward fill live), and around the bars where the
    decisions change (where a peek pays). Returns (the first peek found, bars cut at)."""
    n = len(frame)
    view = _sim_view(frame)
    rng = np.random.default_rng(n)
    spots = set(np.linspace(1, n - 2, PEEK_CHECKPOINTS).astype(int)) | set(range(1, min(n - 1, 12)))
    changes = np.flatnonzero(np.diff(full) != 0) + 1
    if len(changes):
        pick = changes if len(changes) <= PEEK_CHANGES else rng.choice(changes, PEEK_CHANGES, replace=False)
        spots |= {int(c) for c in pick if 0 < c < n - 1} | {int(c) - 1 for c in pick if 1 < c < n}
    for t in sorted(spots):
        part = view.iloc[: t + 1].copy()
        got = _sim_values(_sim_call(fn, part), part, allow_short, "your rule")
        wrong = np.flatnonzero(got != full[: t + 1])
        if len(wrong):
            b = int(wrong[0])
            return {"bar": b, "time": _sim_when(frame, b), "with_all_bars": float(full[b]),
                    "with_bars_so_far": float(got[b])}, len(spots)
    return None, len(spots)


def _sim_ignores_prices(frame: pd.DataFrame, fn: Callable, full: np.ndarray, allow_short: bool) -> bool:
    """True when a rule's (changing) decisions stay exactly the same on entirely
    different prices: then it reads its signal from somewhere else (a variable
    outside the function, a file), and no test on the prices can check it."""
    if len(np.unique(full)) < 2:
        return False                      # always in or always out: nothing to check
    view = _sim_view(frame)
    rng = np.random.default_rng(len(view) + 7)
    close = float(view["close"].iloc[0]) * np.exp(np.cumsum(rng.normal(0, 0.02, len(view))))
    other = view.copy()
    other["close"] = close
    if "open" in other:
        other["open"] = np.r_[close[0], close[:-1]]
    if "high" in other:
        other["high"] = np.maximum(other["open"] if "open" in other else close, close) * 1.002
    if "low" in other:
        other["low"] = np.minimum(other["open"] if "open" in other else close, close) * 0.998
    if "volume" in other:
        other["volume"] = rng.permutation(view["volume"].to_numpy())
    try:
        got = _sim_values(fn(other), other, allow_short, "your rule")
    except Exception:  # the rule failed on other prices: it does read them
        return False
    return bool(np.array_equal(got, full))


# ── fills, trades and their costs ─────────────────────────────────────────────

def _sim_trades(frame: pd.DataFrame, decisions: np.ndarray, fills: np.ndarray, delay: int, fee_pct: float,
                slippage_pct: float) -> List[Dict[str, Any]]:
    """Trades from decisions: a decision at bar t is filled at bar t + delay."""
    n = len(frame)
    held = np.zeros(n)
    held[delay:] = decisions[: n - delay]
    close = frame["close"].to_numpy(float)
    f, s = fee_pct / 100.0, slippage_pct / 100.0
    trades: List[Dict[str, Any]] = []
    cur, start = 0.0, -1

    def finish(entry_i: int, exit_i: int, side: float, exit_px: float, end_of_data: bool) -> None:
        if side > 0:
            buy_fill, sell_fill = fills[entry_i] * (1 + s), exit_px * (1 - s)
            net = sell_fill * (1 - f) / (buy_fill * (1 + f)) - 1.0
        else:
            sell_fill, buy_fill = fills[entry_i] * (1 - s), exit_px * (1 + s)
            net = (sell_fill * (1 - f) - buy_fill * (1 + f)) / sell_fill
        trades.append({"entry_time": _sim_when(frame, entry_i), "exit_time": _sim_when(frame, exit_i),
                       "side": "long" if side > 0 else "short", "entry_price": float(fills[entry_i]),
                       "exit_price": float(exit_px), "return_pct": 100.0 * net, "bars": int(exit_i - entry_i),
                       "entry_bar": int(entry_i), "exit_bar": int(exit_i), "end_of_data": end_of_data})

    for t in range(n):
        if held[t] != cur:
            if cur != 0:
                finish(start, t, cur, fills[t], False)
            if held[t] != 0:
                start = t
            cur = held[t]
    if cur != 0:
        finish(start, n - 1, cur, close[n - 1], True)
    return trades


def _sim_stats(trades: Sequence[Dict[str, Any]], frame: pd.DataFrame, fills: np.ndarray, delay: int,
               lo: int = 0, hi: Optional[int] = None) -> Dict[str, Any]:
    hi = len(frame) if hi is None else hi
    r = np.array([t["return_pct"] for t in trades], float)
    first = min(max(lo, delay), hi - 1)
    hold = 100.0 * (frame["close"].to_numpy(float)[hi - 1] / fills[first] - 1.0)
    in_market = sum(max(0, min(t["exit_bar"], hi) - max(t["entry_bar"], lo)) for t in trades)
    return {"trades": int(len(r)),
            "total_pct": float(100.0 * (np.prod(1.0 + r / 100.0) - 1.0)) if len(r) else 0.0,
            "mean_pct": float(r.mean()) if len(r) else None,
            "win_share": float((r > 0).mean()) if len(r) else None,
            "hold_pct": float(hold), "time_in_market": float(in_market / max(1, hi - lo))}


def _sim_judge(trades: Sequence[Dict[str, Any]], name: str, hold_pct: float, variants: Optional[int],
               fees_included: bool = True) -> Result:
    table = pd.DataFrame({"Entry time": [t["entry_time"] for t in trades], "Exit time": [t["exit_time"] for t in trades],
                          "Side": [t["side"] for t in trades], "Return %": [t["return_pct"] for t in trades]})
    if not isinstance(trades[0]["entry_time"], str):
        table = table.drop(columns=["Entry time", "Exit time"])
    return judge(table, variants_tried=variants, benchmark_return_pct=round(hold_pct, 6), fees_included=fees_included,
                 source="backtest", unit="%", name=name)


def _sim_check(code: str, status: str, text: str, why: str, fix: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "status": status, "text": text, "fix": fix, "why": why}


def _sim_delay_check(base: Sequence[Dict[str, Any]], later: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    why = ("A signal only exists after its bar closes, and real fills come a little later. An edge that disappears "
           "when every fill comes one bar later depends on instant fills or on a hidden look into the future.")
    rb, rl = (np.array([t["return_pct"] for t in x], float) for x in (base, later))
    if len(rb) < MIN_TUNE_TRADES or rb.mean() <= 0:
        return _sim_check("one_bar_later", "pass", "Not checked: there is no positive edge to test.", why)
    now, then = float(rb.mean()), (float(rl.mean()) if len(rl) else 0.0)
    text = (f"Filled one bar later, the average trade goes from {now:+.3f}% to {then:+.3f}%.")
    if then <= 0:
        return _sim_check("one_bar_later", "fail", text + " The edge disappears.", why,
                          "Don't trust this edge: look for a timing trick or a look-ahead in the rule.")
    if then < 0.5 * now:
        return _sim_check("one_bar_later", "warn", text + " Most of the edge is lost.", why,
                          "Expect live results well below the backtest; test it on paper first.")
    return _sim_check("one_bar_later", "pass", text + " It survives.", why)


# ── the backtest ──────────────────────────────────────────────────────────────

class Backtest(Result):
    """An honest backtest and its verdict: read it like a dict, or call
    .to_markdown(), .to_dict(), .to_json(); .judgement is the judge's Result and
    .trades the trades as a table."""

    @property
    def judgement(self) -> Optional[Result]:
        j = self._data.get("judgement")
        return Result(j) if j else None

    @property
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self._data.get("trades") or [])

    def to_markdown(self) -> str:
        return _sim_markdown(self._data)


def _sim_describe_rule(rule: Any) -> Dict[str, Any]:
    if isinstance(rule, Rule):
        return {"kind": "rule language", "text": rule.describe(), "buy": rule.buy, "sell": rule.sell,
                "stop_loss_pct": rule.stop_loss_pct, "take_profit_pct": rule.take_profit_pct,
                "max_hold_bars": rule.max_hold_bars, "side": rule.side, "params": dict(rule.params)}
    name = getattr(rule, "__name__", "your rule")
    return {"kind": "python", "text": f"Python rule {name}", "name": name}


def _sim_run(frame: pd.DataFrame, rule: Any, *, delay: int, fee_pct: float, slippage_pct: float, allow_short: bool,
             check_peeking: bool) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[Dict[str, Any]], np.ndarray]:
    """Decisions at each close and the trades they make, filled ``delay`` bars later.
    The decisions are always the ones made with next-bar fills, so delay=2 runs the
    very same trades one bar later (the one-bar-later check), for either kind of rule."""
    fills, _ = _sim_fill_prices(frame)
    peek = None
    if isinstance(rule, Rule):
        decisions = _sim_rule_decisions(frame, rule, fills, 1)
    else:
        decisions = _sim_values(_sim_call(rule, _sim_view(frame)), frame, allow_short, "your rule")
        if check_peeking:
            peek, checked = _sim_peeks(frame, rule, decisions, allow_short)
            if peek is None and _sim_ignores_prices(frame, rule, decisions, allow_short):
                peek = {"unchecked": True, "checked": checked}
            elif peek is None:
                peek = {"checked": checked, "ok": True}
    return decisions, _sim_trades(frame, decisions, fills, delay, fee_pct, slippage_pct), peek, fills


def _sim_costs_ok(fee_pct: float, slippage_pct: float) -> None:
    for label, value in (("fee_pct", fee_pct), ("slippage_pct", slippage_pct)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuleError(f"{label} is a cost per fill in percent, 0 or more (got {value!r})")


def _sim_peek_stop(out: Dict[str, Any], peek: Dict[str, Any], frame: pd.DataFrame, chosen: bool = False) -> None:
    """Stop: a Python rule that peeks, or one that can't be checked."""
    what = "the chosen rule" if chosen else "the rule"
    if peek.get("unchecked"):
        out["status"] = "rule_unchecked"
        out["headline"] = (f"Stopped: {what}'s decisions don't change when the prices do, so it reads its signal from "
                           "somewhere else and can't be checked for looking ahead.")
        out["checks"].append(_sim_check(
            "no_look_ahead", "fail", "On completely different prices the rule made exactly the same decisions.",
            "The look-ahead test checks how a rule uses the prices it's given. A signal computed beforehand (a "
            "variable outside the function, a column from a file) may have used later prices, and nothing here "
            "can tell.",
            "Compute the signal inside the rule from the prices it's given (f['close'] and so on), or write it in "
            "the rule language."))
        return
    out["status"] = "rule_peeks"
    out["headline"] = f"Stopped: {what} looks into the future, so its backtest can't be trusted."
    out["checks"].append(_sim_check(
        "no_look_ahead", "fail",
        f"At bar {peek['bar']} ({_sim_show(peek['time'])}) the rule decides {peek['with_all_bars']:g} when it can see "
        f"the whole history, but {peek['with_bars_so_far']:g} when it only sees the bars up to then.",
        "A real bot only knows the bars that have closed. A rule whose decision changes when later bars are "
        "added is using the future, and its backtest looks better than any live result can be.",
        "Look for shift(-1), centred rolling windows, backward fills, values normalised over the whole series, or "
        "labels computed from later prices. Or write the rule in the rule language, which can't look ahead."))


def _sim_rule(rule: Any) -> Any:
    if callable(rule) and not isinstance(rule, (Rule, Mapping)):
        return rule
    return rule_from(rule)


def backtest(prices: Any, rule: Any, *, fee_pct: float = DEFAULT_FEE_PCT, slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
             allow_short: bool = False, variants_tried: Any = None, check_peeking: bool = True,
             name: Optional[str] = None, since: Any = None) -> Backtest:
    """Run a rule honestly on prices and judge the trades.

    prices: edgeorluck.prices(...), a DataFrame with a close column (open, high, low,
        volume when there are some), a CSV path or text, or a list of closes.
    rule: a dict in the rule language, like {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}
        (optional: "stop_loss_pct", "take_profit_pct", "max_hold_bars", "side": "short"),
        or a Python function taking the price frame and returning one position per bar
        (1 long, 0 flat, -1 short with allow_short=True, or True/False).
    fee_pct, slippage_pct: per fill, in percent (0.1 = 0.1%).
    variants_tried: how many rules or settings were tried in all before this one.
    since: a forward test: only trades entered after this time count (the rule was
        frozen then, so none of them could have shaped it); earlier bars only warm up.

    A decision at a bar's close is filled at the next bar's open. A Python rule that
    looks into the future is caught and not judged."""
    _sim_costs_ok(fee_pct, slippage_pct)
    frame = _sim_frame(prices)
    rule = _sim_rule(rule)
    label = name or (rule.name if isinstance(rule, Rule) and rule.name else None) or "Backtest"
    decisions, trades, peek, fills = _sim_run(frame, rule, delay=1, fee_pct=fee_pct, slippage_pct=slippage_pct,
                                              allow_short=allow_short, check_peeking=check_peeking)
    if since is not None and not isinstance(frame.index, pd.DatetimeIndex):
        raise PriceError("A forward test needs prices with dates: only the trades after the freeze count.")
    lo = 0 if since is None else int((frame.index < pd.Timestamp(since, tz="UTC").tz_convert(frame.index.tz)).sum())
    trades = [t for t in trades if t["entry_bar"] >= lo]
    _, fill_text = _sim_fill_prices(frame)
    data = {"source": frame.attrs.get("source", "your prices"), "bars": int(len(frame)),
            "first": _sim_when(frame, 0), "last": _sim_when(frame, len(frame) - 1), "filled_at": fill_text}
    out: Dict[str, Any] = {"status": "ok", "headline": "", "rule": _sim_describe_rule(rule), "data": data,
                           "costs": {"fee_pct": fee_pct, "slippage_pct": slippage_pct,
                                     "round_trip_pct": round(2 * (fee_pct + slippage_pct), 6)},
                           "stats": _sim_stats(trades, frame, fills, 1, lo), "checks": [], "trades": trades,
                           "judgement": None, "for_the_assistant": list(SIM_FOR_THE_ASSISTANT)}
    for note in frame.attrs.get("notes", []):
        out["checks"].append(_sim_check("prices", "warn", note, "The backtest runs bar by bar in time order."))
    if peek is not None and not peek.get("ok"):
        _sim_peek_stop(out, peek, frame)
        out["trades"] = []
        return Backtest(out)
    if not isinstance(rule, Rule):
        checked = f"at {peek['checked']} test bars " if peek else ""
        out["checks"].append(_sim_check(
            "no_look_ahead", "pass" if peek else "warn",
            (f"The rule decided the same {checked}with and without the later bars, and its decisions follow the "
             "prices it's given." if peek else "Not checked for looking ahead (check_peeking=False)."),
            "A rule that changes its mind when later bars are added is using the future. A test like this can "
            "miss a rule that peeks only now and then; the rule language can't peek at all."))
    else:
        out["checks"].append(_sim_check(
            "no_look_ahead", "pass", "Written in the rule language, which only looks back.",
            "A rule that uses later bars looks better in a backtest than it can ever be live."))
    if since is not None:
        out["checks"].append(_sim_check("forward", "pass", f"A forward test: only trades entered after "
                                        f"{_sim_show(str(since))} count, so none of them could have shaped the rule.",
                                        "Trades a rule never saw while it was being chosen are the honest test of it."))
    if not trades:
        out["status"] = "no_trades"
        out["headline"] = ("No trades since it was frozen yet." if since is not None else
                           "The rule never traded on these prices: check its thresholds against them.")
        return Backtest(out)
    _, later, _, _ = _sim_run(frame, rule, delay=2, fee_pct=fee_pct, slippage_pct=slippage_pct,
                              allow_short=allow_short, check_peeking=False)
    delay_check = _sim_delay_check(trades, [t for t in later if t["entry_bar"] >= lo])
    out["checks"].append(delay_check)
    free = fee_pct + slippage_pct == 0
    out["checks"].append(_sim_check(
        "costs", "warn" if free else "pass",
        "No fees or slippage were charged." if free else
        f"Every fill pays {fee_pct:g}% fees and {slippage_pct:g}% slippage, {2 * (fee_pct + slippage_pct):.2f}% per "
        "round trip.",
        "A small edge before costs is often a loss after them.",
        "Set fee_pct and slippage_pct to your exchange's real costs." if free else None))
    if trades[-1]["end_of_data"]:
        out["checks"].append(_sim_check(
            "open_at_end", "warn", "The last trade was still open when the prices ended; it's closed at the last "
            "close.", "Its result is not final."))
    j = _sim_judge(trades, label, out["stats"]["hold_pct"], variants_tried, fees_included=not free)
    out["judgement"] = j.to_dict()
    out["headline"] = j.headline
    if delay_check["status"] == "fail":
        # The verdict may say "real edge", but an edge that vanishes one bar later lives on instant fills or a
        # hidden look-ahead: the headline (the line assistants quote) says so first.
        out["status"] = "fragile"
        out["headline"] = ("Don't trust this yet: with every fill one bar later, the edge disappears. The judge on "
                           f"these trades: {j.headline}")
    return Backtest(out)


# ── tuning on one part, judging on the other ─────────────────────────────────

class Tuned(Result):
    """Honest tuning: the settings chosen on the first part of the data, judged once
    on the part it never saw. .judgement is that verdict; .in_sample the judge's
    view of the chosen settings on the data that chose them, with the bar raised
    for every setting tried; .table every setting's results."""

    @property
    def best(self) -> Optional[Dict[str, float]]:
        return self._data.get("best")

    @property
    def judgement(self) -> Optional[Result]:
        j = self._data.get("out_of_sample")
        return Result(j) if j else None

    @property
    def in_sample(self) -> Optional[Result]:
        j = self._data.get("in_sample")
        return Result(j) if j else None

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self._data.get("table") or [])

    def to_markdown(self) -> str:
        return _sim_tune_markdown(self._data)


def _sim_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not isinstance(grid, Mapping) or not grid:
        raise RuleError('grid is a dict of settings to try, like {"low": [20, 25, 30], "high": [65, 70, 75]}')
    names, values = [], []
    for key, vals in grid.items():
        vals = list(vals) if isinstance(vals, (list, tuple, range, np.ndarray, pd.Series, pd.Index)) else [vals]
        vals = [v.item() if isinstance(v, np.generic) else v for v in vals]       # np.int64(10) -> 10
        vals = list(dict.fromkeys(vals))                                          # 15, 15 and 15.0 are one value
        if not vals:
            raise RuleError(f"setting {key} has no values to try")
        names.append(str(key))
        values.append(vals)
    combos = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    if len(combos) > MAX_COMBINATIONS:
        raise RuleError(f"{len(combos):,} combinations is more than {MAX_COMBINATIONS:,}: try fewer values")
    return combos


def _finite(v: float) -> Optional[float]:
    """None for inf/nan: JSON has no infinity (identical results give a t of inf)."""
    return v if math.isfinite(v) else None


def _sim_t(r: np.ndarray, min_n: int) -> float:
    if len(r) < min_n:
        return -math.inf
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(len(r))) if sd > 0 else (math.inf if r.mean() > 0 else -math.inf)


def tune(prices: Any, rule: Any, grid: Mapping[str, Sequence[Any]], *, holdout: float = 0.3,
         fee_pct: float = DEFAULT_FEE_PCT, slippage_pct: float = DEFAULT_SLIPPAGE_PCT, allow_short: bool = False,
         min_trades: int = MIN_TUNE_TRADES, name: Optional[str] = None, variants_tried: Any = None) -> Tuned:
    """Try every combination of settings on the first part of the prices, pick the
    best there, and judge it once on the last part (holdout, 30% by default), which
    played no part in the choice.

    rule: the rule language with {placeholders}, like {"buy": "rsi(14) < {low}",
        "sell": "rsi(14) > {high}"}, or a Python function taking the settings and
        returning a rule (a dict or a Python rule).
    grid: the values to try for each setting, like {"low": [20, 25, 30], "high": [65, 70, 75]}.
    variants_tried: rules judged on these prices in all, this one included (the others saw the unseen part).

    Don't tune again after seeing the verdict on the unseen part: that turns it into
    tuning data too."""
    _sim_costs_ok(fee_pct, slippage_pct)
    frame = _sim_frame(prices)
    if not 0.1 <= holdout <= 0.6:
        raise RuleError("holdout is the share kept unseen for the verdict: between 0.1 and 0.6 (0.3 = the last 30%)")
    n = len(frame)
    cut = int(round(n * (1.0 - holdout)))
    fills, _ = _sim_fill_prices(frame)
    rows, best_i, best_score, built, seen, combos = [], None, -math.inf, [], set(), []
    for params in _sim_grid(grid):
        made = rule(**params) if (callable(rule) and not isinstance(rule, (Rule, Mapping))) else rule_from(rule, params)
        made = _sim_rule(made) if not isinstance(made, Rule) else made
        if isinstance(made, Rule):
            # a setting the rule doesn't use, or a repeated value, gives a rule already tried: it isn't another try
            same = (made.buy, made.sell, made.stop_loss_pct, made.take_profit_pct, made.max_hold_bars, made.side)
            if same in seen:
                continue
            seen.add(same)
        i = len(combos)
        combos.append(params)
        built.append(made)
        _, trades, _, _ = _sim_run(frame, made, delay=1, fee_pct=fee_pct, slippage_pct=slippage_pct,
                                   allow_short=allow_short, check_peeking=False)
        ins = np.array([t["return_pct"] for t in trades if t["exit_bar"] < cut], float)
        outs = np.array([t["return_pct"] for t in trades if t["entry_bar"] >= cut], float)
        score = _sim_t(ins, min_trades)
        rows.append({"settings": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
                     "in_sample_trades": int(len(ins)), "in_sample_mean_pct": float(ins.mean()) if len(ins) else None,
                     "in_sample_t": score if math.isfinite(score) else None,
                     "unseen_trades": int(len(outs)), "unseen_mean_pct": float(outs.mean()) if len(outs) else None,
                     "unseen_t": _finite(_sim_t(outs, 2)) if len(outs) >= 2 else None})
        if score > best_score:
            best_i, best_score = i, score
    first_part = {"from": _sim_when(frame, 0), "to": _sim_when(frame, cut - 1), "bars": cut}
    unseen_part = {"from": _sim_when(frame, cut), "to": _sim_when(frame, n - 1), "bars": n - cut}
    out: Dict[str, Any] = {"status": "ok", "headline": "", "tried": len(combos), "holdout": holdout,
                           "first_part": first_part, "unseen_part": unseen_part,
                           "rule": _sim_describe_rule(built[0]) if built else None, "best": None, "table": rows,
                           "in_sample": None, "out_of_sample": None, "checks": [], "family": None,
                           "costs": {"fee_pct": fee_pct, "slippage_pct": slippage_pct},
                           "for_the_assistant": [
                               "The verdict on the unseen part is the honest one: quote its headline. The first-part "
                               "result only shows how good the settings looked on the data that chose them.",
                               "Don't tune again after seeing the unseen-part verdict: it would become tuning data "
                               "too. If you change the rule anyway, count every version as variants_tried.",
                               "Not financial advice."]}
    if best_i is None:
        out["status"] = "too_few_trades"
        out["headline"] = (f"No setting made {min_trades} trades in the first part: nothing to choose from. Use more "
                           "data or a rule that trades more often.")
        return Tuned(out)
    best = built[best_i]
    out["best"] = rows[best_i]["settings"]
    out["rule"] = _sim_describe_rule(best)
    _, trades, peek, _ = _sim_run(frame, best, delay=1, fee_pct=fee_pct, slippage_pct=slippage_pct,
                                  allow_short=allow_short, check_peeking=not isinstance(best, Rule))
    if peek is not None and not peek.get("ok"):
        _sim_peek_stop(out, peek, frame, chosen=True)
        return Tuned(out)
    ins = [t for t in trades if t["exit_bar"] < cut]
    outs = [t for t in trades if t["entry_bar"] >= cut]
    label = name or "Tuned rule"
    out["in_sample"] = _sim_judge(ins, label + " (first part)", _sim_stats(ins, frame, fills, 1, 0, cut)["hold_pct"],
                                  len(combos)).to_dict()
    unseen_stats = _sim_stats(outs, frame, fills, 1, cut, n)
    out["unseen_stats"] = unseen_stats
    family = [r for r in rows if r["unseen_mean_pct"] is not None and r["unseen_trades"] >= 5]
    if family:
        ranked = sorted(family, key=lambda r: r["unseen_mean_pct"], reverse=True)
        rank = next((k + 1 for k, r in enumerate(ranked) if r is rows[best_i]), None)
        out["family"] = {"settings_with_unseen_trades": len(family),
                         "made_money_unseen": sum(1 for r in family if r["unseen_mean_pct"] > 0),
                         "chosen_rank_unseen": rank}
    if len(outs) < 2:
        out["status"] = "too_few_unseen"
        out["headline"] = ("The chosen settings barely traded in the unseen part, so it can't judge them. Keep more "
                           "data unseen (a bigger holdout) or use a longer history.")
        return Tuned(out)
    j = _sim_judge(outs, label + " (unseen part)", unseen_stats["hold_pct"], variants_tried or 1)
    out["out_of_sample"] = j.to_dict()
    out["headline"] = j.headline
    _, later, _, _ = _sim_run(frame, best, delay=2, fee_pct=fee_pct, slippage_pct=slippage_pct,
                              allow_short=allow_short, check_peeking=False)
    out["checks"].append(_sim_delay_check(outs, [t for t in later if t["entry_bar"] >= cut]))
    return Tuned(out)


# ── Markdown ──────────────────────────────────────────────────────────────────

_MARK = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def _sim_checks_md(checks: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for c in checks:
        lines.append(f"- **{_MARK.get(c['status'], c['status'].upper())}** {c['text']}")
        if c.get("fix"):
            lines.append(f"  Fix: {c['fix']}")
        lines.append(f"  _Why: {c['why']}_")
    return lines


def _sim_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def _sim_markdown(d: Dict[str, Any]) -> str:
    rule, data, st, costs = d["rule"], d["data"], d["stats"], d["costs"]
    lines = [f"## Honest backtest: {rule['text']}", "",
             f"{data['bars']:,} bars of {data['source']}, {_sim_show(data['first'])} to {_sim_show(data['last'])}. Each decision is made at a "
             f"bar's close and filled at {data['filled_at']}, with {costs['fee_pct']:g}% fees and "
             f"{costs['slippage_pct']:g}% slippage on every fill.", ""]
    if d["status"] in ("rule_peeks", "rule_unchecked"):
        lines += [f"**{d['headline']}**", "", "### Honesty checks", ""] + _sim_checks_md(d["checks"])
        return "\n".join(lines) + "\n"
    lines += [f"**Result:** {st['trades']:,} trades, {_sim_pct(st['total_pct'])} in total, against "
              f"{_sim_pct(st['hold_pct'])} for simply holding; in the market {100 * st['time_in_market']:.0f}% of the "
              "time.", ""]
    if d["status"] == "no_trades":
        lines += [f"**{d['headline']}**", ""]
    lines += ["### Honesty checks", ""] + _sim_checks_md(d["checks"]) + [""]
    if d.get("judgement"):
        lines += [Result(d["judgement"]).to_markdown().rstrip(), ""]
    lines += ["### For the assistant (backtest)", ""] + [f"- {x}" for x in d["for_the_assistant"]]
    return "\n".join(lines) + "\n"


def _sim_tune_markdown(d: Dict[str, Any]) -> str:
    fp, up = d["first_part"], d["unseen_part"]
    lines = [f"## Honest tuning: {d['rule']['text'] if d.get('rule') else 'your rule'}", "",
             f"Tried {d['tried']:,} settings on the first part ({_sim_show(fp['from'])} to {_sim_show(fp['to'])}, {fp['bars']:,} bars), then "
             f"judged the best once on the last part ({_sim_show(up['from'])} to {_sim_show(up['to'])}, {up['bars']:,} bars), which played "
             "no part in the choice.", ""]
    if d.get("best"):
        best_row = next((r for r in d["table"] if r["settings"] == d["best"]), None)
        settings = ", ".join(f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}" for k, v in d["best"].items())
        lines.append(f"**Chosen settings:** {settings}" + (
            f" (first part: {_sim_pct(best_row['in_sample_mean_pct'])} per trade over {best_row['in_sample_trades']} "
            "trades)." if best_row and best_row["in_sample_mean_pct"] is not None else "."))
        lines.append("")
    lines += [f"**{d['headline']}**", ""]
    if d.get("out_of_sample"):
        lines += ["### The verdict on the unseen part", "", Result(d["out_of_sample"]).to_markdown().rstrip(), ""]
    fam = d.get("family")
    if fam:
        rank = (f"; the chosen one ranked {fam['chosen_rank_unseen']} of {fam['settings_with_unseen_trades']}"
                if fam.get("chosen_rank_unseen") else "")
        lines += ["### How the other settings did on the unseen part", "",
                  f"{fam['made_money_unseen']} of {fam['settings_with_unseen_trades']} settings made money there"
                  f"{rank}. When few settings make money on unseen data, the first-part winner was mostly luck.", ""]
    if d.get("in_sample"):
        ins = d["in_sample"]
        lines += ["### The chosen settings on the data that chose them", "",
                  f"{ins['verdict']['headline'] if ins.get('verdict') else ins['summary']['headline']} (the bar is "
                  f"raised for all {d['tried']:,} settings tried).", ""]
    if d.get("checks"):
        lines += ["### Honesty checks", ""] + _sim_checks_md(d["checks"]) + [""]
    lines += ["### For the assistant (tuning)", ""] + [f"- {x}" for x in d["for_the_assistant"]]
    return "\n".join(lines) + "\n"


__all__ = ["backtest", "tune", "Backtest", "Tuned", "DEFAULT_FEE_PCT", "DEFAULT_SLIPPAGE_PCT"]
