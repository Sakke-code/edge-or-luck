"""A small, safe language for trading rules, so a rule can arrive as plain text
(from the web page, the command line or an AI assistant) and can never look
into the future.

    buy:  "rsi(14) < 30 and close > sma(200)"
    sell: "rsi(14) > 70"

Every function looks back only (rolling windows, exponential averages, lags),
so a rule written here can't use a later bar by construction. Only the price
names and the functions in RULE_FUNCTIONS are allowed: no attributes, indexing,
imports or code. A function given one number works on the close: sma(20) is
sma(close, 20).
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

RULE_SERIES = ("open", "high", "low", "close", "volume")
MAX_RULE_LENGTH = 500
MAX_WINDOW = 10_000_000        # longer than any price history: a bigger window is a typo, or an attempt to stall
MAX_NUMBER = 1e15

RULE_FUNCTIONS = {
    "sma": "sma(n) or sma(series, n): simple moving average of the last n bars",
    "ema": "ema(n) or ema(series, n): exponential moving average, as TradingView's ta.ema",
    "rsi": "rsi(n) or rsi(series, n): Wilder's relative strength index, 0 to 100, as TradingView's ta.rsi",
    "highest": "highest(n) or highest(series, n): the highest value of the last n bars, this one included "
               "(a breakout above the earlier high is close > prev(highest(high, 20)))",
    "lowest": "lowest(n) or lowest(series, n): the lowest value of the last n bars, this one included",
    "std": "std(n) or std(series, n): standard deviation of the last n bars, as TradingView's ta.stdev",
    "atr": "atr(n): average true range, as TradingView's ta.atr (needs high and low)",
    "change": "change(n) or change(series, n): the difference from n bars ago",
    "pct": "pct(n) or pct(series, n): the change from n bars ago, in percent",
    "prev": "prev(series) or prev(series, n): the value n bars ago (1 by default)",
    "crosses_above": "crosses_above(a, b): a rose above b on this bar",
    "crosses_below": "crosses_below(a, b): a fell below b on this bar",
    "abs": "abs(x)",
    "min": "min(a, b): the smaller, bar by bar",
    "max": "max(a, b): the larger, bar by bar",
}


class RuleError(ValueError):
    """A rule that can't be read, with a plain sentence saying why and how to fix it."""


Value = Union[float, pd.Series]

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def _rb_window(name: str, args: List[Value], frame: pd.DataFrame, *, default_n: Optional[int] = None):
    """(series, n) from f(n) or f(series, n) (or f(series) when there is a default n)."""
    if len(args) == 1 and not isinstance(args[0], pd.Series):
        series, n = frame["close"], args[0]
    elif len(args) == 1 and default_n is not None:
        series, n = args[0], default_n
    elif len(args) == 2 and isinstance(args[0], pd.Series):
        series, n = args
    else:
        raise RuleError(f"{name} takes {RULE_FUNCTIONS[name].split(':')[0]}")
    if isinstance(n, pd.Series) or not math.isfinite(float(n)) or not float(n).is_integer() or n < 1:
        raise RuleError(f"{name}: the window must be a whole number of bars, 1 or more (got {n})")
    if n > MAX_WINDOW:
        raise RuleError(f"{name}: a window of {n:g} bars is longer than any price history ({MAX_WINDOW:,} at most)")
    return _rb_num(series), int(n)


def _rb_num(value: "Value") -> "Value":
    """Conditions are True/False; in arithmetic and functions they count as 1/0
    (pandas would subtract True/False as XOR and difference them oddly)."""
    if isinstance(value, pd.Series) and value.dtype == bool:
        return value.astype(float)
    return value


def _rb_average(series: pd.Series, n: int, alpha: float) -> pd.Series:
    """A running average seeded the way TradingView seeds ta.ema and ta.rma: the
    simple mean of the first n values, then avg = alpha * x + (1 - alpha) * avg.
    Every value uses only the bars up to its own."""
    values = series.to_numpy(float)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < n:
        return pd.Series(np.nan, index=series.index)
    # Seeded at the n-th valid value with the mean of the first n valid values: a gap (a missing volume) early on
    # neither poisons the seed nor makes the start depend on how many bars come later.
    at = int(valid[n - 1])
    seeded = series.astype(float).copy()
    seeded.iloc[:at] = np.nan
    seeded.iloc[at] = float(np.mean(values[valid[:n]]))
    return seeded.ewm(alpha=alpha, adjust=False).mean()


def _rb_rsi(series: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI exactly as TradingView's ta.rsi: 100 when nothing fell, 0 when nothing rose."""
    delta = series.diff()
    up = _rb_average(delta.clip(lower=0), n, 1.0 / n)
    down = _rb_average((-delta).clip(lower=0), n, 1.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + up / down)
    rsi = rsi.where(up != 0, 0.0).where(down != 0, 100.0)
    return rsi.where(up.notna() & down.notna())


def _rb_call(name: str, args: List[Value], frame: pd.DataFrame) -> Value:
    args = [_rb_num(a) for a in args]
    if name in ("sma", "ema", "rsi", "highest", "lowest", "std"):
        series, n = _rb_window(name, args, frame)
        if name == "sma":
            return series.rolling(n, min_periods=n).mean()
        if name == "ema":
            return _rb_average(series, n, 2.0 / (n + 1.0))
        if name == "rsi":
            return _rb_rsi(series, n)
        if name == "highest":
            return series.rolling(n, min_periods=n).max()
        if name == "lowest":
            return series.rolling(n, min_periods=n).min()
        return series.rolling(n, min_periods=n).std(ddof=0)
    if name in ("change", "pct", "prev"):
        series, n = _rb_window(name, args, frame, default_n=1)
        if name == "change":
            return series - series.shift(n)
        if name == "pct":
            return 100.0 * (series / series.shift(n) - 1.0)
        return series.shift(n)
    if name == "atr":
        if len(args) != 1 or isinstance(args[0], pd.Series):
            raise RuleError("atr takes atr(n)")
        for col in ("high", "low"):
            if col not in frame:
                raise RuleError(f"atr needs a {col} column, and this price data has none")
        n = _rb_window("atr", args, frame)[1]
        prev_close = frame["close"].shift(1)
        true_range = pd.concat([frame["high"] - frame["low"], (frame["high"] - prev_close).abs(),
                                (frame["low"] - prev_close).abs()], axis=1).max(axis=1, skipna=False)
        true_range = true_range.where(prev_close.notna(), frame["high"] - frame["low"])
        return _rb_average(true_range, n, 1.0 / n)
    if name in ("crosses_above", "crosses_below"):
        if len(args) != 2:
            raise RuleError(f"{name} takes two things to compare: {name}(a, b)")
        a, b = (x if isinstance(x, pd.Series) else pd.Series(float(x), index=frame.index) for x in args)
        if name == "crosses_above":
            return (a > b) & (a.shift(1) <= b.shift(1))
        return (a < b) & (a.shift(1) >= b.shift(1))
    if name == "abs":
        if len(args) != 1:
            raise RuleError("abs takes one value")
        return abs(args[0])
    if name in ("min", "max"):
        if len(args) != 2:
            raise RuleError(f"{name} takes two values: {name}(a, b)")
        a, b = (x if isinstance(x, pd.Series) else pd.Series(float(x), index=frame.index) for x in args)
        both = pd.concat([a, b], axis=1)
        return both.min(axis=1, skipna=False) if name == "min" else both.max(axis=1, skipna=False)
    raise RuleError(f"unknown function {name!r}")  # never reached: names are checked first


def _rb_bool(value: Value, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value
        return value.fillna(0).astype(float) != 0
    return pd.Series(bool(value), index=frame.index)


def _rb_eval(node: ast.AST, frame: pd.DataFrame) -> Value:
    if isinstance(node, ast.Expression):
        return _rb_eval(node.body, frame)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in frame:
            raise RuleError(f"this price data has no {node.id} column")
        return frame[node.id].astype(float)
    if isinstance(node, ast.UnaryOp):
        value = _rb_eval(node.operand, frame)
        if isinstance(node.op, ast.Not):
            return ~_rb_bool(value, frame)
        return -_rb_num(value) if isinstance(node.op, ast.USub) else _rb_num(value)
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_rb_num(_rb_eval(node.left, frame)), _rb_num(_rb_eval(node.right, frame)))
    if isinstance(node, ast.Compare):
        left, out = _rb_eval(node.left, frame), None
        for op, comparator in zip(node.ops, node.comparators):
            right = _rb_eval(comparator, frame)
            part = _CMP[type(op)](left, right)
            part = part if isinstance(part, pd.Series) else pd.Series(bool(part), index=frame.index)
            out = part if out is None else (out & part)
            left = right
        return out
    if isinstance(node, ast.BoolOp):
        parts = [_rb_bool(_rb_eval(v, frame), frame) for v in node.values]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if isinstance(node.op, ast.And) else (out | p)
        return out
    if isinstance(node, ast.Call):
        return _rb_call(node.func.id, [_rb_eval(a, frame) for a in node.args], frame)
    raise RuleError(f"{type(node).__name__} is not allowed in a rule")


def _rb_check(node: ast.AST, text: str) -> None:
    """Allow only the rule language: numbers, price names, the functions, arithmetic, comparisons, and/or/not."""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Expression, ast.Load, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd)):
            continue
        if isinstance(sub, ast.Constant):
            if isinstance(sub.value, bool) or not isinstance(sub.value, (int, float)):
                raise RuleError(f"only numbers can appear as values in a rule (found {sub.value!r})")
            if not abs(sub.value) <= MAX_NUMBER:       # also catches nan and a 400-digit integer
                raise RuleError(f"a number in {text!r} is too big for a price rule")
            continue
        if isinstance(sub, ast.Name):
            if sub.id not in RULE_SERIES and sub.id not in RULE_FUNCTIONS:
                raise RuleError(f"unknown name {sub.id!r} in {text!r}: use {', '.join(RULE_SERIES)} or a function "
                                f"({', '.join(RULE_FUNCTIONS)})")
            continue
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name) or sub.func.id not in RULE_FUNCTIONS:
                raise RuleError(f"only these functions are allowed: {', '.join(RULE_FUNCTIONS)}")
            if sub.keywords:
                raise RuleError(f"{sub.func.id}: give the values in order, without names (like sma(close, 20))")
            continue
        if isinstance(sub, (ast.UnaryOp, ast.BoolOp, ast.Compare)):
            continue
        if isinstance(sub, ast.BinOp) and type(sub.op) in _BIN:
            continue
        if type(sub) in _BIN or type(sub) in _CMP:
            continue
        raise RuleError(f"{type(sub).__name__} is not allowed in a rule ({text!r}): use numbers, "
                        f"{', '.join(RULE_SERIES)}, the functions, + - * /, comparisons and and/or/not")
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in RULE_FUNCTIONS:
            parents = [p for p in ast.walk(node) if isinstance(p, ast.Call) and p.func is sub]
            if not parents:
                raise RuleError(f"{sub.id} is a function: call it, like {sub.id}(20)")


def compile_rule(text: str) -> Callable[[pd.DataFrame], pd.Series]:
    """A rule's condition, ready to run on price data: True on the bars where it holds.

    Raises RuleError with a plain explanation when the text isn't a valid condition."""
    if not isinstance(text, str) or not text.strip():
        raise RuleError("a rule is a condition in words like: rsi(14) < 30")
    text = text.strip()
    if len(text) > MAX_RULE_LENGTH:
        raise RuleError(f"a rule can be at most {MAX_RULE_LENGTH} characters")
    if "{" in text:
        raise RuleError(f"{text!r} has a {{setting}} with no value: put a number in, or try values with tune()")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"can't read {text!r}: {exc.msg}. Write it like: rsi(14) < 30 and close > sma(200)") from None
    _rb_check(tree, text)
    if not isinstance(tree.body, (ast.Compare, ast.BoolOp)) and not (isinstance(tree.body, ast.UnaryOp)
                                                                      and isinstance(tree.body.op, ast.Not)) \
            and not (isinstance(tree.body, ast.Call) and tree.body.func.id in ("crosses_above", "crosses_below")):
        raise RuleError(f"{text!r} is a number, not a condition: compare it, like {text} < 30")

    def condition(frame: pd.DataFrame) -> pd.Series:
        try:
            with np.errstate(all="ignore"):
                return _rb_bool(_rb_eval(tree, frame), frame).reindex(frame.index).fillna(False).astype(bool)
        except RuleError:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:   # 1/0, an overflow, a mismatched function call
            raise RuleError(f"can't work out {text!r} on these prices: {exc}") from None

    condition.text = text  # type: ignore[attr-defined]
    return condition


@dataclass(frozen=True)
class Rule:
    """When to get in and when to get out. Decisions are made at a bar's close and
    filled at the next bar's open (see simulate.backtest)."""
    buy: str
    sell: Optional[str] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_hold_bars: Optional[int] = None
    side: str = "long"
    name: Optional[str] = None
    params: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        enter = "Sell short" if self.side == "short" else "Buy"
        parts = [f"{enter} when {self.buy}"]
        exits = []
        if self.sell:
            exits.append(f"when {self.sell}")
        if self.stop_loss_pct:
            exits.append(f"at a {self.stop_loss_pct:g}% loss")
        if self.take_profit_pct:
            exits.append(f"at a {self.take_profit_pct:g}% gain")
        if self.max_hold_bars:
            exits.append(f"after {self.max_hold_bars} bars")
        return parts[0] + "; get out " + ", or ".join(exits) + "."


_RULE_KEYS = {"buy", "sell", "stop_loss_pct", "take_profit_pct", "max_hold_bars", "side", "name"}


def rule_from(value: Union[Rule, Mapping[str, Any]], params: Optional[Mapping[str, float]] = None) -> Rule:
    """A Rule from a Rule or a dict like {"buy": "...", "sell": "..."}; with params,
    "{name}" placeholders in the conditions are filled in (numbers only)."""
    if isinstance(value, Rule):
        data: Dict[str, Any] = {k: getattr(value, k) for k in _RULE_KEYS}
    elif isinstance(value, Mapping):
        unknown = set(value) - _RULE_KEYS
        if unknown:
            raise RuleError(f"unknown rule part(s) {sorted(unknown)}: use {sorted(_RULE_KEYS)}")
        data = dict(value)
    else:
        raise RuleError('a rule is a dict like {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}')
    params = dict(params or {})
    for key, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise RuleError(f"setting {key} must be a number (got {v!r})")
    for key in ("buy", "sell"):
        if isinstance(data.get(key), str) and params:
            text = data[key]
            if len(text) > MAX_RULE_LENGTH:
                raise RuleError(f"a rule can be at most {MAX_RULE_LENGTH} characters")
            if re.search(r"\{(?!\w+\})|(?<!\{\w)\}", re.sub(r"\{\w+\}", "", text)):
                raise RuleError(f"the {key} rule has a brace that isn't a {{setting}}: write settings like {{low}}")

            def fill(m: "re.Match[str]") -> str:
                name = m.group(1)
                if name not in params:
                    raise RuleError(f"the {key} rule names the setting {name!r}, which has no values: add it to the "
                                    "grid")
                v = float(params[name])
                return str(int(v)) if v.is_integer() else repr(v)
            # A plain substitution: str.format would also run format specs like {low:>300000000}.
            data[key] = re.sub(r"\{(\w+)\}", fill, text)
    for key in ("stop_loss_pct", "take_profit_pct", "max_hold_bars"):
        v = data.get(key)
        if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
            setting = v.strip()[1:-1]
            if setting not in params:
                raise RuleError(f"{key} names the setting {setting!r}, which has no values: add it to the grid")
            data[key] = params[setting]
        if data.get(key) is not None:
            try:
                v = float(data[key])
            except (TypeError, ValueError):
                raise RuleError(f"{key} must be a number (got {data[key]!r})") from None
            if not np.isfinite(v) or v <= 0:
                raise RuleError(f"{key} must be a positive number (got {data[key]!r})")
            if key == "max_hold_bars" and (not v.is_integer() or v > MAX_WINDOW):
                raise RuleError(f"max_hold_bars is a whole number of bars, 1 or more (got {data[key]!r})")
            data[key] = int(v) if key == "max_hold_bars" else v
    side = str(data.get("side") or "long").lower()
    if side not in ("long", "short"):
        raise RuleError('side is "long" or "short"')
    if not data.get("buy"):
        raise RuleError('say when to get in: {"buy": "rsi(14) < 30", ...}')
    if not (data.get("sell") or data.get("stop_loss_pct") or data.get("take_profit_pct") or data.get("max_hold_bars")):
        raise RuleError("say when to get out: a sell condition, or stop_loss_pct, take_profit_pct or max_hold_bars")
    rule = Rule(buy=str(data["buy"]).strip(), sell=(str(data["sell"]).strip() if data.get("sell") else None),
                stop_loss_pct=data.get("stop_loss_pct"), take_profit_pct=data.get("take_profit_pct"),
                max_hold_bars=data.get("max_hold_bars"), side=side, name=data.get("name"),
                params={k: float(v) for k, v in params.items()})
    compile_rule(rule.buy)
    if rule.sell:
        compile_rule(rule.sell)
    return rule


__all__ = ["Rule", "RuleError", "RULE_FUNCTIONS", "RULE_SERIES", "compile_rule", "rule_from"]
