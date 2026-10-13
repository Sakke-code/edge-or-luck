"""Forgiving readers: whatever a trader or their AI hands over, find the trades.

Accepted: a file path (CSV, TSV, TXT, XLSX, freqtrade JSON or ZIP), CSV/TSV
text or bytes, a pandas DataFrame, a list of P&L numbers (or number strings
like "+12.5%"), a list of dicts, a dict of lists, and, by duck typing and
without importing those libraries: a backtesting.py ``stats`` Series or its
``_trades`` frame, a vectorbt Portfolio or its ``trades.records_readable``
frame, and a freqtrade backtest-result dict.

Recognised layouts: TradingView strategy-tester "List of trades" (old and new
column names, entry and exit rows paired), any table with a P&L column (money
or percent; a strategy column splits it), and raw exchange fills (time, pair,
side, price, quantity, fee) paired into round trips (see fills.py).

Every reader returns canonical trade columns; ``build_trades`` turns those into
Trades. Nothing here raises on messy input: a reader that cannot find trades
says what it saw and what it needs.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fills import round_trips, side_of, split_amount, split_pair

MAX_TRADES = 200_000
MAX_STRATEGIES = 50

CURRENCIES = ("USDT", "USDC", "FDUSD", "BUSD", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "BTC", "ETH",
              "BNB", "SEK", "NOK", "DKK", "PLN", "INR", "TRY", "BRL", "DAI")
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}


@dataclass
class Trades:
    """One strategy's closed trades, oldest first."""
    name: str
    unit: str                                   # "%" or a currency code or "money" (currency unknown)
    pnl: np.ndarray                             # result per closed trade, in ``unit``
    times: Optional[List[pd.Timestamp]] = None  # close times
    entry_times: Optional[List[pd.Timestamp]] = None
    pct: Optional[np.ndarray] = None            # percent per trade, when ``unit`` is money
    fees: Optional[np.ndarray] = None           # fees per trade, in ``unit``
    side: Optional[np.ndarray] = None           # +1 long, -1 short
    size: Optional[np.ndarray] = None           # position value at entry, in ``unit`` (money only)
    entry_price: Optional[np.ndarray] = None
    exit_price: Optional[np.ndarray] = None
    symbols: Optional[List[str]] = None
    strategy_return_pct: Optional[float] = None
    benchmark_pct: Optional[float] = None       # buy-and-hold return from the input itself
    benchmark_source: Optional[str] = None
    fees_included: Optional[bool] = None        # known from the format
    buys_per_trade: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ReadResult:
    strategies: List[Trades]
    format: str
    rows: int = 0
    columns_seen: List[str] = field(default_factory=list)
    columns_used: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    need: Optional[Dict[str, Any]] = None
    source_hint: Optional[str] = None           # "backtest" when the format can only be one


# ── cells ─────────────────────────────────────────────────────────────────────

def number(value: Any) -> float:
    """A number from an export cell: percent signs, currency marks, unicode
    minus, (negatives), thousands separators, decimal commas, trailing coin
    names ("0.0012BTC")."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    for ch in ("−", "–", "—"):
        text = text.replace(ch, "-")
    text = text.replace("\xa0", "").replace(" ", "").replace(" ", "").replace("'", "")
    if not text or text.lower() in ("nan", "none", "null", "-", "n/a", "na"):
        return float("nan")
    negative = (text.startswith("(") and text.endswith(")")) or text.endswith("-")
    text = text.strip("()")
    text = re.sub(r"[A-Za-z]+$", "", text) if re.search(r"\d", text) and not re.search(r"\d[eE][-+]?\d", text) \
        else text
    text = re.sub(r"[^0-9,.\-+eE]", "", text)
    if text.count("-") > 1 or (text.endswith("-") and len(text) > 1):
        text = text.rstrip("-")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        if text.count(",") > 1 or (len(tail) == 3 and head.lstrip("+-").isdigit() and head.lstrip("+-") != "0"):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    try:
        out = float(text)
    except ValueError:
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return -abs(out) if negative else out


def norm(header: Any) -> str:
    """A header in plain lower-case words: "Date(UTC)" -> "date utc",
    "ReturnPct" -> "return pct", "Net P&L USD" -> "net p&l usd"."""
    text = str(header).replace("﻿", "").strip()
    text = re.sub(r"(?i)pnl", " pnl ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w%&/#$€£ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(name: str) -> List[str]:
    return name.split()


def _currency_in(text: str) -> Optional[str]:
    up = str(text).upper()
    for sym, code in SYMBOLS.items():
        if sym in up:
            return code
    for code in CURRENCIES:
        if re.search(rf"(?<![A-Z]){code}(?![A-Z])", up):
            return code
    return None


# ── columns by role ───────────────────────────────────────────────────────────

_NOT_RESULT = {"cum", "cumulative", "run", "runup", "drawdown", "max", "mae", "mfe", "excursion", "favorable",
               "adverse", "unrealized", "unrealised", "open", "fee", "fees", "commission", "total", "equity",
               "balance", "avg", "average", "best", "worst", "trade", "size", "price", "qty", "quantity"}
_PERCENTISH = {"%", "pct", "percent", "percentage", "ratio", "return", "roi", "change"}
_RESULTISH = {"pnl", "p&l", "profit", "p/l", "pl", "gain", "result", "return", "roi", "net", "change", "ratio",
              "profit/loss", "loss"}


def _is_pct_name(name: str) -> bool:
    toks = set(_tokens(name))
    return "%" in name or bool(toks & {"pct", "percent", "percentage", "ratio", "roi"})


def find_result_columns(columns: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """(money P&L column, percent P&L column) by name."""
    money: List[Tuple[int, str]] = []
    pct: List[Tuple[int, str]] = []
    for col in columns:
        name = norm(col)
        toks = set(_tokens(name.replace("%", " % ")))
        if not toks & _RESULTISH:
            continue
        blocked = toks & _NOT_RESULT
        if blocked - {"trade"} and not (blocked <= {"total"} and "net" in toks):
            continue
        if "trade" in blocked and not (toks & {"pnl", "p&l", "profit"}):
            continue
        rank = 0 if toks & {"net", "realized", "realised", "closed"} else 1 if toks & {"pnl", "p&l", "profit"} else 2
        if _is_pct_name(name) or toks == {"return"} or toks <= {"return", "pct"}:
            pct.append((rank, col))
        elif toks & {"pnl", "p&l", "profit", "p/l", "pl", "gain", "result", "profit/loss", "loss", "net"}:
            money.append((rank, col))
    money.sort(key=lambda t: t[0])
    pct.sort(key=lambda t: t[0])
    return (money[0][1] if money else None), (pct[0][1] if pct else None)


def find_column(columns: Sequence[str], candidates: Sequence[str], *, avoid: Iterable[str] = ()) -> Optional[str]:
    """The first column whose normalized name equals a candidate, or starts
    with it followed by a unit word ("price usdt")."""
    avoid = set(avoid)
    normed = [(c, norm(c)) for c in columns if c not in avoid]
    for cand in candidates:
        for col, name in normed:
            if name == cand:
                return col
    for cand in candidates:
        for col, name in normed:
            if name.startswith(cand + " ") and not re.search(r"\b(entry|exit|open|close)\b", name[len(cand):]):
                return col
    return None


_CLOSE_TIME = ("close time", "closed at", "close date", "exit time", "exit date", "exit timestamp", "close date utc",
               "closing time", "closed", "date/time", "date and time", "datetime", "date time", "time", "date",
               "timestamp", "date utc", "time utc", "created time", "trade time", "fill time", "order time",
               "updated time", "execution time", "c time", "ctime", "exit")
_OPEN_TIME = ("open time", "opened at", "open date", "entry time", "entry date", "entry timestamp", "opening time",
              "opened", "entry")
_STRATEGY = ("strategy", "strategy name", "bot", "bot name", "system", "setup", "algo", "model", "book", "tag",
             "label", "variant", "column")
_FEE = ("fees", "fee", "commission", "commissions", "trading fee", "trading fees", "transaction fee", "fee amount",
        "total fee", "fees paid", "fee usdt", "fees usdt")
_SIDE = ("side", "direction", "order side", "trade side", "buy/sell", "b/s", "position side", "type")
_SYMBOL = ("symbol", "pair", "market", "trading pair", "instrument", "ticker", "coin pair", "contract", "product",
           "asset", "coin", "currency pair")
_PRICE = ("price", "avg price", "average price", "fill price", "filled price", "executed price", "execution price",
          "deal price", "trade price", "price avg", "avg fill price", "average filled price", "filled price avg",
          "order price")
_QTY = ("executed", "filled", "quantity", "qty", "size", "filled amount", "filled qty", "filled quantity",
        "executed qty", "executed quantity", "base amount", "fill size", "fill quantity", "volume", "vol",
        "deal amount", "amount")
_QUOTE_TOTAL = ("total", "turnover", "value", "filled value", "trade value", "quote amount", "deal value",
                "filled total", "notional", "cost", "trade amount", "volume usdt", "total usdt", "amount usdt")
_FEE_COIN = ("fee coin", "fee currency", "fee asset", "commission asset", "fee ccy", "fee unit", "fee token",
             "commission currency")


# ── turning anything into a table ─────────────────────────────────────────────

def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _has_words(text: str) -> bool:
    stripped = re.sub(r"(?i)\b(usdt|usdc|usd|eur|gbp|pct|e)\b", "", text)
    return bool(re.search(r"[A-Za-z]{2,}", stripped))


def numbers_from_text(text: str) -> Optional[Tuple[List[float], Optional[str]]]:
    """A pasted list of results ("+12.5, -3, 4.1" or one per line); None when
    the text is not just numbers. Returns (values, unit or None)."""
    if _has_words(text):
        return None
    body = text.strip().strip("[]")
    if not body:
        return None
    if ";" in body:
        parts = re.split(r"[;\n\r]+", body)
    elif "\t" in body:
        parts = re.split(r"[\t\n\r]+", body)
    elif re.search(r"\d,\d", body) and not re.search(r",\s", body) and re.search(r"\s", body.strip()):
        parts = re.split(r"\s+", body)                         # "12,5 -3,2": decimal commas
    elif "\n" in body.strip() and not re.search(r",\s", body):
        parts = [p for line in body.splitlines() for p in re.split(r"\s+(?=[-+−(]?\d)", line.strip())]
    else:
        parts = re.split(r"[,\s]+", body)
    parts = [p.strip() for p in parts if p.strip()]
    if any(re.fullmatch(r"[A-Za-z]+", p) for p in parts):
        return None
    values = [number(p) for p in parts]
    if not values or sum(math.isfinite(v) for v in values) < max(1, 0.9 * len(values)):
        return None
    unit = "%" if "%" in body else _currency_in(body)
    return [v for v in values if math.isfinite(v)], unit


def _sniff(lines: List[str]) -> str:
    best, best_score = ",", -1.0
    sample = [ln for ln in lines[:60] if ln.strip()]
    for delim in (",", ";", "\t", "|"):
        counts = [len(row) for row in csv.reader(sample, delimiter=delim)]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        score = counts.count(modal) / len(counts) * min(modal, 12)
        if score > best_score:
            best, best_score = delim, score
    return best


def table_from_text(text: str) -> pd.DataFrame:
    """A table from CSV/TSV/semicolon text, skipping title lines above the header."""
    text = text.replace("﻿", "")
    lines = text.splitlines()
    delim = _sniff(lines)
    rows = list(csv.reader([ln for ln in lines if ln.strip()], delimiter=delim))
    if not rows:
        return pd.DataFrame()
    counts = [len(r) for r in rows[:60]]
    modal = max(set(counts), key=counts.count)
    header_at = 0
    for i, row in enumerate(rows[:15]):
        cells = [c for c in row if c.strip()]
        if len(row) >= min(modal, 2) and cells and sum(not math.isfinite(number(c)) for c in cells) >= len(cells) / 2:
            header_at = i
            break
    header = [h.strip() or f"column {j + 1}" for j, h in enumerate(rows[header_at])]
    if header and all(math.isfinite(number(h)) for h in header if h.strip()):
        header = [f"column {j + 1}" for j in range(len(header))]
        body = rows[header_at:]
    else:
        body = rows[header_at + 1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body if any(c.strip() for c in r)]
    return pd.DataFrame(body, columns=_dedupe(header), dtype=object)


def _dedupe(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def _excel_tables(source: Any) -> Dict[str, pd.DataFrame]:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ValueError("reading .xlsx needs openpyxl (pip install 'edgeorluck[xlsx]'), or save the sheet as "
                         "CSV") from exc
    sheets = pd.read_excel(source, sheet_name=None, dtype=object, header=None)
    out = {}
    for name, frame in sheets.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        if frame.empty:
            continue
        text = frame.astype(object).where(frame.notna(), "").astype(str)
        buf = io.StringIO()
        text.to_csv(buf, index=False, header=False)
        out[str(name)] = table_from_text(buf.getvalue())
    return out


# ── adapters for what AI-written backtests return (duck typing, no imports) ──

def _is_frame(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame)


def _backtesting_trades(obj: Any) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """backtesting.py: the stats Series from Backtest.run() or its _trades."""
    if isinstance(obj, pd.Series) and "_trades" in obj.index:
        trades = obj["_trades"]
        meta: Dict[str, Any] = {}
        for key, out in (("Buy & Hold Return [%]", "benchmark_pct"), ("Return [%]", "strategy_return_pct")):
            if key in obj.index:
                value = number(obj[key])
                if math.isfinite(value):
                    meta[out] = value
        if "_strategy" in obj.index:
            meta["name"] = re.sub(r"\(.*", "", str(obj["_strategy"])).strip() or None
        if _is_frame(trades):
            return trades, meta
    if _is_frame(obj) and {"EntryBar", "ExitBar", "PnL", "ReturnPct"} <= set(map(str, obj.columns)):
        return obj, {}
    return None


def _vectorbt_trades(obj: Any) -> Optional[pd.DataFrame]:
    if _is_frame(obj):
        cols = set(map(str, obj.columns))
        if {"PnL", "Return"} <= cols and ({"Entry Timestamp", "Exit Timestamp"} <= cols
                                          or {"Entry Date", "Exit Date"} <= cols or "Exit Trade Id" in cols):
            return obj
        return None
    trades = getattr(obj, "trades", None)
    readable = getattr(trades, "records_readable", None) if trades is not None else None
    if readable is not None and _is_frame(readable):
        return readable
    return None


def _freqtrade_result(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("strategy"), dict):
        if any(isinstance(v, dict) and isinstance(v.get("trades"), list) for v in obj["strategy"].values()):
            return obj
    return None


def _from_backtesting(trades: pd.DataFrame, meta: Dict[str, Any], name: Optional[str]) -> ReadResult:
    cols = list(map(str, trades.columns))
    canon = pd.DataFrame({
        "pnl": trades["PnL"].map(number),
        "pct": trades["ReturnPct"].map(number) * 100.0,
        "time": _times(trades["ExitTime"]) if "ExitTime" in cols else pd.NaT,
        "entry_time": _times(trades["EntryTime"]) if "EntryTime" in cols else pd.NaT,
        "side": np.sign(trades["Size"].map(number)) if "Size" in cols else np.nan,
        "entry_price": trades["EntryPrice"].map(number) if "EntryPrice" in cols else np.nan,
        "exit_price": trades["ExitPrice"].map(number) if "ExitPrice" in cols else np.nan,
    })
    if "Size" in cols and "EntryPrice" in cols:
        canon["size"] = (trades["Size"].map(number).abs() * trades["EntryPrice"].map(number)).to_numpy()
    fees_known = None
    if "Commission" in cols:
        fee = trades["Commission"].map(number).abs()
        canon["fee"] = fee.to_numpy()
        fees_known = bool(np.nansum(fee.to_numpy()) > 0) or None
    label = name or meta.get("name") or "Backtest"
    strategies = build_trades(canon, unit="money", name=label)
    for st in strategies:
        st.benchmark_pct = meta.get("benchmark_pct")
        st.benchmark_source = "backtesting.py's Buy & Hold Return [%]" if st.benchmark_pct is not None else None
        st.strategy_return_pct = meta.get("strategy_return_pct")
        st.fees_included = fees_known
    used = {"result": "PnL", "percent": "ReturnPct (a fraction, x100)", "close_time": "ExitTime",
            "entry_time": "EntryTime"}
    return ReadResult(strategies, "backtesting.py", rows=len(trades), columns_seen=cols, columns_used=used,
                      source_hint="backtest")


def _from_vectorbt(trades: pd.DataFrame, name: Optional[str]) -> ReadResult:
    cols = list(map(str, trades.columns))
    warnings = []
    frame = trades
    if "Status" in cols:
        is_open = frame["Status"].astype(str).str.lower().eq("open")
        if is_open.any():
            warnings.append(f"{int(is_open.sum())} trade(s) still open; they are not judged.")
        frame = frame[~is_open]
    et = "Entry Timestamp" if "Entry Timestamp" in cols else "Entry Date" if "Entry Date" in cols else None
    xt = "Exit Timestamp" if "Exit Timestamp" in cols else "Exit Date" if "Exit Date" in cols else None
    ep = "Avg Entry Price" if "Avg Entry Price" in cols else "Entry Price" if "Entry Price" in cols else None
    xp = "Avg Exit Price" if "Avg Exit Price" in cols else "Exit Price" if "Exit Price" in cols else None
    canon = pd.DataFrame({
        "pnl": frame["PnL"].map(number).to_numpy(),
        "pct": (frame["Return"].map(number) * 100.0).to_numpy(),
        "time": _times(frame[xt]).to_numpy() if xt else pd.NaT,
        "entry_time": _times(frame[et]).to_numpy() if et else pd.NaT,
        "entry_price": frame[ep].map(number).to_numpy() if ep else np.nan,
        "exit_price": frame[xp].map(number).to_numpy() if xp else np.nan,
    })
    if "Direction" in cols:
        canon["side"] = np.where(frame["Direction"].astype(str).str.lower().str.startswith("short"), -1, 1)
    if "Size" in cols and ep:
        canon["size"] = (frame["Size"].map(number).abs() * frame[ep].map(number)).to_numpy()
    fee_cols = [c for c in ("Entry Fees", "Exit Fees") if c in cols]
    fees_known = None
    if fee_cols:
        canon["fee"] = sum(frame[c].map(number).abs().to_numpy() for c in fee_cols)
        fees_known = bool(np.nansum(canon["fee"].to_numpy()) > 0) or None
    if "Column" in cols and frame["Column"].astype(str).nunique() > 1:
        canon["strategy"] = frame["Column"].astype(str).to_numpy()
    strategies = build_trades(canon, unit="money", name=name or "Backtest")
    for st in strategies:
        st.fees_included = fees_known
    if "strategy" in canon:
        warnings.append(f"The portfolio has {len(strategies)} columns (parameter settings); each is judged as one "
                        "version, and all of them count as versions tried.")
    used = {"result": "PnL", "percent": "Return (a fraction, x100)"}
    return ReadResult(strategies, "vectorbt", rows=len(trades), columns_seen=cols, columns_used=used,
                      warnings=warnings, source_hint="backtest")


def _from_freqtrade(result: Dict[str, Any], name: Optional[str]) -> ReadResult:
    strategies: List[Trades] = []
    rows = 0
    seen: List[str] = []
    for strat_name, body in result["strategy"].items():
        if not isinstance(body, dict) or not isinstance(body.get("trades"), list):
            continue
        frame = pd.DataFrame(body["trades"])
        rows += len(frame)
        seen = sorted(set(seen) | set(map(str, frame.columns)))
        if frame.empty or "profit_abs" not in frame:
            continue
        canon = pd.DataFrame({
            "pnl": frame["profit_abs"].map(number),
            "pct": frame["profit_ratio"].map(number) * 100.0 if "profit_ratio" in frame else np.nan,
            "time": _times(frame["close_date"]) if "close_date" in frame else pd.NaT,
            "entry_time": _times(frame["open_date"]) if "open_date" in frame else pd.NaT,
            "entry_price": frame["open_rate"].map(number) if "open_rate" in frame else np.nan,
            "exit_price": frame["close_rate"].map(number) if "close_rate" in frame else np.nan,
            "symbol": frame["pair"].astype(str) if "pair" in frame else None,
            "side": np.where(frame["is_short"].astype(bool), -1, 1) if "is_short" in frame else 1,
        })
        if "stake_amount" in frame:
            canon["size"] = frame["stake_amount"].map(number)
        unit = str(body.get("stake_currency") or "money").upper()
        for st in build_trades(canon, unit=unit, name=name or str(strat_name)):
            st.fees_included = True
            total = number(body.get("profit_total"))
            if math.isfinite(total):
                st.strategy_return_pct = 100.0 * total
            change = number(body.get("market_change"))
            if math.isfinite(change):
                st.benchmark_pct = 100.0 * change
                st.benchmark_source = "freqtrade's market change (the average of the pairs, rough)"
            strategies.append(st)
    used = {"result": "profit_abs (net of fees)", "percent": "profit_ratio (x100)", "close_time": "close_date"}
    return ReadResult(strategies, "freqtrade", rows=rows, columns_seen=seen, columns_used=used, source_hint="backtest")


# ── tables of trades ──────────────────────────────────────────────────────────

def _times(series: pd.Series) -> pd.Series:
    """Datetimes from cells: ISO text, "Jan 5, 2024 14:30", day-first dates,
    epoch seconds or milliseconds. Unparseable cells become NaT."""
    s = pd.Series(series).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
        return out.dt.tz_localize("UTC") if out.dt.tz is None else out.dt.tz_convert("UTC")
    nums = s.map(lambda v: number(v) if re.fullmatch(r"\s*\d{9,13}(\.\d+)?\s*", str(v)) else float("nan"))
    if nums.notna().sum() >= max(1, 0.8 * s.astype(str).str.strip().ne("").sum()):
        unit = "ms" if nums.median() > 1e11 else "s"
        return pd.to_datetime(nums, unit=unit, errors="coerce", utc=True)
    text = s.astype(str).str.strip().str.replace(r"\s*\((UTC|GMT)[^)]*\)", "", regex=True)
    dayfirst = bool(text.str.match(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}").mean() > 0.5) and not bool(
        text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").mean() > 0.5 and _us_dates(text))
    try:
        out = pd.to_datetime(text, errors="coerce", utc=True, format="mixed", dayfirst=dayfirst)
    except (TypeError, ValueError):
        out = pd.to_datetime(text, errors="coerce", utc=True)
    return out


def _us_dates(text: pd.Series) -> bool:
    first = text.str.extract(r"^(\d{1,2})/(\d{1,2})/")
    try:
        return bool((first[1].astype(float) > 12).any())
    except (TypeError, ValueError):
        return False


def _tradingview(frame: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    """TradingView "List of trades": an entry row and an exit row per trade."""
    cols = list(frame.columns)
    kind = find_column(cols, ("type",))
    number_col = find_column(cols, ("trade #", "trade", "trade no", "trade number", "#"))
    if kind is None or number_col is None:
        return None
    types = frame[kind].astype(str).str.strip().str.lower()
    if not types.str.startswith(("entry", "exit")).any():
        return None
    money, pct = find_result_columns(cols)
    when = find_column(cols, ("date/time", "date and time", "date time", "datetime", "time", "date"))
    price = find_column(cols, ("price",))
    qty = find_column(cols, ("position size qty", "contracts", "quantity", "qty"))
    value = find_column(cols, ("position size value",))
    cum_pct = next((c for c in cols if re.match(r"^cum(ulative)?\s.*(%|pct)", norm(c))), None)
    out = []
    for trade_no, group in frame.groupby(frame[number_col].astype(str).str.strip(), sort=False):
        t = types.loc[group.index]
        entry = group[t.str.startswith("entry")]
        exit_ = group[t.str.startswith("exit")]
        if exit_.empty:
            continue
        e_row, x_row = (entry.iloc[0] if len(entry) else None), exit_.iloc[-1]
        side_text = (t.loc[x_row.name] if x_row is not None else "")
        row = {
            "trade_no": number(trade_no),
            "pnl": number(x_row[money]) if money else float("nan"),
            "pct": number(x_row[pct]) if pct else float("nan"),
            "time": x_row[when] if when else None,
            "entry_time": e_row[when] if (when and e_row is not None) else None,
            "side": -1 if "short" in side_text else 1,
            "entry_price": number(e_row[price]) if (price and e_row is not None) else float("nan"),
            "exit_price": number(x_row[price]) if price else float("nan"),
            "cum_pct": number(x_row[cum_pct]) if cum_pct else float("nan"),
        }
        if value is not None and e_row is not None:
            row["size"] = number(e_row[value])
        elif qty is not None and price is not None and e_row is not None:
            row["size"] = abs(number(e_row[qty])) * number(e_row[price])
        if not math.isfinite(row["pnl"]) and not math.isfinite(row["pct"]):
            continue
        out.append(row)
    if not out:
        return None
    canon = pd.DataFrame(out)
    canon["time"] = _times(canon["time"]) if when else pd.NaT
    canon["entry_time"] = _times(canon["entry_time"]) if when else pd.NaT
    if canon["time"].isna().all():
        canon = canon.sort_values("trade_no", kind="stable")
    used = {k: v for k, v in (("result", money), ("percent", pct), ("close_time", when), ("type", kind),
                              ("trade_number", number_col), ("price", price)) if v}
    if money is None:
        canon["pnl"] = float("nan")
    return canon, used


def _looks_like_fills(cols: Sequence[str], frame: pd.DataFrame) -> Optional[Dict[str, str]]:
    side = find_column(cols, _SIDE)
    price = find_column(cols, _PRICE)
    if side is None or price is None:
        return None
    sides = frame[side].map(side_of)
    if sides.notna().mean() < 0.8:
        return None
    has_executed = find_column(cols, ("executed", "executed qty", "filled", "filled amount", "filled qty")) is not None
    qty = find_column(cols, _QTY if not has_executed else _QTY[:-1], avoid=[price])
    total_candidates = _QUOTE_TOTAL + (("amount",) if has_executed else ())
    total = find_column(cols, total_candidates, avoid=[c for c in (price, qty) if c])
    if qty is None and total is None:
        return None
    out = {"side": side, "price": price}
    for key, value in (("qty", qty), ("quote_total", total), ("time", find_column(cols, _CLOSE_TIME)),
                       ("pair", find_column(cols, _SYMBOL)), ("fee", find_column(cols, _FEE)),
                       ("fee_coin", find_column(cols, _FEE_COIN))):
        if value is not None:
            out[key] = value
    return out


def _from_fills(frame: pd.DataFrame, colmap: Dict[str, str], name: str) -> ReadResult:
    rows = []
    pair_col = colmap.get("pair")
    times = _times(frame[colmap["time"]]) if "time" in colmap else pd.Series([pd.NaT] * len(frame))
    for i, (_, r) in enumerate(frame.iterrows()):
        side = side_of(r[colmap["side"]])
        if side is None:
            continue
        qty_text, qty_coin = split_amount(r[colmap["qty"]]) if "qty" in colmap else (None, None)
        total_text, total_coin = split_amount(r[colmap["quote_total"]]) if "quote_total" in colmap else (None, None)
        fee_text, fee_coin = split_amount(r[colmap["fee"]]) if "fee" in colmap else (None, None)
        if "fee_coin" in colmap and str(r[colmap["fee_coin"]]).strip():
            fee_coin = str(r[colmap["fee_coin"]]).strip().upper()
        pair = str(r[pair_col]).strip() if pair_col else ""
        if not pair and qty_coin and total_coin:
            pair = f"{qty_coin}/{total_coin}"
        rows.append({"time": times.iloc[i], "pair": pair or "?", "side": side,
                     "price": number(r[colmap["price"]]), "qty": number(qty_text),
                     "fee": number(fee_text), "fee_coin": fee_coin, "quote_total": number(total_text)})
    trips, warnings, facts = round_trips(rows, number)
    quotes = [q for q in facts["quotes"] if q]
    unit = quotes[0] if len(quotes) == 1 else ("money" if not quotes else quotes[0])
    if len(quotes) > 1:
        main = trips["quote"].value_counts().index[0] if len(trips) else quotes[0]
        warnings.append(f"Pairs are quoted in {', '.join(quotes)}; only the {main} pairs are judged (results in "
                        "different coins can't be added up).")
        trips = trips[trips["quote"] == main]
        unit = main
    strategies = build_trades(trips, unit=unit, name=name) if len(trips) else []
    for st in strategies:
        st.fees_included = "fee" in colmap
    if len(trips) and "buys" in trips:
        for st in strategies:
            st.buys_per_trade = trips["buys"].fillna(1).to_numpy(dtype=float)
    used = dict(colmap)
    return ReadResult(strategies, "fills", rows=len(frame), columns_seen=list(map(str, frame.columns)),
                      columns_used=used, warnings=warnings)


def _from_table(frame: pd.DataFrame, name: str, *, unit: Optional[str], kind: Optional[str]) -> ReadResult:
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    cols = list(frame.columns)
    if frame.empty:
        return _need(cols, "The table has no rows.")
    if kind in (None, "auto", "tradingview"):
        tv = _tradingview(frame)
        if tv is not None:
            canon, used = tv
            money_unit = _currency_in(used.get("result", "")) or "money"
            strategies = build_trades(canon, unit=money_unit if used.get("result") else "%", name=name,
                                      prefer=unit)
            cum = canon["cum_pct"].dropna() if "cum_pct" in canon else pd.Series(dtype=float)
            for st in strategies:
                st.strategy_return_pct = float(cum.iloc[-1]) if len(cum) else None
            return ReadResult(strategies, "tradingview", rows=len(frame), columns_seen=cols, columns_used=used,
                              source_hint="backtest")
    money, pct = find_result_columns(cols)
    if kind == "fills" or (kind in (None, "auto") and money is None and pct is None):
        colmap = _looks_like_fills(cols, frame)
        if colmap is not None:
            return _from_fills(frame, colmap, name)
        if kind == "fills":
            return _need(cols, "These rows don't look like fills: we need a side (buy/sell), a price and a quantity.")
    if money is None and pct is None:
        numeric = [c for c in cols if frame[c].map(number).notna().mean() > 0.9]
        if len(cols) == 1 and numeric:
            money = cols[0]
        elif len(numeric) == 1 and len(cols) <= 3:
            money = numeric[0]
        else:
            return _need(cols, "We couldn't find a profit/P&L column, or buy/sell fills.")
    when = find_column(cols, _CLOSE_TIME, avoid=[c for c in (money, pct) if c])
    entry = find_column(cols, _OPEN_TIME, avoid=[c for c in (money, pct, when) if c])
    if entry == when:
        entry = None
    group = find_column(cols, _STRATEGY, avoid=[c for c in (money, pct, when) if c])
    fee = find_column(cols, _FEE, avoid=[c for c in (money, pct) if c])
    side = find_column(cols, ("side", "direction", "position side", "position"), avoid=[c for c in (money, pct) if c])
    symbol = find_column(cols, _SYMBOL, avoid=[c for c in (money, pct, group) if c])
    canon = pd.DataFrame(index=frame.index)
    canon["pnl"] = frame[money].map(number) if money else np.nan
    canon["pct"] = frame[pct].map(number) if pct else np.nan
    if pct and "ratio" in norm(pct) and canon["pct"].abs().max() <= 5:
        canon["pct"] = canon["pct"] * 100.0
    canon["time"] = _times(frame[when]).to_numpy() if when else pd.NaT
    canon["entry_time"] = _times(frame[entry]).to_numpy() if entry else pd.NaT
    if fee:
        canon["fee"] = frame[fee].map(number).abs()
    if side:
        canon["side"] = frame[side].map(lambda v: {1: 1, -1: -1}.get(side_of(v)) if side_of(v) else np.nan)
    if symbol:
        canon["symbol"] = frame[symbol].astype(str).str.strip()
    if group:
        canon["strategy"] = frame[group].astype(str).str.strip().replace("", "(unnamed)")
    # Futures-style fill exports carry realized P&L only on closing rows.
    if money and side and canon["pnl"].eq(0).mean() > 0.3 and canon["pnl"].ne(0).sum() >= 1:
        dropped = int(canon["pnl"].eq(0).sum())
        canon = canon[canon["pnl"].ne(0)]
        warning = [f"Dropped {dropped} rows with a P&L of exactly 0 (opening fills)."]
    else:
        warning = []
    money_unit = unit or (_currency_in(money) if money else None) or _cell_currency(frame[money] if money else None)
    strategies = build_trades(canon, unit=(money_unit or "money") if money else "%", name=name, prefer=unit)
    used = {k: v for k, v in (("result", money), ("percent", pct), ("close_time", when), ("entry_time", entry),
                              ("strategy", group), ("fee", fee), ("side", side), ("symbol", symbol)) if v}
    return ReadResult(strategies, "trades_table", rows=len(frame), columns_seen=cols, columns_used=used,
                      warnings=warning)


def _cell_currency(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    sample = " ".join(series.astype(str).head(20))
    return _currency_in(sample) if re.search(r"[A-Za-z$€£¥]", sample) else None


def _need(columns: Sequence[str], problem: str) -> ReadResult:
    return ReadResult([], "unknown", columns_seen=list(map(str, columns)), need={"problem": problem})


# ── canonical trades into Trades ──────────────────────────────────────────────

def build_trades(canon: pd.DataFrame, *, unit: str, name: str, prefer: Optional[str] = None) -> List[Trades]:
    """Trades from canonical columns (pnl, pct, time, entry_time, fee, side,
    size, entry_price, exit_price, symbol, strategy). Money is the main unit
    when there is any; percent rides along. ``prefer`` = "%" judges percent."""
    canon = canon.reset_index(drop=True)
    if "strategy" in canon and canon["strategy"].notna().any():
        keys = list(dict.fromkeys(canon["strategy"].astype(str)))
    else:
        keys = [None]
    out: List[Trades] = []
    for key in keys:
        part = canon if key is None else canon[canon["strategy"].astype(str) == key]
        money = part["pnl"].to_numpy(dtype=float) if "pnl" in part else np.full(len(part), np.nan)
        pct = part["pct"].to_numpy(dtype=float) if "pct" in part else np.full(len(part), np.nan)
        has_money = bool(np.isfinite(money).sum()) and unit != "%"
        use_pct = (prefer == "%" and np.isfinite(pct).any()) or not has_money
        main = pct if use_pct else money
        keep = np.isfinite(main)
        if not keep.any():
            continue
        part = part[keep]
        main, pct, money = main[keep], pct[keep], money[keep]
        times = part["time"] if "time" in part else pd.Series([pd.NaT] * len(part))
        times = pd.to_datetime(times, errors="coerce", utc=True)
        order = np.arange(len(part))
        has_times = bool(times.notna().all()) and len(part) > 0
        if has_times:
            order = np.argsort(times.to_numpy(), kind="stable")

        def col(name_: str) -> Optional[np.ndarray]:
            if name_ not in part:
                return None
            values = pd.to_numeric(part[name_], errors="coerce").to_numpy(dtype=float)[order]
            return values if np.isfinite(values).any() else None

        entry_times = None
        if "entry_time" in part:
            et = pd.to_datetime(part["entry_time"], errors="coerce", utc=True)
            if et.notna().all():
                entry_times = list(et.iloc[order])
        symbols = None
        if "symbol" in part and part["symbol"].notna().all():
            symbols = [str(s) for s in part["symbol"].iloc[order]]
        st = Trades(
            name=(str(key)[:60] if key is not None else name),
            unit="%" if use_pct else (prefer if prefer and prefer != "%" else unit),
            pnl=main[order].astype(float),
            times=list(times.iloc[order]) if has_times else None,
            entry_times=entry_times,
            pct=(pct[order] if (not use_pct and np.isfinite(pct).sum() >= max(1, 0.9 * len(pct))) else None),
            fees=col("fee") if not use_pct else None,
            side=col("side"),
            size=col("size") if not use_pct else None,
            entry_price=col("entry_price"),
            exit_price=col("exit_price"),
            symbols=symbols,
        )
        if use_pct and np.isfinite(money).sum() >= max(1, 0.9 * len(money)):
            st.notes.append("money_column_kept")
            st.size = None
            st.fees = None
            st.money_shadow = money[order]  # type: ignore[attr-defined]
        out.append(st)
    return out


# ── the one entry point ───────────────────────────────────────────────────────

def _frame_from_records(data: Any) -> Optional[pd.DataFrame]:
    if isinstance(data, dict):
        if data and all(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in data.values()):
            lengths = {len(v) for v in data.values()}
            if len(lengths) == 1:
                return pd.DataFrame({str(k): list(v) for k, v in data.items()}, dtype=object)
        for key in ("trades", "data", "rows", "results"):
            if isinstance(data.get(key), list):
                return _frame_from_records(data[key])
        return None
    if isinstance(data, (list, tuple)) and data and all(isinstance(r, dict) for r in data):
        return pd.DataFrame(list(data), dtype=object)
    return None


def _looks_like_path(text: str) -> bool:
    if "\n" in text or len(text) > 1024:
        return False
    return bool(re.search(r"\.(csv|tsv|txt|xlsx|xlsm|xls|json|zip)$", text.strip(), re.I)) or os.sep in text \
        or "/" in text


def read_any(data: Any, *, kind: Optional[str] = None, unit: Optional[str] = None,
             name: Optional[str] = None) -> ReadResult:
    """Find the trades in ``data``. Never raises on bad input: a result with
    ``need`` set says what is missing."""
    kind = (kind or "auto").strip().lower()
    unit = _normalize_unit(unit)
    label = name or "Your strategy"
    if data is None:
        return _need([], "Nothing was given.")
    # adapters first: objects from backtesting libraries
    bt = _backtesting_trades(data)
    if bt is not None:
        return _finish(_from_backtesting(bt[0], bt[1], name), unit)
    vbt = _vectorbt_trades(data)
    if vbt is not None:
        return _finish(_from_vectorbt(vbt, name), unit)
    ft = _freqtrade_result(data)
    if ft is not None:
        return _finish(_from_freqtrade(ft, name), unit)
    if isinstance(data, (str, os.PathLike)) and not isinstance(data, str):
        return _read_path(Path(data), kind=kind, unit=unit, name=name)
    if isinstance(data, str):
        text = data
        if _looks_like_path(text):
            path = Path(text.strip().strip('"'))
            if path.exists():
                return _read_path(path, kind=kind, unit=unit, name=name)
            return _need([], f"No file at {path.name!r}. Check the path, or paste the file's text instead.")
        return _read_text(text, kind=kind, unit=unit, name=label)
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        if raw[:2] == b"PK":
            return _read_xlsx_or_zip(io.BytesIO(raw), kind=kind, unit=unit, name=label)
        return _read_text(_decode(raw), kind=kind, unit=unit, name=label)
    if isinstance(data, pd.Series) and not isinstance(data.index, pd.RangeIndex) and data.dtype == object:
        data = data.to_frame().T
    if isinstance(data, pd.Series):
        data = list(data.to_numpy())
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex) and not any(norm(c) in _CLOSE_TIME for c in data.columns):
            data = data.reset_index().rename(columns={"index": "time"})
        return _finish(_from_table(data, label, unit=unit, kind=kind), unit)
    frame = _frame_from_records(data)
    if frame is not None:
        return _finish(_from_table(frame, label, unit=unit, kind=kind), unit)
    if isinstance(data, (list, tuple)):
        values, seen_unit = [], None
        for v in data:
            if isinstance(v, str):
                if "%" in v:
                    seen_unit = "%"
                elif _currency_in(v):
                    seen_unit = seen_unit or _currency_in(v)
            values.append(number(v))
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return _need([], "The list holds no numbers.")
        st_unit = unit or seen_unit or "money"
        st = Trades(name=label, unit=st_unit, pnl=np.array(finite, dtype=float))
        result = ReadResult([st], "pnl_list", rows=len(values), columns_used={"result": "the list"})
        if len(finite) < len(values):
            result.warnings.append(f"Skipped {len(values) - len(finite)} entries that were not numbers.")
        return _finish(result, unit)
    return _need([], f"Can't read a {type(data).__name__}. Give a file path, CSV text, a table, or a list of "
                     "results per trade.")


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return None
    if text in ("%", "pct", "percent", "percentage") or text.lower() in ("pct", "percent", "percentage"):
        return "%"
    return SYMBOLS.get(text, text.upper())


def _read_text(text: str, *, kind: str, unit: Optional[str], name: str) -> ReadResult:
    stripped = text.strip()
    if not stripped:
        return _need([], "The text is empty.")
    if stripped[:1] in "[{":
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if parsed is not None and not (isinstance(parsed, list) and all(isinstance(v, (int, float)) for v in parsed)):
            return read_any(parsed, kind=kind, unit=unit, name=name)
    pasted = numbers_from_text(stripped) if kind in ("auto", "pnl") else None
    if pasted is not None:
        values, seen_unit = pasted
        st = Trades(name=name, unit=unit or seen_unit or "money", pnl=np.array(values, dtype=float))
        return _finish(ReadResult([st], "pnl_list", rows=len(values), columns_used={"result": "the pasted numbers"}),
                       unit)
    try:
        frame = table_from_text(stripped)
    except Exception as exc:  # a malformed paste is the trader's to fix, not a crash
        return _need([], f"Could not read the text as a table ({exc}).")
    return _finish(_from_table(frame, name, unit=unit, kind=kind), unit)


def _read_xlsx_or_zip(source: Any, *, kind: str, unit: Optional[str], name: str) -> ReadResult:
    try:
        with zipfile.ZipFile(source) as zf:
            names = zf.namelist()
            if not any(n.startswith("xl/") for n in names):
                jsons = [n for n in names if n.endswith(".json") and "_config" not in n and "market_change" not in n
                         and "meta" not in n]
                for n in jsons:
                    try:
                        parsed = json.loads(zf.read(n).decode("utf-8"))
                    except ValueError:
                        continue
                    found = read_any(parsed, kind=kind, unit=unit, name=None if name == "Your strategy" else name)
                    if found.strategies:
                        return found
                csvs = [n for n in names if n.lower().endswith((".csv", ".tsv", ".txt"))]
                if csvs:
                    return _read_text(_decode(zf.read(csvs[0])), kind=kind, unit=unit, name=name)
                return _need([], "The ZIP holds no trade file we know.")
    except zipfile.BadZipFile:
        return _need([], "The file looks like a ZIP or XLSX but can't be opened.")
    source.seek(0)
    try:
        sheets = _excel_tables(source)
    except ValueError as exc:
        return _need([], str(exc))
    except Exception as exc:  # a broken workbook
        return _need([], f"Could not read the workbook ({exc}).")
    order = sorted(sheets, key=lambda s: (0 if "list of trades" in s.lower() else 1 if "trade" in s.lower() else 2))
    last: Optional[ReadResult] = None
    for sheet in order:
        found = _from_table(sheets[sheet], name, unit=unit, kind=kind)
        if found.strategies:
            found.warnings.append(f"Read the sheet '{sheet}'.")
            return _finish(found, unit)
        last = last or found
    return last or _need([], "The workbook has no sheet with trades.")


def _read_path(path: Path, *, kind: str, unit: Optional[str], name: Optional[str]) -> ReadResult:
    if not path.exists():
        return _need([], f"No file at {path.name!r}.")
    if path.is_dir():
        return _need([], f"{path.name!r} is a folder; give the export file itself.")
    if path.stat().st_size > 50 * 1024 * 1024:
        return _need([], "The file is over 50 MB; export a shorter period or only the closed trades.")
    label = name or re.sub(r"[_\-]+", " ", path.stem).strip()[:60] or "Your strategy"
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix == ".xls" and raw[:4] == b"\xd0\xcf\x11\xe0":
        return _need([], "Old .xls workbooks can't be read here. Open it and save as .xlsx or CSV.")
    if raw[:2] == b"PK":
        return _read_xlsx_or_zip(io.BytesIO(raw), kind=kind, unit=unit, name=label)
    return _read_text(_decode(raw), kind=kind, unit=unit, name=label)


def _finish(result: ReadResult, unit: Optional[str]) -> ReadResult:
    if result.need is None and not result.strategies:
        result.need = {"problem": "We found the columns but no rows with a result we could read."}
    names: Dict[str, int] = {}
    for st in result.strategies:
        names[st.name] = names.get(st.name, 0) + 1
        if names[st.name] > 1:
            st.name = f"{st.name} ({names[st.name]})"
    if len(result.strategies) > MAX_STRATEGIES:
        result.strategies.sort(key=lambda st: -len(st.pnl))
        result.warnings.append(f"Judged the {MAX_STRATEGIES} strategies with the most trades of "
                               f"{len(result.strategies)}.")
        result.strategies = result.strategies[:MAX_STRATEGIES]
    total = sum(len(st.pnl) for st in result.strategies)
    if total > MAX_TRADES:
        result.need = {"problem": f"{total:,} trades is more than the {MAX_TRADES:,} one check reads."}
        result.strategies = []
    return result
