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
MAX_INFLATED = 50 * 1024 * 1024    # a ZIP or XLSX may unpack to at most this

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
    summary: Optional[str] = None               # what was read, in one sentence


# ── cells ─────────────────────────────────────────────────────────────────────

def number(value: Any) -> float:
    """A number from an export cell: percent signs, currency marks, unicode
    minus, (negatives), thousands separators, decimal commas, trailing coin
    names ("0.0012BTC")."""
    if value is None or isinstance(value, (list, tuple, dict, set)):   # never glue [1, 2.5] into 12.5
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    for ch in ("−", "–", "—", "‐", "‑", "‒", "﹣", "－"):   # every minus sign means minus
        text = text.replace(ch, "-")
    # a currency or coin written in front: "EUR 12.50", "-EUR 3", "ETH 0.5"
    text = re.sub(r"^([+-]?)\s*[A-Za-z]{2,6}\.?\s*(?=[-+(]?\d)", r"\1", text)
    text = text.replace("\xa0", "").replace(" ", "").replace(" ", "").replace("'", "")
    if not text or text.lower() in ("nan", "none", "null", "-", "n/a", "na"):
        return float("nan")
    if re.search(r"\d[:/]\d", text) or text[:1] in "[{":   # a clock time, a date or fraction, a list: not an amount
        return float("nan")
    text = re.sub(r"^\+-", "-", text)
    if text.count(".") > 1 and "," not in text:     # 1.234.567: dots as thousands separators
        text = text.replace(".", "")
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

_NOT_RESULT = {"cum", "cumulative", "running", "accumulated", "acc", "rolling", "cash", "amount", "position",
               "value", "notional", "proceeds",
               "run", "runup", "drawdown", "max", "mae", "mfe", "excursion", "favorable",
               "adverse", "unrealized", "unrealised", "open", "fee", "fees", "commission", "total", "equity",
               "balance", "avg", "average", "best", "worst", "trade", "size", "price", "qty", "quantity"}
_PERCENTISH = {"%", "pct", "percent", "percentage", "ratio", "return", "roi", "change"}
# A trade's result in the languages of European brokers' exports (Nordnet: Tulos, Resultat; Degiro, Comdirect ...)
_RESULT_WORDS = {"tulos", "voitto", "resultat", "vinst", "gevinst", "gewinn", "ergebnis", "winst", "resultaat",
                 "beneficio", "resultado", "risultato", "utile", "résultat", "zysk", "wynik"}
_RESULTISH = {"pnl", "p&l", "profit", "p/l", "pl", "gain", "result", "return", "roi", "net", "change", "ratio",
              "profit/loss", "loss"} | _RESULT_WORDS


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
        core = toks & {"pnl", "p&l", "profit", "p/l", "pl", "profit/loss"}
        # realized/closed P&L first, then net P&L; "net" alone (Net Amount, NetCash) is not a result
        rank = (0 if core and toks & {"realized", "realised", "closed"} else 1 if core and "net" in toks
                else 2 if core else 3)
        if _is_pct_name(name) or toks == {"return"} or toks <= {"return", "pct"}:
            pct.append((rank, col))
        elif toks & ({"pnl", "p&l", "profit", "p/l", "pl", "gain", "result", "profit/loss", "loss", "net"}
                     | _RESULT_WORDS):
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
            rest = name[len(cand):]
            # "price usdt" is a price; "exit price" is no exit time, and "time opened" no close time
            # ... and "Position ID" no side, "Market Spread (USD)" no market
            if name.startswith(cand + " ") and not re.search(r"\b(entry|exit|open\w*|clos\w*)\b", rest) \
                    and not re.search(r"\b(price|qty|quantity|size|amount|value|fees?|pnl|profit|id|no|number|"
                                      r"spread|rate)\b", rest):
                return col
    return None


_CLOSE_TIME = ("close time", "closed at", "close date", "exit time", "exit date", "exit timestamp", "close date utc",
               "closing time", "closing date", "time closed", "date closed", "closed time", "closed date",
               "close timestamp", "closed", "trade day", "trade date", "kauppapäivä", "affärsdag", "handelsdag",
               "handelsdato", "date/time", "date and time", "datetime", "date time", "time", "date",
               "timestamp", "date utc", "time utc", "created time", "trade time", "fill time", "order time",
               "updated time", "execution time", "c time", "ctime", "exit")
_OPEN_TIME = ("open time", "opened at", "open date", "entry time", "entry date", "entry timestamp", "opening time",
              "opening date", "time opened", "date opened", "opened time", "opened date", "open timestamp", "opened",
              "entry")
_STRATEGY = ("strategy", "strategy name", "bot", "bot name", "system", "setup", "algo", "model", "book", "tag",
             "label", "variant", "column")
_FEE = ("fees", "fee", "commission", "commissions", "trading fee", "trading fees", "transaction fee", "fee amount",
        "total fee", "fees paid", "fee usdt", "fees usdt", "transaction and/or third party fees", "transaction costs",
        "total fees", "kokonaiskulut", "courtage")
_SIDE = ("side", "direction", "order side", "trade side", "buy/sell", "b/s", "position side", "type")
_SYMBOL = ("symbol", "pair", "market", "trading pair", "instrument", "ticker", "coin pair", "contract", "product",
           "asset", "coin", "currency pair", "security", "arvopaperi", "värdepapper", "verdipapir", "værdipapir")
# What a fill was done at comes first: an order history's "Price" is the limit price, or the word "Market".
_PRICE = ("avg price", "average price", "fill price", "filled price", "executed price", "execution price",
          "deal price", "trade price", "price avg", "avg fill price", "average filled price", "filled price avg",
          "price", "order price")
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
        return raw.decode("utf-16", errors="replace")       # a cut-off file has an odd byte at the end
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
    lines = [line for line in body.splitlines() if line.strip()]
    if len(lines) >= 2 and sum(bool(re.match(r"\s*#?\d+[.):]\s+\S", line)) for line in lines) >= 0.8 * len(lines):
        # a numbered list ("1. +2.5%", "2) -45"): the list numbers are not results
        body = "\n".join(re.sub(r"^\s*#?\d+[.):]\s+", "", line) for line in lines)
    if ";" in body:
        parts = re.split(r"[;\n\r]+", body)
    elif "\t" in body:
        parts = re.split(r"[\t\n\r]+", body)
    elif re.search(r"\d,\d", body) and not re.search(r",\s", body) and re.search(r"\s", body.strip()):
        parts = re.split(r"\s+", body)                         # "12,5 -3,2": decimal commas
    elif "\n" in body.strip() and not re.search(r",\s", body):
        parts = [p for line in body.splitlines() for p in re.split(r"\s+(?=[-+−(]?\d)", line.strip())]
    elif re.search(r",\s", body) and re.search(r"\d,\d{3}\b", body):
        parts = re.split(r",\s+|\s*[\n\r]+\s*", body)        # "1,234.50, -300.00": a comma and a space separate
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


def _numbers_with_words_around(text: str) -> Tuple[Optional[Tuple[List[float], Optional[str]]], List[str]]:
    """A paste like "here are my trades (usdt):" followed by one number per
    line: set aside up to three lines of words at the top and bottom, read
    the numbers, and take a unit hint ("usdt", "%") from the words."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return None, []
    head, tail = [], []
    while lines and len(head) < 3 and _has_words(lines[0]):
        head.append(lines.pop(0))
    while lines and len(tail) < 3 and _has_words(lines[-1]):
        tail.insert(0, lines.pop())
    if not (head or tail) or len(lines) < 3:
        return None, []
    found = numbers_from_text("\n".join(lines))
    if found is None:
        return None, []
    values, unit = found
    words = " ".join(head + tail)
    if unit is None:
        unit = "%" if ("%" in words or re.search(r"(?i)\b(percent|pct)\b", words)) else _currency_in(words)
    return (values, unit), head + tail


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
    need = max(2, int(0.6 * modal)) if modal >= 2 else 1     # a title line padded with commas isn't a header
    for i, row in enumerate(rows[:15]):
        cells = [c for c in row if c.strip()]
        if len(row) >= min(modal, 2) and len(cells) >= need and sum(not _data_like(c) for c in cells) >= len(cells) / 2:
            header_at = i
            break
    header = [h.strip() or f"column {j + 1}" for j, h in enumerate(rows[header_at])]
    if header and all(_data_like(h) for h in header if h.strip()):    # no header at all: the first row is a trade
        header = [f"column {j + 1}" for j in range(len(header))]
        body = rows[header_at:]
    else:
        body = rows[header_at + 1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body if any(c.strip() for c in r)]
    return pd.DataFrame(body, columns=_dedupe(header), dtype=object)


_DATE_CELL = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T].*)?|\d{9,13}(?:\.\d+)?")


def _data_like(cell: str) -> bool:
    """A cell that is a value (a number or a date), not a column name."""
    text = str(cell).strip()
    return math.isfinite(number(text)) or bool(_DATE_CELL.fullmatch(text))


def _dedupe(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def _excel_cell(cell: Any) -> str:
    """A workbook cell as the text a CSV export would show: percent-formatted
    numbers as "1.5%", dates as ISO text."""
    value = getattr(cell, "value", None)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if "%" in str(getattr(cell, "number_format", "") or ""):
            return f"{value * 100:.12g}%"
        return f"{value:.15g}" if isinstance(value, float) else str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
    return str(value)


def _excel_tables(source: Any) -> Dict[str, pd.DataFrame]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ValueError("reading .xlsx needs openpyxl (pip install 'edgeorluck[xlsx]'), or save the sheet as "
                         "CSV") from exc
    out: Dict[str, pd.DataFrame] = {}
    try:
        # openpyxl sees each cell's format: a cell showing 1.5% holds 0.015, and pandas alone would read 0.015 %
        workbook = openpyxl.load_workbook(source, data_only=True, read_only=True)
    except Exception:  # not an .xlsx openpyxl can open: pandas' readers below
        workbook = None
    if workbook is not None:
        for sheet in workbook.worksheets:
            rows = [[_excel_cell(c) for c in row] for row in sheet.iter_rows()]
            rows = [r for r in rows if any(v.strip() for v in r)]
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            keep = [j for j in range(width) if any(r[j].strip() for r in rows)]
            buf = io.StringIO()
            csv.writer(buf).writerows([[r[j] for j in keep] for r in rows])
            out[str(sheet.title)] = table_from_text(buf.getvalue())
        workbook.close()
        return out
    if hasattr(source, "seek"):
        source.seek(0)
    sheets = pd.read_excel(source, sheet_name=None, dtype=object, header=None)
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
        if frame.empty or ("profit_abs" not in frame and "profit_ratio" not in frame):
            continue
        canon = pd.DataFrame({
            "pnl": frame["profit_abs"].map(number) if "profit_abs" in frame else np.nan,
            "pct": frame["profit_ratio"].map(number) * 100.0 if "profit_ratio" in frame else np.nan,
            "time": _times(frame["close_date"]) if "close_date" in frame else pd.NaT,
            "entry_time": _times(frame["open_date"]) if "open_date" in frame else pd.NaT,
            "entry_price": frame["open_rate"].map(number) if "open_rate" in frame else np.nan,
            "exit_price": frame["close_rate"].map(number) if "close_rate" in frame else np.nan,
            "symbol": frame["pair"].astype(str) if "pair" in frame else None,
            # a trade without is_short is a long one (NaN would count as True)
            "side": np.where(frame["is_short"].fillna(False).astype(bool), -1, 1) if "is_short" in frame else 1,
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
    # Day first or month first is decided once for the whole column (row by row, 01/13/25 and 01/05/25 came out
    # as Jan 13 and May 1): a first number above 12 means day first, a second above 12 month first. Undecidable
    # columns are read day first, the way most of the world writes dates.
    parts = text.str.extract(r"^(\d{1,2})[./-](\d{1,2})[./-]\d{2,4}\b")
    dayfirst = False
    if parts[0].notna().mean() > 0.5:
        first, second = parts[0].astype(float), parts[1].astype(float)
        dayfirst = not ((second > 12).any() and not (first > 12).any())
    try:
        out = pd.to_datetime(text, errors="coerce", utc=True, format="mixed", dayfirst=dayfirst)
    except (TypeError, ValueError):
        out = pd.to_datetime(text, errors="coerce", utc=True)
    return out


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
    signal = find_column(cols, ("signal",))
    numbers = frame[number_col].astype(str).str.strip()
    # Two exports pasted together repeat their trade numbers: pair the k-th entry of a number with its k-th exit.
    side_kind = types.str.startswith("entry").map({True: "entry", False: "exit"})
    nth = frame.groupby([numbers, side_kind], sort=False).cumcount()
    notes: List[str] = []
    if (nth > 0).any():
        notes.append("Trade numbers repeat (two exports pasted together?): each number's entries and exits were "
                     "paired in file order.")
    still_open = 0
    out = []
    for (trade_no, _), group in frame.groupby([numbers, nth], sort=False):
        t = types.loc[group.index]
        entry = group[t.str.startswith("entry")]
        exit_ = group[t.str.startswith("exit")]
        if exit_.empty:
            continue
        if signal and str(exit_.iloc[-1][signal]).strip().lower() == "open":
            still_open += 1                 # TradingView lists an open trade's unrealised result: not a closed trade
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
    if still_open:
        notes.append(f"Left out {still_open} trade{'s' if still_open != 1 else ''} still open (its result isn't final).")
    canon = pd.DataFrame(out)
    canon.attrs["notes"] = notes
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
    if price is None:
        return None
    has_executed = find_column(cols, ("executed", "executed qty", "filled", "filled amount", "filled qty")) is not None
    qty = find_column(cols, _QTY if not has_executed else _QTY[:-1], avoid=[price])
    signed = None
    if side is None or frame[side].map(side_of).notna().mean() < 0.8:
        # No side column, but buys and sells told apart by the sign of the quantity (Degiro: 10 bought, -10 sold)
        amounts = numbers_of(frame[qty]) if qty else None
        if amounts is None or amounts.notna().mean() < 0.9 or (amounts > 0).mean() < 0.2 or (amounts < 0).mean() < 0.2:
            return None
        side = signed = qty
    total_candidates = _QUOTE_TOTAL + (("amount",) if has_executed else ())
    # A signed fill's own totals are often in another currency than its price (Degiro: Value in EUR, Price in USD)
    total = None if signed else find_column(cols, total_candidates, avoid=[c for c in (price, qty) if c])
    if qty is None and total is None:
        return None
    out = {"side": side, "price": price, **({"signed_qty": signed} if signed else {})}
    taken = [c for c in (side, price, qty, total) if c]
    when = find_column(cols, _CLOSE_TIME) or _fill_time_column(cols, frame, taken)
    pair = find_column(cols, _SYMBOL) or _fill_pair_column(cols, frame, taken)
    for key, value in (("qty", qty), ("quote_total", total), ("time", when), ("pair", pair),
                       ("fee", find_column(cols, _FEE)), ("fee_coin", find_column(cols, _FEE_COIN))):
        if value is not None:
            out[key] = value
    return out


def _fill_time_column(cols: Sequence[str], frame: pd.DataFrame, taken: Sequence[str]) -> Optional[str]:
    """A fill's time under another name (tradeCreatedAt, Created, Date(UTC)): a header
    that says time or date, else a column whose cells are nearly all dates."""
    rest = [c for c in cols if c not in taken and "price" not in norm(c)]
    for c in rest:
        if re.search(r"time|date|created|timestamp|(?:^|[^a-z])at$", norm(c)):
            return c
    for c in rest:
        # Row ids and ticket numbers (9001, 9002 ...) read as 1970 epoch times: not dates
        small_ints = frame[c].astype(str).str.strip().str.fullmatch(r"\d{1,8}").mean() > 0.5
        if not pd.api.types.is_numeric_dtype(frame[c]) and not small_ints and _times(frame[c]).notna().mean() >= 0.9:
            return c
    return None


_PAIR_CELL = re.compile(r"[A-Z0-9]{2,12}[-/_]?(?:USDT|USDC|FDUSD|BUSD|USD|EUR|GBP|TRY|BTC|ETH|BNB)", re.I)


def _fill_pair_column(cols: Sequence[str], frame: pd.DataFrame, taken: Sequence[str]) -> Optional[str]:
    """The traded pair under another name (Spot Pairs, Contracts): a header with the
    word pair or symbol in it, else a column whose cells nearly all look like pairs."""
    rest = [c for c in cols if c not in taken]
    for c in rest:
        if set(_tokens(norm(c))) & {"pair", "pairs", "symbol", "symbols", "market", "instrument"}:
            return c
    for c in rest:
        cells = frame[c].dropna().astype(str).str.strip()
        if len(cells) and cells.map(lambda v: bool(_PAIR_CELL.fullmatch(v))).mean() >= 0.9:
            return c
    return None


def _from_fills(frame: pd.DataFrame, colmap: Dict[str, str], name: str) -> ReadResult:
    cols = list(map(str, frame.columns))
    if "time" not in colmap:
        # Exports are usually newest first, so pairing in file order would run backwards: don't guess.
        return _need(cols, "These look like fills, but there is no time column, so buys can't be paired with the "
                           "sells that came after them. Export the fills with their date and time.")
    if "pair" not in colmap:
        prices = numbers_of(frame[colmap["price"]])
        if prices.gt(0).any() and prices[prices > 0].max() > 10 * prices[prices > 0].min():
            return _need(cols, "These fills have prices far apart but no pair or symbol column, so they may be "
                               "several coins mixed together. Export them with the pair (like BTCUSDT).")
    rows = []
    pair_col = colmap.get("pair")
    times = _times(frame[colmap["time"]])
    # Decimal commas are decided per column, not per cell: a price of "2,345" (EUR) next to "100,5" is 2.345.
    commas = {k: _decimal_commas(frame[colmap[k]].map(lambda v: split_amount(v)[0]))
              for k in ("price", "qty", "fee", "quote_total") if k in colmap}

    def num(key: str, text: Any) -> float:
        return _comma_number(text) if commas.get(key) and text is not None else number(text)

    # A stock broker's product is a name, not a pair ("ASML HOLDING" is no ASML quoted in HOLDING): its currency is
    # the column right after the price when there is one (Degiro), else unknown.
    currency = None
    if "signed_qty" in colmap and cols.index(colmap["price"]) + 1 < len(cols):
        cells = frame[cols[cols.index(colmap["price"]) + 1]].astype(str).str.strip().str.upper()
        currency = cells if cells.str.fullmatch(r"[A-Z]{3}").mean() > 0.9 else None

    for i, (_, r) in enumerate(frame.iterrows()):
        qty_text, qty_coin = split_amount(r[colmap["qty"]]) if "qty" in colmap else (None, None)
        if "signed_qty" in colmap:
            amount = num("qty", qty_text)
            side = (1 if amount > 0 else -1) if math.isfinite(amount) and amount != 0 else None
        else:
            side = side_of(r[colmap["side"]])
        if side is None:
            continue
        total_text, total_coin = split_amount(r[colmap["quote_total"]]) if "quote_total" in colmap else (None, None)
        fee_text, fee_coin = split_amount(r[colmap["fee"]]) if "fee" in colmap else (None, None)
        if "fee_coin" in colmap and str(r[colmap["fee_coin"]]).strip():
            fee_coin = str(r[colmap["fee_coin"]]).strip().upper()
        pair = str(r[pair_col]).strip() if pair_col else ""
        if not pair and qty_coin and total_coin:
            pair = f"{qty_coin}/{total_coin}"
        if "signed_qty" in colmap:
            pair = f"{pair.replace('/', ' ') or '?'}/{currency.iloc[i] if currency is not None else ''}"
        rows.append({"time": times.iloc[i], "pair": pair or "?", "side": side,
                     "price": num("price", r[colmap["price"]]), "qty": abs(num("qty", qty_text)),   # sells as -0.5
                     "fee": num("fee", fee_text), "fee_coin": fee_coin, "quote_total": num("quote_total", total_text)})
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
    pairs = sorted({"/".join(x for x in split_pair(p) if x) for p in facts["pairs"] if p and p != "?"})
    summary = (f"{facts['fills']:,} fills in {len(pairs)} pair{'s' if len(pairs) != 1 else ''} "
               f"({', '.join(pairs[:6])}{', ...' if len(pairs) > 6 else ''}) paired into "
               f"{len(trips):,} round trips, first in first out, net of fees.")
    return ReadResult(strategies, "fills", rows=len(frame), columns_seen=list(map(str, frame.columns)),
                      columns_used=used, warnings=warnings, summary=summary)


def numbers_of(series: pd.Series) -> pd.Series:
    """A column of numbers. Whether a comma is a decimal comma is decided once for
    the whole column: "0,750" and "1,25" mark decimal commas, so "1,250" in the
    same column is 1.25, not 1,250."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    if _decimal_commas(series):
        return series.astype(str).map(lambda v: _comma_number(v)).astype(float)
    return series.map(number).astype(float)


def _decimal_commas(series: pd.Series) -> bool:
    """Does this column write decimals with a comma? Yes when no cell has a decimal
    point and some cell has a comma that can't be a thousands separator."""
    text = series.dropna().astype(str).str.strip()
    if text.str.contains(r"\d\.\d").any():
        return False
    return bool(text.str.contains(r"(?:^|[^\d])0,\d").any() or text.str.contains(r"\d,(?:\d{1,2}|\d{4,})(?!\d)").any())


def _comma_number(value: Any) -> float:
    return number(str(value).replace(".", "").replace(",", "."))


_TOTAL_ROW = re.compile(r"(?:grand\s+)?(?:sub)?totals?|sum|summary", re.I)


def _join_date_and_clock(frame: pd.DataFrame) -> pd.DataFrame:
    """A date column next to a clock column (Degiro: Date 27-01-2024, Time 15:30) is one time: on its own, a
    newest-first file's sell at 17:20 would come before the buy it closes at 15:30 on the same day."""
    names = {norm(c): c for c in frame.columns}
    day = next((names[n] for n in ("date", "trade date", "datum", "päivä", "dato") if n in names), None)
    clock = next((names[n] for n in ("time", "trade time", "tid", "aika", "zeit", "klokkeslett") if n in names), None)
    if day is None or clock is None:
        return frame
    clocks = frame[clock].astype(str).str.strip()
    if clocks.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?").mean() < 0.9:
        return frame
    frame[day] = frame[day].astype(str).str.strip() + " " + clocks
    return frame.drop(columns=[clock])


def _opening_rows(frame: pd.DataFrame, cols: Sequence[str], avoid: Sequence[str]) -> Optional[pd.Series]:
    """Executions a broker marks as opening (IBKR: Open/CloseIndicator O, or O in its Code column): they carry
    no result of their own, so they are no trades. None when no such column is there."""
    col = find_column(cols, ("open/close indicator", "open/close", "open close indicator", "code"), avoid=avoid)
    if col is None:
        return None
    codes = frame[col].astype(str).str.strip().str.upper()
    shaped = codes[codes != ""].str.fullmatch(r"[A-Z]{1,3}(;[A-Z]{1,3})*")
    if len(shaped) == 0 or shaped.mean() < 0.8:
        return None
    parts = codes.str.split(";")
    opening = parts.map(lambda p: "O" in p and "C" not in p)
    return opening if opening.any() and parts.map(lambda p: "C" in p).any() else None


def _from_table(frame: pd.DataFrame, name: str, *, unit: Optional[str], kind: Optional[str]) -> ReadResult:
    frame = _join_date_and_clock(frame.copy())
    frame.columns = [str(c) for c in frame.columns]
    cols = list(frame.columns)
    if frame.empty:
        return _need(cols, "The table has no rows.")
    notes: List[str] = []
    # A "Total" row at the bottom of an export is not one more trade.
    cells = frame.select_dtypes(exclude="number").astype(str).apply(lambda c: c.str.strip().str.strip("*").str.strip())
    totals = cells.apply(lambda c: c.str.fullmatch(_TOTAL_ROW)).any(axis=1) if not cells.empty else None
    if totals is not None and totals.any() and totals.sum() < len(frame):
        frame = frame[~totals.to_numpy()]
        notes.append(f"Left out {int(totals.sum())} total row{'s' if totals.sum() != 1 else ''}.")
    if kind != "fills":                         # TradingView's entry/exit rows are paired whatever kind was given
        tv = _tradingview(frame)
        if tv is not None:
            canon, used = tv
            money_unit = _currency_in(used.get("result", "")) or "money"
            strategies = build_trades(canon, unit=money_unit if used.get("result") else "%", name=name,
                                      prefer=unit)
            last = None
            if "cum_pct" in canon and canon["cum_pct"].notna().any():
                order = canon["time"] if canon["time"].notna().all() else canon["trade_no"]
                last = float(canon.loc[order.idxmax(), "cum_pct"])          # exports are newest first
                last = last if math.isfinite(last) else None
            for st in strategies:
                st.strategy_return_pct = last
            return ReadResult(strategies, "tradingview", rows=len(frame), columns_seen=cols, columns_used=used,
                              source_hint="backtest", warnings=notes + list(canon.attrs.get("notes", [])))
    money, pct = find_result_columns(cols)
    # A result column that is all empty or zero (a spot fills export with a "Realized PnL" column) is not the result.
    blank_result = bool(money) and pct is None and numbers_of(frame[money]).fillna(0).eq(0).all()
    if kind == "fills" or (kind in (None, "auto") and ((money is None and pct is None) or blank_result)):
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
    if when is None and entry is None:      # no time-ish header (a headerless file): a column of dates will do
        when = _fill_time_column(cols, frame, [c for c in (money, pct) if c])
    if entry == when:
        entry = None
    if when and f"{when} (2)" in cols and entry is None:
        # MT5 lists "Time" twice: when the position opened, then when it closed
        dupes = [when] + [c for c in cols if re.fullmatch(re.escape(when) + r" \(\d+\)", c)]
        entry, when = dupes[0], dupes[-1]
    group = find_column(cols, _STRATEGY, avoid=[c for c in (money, pct, when) if c])
    if group is not None:
        # A strategy column names a few strategies, each with several trades. A tag or label on every row (an order
        # id, a note) is not one: split by it, 32,000 trades became 32,000 "strategies" of one trade each.
        sizes = frame[group].astype(str).str.strip().value_counts()
        if len(sizes) > max(1, len(frame) // 5) or (sizes >= 3).mean() < 0.5:
            group = None
    fee = find_column(cols, _FEE, avoid=[c for c in (money, pct) if c])
    side = find_column(cols, ("side", "direction", "position side", "long / short", "long/short", "position"),
                       avoid=[c for c in (money, pct) if c])
    symbol = find_column(cols, _SYMBOL, avoid=[c for c in (money, pct, group) if c])
    canon = pd.DataFrame(index=frame.index)
    canon["pnl"] = numbers_of(frame[money]) if money else np.nan
    canon["pct"] = numbers_of(frame[pct]) if pct else np.nan
    if pct and "ratio" in norm(pct) and canon["pct"].abs().max() <= 5 \
            and not frame[pct].astype(str).str.contains("%").any():
        canon["pct"] = canon["pct"] * 100.0
    # Results in percent: the user said so, or the result cells end in "%" ("+1.5%").
    pct_cells = bool(money) and frame[money].astype(str).str.contains("%").mean() >= 0.9
    if money and (unit == "%" or pct_cells) and (pct is None or canon["pct"].isna().all()):
        canon["pct"], canon["pnl"] = canon["pnl"], np.nan
        pct, money = money, None
    canon["time"] = _times(frame[when]).to_numpy() if when else pd.NaT
    canon["entry_time"] = _times(frame[entry]).to_numpy() if entry else pd.NaT
    if fee:
        canon["fee"] = numbers_of(frame[fee]).abs()
    if side:
        canon["side"] = frame[side].map(lambda v: {1: 1, -1: -1}.get(side_of(v)) if side_of(v) else np.nan)
    if symbol:
        canon["symbol"] = frame[symbol].astype(str).str.strip()
    if group:
        canon["strategy"] = frame[group].astype(str).str.strip().replace("", "(unnamed)")
    warning = list(notes)
    opening = _opening_rows(frame, cols, [c for c in (money, pct, when, entry, side, symbol) if c])
    if opening is not None:
        empty = canon["pnl"].fillna(0).eq(0) & canon["pct"].fillna(0).eq(0)
        drop = opening.to_numpy() & empty.to_numpy()
        if drop.any():
            canon = canon[~drop]
            warning.append(f"Left out {int(drop.sum())} opening execution{'s' if drop.sum() != 1 else ''}: the "
                           "broker marks them as opening, and only the closing ones carry a trade's result.")
    # Futures-style fill exports carry realized P&L only on closing rows: drop the opening rows, but only for
    # files that look like fills (a price and a quantity), never breakeven trades in a trade table.
    fill_shaped = _looks_like_fills(cols, frame) is not None
    gross_fills = False
    if money and side and fill_shaped and canon["pnl"].eq(0).mean() > 0.3 and canon["pnl"].ne(0).sum() >= 1:
        dropped = int(canon["pnl"].eq(0).sum())
        if "fee" in canon:
            # An opening fill's fee belongs to the trade it opens: carry it to that symbol's next closing row
            # (dropping the opening rows used to drop their fees too).
            pnl, fees = canon["pnl"].to_numpy(), canon["fee"].fillna(0).to_numpy(copy=True)
            keys = canon["symbol"].to_numpy() if "symbol" in canon else np.full(len(canon), "")
            order = np.argsort(canon["time"].to_numpy(), kind="stable") if canon["time"].notna().all() \
                else np.arange(len(canon))
            pending: Dict[Any, float] = {}
            for i in order:
                if pnl[i] == 0:
                    pending[keys[i]] = pending.get(keys[i], 0.0) + fees[i]
                else:
                    fees[i] += pending.pop(keys[i], 0.0)
            canon["fee"] = fees
            gross_fills = True
        canon = canon[canon["pnl"].ne(0)]
        warning.append(f"Dropped {dropped} rows with a P&L of exactly 0 (opening fills).")
    money_unit = (unit if unit != "%" else None) or (_currency_in(money) if money else None) or \
        _cell_currency(frame[money] if money else None) or (_next_currency(frame, money) if money else None)
    ccy = find_column(cols, ("currency", "currency primary", "ccy", "valuutta", "valuta"), avoid=[money] if money else [])
    if money and not money_unit and ccy is not None:            # IBKR: one Currency column for the whole statement
        cells = frame.loc[canon.index, ccy].astype(str).str.strip().str.upper()
        money_unit = cells.iloc[0] if len(cells) and cells.nunique() == 1 and re.fullmatch(r"[A-Z]{3}",
                                                                                          cells.iloc[0]) else None
    strategies = build_trades(canon, unit=(money_unit or "money") if money else "%", name=name,
                              prefer="%" if not money else unit)
    if gross_fills:
        # Exchange fills with realized P&L on the closing rows (futures trade history) report it before fees.
        for st in strategies:
            if st.fees is not None:
                st.fees_included = False
        warning.append("Realized profit on these fills is before fees, so the fee column (opening fills included) "
                       "is taken off. If the profit already includes fees, say fees_included=yes.")
    if any("dates_partial" in st.notes for st in strategies):
        warning.append("Some rows have no readable close date, so the trades are taken in file order (a newest-first "
                       "file is turned around).")
    used = {k: v for k, v in (("result", money), ("percent", pct), ("close_time", when), ("entry_time", entry),
                              ("strategy", group), ("fee", fee), ("side", side), ("symbol", symbol)) if v}
    return ReadResult(strategies, "trades_table", rows=len(frame), columns_seen=cols, columns_used=used,
                      warnings=warning)


def _next_currency(frame: pd.DataFrame, col: str) -> Optional[str]:
    """The currency in the column right after a result column (Trading 212: Result, Currency (Result); Nordnet:
    Tulos, Valuutta), when it names one currency throughout."""
    cols = list(frame.columns)
    i = cols.index(col)
    if i + 1 >= len(cols):
        return None
    cells = frame[cols[i + 1]].astype(str).str.strip().str.upper()
    cells = cells[(cells != "") & (cells != "NAN")]
    return cells.iloc[0] if len(cells) and cells.nunique() == 1 and re.fullmatch(r"[A-Z]{3}", cells.iloc[0]) else None


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
        groups = list(canon.groupby(canon["strategy"].astype(str), sort=False))     # one pass, in file order
    else:
        groups = [(None, canon)]
    out: List[Trades] = []
    for key, part in groups:
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
        partial = False
        if has_times:
            order = np.argsort(times.to_numpy(), kind="stable")
        elif times.notna().sum() >= 2:
            # One blank date used to drop the time order altogether, so a newest-first export was judged backwards.
            # The dates there still say which way the file runs.
            dated = times[times.notna()]
            steps = np.diff((dated - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy())
            if (steps < 0).sum() > (steps > 0).sum():
                order = order[::-1]
            partial = True

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
        if partial:
            st.notes.append("dates_partial")
        if use_pct and np.isfinite(money).sum() >= max(1, 0.9 * len(money)):
            st.notes.append("money_column_kept")
            st.size = None
            st.fees = None
            st.money_shadow = money[order]  # type: ignore[attr-defined]
            st.money_unit = unit if unit and unit != "%" else "money"  # type: ignore[attr-defined]
        out.append(st)
    return out


# ── the one entry point ───────────────────────────────────────────────────────

def _frame_from_records(data: Any) -> Optional[pd.DataFrame]:
    if isinstance(data, dict):
        for key in ("trades", "data", "rows", "results", "closed_trades", "orders", "fills"):
            value = data.get(key)
            if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
                return _frame_from_records(value)
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
             name: Optional[str] = None, allow_paths: bool = True) -> ReadResult:
    """Find the trades in ``data``. Never raises on bad input: a result with
    ``need`` set says what is missing.

    allow_paths=False: a string is only ever text, never a file name. The web app
    reads visitors' input this way, so nobody can make it open a file on the server."""
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
        if not allow_paths:
            return _need([], "Send the file's contents, not a file name.")
        return _read_path(Path(data), kind=kind, unit=unit, name=name)
    if isinstance(data, str):
        text = data
        if allow_paths and _looks_like_path(text):
            path = Path(text.strip().strip('"'))
            if path.exists():
                return _read_path(path, kind=kind, unit=unit, name=name)
            return _need([], f"No file at {path.name!r}. Check the path, or paste the file's text instead.")
        return _read_text(text, kind=kind, unit=unit, name=label, explicit=name)
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        if raw[:2] == b"PK":
            return _read_xlsx_or_zip(io.BytesIO(raw), kind=kind, unit=unit, name=label, explicit=name)
        return _read_text(_decode(raw), kind=kind, unit=unit, name=label, explicit=name)
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
    text = SYMBOLS.get(text, text.upper())
    # A unit is a short currency code or symbol (USDT, BTC, EUR, $). Anything else is ignored: the unit is
    # repeated in every sentence of the answer and kept with a web forecast, so free text can't be one.
    return text if re.fullmatch(r"[A-Z0-9$€£¥]{1,10}", text) else None


def _read_text(text: str, *, kind: str, unit: Optional[str], name: str,
               explicit: Optional[str] = None) -> ReadResult:
    stripped = text.strip()
    if not stripped:
        return _need([], "The text is empty.")
    if stripped[:1] in "[{":
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if parsed is not None and not (isinstance(parsed, list) and all(isinstance(v, (int, float)) for v in parsed)):
            # text inside text is never a file name
            return read_any(parsed, kind=kind, unit=unit, name=explicit, allow_paths=False)
    pasted = numbers_from_text(stripped) if kind in ("auto", "pnl") else None
    skipped: List[str] = []
    if pasted is None and kind in ("auto", "pnl"):
        pasted, skipped = _numbers_with_words_around(stripped)
    if pasted is not None:
        values, seen_unit = pasted
        st = Trades(name=name, unit=unit or seen_unit or "money", pnl=np.array(values, dtype=float))
        result = ReadResult([st], "pnl_list", rows=len(values), columns_used={"result": "the pasted numbers"})
        if skipped:
            result.warnings.append("Read the numbers and skipped the line(s) with words: "
                                   + "; ".join(repr(s[:60]) for s in skipped) + ".")
        return _finish(result, unit)
    try:
        frame = table_from_text(stripped)
    except Exception as exc:  # a malformed paste is the trader's to fix, not a crash
        return _need([], f"Could not read the text as a table ({exc}).")
    return _finish(_from_table(frame, name, unit=unit, kind=kind), unit)


def _read_xlsx_or_zip(source: Any, *, kind: str, unit: Optional[str], name: str,
                      explicit: Optional[str] = None) -> ReadResult:
    try:
        with zipfile.ZipFile(source) as zf:
            names = zf.namelist()
            # A small ZIP can inflate to gigabytes. Members never inflate past their declared size, so a cap on
            # the declared total bounds the memory (XLSX files are ZIPs too).
            if sum(info.file_size for info in zf.infolist()) > MAX_INFLATED:
                return _need([], "The file unpacks to over 50 MB; export a shorter period or only the closed trades.")
            if not any(n.startswith("xl/") for n in names):
                jsons = [n for n in names if n.endswith(".json") and "_config" not in n and "market_change" not in n
                         and "meta" not in n]
                for n in jsons:
                    try:
                        parsed = json.loads(zf.read(n).decode("utf-8"))
                    except ValueError:
                        continue
                    found = read_any(parsed, kind=kind, unit=unit, name=explicit, allow_paths=False)
                    if found.strategies:
                        return found
                csvs = sorted(n for n in names if n.lower().endswith((".csv", ".tsv", ".txt")))
                if csvs:
                    texts = [_decode(zf.read(n)) for n in csvs]
                    heads = {t.lstrip("﻿").splitlines()[0].strip() if t.strip() else "" for t in texts}
                    if len(texts) > 1 and len(heads) == 1 and "" not in heads:
                        # monthly exports of one account: one table, the header once
                        first, *rest = texts
                        text = first.rstrip("\n") + "\n" + "\n".join(t.split("\n", 1)[1].strip("\n")
                                                                       for t in rest if "\n" in t)
                        found = _read_text(text, kind=kind, unit=unit, name=name, explicit=explicit)
                        found.warnings.append(f"Read {len(texts)} files from the ZIP as one table.")
                        return found
                    found = _read_text(texts[0], kind=kind, unit=unit, name=name, explicit=explicit)
                    if len(texts) > 1:
                        found.warnings.append(f"The ZIP holds {len(texts)} files with different columns; read "
                                              f"{csvs[0]!r} only.")
                    return found
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
    if not path.is_file():
        return _need([], f"{path.name!r} is not a regular file; give the export file itself.")
    if path.stat().st_size > 50 * 1024 * 1024:
        return _need([], "The file is over 50 MB; export a shorter period or only the closed trades.")
    label = name or re.sub(r"[_\-]+", " ", path.stem).strip()[:60] or "Your strategy"
    suffix = path.suffix.lower()
    with path.open("rb") as handle:             # read at most the cap, whatever the file claims its size is
        raw = handle.read(50 * 1024 * 1024 + 1)
    if len(raw) > 50 * 1024 * 1024:
        return _need([], "The file is over 50 MB; export a shorter period or only the closed trades.")
    if suffix == ".xls" and raw[:4] == b"\xd0\xcf\x11\xe0":
        return _need([], "Old .xls workbooks can't be read here. Open it and save as .xlsx or CSV.")
    if raw[:2] == b"PK":
        return _read_xlsx_or_zip(io.BytesIO(raw), kind=kind, unit=unit, name=label, explicit=name)
    return _read_text(_decode(raw), kind=kind, unit=unit, name=label, explicit=name)


FORMAT_WORDS = {"tradingview": "a TradingView list of trades", "trades_table": "a table of closed trades",
                "pnl_list": "a list of results", "backtesting.py": "a backtesting.py result",
                "vectorbt": "a vectorbt portfolio", "freqtrade": "a freqtrade backtest result"}


def _finish(result: ReadResult, unit: Optional[str]) -> ReadResult:
    if result.summary is None and result.strategies:
        n = sum(len(st.pnl) for st in result.strategies)
        k = len(result.strategies)
        what = FORMAT_WORDS.get(result.format, "the input")
        result.summary = (f"Read {what}: {n:,} closed trades" + (f" in {k} strategies" if k > 1 else "") +
                          f", results in {result.strategies[0].unit if result.strategies[0].unit != 'money' else 'money'}"
                          + (" (money first, percent alongside)." if result.strategies[0].pct is not None else "."))
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
