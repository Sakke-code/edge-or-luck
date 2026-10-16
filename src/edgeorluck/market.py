"""Price history for honest backtests.

    edgeorluck.prices("BTCUSDT", "1h", start="2024-01-01")    # crypto: Binance spot
    edgeorluck.prices("SPY", "1d", start="2015-01-01")        # stocks, ETFs, indices: Yahoo Finance
    edgeorluck.prices("EURUSD", "1d")                          # currencies: the ECB's daily reference rates

Prices are fetched on your own machine from free public sources, with no
account and no key, and are never stored or passed on by edge or luck:
Binance's public market data, Yahoo Finance's public chart data (unofficial;
split- and dividend-adjusted), and the European Central Bank's reference
rates through Frankfurter. Or bring your own: load_prices() reads a CSV or a
DataFrame with a close column (open, high, low and volume when there are some).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .readers import number

KLINE_HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")
KLINE_LIMIT = 1000
FETCH_WORKERS = 6
MAX_BARS = 300_000
CACHE_SIZE = 32
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
               "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
               "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000}
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
YAHOO_INTERVALS = {"1h": "60m", "1d": "1d", "1w": "1wk"}      # Yahoo keeps hourly bars for about two years
ECB = "https://api.frankfurter.dev/v1/"
ECB_CODES = set("AUD BGN BRL CAD CHF CNY CZK DKK EUR GBP HKD HUF IDR ILS INR ISK JPY KRW MXN MYR NOK NZD PHP PLN RON "
                "SEK SGD THB TRY USD ZAR".split())
CRYPTO_QUOTES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "BTC", "ETH", "BNB", "EUR", "TRY")
_UA = {"User-Agent": "edgeorluck (+https://github.com/Sakke-code/edge-or-luck)", "Accept": "application/json"}
_mk_cache: Dict[tuple, pd.DataFrame] = {}


class PriceError(RuntimeError):
    """Prices couldn't be fetched or read, with a plain sentence saying what to do."""


def _mk_ms(value: Any, default: datetime) -> int:
    if value is None:
        moment = default
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    else:
        try:
            moment = pd.Timestamp(value).to_pydatetime()
        except (ValueError, TypeError):
            raise PriceError(f"Can't read the date {value!r}: write it like 2024-01-31.") from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def _mk_get(path: str) -> Any:
    last: Optional[Exception] = None
    for host in KLINE_HOSTS:
        request = urllib.request.Request(host + path, headers={"User-Agent": "edgeorluck"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 400:
                    body = exc.read().decode("utf-8", "replace")
                    raise PriceError(f"Binance refused the request ({body[:160]}). Check the symbol (like BTCUSDT) "
                                     f"and the interval ({', '.join(INTERVAL_MS)}).") from None
                if exc.code in (429, 418):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                time.sleep(1 + attempt)
    raise PriceError(f"Couldn't reach Binance's public market data ({last}). Check the internet connection, or "
                     "bring your own prices with load_prices().")


def kind_of(symbol: str) -> Tuple[str, str]:
    """Which market a symbol is, and how its source writes it: ("crypto", "BTCUSDT") for a Binance
    spot pair, ("forex", "EURUSD") for two ECB currencies, ("stock", "AAPL") for anything else
    (stocks, ETFs and indices as Yahoo Finance writes them: SPY, NOKIA.HE, ^GSPC). A prefix
    (crypto:, forex:, stock:) settles a symbol that could be either."""
    raw = str(symbol).strip()
    kind, _, rest = raw.partition(":") if ":" in raw else ("", "", raw)
    compact = rest.upper().replace("/", "").replace("-", "").replace("_", "")
    if kind.lower() in ("forex", "fx") or (not kind and re.fullmatch(r"[A-Z]{6}", compact)
                                            and compact[:3] in ECB_CODES and compact[3:] in ECB_CODES):
        if not (re.fullmatch(r"[A-Z]{6}", compact) and compact[:3] in ECB_CODES and compact[3:] in ECB_CODES):
            raise PriceError(f"{rest!r} isn't two of the ECB's currencies, like EURUSD or USDJPY.")
        return "forex", compact
    if kind.lower() == "crypto" or (not kind and re.fullmatch(r"[A-Z0-9]{6,20}", compact)
                                    and any(compact.endswith(q) for q in CRYPTO_QUOTES)):
        if not re.fullmatch(r"[A-Z0-9]{2,20}", compact):
            raise PriceError(f"{rest!r} isn't a Binance spot pair: use letters and digits, like BTCUSDT.")
        return "crypto", compact
    if kind and kind.lower() != "stock":
        raise PriceError(f"Unknown market {kind!r}: write crypto:, forex: or stock: before the symbol, or neither.")
    if not re.fullmatch(r"\^?[A-Za-z0-9.\-=]{1,20}", rest):
        raise PriceError(f"{rest!r} isn't a ticker: write it as Yahoo Finance does, like AAPL, SPY, NOKIA.HE or ^GSPC.")
    return "stock", rest.upper()


def prices(symbol: str = "BTCUSDT", interval: str = "1h", start: Any = None, end: Any = None) -> pd.DataFrame:
    """Price bars, oldest first, indexed by each bar's open time (UTC); finished bars only.

    symbol: a Binance spot pair (BTCUSDT): open, high, low, close, volume at any Binance interval;
        a stock, ETF or index as Yahoo Finance writes it (AAPL, SPY, NOKIA.HE, ^GSPC): split- and
        dividend-adjusted open, high, low, close and volume, at 1h (about the last two years), 1d or 1w;
        or two currencies (EURUSD, USDJPY): the European Central Bank's daily reference rate as the
        close, at 1d or 1w. See kind_of().
    start defaults to two years ago, end to now."""
    kind, symbol = kind_of(symbol)
    allowed = {"crypto": INTERVAL_MS, "stock": YAHOO_INTERVALS, "forex": ("1d", "1w")}[kind]
    if interval not in allowed:
        raise PriceError(f"{symbol} has {', '.join(allowed)} bars, not the interval {interval!r}.")
    step = INTERVAL_MS[interval]
    now = datetime.now(timezone.utc)
    start_ms = _mk_ms(start, now - timedelta(days=720 if (kind, interval) == ("stock", "1h") else 730))
    if start is None:
        start_ms -= start_ms % step         # "two years ago" on a bar boundary, so repeated calls share the cache
    end_ms = min(_mk_ms(end, now), int(now.timestamp() * 1000))
    if start_ms >= end_ms:
        raise PriceError("start must be before end.")
    if (end_ms - start_ms) / step > MAX_BARS:
        raise PriceError(f"That is more than {MAX_BARS:,} bars: pick a later start or a longer interval.")
    if kind == "stock" and interval == "1h" and start_ms < (now - timedelta(days=729)).timestamp() * 1000:
        raise PriceError("Yahoo Finance keeps hourly bars for about the last two years: start later, or use 1d.")
    # A period that ends near now also depends on which bars have finished since: the bar count now is in the key,
    # so a frame cached while a bar was forming isn't handed out after it has closed. A stock's day closes with its
    # session, not on a bar boundary, so its recent periods are fetched again each hour.
    now_ms = int(now.timestamp() * 1000)
    slot = min(step, 3_600_000) if kind == "stock" else step
    key = (kind, symbol, interval, start_ms, end_ms // step, now_ms // slot if end_ms > now_ms - 2 * step else None)
    if key in _mk_cache:
        return _mk_cache[key].copy()
    fetch = {"crypto": _binance, "stock": _yahoo, "forex": _ecb}[kind]
    frame = fetch(symbol, interval, start_ms, end_ms, now_ms)
    frame.index.name = "time"
    while len(_mk_cache) >= CACHE_SIZE:           # a long-running MCP server must not grow without end
        _mk_cache.pop(next(iter(_mk_cache)))
    _mk_cache[key] = frame
    return frame.copy()


def _binance(symbol: str, interval: str, start_ms: int, end_ms: int, now_ms: int) -> pd.DataFrame:
    # Fixed windows of KLINE_LIMIT bars, fetched side by side: a window that comes back
    # short (the exchange was down for a while) doesn't end the history early.
    span = KLINE_LIMIT * INTERVAL_MS[interval]
    windows = [(lo, min(lo + span - 1, end_ms)) for lo in range(start_ms, end_ms, span)]
    paths = [f"/api/v3/klines?symbol={symbol}&interval={interval}&startTime={lo}&endTime={hi}&limit={KLINE_LIMIT}"
             for lo, hi in windows]
    with ThreadPoolExecutor(max_workers=max(1, min(FETCH_WORKERS, len(paths)))) as pool:
        rows: List[list] = [row for batch in pool.map(_mk_get, paths) for row in (batch or [])]
    if not rows:
        raise PriceError(f"No {interval} candles for {symbol} in that period. Is it a Binance spot pair, like "
                         "BTCUSDT or ETHUSDT?")
    frame = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6])]
                          for r in rows],
                         columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    frame = frame[frame["close_time"] < now_ms]          # finished bars only
    frame.index = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame.attrs["source"] = f"Binance spot {symbol} {interval}"
    return frame


def _get_json(url: str, who: str) -> Any:
    """GET a JSON document, waiting and retrying when the source says it is busy (429) or failing."""
    last: Any = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404, 422):
                return json.loads(exc.read().decode("utf-8", "replace") or "{}")
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise PriceError(f"Couldn't get prices from {who} ({last}). Try again in a minute, or bring your own prices "
                     "with load_prices().")


def _yahoo(symbol: str, interval: str, start_ms: int, end_ms: int, now_ms: int) -> pd.DataFrame:
    from urllib.parse import quote
    data = _get_json(f"{YAHOO}{quote(symbol)}?interval={YAHOO_INTERVALS[interval]}&period1={start_ms // 1000}"
                     f"&period2={end_ms // 1000}&events=div,split&includeAdjustedClose=true", "Yahoo Finance")
    result = ((data or {}).get("chart") or {}).get("result") or [{}]
    q = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
    if not result[0].get("timestamp") or not q.get("close"):
        raise PriceError(f"Yahoo Finance has no {interval} prices for {symbol} in that period: write the ticker as "
                         "Yahoo does (AAPL, SPY, NOKIA.HE, ^GSPC), or bring your own prices with load_prices().")
    frame = pd.DataFrame({k: pd.to_numeric(pd.Series(q.get(k)), errors="coerce")
                          for k in ("open", "high", "low", "close", "volume")})
    adj = ((result[0]["indicators"].get("adjclose") or [{}])[0]).get("adjclose")
    if adj is not None:                           # splits and dividends: every price on the adjusted scale
        factor = (pd.to_numeric(pd.Series(adj), errors="coerce") / frame["close"]).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col] * factor
    frame.index = pd.to_datetime(pd.Series(result[0]["timestamp"]), unit="s", utc=True)
    done = frame.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    meta = result[0].get("meta") or {}
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    offset = regular.get("gmtoffset", meta.get("gmtoffset"))
    if interval == "1d" and isinstance(regular.get("end"), (int, float)) and isinstance(offset, (int, float)):
        # A day's bar is done at its session's close on the exchange's clock. Yahoo stamps it at the open, so
        # "a day after the stamp" hid yesterday's session until today's open.
        shift = pd.Timedelta(seconds=offset)
        done = (frame.index + shift).floor("D") + pd.Timedelta(seconds=(regular["end"] + offset) % 86_400) - shift
    finished = done <= pd.Timestamp(now_ms, unit="ms", tz="UTC")
    frame = frame[np.asarray(finished) & frame["close"].notna().to_numpy()]
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["source"] = f"Yahoo Finance {symbol} {interval} (split- and dividend-adjusted)"
    return frame


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _ecb(symbol: str, interval: str, start_ms: int, end_ms: int, now_ms: int) -> pd.DataFrame:
    base, quote_ccy = symbol[:3], symbol[3:]
    if base == quote_ccy:
        raise PriceError(f"{symbol} is one currency twice: write two, like EURUSD.")
    data = _get_json(f"{ECB}{_day(start_ms)}..{_day(end_ms)}?base={base}&symbols={quote_ccy}",
                     "the ECB (Frankfurter)")
    rates = {d: r.get(quote_ccy) for d, r in ((data or {}).get("rates") or {}).items()}
    if not rates:
        raise PriceError(f"No ECB reference rates for {symbol} in that period.")
    close = pd.Series(rates, dtype=float).dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    frame = close.sort_index().to_frame("close")
    if interval == "1w":
        frame = frame.resample("W-MON", label="left", closed="left").last().dropna()
    frame.attrs["source"] = f"ECB reference rates {base}/{quote_ccy}, {interval} (via Frankfurter)"
    return frame


_PRICE_NAMES = {"time": ("time", "date", "datetime", "timestamp", "open_time", "open time", "date (utc)", "date utc"),
                "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
                "close": ("close", "c", "price", "adj close", "last", "close price"),
                "volume": ("volume", "vol", "v")}


def _mk_read_csv(data: str) -> pd.DataFrame:
    """A price CSV from a path or its text. The separator is picked from the header
    among , ; tab | (a one-column file has none: pandas' own sniffer would split
    "close" at the s)."""
    import io
    import os
    if os.path.exists(data):
        with open(data, encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read()
    elif "\n" not in data.strip():
        raise PriceError(f"No price file at {data}.")
    else:
        text = data
    lines = text.strip().splitlines()
    header = lines[0] if lines else ""
    sep = max(",;\t|", key=header.count) if any(c in header for c in ",;\t|") else ","
    decimal = "," if sep == ";" and any("," in line for line in lines[1:6]) else "."
    return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)


def load_prices(data: Union[str, pd.DataFrame, pd.Series, list, tuple, np.ndarray]) -> pd.DataFrame:
    """Your own price history as a frame with a close column (open, high, low and
    volume when given), oldest first. Accepts a CSV path or text, a DataFrame, a
    Series of closes, or a list of closes."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.astype(float).to_frame("close")
    elif isinstance(data, (list, tuple, np.ndarray)):
        frame = pd.DataFrame({"close": np.asarray(data, dtype=float)})
    elif isinstance(data, str):
        frame = _mk_read_csv(data)
    else:
        raise PriceError("Give prices as a CSV path or text, a DataFrame, or a list of closing prices.")
    lower = {str(c).strip().lower(): c for c in frame.columns}
    picked = {}
    for role, names in _PRICE_NAMES.items():
        for name in names:
            if name in lower:
                picked[role] = lower[name]
                break
    if "close" not in picked:
        raise PriceError(f"The prices need a close column (found {', '.join(map(str, frame.columns))}).")
    notes: List[str] = []
    out = pd.DataFrame({role: _mk_numbers(frame[col]) for role, col in picked.items() if role != "time"})
    out.index = frame.index
    if "time" in picked:
        times = _mk_times(frame[picked["time"]])
        good = times.notna().to_numpy()
        if good.mean() >= 0.95:
            if not good.all():          # a footer like "Total": drop the row, keep the dates
                notes.append(f"{int((~good).sum()):,} row(s) without a readable date were left out.")
            out = out[good]
            out.index = pd.DatetimeIndex(times[good], name="time")
        else:
            notes.append("The dates couldn't be read, so the prices are used in the order given.")
    elif isinstance(frame.index, pd.DatetimeIndex):
        out.index = frame.index if frame.index.tz is not None else frame.index.tz_localize("UTC")
    else:
        out = out.reset_index(drop=True)
    unreadable = out["close"].isna()
    if unreadable.sum() > max(2, 0.01 * len(out)):
        example = frame[picked["close"]][unreadable.to_numpy()].iloc[0] if len(frame) == len(unreadable) else ""
        raise PriceError(f"{int(unreadable.sum()):,} closing prices couldn't be read (like {example!r}). Use plain "
                         "numbers such as 1234.56.")
    if unreadable.any():
        notes.append(f"{int(unreadable.sum())} row(s) without a readable close were left out.")
    out = out[~unreadable]
    if isinstance(out.index, pd.DatetimeIndex):
        if not out.index.is_monotonic_increasing:
            notes.append("The prices were not in time order; they were sorted oldest first.")
            out = out.sort_index(kind="stable")
        out = out[~out.index.duplicated(keep="last")]
    if len(out) < 3:
        raise PriceError("Too few prices: a backtest needs a history of bars.")
    if (out["close"] <= 0).any():
        raise PriceError("Prices must be positive.")
    out.attrs["notes"] = notes
    return out


def _mk_numbers(col: pd.Series) -> pd.Series:
    """Prices as numbers, including "1,010.25", "1.010,25" and "$12.50"."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    return col.map(number).astype(float)


def _mk_times(col: pd.Series) -> pd.Series:
    """Dates or epoch numbers; an epoch's unit (s, ms, us, ns) is read from its size."""
    if pd.api.types.is_numeric_dtype(col):
        size = float(np.nanmedian(np.abs(col.to_numpy(float)))) if col.notna().any() else 0.0
        unit = "s" if size < 1e11 else "ms" if size < 1e14 else "us" if size < 1e17 else "ns"
        return pd.Series(pd.to_datetime(col, unit=unit, utc=True, errors="coerce"), index=col.index)
    return pd.Series(pd.to_datetime(col, utc=True, errors="coerce"), index=col.index)


def between(frame: pd.DataFrame, start: Any = None, end: Any = None) -> pd.DataFrame:
    """The part of a price history from start to end (either may be None)."""
    if start is None and end is None:
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise PriceError("These prices have no dates, so start and end can't pick a period.")

    def stamp(value: Any) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise PriceError(f"Can't read the date {value!r}: write it like 2024-01-31.") from None
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    lo = stamp(start) if start is not None else frame.index[0]
    hi = stamp(end) if end is not None else frame.index[-1]
    part = frame[(frame.index >= lo) & (frame.index <= hi)]
    if len(part) < 3:
        raise PriceError("There are almost no prices between that start and end.")
    part.attrs = dict(frame.attrs)
    return part


__all__ = ["prices", "load_prices", "between", "kind_of", "PriceError", "INTERVAL_MS"]
