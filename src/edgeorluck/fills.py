"""Exchange fill exports into closed round trips (spot, long-only, FIFO).

A fill export has one row per execution: time, pair, side, price, quantity
and a fee. Per pair, buys open lots and sells close them first-in-first-out.
A round trip runs from flat to flat: the buys that build a position and the
sells that take it back to (nearly) zero are one result, in the quote
currency, net of fees.

* A fee in the quote coin is a cost. A fee in the base coin is coins you did
  not get: the lot is that much smaller and the fee is valued at the fill
  price. A fee in a third coin (BNB, BGB...) cannot be valued from the file
  and is reported, not guessed.
* Sells with no earlier buy in the file (bought before the export starts)
  are not counted, and say so.
* A position still open at the end: its sells so far count one by one, and
  what is left is reported as open.
"""
from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

QUOTES = ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USDE", "DAI", "USD", "EUR", "GBP", "TRY", "BRL", "JPY",
          "AUD", "BTC", "ETH", "BNB", "BGB", "SOL")
DUST = 0.005             # a position below 0.5% of its peak quantity counts as closed


def split_pair(pair: str) -> Tuple[str, str]:
    """('BTC', 'USDT') from 'BTCUSDT', 'BTC/USDT', 'btc-usdt', 'BTCUSDT_SPBL'."""
    text = str(pair).strip().upper()
    text = re.sub(r"_(SPBL|SPOT|UMCBL|DMCBL)$", "", text)
    for sep in ("/", "-", "_", ":", " "):
        if sep in text:
            base, _, quote = text.partition(sep)
            return base.strip(), quote.strip()
    for quote in QUOTES:
        if text.endswith(quote) and len(text) > len(quote):
            return text[: -len(quote)], quote
    return text, ""


_AMOUNT = re.compile(r"^\s*([-+−]?[\d.,\s]*\d[\d.,]*(?:[eE][-+]?\d+)?)\s*([A-Za-z][A-Za-z0-9]*)?\s*$")


def split_amount(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """'0.00100000BTC' -> ('0.00100000', 'BTC'); '12.5' -> ('12.5', None)."""
    if value is None:
        return None, None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return str(value), None
    text = str(value).strip()
    if not text:
        return None, None
    m = _AMOUNT.match(text)
    if m:
        return m.group(1), (m.group(2) or None) and m.group(2).upper()
    return text, None


def side_of(value: Any) -> Optional[int]:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("buy", "b", "bid", "long", "open long", "buy long") or text.startswith("buy"):
        return 1
    if text in ("sell", "s", "ask", "short", "close long", "sell short") or text.startswith("sell"):
        return -1
    return None


def round_trips(rows: List[Dict[str, Any]], number) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Closed round trips from fills.

    ``rows``: dicts with time (Timestamp or NaT), pair, side (+1/-1), price,
    qty, fee (float, may be nan), fee_coin (str or None), quote_total (float or
    nan). ``number`` turns a cell into a float. Returns the trades as a frame
    in the canonical columns, warnings, and facts about the fills."""
    warnings: List[str] = []
    facts: Dict[str, Any] = {"fills": 0, "pairs": [], "quotes": [], "third_coin_fees": 0,
                             "unmatched_sells": 0, "open_positions": []}
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    # Fills at the same second keep the file's own time direction: in a newest-first export the later row is the
    # earlier fill (a buy and its sell in one second would otherwise be paired the wrong way round).
    stamps = [r["time"] for r in rows if not pd.isna(r["time"])]
    newest_first = sum(b < a for a, b in zip(stamps, stamps[1:])) > sum(b > a for a, b in zip(stamps, stamps[1:]))
    for i, r in enumerate(rows):
        r = dict(r, _i=-i if newest_first else i)
        by_pair.setdefault(r["pair"], []).append(r)
    trades: List[Dict[str, Any]] = []
    unknown_fee_coins: Dict[str, int] = {}
    for pair, fills in by_pair.items():
        base, quote = split_pair(pair)
        facts["pairs"].append(pair)
        if quote and quote not in facts["quotes"]:
            facts["quotes"].append(quote)
        fills.sort(key=lambda r: (r["time"] if not pd.isna(r["time"]) else pd.Timestamp.min.tz_localize("UTC"),
                                  r["_i"]))
        lots: Deque[List[float]] = deque()      # [qty, cost per coin (quote, fees in)]
        held = 0.0                              # coins in the lots, kept as a running total (summing the lots
                                                # on every fill made a long never-flat history quadratic)
        episode: Optional[Dict[str, Any]] = None
        peak = 0.0
        for f in fills:
            price, qty = f["price"], f["qty"]
            if not (math.isfinite(price) and price > 0):
                continue
            if not math.isfinite(qty) or qty <= 0:
                total = f.get("quote_total", float("nan"))
                if math.isfinite(total) and total > 0:
                    qty = total / price
                else:
                    continue
            facts["fills"] += 1
            fee, coin = f.get("fee", float("nan")), (f.get("fee_coin") or "").upper()
            fee = abs(fee) if math.isfinite(fee) else 0.0
            fee_in_base = fee if coin and coin == base else 0.0
            fee_in_quote = fee if (not coin or coin == quote) else 0.0
            if fee and coin and coin not in (base, quote):
                unknown_fee_coins[coin] = unknown_fee_coins.get(coin, 0) + 1
            fee_value = fee_in_quote + fee_in_base * price
            if f["side"] > 0:
                got = qty - fee_in_base
                if got <= 0:
                    continue
                if episode is None:
                    episode = {"entry_time": f["time"], "cost": 0.0, "pnl": 0.0, "fees": 0.0, "sells": [],
                               "first_price": price, "buys": 0}
                    peak = 0.0
                cost = qty * price + fee_in_quote
                lots.append([got, cost / got])
                held += got
                episode["fees"] += fee_value
                episode["buys"] += 1
                peak = max(peak, held)
                continue
            # a sell
            if held <= 0 or not lots:
                facts["unmatched_sells"] += 1
                continue
            sold = min(qty, held)
            if qty > held * (1 + 1e-9) + 1e-12:
                facts["unmatched_sells"] += 1
            proceeds = sold * price - (fee_in_quote + fee_in_base * price) * (sold / qty)
            remaining, basis = sold, 0.0
            while remaining > 1e-15 and lots:
                take = min(remaining, lots[0][0])
                basis += take * lots[0][1]
                lots[0][0] -= take
                held -= take
                remaining -= take
                if lots[0][0] <= 1e-15:
                    held -= lots[0][0]
                    lots.popleft()
            assert episode is not None
            episode["cost"] += basis
            episode["pnl"] += proceeds - basis
            episode["fees"] += fee_value * (sold / qty)
            episode["sells"].append({"time": f["time"], "price": price, "pnl": proceeds - basis, "basis": basis})
            if not lots:
                held = 0.0
            left = held
            if left <= DUST * peak:
                lots.clear()
                held = 0.0
                trades.append(_trip(pair, quote, episode))
                episode = None
        if episode is not None:
            left = sum(l[0] for l in lots)       # once per pair: exact, not the running total
            if episode["sells"]:
                for s in episode["sells"]:            # partly closed: each sell is its own result
                    trades.append({"symbol": pair, "quote": quote, "entry_time": episode["entry_time"],
                                   "time": s["time"], "pnl": s["pnl"],
                                   "pct": 100.0 * s["pnl"] / s["basis"] if s["basis"] > 0 else float("nan"),
                                   "fee": float("nan"), "size": s["basis"], "side": 1,
                                   "entry_price": episode["first_price"], "exit_price": s["price"]})
            if left > 0:
                facts["open_positions"].append({"pair": pair, "quantity": round(left, 10)})
    if facts["unmatched_sells"]:
        warnings.append(f"{facts['unmatched_sells']} sell(s) had no earlier buy in the file (bought before the "
                        "export starts?). They are not counted.")
    if facts["open_positions"]:
        names = ", ".join(p["pair"] for p in facts["open_positions"][:5])
        warnings.append(f"Still open at the end of the file: {names}. Open positions are not judged.")
    if unknown_fee_coins:
        coins = ", ".join(sorted(unknown_fee_coins))
        n = sum(unknown_fee_coins.values())
        facts["third_coin_fees"] = n
        warnings.append(f"{n} fill(s) paid fees in {coins}. The file does not say what that was worth, so those "
                        "fees are not subtracted.")
    frame = pd.DataFrame(trades, columns=["symbol", "quote", "entry_time", "time", "pnl", "pct", "fee", "size",
                                          "side", "entry_price", "exit_price"])
    if len(frame):
        frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    return frame, warnings, facts


def _trip(pair: str, quote: str, ep: Dict[str, Any]) -> Dict[str, Any]:
    last = ep["sells"][-1]
    return {"symbol": pair, "quote": quote, "entry_time": ep["entry_time"], "time": last["time"],
            "pnl": ep["pnl"], "pct": 100.0 * ep["pnl"] / ep["cost"] if ep["cost"] > 0 else float("nan"),
            "fee": ep["fees"], "size": ep["cost"], "side": 1,
            "entry_price": ep["first_price"], "exit_price": last["price"], "buys": ep["buys"]}
