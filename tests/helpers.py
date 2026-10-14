"""Builders for test inputs: exports as exchanges and tools write them.

None of these come from real accounts. Header styles follow public examples
of each export as best we know them (Binance spot trade history old and new,
Bitget spot fills and order history, TradingView's list of trades old and
new); the README asks traders to send a real export when theirs fails.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

Fill = Tuple[str, str, str, float, float]          # (time, pair, side, price, base quantity)


def csv_text(rows: Sequence[Sequence], sep: str = ",") -> str:
    return "\n".join(sep.join(map(str, row)) for row in rows) + "\n"


def binance_new(fills: Iterable[Fill], fee_rate: float = 0.001) -> str:
    """Date(UTC),Pair,Side,Price,Executed,Amount,Fee: coins in the cells,
    buy fees in the base coin, sell fees in USDT."""
    rows = [["Date(UTC)", "Pair", "Side", "Price", "Executed", "Amount", "Fee"]]
    for t, pair, side, price, qty in fills:
        base = pair.replace("USDT", "")
        fee = f"{qty * fee_rate:.8f}{base}" if side == "BUY" else f"{price * qty * fee_rate:.8f}USDT"
        rows.append([t, pair, side, f"{price:.2f}", f"{qty:.8f}{base}", f"{price * qty:.8f}USDT", fee])
    return csv_text(rows)


def binance_old(fills: Iterable[Fill], fee_rate: float = 0.001) -> str:
    """Date(UTC),Market,Type,Price,Amount,Total,Fee,Fee Coin (2018-2021)."""
    rows = [["Date(UTC)", "Market", "Type", "Price", "Amount", "Total", "Fee", "Fee Coin"]]
    for t, pair, side, price, qty in fills:
        rows.append([t, pair, side, price, qty, round(price * qty, 8), round(price * qty * fee_rate, 8), "USDT"])
    return csv_text(rows)


def bitget_fills(fills: Iterable[Fill], fee_rate: float = 0.001) -> str:
    """Bitget-style spot fills: Date,Trading pair,Direction,Price,Amount,
    Total,Fee,Fee coin (fees in the coin received)."""
    rows = [["Date", "Trading pair", "Direction", "Price", "Amount", "Total", "Fee", "Fee coin"]]
    for t, pair, side, price, qty in fills:
        base, quote = pair.split("/")
        if side == "BUY":
            rows.append([t, pair.replace("/", ""), "Buy", price, qty, round(price * qty, 6), round(qty * fee_rate, 8), base])
        else:
            rows.append([t, pair.replace("/", ""), "Sell", price, qty, round(price * qty, 6),
                         round(price * qty * fee_rate, 8), quote])
    return csv_text(rows)


def bitget_orders(fills: Iterable[Fill], cancelled: int = 2) -> str:
    """Bitget-style order history with a status column and some cancelled
    orders (nothing filled), epoch-millisecond times."""
    rows = [["Order ID", "Time", "Pair", "Side", "Order type", "Avg. price", "Filled", "Total", "Fee", "Status"]]
    for i, (t, pair, side, price, qty) in enumerate(fills):
        ms = int(pd.Timestamp(t, tz="UTC").value // 1_000_000)
        rows.append([f"10{i:06d}", ms, pair.replace("/", "_") + "_SPBL", side.lower(), "market", price, qty,
                     round(price * qty, 6), f"-{round(price * qty * 0.001, 8)}", "Filled"])
        if i < cancelled:
            rows.append([f"20{i:06d}", ms + 1000, pair.replace("/", "_") + "_SPBL", "buy", "limit", 0, 0, 0, 0,
                         "Cancelled"])
    return csv_text(rows)


def round_trips(n: int, seed: int, pair: str = "BTCUSDT", drift: float = 0.002, p0: float = 50_000.0,
                qty: float = 0.002) -> List[Fill]:
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2025-01-01 08:00")
    price = p0
    out: List[Fill] = []
    for _ in range(n):
        price *= float(np.exp(rng.normal(0, 0.01)))
        out.append((f"{t:%Y-%m-%d %H:%M:%S}", pair, "BUY", round(price, 2), qty))
        t += pd.Timedelta(hours=float(rng.uniform(2, 20)))
        price *= float(np.exp(rng.normal(drift, 0.02)))
        out.append((f"{t:%Y-%m-%d %H:%M:%S}", pair, "SELL", round(price, 2), round(qty * 0.999, 8)))
        t += pd.Timedelta(hours=float(rng.uniform(1, 10)))
    return out


def tradingview_old(returns_pct: Sequence[float], *, start: str = "2024-01-01 00:00", rising: float = 0.0,
                    seed: int = 0, side: str = "long") -> str:
    """Trade #,Type,Signal,Date/Time,Price USDT,Contracts,Profit USDT,Profit %,
    Cum. Profit USDT,Cum. Profit %,Run-up USDT,Run-up %,Drawdown USDT,Drawdown %
    newest first, entry and exit rows. ``rising`` adds a market drift per
    hour between and during trades."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(start)
    price, cum = 100.0, 0.0
    trades = []
    for i, r in enumerate(returns_pct, start=1):
        gap = float(rng.uniform(2, 12))
        price *= (1 + rising) ** gap
        t += pd.Timedelta(hours=gap)
        entry_t, entry_p = t, price
        hold = float(rng.uniform(4, 30))
        t += pd.Timedelta(hours=hold)
        exit_p = entry_p * (1 + r / 100.0)
        price = exit_p
        qty = 10.0
        pnl = qty * (exit_p - entry_p) * (1 if side == "long" else -1)
        cum += pnl
        trades.append((i, entry_t, entry_p, t, exit_p, qty, pnl, r, cum))
    rows = [["Trade #", "Type", "Signal", "Date/Time", "Price USDT", "Contracts", "Profit USDT", "Profit %",
             "Cum. Profit USDT", "Cum. Profit %", "Run-up USDT", "Run-up %", "Drawdown USDT", "Drawdown %"]]
    for i, et, ep, xt, xp, qty, pnl, r, cum in reversed(trades):
        for kind, when, p in ((f"Exit {side}", xt, xp), (f"Entry {side}", et, ep)):
            rows.append([i, kind, "sig", f"{when:%Y-%m-%d %H:%M}", f"{p:.4f}", qty, f"{pnl:.2f}", f"{r:.2f}",
                         f"{cum:.2f}", f"{cum / 10:.2f}", "0", "0", "0", "0"])
    return csv_text(rows)


def backtesting_stats(n: int = 80, seed: int = 0, edge: float = 0.003, hold_return: float = 34.5,
                      commission: bool = True) -> pd.Series:
    """What backtesting.py's Backtest.run() returns: a Series of numbers plus
    _strategy, _equity_curve and _trades (Size, EntryBar, ExitBar, EntryPrice,
    ExitPrice, SL, TP, PnL, Commission, ReturnPct, EntryTime, ExitTime,
    Duration, Tag)."""
    rng = np.random.default_rng(seed)
    entry_bar = np.sort(rng.choice(np.arange(10, 10 * n), size=n, replace=False))
    exit_bar = entry_bar + rng.integers(1, 9, size=n)
    entry_price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ret = rng.normal(edge, 0.02, n)
    exit_price = entry_price * (1 + ret)
    size = np.where(rng.random(n) < 0.8, 10, -10)
    pnl = size * (exit_price - entry_price)
    comm = np.abs(size) * (entry_price + exit_price) * 0.001 if commission else np.zeros(n)
    times = pd.date_range("2024-01-01", periods=10 * n + 20, freq="h")
    trades = pd.DataFrame({
        "Size": size, "EntryBar": entry_bar, "ExitBar": exit_bar, "EntryPrice": entry_price,
        "ExitPrice": exit_price, "SL": np.nan, "TP": np.nan, "PnL": pnl - comm, "Commission": comm,
        "ReturnPct": np.sign(size) * (exit_price / entry_price - 1),
        "EntryTime": times[entry_bar], "ExitTime": times[exit_bar], "Duration": times[exit_bar] - times[entry_bar],
        "Tag": None,
    })
    return pd.Series({
        "Start": times[0], "End": times[-1], "Return [%]": float(pnl.sum() / 100), "Buy & Hold Return [%]": hold_return,
        "# Trades": n, "Win Rate [%]": float((pnl > 0).mean() * 100), "_strategy": "SmaCross(n1=10,n2=20)",
        "_equity_curve": pd.DataFrame({"Equity": 10_000 + np.cumsum(np.zeros(5))}), "_trades": trades,
    }, dtype=object)


def vectorbt_portfolio(n: int = 60, seed: int = 1, columns: int = 1) -> SimpleNamespace:
    """A stand-in for a vectorbt Portfolio: .trades.records_readable with the
    0.2x column names, some trades still open, one or more columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(columns):
        t = pd.Timestamp("2024-02-01")
        for i in range(n):
            entry = 100 * float(np.exp(rng.normal(0, 0.05)))
            ret = float(rng.normal(0.004, 0.02))
            exit_ = entry * (1 + ret)
            size = 5.0
            fees = 0.001 * size * (entry + exit_)
            rows.append({"Exit Trade Id": i, "Column": (10 + c, 30) if columns > 1 else 0, "Size": size,
                         "Entry Timestamp": t, "Avg Entry Price": entry, "Entry Fees": fees / 2,
                         "Exit Timestamp": t + pd.Timedelta(hours=6), "Avg Exit Price": exit_, "Exit Fees": fees / 2,
                         "PnL": size * (exit_ - entry) - fees, "Return": ret - 0.002, "Direction": "Long",
                         "Status": "Open" if i == n - 1 else "Closed", "Position Id": i})
            t += pd.Timedelta(hours=10)
    return SimpleNamespace(trades=SimpleNamespace(records_readable=pd.DataFrame(rows)))


def freqtrade_result(n: int = 50, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    trades = []
    t = pd.Timestamp("2024-03-01", tz="UTC")
    for i in range(n):
        ratio = float(rng.normal(0.003, 0.02))
        stake = 100.0
        trades.append({"pair": "ETH/USDT" if i % 2 else "BTC/USDT", "stake_amount": stake, "amount": 0.05,
                       "open_date": t.isoformat(), "close_date": (t + pd.Timedelta(hours=5)).isoformat(),
                       "open_rate": 2000.0, "close_rate": 2000.0 * (1 + ratio), "fee_open": 0.001,
                       "fee_close": 0.001, "trade_duration": 300, "profit_ratio": ratio,
                       "profit_abs": round(stake * ratio, 6), "exit_reason": "roi", "is_open": False,
                       "is_short": False})
        t += pd.Timedelta(hours=9)
    return {"strategy": {"SampleStrategy": {"trades": trades, "stake_currency": "USDT", "profit_total": 0.12,
                                            "market_change": 0.25, "total_trades": n}},
            "strategy_comparison": [{"key": "SampleStrategy", "trades": n}]}


# ── stock brokers (public column layouts; synthetic rows) ─────────────────────

def stock_results(n: int = 40, seed: int = 5) -> List[float]:
    rng = np.random.default_rng(seed)
    return [round(float(rng.normal(3, 20)), 2) for _ in range(n)]


def trading212(results: Sequence[float]) -> str:
    rows = ["Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,"
            "Result,Currency (Result),Total,Currency (Total)"]
    rows.append('Deposit,2024-01-02 09:00:00,,,,,"d1",,,,,,,1000.00,"EUR"')
    for i, r in enumerate(results):
        day = (pd.Timestamp("2024-01-03") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append(f'Market buy,{day} 15:30:01,US0378331005,AAPL,"Apple",,b{i},2.0,185.10,USD,1.09,,"EUR",340.12,"EUR"')
        rows.append(f'Market sell,{day} 20:59:01,US0378331005,AAPL,"Apple",,s{i},2.0,186.20,USD,1.09,{r},"EUR",341.50,'
                    '"EUR"')
    rows.append('Dividend (Dividend),2024-03-15 10:00:00,US0378331005,AAPL,"Apple",,,2.0,0.24,USD,1.09,,"EUR",0.41,"EUR"')
    return "\n".join(rows)


def ibkr_activity(results: Sequence[float]) -> str:
    rows = ["Statement,Header,Field Name,Field Value", "Statement,Data,Title,Activity Statement",
            "Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,"
            "Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code"]
    for i, r in enumerate(results):
        day = (pd.Timestamp("2024-01-03") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append(f'Trades,Data,Order,Stocks,USD,MSFT,"{day}, 15:30:01",10,400.1,400.5,-4001,-1,4002,0,4,O')
        rows.append(f'Trades,Data,Order,Stocks,USD,MSFT,"{day}, 19:59:01",-10,401.2,401.2,4012,-1,-4002,{r},0,C;P')
    rows += [f"Trades,SubTotal,,Stocks,USD,MSFT,,0,,,,,,{sum(results):.2f},,",
             f"Trades,Total,,Stocks,USD,,,,,,,,,{sum(results):.2f},,"]
    return "\n".join(rows)


def nordnet_finnish(results: Sequence[float]) -> str:
    rows = ["Id\tKirjauspäivä\tKauppapäivä\tMaksupäivä\tSalkku\tTapahtumatyyppi\tArvopaperi\tISIN\tMäärä\tKurssi\tKorko\t"
            "Kokonaiskulut\tValuutta\tSumma\tValuutta\tHankinta-arvo\tValuutta\tTulos\tValuutta"]
    for i, r in enumerate(results):
        day = (pd.Timestamp("2024-02-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append(f"{9000 + 2 * i}\t{day}\t{day}\t{day}\t123\tOSTO\tNokia\tFI0009000681\t100\t3,50\t0\t9\tEUR\t-359,00\t"
                    "EUR\t359\tEUR\t\t")
        rows.append(f"{9001 + 2 * i}\t{day}\t{day}\t{day}\t123\tMYYNTI\tNokia\tFI0009000681\t100\t3,55\t0\t9\tEUR\t"
                    f"346,00\tEUR\t359\tEUR\t{str(r).replace('.', ',')}\tEUR")
    return "\n".join(rows)


def degiro(prices: Sequence[Tuple[float, float]]) -> str:
    """Transactions.csv: newest first, Date and Time apart, signed quantities, the currency after each amount."""
    rows = ["Date,Time,Product,ISIN,Reference exchange,Venue,Quantity,Price,,Local value,,Value,,Exchange rate,"
            "Transaction and/or third party fees,,Total,,Order ID"]
    for i, (buy, sell) in reversed(list(enumerate(prices))):
        day = (pd.Timestamp("2024-01-03") + pd.Timedelta(days=i)).strftime("%d-%m-%Y")
        rows.append(f"{day},17:20,ASML HOLDING,NL0010273215,EAM,XAMS,-10,{sell},EUR,{sell * 10:.2f},EUR,"
                    f"{sell * 10:.2f},EUR,,-2.00,EUR,{sell * 10 - 2:.2f},EUR,s{i}")
        rows.append(f"{day},15:30,ASML HOLDING,NL0010273215,EAM,XAMS,10,{buy},EUR,-{buy * 10:.2f},EUR,"
                    f"-{buy * 10:.2f},EUR,,-2.00,EUR,-{buy * 10 + 2:.2f},EUR,b{i}")
    return "\n".join(rows)
