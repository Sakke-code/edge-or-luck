"""Made-up records whose truth is known, generated from fixed seeds.

Every sample is a CSV export judged through the same public judge(), so a
click on the page shows exactly what an upload of that file would. The tests
hold the judge to the known answers.

* lucky: no edge at all, picked from 200 such records for looking good;
* real: a small, steady edge;
* faded: an edge that stopped working;
* trader: four strategies, two of which are one bet, one of which faded;
* small_wins_big_losses: positive percent per trade, money lost (the losing
  trades are the big ones);
* fills: a raw exchange fill export (Binance-style columns);
* compare: a TradingView backtest and the live trades that followed.

Private samples (never in the repository): set EDGEORLUCK_PRIVATE_SAMPLES to
a folder of .csv files (or one .csv). Each becomes a sample on the local web
app, named after its file.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SAMPLES = [
    {"id": "lucky", "title": "A lucky streak", "blurb": "Looks great. Is it?", "group": "main"},
    {"id": "real", "title": "A real edge", "blurb": "Small and steady. Real?", "group": "main"},
    {"id": "faded", "title": "One that faded", "blurb": "It worked. Until it didn't.", "group": "main"},
    {"id": "trader", "title": "4 strategies at once", "blurb": "A trader's whole book.", "group": "main"},
    {"id": "small_wins_big_losses", "title": "Small wins, big losses",
     "blurb": "+1.7% a trade, and still losing money.", "group": "main"},
    {"id": "fills", "title": "An exchange export", "blurb": "Raw buys and sells, paired for you.",
     "group": "extra"},
    {"id": "compare", "title": "Backtest vs live", "blurb": "Did live keep the backtest's promise?",
     "group": "extra"},
]


def _times(n: int, rng: np.random.Generator, start: str = "2025-03-03") -> pd.DatetimeIndex:
    gaps = rng.exponential(scale=9.0, size=n)
    return pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.cumsum(gaps), unit="h")


def _csv(frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def _percent_csv(x: np.ndarray, times: pd.DatetimeIndex, strategy: Optional[List[str]] = None) -> str:
    frame = pd.DataFrame({"close_time": [t.strftime("%Y-%m-%d %H:%M") for t in times],
                          "pnl %": np.round(x, 3)})
    if strategy is not None:
        frame.insert(0, "strategy", strategy)
    return _csv(frame)


def lucky() -> str:
    """No edge: the 10th best of 200 noise records of 80 trades."""
    finals = sorted(((float(np.random.default_rng(s).normal(0.0, 2.4, 80).sum()), s) for s in range(200)),
                    reverse=True)
    seed = finals[9][1]
    x = np.random.default_rng(seed).normal(0.0, 2.4, 80)
    return _percent_csv(x, _times(80, np.random.default_rng(1000 + seed)))


def real() -> str:
    """+0.35% a trade with a 1.8% swing, 600 trades, from the first seed
    whose average is within 0.02 of the truth (a typical draw)."""
    seed = next(s for s in range(1000)
                if abs(np.random.default_rng(s).normal(0.35, 1.8, 600).mean() - 0.35) < 0.02)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.35, 1.8, 600)
    return _percent_csv(x, _times(600, rng))


def faded() -> str:
    rng = np.random.default_rng(11)
    x = np.concatenate((rng.normal(0.5, 2.0, 300), rng.normal(-0.8, 2.0, 60)))
    return _percent_csv(x, _times(360, rng))


def trader() -> str:
    """Four strategies: a real edge, its twin on the same days (one bet), no
    edge at all, and one that faded."""
    rng = np.random.default_rng(23)
    days = pd.date_range("2025-01-06", periods=300, freq="D", tz="UTC")
    shock = rng.normal(0.0, 1.0, len(days))
    rows: List[Tuple[str, pd.Timestamp, float]] = []
    for i, d in enumerate(days):
        when = d + pd.Timedelta(hours=float(rng.uniform(1, 23)))
        rows.append(("Breakout BTC", when, 0.75 + 1.4 * shock[i] + rng.normal(0, 1.0)))
        rows.append(("Trend ETH", when + pd.Timedelta(minutes=7), 0.15 + 1.4 * shock[i] + rng.normal(0, 1.1)))
    sol_t = days[0] + pd.to_timedelta(np.sort(rng.uniform(0, 24 * 300, 260)), unit="h")
    rows += [("Mean-revert SOL", t, float(v)) for t, v in zip(sol_t, rng.normal(-0.05, 1.8, 260))]
    news = np.concatenate((rng.normal(0.6, 2.0, 300), rng.normal(-0.9, 2.0, 60)))
    news_t = days[0] + pd.to_timedelta(np.sort(rng.uniform(0, 24 * 300, len(news))), unit="h")
    rows += [("News scalper", t, float(v)) for t, v in zip(news_t, news)]
    rows.sort(key=lambda r: (r[1], r[0]))
    frame = pd.DataFrame({"strategy": [r[0] for r in rows],
                          "close_time": [r[1].strftime("%Y-%m-%d %H:%M") for r in rows],
                          "pnl %": np.round([r[2] for r in rows], 3)})
    return _csv(frame)


def small_wins_big_losses() -> str:
    """Three in four trades win about 3.5% on small positions; the losers
    lose about 3% on positions six times bigger. Percent per trade is about
    +1.7%; the money is negative."""
    rng = np.random.default_rng(7)
    n = 160
    win = rng.random(n) < 0.75
    pct = np.where(win, rng.normal(3.5, 0.8, n), rng.normal(-3.0, 1.0, n))
    size = np.where(win, rng.normal(40, 6, n), rng.normal(240, 30, n))
    money = pct / 100.0 * size
    times = _times(n, rng, "2025-05-01")
    return _csv(pd.DataFrame({"close_time": [t.strftime("%Y-%m-%d %H:%M") for t in times],
                              "symbol": "ALT/USDT", "position_usdt": np.round(size, 2),
                              "pnl_usdt": np.round(money, 4), "pnl %": np.round(pct, 3)}))


def fills() -> str:
    """A Binance-style spot trade history: Date(UTC), Pair, Side, Price,
    Executed, Amount, Fee, newest first, fees in the base coin on buys and in
    USDT on sells, one sell from before the export and one position left open."""
    rng = np.random.default_rng(5)
    rows = []
    t = pd.Timestamp("2025-02-03 09:00")
    price = {"BTCUSDT": 96000.0, "ETHUSDT": 2700.0}
    rows.append((t - pd.Timedelta(days=2), "ETHUSDT", "SELL", 2750.0, 0.05))
    for k in range(70):
        pair = "BTCUSDT" if k % 3 else "ETHUSDT"
        base = 0.002 if pair == "BTCUSDT" else 0.06
        price[pair] *= float(np.exp(rng.normal(0.0, 0.015)))
        buys = 2 if k % 7 == 0 else 1
        held = 0.0
        for b in range(buys):
            rows.append((t, pair, "BUY", round(price[pair], 2), base))
            held += base * 0.999
            t += pd.Timedelta(hours=2)
            price[pair] *= float(np.exp(rng.normal(-0.002, 0.01)))
        t += pd.Timedelta(hours=float(rng.uniform(3, 30)))
        price[pair] *= float(np.exp(rng.normal(0.003, 0.025)))
        rows.append((t, pair, "SELL", round(price[pair], 2), round(held, 6)))
        t += pd.Timedelta(hours=float(rng.uniform(2, 20)))
    rows.append((t, "BTCUSDT", "BUY", round(price["BTCUSDT"], 2), 0.002))
    rows.sort(key=lambda r: r[0], reverse=True)
    lines = ["Date(UTC),Pair,Side,Price,Executed,Amount,Fee"]
    for when, pair, side, p, q in rows:
        coin = pair[:-4]
        fee = f"{q * 0.001:.8f}{coin}" if side == "BUY" else f"{p * q * 0.001:.8f}USDT"
        lines.append(f"{when:%Y-%m-%d %H:%M:%S},{pair},{side},{p:.2f},{q:.8f}{coin},{p * q:.8f}USDT,{fee}")
    return "\n".join(lines) + "\n"


def tradingview(x_pct: np.ndarray, start: str, seed: int, price0: float = 100.0) -> str:
    """A TradingView "List of trades" (new column names), newest first."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(start)
    price, cum, cum_pct, capital = price0, 0.0, 0.0, 10_000.0
    out = []
    for i, r in enumerate(x_pct, start=1):
        price *= float(np.exp(rng.normal(0.0, 0.01)))
        entry_t, entry_p = t, price
        t += pd.Timedelta(hours=float(rng.uniform(4, 40)))
        exit_p = entry_p * (1 + r / 100.0)
        price = exit_p
        qty = round(1000.0 / entry_p, 4)
        pnl = qty * (exit_p - entry_p)
        cum += pnl
        cum_pct = 100.0 * cum / capital
        out.append((i, entry_t, entry_p, t, exit_p, qty, pnl, r, cum, cum_pct))
        t += pd.Timedelta(hours=float(rng.uniform(1, 20)))
    lines = ["Trade #,Type,Date/Time,Signal,Price USDT,Position size (qty),Position size (value),Net P&L USDT,"
             "Net P&L %,Run-up USDT,Run-up %,Drawdown USDT,Drawdown %,Cumulative P&L USDT,Cumulative P&L %"]
    for i, et, ep, xt, xp, qty, pnl, r, cum, cp in reversed(out):
        value = qty * ep
        lines.append(f"{i},Exit long,{xt:%Y-%m-%d %H:%M},Close,{xp:.4f},{qty},{value:.2f},{pnl:.2f},{r:.2f},"
                     f"{max(pnl, 0):.2f},{max(r, 0):.2f},{min(pnl, 0):.2f},{min(r, 0):.2f},{cum:.2f},{cp:.2f}")
        lines.append(f"{i},Entry long,{et:%Y-%m-%d %H:%M},Long,{ep:.4f},{qty},{value:.2f},{pnl:.2f},{r:.2f},"
                     f"{max(pnl, 0):.2f},{max(r, 0):.2f},{min(pnl, 0):.2f},{min(r, 0):.2f},{cum:.2f},{cp:.2f}")
    return "\n".join(lines) + "\n"


def compare() -> Tuple[str, str]:
    """(live trades, backtest): the backtest promised +0.5% a trade; live
    delivers about -0.35% over 120 trades."""
    rng = np.random.default_rng(41)
    backtest = tradingview(rng.normal(0.5, 1.4, 400), "2024-01-02 09:00", 1)
    live = rng.normal(-0.35, 1.4, 120)
    times = _times(120, rng, "2025-06-01")
    return _percent_csv(live, times), backtest


MAKERS = {"lucky": lucky, "real": real, "faded": faded, "trader": trader,
          "small_wins_big_losses": small_wins_big_losses, "fills": fills}
NAMES = {"lucky": "Lucky streak", "real": "Steady edge", "faded": "Faded breakout", "trader": "My strategies",
         "small_wins_big_losses": "Grid bot", "fills": "Spot account", "compare": "Live"}


def _private_dir() -> Optional[Path]:
    env = os.environ.get("EDGEORLUCK_PRIVATE_SAMPLES")
    return Path(env) if env else None


def _private() -> Dict[str, Path]:
    root = _private_dir()
    if root is None or not root.exists():
        return {}
    files = [root] if root.is_file() else sorted(p for p in root.iterdir() if p.suffix.lower() in (".csv", ".tsv"))
    return {"private:" + re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_"): p for p in files}


def available() -> List[Dict[str, str]]:
    out = [dict(s) for s in SAMPLES]
    for sid, path in _private().items():
        out.append({"id": sid, "title": path.stem.replace("_", " "), "blurb": "Private sample (local only).",
                    "group": "private"})
    return out


def sample_input(sample_id: str) -> Dict[str, Any]:
    """The judge() arguments for a sample: {"data": ..., "name": ..., and for
    compare "backtest": ...}. KeyError for an unknown id."""
    if sample_id == "compare":
        live, backtest = compare()
        return {"data": live, "backtest": backtest, "name": NAMES["compare"], "source": "live"}
    if sample_id in MAKERS:
        args: Dict[str, Any] = {"data": MAKERS[sample_id](), "name": NAMES[sample_id]}
        if sample_id == "small_wins_big_losses":
            args["source"] = "live"
        return args
    private = _private()
    if sample_id in private:
        return {"data": private[sample_id].read_bytes(), "name": private[sample_id].stem.replace("_", " ")}
    raise KeyError(sample_id)
