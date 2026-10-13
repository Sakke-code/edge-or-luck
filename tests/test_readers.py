"""Forgiving readers: every shape a trader or an AI hands over."""
from __future__ import annotations

import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

import edgeorluck
from edgeorluck.fills import split_amount, split_pair
from edgeorluck.readers import number, read_any

from helpers import (backtesting_stats, binance_new, binance_old, bitget_fills, bitget_orders, csv_text,
                     freqtrade_result, round_trips, tradingview_old, vectorbt_portfolio)


# ── cells ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell, value", [
    ("12.5", 12.5), ("-3", -3.0), ("−12,5", -12.5), ("1.234,5", 1234.5), ("1,234.5", 1234.5), ("1,234", 1234.0),
    ("(4.20)", -4.2), ("$1,000", 1000.0), ("12.5%", 12.5), ("0.00100000BTC", 0.001), ("+7", 7.0),
    ("-0,8%", -0.8), ("1 234,5", 1234.5), ("", math.nan), ("n/a", math.nan), ("abc", math.nan), (3, 3.0),
    ("0,5", 0.5), ("5-", -5.0),
])
def test_numbers_from_export_cells(cell, value):
    got = number(cell)
    assert (math.isnan(got) and math.isnan(value)) or got == pytest.approx(value)


def test_pairs_and_amounts():
    assert split_pair("BTCUSDT") == ("BTC", "USDT")
    assert split_pair("eth/usdt") == ("ETH", "USDT")
    assert split_pair("BTCUSDT_SPBL") == ("BTC", "USDT")
    assert split_pair("SOL-FDUSD") == ("SOL", "FDUSD")
    assert split_amount("0.00100000BTC") == ("0.00100000", "BTC")
    assert split_amount("12.5 USDT") == ("12.5", "USDT")


# ── TradingView ───────────────────────────────────────────────────────────────

def test_a_tradingview_export_pairs_entry_and_exit_rows():
    text = tradingview_old([1.5 if i % 3 else -1.0 for i in range(40)], seed=1)
    r = edgeorluck.judge(text.encode(), name="BTC breakout")
    assert r["input"]["format"] == "tradingview" and r.status == "ok"
    card = r["strategies"][0]
    assert card["n"] == 40 and card["unit"] == "USDT"                    # money first
    assert card["money"]["percent_mean"] == pytest.approx(np.mean([1.5 if i % 3 else -1.0 for i in range(40)]),
                                                          abs=0.01)
    assert r["numbers"]["first_trade"] < r["numbers"]["last_trade"]       # newest-first file, oldest-first trades
    assert r["input"]["source_hint"] if "source_hint" in r["input"] else True


def test_the_newer_tradingview_columns_with_european_numbers():
    rows = [["Trade #", "Type", "Date/Time", "Signal", "Price USD", "Net P&L USD", "Net P&L %",
             "Cumulative P&L USD", "Cumulative P&L %"]]
    for i in range(1, 36):
        pnl, pct = ('"1.234,5"', "1,2%") if i == 1 else ("−12,5", "−0,8%")
        rows.append([i, "Entry short", f"2026-02-{1 + i % 27:02d} 09:00", "S", "100,0", pnl, pct, "0", "0"])
        rows.append([i, "Exit short", f"2026-02-{1 + i % 27:02d} 09:30", "C", "99,0", pnl, pct, "0", "0"])
    r = read_any(csv_text(rows, sep=";"))
    [st] = r.strategies
    assert r.format == "tradingview" and st.unit == "USD"
    assert sorted(np.round(st.pnl, 6).tolist()) == [-12.5] * 34 + [1234.5]
    assert sorted(np.round(st.pct, 6).tolist()) == [-0.8] * 34 + [1.2]
    assert set(st.side.tolist()) == {-1.0}


def test_percent_only_tradingview_is_judged_in_percent():
    rows = [["Trade #", "Type", "Date/Time", "Profit %"]]
    for i in range(1, 41):
        rows += [[i, "Entry long", f"2025-01-{1 + i % 28:02d} 10:00", "0.5"],
                 [i, "Exit long", f"2025-01-{1 + i % 28:02d} 12:00", f"{(-1) ** i * 0.8 + 0.1:.2f}"]]
    [st] = read_any(csv_text(rows)).strategies
    assert st.unit == "%" and len(st.pnl) == 40


# ── tables of trades ──────────────────────────────────────────────────────────

def test_a_strategy_column_splits_one_file_and_times_order_the_trades():
    rows = [["strategy", "close_time", "pnl"]]
    for i in range(60):
        rows.append(["alpha" if i % 2 else "beta", f"2026-03-{1 + (59 - i) % 28:02d}T{i % 24:02d}:00:00Z", i])
    r = read_any(csv_text(rows))
    assert sorted(st.name for st in r.strategies) == ["alpha", "beta"]
    for st in r.strategies:
        assert st.unit == "money" and all(a <= b for a, b in zip(st.times, st.times[1:]))


def test_fee_column_unit_from_header_and_a_title_line_above_the_header():
    text = "My export, Jan 2025\n\nclosed at;symbol;net profit (USDT);fee\n" + "\n".join(
        f"2025-01-{1 + i % 28:02d} 10:{i % 60:02d};BTC;{(i % 5) - 1.5:.2f};0,05" for i in range(40))
    r = read_any(text)
    [st] = r.strategies
    assert st.unit == "USDT" and st.fees is not None and st.fees.sum() == pytest.approx(2.0)


def test_pasted_lists_of_numbers():
    for text, n, unit in [("12.5, -3, 4.1, -2, 7", 5, "money"), ("+12.5%\n-3%\n4.1%", 3, "%"),
                          ("12,5\n-3,2\n4,1", 3, "money"), ("12,5; -3,2; 4,1", 3, "money"),
                          ("[1.5, -2, 3]", 3, "money"), ("1 -2 3 4", 4, "money"), ("$12 -$3 $4", 3, "USD")]:
        r = read_any(text)
        assert r.format == "pnl_list", text
        assert len(r.strategies[0].pnl) == n and r.strategies[0].unit == unit, text
    assert read_any("12,5\n-3,2").strategies[0].pnl.tolist() == [12.5, -3.2]


def test_python_shapes():
    assert read_any([1.0, -2.0, "3%"]).strategies[0].unit == "%"
    assert len(read_any(np.array([1.0, 2.0, -1.0])).strategies[0].pnl) == 3
    assert len(read_any(pd.Series([1.0, 2.0, -1.0])).strategies[0].pnl) == 3
    recs = [{"symbol": "BTC", "pnl": 1.2, "closed_at": "2025-01-02"}, {"symbol": "ETH", "pnl": -0.4,
                                                                          "closed_at": "2025-01-01"}]
    st = read_any(recs).strategies[0]
    assert st.pnl.tolist() == [-0.4, 1.2]                               # sorted by close time
    assert len(read_any({"pnl": [1, 2, -3], "time": ["2025-01-01", "2025-01-02", "2025-01-03"]}).strategies[0].pnl) == 3
    assert len(read_any({"trades": recs}).strategies[0].pnl) == 2
    assert len(read_any(json.dumps(recs)).strategies[0].pnl) == 2
    frame = pd.DataFrame({"profit": [1.0, -2.0, 0.5]}, index=pd.date_range("2025-01-01", periods=3, tz="UTC"))
    st = read_any(frame).strategies[0]
    assert st.times is not None and len(st.pnl) == 3


def test_files_by_path_bytes_and_xlsx(tmp_path):
    rows = [["close time", "pnl %"]] + [[f"2025-02-{1 + i % 28:02d} 10:00", (i % 7) - 2.5] for i in range(50)]
    path = tmp_path / "my_trades.csv"
    path.write_text(csv_text(rows), encoding="utf-8")
    a = edgeorluck.judge(str(path))
    b = edgeorluck.judge(path)
    c = edgeorluck.judge(path.read_bytes())
    d = edgeorluck.judge(path.read_text(encoding="utf-8"))
    assert a["input_fingerprint"] == b["input_fingerprint"] == c["input_fingerprint"] == d["input_fingerprint"]
    assert a["strategies"][0]["name"] == "my trades"
    pytest.importorskip("openpyxl")
    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame({"Notes": ["nothing here"]}).to_excel(writer, sheet_name="Overview", index=False)
        pd.DataFrame(rows[1:], columns=rows[0]).to_excel(writer, sheet_name="List of trades", index=False)
    x = edgeorluck.judge(str(xlsx))
    assert x.status == "ok" and x["strategies"][0]["n"] == 50
    assert x["input_fingerprint"] == a["input_fingerprint"]


def test_utf16_and_tab_separated_text():
    rows = [["Time", "Profit"]] + [[f"2025-03-{1 + i % 28:02d}", (i % 5) - 1] for i in range(40)]
    raw = csv_text(rows, sep="\t").encode("utf-16")
    r = read_any(raw)
    assert r.format == "trades_table" and len(r.strategies[0].pnl) == 40


# ── exchange fills ────────────────────────────────────────────────────────────

def test_fifo_round_trip_with_base_and_quote_fees_by_hand():
    text = csv_text([
        ["Date(UTC)", "Pair", "Side", "Price", "Executed", "Amount", "Fee"],
        ["2025-01-01 10:00:00", "BTCUSDT", "BUY", "100", "1.0BTC", "100USDT", "0.001BTC"],
        ["2025-01-01 11:00:00", "BTCUSDT", "BUY", "110", "1.0BTC", "110USDT", "0.11USDT"],
        ["2025-01-02 10:00:00", "BTCUSDT", "SELL", "120", "1.999BTC", "239.88USDT", "0.23988USDT"],
    ])
    r = read_any(text)
    assert r.format == "fills"
    [st] = r.strategies
    assert st.unit == "USDT" and len(st.pnl) == 1
    assert st.pnl[0] == pytest.approx(1.999 * 120 - 0.23988 - (100 + 110.11))
    assert st.fees[0] == pytest.approx(0.001 * 100 + 0.11 + 0.23988)
    assert st.pct[0] == pytest.approx(100 * st.pnl[0] / 210.11)
    assert r.warnings == []


def test_partial_sells_are_one_round_trip_and_odd_rows_warn():
    text = csv_text([
        ["time", "symbol", "side", "price", "qty", "fee", "fee currency"],
        ["2025-01-01 09:00", "ETH/USDT", "sell", "10", "1", "0", "USDT"],          # sold before the export
        ["2025-01-01 10:00", "BTC/USDT", "buy", "10", "2", "0", "USDT"],
        ["2025-01-01 11:00", "BTC/USDT", "sell", "12", "1", "0", "USDT"],
        ["2025-01-01 12:00", "BTC/USDT", "sell", "9", "1", "0", "USDT"],
        ["2025-01-02 10:00", "BTC/USDT", "buy", "10", "1", "0.5", "BNB"],          # fee in a third coin
        ["2025-01-02 11:00", "BTC/USDT", "sell", "11", "1", "0", "USDT"],
        ["2025-01-03 10:00", "BTC/USDT", "buy", "10", "1", "0", "USDT"],           # still open
    ])
    r = read_any(text)
    [st] = r.strategies
    assert st.pnl.tolist() == pytest.approx([1.0, 1.0])
    text_w = " ".join(r.warnings)
    assert "no earlier buy" in text_w and "Still open" in text_w and "BNB" in text_w


def test_never_flat_positions_count_each_sell():
    text = csv_text([["time", "pair", "side", "price", "amount"],
                     ["2025-01-01", "SOLUSDT", "buy", "10", "10"],
                     ["2025-01-02", "SOLUSDT", "sell", "12", "2"],
                     ["2025-01-03", "SOLUSDT", "sell", "8", "2"]])
    r = read_any(text)
    assert r.strategies[0].pnl.tolist() == pytest.approx([4.0, -4.0]) and "Still open" in r.warnings[0]


@pytest.mark.parametrize("maker", [binance_new, binance_old])
def test_binance_trade_history_exports(maker):
    fills = round_trips(40, seed=3)
    text = maker(sorted(fills, reverse=True))                                  # newest first, as exported
    r = edgeorluck.judge(text)
    assert r["input"]["format"] == "fills" and r.status == "ok"
    card = r["strategies"][0]
    assert card["n"] == 40 and card["unit"] == "USDT" and card["numbers"]["fees"]["paid"] > 0
    assert not any(q["id"] == "fees_included" for q in r["ask_the_user"])     # fees are in the file


def test_bitget_fills_and_order_history():
    fills = [(t, "BTC/USDT", side, price, qty) for t, _, side, price, qty in round_trips(35, seed=4)]
    a = read_any(bitget_fills(fills))
    assert a.format == "fills" and len(a.strategies[0].pnl) == 35
    b = read_any(bitget_orders(fills))
    assert b.format == "fills" and len(b.strategies[0].pnl) == 35             # cancelled orders filled nothing
    assert b.strategies[0].times[0].year == 2025                              # epoch milliseconds


def test_two_quote_currencies_judge_the_main_one():
    fills = round_trips(30, seed=5) + [("2025-06-01 10:00:00", "ETHBTC", "BUY", 0.05, 1.0),
                                       ("2025-06-02 10:00:00", "ETHBTC", "SELL", 0.051, 0.999)]
    r = read_any(binance_new(fills))
    assert r.strategies[0].unit == "USDT" and len(r.strategies[0].pnl) == 30
    assert any("quoted in" in w for w in r.warnings)


# ── what AI-written backtests return ──────────────────────────────────────────

def test_backtesting_py_stats_series_and_its_trades():
    stats = backtesting_stats(n=80, seed=1)
    r = edgeorluck.judge(stats)
    assert r["input"]["format"] == "backtesting.py" and r.status == "ok"
    card = r["strategies"][0]
    assert card["name"] == "SmaCross" and card["n"] == 80
    assert card["benchmark"]["hold_pct"] == 34.5 and "Buy & Hold" in card["benchmark"]["hold_source"]
    assert r["settings"]["source"] is None and any(c["code"] == "backtest_only" for c in r["checks"])
    assert not any(q["id"] == "fees_included" for q in r["ask_the_user"])     # Commission column is filled
    t = edgeorluck.judge(stats["_trades"])
    assert t["input"]["format"] == "backtesting.py" and t["strategies"][0]["n"] == 80


def test_vectorbt_portfolio_and_records():
    pf = vectorbt_portfolio(n=60)
    r = edgeorluck.judge(pf)
    assert r["input"]["format"] == "vectorbt" and r["strategies"][0]["n"] == 59      # the open trade is left out
    assert any("still open" in w for w in r["warnings"])
    many = edgeorluck.judge(vectorbt_portfolio(n=40, columns=3))
    assert len(many["strategies"]) == 3 and many["strategies"][0]["numbers"]["evidence"]["versions_or_strategies"] == 3
    assert edgeorluck.judge(pf.trades.records_readable)["input"]["format"] == "vectorbt"


def test_freqtrade_result_dict_file_and_zip(tmp_path):
    result = freqtrade_result()
    r = edgeorluck.judge(result)
    assert r["input"]["format"] == "freqtrade" and r["strategies"][0]["name"] == "SampleStrategy"
    card = r["strategies"][0]
    assert card["unit"] == "USDT" and card["benchmark"]["hold_pct"] == 25.0
    assert not any(q["id"] == "fees_included" for q in r["ask_the_user"])
    path = tmp_path / "backtest-result-2024-03-01_10-00-00.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert edgeorluck.judge(str(path))["input_fingerprint"] == r["input_fingerprint"]
    zpath = tmp_path / "backtest-result.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("backtest-result_config.json", "{}")
        zf.writestr("backtest-result.json", json.dumps(result))
    assert edgeorluck.judge(str(zpath))["input_fingerprint"] == r["input_fingerprint"]


def test_real_backtesting_py_end_to_end():
    bt_mod = pytest.importorskip("backtesting")
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    from backtesting.test import GOOG, SMA

    class SmaCross(Strategy):
        def init(self):
            self.fast = self.I(SMA, self.data.Close, 10)
            self.slow = self.I(SMA, self.data.Close, 20)

        def next(self):
            if crossover(self.fast, self.slow):
                self.position.close()
                self.buy()
            elif crossover(self.slow, self.fast):
                self.position.close()
                self.sell()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = Backtest(GOOG, SmaCross, cash=10_000, commission=0.002).run()
    import time
    start = time.perf_counter()
    r = edgeorluck.judge(stats)
    elapsed = time.perf_counter() - start
    assert r.status == "ok" and r["input"]["format"] == "backtesting.py"
    assert r["strategies"][0]["n"] == len(stats["_trades"])
    assert r["strategies"][0]["benchmark"]["hold_pct"] == pytest.approx(stats["Buy & Hold Return [%]"])
    assert elapsed < 1.0, f"judge() took {elapsed:.2f}s on a backtesting.py result"
    assert bt_mod is not None


# ── never crash ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("junk", [
    None, "", "   ", "a,b\n1,2\n", "hello world", b"\x00\xff\xfe\x00garbage", 12, 3.5, {"x": 1}, [], [None, "abc"],
    {"a": [1, 2], "b": [1]}, pd.DataFrame(), pd.DataFrame({"a": ["x", "y"]}), "C:/no/such/file.csv",
    "trades.csv", object(), [{"weird": "row"}], "Trade #,Type\n1,Entry long\n", "pnl\n", b"PK\x03\x04broken",
    "time,side,price,qty\n2025-01-01,buy,abc,1\n", float("nan"), [float("inf"), float("nan")],
])
def test_messy_input_never_raises_and_says_what_it_needs(junk):
    r = edgeorluck.judge(junk)
    d = r.to_dict()
    assert d["status"] in ("ok", "need_input")
    json.dumps(d, allow_nan=False)
    if d["status"] == "need_input":
        assert d["need"]["problem"] and d["need"]["needs"] and d["need"]["question_for_trader"]
        assert d["ask_the_user"] and d["ask_the_user"][0]["why"]
        assert r.to_markdown().startswith("## Edge or luck: need the trades")


def test_one_check_reads_a_bounded_number_of_trades(monkeypatch):
    from edgeorluck import readers
    monkeypatch.setattr(readers, "MAX_TRADES", 50)
    r = edgeorluck.judge(list(range(60)))
    assert r.status == "need_input" and "more than the 50" in r["need"]["problem"]
