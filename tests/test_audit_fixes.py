"""Defects found by the code audit of 2026-10-14, each pinned so it can't come back.

Every test names what used to go wrong."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import edgeorluck as eol
from edgeorluck import samples
from edgeorluck.judge import judge_dict
from edgeorluck.rulebook import RuleError, compile_rule, rule_from
from edgeorluck.words import fmt, how_often, share

flask = pytest.importorskip("flask")
from edgeorluck.web.app import create_app  # noqa: E402
from edgeorluck.web.ledger import Ledger  # noqa: E402


def walk(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": rng.lognormal(0, 0.5, n)},
                        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))


# ── the web app ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    return create_app(Ledger(root=tmp_path / "ledger"), rate_limit=1000).test_client()


def test_the_api_never_opens_a_file_named_in_a_request(client, tmp_path):
    # It used to: a path in csv_text, pnl, data or ?csv= was read from the server's disk and judged, and the
    # file's first line came back as "Columns we saw".
    secret = tmp_path / "server_side.csv"
    secret.write_text("api_key_ABC123,owner_x\n" + samples.lucky(), encoding="utf-8")
    for field in ("csv_text", "pnl", "data"):
        j = client.post("/api/judge", json={field: str(secret)}).get_json()
        assert j["status"] == "need_input", field
        assert "api_key_ABC123" not in json.dumps(j)
    j = client.post("/api/judge", json={"pnl": [1, -1] * 20, "backtest_csv_text": str(secret)}).get_json()
    assert j["compare"]["code"] == "need_input"
    for query in ("csv", "pnl"):
        j = client.get(f"/api/judge?format=json&{query}=" + str(secret)).get_json()
        assert j["status"] == "need_input" and "api_key_ABC123" not in json.dumps(j)


def test_the_web_tools_never_open_a_file_named_in_a_request(client, tmp_path):
    # 2026-10-16: the JSON tools (/api/tools/*) and the hosted MCP server (/mcp) read a path-looking csv_text from
    # the server's disk and echoed it as "Columns we saw"; /proc/self/environ, one line, gave the signing secret.
    from edgeorluck import mcp_server
    from edgeorluck.web import tools
    secret = tmp_path / "server_side.csv"
    secret.write_text("api_key_ABC123,owner_x\n" + samples.lucky(), encoding="utf-8")
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\0EDGEORLUCK_SECRET=api_key_ABC123\0PORT=8080\0")
    for path in (secret.as_posix(), environ.as_posix()):
        assert "api_key_ABC123" not in tools.judge(csv_text=path)["markdown"]
        assert "api_key_ABC123" not in tools.judge(pnl=[1, -1] * 20, backtest_csv_text=path)["markdown"]
        assert "api_key_ABC123" not in client.post("/api/tools/judge", json={"csv_text": path}).get_data(as_text=True)
    assert "api_key_ABC123" in mcp_server.judge_trades(file_path=environ.as_posix())   # your own computer: still read
    assert "api_key_ABC123" in mcp_server.judge_trades(csv_text=environ.as_posix())    # ... the same, as before


def test_odd_tool_values_and_huge_results_get_plain_answers(client):
    # 2026-10-16 audit: variants_tried "abc" answered with Python's own error text, results of 1e308 summed to
    # an infinity JSON can't hold, and the hosted tools asked for a file_path they don't take.
    from edgeorluck.web import tools
    r = client.post("/api/tools/judge", json={"pnl": [1, -1] * 20, "variants_tried": "abc"})
    assert r.status_code == 400 and r.get_json()["error"] == "a field has a value of the wrong kind"
    assert client.post("/api/tools/backtest", json={}).get_json()["error"] == "missing field(s) buy"
    huge = eol.judge([1e308] * 20 + [-1e308] * 20, include_chart=True)
    assert None in json.loads(huge.to_json(None))["strategies"][0]["chart"]["you"]
    assert "file_path" not in tools.judge()["markdown"]


def test_the_site_sends_safety_headers_and_never_shows_private_samples_behind_a_proxy(tmp_path, monkeypatch):
    # 2026-10-16 audit: no security headers; an odd thank-you token gave a 500; behind a proxy "this computer"
    # came from a header; a link's trade numbers went into the server's access log.
    from pathlib import Path
    c = create_app(Ledger(root=tmp_path / "ledger"), rate_limit=1000).test_client()
    page = c.get("/")
    assert page.headers["X-Content-Type-Options"] == "nosniff" and page.headers["X-Frame-Options"] == "DENY"
    assert page.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Strict-Transport-Security" not in page.headers
    assert c.get("/", base_url="https://example.org").headers["Strict-Transport-Security"].startswith("max-age=")
    assert c.get("/thanks/a.%C3%A9").status_code == 404
    private = tmp_path / "private"
    private.mkdir()
    (private / "mine.csv").write_text("pnl\n" + "\n".join(["1", "-1"] * 20), encoding="utf-8")
    monkeypatch.setenv("EDGEORLUCK_PRIVATE_SAMPLES", str(private))
    assert "private:mine" in {s["id"] for s in c.get("/api/samples").get_json()}        # on this computer
    monkeypatch.setenv("EDGEORLUCK_PROXIES", "1")
    hosted = create_app(Ledger(root=tmp_path / "hosted"), rate_limit=1000).test_client()
    local_looking = {"X-Forwarded-For": "127.0.0.1"}
    assert "private:mine" not in {s["id"] for s in hosted.get("/api/samples", headers=local_looking).get_json()}
    assert hosted.post("/api/judge", json={"sample": "private:mine"}, headers=local_looking).status_code == 404
    assert "request log" in c.get("/privacy").get_data(as_text=True)
    assert "--no-access-log" in (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")


def test_the_rate_limit_ignores_a_made_up_forwarded_for(tmp_path):
    c = create_app(Ledger(root=tmp_path), rate_limit=3).test_client()
    codes = [c.get("/api/judge?pnl=1,-1,2", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_odd_fields_get_a_plain_answer_not_a_500(client):
    assert client.post("/api/judge", json={"pnl": [1, -1] * 20, "kind": ["x"]}).status_code == 400
    assert client.post("/api/judge", json={"pnl": [1, -1] * 20, "unit": "Q" * 5000}).status_code == 400
    assert client.post("/api/judge", json={"pnl": [1, -1] * 20, "variants_tried": 10 ** 400}).status_code == 200
    deep = "[" * 100_000 + "]" * 100_000
    r = client.post("/api/judge", data=deep, content_type="application/json")
    assert r.status_code in (400, 422) and r.is_json


def test_the_ledger_keeps_no_free_text_and_survives_a_torn_line(tmp_path):
    from edgeorluck.readers import Trades
    book = Ledger(tmp_path)
    times = list(pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.arange(60) * 6, unit="h"))
    x = np.random.default_rng(1).normal(0.3, 2.0, 60)
    book.observe(Trades("s", "Alice Smith acct 4111-1111", x, times), now=(times[-1] + pd.Timedelta(hours=1)).to_pydatetime())
    stored = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "Alice" not in stored and '"unit":"other"' in stored
    with (tmp_path / "calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"id": "half a line')
    again = Ledger(tmp_path)                  # used to raise, and the app wouldn't start
    record = again.track_record()
    assert record["predictions"] == 1 and record["chain_intact"] is False


def test_the_chatgpt_download_link_points_at_the_real_file(client):
    page = client.get("/").get_data(as_text=True)
    assert 'href = "/edgeorluck_single.py"' in page and "/assets/edgeorluck_single.py" not in page
    assert client.get("/edgeorluck_single.py").status_code == 200


# ── the readers ───────────────────────────────────────────────────────────────

def test_a_small_zip_that_unpacks_to_a_lot_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trades.csv", "pnl\n" + "0\n" * 30_000_000)
    assert len(buf.getvalue()) < 2_000_000
    r = eol.judge(buf.getvalue())
    assert r["status"] == "need_input" and "50 MB" in r["need"]["problem"]


def test_monthly_csvs_in_one_zip_are_read_together():
    rows = samples.lucky().strip().splitlines()
    head, body = rows[0], rows[1:]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k in range(2):
            part = body[k * len(body) // 2:(k + 1) * len(body) // 2]
            zf.writestr(f"month{k}.csv", "\n".join([head] + part))
    r = eol.judge(buf.getvalue())
    assert r["numbers"]["trades"] == len(body)


def test_cash_and_running_columns_are_not_taken_for_the_result():
    from edgeorluck.readers import find_result_columns
    assert find_result_columns(["Close time", "Net Amount", "Realized PnL"])[0] == "Realized PnL"
    assert find_result_columns(["Close time", "Running PnL", "PnL"])[0] == "PnL"
    assert find_result_columns(["Close time", "Accumulated profit", "Profit"])[0] == "Profit"


def test_a_total_row_is_not_a_trade():
    text = "close time,pnl\n" + "\n".join(f"2025-01-{d:02d},{0.1 if d % 2 else -0.0}" for d in range(1, 41)) + \
           "\nTotal,2.0\n"
    r = eol.judge(text)
    assert r["numbers"]["trades"] == 40 and any("total row" in w for w in r["warnings"])


def test_decimal_commas_are_decided_per_column():
    # "1,250" alone could be a thousand and a quarter; next to "-0,750" in a semicolon file it is 1.25
    from edgeorluck.readers import read_any
    text = "date;pnl\n" + "\n".join(f"2025-01-0{k + 1};{v}" for k, v in
                                     enumerate(["1,250", "-0,750", "2,125", "-1,500", "0,375"]))
    assert list(read_any(text).strategies[0].pnl) == [1.25, -0.75, 2.125, -1.5, 0.375]


def test_one_blank_date_doesnt_turn_a_newest_first_file_around():
    from edgeorluck.readers import read_any
    rows = [f"2025-01-{d:02d},{d}" for d in range(1, 31)][::-1]       # newest first, results 30 .. 1
    rows[5] = "," + rows[5].split(",")[1]                              # one date missing
    st = read_any("close date,pnl\n" + "\n".join(rows)).strategies[0]
    assert st.pnl[0] == 1 and st.pnl[-1] == 30


def test_breakeven_trades_stay_in_a_trade_table():
    text = "close time,side,pnl\n" + "\n".join(f"2025-02-{d % 28 + 1:02d},long,{[1.0, 0.0, -0.5][d % 3]}"
                                               for d in range(60))
    assert eol.judge(text)["numbers"]["trades"] == 60


def test_percent_results_are_read_as_percent():
    text = "date,pnl\n" + "\n".join(f"2025-03-{d:02d},{'+' if d % 2 else '-'}1.5%" for d in range(1, 29))
    from edgeorluck.readers import read_any
    assert read_any(text).strategies[0].unit == "%"
    assert read_any("close time,pnl\n" + "\n".join(f"2025-03-{d:02d},{d % 3 - 1}" for d in range(1, 29)),
                    unit="%").strategies[0].unit == "%"


def test_a_numbered_list_is_not_read_as_its_numbers():
    from edgeorluck.readers import read_any
    assert list(read_any("1. +2.5%\n2. -1.2%\n3. +0.8%\n4. -0.4%").strategies[0].pnl) == [2.5, -1.2, 0.8, -0.4]


def test_a_fee_column_is_asked_about_and_taken_off_when_told():
    rng = np.random.default_rng(1)
    profit = rng.normal(3.3, 6, 80)
    text = "close time,profit,commission\n" + "\n".join(
        f"{pd.Timestamp('2025-01-01') + pd.Timedelta(hours=6 * k)},{p:.2f},-7" for k, p in enumerate(profit))
    asked = eol.judge(text)
    assert any(q["field"] == "fees_included" for q in asked["ask_the_user"])
    assert next(c for c in asked["checks"] if c["code"] == "fees")["status"] == "warn"
    net = eol.judge(text, fees_included="no")
    assert net["numbers"]["mean_per_trade"]["value"] == pytest.approx(asked["numbers"]["mean_per_trade"]["value"] - 7,
                                                                      abs=1e-4)
    assert net["verdict"]["code"] != "real_edge"


@pytest.mark.parametrize("cell,want", [("EUR 12.50", 12.5), ("-EUR 3", -3.0), ("ETH 0.5", 0.5), ("‐5", -5.0),
                                       ("－5", -5.0), ("1.234.567", 1234567.0), ("+-5", -5.0), ("$1,234.50", 1234.5)])
def test_cells_read_as_the_amount_they_show(cell, want):
    from edgeorluck.readers import number
    assert number(cell) == want


@pytest.mark.parametrize("cell", ["12:30", "1/2", "[1, 2.5]", "2025-01-03"])
def test_cells_that_are_not_amounts_read_as_nothing(cell):
    import math
    from edgeorluck.readers import number
    assert not math.isfinite(number(cell)) and not math.isfinite(number([1, 2.5]))


def test_time_and_price_columns_are_the_right_ones():
    from edgeorluck.readers import read_any
    rows = "\n".join(f"2025-01-{d:02d} 09:00,2025-01-{d:02d} 17:00,{(-1) ** d}" for d in range(1, 31))
    r = read_any("Time Opened,Time Closed,PnL\n" + rows)
    assert r.columns_used["close_time"] == "Time Closed" and r.columns_used["entry_time"] == "Time Opened"
    r = read_any("Exit Price,Symbol,PnL\n" + "\n".join(f"10{d},BTC,{(-1) ** d}" for d in range(30)))
    assert "close_time" not in r.columns_used


def test_dates_are_read_the_same_way_down_a_column():
    from edgeorluck.readers import _times
    us = _times(pd.Series(["01/05/25", "01/13/25", "02/03/25"])).dt.strftime("%Y-%m-%d").tolist()
    eu = _times(pd.Series(["05-01-2025", "13-01-2025", "03-02-2025"])).dt.strftime("%Y-%m-%d").tolist()
    assert us == ["2025-01-05", "2025-01-13", "2025-02-03"] and eu == ["2025-01-05", "2025-01-13", "2025-02-03"]


def test_tradingview_open_trades_and_pasted_exports():
    from edgeorluck.readers import read_any
    head = "Trade #,Type,Signal,Date/Time,Price USDT,Contracts,Profit USDT"
    rows = []
    for i in range(1, 31):
        rows += [f"{i},Exit long,Close,2025-01-{i:02d} 12:00,101,1,{(-1) ** i}",
                 f"{i},Entry long,Long,2025-01-{i:02d} 09:00,100,1,{(-1) ** i}"]
    open_rows = ["31,Exit long,Open,2025-01-31 23:00,70,1,-300", "31,Entry long,Long,2025-01-31 09:00,100,1,-300"]
    r = read_any(head + "\n" + "\n".join(reversed(rows + open_rows)))
    assert len(r.strategies[0].pnl) == 30 and -300 not in r.strategies[0].pnl
    twice = read_any(head + "\n" + "\n".join(reversed(rows)) + "\n" + "\n".join(reversed(rows)))
    assert len(twice.strategies[0].pnl) == 60


def test_a_tag_on_every_row_is_not_a_strategy():
    text = "close time,tag,pnl\n" + "\n".join(f"2025-01-{d % 28 + 1:02d},order-{d},{(-1) ** d}" for d in range(200))
    r = eol.judge(text)
    assert r["summary"]["strategies"] == 1 and r["summary"]["trades"] == 200


def test_headerless_files_and_padded_title_lines():
    from edgeorluck.readers import read_any
    body = "\n".join(f"2025-01-{d:02d},{(-1) ** d * 1.5}" for d in range(1, 9))
    r = read_any(body)
    assert len(r.strategies[0].pnl) == 8 and r.strategies[0].times is not None
    r = read_any("My trades export,,\nclose time,symbol,pnl\n" +
                 "\n".join(f"2025-01-{d:02d},BTC,{(-1) ** d}" for d in range(1, 31)))
    assert len(r.strategies[0].pnl) == 30


def test_fills_with_signed_sells_and_unlabelled_fees():
    from edgeorluck.readers import read_any
    head = "Time,Pair,Side,Price,Quantity,Fee"
    rows = []
    for k in range(20):     # the buy pays 0.001 BTC (0.1% of the coins), the sell 59.94 USDT (0.1% of the money)
        rows += [f"2025-01-{k + 1:02d} 09:00,BTCUSDT,BUY,60000,1,0.001",
                 f"2025-01-{k + 1:02d} 10:00,BTCUSDT,SELL,60000,-0.999,59.94"]
    st = read_any(head + "\n" + "\n".join(rows)).strategies[0]
    assert len(st.pnl) == 20
    assert st.pnl[0] == pytest.approx(-119.94, abs=1e-6)  # a flat trade loses its two 0.1% fees


def test_futures_fills_have_their_opening_fees_taken_off():
    head = "Date(UTC),Symbol,Side,Price,Quantity,Amount,Fee,Realized Profit"
    rows = []
    for i in range(40):
        gross = 5.0 if i % 2 else -3.0
        rows += [f"2025-01-{1 + i % 28:02d} {i // 28 * 2 + 8:02d}:00:00,BTCUSDT,BUY,100,10,1000,0.8,0",
                 f"2025-01-{1 + i % 28:02d} {i // 28 * 2 + 9:02d}:00:00,BTCUSDT,SELL,{100 + gross / 10},10,{1000 + gross},0.8,{gross}"]
    r = eol.judge(head + "\n" + "\n".join(rows))
    assert r["numbers"]["total"] == pytest.approx(-24.0) and r["numbers"]["fees"]["paid"] == pytest.approx(64.0)


def test_freqtrade_trades_without_is_short_are_long():
    trades = [{"profit_abs": 1.0 if k % 2 else -0.5, "profit_ratio": 0.01, "close_date": f"2025-01-{k + 1:02d}",
               **({"is_short": False} if k % 3 else {})} for k in range(28)]
    r = eol.judge({"strategy": {"s": {"trades": trades, "stake_currency": "USDT"}}})
    assert r["strategies"][0]["numbers"]["trades"] == 28
    from edgeorluck.readers import read_any
    assert set(read_any({"strategy": {"s": {"trades": trades}}}).strategies[0].side) == {1.0}


def test_percent_formatted_excel_cells_keep_their_scale(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    import datetime as dt
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Close date", "Return %"])
    for i, v in enumerate([0.015, -0.01, 0.02, -0.005]):
        ws.append([dt.datetime(2025, 1, 1 + i, 10), v])
        ws.cell(row=i + 2, column=2).number_format = "0.00%"
    path = tmp_path / "sheet.xlsx"
    wb.save(path)
    from edgeorluck.readers import read_any
    st = read_any(str(path)).strategies[0]
    assert st.unit == "%" and list(st.pnl) == [1.5, -1.0, 2.0, -0.5]


def test_private_samples_are_only_for_this_computer(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    (private / "my_bot.csv").write_text(samples.lucky(), encoding="utf-8")
    monkeypatch.setenv("EDGEORLUCK_PRIVATE_SAMPLES", str(private))
    c = create_app(Ledger(root=tmp_path / "l"), rate_limit=1000).test_client()
    local = [s["id"] for s in c.get("/api/samples").get_json()]
    away = [s["id"] for s in c.get("/api/samples", environ_base={"REMOTE_ADDR": "203.0.113.9"}).get_json()]
    assert any(i.startswith("private:") for i in local) and not any(i.startswith("private:") for i in away)
    pid = next(i for i in local if i.startswith("private:"))
    assert c.post("/api/judge", json={"sample": pid}, environ_base={"REMOTE_ADDR": "203.0.113.9"}).status_code == 404


def test_a_unit_is_a_short_code_never_free_text():
    r = eol.judge([1.0, -0.5] * 20, unit="USDT")
    assert r["numbers"]["unit"] == "USDT"
    r = eol.judge([1.0, -0.5] * 20, unit="my account at the bank")
    assert "my account" not in r.to_json()


# ── the judge ─────────────────────────────────────────────────────────────────

def test_every_strategy_called_real_shows_a_range_above_zero():
    # e-BH picks the 2nd and 3rd of several strategies at a lower bar than 40 N; their ranges used to be shown
    # at 40 N and could reach below zero while the sentence said "all above zero".
    rng = np.random.default_rng(2)
    rows = [{"strategy": f"S{k}", "pnl": float(v)} for k in range(3) for v in rng.normal(0.45, 1.0, 80)]
    r = judge_dict(pd.DataFrame(rows))
    real = [c for c in r["strategies"] if c["verdict"]["code"] == "real_edge"]
    assert len(real) == 3
    for c in real:
        assert c["edge"]["adjusted_lower"] > 0 and "all above zero" in c["verdict"]["plain"]
        assert not any("more than 0 versions" in w["text"] for w in c["what_would_change_it"])


def test_market_words_fit_a_falling_market_and_shorts():
    rng = np.random.default_rng(4)
    n = 80
    entries = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    exits = entries + pd.Timedelta(hours=20)
    pct = rng.normal(0.7, 1.0, n)
    df = pd.DataFrame({"entry time": entries, "exit time": exits, "side": "short", "profit %": pct})
    r = judge_dict(df, benchmark_return_pct=-60)
    text = json.dumps(r)
    assert "rose -" not in text and "price rose -" not in text


def test_numbers_are_never_rounded_into_a_lie():
    assert share(0.9967) == "over 99%" and share(0.0005) == "under 1%" and share(0.5) == "50%"
    assert fmt(4.41e-7, "BTC") == "+0.000000441 BTC"
    assert how_often(0.9, 20_000) == "most of the time" and how_often(0.25, 20_000) == "about 1 in 4 times"


def test_a_sorted_list_of_40000_results_is_judged_quickly():
    import time
    x = np.sort(np.random.default_rng(1).normal(0.1, 1, 40_000))
    t = time.time()
    judge_dict(list(x), unit="USDT")
    assert time.time() - t < 15          # the losing-streak odds used to take over 30 seconds


def test_the_same_record_gets_the_same_evidence_at_any_scale():
    x = np.random.default_rng(3).normal(0.15, 1, 300)
    evidence = [judge_dict(list(x * s))["numbers"]["evidence"]["e_value"] for s in (1e-9, 1, 1e9)]
    assert max(evidence) / min(evidence) < 1.0001


# ── the backtester ────────────────────────────────────────────────────────────

def test_a_signal_computed_outside_the_rule_is_not_waved_through():
    px = walk()
    px["next_up"] = (px["close"].shift(-1) > px["close"] * 1.002).astype(float)
    with pytest.raises(RuleError, match="only the prices"):
        eol.backtest(px, lambda f: f["next_up"])            # used to pass the look-ahead test: +1.3% a trade
    signal = (px["close"].shift(-1) > px["close"] * 1.002).astype(float)
    bt = eol.backtest(px.drop(columns="next_up"), lambda f: signal)
    assert bt.status == "rule_unchecked" and bt["judgement"] is None


@pytest.mark.parametrize("seed", range(10))
def test_a_rare_peek_is_caught_on_every_price_path(seed):
    # Peeking only when the next bar rises 2% used to slip past 20 test bars on half of the paths.
    rule = lambda f: ((f["close"].shift(-1) / f["close"] - 1) > 0.02).astype(float)
    assert eol.backtest(walk(seed=seed), rule).status == "rule_peeks"


def test_a_backward_fill_in_the_warm_up_is_caught():
    assert eol.backtest(walk(), lambda f: (f["close"] > f["close"].rolling(400).mean().bfill()).astype(float)
                        ).status == "rule_peeks"


def test_prices_newest_first_or_with_repeats_are_put_right():
    px = walk(600)
    bt = eol.backtest(px.iloc[::-1], {"buy": "rsi(3) < 20", "sell": "rsi(3) > 60"})
    assert all(t["exit_bar"] > t["entry_bar"] for t in bt["trades"])
    assert any(c["code"] == "prices" for c in bt["checks"])
    doubled = eol.backtest(px.loc[px.index.repeat(2)], {"buy": "close > open * 1.004", "max_hold_bars": 1})
    clean = eol.backtest(px, {"buy": "close > open * 1.004", "max_hold_bars": 1})
    assert doubled["stats"] == clean["stats"]


def test_costs_must_be_real():
    with pytest.raises(RuleError):
        eol.backtest(walk(), {"buy": "rsi(3) < 20", "sell": "rsi(3) > 60"}, fee_pct=-0.2)
    free = eol.backtest(walk(), {"buy": "rsi(3) < 20", "sell": "rsi(3) > 60"}, fee_pct=0, slippage_pct=0)
    assert next(c for c in free["checks"] if c["code"] == "costs")["status"] == "warn"
    assert free["judgement"]["settings"]["fees_included"] is False


@pytest.mark.parametrize("text", ["close > 1/0", "close > " + "9" * 400, "sma(1e19) > 0", "prev(close, 1e19) > 0"])
def test_rule_mistakes_are_rule_errors(text):
    with pytest.raises(RuleError):
        compile_rule(text)(walk(100))


def test_true_false_count_as_one_and_zero():
    px = walk(300)
    both = compile_rule("(close > open) + (high > low) >= 2")(px)
    assert both.equals((px["close"] > px["open"]) & (px["high"] > px["low"]))


def test_a_setting_is_only_ever_a_number():
    with pytest.raises(RuleError):
        rule_from({"buy": "close > {low:>300000000}", "sell": "close < 0"}, {"low": 1})
    with pytest.raises(RuleError):
        rule_from({"buy": "close > 1", "max_hold_bars": 0.5})


def test_price_files_with_thousands_separators_epochs_and_a_total_row():
    text = "Date,Close\n" + "\n".join(f'2024-01-{d:02d},"1,0{d:02d}.25"' for d in range(1, 21)) + "\nTotal,\n"
    px = eol.load_prices(text)
    assert len(px) == 20 and px["close"].iloc[0] == 1001.25 and isinstance(px.index, pd.DatetimeIndex)
    ms = "open_time,close\n" + "\n".join(f"{1704067200000 + 3_600_000 * k},{100 + k}" for k in range(10))
    assert eol.load_prices(ms).index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_tuning_counts_distinct_rules_and_writes_valid_json():
    px = walk(2500, seed=4)
    tu = eol.tune(px, {"buy": "rsi(3) < {low}", "sell": "rsi(3) > 60"},
                  {"low": np.array([15, 15, 20]), "typo": [1, 2, 3]})
    assert tu["tried"] == 2
    json.loads(tu.to_json())


def test_an_edge_that_vanishes_one_bar_later_changes_the_headline():
    zig = 100 * (1 + 0.02 * (-1.0) ** np.arange(400)) + np.random.default_rng(1).normal(0, 0.05, 400)
    px = pd.DataFrame({"close": zig, "open": np.r_[zig[0], zig[:-1]]},
                      index=pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC"))
    bt = eol.backtest(px, {"buy": "change(1) < 0", "sell": "change(1) > 0"})
    assert bt.status == "fragile" and bt["headline"].startswith("Don't trust this yet")
