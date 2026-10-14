"""The league (rules frozen in public, judged on their later trades) and the MCP server over the web."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import edgeorluck as eol
from edgeorluck import market
from edgeorluck.market import PriceError, between
from edgeorluck.rulebook import RuleError
from edgeorluck.web.app import create_app
from edgeorluck.web.league import League
from edgeorluck.web.ledger import Ledger

RSI = {"buy": "rsi(3) < 15", "sell": "rsi(3) > 60"}


def closes(n=3000, seed=4, drift=0.0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, 0.01, n)))


def hourly(c, end=None):
    """Hourly bars ending at the current hour (or at end), like Binance's finished bars."""
    end = pd.Timestamp.now(tz="UTC").floor("h") if end is None else pd.Timestamp(end, tz="UTC")
    idx = pd.date_range(end=end, periods=len(c), freq="h")
    opens = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, c) * 1.001, "low": np.minimum(opens, c) * 0.999,
                         "close": c, "volume": 1.0}, index=idx)


def fetcher(books):
    def fetch(symbol, interval, start=None, end=None):
        if symbol not in books:
            raise PriceError(f"{symbol} isn't a Binance spot pair.")
        return between(books[symbol], start, end)
    return fetch


@pytest.fixture
def league(tmp_path):
    return League(tmp_path, fetch=fetcher({"BTCUSDT": hourly(closes(), end="2024-05-05")}))


FROZEN_AT = datetime(2024, 3, 1, 12, 30, tzinfo=timezone.utc)


def test_freezing_is_checked_kept_in_a_chain_and_never_doubled(league, tmp_path):
    a = league.freeze(RSI, "btc/usdt", "1h", name="RSI dip\x07", now=FROZEN_AT)
    assert a["owner_token"] and a["url"] == f"/league/{a['id']}" and "Frozen in public" in a["sentence"]
    again = league.freeze(RSI, "BTCUSDT", "1h", owner_token=a["owner_token"], now=FROZEN_AT)
    assert again["id"] == a["id"] and "owner_token" not in again and len(league.rules) == 1
    league.freeze({"buy": "rsi(5) < 20", "max_hold_bars": 6}, "BTCUSDT", owner_token=a["owner_token"], now=FROZEN_AT)
    assert league.owner_rules(a["owner"]) == 2 and league.rules[a["id"]]["name"] == "RSI dip"
    for rule, symbol, interval, error in ((dict(RSI, buy="rsi(3) < {x}"), "BTCUSDT", "1h", "fixed numbers"),
                                          ({"buy": "close >"}, "BTCUSDT", "1h", "get out"),
                                          (RSI, "ETHUSDT", "1h", "isn't a Binance"),
                                          (RSI, "BTCUSDT", "1m", "runs 15m")):
        with pytest.raises((RuleError, PriceError), match=error):
            league.freeze(rule, symbol, interval, now=FROZEN_AT)
    with pytest.raises(RuleError, match="owner_token"):
        league.freeze(RSI, "BTCUSDT", owner_token="short", now=FROZEN_AT)
    reloaded = League(tmp_path, fetch=league.fetch)
    assert reloaded.head == league.head and reloaded.chain_breaks == 0 and list(reloaded.rules) == list(league.rules)
    lines = league.path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("RSI dip", "RSI dip!")                       # an edit after the fact
    league.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert League(tmp_path, fetch=league.fetch).chain_breaks == 1


def test_only_trades_after_the_freeze_count_and_anyone_can_recompute_them(league):
    a = league.freeze(RSI, "BTCUSDT", "1h", now=FROZEN_AT)
    d = league.detail(a["id"])
    trades = d["backtest"]["trades"]
    assert trades and min(t["entry_time"] for t in trades) >= FROZEN_AT.isoformat()
    assert d["row"]["trades"] == len(trades) and len(d["equity"]) == len(trades) + 1
    px = league.fetch("BTCUSDT", "1h", start=d["entry"]["warmup_from"])
    again = eol.backtest(px, RSI, since=d["entry"]["at"], variants_tried=1)            # what the page tells you to run
    assert [t["return_pct"] for t in again["trades"]] == [t["return_pct"] for t in trades]
    assert "from edgeorluck import backtest, prices" in d["recompute"] and league.detail("nope") is None
    assert f"League rule `{a['id']}`" in league.markdown(a["id"]) and "No rule" in league.markdown("nope")


def test_the_league_wide_verdict_counts_every_rule_anyone_froze(tmp_path):
    books = {"BTCUSDT": hourly(closes(drift=0.003), end="2024-05-05"), "ETHUSDT": hourly(closes(seed=9), end="2024-05-05")}
    league = League(tmp_path, fetch=fetcher(books))
    real = league.freeze({"buy": "close > 0", "max_hold_bars": 3}, "BTCUSDT", "1h", now=FROZEN_AT)
    noise = [league.freeze({"buy": f"rsi({k}) < 30", "sell": f"rsi({k}) > 55"}, "ETHUSDT", "1h", now=FROZEN_AT)
             for k in (3, 5, 8, 13)]
    st = league.standings(wait=True)
    assert st["rules"] == 5 and st["passing"] == 1 and st["rows"][0]["id"] == real["id"]
    assert st["rows"][0]["league_wide"] and not any(r["league_wide"] for r in st["rows"][1:])
    assert {league.detail(n["id"])["row"]["owner_rules"] for n in noise} == {1}        # each a separate owner
    assert "1 shows a real edge across the whole league" in st["sentence"] and st["chain_intact"]


def test_the_league_pages_and_api(tmp_path):
    books = {"BTCUSDT": hourly(closes())}
    league = League(tmp_path / "league", fetch=fetcher(books))
    client = create_app(Ledger(tmp_path / "ledger"), league=league).test_client()
    assert b"Nothing frozen yet" in client.get("/league").data
    made = client.post("/api/league/freeze", json=dict(RSI, symbol="BTCUSDT", interval="1h", name="My <b>rule</b>"))
    receipt = made.get_json()
    assert made.status_code == 201 and receipt["owner_token"]
    page = client.get(receipt["url"]).get_data(as_text=True)
    assert "My &lt;b&gt;rule&lt;/b&gt;" in page and "Check it yourself" in page and "/league?owner=" in page
    board = client.get("/league").get_data(as_text=True)
    assert receipt["id"] in board and 'aria-current=page>League' in board
    assert client.get(f"/league?owner={receipt['owner']}").status_code == 200
    assert client.get("/api/league").get_json()["rules"] == 1
    assert client.get(f"/api/league/{receipt['id']}").get_json()["entry"]["id"] == receipt["id"]
    assert json.loads(client.get("/api/league/chain").get_data(as_text=True).splitlines()[0])["id"] == receipt["id"]
    assert client.get("/league/nope").status_code == 404 and client.get("/api/league/nope").status_code == 404
    bad = client.post("/api/league/freeze", json={"buy": "close >", "symbol": "BTCUSDT"})
    assert bad.status_code == 400 and bad.get_json()["why"]
    assert client.post("/api/league/freeze", data="x").status_code == 400
    codes = [client.post("/api/league/freeze", json={"buy": "close >", "symbol": "BTCUSDT"}).status_code
             for _ in range(12)]
    assert 429 in codes                                                             # 10 freezes an hour per visitor


# ── the MCP server over the web ───────────────────────────────────────────────

def _text(result):
    return "".join(getattr(c, "text", "") for c in result.content)


def test_the_web_mcp_counts_versions_per_study_and_freezes_into_the_league(tmp_path, monkeypatch):
    pytest.importorskip("a2wsgi")
    from mcp import Client
    from edgeorluck.web import remote
    fetch = fetcher({"BTCUSDT": hourly(closes())})
    monkeypatch.setattr(market, "prices", fetch)
    league = League(tmp_path, fetch=fetch)
    server = remote.build(league)

    async def run():
        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            assert [t.name for t in tools] == ["judge_trades", "backtest_rule", "forward_test", "tune_rule",
                                               "rule_language"]
            assert not any("file" in p for t in tools for p in t.input_schema["properties"])
            first = _text(await client.call_tool("judge_trades", {"pnl": [1.0, -0.5, 2.0] * 20}))
            study = first.split("Study `")[1][:12]
            second = _text(await client.call_tool("judge_trades", {"pnl": [1.0, -0.4, 2.0] * 20, "study": study}))
            assert "This session has judged 2" in second and f"Study `{study}`" in second
            fresh = _text(await client.call_tool("judge_trades", {"pnl": [1.0, -0.3, 2.0] * 20}))
            assert "This session" not in fresh                                      # no study: a new one
            frozen = _text(await client.call_tool("backtest_rule", dict(RSI, freeze=True, name="web", study=study)))
            assert "Frozen in public" in frozen and "owner_token: `" in frozen and "has judged 3" in frozen
            rule_id = frozen.split("Its id for forward_test: `")[1][:12]
            assert f"League rule `{rule_id}`" in _text(await client.call_tool("forward_test", {"frozen_id": rule_id}))
        monkeypatch.setattr(remote, "CALLS_PER_MINUTE", 0)
        async with Client(remote.build(league)) as client:
            assert "Too many calls" in _text(await client.call_tool("judge_trades", {"pnl": [1.0, -1.0] * 20}))
        return rule_id
    rule_id = asyncio.run(run())
    assert len(league.rules) == 1 and league.rules[rule_id]["name"] == "web"


@pytest.mark.slow
def test_the_hosted_app_serves_the_site_and_mcp_over_http(tmp_path, monkeypatch):
    pytest.importorskip("a2wsgi")
    import socket
    import threading
    import time
    import urllib.request
    import uvicorn
    from mcp import Client
    from edgeorluck.web import remote
    monkeypatch.setenv("EDGEORLUCK_DATA_DIR", str(tmp_path))
    app = remote.make_app(League(tmp_path, fetch=fetcher({"BTCUSDT": hourly(closes())})))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz"))["ok"]
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/league").status == 200

        async def run():
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                return [t.name for t in (await client.list_tools()).tools]
        assert "forward_test" in asyncio.run(run())
    finally:
        server.should_exit = True
