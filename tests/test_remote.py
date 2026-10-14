"""The assistant tools over the web: the JSON API with its OpenAPI description, the MCP server at /mcp, studies
that count versions without a session, and signed receipts for frozen rules."""
from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pandas as pd
import pytest

from edgeorluck import market
from edgeorluck.market import between
from edgeorluck.web import tools
from edgeorluck.web.app import create_app
from edgeorluck.web.ledger import Ledger

RSI = {"buy": "rsi(3) < 15", "sell": "rsi(3) > 60"}
A, B = [1.0, -0.5, 2.0] * 20, [1.0, -0.4, 2.0] * 20


@pytest.fixture(autouse=True)
def binance(monkeypatch):
    """Hourly prices up to the current hour, in place of Binance."""
    c = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 3000)))
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=len(c), freq="h")
    px = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.002, "low": c * 0.998, "close": c, "volume": 1.0},
                      index=idx)
    monkeypatch.setattr(market, "prices", lambda symbol, interval, start=None, end=None: between(px, start, end))
    return px


def frozen_earlier(spec, days):
    """A receipt as if the rule had been frozen `days` ago."""
    data = dict(spec, since=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat(timespec="seconds"))
    body = base64.urlsafe_b64encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).decode()
    return body.rstrip("=") + "." + tools._sign(body.rstrip("="))


def test_a_study_counts_every_version_judged_under_it():
    first = tools.judge(pnl=A)
    assert "This session" not in first["markdown"] and first["study"]
    second = tools.judge(pnl=B, study=first["study"])
    assert "has judged 2" in second["markdown"] and second["study"] == first["study"]
    assert "This session" not in tools.judge(pnl=B, study="bad id!")["markdown"]          # a malformed id: new study


def test_a_frozen_rule_is_a_signed_receipt_judged_on_its_later_trades_only():
    out = tools.backtest(**RSI, freeze=True)
    assert out["receipt"] and tools.open_receipt(out["receipt"])["rule"] == dict(RSI, side="long")
    assert "No trades since it was frozen yet" in tools.forward(out["receipt"])["markdown"]
    spec = {k: v for k, v in tools.open_receipt(out["receipt"]).items() if k != "since"}
    later = tools.forward(frozen_earlier(spec, 30))["markdown"]
    assert "only its trades after that count" in later and "## Edge or luck" in later
    body, mac = out["receipt"].split(".")
    data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    data["since"] = "2020-01-01T00:00:00+00:00"                                          # back-dated: refused
    forged = base64.urlsafe_b64encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).decode()
    for bad in (forged.rstrip("=") + "." + mac, "nonsense", ""):
        assert "isn't a receipt this server signed" in tools.forward(bad)["markdown"]
    assert "Its receipt: `" in tools.as_text(out) and "study=" in tools.as_text(out)


def test_the_json_tools_and_their_openapi_description(tmp_path):
    client = create_app(Ledger(tmp_path), rate_limit=1000).test_client()
    spec = client.get("/openapi.json").get_json()
    ops = {op["operationId"] for path in spec["paths"].values() for op in path.values()}
    assert ops == {"judgeTrades", "backtestRule", "tuneRule", "forwardTest", "ruleLanguage"}
    judge_fields = spec["paths"]["/api/tools/judge"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert "pnl" in judge_fields["properties"] and not any("file" in f for f in judge_fields["properties"])
    assert spec["paths"]["/api/tools/forward"]["post"]["requestBody"]["content"]["application/json"]["schema"][
        "required"] == ["receipt"]
    first = client.post("/api/tools/judge", json={"pnl": A}).get_json()
    again = client.post("/api/tools/judge", json={"pnl": B, "study": first["study"]}).get_json()
    assert first["markdown"].startswith("## Edge or luck") and "has judged 2" in again["markdown"]
    frozen = client.post("/api/tools/backtest", json=dict(RSI, freeze=True)).get_json()
    forward = client.post("/api/tools/forward", json={"receipt": frozen["receipt"]}).get_json()
    assert "No trades since it was frozen yet" in forward["markdown"]                  # frozen this very hour
    assert "Functions" in client.get("/api/tools/rules").get_json()["markdown"]
    assert client.post("/api/tools/judge", json={"file_path": "/etc/passwd"}).status_code == 400   # no files
    assert client.post("/api/tools/backtest", json={"sell": "x"}).status_code == 400              # buy is required
    assert client.post("/api/tools/nope", json={}).status_code == 404
    assert client.post("/api/tools/judge", data="x").status_code == 400


def _text(result):
    return "".join(getattr(c, "text", "") for c in result.content)


def test_the_mcp_server_for_the_web(monkeypatch):
    pytest.importorskip("a2wsgi")
    from mcp import Client
    from edgeorluck.web import remote

    async def run():
        async with Client(remote.build()) as client:
            listed = (await client.list_tools()).tools
            assert [t.name for t in listed] == ["judge_trades", "backtest_rule", "forward_test", "tune_rule",
                                                "rule_language"]
            assert not any("file" in p for t in listed for p in t.input_schema["properties"])
            first = _text(await client.call_tool("judge_trades", {"pnl": A}))
            study = first.split("Study `")[1][:12]
            second = _text(await client.call_tool("judge_trades", {"pnl": B, "study": study}))
            assert "has judged 2" in second
            frozen = _text(await client.call_tool("backtest_rule", dict(RSI, freeze=True, study=study)))
            assert "has judged 3" in frozen and "Its receipt: `" in frozen
            receipt = frozen.split("Its receipt: `")[1].split("`")[0]
            assert "No trades since" in _text(await client.call_tool("forward_test", {"receipt": receipt}))
        monkeypatch.setattr(remote, "CALLS_PER_MINUTE", 0)
        async with Client(remote.build()) as client:
            assert "Too many calls" in _text(await client.call_tool("judge_trades", {"pnl": A}))
    asyncio.run(run())


@pytest.mark.slow
def test_the_hosted_app_serves_the_site_the_api_and_mcp_over_http(tmp_path, monkeypatch):
    pytest.importorskip("a2wsgi")
    import socket
    import threading
    import time
    import urllib.request
    import uvicorn
    from mcp import Client
    from edgeorluck.web import remote
    monkeypatch.setenv("EDGEORLUCK_DATA_DIR", str(tmp_path))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(remote.make_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert json.load(urllib.request.urlopen(base + "/healthz"))["ok"]
        assert json.load(urllib.request.urlopen(base + "/openapi.json"))["servers"][0]["url"] == base

        async def run():
            async with Client(base + "/mcp") as client:
                return [t.name for t in (await client.list_tools()).tools]
        assert "forward_test" in asyncio.run(run())
    finally:
        server.should_exit = True


def test_usage_is_counted_without_knowing_who(tmp_path, monkeypatch):
    from edgeorluck.web.usage import Usage, assistant
    assert assistant("Mozilla/5.0; compatible; ChatGPT-User/1.0") == "ChatGPT" and assistant("claude-ai") == "Claude"
    assert assistant("curl/8.4") == ""
    client = create_app(Ledger(tmp_path), rate_limit=1000).test_client()
    assert client.get("/api/usage").get_json()["total"] == 0
    client.get("/api/judge?pnl=" + ",".join(map(str, A)), headers={"User-Agent": "ChatGPT-User/1.0"})
    client.post("/api/tools/judge", json={"pnl": A}, headers={"User-Agent": "curl/8.4"})
    usage = client.get("/api/usage").get_json()
    assert usage["total"] == 2 and usage["for_assistants"] == 1 and usage["by_assistant"] == {"ChatGPT": 1}
    assert usage["by_way"] == {"links": 1, "the JSON tools": 1} and "2 checks so far" in usage["sentence"]
    client.application.config["EDGEORLUCK_USAGE"].save()
    stored = Usage(tmp_path)                  # nothing but counts reaches the disk
    assert all(set(k.split(":")[0] for k in day) <= {"page", "link", "tools", "mcp"} for day in stored.days.values())
    assert "pnl" not in (tmp_path / "usage.json").read_text(encoding="utf-8")
    assert client.get("/api/engine").get_json()["tip_url"] is None
    monkeypatch.setenv("EDGEORLUCK_TIP_URL", "https://ko-fi.com/edgeorluck")
    assert "Keep it free" in client.get("/strategies").get_data(as_text=True)
    assert client.get("/api/engine").get_json()["tip_url"] == "https://ko-fi.com/edgeorluck"
