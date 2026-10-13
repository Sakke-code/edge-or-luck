"""The web app: its API, its limits, and the ledger that grades its forecasts."""
from __future__ import annotations

import hashlib
import io
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flask")

from edgeorluck import samples
from edgeorluck.readers import Trades
from edgeorluck.web import ledger as ledger_module
from edgeorluck.web.app import PRIVACY, create_app
from edgeorluck.web.ledger import Ledger, forward_probabilities

NOW = pd.Timestamp("2026-03-01", tz="UTC").to_pydatetime()


def strategy(x, start="2026-01-01"):
    times = list(pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.arange(len(x)) * 6, unit="h"))
    return Trades("s", "%", np.asarray(x, dtype=float), times)


def recent_rows(n):
    end = pd.Timestamp.now(tz="UTC").floor("h")
    return [end - pd.Timedelta(hours=3 * (n - i)) for i in range(n)]


# ── the ledger ────────────────────────────────────────────────────────────────

def test_a_forecast_is_made_once_kept_on_reupload_and_graded_when_enough_trades_follow(tmp_path):
    book = Ledger(tmp_path)
    x = np.random.default_rng(1).normal(0.3, 2.0, 80)
    first = book.observe(strategy(x), now=NOW)
    again = book.observe(strategy(x), now=NOW)
    assert len(book.calls) == 1 and "(made " in again["sentence"] and first["probability"] == again["probability"]
    book.observe(strategy(np.concatenate((x, np.full(10, 1.0)))), now=NOW)
    assert book.grades == []
    longer = np.concatenate((x, np.full(30, -1.0), np.random.default_rng(2).normal(0.3, 2.0, 5)))
    graded = book.observe(strategy(longer), now=NOW)
    [grade] = book.grades
    assert grade["outcome"] == 0 and graded["graded_now"] and "ended negative" in graded["sentence"]
    assert len(book.calls) == 2 and len(grade["brier"]) == len(ledger_module.SHRINKAGE)


def test_a_different_history_behind_the_same_opening_trades_is_not_graded(tmp_path):
    book = Ledger(tmp_path)
    x = np.random.default_rng(1).normal(0.3, 2.0, 80)
    book.observe(strategy(x), now=NOW)
    other = np.concatenate((x[:30], np.random.default_rng(9).normal(0, 2, 90)))
    view = book.observe(strategy(other), now=NOW)
    assert book.grades == [] and "(made " in view["sentence"]


def test_the_record_is_chained_and_survives_a_restart(tmp_path):
    book = Ledger(tmp_path)
    for seed in range(3):
        x = np.random.default_rng(seed).normal(0.1, 2.0, 60)
        book.observe(strategy(x), now=NOW)
        book.observe(strategy(np.concatenate((x, np.full(30, 0.5)))), now=NOW)
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    for before, after in zip(lines, lines[1:]):
        assert json.loads(after)["prev"] == hashlib.sha256(before.encode("utf-8")).hexdigest()
    again = Ledger(tmp_path)
    assert again.track_record() == book.track_record() and set(again.open) == set(book.open)
    stored = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "0.1" not in json.loads(lines[0]).get("pnl", "") and "pnl" not in stored     # no trades kept


def test_the_ledger_picks_the_shrinkage_its_own_graded_forecasts_favour(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "MIN_GRADED", 20)
    book = Ledger(tmp_path)
    assert book.chosen_k() == ledger_module.DEFAULT_K
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 2.0, 60)
        x[:30] += 0.8
        book.observe(strategy(x[:30]), now=NOW)
        book.observe(strategy(np.concatenate((x[:30], rng.normal(0.0, 2.0, 30)))), now=NOW)
    assert len(book.grades) == 40 and book.chosen_k() >= 400
    record = book.track_record()
    assert record["graded"] == 40 and record["brier"] is not None and record["calibration"]


def test_a_history_that_stopped_trading_gets_no_forecast(tmp_path):
    book = Ledger(tmp_path)
    view = book.observe(strategy(np.random.default_rng(3).normal(0.2, 2.0, 60)), now=NOW + pd.Timedelta(days=90))
    assert view["probability"] is None and "hasn't traded since" in view["sentence"] and book.calls == {}


def test_forward_probabilities_shrink_toward_a_coin_flip():
    x = np.random.default_rng(4).normal(0.5, 2.0, 100)
    probs = forward_probabilities(x, [0.0, 100.0, 10_000.0, math.inf])
    assert probs[0] > probs[1] > probs[2] > 0.5 and probs[3] == 0.5


# ── the API ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGEORLUCK_PRIVATE_SAMPLES", raising=False)
    app = create_app(Ledger(tmp_path / "ledger"))
    return app.test_client()


def upload(client, text, name="my strategy.csv", **fields):
    data = {"files": (io.BytesIO(text.encode("utf-8")), name), **{k: str(v) for k, v in fields.items()}}
    return client.post("/api/judge", data=data, content_type="multipart/form-data")


def test_the_samples_engine_schema_health_and_llms(client):
    listed = client.get("/api/samples").get_json()
    assert [s["id"] for s in listed if s["group"] == "main"] == ["lucky", "real", "faded", "trader",
                                                                   "small_wins_big_losses"]
    assert all({"id", "title", "blurb"} <= set(s) for s in listed)
    info = client.get("/api/engine").get_json()
    assert info["version"] and info["changelog"]
    assert client.get("/api/schema").get_json()["title"].startswith("edge or luck")
    assert client.get("/healthz").get_json()["ok"] is True
    llms = client.get("/llms.txt")
    assert llms.status_code == 200 and llms.mimetype == "text/plain" and b"edge or luck" in llms.data.lower()


def test_a_sample_is_judged_and_never_enters_the_track_record(client):
    report = client.post("/api/judge", json={"sample": "trader"}).get_json()
    assert report["summary"]["strategies"] == 4 and report["portfolio"]["pairs"]
    assert all(c["prediction"] is None for c in report["strategies"])
    assert all(c["chart"]["fan"] for c in report["strategies"])
    assert report["markdown"].startswith("## Edge or luck") and report["privacy"] == PRIVACY
    assert client.get("/api/track-record").get_json()["predictions"] == 0
    missing = client.post("/api/judge", json={"sample": "nope"})
    assert missing.status_code == 404 and missing.get_json()["why"]
    tried = client.post("/api/judge", json={"sample": "lucky", "variants_tried": 200}).get_json()
    assert tried["settings"]["variants_tried"] == 200
    assert client.post("/api/judge", json={"sample": "compare"}).get_json()["compare"]["code"] == \
        "worse_than_backtest"


def test_an_upload_is_judged_and_its_forecast_recorded(client):
    rows = ["close_time,pnl %"] + [f"{t.isoformat()},{(i % 5) - 1.5}" for i, t in enumerate(recent_rows(60))]
    report = upload(client, "\n".join(rows), fees_included="false", source="backtest", variants_tried="12").get_json()
    [card] = report["strategies"]
    assert card["name"] == "my strategy" and card["prediction"]["horizon"] == 30
    assert report["track_record"]["predictions"] == 1 and "never stored" in report["privacy"]
    assert report["settings"]["fees_included"] is False and report["settings"]["source"] == "backtest"
    assert report["settings"]["variants_tried"] == 12


def test_several_files_and_a_backtest_in_one_upload(client):
    a = "pnl %\n" + "\n".join(str((i % 5) - 1.6) for i in range(50))
    b = "pnl %\n" + "\n".join(str((i % 3) - 0.9) for i in range(50))
    bt = "pnl %\n" + "\n".join(str((i % 7) - 2.0) for i in range(200))
    data = {"files": [(io.BytesIO(a.encode()), "alpha.csv"), (io.BytesIO(b.encode()), "beta.csv")]}
    report = client.post("/api/judge", data=data, content_type="multipart/form-data").get_json()
    assert [c["name"] for c in report["strategies"]] == ["alpha", "beta"]
    data = {"files": (io.BytesIO(a.encode()), "live.csv"), "backtest": (io.BytesIO(bt.encode()), "bt.csv")}
    report = client.post("/api/judge", data=data, content_type="multipart/form-data").get_json()
    assert report["compare"]["code"] in ("consistent", "worse_than_backtest")


def test_json_bodies(client):
    r = client.post("/api/judge", json={"pnl": [1.0, -0.5, 2.0] * 15, "unit": "USDT", "chart": False}).get_json()
    assert r["strategies"][0]["unit"] == "USDT" and r["strategies"][0]["chart"] is None
    r = client.post("/api/judge", json={"csv_text": "pnl\n1\n-2\n3\n"}).get_json()
    assert r["verdict"]["code"] == "too_early"
    r = client.post("/api/judge", json={"pnl": [0.2, -0.1] * 40, "backtest_pnl": [0.3, -0.1] * 100}).get_json()
    assert r["compare"] is not None


def test_bad_requests_are_answered_plainly(client):
    for bad in (client.post("/api/judge", json={}), client.post("/api/judge", data={}),
                client.post("/api/judge", json=[1, 2])):
        assert bad.status_code == 400 and {"error", "why"} <= set(bad.get_json())
    big = {"files": (io.BytesIO(b"pnl\n" + b"1.0\n" * 3_000_000), "big.csv")}
    too_big = client.post("/api/judge", data=big, content_type="multipart/form-data")
    assert too_big.status_code == 413 and too_big.get_json()["why"]
    odd = upload(client, "a,b\n1,2\n", "odd.csv").get_json()
    assert odd["status"] == "need_input" and odd["need"]["problem"] and odd["strategies"] == []
    assert client.get("/nope").status_code == 404 and client.get("/nope").get_json()["error"]
    assert client.get("/api/judge").status_code == 405


def test_uploads_stay_in_memory(client, monkeypatch):
    import tempfile
    def no_disk(*a, **k):
        raise AssertionError("an upload was spooled to disk")
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", no_disk)
    monkeypatch.setattr(tempfile, "TemporaryFile", no_disk)
    import werkzeug.formparser as formparser
    monkeypatch.setattr(formparser, "SpooledTemporaryFile", no_disk, raising=False)
    monkeypatch.setattr(formparser, "TemporaryFile", no_disk, raising=False)
    lines = [f"{(i % 9) - 3.9:.4f}" for i in range(120_000)]
    big = "\n".join(["pnl %"] + lines)                                      # about 1 MB
    report = upload(client, big, "big.csv").get_json()
    assert report["status"] == "ok" and report["strategies"][0]["n"] == 120_000


def test_the_rate_limit(tmp_path):
    client = create_app(Ledger(tmp_path), rate_limit=3).test_client()
    codes = [client.post("/api/judge", json={"pnl": [1, 2, 3]}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert client.post("/api/judge", json={"pnl": [1]}).get_json()["why"]


def test_the_page_assets_and_the_one_file_download(client):
    page = client.get("/")
    assert page.status_code == 200 and b"edge or luck" in page.data.lower()
    single = client.get("/edgeorluck_single.py")
    assert single.status_code == 200 and "attachment" in single.headers["Content-Disposition"]
    compile(single.data.decode("utf-8"), "edgeorluck_single.py", "exec")


def test_validation_endpoint(client):
    r = client.get("/api/validation")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "false_real_edge" in r.get_json()


def test_private_samples_load_from_outside_the_repo(tmp_path, monkeypatch):
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "my_bot.csv").write_text("pnl_usdt\n" + "\n".join(str((i % 5) - 2.2) for i in range(50)),
                                       encoding="utf-8")
    monkeypatch.setenv("EDGEORLUCK_PRIVATE_SAMPLES", str(folder))
    client = create_app(Ledger(tmp_path / "ledger")).test_client()
    listed = client.get("/api/samples").get_json()
    [private] = [s for s in listed if s["group"] == "private"]
    assert private["id"] == "private:my_bot"
    report = client.post("/api/judge", json={"sample": "private:my_bot"}).get_json()
    assert report["status"] == "ok" and report["strategies"][0]["unit"] == "USDT"
    assert client.get("/api/track-record").get_json()["predictions"] == 0
    monkeypatch.delenv("EDGEORLUCK_PRIVATE_SAMPLES")
    assert all(s["group"] != "private" for s in samples.available())
