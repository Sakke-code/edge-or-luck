"""GET /api/judge: the judge from a plain link, for assistants that can only fetch URLs."""
from urllib.parse import quote

import pytest

from edgeorluck.web.app import _pnl_lines, create_app
from edgeorluck.web.ledger import Ledger

PNL = [1.2, 0.85, -2.4, 0.95, 1.1, -0.3, 0.7, 1.05, -3.1, 0.9, 0.8, 1.15, -0.45, 0.75, 1.3, -2.8, 0.65, 0.95,
       1.0, -0.2, 0.85, 1.1, -2.95, 0.9, 0.7, 1.25, 0.6, -0.35, 0.95, 1.05, -3.2, 0.8, 1.15, 0.7, 0.9]


@pytest.fixture()
def client(tmp_path):
    app = create_app(Ledger(root=tmp_path), rate_limit=1000)
    return app.test_client()


def test_pnl_link_answers_in_markdown(client):
    r = client.get("/api/judge?pnl=" + ",".join(str(v) for v in PNL))
    text = r.get_data(as_text=True)
    assert r.status_code == 200 and r.mimetype == "text/markdown"
    assert "Edge or luck" in text and "35" in text and "For the assistant" in text


def test_json_on_request_matches_post_and_settings_travel_in_the_link(client):
    link = "/api/judge?format=json&pnl=" + ",".join(str(v) for v in PNL)
    got, tried = client.get(link).get_json(), client.get(link + "&variants_tried=40").get_json()
    posted = client.post("/api/judge", json={"pnl": PNL, "chart": False}).get_json()
    assert got["status"] == posted["status"] == "ok" and got["verdict"]["code"] == posted["verdict"]["code"]
    assert got["input_fingerprint"] == posted["input_fingerprint"]
    assert tried["settings"]["variants_tried"] == 40 and got["settings"]["variants_tried"] is None


def test_decimal_commas_need_semicolons(client):
    text = ";".join(f"{v:.2f}".replace(".", ",") for v in PNL)
    got = client.get("/api/judge?format=json&pnl=" + quote(text)).get_json()
    assert got["status"] == "ok" and got["numbers"]["trades"] == 35
    assert got["numbers"]["total"] == pytest.approx(sum(PNL), abs=1e-6)


def test_csv_and_sample_links(client):
    csv = "close_time,pnl %\n" + "\n".join(f"2026-06-{(i % 28) + 1:02d},{v}" for i, v in enumerate(PNL))
    assert client.get("/api/judge?format=json&csv=" + quote(csv)).get_json()["status"] == "ok"
    sample = client.get("/api/judge?format=json&sample=lucky").get_json()
    assert sample["status"] == "ok" and sample["sample"] == "lucky"
    assert client.get("/api/judge?sample=nope").status_code == 404


def test_pnl_lines_splitting():
    assert _pnl_lines("1.2,-0.8,0.5") == "1.2\n-0.8\n0.5"
    assert _pnl_lines("1,20;-0,80") == "1,20\n-0,80"
    assert _pnl_lines("1,20 -0,80\n0,5") == "1,20\n-0,80\n0,5"
