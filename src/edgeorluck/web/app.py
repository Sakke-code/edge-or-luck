"""The web app: the page, and a small JSON API around the same judge().

GET  /                      the page (static/index.html)
GET  /assets/<path>         the page's assets
GET  /healthz               {"ok": true}
GET  /llms.txt              what this is, for AI assistants
GET  /api/schema            the result's JSON Schema
GET  /api/samples           [{id, title, blurb, group}]
GET  /api/engine            {version, changelog}
GET  /api/track-record      how good the app's own forecasts have been
GET  /api/validation        how often the judge is wrong, measured (VALIDATION.md)
GET  /edgeorluck_single.py  the judge in one file, for chatbots without installs
POST /api/judge             multipart: files (one or more), backtest (optional),
                            and fields variants_tried, benchmark_return_pct,
                            fees_included, source, unit; or JSON: {sample} or
                            {csv_text | pnl | rows, ...the same fields...,
                            backtest_csv_text | backtest_pnl, chart}

Uploads are read in memory and never written anywhere. For uploads the
ledger keeps a one-way fingerprint and a forecast (see ledger.py). Errors are
JSON {"error", "why"} with a 4xx code.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from flask import Flask, Request, Response, jsonify, request, send_from_directory

from .. import samples
from ..judge import CHANGELOG, ENGINE_VERSION, analyze, make_settings
from ..readers import ReadResult, read_any
from ..report import to_markdown
from ..schema import schema
from .ledger import Ledger

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
PACKAGE = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 10 * 1024 * 1024
RATE_LIMIT = 30                 # judge calls per address per window
RATE_WINDOW_S = 60.0

PRIVACY = ("Your file is read in memory and never stored. For each strategy with 30 or more trades we keep only: a "
           "one-way fingerprint of its first 30 trades and of the whole list, the number of trades, the unit, our "
           "forecast for its next 30 trades, and the time. If you come back with 30 more trades, we also keep "
           "whether those trades ended up or down, to grade the forecast. No trades, no file, no IP address (the "
           "rate limiter holds addresses in memory for one minute).")


class InMemoryRequest(Request):
    """Uploads stay in memory: Werkzeug would spool files over 500 KB to a
    temporary file on disk, and the page promises files are never stored.
    The 10 MB cap bounds the memory."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window_s: float = RATE_WINDOW_S) -> None:
        self.limit, self.window = limit, window_s
        self.hits: Dict[str, Deque[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 10_000:
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
            return True


def _error(message: str, why: str, code: int) -> Tuple[Response, int]:
    return jsonify({"error": message, "why": why}), code


def _merge(reads: List[ReadResult]) -> ReadResult:
    """Several uploaded files: every strategy in one judgement."""
    if len(reads) == 1:
        return reads[0]
    found = [r for r in reads if r.strategies]
    merged = ReadResult(strategies=[st for r in found for st in r.strategies],
                        format=found[0].format if found and all(r.format == found[0].format for r in found)
                        else "mixed",
                        rows=sum(r.rows for r in reads),
                        columns_seen=sorted({c for r in reads for c in r.columns_seen}),
                        columns_used=found[0].columns_used if found else {},
                        warnings=[w for r in reads for w in r.warnings],
                        source_hint=found[0].source_hint if found else None)
    for r in reads:
        if r.need and not r.strategies:
            merged.warnings.append(r.need.get("problem", "A file could not be read."))
    names: Dict[str, int] = {}
    for st in merged.strategies:
        names[st.name] = names.get(st.name, 0) + 1
        if names[st.name] > 1:
            st.name = f"{st.name} ({names[st.name]})"
    if not merged.strategies:
        merged.need = (reads[0].need or {"problem": "No trades found in the files."})
    return merged


def _field(payload: Dict[str, Any], name: str) -> Any:
    value = payload.get(name)
    if isinstance(value, str) and not value.strip():
        return None
    return value


_single_cache: Dict[str, str] = {}


def create_app(ledger: Optional[Ledger] = None, *, rate_limit: int = RATE_LIMIT) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.request_class = InMemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    app.json.sort_keys = False  # type: ignore[attr-defined]
    book = ledger if ledger is not None else Ledger()
    limiter = RateLimiter(rate_limit)
    app.config["EDGEORLUCK_LEDGER"] = book

    @app.get("/")
    def page():
        if (STATIC / "index.html").exists():
            return send_from_directory(STATIC, "index.html")
        return "<!doctype html><title>edge or luck</title><p>edge or luck: the page is not built yet. The API is at " \
               "<a href='/api/samples'>/api/samples</a>.</p>"

    @app.get("/assets/<path:name>")
    def assets(name: str):
        return send_from_directory(STATIC / "assets", name)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "engine": ENGINE_VERSION})

    @app.get("/llms.txt")
    def llms():
        path = PACKAGE / "llms.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else "edge or luck\n"
        return Response(text, mimetype="text/plain; charset=utf-8")

    @app.get("/api/schema")
    def api_schema():
        return jsonify(schema())

    @app.get("/api/samples")
    def api_samples():
        return jsonify(samples.available())

    @app.get("/api/engine")
    def api_engine():
        return jsonify({"name": "edge or luck", "version": ENGINE_VERSION, "changelog": CHANGELOG})

    @app.get("/api/track-record")
    def api_track_record():
        return jsonify(book.track_record())

    @app.get("/api/validation")
    def api_validation():
        path = PACKAGE / "data" / "validation.json"
        if not path.exists():
            return _error("no validation results yet", "Run tools/validate.py to measure how often the judge is "
                                                       "wrong; it writes VALIDATION.md and this file.", 404)
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/edgeorluck_single.py")
    def single_file():
        if ENGINE_VERSION not in _single_cache:
            from ..single import build
            _single_cache[ENGINE_VERSION] = build()
        return Response(_single_cache[ENGINE_VERSION], mimetype="text/x-python; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=edgeorluck_single.py"})

    @app.post("/api/judge")
    def api_judge():
        if not limiter.allow(request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0]):
            return _error("too many requests", f"Up to {limiter.limit} checks a minute per visitor. Wait a minute "
                                               "and try again.", 429)
        payload: Dict[str, Any] = {}
        backtest = None
        source_kind = "upload"
        sample_id = None
        include_chart = True
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return _error("send a JSON object", "Expected {\"sample\": id} or {\"csv_text\": ...}.", 400)
            include_chart = payload.get("chart", True) is not False
            sample_id = payload.get("sample")
            if sample_id:
                try:
                    args = samples.sample_input(str(sample_id))
                except KeyError:
                    return _error(f"no sample called {sample_id!r}", "See GET /api/samples for the ids.", 404)
                source_kind = "sample"
                reads = [read_any(args["data"], name=args.get("name"))]
                if args.get("backtest") is not None:
                    backtest = read_any(args["backtest"], name="Backtest")
                payload.setdefault("source", args.get("source"))
            else:
                data = next((payload[k] for k in ("csv_text", "pnl", "rows", "data") if payload.get(k) is not None),
                            None)
                if data is None:
                    return _error("nothing to judge", "Send csv_text, pnl (a list of results), rows (a list of "
                                                      "trades) or a sample id.", 400)
                if isinstance(data, str) and len(data) > MAX_UPLOAD:
                    return _error("too large", "Up to 10 MB per check.", 413)
                reads = [read_any(data, kind=_field(payload, "kind"), unit=_field(payload, "unit"),
                                  name=_field(payload, "name"))]
                bt = next((payload[k] for k in ("backtest_csv_text", "backtest_pnl", "backtest") if
                           payload.get(k) is not None), None)
                if bt is not None:
                    backtest = read_any(bt, name="Backtest")
        else:
            payload = {k: v for k, v in request.form.items()}
            files = [(f.filename or "trades.csv", f.read()) for f in request.files.getlist("files")]
            files += [(f.filename or "trades.csv", f.read()) for f in request.files.getlist("file")]
            if not files and payload.get("csv_text"):
                files = [("pasted", payload["csv_text"].encode("utf-8"))]
            if not files:
                return _error("no file", "Send one or more trade exports as 'files', or a sample.", 400)
            reads = [read_any(raw, unit=_field(payload, "unit"), kind=_field(payload, "kind"),
                              name=_display_name(fname)) for fname, raw in files]
            bt_files = [(f.filename or "backtest.csv", f.read()) for f in request.files.getlist("backtest")]
            if bt_files:
                backtest = _merge([read_any(raw, name="Backtest") for _, raw in bt_files])
        read = _merge(reads)
        settings = make_settings(_field(payload, "variants_tried"), _field(payload, "benchmark_return_pct"),
                                 _field(payload, "fees_included"), _field(payload, "source"),
                                 _field(payload, "unit"), _field(payload, "kind"), include_chart)
        try:
            result = analyze(read, settings, backtest=backtest)
        except Exception:  # never a bare 500 for a trader's file
            logger.exception("judge failed")
            return _error("the judge failed on this file", "This is our bug, not your file. Try a CSV export of "
                                                           "the closed trades.", 422)
        if source_kind == "upload" and result.get("status") == "ok":
            for card, st in zip(result["strategies"], read.strategies):
                try:
                    card["prediction"] = book.observe(st, source="upload")
                except Exception:
                    logger.exception("ledger observe failed")
                    card["prediction"] = None
        else:
            for card in result.get("strategies", []):
                card["prediction"] = None
        result["markdown"] = to_markdown(result)
        result["sample"] = sample_id
        result["track_record"] = book.track_record()
        result["privacy"] = PRIVACY
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return _error("too large", f"Files over {MAX_UPLOAD // (1024 * 1024)} MB are not read. Export a shorter "
                                   "period, or only the closed trades.", 413)

    @app.errorhandler(404)
    def not_found(_):
        return _error("not found", "See / for the page and /api/samples for the API.", 404)

    @app.errorhandler(405)
    def wrong_method(_):
        return _error("method not allowed", "POST to /api/judge; GET everything else.", 405)

    return app


def _display_name(filename: str) -> str:
    import re
    stem = Path(filename).stem if filename else "Your strategy"
    return re.sub(r"[_\-]+", " ", stem).strip()[:60] or "Your strategy"


def serve(host: str = "127.0.0.1", port: Optional[int] = None, threads: int = 8) -> None:
    """Run the app with waitress (never the Flask dev server)."""
    from waitress import serve as waitress_serve
    port = int(port or os.environ.get("PORT") or 5078)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    logger.info("edge or luck on http://%s:%d (ledger: %s)", host, port, app.config["EDGEORLUCK_LEDGER"].root)
    # waitress buffers request bodies over 512 KB (and responses over 1 MB) in temporary files; keep both in
    # memory so an upload never touches the disk.
    waitress_serve(app, host=host, port=port, threads=threads, max_request_body_size=MAX_UPLOAD + 1024 * 1024,
                   inbuf_overflow=MAX_UPLOAD + 2 * 1024 * 1024, outbuf_overflow=16 * 1024 * 1024)
