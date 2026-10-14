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
GET  /story                 Flip's 45-second animation: luck, the bar, and a record that clears it
GET  /play                  Flip Flip: call coin flips and watch luck even out inside the funnel
GET  /guides                the mini guides; /guides/<slug> one guide, /guides/<slug>.md its Markdown
GET  /api/guides            [{slug, title, summary, minutes, group, url, markdown_url}]
GET  /how                   the page where Flip shows how it checks and where it can be wrong
GET  /strategies            popular strategies, judged (past prices, every try counted); /strategies/<slug> one,
                            add .md for Markdown; GET /api/strategies the numbers
GET  /api/usage             how much it is used: checks per day, by way in and kind of assistant (counts only)
GET  /openapi.json          the assistant tools as an OpenAPI description (for GPT actions and the like)
POST /api/tools/<tool>      judge | backtest | tune | forward: the MCP tools over plain HTTP, the same fields,
                            answering {markdown, study, receipt?}; GET /api/tools/rules the rule language
GET  /api/claims            that page's claims: proved (Lean), measured, our choice, can't check
GET  /proofs/<file>         the Lean proofs as plain text (when running from the repository)
GET  /edgeorluck_single.py  the judge in one file, for chatbots without installs
GET  /api/judge             for assistants that can only fetch links:
                            ?pnl=1.2,-0.8,0.5 | ?csv=<url-encoded CSV> | ?sample=<id>,
                            plus the fields below; Markdown, or format=json
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

import html
import inspect
import io
import json
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from flask import Flask, Request, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from .. import samples
from ..judge import CHANGELOG, ENGINE_VERSION, analyze, make_settings
from ..mcp_server import rule_language
from ..readers import MAX_STRATEGIES, MAX_TRADES, ReadResult, read_any
from ..report import to_markdown
from ..schema import schema
from . import guides, openapi, strategies, tools
from .ledger import Ledger
from .tools import TOOLS
from .usage import Usage

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
PACKAGE = Path(__file__).resolve().parents[1]
PROOFS = PACKAGE.parents[1] / "proofs"          # the Lean proofs, present when running from the repository
MAX_UPLOAD = 10 * 1024 * 1024
RATE_LIMIT = 30                 # judge calls per address per window
RATE_WINDOW_S = 60.0

PRIVACY = ("Your file is read in memory and never stored. For each strategy with 30 or more trades we keep only: a "
           "one-way fingerprint of its first 30 trades and of the whole list, the number of trades, the unit, our "
           "forecast for its next 30 trades, and the time. If you come back with 30 more trades, we also keep "
           "whether those trades ended up or down, to grade the forecast. Each check also adds one to a daily count "
           "(the way it came in and the kind of assistant, if any: public at /api/usage). No trades, no file, no IP "
           "address (the rate limiter holds addresses in memory for one minute).")


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
        self.swept = time.monotonic()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            # Forget every address idle for a full window, at least once a window: the privacy note promises
            # addresses are held for a minute, not until 10,000 have piled up.
            if now - self.swept > self.window:
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
                self.swept = now
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
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
    app.config["MAX_FORM_MEMORY_SIZE"] = MAX_UPLOAD     # a pasted csv_text form field may be as big as a file
    proxies = int(os.environ.get("EDGEORLUCK_PROXIES", "0") or 0)
    if proxies > 0:                                     # behind N trusted proxies: their view of the visitor
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxies)  # type: ignore[method-assign]
    app.json.sort_keys = False  # type: ignore[attr-defined]
    book = ledger if ledger is not None else Ledger()
    limiter = RateLimiter(rate_limit)
    app.config["EDGEORLUCK_LEDGER"] = book
    usage = app.config["EDGEORLUCK_USAGE"] = Usage(book.root)

    def _counted(way: str) -> None:
        usage.count(way, request.headers.get("User-Agent", ""))

    @app.get("/api/usage")
    def api_usage():
        return jsonify(usage.summary())

    @app.get("/privacy")
    def privacy_page():
        e = html.escape
        kept = [PRIVACY, "Through the assistant tools (/mcp and /api/tools): the trades and rules you send are used for "
                "the answer and dropped. A frozen rule is not kept by us either: its receipt, signed by the server, "
                "holds the rule and the time, and you keep it.", "Prices come from public sources (Binance, Yahoo "
                "Finance, the European Central Bank) and are fetched by the server, not linked to you.",
                "No accounts, no cookies for tracking, no ads, no selling of anything. The code that does all this is "
                "open: github.com/Sakke-code/edge-or-luck."]
        body = ('<article class="wrap guide"><header class="guide-head"><div><h1>Privacy</h1><p class="lede">What edge '
                'or luck keeps, and what it doesn\'t.</p></div></header><div class="prose">'
                + "".join(f"<p>{e(p)}</p>" for p in kept) + "</div></article>")
        return Response(guides._shell("Privacy · edge or luck", "What edge or luck keeps, and what it doesn't.", body,
                                      current="/privacy"), mimetype="text/html")
    @app.get("/strategies")
    @app.get("/strategies.md", defaults={"as_md": True})
    def strategies_hub(as_md: bool = False):
        data = strategies.load()
        if data is None:
            return _error("not computed yet", "Run tools/write_strategies.py.", 404)
        return Response(strategies.markdown(data), mimetype="text/markdown") if as_md else \
            Response(strategies.hub_page(data), mimetype="text/html")

    @app.get("/strategies/<slug>.md", defaults={"as_md": True})
    @app.get("/strategies/<slug>")
    def strategy(slug: str, as_md: bool = False):
        data = strategies.load()
        found = strategies.find(data, slug) if data else None
        if found is None:
            return _error("no such strategy", "They are listed at /strategies.", 404)
        return Response(strategies.markdown(data, found), mimetype="text/markdown") if as_md else \
            Response(strategies.strategy_page(data, found), mimetype="text/html")

    @app.get("/api/strategies")
    def api_strategies():
        data = strategies.load()
        return jsonify(data) if data else _error("not computed yet", "Run tools/write_strategies.py.", 404)

    @app.get("/openapi.json")
    def openapi_spec():
        return jsonify(openapi.spec(request.host_url.rstrip("/")))

    @app.get("/api/tools/rules")
    def api_tools_rules():
        return jsonify({"markdown": rule_language()})

    @app.post("/api/tools/<name>")
    def api_tools(name: str):
        fn = TOOLS.get(name)
        if fn is None:
            return _error("no such tool", f"The tools are {', '.join(TOOLS)} and rules; see /openapi.json.", 404)
        if not limiter.allow(_visitor()):
            return _error("too many requests", f"Up to {limiter.limit} calls a minute per visitor. Wait a minute "
                                               "and try again.", 429)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("send a JSON object", "Its fields are listed in /openapi.json.", 400)
        unknown = sorted(set(payload) - set(inspect.signature(fn).parameters))
        if unknown:
            return _error(f"unknown field(s) {', '.join(unknown)}", "The fields are listed in /openapi.json.", 400)
        try:
            answer = fn(**payload)
        except TypeError as exc:                      # a required field left out
            return _error("a required field is missing", str(exc), 400)
        _counted("tools")
        return jsonify(answer)

    @app.get("/", defaults={"name": "index"})
    @app.get("/<any(how, story, play):name>")
    def page(name: str):
        return send_from_directory(STATIC, f"{name}.html")

    @app.get("/assets/<path:name>")
    def assets(name: str):
        return send_from_directory(STATIC / "assets", name)

    @app.get("/guides")
    def guides_hub():
        return Response(guides.hub_page(guides.load()), mimetype="text/html")

    @app.get("/guides/<slug>.md")
    def guide_markdown(slug: str):
        guide = guides.find(slug)
        if guide is None:
            return _error("no such guide", "The guides are listed at /api/guides.", 404)
        return Response(guide.full_markdown(), mimetype="text/markdown")

    @app.get("/guides/<slug>")
    def guide_page(slug: str):
        every = guides.load()
        guide = next((g for g in every if g.slug == slug), None)
        if guide is None:
            return _error("no such guide", "The guides are listed at /guides.", 404)
        return Response(guides.guide_page(guide, every), mimetype="text/html")

    @app.get("/api/guides")
    def api_guides():
        return jsonify([g.as_dict() for g in guides.load()])

    @app.get("/api/claims")
    def api_claims():
        return jsonify(json.loads((PACKAGE / "data" / "claims.json").read_text(encoding="utf-8")))

    @app.get("/proofs/<path:name>")
    def proofs(name: str):
        """The Lean proofs, readable as plain text, when the site runs from the repository."""
        if not (name.endswith(".lean") or name.endswith(".md")) or not PROOFS.is_dir():
            return _error("not found", "The proofs are .lean files under /proofs/EdgeOrLuck/.", 404)
        return send_from_directory(PROOFS, name, mimetype="text/plain")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "engine": ENGINE_VERSION})

    @app.get("/llms.txt")
    def llms():
        path = PACKAGE / "llms.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else "edge or luck\n"
        return Response(text, mimetype="text/plain")

    @app.get("/api/schema")
    def api_schema():
        return jsonify(schema())

    @app.get("/api/samples")
    def api_samples():
        listed = samples.available()
        if not _on_this_computer():
            listed = [s for s in listed if s.get("group") != "private"]
        return jsonify(listed)

    @app.get("/api/engine")
    def api_engine():
        return jsonify({"name": "edge or luck", "version": ENGINE_VERSION, "changelog": CHANGELOG,
                        "tip_url": os.environ.get("EDGEORLUCK_TIP_URL") or None})

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
        return Response(_single_cache[ENGINE_VERSION], mimetype="text/x-python",
                        headers={"Content-Disposition": "attachment; filename=edgeorluck_single.py"})

    def _on_this_computer() -> bool:
        """Private samples (EDGEORLUCK_PRIVATE_SAMPLES: your own records) are shown only to a browser on the
        same computer, never to visitors of a hosted site."""
        return (request.remote_addr or "") in ("127.0.0.1", "::1")

    def _sample(sample_id: str) -> Dict[str, Any]:
        if str(sample_id).startswith("private:") and not _on_this_computer():
            raise KeyError(sample_id)
        return samples.sample_input(str(sample_id))

    def _visitor() -> str:
        # The connecting address. Behind a proxy, set EDGEORLUCK_PROXIES to the number of proxies in front of the
        # app: ProxyFix then takes the address the nearest trusted proxy saw. A raw X-Forwarded-For header is
        # whatever the visitor typed, so it is never used directly.
        return request.remote_addr or "?"

    def _bad_fields(fields: Dict[str, Any]) -> Optional[Tuple[Response, int]]:
        for key, limit in (("name", 80), ("kind", 20), ("unit", 10), ("source", 20)):
            value = fields.get(key)
            if value is not None and not isinstance(value, str):
                return _error(f"{key} must be text", f"Send {key} as a short string.", 400)
            if isinstance(value, str) and len(value) > limit:
                return _error(f"{key} is too long", f"{key} is at most {limit} characters.", 400)
        return None

    @app.post("/api/judge")
    def api_judge():
        if not limiter.allow(_visitor()):
            return _error("too many requests", f"Up to {limiter.limit} checks a minute per visitor. Wait a minute "
                                               "and try again.", 429)
        try:
            _counted("page")
            return _judge_post()
        except HTTPException:     # 413 too large and the like: their own handlers answer
            raise
        except RecursionError:
            return _error("the request is nested too deeply", "Send a flat list of results or a CSV.", 400)
        except Exception:  # never a bare 500 for a trader's file
            logger.exception("judge request failed")
            return _error("the judge failed on this input", "This is our bug, not your file. Try a CSV export of "
                                                            "the closed trades.", 422)

    def _judge_post():
        payload: Dict[str, Any] = {}
        backtest = None
        source_kind = "upload"
        sample_id = None
        include_chart = True
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return _error("send a JSON object", "Expected {\"sample\": id} or {\"csv_text\": ...}.", 400)
            bad = _bad_fields(payload)
            if bad:
                return bad
            include_chart = payload.get("chart", True) is not False
            sample_id = payload.get("sample")
            if sample_id:
                try:
                    args = _sample(str(sample_id))
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
                                  name=_field(payload, "name"), allow_paths=False)]
                bt = next((payload[k] for k in ("backtest_csv_text", "backtest_pnl", "backtest") if
                           payload.get(k) is not None), None)
                if bt is not None:
                    backtest = read_any(bt, name="Backtest", allow_paths=False)
        else:
            payload = {k: v for k, v in request.form.items()}
            bad = _bad_fields(payload)
            if bad:
                return bad
            files = [(f.filename or "trades.csv", f.read()) for f in request.files.getlist("files")]
            files += [(f.filename or "trades.csv", f.read()) for f in request.files.getlist("file")]
            if not files and payload.get("csv_text"):
                files = [("pasted", payload["csv_text"].encode("utf-8"))]
            if not files:
                return _error("no file", "Send one or more trade exports as 'files', or a sample.", 400)
            reads = [read_any(raw, unit=_field(payload, "unit"), kind=_field(payload, "kind"),
                              name=_display_name(fname), allow_paths=False) for fname, raw in files]
            bt_files = [(f.filename or "backtest.csv", f.read()) for f in request.files.getlist("backtest")]
            if bt_files:
                backtest = _merge([read_any(raw, name="Backtest", allow_paths=False) for _, raw in bt_files])
        result = _run(_merge(reads), payload, backtest, source_kind, sample_id, include_chart)
        return result if isinstance(result, tuple) else jsonify(result)

    @app.get("/api/judge")
    def api_judge_get():
        """The same judge from a plain link, for assistants that can only fetch
        URLs: ?pnl=1.2,-0.8,0.5 (or separated by ; or spaces, which allows
        decimal commas) or ?csv=<url-encoded CSV> or ?sample=<id>, plus the
        usual fields. Markdown by default; format=json for the JSON result."""
        if not limiter.allow(_visitor()):
            return _error("too many requests", f"Up to {limiter.limit} checks a minute per visitor. Wait a minute "
                                               "and try again.", 429)
        try:
            _counted("link")
            return _judge_get()
        except HTTPException:
            raise
        except Exception:
            logger.exception("judge request failed")
            return _error("the judge failed on this input", "This is our bug, not your data.", 422)

    def _judge_get():
        args = {k: v for k, v in request.args.items()}
        bad = _bad_fields(args)
        if bad:
            return bad
        backtest = None
        source_kind, sample_id = "upload", None
        if _field(args, "sample"):
            sample_id = args["sample"]
            try:
                given = _sample(str(sample_id))
            except KeyError:
                return _error(f"no sample called {sample_id!r}", "See GET /api/samples for the ids.", 404)
            source_kind = "sample"
            reads = [read_any(given["data"], name=given.get("name"))]
            if given.get("backtest") is not None:
                backtest = read_any(given["backtest"], name="Backtest")
            args.setdefault("source", given.get("source"))
        elif _field(args, "csv"):
            reads = [read_any(args["csv"], kind=_field(args, "kind"), unit=_field(args, "unit"),
                              name=_field(args, "name"), allow_paths=False)]
        elif _field(args, "pnl"):
            reads = [read_any(_pnl_lines(args["pnl"]), unit=_field(args, "unit"), name=_field(args, "name"),
                              allow_paths=False)]
        else:
            return _error("nothing to judge", "Add ?pnl=1.2,-0.8,0.5 (one result per trade), ?csv=<url-encoded "
                                              "CSV> or ?sample=<id>. See /llms.txt.", 400)
        result = _run(_merge(reads), args, backtest, source_kind, sample_id, include_chart=False)
        if isinstance(result, tuple):
            return result
        if (args.get("format") or "md").lower() == "json":
            return jsonify(result)
        return Response(result["markdown"], mimetype="text/markdown")

    def _run(read: ReadResult, payload: Dict[str, Any], backtest: Optional[ReadResult], source_kind: str,
             sample_id: Optional[str], include_chart: bool):
        """Judge, let the ledger make its forecast for real uploads, and add what
        every answer carries: the Markdown, the track record, the privacy note."""
        # The caps hold per check, after several files are merged (each file is capped on its own when read).
        for part in (read, backtest):
            if part is not None and len(part.strategies) > MAX_STRATEGIES:
                part.strategies = part.strategies[:MAX_STRATEGIES]
                part.warnings.append(f"Only the first {MAX_STRATEGIES} strategies were judged.")
            if part is not None and sum(len(st.pnl) for st in part.strategies) > MAX_TRADES:
                return _error("too many trades", f"Up to {MAX_TRADES:,} trades per check. Export a shorter period.",
                              413)
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
        return result

    for code, error, why in (
            (500, "something went wrong on our side", "This is our bug. Try again, or try a CSV export of the closed "
                                                      "trades."),
            (413, "too large", f"Files over {MAX_UPLOAD // (1024 * 1024)} MB are not read. Export a shorter period, or "
                               "only the closed trades."),
            (404, "not found", "See / for the page and /api/samples for the API."),
            (405, "method not allowed", "GET or POST /api/judge; GET everything else.")):
        app.register_error_handler(code, lambda _, c=code, e=error, w=why: _error(e, w, c))
    return app


def _pnl_lines(text: str) -> str:
    """One result per line. Commas separate results unless the text also has
    semicolons, spaces or line breaks, in which case commas may be decimal
    commas ("1,20; -0,80")."""
    text = text.strip()
    parts = re.split(r"[;\s]+", text) if re.search(r"[;\s]", text) else text.split(",")
    return "\n".join(p for p in parts if p)


def _display_name(filename: str) -> str:
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
