"""The hosted site as one ASGI app: the web app at /, and the MCP server at /mcp for assistants that connect to a
URL (ChatGPT, Claude and others):

    uvicorn --factory edgeorluck.web.remote:make_app --host 0.0.0.0 --port $PORT --proxy-headers

Over the web the tools never read files on anyone's computer: a record comes as csv_text or pnl, prices from
Binance. The protocol keeps no session, so every answer names a study and asks the assistant to pass it back:
versions tried are counted per study (or per session, for clients that keep one). freeze=true freezes a rule in
the public league. EDGEORLUCK_HOSTS (comma-separated host names) turns on the MCP transport's host check; unset,
any host is served. Needs pip install 'edgeorluck[host]'.
"""
from __future__ import annotations

import contextlib
import os
import re
import secrets
from typing import Any, Iterator, List, Optional, Tuple, Union

from a2wsgi import WSGIMiddleware
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Mount

from .. import mcp_server as local
from ..market import PriceError
from ..rulebook import RuleError
from .app import FREEZES_PER_HOUR, RateLimiter, create_app
from .league import League

CALLS_PER_MINUTE = 20
BUSY = f"Too many calls from here: up to {CALLS_PER_MINUTE} a minute. Wait a minute, then call again."
_STUDY = re.compile(r"^[A-Za-z0-9_-]{6,40}$")
STUDY = ("study: the id an earlier answer in this conversation gave. Pass it every time: the server can't see the "
         "conversation, and it counts every version tried under one study.")


def _swap(text: str, *pairs: str) -> str:
    """The local tool's description with the parts about local files rewritten (each must be there)."""
    for old, new in zip(pairs[::2], pairs[1::2]):
        if old not in text:
            raise ValueError(f"the description changed: {old[:40]!r} is gone")
        text = text.replace(old, new)
    return text


JUDGE_DESCRIPTION = _swap(
    local.JUDGE_DESCRIPTION,
    "file_path (an export on this machine: CSV/TSV/XLSX/freqtrade JSON; TradingView 'List of trades' and raw exchange "
    "fills are recognised), csv_text (the file's text)",
    "csv_text (an export's text: TradingView 'List of trades', exchange fills, or any CSV with a P&L column)",
    "The server counts the ones judged in this session itself.", STUDY,
    "backtest_file_path / backtest_csv_text / backtest_pnl", "backtest_csv_text / backtest_pnl")
BACKTEST_DESCRIPTION = _swap(
    local.BACKTEST_DESCRIPTION,
    " or read a price file", "",
    "start/end dates, or prices_file (a CSV on this machine with a close column).", "start/end dates.",
    "variants_tried: versions tried before this session; the server counts the rules tested in it (the bar rises "
    "with them).", "variants_tried: versions tried outside this study. " + STUDY,
    "freeze=true keeps the rule for forward_test, which judges only its trades after now.",
    "freeze=true freezes it in the public league: from now on only its new trades count, anyone can check them at "
    "its page, and forward_test judges them. Give the user the owner_token a first freeze returns, and pass it with "
    "their later freezes (with a title in name, if they like).")
FORWARD_DESCRIPTION = _swap(local.FORWARD_DESCRIPTION, "(a wrong id lists the frozen ones)",
                            "(every frozen rule is listed at /league)")
TUNE_DESCRIPTION = _swap(local.TUNE_DESCRIPTION, "Don't tune again", STUDY + " Don't tune again")


def _studied(md: str, study: str) -> str:
    return md + f"\n_Study `{study}`: pass study=\"{study}\" in every later call about this idea, so each version " \
                "tried counts._\n"


def build(league: League) -> MCPServer:
    """The MCP tools for the web: the local ones without files, counted per study, freezing into the league."""
    calls, freezes = RateLimiter(CALLS_PER_MINUTE, 60.0), RateLimiter(FREEZES_PER_HOUR, 3600.0)

    @contextlib.contextmanager
    def caller(ctx: Context, study: Optional[str] = None) -> Iterator[Optional[Tuple[str, str]]]:
        """(address, study) while the study counts the versions tried; None when over the rate limit."""
        host = getattr(getattr(ctx.request_context.request, "client", None), "host", None) or "?"
        key = study if study and _STUDY.match(study) else (ctx.headers or {}).get("mcp-session-id", "")[:40]
        key = key or secrets.token_hex(6)
        token = local._CURRENT.set(key)
        try:
            yield (host, key) if calls.allow(host) else None
        finally:
            local._CURRENT.reset(token)

    def judge_trades(csv_text: Optional[str] = None, pnl: Optional[List[float]] = None,
                     variants_tried: Optional[int] = None, benchmark_return_pct: Optional[float] = None,
                     fees_included: Optional[bool] = None, source: Optional[str] = None, unit: Optional[str] = None,
                     kind: Optional[str] = None, backtest_csv_text: Optional[str] = None,
                     backtest_pnl: Optional[List[float]] = None, detail: str = "markdown", study: Optional[str] = None,
                     ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as who:
            return BUSY if who is None else _studied(local.judge_trades(
                csv_text=csv_text, pnl=pnl, variants_tried=variants_tried, benchmark_return_pct=benchmark_return_pct,
                fees_included=fees_included, source=source, unit=unit, kind=kind,
                backtest_csv_text=backtest_csv_text, backtest_pnl=backtest_pnl, detail=detail), who[1])

    def backtest_rule(buy: str, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
                      start: Optional[str] = None, end: Optional[str] = None, stop_loss_pct: Optional[float] = None,
                      take_profit_pct: Optional[float] = None, max_hold_bars: Optional[int] = None,
                      side: str = "long", fee_pct: float = 0.1, slippage_pct: float = 0.05,
                      variants_tried: Optional[int] = None, freeze: bool = False, name: Optional[str] = None,
                      owner_token: Optional[str] = None, study: Optional[str] = None,
                      ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as who:
            if who is None:
                return BUSY
            md = local.backtest_rule(buy=buy, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
                                     stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                                     max_hold_bars=max_hold_bars, side=side, fee_pct=fee_pct,
                                     slippage_pct=slippage_pct, variants_tried=variants_tried)
            if not freeze or md.startswith("Couldn't"):
                return _studied(md, who[1])
            if not freezes.allow(who[0]):
                return _studied(md + f"\n**Not frozen:** up to {FREEZES_PER_HOUR} freezes an hour from here.\n", who[1])
            try:
                receipt = league.freeze(local._rule_dict(buy, sell, stop_loss_pct, take_profit_pct, max_hold_bars,
                                                         side), symbol, interval, fee_pct=fee_pct,
                                        slippage_pct=slippage_pct, name=name, owner_token=owner_token)
            except (RuleError, PriceError) as exc:
                return _studied(md + f"\n**Not frozen:** {exc}\n", who[1])
            token = receipt.get("owner_token")
            return _studied(md + f"\n**{receipt['sentence']}** Its id for forward_test: `{receipt['id']}`.\n" + (
                f"\nowner_token: `{token}`. Give it to the user to keep: it is shown once.\n" if token else ""), who[1])

    def forward_test(frozen_id: str, ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx) as who:
            return BUSY if who is None else league.markdown(str(frozen_id).strip().strip("`\""))

    def tune_rule(buy: str, grid: Union[dict, str], sell: Optional[str] = None, symbol: str = "BTCUSDT",
                  interval: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                  stop_loss_pct: Optional[Union[float, str]] = None,
                  take_profit_pct: Optional[Union[float, str]] = None,
                  max_hold_bars: Optional[Union[int, str]] = None, side: str = "long", holdout: float = 0.3,
                  fee_pct: float = 0.1, slippage_pct: float = 0.05, study: Optional[str] = None,
                  ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as who:
            return BUSY if who is None else _studied(local.tune_rule(
                buy=buy, grid=grid, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
                stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, max_hold_bars=max_hold_bars, side=side,
                holdout=holdout, fee_pct=fee_pct, slippage_pct=slippage_pct), who[1])

    server = MCPServer("edgeorluck", instructions=local.INSTRUCTIONS + " Pass back the study id each answer names.",
                       website_url=os.environ.get("EDGEORLUCK_URL"))
    server.tool(name="judge_trades", description=JUDGE_DESCRIPTION)(judge_trades)
    server.tool(name="backtest_rule", description=BACKTEST_DESCRIPTION)(backtest_rule)
    server.tool(name="forward_test", description=FORWARD_DESCRIPTION)(forward_test)
    server.tool(name="tune_rule", description=TUNE_DESCRIPTION)(tune_rule)
    server.tool(name="rule_language", description="The rule language for backtest_rule and tune_rule: the price names, "
                                                  "every function, and examples.")(local.rule_language)
    return server


def make_app(league: Optional[League] = None) -> Any:
    """The web app and the MCP server, in one ASGI app (data in EDGEORLUCK_DATA_DIR)."""
    site = create_app(league=league)
    hosts = [h.strip() for h in os.environ.get("EDGEORLUCK_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(hosts),
                                         allowed_hosts=hosts + [f"{h}:*" for h in hosts],
                                         allowed_origins=[f"https://{h}" for h in hosts])
    app = build(site.config["EDGEORLUCK_LEAGUE"]).streamable_http_app(transport_security=security, host="0.0.0.0")
    app.router.routes.append(Mount("/", app=WSGIMiddleware(site)))
    return app


__all__ = ["build", "make_app"]
