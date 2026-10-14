"""The hosted site as one ASGI app: the web app at /, and the MCP server at /mcp for assistants that connect to a
URL (ChatGPT, Claude and others):

    uvicorn --factory edgeorluck.web.remote:make_app --host 0.0.0.0 --port $PORT --proxy-headers

The tools are the local ones without files on anyone's computer (web/tools.py): versions tried are counted per
study id the answers hand out, and a frozen rule comes back as a signed receipt. EDGEORLUCK_HOSTS (comma-separated
host names) turns on the MCP transport's host check; unset, any host is served. Needs 'edgeorluck[host]'.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, List, Optional, Union

from a2wsgi import WSGIMiddleware
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Mount

from .. import mcp_server as local
from . import tools
from .app import RateLimiter, create_app
from .usage import Usage

CALLS_PER_MINUTE = 20
BUSY = f"Too many calls from here: up to {CALLS_PER_MINUTE} a minute. Wait a minute, then call again."
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
    "backtest_file_path / backtest_csv_text / backtest_pnl", "backtest_csv_text / backtest_pnl",
    "- detail: 'markdown' (default) or 'full' (the JSON result too: rarely needed, it repeats the Markdown).\n", "")
BACKTEST_DESCRIPTION = _swap(
    local.BACKTEST_DESCRIPTION,
    " or read a price file", "",
    "start/end dates, or prices_file (a CSV on this machine with a close column).", "start/end dates.",
    "variants_tried: versions tried before this session; the server counts the rules tested in it (the bar rises "
    "with them).", "variants_tried: versions tried outside this study. " + STUDY,
    "freeze=true keeps the rule for forward_test, which judges only its trades after now.",
    "freeze=true also returns a signed receipt: give it to the user, and forward_test(receipt) later judges only the "
    "trades the rule makes from now on.")
FORWARD_DESCRIPTION = (
    "Judge a frozen rule on the trades it made after it was frozen (backtest_rule with freeze=true gave a receipt), "
    "on fresh prices. None of them could have shaped the rule, so this is the honest test after many tries.\n\n"
    "CALL THIS when the user comes back with a receipt or asks how a frozen rule is doing. receipt: exactly as given. "
    "variants_tried: how many rules the user froze in all (ask; the bar rises with them). Returns Markdown: quote the "
    "verdict's headline. Not financial advice.")
TUNE_DESCRIPTION = _swap(local.TUNE_DESCRIPTION, "Don't tune again", STUDY + " Don't tune again")


def _client_name(ctx: Context) -> str:
    """The name the assistant's MCP client gives (claude-ai, openai-mcp, cursor ...), else its user agent."""
    params = getattr(ctx.session, "client_params", None)
    info = getattr(params, "client_info", None) or getattr(params, "clientInfo", None)
    return str(getattr(info, "name", "") or (ctx.headers or {}).get("user-agent", ""))


def build(usage: Optional[Usage] = None) -> MCPServer:
    """The MCP tools for the web: tools.py behind a rate limit, counted per study (or per session id)."""
    calls = RateLimiter(CALLS_PER_MINUTE, 60.0)

    @contextlib.contextmanager
    def caller(ctx: Context, study: Optional[str]) -> Iterator[Optional[str]]:
        """The study to count under, or None when the caller is over the rate limit."""
        host = getattr(getattr(ctx.request_context.request, "client", None), "host", None) or "?"
        allowed = calls.allow(host)
        if allowed and usage is not None:
            usage.count("mcp", _client_name(ctx) or "mcp")
        yield (study or (ctx.headers or {}).get("mcp-session-id", "")[:40] or "") if allowed else None

    def judge_trades(csv_text: Optional[str] = None, pnl: Optional[List[float]] = None,
                     variants_tried: Optional[int] = None, benchmark_return_pct: Optional[float] = None,
                     fees_included: Optional[bool] = None, source: Optional[str] = None, unit: Optional[str] = None,
                     kind: Optional[str] = None, backtest_csv_text: Optional[str] = None,
                     backtest_pnl: Optional[List[float]] = None, study: Optional[str] = None,
                     ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as key:
            return BUSY if key is None else tools.as_text(tools.judge(
                csv_text=csv_text, pnl=pnl, variants_tried=variants_tried, benchmark_return_pct=benchmark_return_pct,
                fees_included=fees_included, source=source, unit=unit, kind=kind,
                backtest_csv_text=backtest_csv_text, backtest_pnl=backtest_pnl, study=key))

    def backtest_rule(buy: str, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
                      start: Optional[str] = None, end: Optional[str] = None, stop_loss_pct: Optional[float] = None,
                      take_profit_pct: Optional[float] = None, max_hold_bars: Optional[int] = None,
                      side: str = "long", fee_pct: float = 0.1, slippage_pct: float = 0.05,
                      variants_tried: Optional[int] = None, freeze: bool = False, study: Optional[str] = None,
                      ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as key:
            return BUSY if key is None else tools.as_text(tools.backtest(
                buy=buy, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
                stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, max_hold_bars=max_hold_bars, side=side,
                fee_pct=fee_pct, slippage_pct=slippage_pct, variants_tried=variants_tried, freeze=freeze, study=key))

    def forward_test(receipt: str, variants_tried: Optional[int] = None,
                     ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, None) as key:
            return BUSY if key is None else tools.as_text(tools.forward(receipt, variants_tried))

    def tune_rule(buy: str, grid: Union[dict, str], sell: Optional[str] = None, symbol: str = "BTCUSDT",
                  interval: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                  stop_loss_pct: Optional[Union[float, str]] = None,
                  take_profit_pct: Optional[Union[float, str]] = None,
                  max_hold_bars: Optional[Union[int, str]] = None, side: str = "long", holdout: float = 0.3,
                  fee_pct: float = 0.1, slippage_pct: float = 0.05, study: Optional[str] = None,
                  ctx: Context = None) -> str:  # type: ignore[assignment]
        with caller(ctx, study) as key:
            return BUSY if key is None else tools.as_text(tools.tune(
                buy=buy, grid=grid, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
                stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, max_hold_bars=max_hold_bars, side=side,
                holdout=holdout, fee_pct=fee_pct, slippage_pct=slippage_pct, study=key))

    server = MCPServer("edgeorluck", instructions=local.INSTRUCTIONS + " Pass back the study id each answer names.",
                       website_url=os.environ.get("EDGEORLUCK_URL"))
    server.tool(name="judge_trades", description=JUDGE_DESCRIPTION)(judge_trades)
    server.tool(name="backtest_rule", description=BACKTEST_DESCRIPTION)(backtest_rule)
    server.tool(name="forward_test", description=FORWARD_DESCRIPTION)(forward_test)
    server.tool(name="tune_rule", description=TUNE_DESCRIPTION)(tune_rule)
    server.tool(name="rule_language", description="The rule language for backtest_rule and tune_rule: the price names, "
                                                  "every function, and examples.")(local.rule_language)
    return server


def make_app() -> Any:
    """The web app and the MCP server, in one ASGI app (data in EDGEORLUCK_DATA_DIR)."""
    hosts = [h.strip() for h in os.environ.get("EDGEORLUCK_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(hosts),
                                         allowed_hosts=hosts + [f"{h}:*" for h in hosts],
                                         allowed_origins=[f"https://{h}" for h in hosts])
    site = create_app()
    app = build(site.config["EDGEORLUCK_USAGE"]).streamable_http_app(transport_security=security, host="0.0.0.0")
    app.router.routes.append(Mount("/", app=WSGIMiddleware(site)))
    return app


__all__ = ["build", "make_app"]
