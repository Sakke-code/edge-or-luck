"""The assistant tools (web/tools.py) as an OpenAPI 3.1 description, for GPT actions and any other caller that
reads one. The fields come from the tools' own signatures, so the description can't drift from the code."""
from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Dict

from ..judge import ENGINE_VERSION
from . import tools

FIELDS = {
    "csv_text": "The trade export's text (TradingView 'List of trades', exchange fills, any CSV with a P&L column).",
    "pnl": "Or the result of each closed trade, oldest first.",
    "variants_tried": "How many versions or settings were tried in all. Ask the user; the bar rises with it.",
    "benchmark_return_pct": "What simply holding returned over the same dates, in percent.",
    "fees_included": "Whether the results already include fees.",
    "source": "backtest, paper or live.",
    "unit": "% or a currency like USDT, when the export doesn't say.",
    "kind": "auto, trades, fills, tradingview or pnl.",
    "backtest_csv_text": "The backtest's trades, to check live trades against it.",
    "backtest_pnl": "Or the backtest's results per trade.",
    "study": "The study id an earlier answer gave. Pass it every time, so each version tried counts.",
    "buy": "When to enter, in the rule language (see ruleLanguage), e.g. rsi(14) < 30 and close > sma(200).",
    "sell": "When to get out, in the rule language.",
    "symbol": "A stock, ETF or index as Yahoo Finance writes it (AAPL, SPY, ^GSPC), two currencies (EURUSD: ECB daily "
              "rates), or a Binance spot pair (BTCUSDT).",
    "interval": "Bar size: 1h, 1d or 1w for stocks; 1d or 1w for currencies; 15m, 1h, 4h, 1d and more for crypto.",
    "start": "First date, like 2024-01-01 (default: two years ago).",
    "end": "Last date (default: now).",
    "stop_loss_pct": "Get out at this loss, in percent.",
    "take_profit_pct": "Get out at this gain, in percent.",
    "max_hold_bars": "Get out after this many bars.",
    "side": "long or short.",
    "fee_pct": "Fee per fill, in percent (0.1 = 0.1%).",
    "slippage_pct": "Slippage per fill, in percent.",
    "freeze": "Also return a signed receipt: later, forwardTest judges only the trades after now.",
    "grid": "The values to try for each {placeholder}, like {\"low\": [20, 25, 30]}.",
    "holdout": "The share of the history kept unseen for the verdict (0.3 = the last 30%).",
    "receipt": "The receipt a freeze returned, exactly as given.",
}

OPERATIONS = {
    "judge": ("judgeTrades", "Judge whether a trading record is a real edge or luck. Call it whenever the user shows "
                             "trades or a backtest and asks if it is good. Quote the headline; ask its questions."),
    "backtest": ("backtestRule", "Test a trading idea honestly on stock, ETF, currency or crypto prices (fills at the "
                                 "next bar's open, with costs, no look-ahead) and judge it. Use it instead of writing "
                                 "your own backtest."),
    "tune": ("tuneRule", "Pick a rule's settings on the first part of the prices and judge the choice once on the "
                         "unseen last part. Use it instead of running many backtests and keeping the best."),
    "forward": ("forwardTest", "Judge a frozen rule only on the trades it made after it was frozen, from its "
                               "receipt: the honest test after many tries."),
}


def _schema(hint: Any) -> Dict[str, Any]:
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if typing.get_origin(hint) is typing.Union:
        return _schema(args[0]) if len(args) == 1 else {"oneOf": [_schema(a) for a in args]}
    if typing.get_origin(hint) in (list, typing.List):
        return {"type": "array", "items": _schema(args[0])}
    return {str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"}, bool: {"type": "boolean"},
            dict: {"type": "object"}}.get(typing.get_origin(hint) or hint, {"type": "object"})


def _body(fn: Callable[..., Any]) -> Dict[str, Any]:
    hints = typing.get_type_hints(fn)
    params = inspect.signature(fn).parameters
    props = {name: dict(_schema(hints[name]), description=FIELDS[name]) for name in params}
    required = [n for n, p in params.items() if p.default is inspect.Parameter.empty]
    return {"type": "object", "properties": props, **({"required": required} if required else {})}


ANSWER = {"type": "object", "properties": {
    "markdown": {"type": "string", "description": "The answer, ready to relay: quote its headline word for word."},
    "study": {"type": "string", "description": "Pass it back in later calls about this idea."},
    "receipt": {"type": "string", "description": "A frozen rule's receipt: give it to the user to keep."}}}


def spec(server: str) -> Dict[str, Any]:
    paths: Dict[str, Any] = {
        f"/api/tools/{name}": {"post": {
            "operationId": op, "summary": op, "description": text,
            "requestBody": {"required": True, "content": {"application/json": {"schema": _body(tools.TOOLS[name])}}},
            "responses": {"200": {"description": "The answer", "content": {"application/json": {"schema": ANSWER}}}}}}
        for name, (op, text) in OPERATIONS.items()}
    paths["/api/tools/rules"] = {"get": {
        "operationId": "ruleLanguage", "summary": "ruleLanguage",
        "description": "The rule language for backtestRule and tuneRule: price names, functions, examples.",
        "responses": {"200": {"description": "The rule language", "content": {"application/json": {"schema": ANSWER}}}}}}
    return {"openapi": "3.1.0",
            "info": {"title": "edge or luck", "version": ENGINE_VERSION,
                     "description": "Is a trading record a real edge, or luck? An honest judge and backtester for "
                                    "traders and their AI assistants. Not financial advice."},
            "servers": [{"url": server}], "paths": paths}


__all__ = ["spec"]
