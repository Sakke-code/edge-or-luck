"""The MCP server (stdio): lets an AI assistant judge a trader's results.

    pip install 'edgeorluck[mcp]'
    edgeorluck-mcp                         # or: python -m edgeorluck.mcp_server

It runs on the trader's own machine, so the assistant can pass the path of
an export it can see. Tools:

* judge_trades: the verdict, in Markdown followed by the full JSON result;
* result_schema: the JSON Schema of that result;
* backtest_rule: an honest backtest of a rule (fetched or local prices), judged;
* tune_rule: honest tuning, judged on a part the choice never saw;
* rule_language: the rule language's names, functions and examples.

Works with the official MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""
import json
import os
from typing import List, Optional

from .judge import ENGINE_VERSION
from .result import judge
from .schema import schema

INSTRUCTIONS = (
    "edge or luck judges whether a trading record is a real edge or luck, and tests trading ideas honestly. Call "
    "judge_trades whenever the user shows trading results, a trade list, a backtest or an exchange export and asks "
    "whether it is good, real, worth running, or why it lost. Call backtest_rule when the user has an idea or a rule "
    "but no trades, and tune_rule when they want the best settings, instead of writing your own backtest code. Then "
    "answer from the result: quote the verdict's headline, explain it in short words, give the next step, and ask "
    "the questions it lists. Don't compute your own t-test or Sharpe ratio on top. Not financial advice."
)

JUDGE_DESCRIPTION = (
    "Judge whether a trading strategy's results are a real edge or luck, honestly and reproducibly.\n\n"
    "CALL THIS whenever the user shows trading results, a list of trades, a backtest (TradingView strategy "
    "tester, backtesting.py, vectorbt, freqtrade, a bot's log) or an exchange trade export, and asks if it is "
    "good, real, profitable, worth running live, overfit, lucky, or why it lost money. Also call it to check "
    "live trades against a backtest (pass both).\n\n"
    "Give ONE of: file_path (an export on this machine: CSV/TSV/XLSX/freqtrade JSON; TradingView 'List of trades' "
    "and raw exchange fills are recognised), csv_text (the file's text), or pnl (results per trade, oldest first).\n"
    "Optional, and each one makes the verdict more honest:\n"
    "- variants_tried: how many versions or parameter settings were tried in all. The best of many looks good by "
    "luck, so this raises the bar. Ask the user if unknown.\n"
    "- benchmark_return_pct: what buying and holding returned over the same dates, in percent (34.5 = +34.5%). "
    "Catches a long-only strategy that only rode a rising market.\n"
    "- fees_included (true/false), source ('backtest', 'paper' or 'live'), unit ('%' or 'USDT' etc.), "
    "kind ('auto', 'trades', 'fills', 'tradingview', 'pnl').\n"
    "- backtest_file_path / backtest_csv_text / backtest_pnl: the backtest, to check the live trades against it.\n"
    "- detail: 'full' (default: Markdown, then the JSON result) or 'markdown' (shorter: the Markdown only).\n\n"
    "Returns Markdown (ready to relay) followed by the full JSON result. Answer the user from it: quote "
    "verdict.headline word for word, explain verdict.plain in short words, give next_step, and ask the questions "
    "in ask_the_user (each says which parameter to set when you call again). Never present 'cant_tell' as a win. "
    "Not financial advice."
)


def _path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value.strip().strip('"'))))


def judge_trades(file_path: Optional[str] = None, csv_text: Optional[str] = None,
                 pnl: Optional[List[float]] = None, variants_tried: Optional[int] = None,
                 benchmark_return_pct: Optional[float] = None, fees_included: Optional[bool] = None,
                 source: Optional[str] = None, unit: Optional[str] = None, kind: Optional[str] = None,
                 backtest_file_path: Optional[str] = None, backtest_csv_text: Optional[str] = None,
                 backtest_pnl: Optional[List[float]] = None, detail: str = "full") -> str:
    """Judge a trading record: Markdown, then the full JSON result (detail="markdown": the Markdown only)."""
    data = _path(file_path) or csv_text or pnl
    if data is None:
        return ("Nothing to judge: pass file_path (an export on this machine), csv_text (its text) or pnl (a list "
                "of results per trade, oldest first).")
    if isinstance(data, str) and file_path and not os.path.exists(data):
        return (f"No file at {data}. Check the path (an absolute path is safest), or pass the file's text as "
                "csv_text.")
    backtest = _path(backtest_file_path) or backtest_csv_text or backtest_pnl
    result = judge(data, variants_tried=variants_tried, benchmark_return_pct=benchmark_return_pct,
                   fees_included=fees_included, source=source, unit=unit, kind=kind, backtest=backtest)
    if str(detail).strip().lower().startswith("markdown"):
        return result.to_markdown()
    return result.to_markdown() + "\n\n## Full result (JSON)\n\n```json\n" + result.to_json(indent=None) + "\n```\n"


def result_schema() -> str:
    """The JSON Schema of judge_trades' result."""
    return json.dumps(schema(), indent=1)


BACKTEST_DESCRIPTION = (
    "Test a trading idea honestly and judge it, in one call: fetch prices (Binance spot, public data, no key) or read "
    "a price file, run the rule, and judge the trades.\n\n"
    "CALL THIS whenever the user has a trading idea or rule but no trades yet (\"buy when RSI is below 30\", \"does "
    "a moving-average cross work on ETH?\"), INSTEAD of writing your own backtest code. Hand-written backtests "
    "usually trade at the close of the bar the signal used, forget fees, and look better than reality; this one "
    "can't: decisions at a bar's close are filled at the next bar's open, with fees and slippage on every fill.\n\n"
    "buy / sell: conditions in the rule language (call rule_language for the full list), e.g. buy=\"rsi(14) < 30 "
    "and close > sma(200)\", sell=\"rsi(14) > 70\". Optional exits: stop_loss_pct, take_profit_pct, max_hold_bars. "
    "side=\"short\" sells first.\n"
    "Prices: symbol (Binance spot pair like BTCUSDT) + interval (1h, 4h, 1d, ...) + start/end dates, or "
    "prices_file (a CSV on this machine with a close column).\n"
    "variants_tried: how many rules or settings were tried in all so far in this conversation, this one included "
    "(the bar rises with it).\n\n"
    "Returns Markdown: the backtest, its honesty checks (look-ahead, one bar later, costs) and the verdict. Quote "
    "the verdict's headline. To compare settings, use tune_rule instead of calling this many times. Not financial "
    "advice."
)

TUNE_DESCRIPTION = (
    "Tune a rule's settings honestly: try every combination on the first part of the prices only, then judge the "
    "chosen settings once on the last part, which played no part in the choice.\n\n"
    "CALL THIS whenever the user wants the best settings for a rule (\"which RSI level works best?\"), INSTEAD of "
    "running many backtests and keeping the best: that finds luck. Use {placeholders} in buy/sell and give their "
    "values in grid, e.g. buy=\"rsi(14) < {low}\", sell=\"rsi(14) > {high}\", grid={\"low\": [20, 25, 30], "
    "\"high\": [65, 70, 75]}. stop_loss_pct / take_profit_pct / max_hold_bars may also be \"{name}\".\n"
    "holdout: the share of the history kept unseen (0.3 = the last 30%).\n\n"
    "Returns Markdown: the chosen settings, the verdict on the unseen part (the honest one: quote it), how the "
    "other settings did there, and checks. Don't tune again after seeing the unseen verdict. Not financial advice."
)


def _prices_for(symbol: Optional[str], interval: str, start: Optional[str], end: Optional[str],
                prices_file: Optional[str]):
    from .market import load_prices, prices
    if prices_file:
        path = _path(prices_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file at {path}.")
        return load_prices(path)
    return prices(symbol or "BTCUSDT", interval, start=start, end=end)


def _rule_dict(buy: str, sell: Optional[str], stop_loss_pct, take_profit_pct, max_hold_bars, side: str):
    return {k: v for k, v in {"buy": buy, "sell": sell, "stop_loss_pct": stop_loss_pct,
                              "take_profit_pct": take_profit_pct, "max_hold_bars": max_hold_bars,
                              "side": side}.items() if v not in (None, "")}


def backtest_rule(buy: str, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
                  start: Optional[str] = None, end: Optional[str] = None, prices_file: Optional[str] = None,
                  stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None,
                  max_hold_bars: Optional[int] = None, side: str = "long", fee_pct: float = 0.1,
                  slippage_pct: float = 0.05, variants_tried: Optional[int] = None) -> str:
    """An honest backtest of a rule, judged: Markdown."""
    from .market import PriceError
    from .rulebook import RuleError
    from .simulate import backtest
    try:
        px = _prices_for(symbol, interval, start, end, prices_file)
        result = backtest(px, _rule_dict(buy, sell, stop_loss_pct, take_profit_pct, max_hold_bars, side),
                          fee_pct=fee_pct, slippage_pct=slippage_pct, variants_tried=variants_tried)
    except (RuleError, PriceError, FileNotFoundError) as exc:
        return f"Couldn't run the backtest: {exc}"
    return result.to_markdown()


def tune_rule(buy: str, grid: dict, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
              start: Optional[str] = None, end: Optional[str] = None, prices_file: Optional[str] = None,
              stop_loss_pct: Optional[str] = None, take_profit_pct: Optional[str] = None,
              max_hold_bars: Optional[str] = None, side: str = "long", holdout: float = 0.3, fee_pct: float = 0.1,
              slippage_pct: float = 0.05) -> str:
    """Honest tuning with a held-back part, judged: Markdown."""
    from .market import PriceError
    from .rulebook import RuleError
    from .simulate import tune
    try:
        if isinstance(grid, str):
            grid = json.loads(grid)
        px = _prices_for(symbol, interval, start, end, prices_file)
        result = tune(px, _rule_dict(buy, sell, stop_loss_pct, take_profit_pct, max_hold_bars, side), grid,
                      holdout=holdout, fee_pct=fee_pct, slippage_pct=slippage_pct)
    except (RuleError, PriceError, FileNotFoundError, ValueError) as exc:
        return f"Couldn't tune: {exc}"
    return result.to_markdown()


def rule_language() -> str:
    """The rule language: names, functions and examples."""
    from .rulebook import RULE_FUNCTIONS, RULE_SERIES
    lines = ["# The rule language", "",
             "A rule is a condition checked at each bar's close; it can only look back, never ahead. Stops, targets "
             "and max_hold_bars are checked at each close too. Every decision is filled at the next bar's open, "
             "with fees and slippage.", "",
             f"Prices: {', '.join(RULE_SERIES)}. Numbers. Arithmetic: + - * /. Comparisons: < <= > >=. "
             "Combine with and, or, not. A function given one number works on the close (sma(20) = sma(close, 20)).",
             "", "## Functions", ""]
    lines += [f"- {text}" for text in RULE_FUNCTIONS.values()]
    lines += ["", "## Examples", "",
              '- buy "rsi(14) < 30", sell "rsi(14) > 70"',
              '- buy "crosses_above(ema(12), ema(26))", sell "crosses_below(ema(12), ema(26))"',
              '- buy "close > highest(high, 20) and volume > 2 * sma(volume, 20)", stop_loss_pct 5, max_hold_bars 48',
              '- buy "pct(24) < -8", take_profit_pct 6, max_hold_bars 72   (buy a 24-bar drop of 8%)']
    return "\n".join(lines)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
    server = Server("edgeorluck", instructions=INSTRUCTIONS)
    server.tool(name="judge_trades", description=JUDGE_DESCRIPTION)(judge_trades)
    server.tool(name="result_schema", description="The JSON Schema of judge_trades' result (fields and what they "
                                                  "mean). Rarely needed: the result explains itself.")(result_schema)
    server.tool(name="backtest_rule", description=BACKTEST_DESCRIPTION)(backtest_rule)
    server.tool(name="tune_rule", description=TUNE_DESCRIPTION)(tune_rule)
    server.tool(name="rule_language", description="The rule language for backtest_rule and tune_rule: the price "
                                                  "names, every function, and examples. Read it before writing a "
                                                  "rule.")(rule_language)
    return server


def main() -> int:
    try:
        server = build_server()
    except ImportError:
        import sys
        print("The MCP server needs: pip install 'edgeorluck[mcp]'", file=sys.stderr)
        return 1
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["judge_trades", "result_schema", "backtest_rule", "tune_rule", "rule_language", "build_server", "main",
           "ENGINE_VERSION"]
