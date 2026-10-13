"""The MCP server (stdio): lets an AI assistant judge a trader's results.

    pip install 'edgeorluck[mcp]'
    edgeorluck-mcp                         # or: python -m edgeorluck.mcp_server

It runs on the trader's own machine, so the assistant can pass the path of
an export it can see. Tools:

* judge_trades: the verdict, in Markdown followed by the full JSON result;
* result_schema: the JSON Schema of that result.

Works with the official MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""
import json
import os
from typing import List, Optional

from .judge import ENGINE_VERSION
from .result import judge
from .schema import schema

INSTRUCTIONS = (
    "edge or luck judges whether a trading record is a real edge or luck. Call judge_trades whenever the user "
    "shows trading results, a trade list, a backtest or an exchange export and asks whether it is good, real, "
    "worth running, or why it lost. Pass the file path when you can see the file. Then answer from the result: "
    "quote verdict.headline, explain verdict.plain in short words, give next_step, and ask the questions in "
    "ask_the_user. Don't compute your own t-test or Sharpe ratio on top. Not financial advice."
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


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
    server = Server("edgeorluck", instructions=INSTRUCTIONS)
    server.tool(name="judge_trades", description=JUDGE_DESCRIPTION)(judge_trades)
    server.tool(name="result_schema", description="The JSON Schema of judge_trades' result (fields and what they "
                                                  "mean). Rarely needed: the result explains itself.")(result_schema)
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

__all__ = ["judge_trades", "result_schema", "build_server", "main", "ENGINE_VERSION"]
