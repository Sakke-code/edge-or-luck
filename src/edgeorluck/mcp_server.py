"""The MCP server (stdio): lets an AI assistant judge a trader's results.

    uvx edgeorluck mcp                     # or, installed: edgeorluck-mcp, python -m edgeorluck.mcp_server

It runs on the trader's own machine, so the assistant can pass the path of
an export it can see. Tools:

* judge_trades: the verdict, in Markdown (the JSON result too, on request);
* backtest_rule: an honest backtest of a rule (fetched or local prices), judged; freeze=true keeps the rule;
* forward_test: a frozen rule judged on its trades since it was frozen only;
* tune_rule: honest tuning, judged on a part the choice never saw;
* rule_language: the rule language's names, functions and examples.

Every distinct record and rule judged in a session counts as a version tried,
so the bar rises with them even when nobody says how many there were.
Works with the official MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""
import hashlib
import json
import os
import threading
from collections import OrderedDict
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from .judge import ENGINE_VERSION
from .result import judge

INSTRUCTIONS = (
    "edge or luck judges whether a trading record is a real edge or luck, and tests trading ideas honestly. Call "
    "judge_trades whenever the user shows trading results, a trade list, a backtest or an exchange export and asks "
    "whether it is good, real, worth running, or why it lost. Call backtest_rule when the user has an idea or a rule "
    "but no trades, and tune_rule when they want the best settings, instead of writing your own backtest code. To "
    "test a rule on trades it never saw, freeze it (backtest_rule freeze=true) and call forward_test later. Then "
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
    "luck, so this raises the bar. Ask the user if unknown. The server counts the ones judged in this session itself.\n"
    "- benchmark_return_pct: what buying and holding returned over the same dates, in percent (34.5 = +34.5%). "
    "Catches a long-only strategy that only rode a rising market.\n"
    "- fees_included (true/false), source ('backtest', 'paper' or 'live'), unit ('%' or 'USDT' etc.), "
    "kind ('auto', 'trades', 'fills', 'tradingview', 'pnl').\n"
    "- backtest_file_path / backtest_csv_text / backtest_pnl: the backtest, to check the live trades against it.\n"
    "- detail: 'markdown' (default) or 'full' (the JSON result too: rarely needed, it repeats the Markdown).\n\n"
    "Returns Markdown, ready to relay. Answer the user from it: quote verdict.headline word for word, explain "
    "verdict.plain in short words, give next_step, and ask the questions in ask_the_user (each says which "
    "parameter to set when you call again). Never present 'cant_tell' as a win. Not financial advice."
)

SESSION: set = set()     # each distinct record or rule judged since the server started: a version tried
_CURRENT: ContextVar[Optional[str]] = ContextVar("edgeorluck_session", default=None)  # over the web: the session id
_SESSIONS: "OrderedDict[str, set]" = OrderedDict()     # over the web, the most recent MAX_SESSIONS sessions
MAX_SESSIONS = 10_000
_SESSIONS_LOCK = threading.Lock()
FROZEN = os.path.join(os.path.expanduser("~"), ".edgeorluck", "frozen.json")


def _path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value.strip().strip('"'))))


def _tried() -> set:
    """What this session has judged: one set over stdio, one per assistant session over the web ("" when the
    client keeps no session: then nothing carries over from one call to the next)."""
    current = _CURRENT.get()
    if not current:
        return SESSION if current is None else set()
    with _SESSIONS_LOCK:                                # tools run in worker threads
        _SESSIONS[current] = _SESSIONS.pop(current, None) or set()
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
        return _SESSIONS[current]


def _versions(said: Optional[int]) -> Optional[int]:
    """The versions tried: never fewer than this session has judged (None: unknown, and one so far)."""
    return max(said or 0, len(_tried())) if (said or len(_tried()) > 1) else None


def _note() -> str:
    return "" if len(_tried()) < 2 else (
        f"\n_This session has judged {len(_tried())} records and rules. Each is a version tried, so the bar is raised "
        "to match. Versions tried before the session aren't counted: ask, and pass the total as variants_tried._\n")


def judge_trades(file_path: Optional[str] = None, csv_text: Optional[str] = None,
                 pnl: Optional[List[float]] = None, variants_tried: Optional[int] = None,
                 benchmark_return_pct: Optional[float] = None, fees_included: Optional[bool] = None,
                 source: Optional[str] = None, unit: Optional[str] = None, kind: Optional[str] = None,
                 backtest_file_path: Optional[str] = None, backtest_csv_text: Optional[str] = None,
                 backtest_pnl: Optional[List[float]] = None, detail: str = "markdown") -> str:
    """Judge a trading record: Markdown (detail="full": then the JSON result)."""
    data = _path(file_path) or csv_text or pnl
    if data is None:
        return ("Nothing to judge: pass file_path (an export on this machine), csv_text (its text) or pnl (a list "
                "of results per trade, oldest first).")
    if isinstance(data, str) and file_path and not os.path.exists(data):
        return (f"No file at {data}. Check the path (an absolute path is safest), or pass the file's text as "
                "csv_text.")
    run = partial(judge, data, benchmark_return_pct=benchmark_return_pct, fees_included=fees_included, source=source,
                  unit=unit, kind=kind, backtest=_path(backtest_file_path) or backtest_csv_text or backtest_pnl)
    result = run(variants_tried=variants_tried)
    if result.status == "ok":           # the same trades judged again (say, with an answer added) are no new version
        _tried().add(result["input_fingerprint"])
        if _versions(variants_tried) != variants_tried:
            result = run(variants_tried=_versions(variants_tried))
    full = str(detail).strip().lower().startswith("full")
    return result.to_markdown() + _note() + (f"\n## Full result (JSON)\n\n```json\n{result.to_json(None)}\n```\n"
                                             if full else "")


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
    "variants_tried: versions tried before this session; the server counts the rules tested in it (the bar rises "
    "with them). freeze=true keeps the rule for forward_test, which judges only its trades after now.\n\n"
    "Returns Markdown: the backtest, its honesty checks (look-ahead, one bar later, costs) and the verdict. Quote "
    "the verdict's headline. To compare settings, use tune_rule instead of calling this many times. Not financial "
    "advice."
)

FORWARD_DESCRIPTION = (
    "Judge a frozen rule (backtest_rule with freeze=true) on the trades it made after it was frozen, on fresh "
    "prices. None of them could have shaped the rule, so its bar is set by how many rules were frozen, not by every "
    "version tried: the honest fast way to a verdict after many tries.\n\n"
    "CALL THIS when the user comes back to a frozen rule or asks how it is doing. frozen_id: the id backtest_rule "
    "gave (a wrong id lists the frozen ones). Returns Markdown: quote the verdict's headline. Not financial advice."
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
    from .market import between, load_prices, prices
    if prices_file:
        path = _path(prices_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file at {path}.")
        return between(load_prices(path), start, end)
    return prices(symbol or "BTCUSDT", interval, start=start, end=end)


def _rule_dict(buy: str, sell: Optional[str], stop_loss_pct, take_profit_pct, max_hold_bars, side: str):
    return {k: v for k, v in {"buy": buy, "sell": sell, "stop_loss_pct": stop_loss_pct,
                              "take_profit_pct": take_profit_pct, "max_hold_bars": max_hold_bars,
                              "side": side}.items() if v not in (None, "")}


def _run(failed: str, work: Callable[[], Any], key: Any = None) -> str:
    """A tool's work as Markdown; key counts it as a version tried this session, unless it fails."""
    from .market import PriceError
    from .rulebook import RuleError
    key = None if key is None else json.dumps(key, sort_keys=True, default=str)
    new = key is not None and key not in _tried()
    _tried().update([key] if new else [])
    try:
        return work().to_markdown() + _note()
    except (RuleError, PriceError, ValueError, TypeError, OSError) as exc:   # ValueError: a CSV pandas can't parse
        _tried().difference_update([key] if new else [])
        return f"{failed}: {exc}"


def _frozen() -> Dict[str, Dict[str, Any]]:
    try:
        with open(FROZEN, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def backtest_rule(buy: str, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
                  start: Optional[str] = None, end: Optional[str] = None, prices_file: Optional[str] = None,
                  stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None,
                  max_hold_bars: Optional[int] = None, side: str = "long", fee_pct: float = 0.1,
                  slippage_pct: float = 0.05, variants_tried: Optional[int] = None, freeze: bool = False) -> str:
    """An honest backtest of a rule, judged: Markdown. freeze=True keeps the rule for forward_test."""
    from .simulate import backtest
    spec = {"rule": _rule_dict(buy, sell, stop_loss_pct, take_profit_pct, max_hold_bars, side), "symbol": symbol,
            "interval": interval, "start": start, "prices_file": _path(prices_file), "fee_pct": fee_pct,
            "slippage_pct": slippage_pct}
    md = _run("Couldn't run the backtest", lambda: backtest(
        _prices_for(symbol, interval, start, end, prices_file), spec["rule"], fee_pct=fee_pct,
        slippage_pct=slippage_pct, variants_tried=_versions(variants_tried)), dict(spec, end=end))
    if freeze and not md.startswith("Couldn't"):
        book, key = _frozen(), hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
        book.setdefault(key, dict(spec, since=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
        with open(FROZEN, "w", encoding="utf-8") as f:
            json.dump(book, f, indent=1)
        md += (f"\n**Frozen as `{key}`** on {book[key]['since'][:16].replace('T', ' ')} UTC. From then on only its new "
               f"trades count: call forward_test(\"{key}\") later. Its bar is set by the {len(book)} rule(s) frozen so "
               "far, not by every version tried. Change the rule and it is a new one.\n")
    return md


def forward_test(frozen_id: str) -> str:
    """A frozen rule judged on its trades since it was frozen only: Markdown."""
    from .simulate import backtest
    book = _frozen()
    spec = book.get(str(frozen_id).strip().strip("`\""))
    if spec is None:
        return f"No frozen rule {frozen_id!r}. Frozen: {', '.join(book) or 'none yet (backtest_rule freeze=true)'}."
    return _run("Couldn't run the forward test", lambda: backtest(
        _prices_for(spec["symbol"], spec["interval"], spec["start"], None, spec["prices_file"]), spec["rule"],
        fee_pct=spec["fee_pct"], slippage_pct=spec["slippage_pct"], variants_tried=len(book), since=spec["since"]))


def tune_rule(buy: str, grid: dict, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
              start: Optional[str] = None, end: Optional[str] = None, prices_file: Optional[str] = None,
              stop_loss_pct: Optional[Union[float, str]] = None, take_profit_pct: Optional[Union[float, str]] = None,
              max_hold_bars: Optional[Union[int, str]] = None, side: str = "long", holdout: float = 0.3,
              fee_pct: float = 0.1, slippage_pct: float = 0.05) -> str:
    """Honest tuning with a held-back part, judged: Markdown."""
    from .simulate import tune
    rule = _rule_dict(buy, sell, stop_loss_pct, take_profit_pct, max_hold_bars, side)
    return _run("Couldn't tune", lambda: tune(
        _prices_for(symbol, interval, start, end, prices_file), rule, json.loads(grid) if isinstance(grid, str)
        else grid, holdout=holdout, fee_pct=fee_pct, slippage_pct=slippage_pct, variants_tried=_versions(None)),
        {"tune": rule, "grid": grid, "at": [symbol, interval, start, end, prices_file, holdout]})


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
              '- buy "close > prev(highest(high, 20)) and volume > 2 * sma(volume, 20)", stop_loss_pct 5, '
              'max_hold_bars 48   (highest includes this bar, so a breakout compares with the bar before)',
              '- buy "pct(24) < -8", take_profit_pct 6, max_hold_bars 72   (buy a 24-bar drop of 8%)']
    return "\n".join(lines)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
    server = Server("edgeorluck", instructions=INSTRUCTIONS)
    server.tool(name="judge_trades", description=JUDGE_DESCRIPTION)(judge_trades)
    server.tool(name="backtest_rule", description=BACKTEST_DESCRIPTION)(backtest_rule)
    server.tool(name="forward_test", description=FORWARD_DESCRIPTION)(forward_test)
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
        print("The MCP server needs the mcp package: pip install 'mcp>=1.2'", file=sys.stderr)
        return 1
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["judge_trades", "backtest_rule", "forward_test", "tune_rule", "rule_language", "build_server", "main",
           "ENGINE_VERSION"]
