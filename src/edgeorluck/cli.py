"""The command line.

    edgeorluck trades.csv [--tried 50] [--benchmark 34.5] [--json | --md]
    edgeorluck live.csv --backtest backtest.csv
    edgeorluck -                       # read the trades from stdin
    edgeorluck --schema                # the result's JSON Schema
    edgeorluck --serve [--port 5078]   # the web app (needs edgeorluck[web])
    edgeorluck --single-file out.py    # the judge in one file, for chatbots

An idea with no trades yet: test it honestly, or tune it on one part of the
history and judge it on the rest (prices from Binance, or --prices file.csv):

    edgeorluck backtest --buy "rsi(14) < 30" --sell "rsi(14) > 70" --symbol ETHUSDT --start 2023-01-01
    edgeorluck tune --buy "rsi(14) < {low}" --sell "rsi(14) > {high}" --grid "low=20,25,30 high=65,70,75"

Several files are judged one by one. Exit code 0 when judged, 2 when the
input needs something (status need_input) or a backtest had nothing to judge,
1 on an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .judge import ENGINE_VERSION
from .result import judge
from .schema import schema


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgeorluck",
                                description="Is a trading record a real edge, or luck? Give it a trade export "
                                            "(TradingView list of trades, exchange fills, any CSV with a P&L "
                                            "column, XLSX, freqtrade JSON) or '-' for stdin.",
                                epilog="An idea with no trades yet? edgeorluck backtest --help, edgeorluck tune "
                                       "--help.")
    p.add_argument("files", nargs="*", help="trade exports to judge, one by one")
    p.add_argument("--tried", "--variants-tried", dest="tried",
                   help="how many versions or settings were tried in all (raises the bar)")
    p.add_argument("--benchmark", help="what buying and holding returned over the same dates, in percent")
    p.add_argument("--fees", help="are fees included? yes or no")
    p.add_argument("--source", help="backtest, paper or live")
    p.add_argument("--unit", help="'%%' or a currency like USDT, when the file doesn't say")
    p.add_argument("--kind", help="auto, trades, fills, tradingview or pnl")
    p.add_argument("--backtest", help="the backtest's trades, to check live trades against")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print the JSON result")
    out.add_argument("--md", action="store_true", help="print Markdown (the default)")
    p.add_argument("--schema", action="store_true", help="print the result's JSON Schema and exit")
    p.add_argument("--serve", action="store_true", help="run the web app with waitress")
    p.add_argument("--host", default="127.0.0.1", help="web app host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="web app port (default $PORT or 5078)")
    p.add_argument("--single-file", nargs="?", const="edgeorluck_single.py", metavar="PATH",
                   help="write the one-file build and exit")
    p.add_argument("--version", action="version", version=f"edgeorluck {ENGINE_VERSION}")
    return p


def _print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))


def _rule_parser(command: str) -> argparse.ArgumentParser:
    if command == "backtest":
        p = argparse.ArgumentParser(prog="edgeorluck backtest",
                                    description="Test a rule honestly and judge it: each decision is made at a bar's "
                                                "close and filled at the next bar's open, with fees and slippage on "
                                                "every fill.",
                                    epilog='Example: edgeorluck backtest --buy "rsi(14) < 30" --sell "rsi(14) > 70" '
                                           "--symbol ETHUSDT --interval 4h --start 2023-01-01")
    else:
        p = argparse.ArgumentParser(prog="edgeorluck tune",
                                    description="Tune a rule honestly: try every setting on the first part of the "
                                                "prices, then judge the chosen one once on the last part, which "
                                                "played no part in the choice.",
                                    epilog='Example: edgeorluck tune --buy "rsi(14) < {low}" --sell "rsi(14) > {high}" '
                                           '--grid "low=20,25,30 high=65,70,75"')
    p.add_argument("--buy", required=True, help='when to get in, in the rule language: "rsi(14) < 30"')
    p.add_argument("--sell", help='when to get out: "rsi(14) > 70"')
    p.add_argument("--stop", help="stop loss in percent, checked at each close")
    p.add_argument("--take", help="take profit in percent, checked at each close")
    p.add_argument("--hold", help="the most bars to hold a trade")
    p.add_argument("--short", action="store_true", help="sell first (a short) instead of buying")
    p.add_argument("--symbol", default="BTCUSDT", help="a Binance spot pair (default BTCUSDT)")
    p.add_argument("--interval", default="1h", help="bar length: 1m 5m 15m 1h 4h 1d ... (default 1h)")
    p.add_argument("--start", help="first date, like 2023-01-01 (default: two years ago)")
    p.add_argument("--end", help="last date (default: now)")
    p.add_argument("--prices", help="a price CSV to use instead of fetching (needs a close column)")
    p.add_argument("--fee", type=float, default=0.1, help="fee per fill in percent (default 0.1)")
    p.add_argument("--slippage", type=float, default=0.05, help="slippage per fill in percent (default 0.05)")
    if command == "backtest":
        p.add_argument("--tried", "--variants-tried", dest="tried",
                       help="how many rules or settings were tried in all, this one included")
    else:
        p.add_argument("--grid", action="append", required=True,
                       help='the values to try: "low=20,25,30 high=65,70" (or JSON); may be repeated')
        p.add_argument("--holdout", type=float, default=0.3, help="the share kept unseen (default 0.3)")
    p.add_argument("--json", action="store_true", help="print the JSON result")
    return p


def _grid(parts: List[str]) -> dict:
    grid: dict = {}
    for part in parts:
        part = part.strip()
        if part.startswith("{"):
            grid.update(json.loads(part))
            continue
        for item in part.split():
            name, sep, values = item.partition("=")
            if not sep or not values:
                raise ValueError(f"can't read {item!r} in --grid: write name=1,2,3")
            nums = [float(v) for v in values.split(",") if v]
            grid[name] = [int(v) if v.is_integer() else v for v in nums]
    return grid


def _rule_main(command: str, argv: List[str]) -> int:
    from .market import PriceError, load_prices, prices
    from .rulebook import RuleError
    from .simulate import backtest, tune
    args = _rule_parser(command).parse_args(argv)
    rule = {k: v for k, v in {"buy": args.buy, "sell": args.sell, "stop_loss_pct": args.stop,
                              "take_profit_pct": args.take, "max_hold_bars": args.hold,
                              "side": "short" if args.short else "long"}.items() if v is not None}
    try:
        px = load_prices(args.prices) if args.prices else prices(args.symbol, args.interval, args.start, args.end)
        if command == "backtest":
            result = backtest(px, rule, fee_pct=args.fee, slippage_pct=args.slippage, variants_tried=args.tried)
        else:
            result = tune(px, rule, _grid(args.grid), holdout=args.holdout, fee_pct=args.fee,
                          slippage_pct=args.slippage)
    except (RuleError, PriceError, ValueError, OSError) as exc:
        print(f"edgeorluck {command}: {exc}", file=sys.stderr)
        return 1
    _print(result.to_json() if args.json else result.to_markdown())
    return 0 if result.status == "ok" else 2


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("backtest", "tune"):
        return _rule_main(argv[0], argv[1:])
    args = _parser().parse_args(argv)
    if args.schema:
        print(json.dumps(schema(), indent=2))
        return 0
    if args.single_file:
        from .single import write
        print(write(args.single_file))
        return 0
    if args.serve:
        try:
            from .web.app import serve
        except ImportError:
            print("The web app needs: pip install 'edgeorluck[web]'", file=sys.stderr)
            return 1
        serve(host=args.host, port=args.port)
        return 0
    if not args.files:
        _parser().print_help()
        return 2
    worst = 0
    for i, path in enumerate(args.files):
        data = sys.stdin.buffer.read() if path == "-" else path
        result = judge(data, variants_tried=args.tried, benchmark_return_pct=args.benchmark, kind=args.kind,
                       unit=args.unit, fees_included=args.fees, source=args.source, backtest=args.backtest)
        if i:
            print()
        _print(result.to_json() if args.json else result.to_markdown())
        if result.status == "need_input":
            worst = max(worst, 2)
        elif result.status == "error":
            worst = max(worst, 1) if worst != 2 else worst
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
