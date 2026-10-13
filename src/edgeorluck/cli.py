"""The command line.

    edgeorluck trades.csv [--tried 50] [--benchmark 34.5] [--json | --md]
    edgeorluck live.csv --backtest backtest.csv
    edgeorluck -                       # read the trades from stdin
    edgeorluck --schema                # the result's JSON Schema
    edgeorluck --serve [--port 5078]   # the web app (needs edgeorluck[web])
    edgeorluck --single-file out.py    # the judge in one file, for chatbots

Several files are judged one by one. Exit code 0 when judged, 2 when the
input needs something (status need_input), 1 on an error.
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
                                            "column, XLSX, freqtrade JSON) or '-' for stdin.")
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


def main(argv: Optional[List[str]] = None) -> int:
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
        text = result.to_json() if args.json else result.to_markdown()
        try:
            print(text)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        if result.status == "need_input":
            worst = max(worst, 2)
        elif result.status == "error":
            worst = max(worst, 1) if worst != 2 else worst
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
