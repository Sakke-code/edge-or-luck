"""Bundle the judge into one file, edgeorluck_single.py, for chatbots that can
run Python but can't install packages (ChatGPT's sandbox has numpy and
pandas, and no internet).

    python -m edgeorluck.single out.py     # or: edgeorluck --single-file out.py

The bundle is the package's own modules, concatenated in dependency order
with the package-relative imports removed. Building fails loudly when two
modules define the same top-level name, so the bundle can't silently differ
from the package; a test checks both give the same answers.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional

MODULES = ["evidence", "words", "fills", "readers", "compare", "judge", "report", "schema", "result", "rulebook",
           "market", "simulate"]
HERE = Path(__file__).resolve().parent

HEADER = '''"""edge or luck, in one file: is a trading record a real edge, or luck?

For an AI assistant with a Python sandbox (numpy + pandas) and no internet.
Upload this file together with the trades (a CSV/XLSX export, a TradingView
"List of trades", exchange fills, or a list of results), then run:

    import sys; sys.path.insert(0, "/mnt/data")
    import edgeorluck_single as eol
    r = eol.judge("/mnt/data/trades.csv", variants_tried=None)
    print(r.to_markdown())

or from a shell:  python edgeorluck_single.py trades.csv [--tried 40] [--json]

An idea and a price file (a CSV with a close column), but no trades? Test it
honestly; decisions at a bar's close are filled at the next bar's open:

    bt = eol.backtest("/mnt/data/prices.csv", {{"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}})
    print(bt.to_markdown())

Answer the trader from the result alone: quote verdict.headline, explain
verdict.plain in short words, give next_step, and ask the questions in
ask_the_user. Not financial advice: a statistical check of past trades.

Open source (MIT): https://github.com/Sakke-code/edge-or-luck
Built from edgeorluck {version}. Generated file: do not edit; rebuild it.
"""
from __future__ import annotations
'''

MAIN = '''

# ── command line ──────────────────────────────────────────────────────────────

def _single_main(argv=None):
    import argparse
    import glob
    import os
    parser = argparse.ArgumentParser(prog="edgeorluck_single", description="Is a trading record a real edge, or luck?")
    parser.add_argument("files", nargs="*", help="trade exports; default: every csv/tsv/xlsx/json next to this file "
                                                 "and in /mnt/data")
    parser.add_argument("--tried", help="how many versions/settings were tried in all")
    parser.add_argument("--benchmark", help="buy-and-hold return over the same dates, in percent")
    parser.add_argument("--fees", help="fees included? yes/no")
    parser.add_argument("--source", help="backtest, paper or live")
    parser.add_argument("--backtest", help="the backtest's trades, to compare live trades with")
    parser.add_argument("--json", action="store_true", help="print the JSON result")
    args = parser.parse_args(argv)
    files = list(args.files)
    if not files:
        here = os.path.dirname(os.path.abspath(__file__))
        for folder in dict.fromkeys([here, "/mnt/data", os.getcwd()]):
            for pattern in ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.json"):
                files += [f for f in glob.glob(os.path.join(folder, pattern)) if "edgeorluck" not in f]
    if not files:
        print("No trade files found. Pass a path: python edgeorluck_single.py trades.csv")
        return 2
    for path in files:
        result = judge(path, variants_tried=args.tried, benchmark_return_pct=args.benchmark,
                       fees_included=args.fees, source=args.source, backtest=args.backtest)
        print(result.to_json() if args.json else result.to_markdown())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_single_main())
'''


def _strip(source: str, name: str) -> str:
    tree = ast.parse(source)
    drop: List[range] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.level or 0) >= 1:
            drop.append(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            drop.append(range(node.lineno, node.end_lineno + 1))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or 0) >= 1 and node not in tree.body:
            raise ValueError(f"{name}: a package-relative import inside a function can't be bundled "
                             f"(line {node.lineno})")
    lines = source.splitlines()
    kept = [line for i, line in enumerate(lines, start=1) if not any(i in r for r in drop)]
    return "\n".join(kept).strip("\n")


def _names(source: str) -> List[str]:
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out.append(node.name)
        elif isinstance(node, ast.Assign):
            out += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
    return out


def build() -> str:
    from .judge import ENGINE_VERSION
    seen: Dict[str, str] = {}
    parts = [HEADER.format(version=ENGINE_VERSION)]
    for name in MODULES:
        source = (HERE / f"{name}.py").read_text(encoding="utf-8")
        for top in _names(source):
            if top in seen and top != "__all__":
                raise ValueError(f"{name}.{top} clashes with {seen[top]}.{top}: rename one to bundle them")
            seen[top] = name
        parts.append(f"\n\n# ── {name}.py " + "─" * max(0, 66 - len(name)) + "\n\n" + _strip(source, name))
    parts.append(MAIN)
    return "\n".join(parts).rstrip() + "\n"


def write(path: Optional[str] = None) -> Path:
    target = Path(path or "edgeorluck_single.py")
    target.write_text(build(), encoding="utf-8", newline="\n")
    return target


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print(write(argv[0] if argv else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
