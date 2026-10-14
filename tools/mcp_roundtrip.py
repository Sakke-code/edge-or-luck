"""Start edgeorluck-mcp over stdio with the official MCP client, list its
tools, judge a file, and backtest and tune a rule on a price file: proof the
server works the way an assistant uses it (no internet needed).

    python tools/mcp_roundtrip.py [path-to-trades.csv]
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _price_file() -> str:
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.01, 1500)))
    path = Path(tempfile.mkdtemp()) / "prices.csv"
    path.write_text("close\n" + "\n".join(f"{c:.4f}" for c in closes) + "\n", encoding="utf-8")
    return str(path)


async def roundtrip(path: str) -> str:
    params = StdioServerParameters(command=sys.executable, args=["-m", "edgeorluck.mcp_server"])
    prices = _price_file()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            texts = []
            for name, args in (("judge_trades", {"file_path": path, "variants_tried": 20}),
                               ("backtest_rule", {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70",
                                                  "prices_file": prices}),
                               ("tune_rule", {"buy": "rsi(14) < {low}", "sell": "rsi(14) > 70",
                                              "grid": {"low": [25, 30, 35]}, "prices_file": prices})):
                result = await session.call_tool(name, args)
                texts.append("".join(getattr(c, "text", "") for c in result.content)[:1200])
            return f"tools: {names}\n" + "\n\n".join(texts)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import tempfile
        from edgeorluck import samples
        tmp = Path(tempfile.mkdtemp()) / "lucky.csv"
        tmp.write_text(samples.lucky(), encoding="utf-8")
        path = str(tmp)
    out = asyncio.run(roundtrip(path))
    print(out)
    return 0 if all(s in out for s in ("## Edge or luck", "## Honest backtest:", "## Honest tuning:")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
