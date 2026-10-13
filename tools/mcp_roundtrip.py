"""Start edgeorluck-mcp over stdio with the official MCP client, list its
tools and judge a file: proof the server works the way an assistant uses it.

    python tools/mcp_roundtrip.py [path-to-trades.csv]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def roundtrip(path: str) -> str:
    params = StdioServerParameters(command=sys.executable, args=["-m", "edgeorluck.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            result = await session.call_tool("judge_trades", {"file_path": path, "variants_tried": 20})
            text = "".join(getattr(c, "text", "") for c in result.content)
            return f"tools: {names}\n" + text


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import tempfile
        from edgeorluck import samples
        tmp = Path(tempfile.mkdtemp()) / "lucky.csv"
        tmp.write_text(samples.lucky(), encoding="utf-8")
        path = str(tmp)
    out = asyncio.run(roundtrip(path))
    print(out[:1500])
    return 0 if "## Edge or luck" in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
