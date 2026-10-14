"""How much edge or luck is used, counted without knowing by whom: checks per day, by the way they came in (the
page, a link, the JSON tools, MCP over the web) and by the kind of assistant that asked (from the name it gives).
No addresses, no trades, no contents: counts only. GET /api/usage shows them: the public record of being used."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ASSISTANTS = (("chatgpt", "ChatGPT"), ("openai", "ChatGPT"), ("claude", "Claude"), ("anthropic", "Claude"),
              ("cursor", "Cursor"), ("codex", "Codex"), ("copilot", "Copilot"), ("perplexity", "Perplexity"),
              ("gemini", "Gemini"), ("google", "Gemini"), ("windsurf", "Windsurf"), ("mcp", "another assistant"))
WAYS = {"page": "the page", "link": "links", "tools": "the JSON tools", "mcp": "MCP"}


def assistant(declared: str) -> str:
    """The kind of assistant a user agent or MCP client name declares ("" for people and scripts)."""
    text = str(declared or "").lower()
    return next((name for key, name in ASSISTANTS if key in text), "")


class Usage:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "usage.json"
        self._lock = threading.Lock()
        try:
            self.days: Dict[str, Dict[str, int]] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.days = {}
        self._unsaved, self._saved = 0, time.monotonic()

    def count(self, way: str, declared: str = "") -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{way}:{assistant(declared) or 'people'}"
        with self._lock:
            today = self.days.setdefault(day, {})
            today[key] = today.get(key, 0) + 1
            self._unsaved += 1
            if self._unsaved >= 25 or time.monotonic() - self._saved > 60:
                self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.days, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        self._unsaved, self._saved = 0, time.monotonic()

    def summary(self) -> Dict[str, object]:
        with self._lock:
            ways: Dict[str, int] = {}
            who: Dict[str, int] = {}
            for counts in self.days.values():
                for key, n in counts.items():
                    way, _, name = key.partition(":")
                    ways[way] = ways.get(way, 0) + n
                    who[name] = who.get(name, 0) + n
            daily = {day: sum(counts.values()) for day, counts in sorted(self.days.items())[-30:]}
        total = sum(ways.values())
        ai = total - who.get("people", 0)
        named = ", ".join(f"{name} {n:,}" for name, n in sorted(who.items(), key=lambda kv: -kv[1]) if name != "people")
        sentence = (f"{total:,} check{'' if total == 1 else 's'} so far, {ai:,} of them for AI assistants"
                    + (f" ({named})." if named else ".")) if total else "No checks counted yet."
        return {"total": total, "for_assistants": ai, "by_way": {WAYS.get(k, k): v for k, v in ways.items()},
                "by_assistant": {k: v for k, v in who.items() if k != "people"}, "last_30_days": daily,
                "since": min(self.days) if self.days else None, "sentence": sentence}


__all__ = ["Usage", "assistant"]
