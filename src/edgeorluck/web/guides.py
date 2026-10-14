"""The mini guides: short Markdown files in edgeorluck/guides, served as pages.

Each guide is plain Markdown with a small header (title, summary, Flip's line,
mood, minutes, group, order). The pages are rendered on the server, so an
assistant that fetches a guide without running scripts still reads all of it,
and each guide is also served as Markdown (/guides/<slug>.md).

Numbers the judge has measured are written as {{name}} and filled in from
data/validation.json, so a guide can't quote a stale error rate.

The Markdown is a small subset, rendered here with no dependency: ## and ###
headings, paragraphs, - and 1. lists, | tables |, ``` code, **bold**, *italic*,
`code`, [links](/path), and two blocks of our own:

    :::flip worried          Flip says something (the word after flip is the mood)
    ...
    :::
    :::try                   a "Try it" box
    ...
    :::
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PACKAGE = Path(__file__).resolve().parents[1]
GUIDES = PACKAGE / "guides"
GROUPS = {"backtest": "Before you trust a backtest", "verdict": "Reading your verdict"}
FIELDS = ("title", "summary", "flip", "mood", "minutes", "group", "order")


@dataclass
class Guide:
    slug: str
    title: str
    summary: str
    flip: str
    mood: str
    minutes: int
    group: str
    order: int
    markdown: str          # the body, numbers filled in

    def as_dict(self) -> Dict[str, Any]:
        return {"slug": self.slug, "title": self.title, "summary": self.summary, "minutes": self.minutes,
                "group": GROUPS.get(self.group, self.group), "url": f"/guides/{self.slug}",
                "markdown_url": f"/guides/{self.slug}.md"}

    def full_markdown(self) -> str:
        return f"# {self.title}\n\n{self.summary}\n\n{self.markdown.strip()}\n"

    def for_github(self) -> str:
        """The guide for docs/guides/: numbers filled in, the site's links pointed at the repository."""
        text = self.full_markdown()
        text = re.sub(r"\]\(/guides/([a-z0-9-]+)\)", r"](\1.md)", text)
        text = text.replace("](/how)", "](../../README.md#how-it-judges-the-short-version)")
        text = text.replace("](/)", "](../../README.md#use-it)")
        return ("<!-- Generated from src/edgeorluck/guides by tools/write_guides.py: edit the source, not this. -->\n"
                + text)


# ── numbers from the validation run ───────────────────────────────────────────

def share(rate: float) -> str:
    """0.0015 -> 0.15%, 0.088 -> 8.8%, 0.603 -> 60%, 0.997 -> 99.7% (never rounded up to 100% or down to 0%)."""
    v = rate * 100
    if v == 0:
        return "0%"
    if v < 0.01:
        return "under 0.01%"
    if v < 1:
        return f"{v:.2f}".rstrip("0").rstrip(".") + "%"
    if v < 10 or 99.5 <= v < 100:
        return f"{v:.1f}".rstrip("0").rstrip(".") + "%"
    return f"{round(v)}%"


def numbers() -> Dict[str, str]:
    path = PACKAGE / "data" / "validation.json"
    v = json.loads(path.read_text(encoding="utf-8"))
    h = v["headline"]
    return {
        "false_typical": share(h["false_real_edge_typical"]),
        "false_target": share(h["false_real_edge_target"]),
        "checked_often_worst": share(v["checked_often"]["autocorrelated"]["rate"]),
        "checked_every_10": share(v["checked_often"]["normal"]["rate"]),
        "best10_without": share(v["best_of_n"]["10"]["without"]["rate"]),
        "best10_with": share(v["best_of_n"]["10"]["with"]["rate"]),
        "best100_without": share(v["best_of_n"]["100"]["without"]["rate"]),
        "best100_with": share(v["best_of_n"]["100"]["with"]["rate"]),
        "best1000_without": share(v["best_of_n"]["1000"]["without"]["rate"]),
        "best1000_with": share(v["best_of_n"]["1000"]["with"]["rate"]),
        "power_300": share(v["power"]["0.2"]["300"]["rate"]),
        "power_1000": share(v["power"]["0.2"]["1000"]["rate"]),
        "power_100": share(v["power"]["0.2"]["100"]["rate"]),
        "faded_caught": share(v["faded"]["real_fade_caught"]["rate"]),
        "faded_false": share(v["faded"]["steady_edge_called_faded"]["rate"]),
        "compare_caught": share(v["compare"]["worse_caught_by_200"]["rate"]),
        "compare_false": share(v["compare"]["false_worse_checked_every_10"]["rate"]),
    }


def _fill(text: str, values: Dict[str, str], where: str) -> str:
    def one(m: "re.Match[str]") -> str:
        if m.group(1) not in values:
            raise KeyError(f"{where}: no number called {{{{{m.group(1)}}}}}")
        return values[m.group(1)]
    return re.sub(r"\{\{\s*([a-z0-9_]+)\s*\}\}", one, text)


# ── loading ───────────────────────────────────────────────────────────────────

def _parse(path: Path, values: Dict[str, str]) -> Guide:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError(f"{path.name}: a guide starts with a --- header ---")
    head: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            head[key.strip()] = value.strip()
    missing = [f for f in FIELDS if not head.get(f)]
    if missing:
        raise ValueError(f"{path.name}: the header needs {', '.join(missing)}")
    slug = re.sub(r"^\d+-", "", path.stem)
    return Guide(slug=slug, title=_fill(head["title"], values, path.name),
                 summary=_fill(head["summary"], values, path.name), flip=_fill(head["flip"], values, path.name),
                 mood=head["mood"], minutes=int(head["minutes"]), group=head["group"], order=int(head["order"]),
                 markdown=_fill(text[m.end():], values, path.name))


def load() -> List[Guide]:
    values = numbers()
    guides = [_parse(p, values) for p in sorted(GUIDES.glob("*.md"))]
    return sorted(guides, key=lambda g: (list(GROUPS).index(g.group) if g.group in GROUPS else 99, g.order))


def find(slug: str) -> Optional[Guide]:
    return next((g for g in load() if g.slug == slug), None)


# ── Markdown → HTML (the small subset above) ──────────────────────────────────

_SAFE_URL = re.compile(r"^(/|#|https?://)")


def _inline(text: str) -> str:
    out = []
    for i, part in enumerate(re.split(r"(`[^`]+`)", text)):
        if i % 2:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        s = html.escape(part, quote=False)

        def link(m: "re.Match[str]") -> str:
            url = html.unescape(m.group(2))
            if not _SAFE_URL.match(url):
                return m.group(0)
            ext = ' rel="noopener"' if url.startswith("http") else ""
            return f'<a href="{html.escape(url)}"{ext}>{m.group(1)}</a>'
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def _anchor(plain: str, used: Dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "part"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base}-{used[base]}"


def _table(rows: List[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    divider = len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ")
    head, body = cells[0], (cells[2:] if divider else cells[1:])
    out =['<div class="table"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def render(markdown: str) -> Tuple[str, List[Tuple[str, str]]]:
    """HTML for a guide body, and its (anchor, heading) list."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    toc: List[Tuple[str, str]] = []
    used: Dict[str, int] = {}
    para: List[str] = []
    i = 0

    def flush() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f'<div class="code"><pre><code{cls}>{html.escape(chr(10).join(body))}</code></pre>'
                       '<button class="copy" type="button">Copy</button></div>')
        elif stripped.startswith(":::") and stripped != ":::":
            flush()
            kind, _, arg = stripped[3:].strip().partition(" ")
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != ":::":
                body.append(lines[i])
                i += 1
            inner, _ = render("\n".join(body))
            if kind == "flip":
                mood = re.sub(r"[^a-z]", "", arg) or "idle"
                out.append(f'<aside class="flip-says"><div class="mini-pet" data-mood="{mood}" aria-hidden="true">'
                           f'</div><div class="bubble"><b class="who">Flip</b>{inner}</div></aside>')
            elif kind == "try":
                out.append(f'<aside class="try"><b class="label">Try it</b>{inner}</aside>')
            else:
                out.append(f'<aside class="note">{inner}</aside>')
        elif stripped.startswith("### ") or stripped.startswith("## "):
            flush()
            level = 3 if stripped.startswith("### ") else 2
            text = _inline(stripped[level + 1:].strip())
            plain = html.unescape(re.sub(r"<[^>]+>", "", text))
            anchor = _anchor(plain, used)
            if level == 2:
                toc.append((anchor, plain))
            out.append(f'<h{level} id="{anchor}">{text}</h{level}>')
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_table(rows))
            continue
        elif re.match(r"^(-|\d+\.)\s", stripped):
            flush()
            ordered = not stripped.startswith("-")
            items: List[List[str]] = []          # [text, sub-item, sub-item, ...]: one level of nesting
            while i < len(lines):
                raw, s = lines[i], lines[i].strip()
                nested = raw.startswith("  ") and bool(re.match(r"^(-|\d+\.)\s", s))
                if nested and items:
                    items[-1].append(re.sub(r"^(-|\d+\.)\s+", "", s))
                elif re.match(r"^(-|\d+\.)\s", s) and not raw.startswith(" ") and s[0].isdigit() == ordered:
                    items.append([re.sub(r"^(-|\d+\.)\s+", "", s)])
                elif s and raw.startswith("  ") and items:
                    items[-1][-1] += " " + s
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            parts = []
            for text, *subs in items:
                sub = ("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in subs) + "</ul>") if subs else ""
                parts.append(f"<li>{_inline(text)}{sub}</li>")
            out.append(f"<{tag}>" + "".join(parts) + f"</{tag}>")
            continue
        elif not stripped:
            flush()
        else:
            para.append(stripped)
        i += 1
    flush()
    return "\n".join(out), toc


# ── the pages ─────────────────────────────────────────────────────────────────

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' "
           "cy='16' r='12.5' fill='%23F3F7F4' stroke='%2316294A' stroke-width='2.6'/%3E%3Cpath d='M8.5 20.5 13 16.5l3 "
           "2.4 7-8.4' fill='none' stroke='%2316294A' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'/"
           "%3E%3C/svg%3E")


def _shell(title: str, description: str, body: str, *, alternate: Optional[str] = None) -> str:
    e = html.escape
    alt = f'<link rel="alternate" type="text/markdown" href="{e(alternate)}">' if alternate else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<link rel="icon" href="{FAVICON}">
<link rel="stylesheet" href="/assets/site.css">
<link rel="stylesheet" href="/assets/guides.css">
{alt}
</head>
<body>
<header class="top">
  <div class="wrap">
    <a class="brand" href="/" aria-label="edge or luck, home">
      <svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="12.5" fill="none" stroke="currentColor" stroke-width="2.2"/><path d="M8.6 20.4c1.4-1.2 2.9-2.6 4.3-3.8.9.7 1.9 1.6 2.9 2.3 2.3-2.6 4.6-5.6 6.8-8.4" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      edge or luck
    </a>
    <nav class="nav" aria-label="Site">
      <a class="opt" href="/">Check your trades</a>
      <a class="opt" href="/guides" aria-current="page">Guides</a>
      <a class="opt" href="/story">Story</a>
      <a class="opt" href="/play">Play</a>
      <a class="opt" href="/how">How it checks</a>
      <a href="https://github.com/Sakke-code/edge-or-luck" rel="noopener">GitHub</a>
      <button class="theme" id="theme" type="button" aria-label="Switch to blueprint (dark) look">
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.8a6.2 6.2 0 0 1 0 12.4z" fill="currentColor"/></svg>
        <span id="themeLabel">Blueprint</span>
      </button>
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer>
  <div class="wrap">
    <div><p><b>Not financial advice.</b> edge or luck judges past results. It doesn't tell you what to trade or how much.</p></div>
    <div><p>Open source under the MIT licence. <a href="https://github.com/Sakke-code/edge-or-luck" rel="noopener">Source on GitHub</a></p></div>
  </div>
</footer>
<script src="/assets/site.js"></script>
<script src="/assets/guides.js"></script>
</body>
</html>
"""


def hub_page(guides: List[Guide]) -> str:
    e = html.escape
    groups = []
    n = 0
    for key, label in GROUPS.items():
        cards = []
        for g in (g for g in guides if g.group == key):
            n += 1
            cards.append(f'<li><a class="card" href="/guides/{e(g.slug)}"><span class="num">{n}</span>'
                         f'<span class="card-body"><span class="card-title">{e(g.title)}</span>'
                         f'<span class="card-sum">{e(g.summary)}</span>'
                         f'<span class="card-min">{g.minutes} min read</span></span></a></li>')
        if cards:
            groups.append(f'<section class="group wrap" aria-labelledby="g-{key}"><h2 id="g-{key}">{e(label)}</h2>'
                          f'<ul class="cards">{"".join(cards)}</ul></section>')
    body = f"""<section class="intro wrap" aria-labelledby="title">
  <div>
    <h1 id="title">Mini guides</h1>
    <p class="lede">Short reads on testing a trading idea without fooling yourself: how backtests lie, why the judge is strict, and what to do with its answer. A few minutes each, no maths needed.</p>
    <p class="watch"><a href="/story">Rather watch? Flip explains it all in 45 seconds →</a></p>
    <p class="for-ai">For AI assistants: every guide is also plain Markdown (add <code>.md</code> to its address), and <a href="/api/guides">/api/guides</a> lists them.</p>
  </div>
  <div class="intro-pet" aria-hidden="true">
    <div class="bubble"><span>Pick one. I'll keep it short, promise.</span><svg class="tail" viewBox="0 0 16 14"><path d="M0 1.5C6 3 10.5 7 15 13 9.5 11 5 10 0 10.5"/></svg></div>
    <div class="pet big-pet" data-mood="happy"></div>
  </div>
</section>
{"".join(groups)}"""
    return _shell("Mini guides · edge or luck",
                  "Short guides on testing a trading strategy honestly: backtests that lie, tuning, versions tried, "
                  "riding the market, losing streaks.", body)


def guide_page(guide: Guide, guides: List[Guide]) -> str:
    e = html.escape
    content, toc = render(guide.markdown)
    i = next(k for k, g in enumerate(guides) if g.slug == guide.slug)
    nxt = guides[(i + 1) % len(guides)]
    contents = "".join(f'<li><a href="#{a}">{e(t)}</a></li>' for a, t in toc)
    body = f"""<article class="guide wrap">
  <p class="crumbs"><a href="/guides">All guides</a> · {e(GROUPS.get(guide.group, guide.group))}</p>
  <header class="guide-head">
    <div>
      <h1>{e(guide.title)}</h1>
      <p class="lede">{e(guide.summary)}</p>
      <p class="meta">{guide.minutes} min read · <a href="/guides/{e(guide.slug)}.md">Markdown for your AI</a></p>
    </div>
    <div class="intro-pet" aria-hidden="true">
      <div class="bubble"><span>{e(guide.flip)}</span><svg class="tail" viewBox="0 0 16 14"><path d="M0 1.5C6 3 10.5 7 15 13 9.5 11 5 10 0 10.5"/></svg></div>
      <div class="pet big-pet" data-mood="{e(guide.mood)}"></div>
    </div>
  </header>
  <div class="guide-grid">
    <nav class="contents" aria-label="In this guide"><b>In this guide</b><ol>{contents}</ol></nav>
    <div class="prose">
{content}
    </div>
  </div>
  <a class="next" href="/guides/{e(nxt.slug)}"><span class="next-label">Next guide</span><span class="next-title">{e(nxt.title)}</span><span class="next-sum">{e(nxt.summary)}</span></a>
</article>"""
    return _shell(f"{guide.title} · edge or luck", guide.summary, body, alternate=f"/guides/{guide.slug}.md")


__all__ = ["Guide", "load", "find", "render", "hub_page", "guide_page", "numbers", "share", "GROUPS"]
