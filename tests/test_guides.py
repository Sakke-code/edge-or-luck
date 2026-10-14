"""The mini guides can't drift from the code: every rule they show compiles and
fires on real-looking prices, every Python snippet parses, every number comes
from the validation run, and every link goes somewhere."""
import ast
import re

import numpy as np
import pandas as pd
import pytest

from edgeorluck.rulebook import RULE_FUNCTIONS, RULE_SERIES, compile_rule
from edgeorluck.web import guides
from edgeorluck.web.app import create_app
from edgeorluck.web.ledger import Ledger

GUIDES = guides.load()
MOODS = {"idle", "excited", "flipping", "happy", "shrug", "meh", "worried", "sleepy", "wait", "confused"}
WORDS = set(RULE_SERIES) | set(RULE_FUNCTIONS) | {"and", "or", "not"}


@pytest.fixture()
def client(tmp_path):
    return create_app(Ledger(root=tmp_path)).test_client()


def ohlcv(n=20000, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.004, n))
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, close) * (1 + wick),
                         "low": np.minimum(opens, close) * (1 - wick), "close": close,
                         "volume": rng.lognormal(0, 0.6, n)},
                        index=pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"))


def rules_in(text):
    """Code spans and double-quoted strings written in the rule language."""
    found = set(re.findall(r"`([^`]+)`", text)) | set(re.findall(r'"([^"{}]+)"', text))
    out = []
    for s in found:
        s = s.strip()
        if s.startswith("add "):
            s = s[4:]
        if s.startswith("and "):
            s = s[4:]
        idents = set(re.findall(r"[A-Za-z_]+", s))
        if idents and idents <= WORDS and re.search(r"[<>]|crosses_", s):
            out.append(s)
    return sorted(set(out))


def test_there_are_guides_with_full_headers():
    assert len(GUIDES) >= 8
    assert len({g.slug for g in GUIDES}) == len(GUIDES)
    for g in GUIDES:
        assert g.group in guides.GROUPS and g.mood in MOODS, g.slug
        assert 1 <= g.minutes <= 6 and len(g.summary) < 220, g.slug
        assert "{{" not in g.markdown and "}}" not in g.markdown.replace("{{", ""), g.slug


@pytest.mark.parametrize("guide", GUIDES, ids=lambda g: g.slug)
def test_every_rule_in_a_guide_compiles_and_fires(guide):
    px = ohlcv()
    rules = rules_in(guide.markdown)
    for text in rules:
        fired = compile_rule(text)(px)
        assert fired.any(), f"{guide.slug}: {text!r} is never true on 20,000 bars"


def test_the_rule_guide_shows_rules_and_every_function():
    guide = next(g for g in GUIDES if g.slug == "write-your-idea-as-a-rule")
    assert len(rules_in(guide.markdown)) >= 10
    for name in RULE_FUNCTIONS:
        assert f"`{name}(" in guide.markdown, name


def test_the_rule_language_tool_examples_fire_too():
    from edgeorluck.mcp_server import rule_language
    rules = re.findall(r'(?:buy|sell) "([^"]+)"', rule_language())
    assert len(rules) >= 5
    px = ohlcv()
    for text in rules:
        assert compile_rule(text)(px).any(), text


@pytest.mark.parametrize("guide", GUIDES, ids=lambda g: g.slug)
def test_python_snippets_parse(guide):
    for lang, body in re.findall(r"```(\w*)\n(.*?)```", guide.markdown, re.S):
        if lang == "python":
            ast.parse(body)


def test_links_go_somewhere(client):
    slugs = {g.slug for g in GUIDES}
    for g in GUIDES:
        for url in re.findall(r"\]\((/[^)]*)\)", g.markdown):
            path = url.split("#")[0]
            if path.startswith("/guides/"):
                assert path[len("/guides/"):] in slugs, (g.slug, url)
            else:
                assert client.get(path).status_code == 200, (g.slug, url)


def test_the_pages_are_served(client):
    hub = client.get("/guides")
    assert hub.status_code == 200 and hub.mimetype == "text/html"
    text = hub.get_data(as_text=True)
    for g in GUIDES:
        assert f'href="/guides/{g.slug}"' in text
    listing = client.get("/api/guides").get_json()
    assert [x["slug"] for x in listing] == [g.slug for g in GUIDES]
    for g in GUIDES:
        page = client.get(f"/guides/{g.slug}")
        body = page.get_data(as_text=True)
        assert page.status_code == 200 and "<h1>" in body and "{{" not in body and "**" not in body
        assert f'href="/guides/{g.slug}.md"' in body
        md = client.get(f"/guides/{g.slug}.md")
        assert md.status_code == 200 and md.mimetype == "text/markdown"
        assert md.get_data(as_text=True).startswith(f"# {g.title}\n")
    assert client.get("/guides/nope").status_code == 404
    assert client.get("/guides/nope.md").status_code == 404


def test_flips_story_is_served_and_linked(client):
    page = client.get("/story")
    text = page.get_data(as_text=True)
    assert page.status_code == 200 and "Edge or luck, in 45 seconds" in text and "EOLStory" in text
    assert "/api/validation" in text            # the false-alarm rate it quotes is the measured one
    for path in ("/", "/how", "/guides", f"/guides/{GUIDES[0].slug}"):
        assert 'href="/story"' in client.get(path).get_data(as_text=True), path


def test_the_flip_game_is_served_and_linked(client):
    page = client.get("/play")
    text = page.get_data(as_text=True)
    assert page.status_code == 200 and "Flip Flip" in text and "crypto.getRandomValues" in text
    for path in ("/", "/how", "/story", "/guides", f"/guides/{GUIDES[0].slug}"):
        assert 'href="/play"' in client.get(path).get_data(as_text=True), path


def test_numbers_come_from_the_validation_run():
    values = guides.numbers()
    strict = next(g for g in GUIDES if g.slug == "why-the-judge-is-strict")
    assert values["false_typical"] in strict.markdown and values["power_300"] in strict.markdown
    assert guides.share(0.0015) == "0.15%" and guides.share(0.088) == "8.8%" and guides.share(0.603) == "60%"
    assert guides.share(0.0) == "0%" and guides.share(0.997) == "99.7%" and guides.share(1.0) == "100%"


def test_the_github_copies_are_up_to_date():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("write_guides", root / "tools" / "write_guides.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    want = tool.expected()
    have = {p.name: p.read_text(encoding="utf-8") for p in (root / "docs" / "guides").glob("*.md")}
    assert have == want, "docs/guides is stale: run python tools/write_guides.py"
    for text in want.values():
        assert "{{" not in text and "](/" not in text


def test_render_escapes_and_keeps_links_safe():
    html, toc = guides.render("## A <b> title\n\nText with <script>x</script> and [bad](javascript:alert(1)) "
                              "and [good](/guides) and `a < b`.\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'href="javascript' not in html and '<a href="/guides">good</a>' in html
    assert "<code>a &lt; b</code>" in html and "<ul><li>one</li><li>two</li></ul>" in html
    assert "<th>a</th>" in html and "<td>2</td>" in html
    nested, _ = guides.render("- way in\n- way out:\n  - sell\n  - stop\n- last\n")
    assert nested == "<ul><li>way in</li><li>way out:<ul><li>sell</li><li>stop</li></ul></li><li>last</li></ul>"
    assert toc == [("a-b-title", "A <b> title")]          # plain text: the page escapes it once
    assert '<h2 id="a-b-title">A &lt;b&gt; title</h2>' in html
