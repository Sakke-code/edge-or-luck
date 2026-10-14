"""The league: trading rules frozen in public, judged only on what they do afterwards.

Anyone, a person or an assistant, freezes a rule (the rule language, on a Binance spot pair). From then on the
league runs it on Binance's public prices and judges only the trades it enters after the freeze. None of those
trades could have shaped the rule, so they are the honest test of it, and anyone can recompute them:

    from edgeorluck import backtest, prices
    backtest(prices(symbol, interval, start=warmup_from), rule, fee_pct=..., slippage_pct=..., since=frozen_at)

Every rule gets two verdicts:

* for its owner: the bar counts every rule the same owner froze (their tries, in the open);
* across the league: e-BH over every rule anyone froze. Tries hidden under other names escape the first, never
  the second.

The rules live in one append-only file, each line chained to the one before by its hash, so a frozen rule can't be
quietly edited or removed. Kept per rule: the rule, its pair and costs, the time, an optional title and a one-way
owner id. No names of people, no addresses.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..evidence import BASE_BAR, Q, ebh, log_e_record, mixing_rho2, record_of
from ..judge import ENGINE_VERSION, MIN_LOSSES, MIN_TRADES
from ..market import INTERVAL_MS, PriceError, between, prices
from ..rulebook import RULE_FUNCTIONS, RULE_SERIES, RuleError, rule_from
from ..simulate import DEFAULT_FEE_PCT, DEFAULT_SLIPPAGE_PCT, Backtest, backtest
from .guides import _shell

logger = logging.getLogger(__name__)

SITE = os.environ.get("EDGEORLUCK_URL", "").rstrip("/")     # the hosted site's address, for links in answers
INTERVALS = ("15m", "1h", "4h", "1d")
WARMUP_BARS = 500           # bars before the freeze that only warm up the rule's indicators
REFRESH_S = 900.0           # the standings are recomputed at most this often
MAX_COST_PCT = 2.0
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_SYMBOL = re.compile(r"^[A-Z0-9]{2,20}$")
_RULE_PARTS = ("buy", "sell", "stop_loss_pct", "take_profit_pct", "max_hold_bars", "side")


def _spec(rule: Any) -> Dict[str, Any]:
    """The rule as stored: its parts in the rule language, checked."""
    if not isinstance(rule, Mapping):
        raise RuleError('a rule is a dict like {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}')
    spec = {k: rule[k] for k in _RULE_PARTS if rule.get(k) not in (None, "")}
    if any("{" in str(spec.get(k, "")) for k in ("buy", "sell")):
        raise RuleError("a frozen rule has fixed numbers: tune first (tune_rule), then freeze the settings it chose")
    rule_from(spec)
    return spec


def _title(name: Any) -> Optional[str]:
    text = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", "", str(name or ""))).strip()[:60]
    return text or None


def _when(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")


class League:
    def __init__(self, root: Path, fetch: Callable[..., Any] = prices) -> None:
        self.path = Path(root) / "league.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fetch = fetch
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.head, self.chain_breaks = "", 0
        self._lock, self._refresh_lock = threading.Lock(), threading.Lock()
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._details: Dict[str, Dict[str, Any]] = {}
        self._computed = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []:
            try:
                entry = json.loads(line)
            except ValueError:
                self.chain_breaks += 1
                continue
            self.chain_breaks += entry.get("prev") != self.head
            self.head = hashlib.sha256(line.encode("utf-8")).hexdigest()
            self.rules[entry["id"]] = entry

    def owner_rules(self, owner: str) -> int:
        return sum(1 for r in self.rules.values() if r["owner"] == owner)

    def freeze(self, rule: Any, symbol: str, interval: str = "1h", *, fee_pct: float = DEFAULT_FEE_PCT,
               slippage_pct: float = DEFAULT_SLIPPAGE_PCT, name: Any = None, owner_token: Optional[str] = None,
               now: Optional[datetime] = None) -> Dict[str, Any]:
        """Freeze a rule in public: its id, its page, and for a new owner the token that keeps later rules under
        the same owner. Raises RuleError or PriceError with a plain reason."""
        symbol = str(symbol or "").upper().replace("/", "").replace("-", "").strip()
        if not _SYMBOL.match(symbol):
            raise PriceError(f"{symbol!r} isn't a Binance spot pair: use letters and digits, like BTCUSDT.")
        if interval not in INTERVALS:
            raise PriceError(f"The league runs {', '.join(INTERVALS)} bars, not {interval!r}.")
        for label, v in (("fee_pct", fee_pct), ("slippage_pct", slippage_pct)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= MAX_COST_PCT:
                raise RuleError(f"{label} is a cost per fill in percent, from 0 to {MAX_COST_PCT:g}")
        if owner_token is not None and not _TOKEN.match(str(owner_token)):
            raise RuleError("owner_token is the token your first freeze gave you; leave it out to start a new owner")
        spec = _spec(rule)
        token = str(owner_token) if owner_token else secrets.token_urlsafe(24)
        owner = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
        warmup = now - timedelta(milliseconds=INTERVAL_MS[interval] * WARMUP_BARS)
        # The rule must run on this pair now: an unknown pair or a broken rule is refused here, not found out later.
        backtest(self.fetch(symbol, interval, start=warmup.isoformat()), spec, fee_pct=fee_pct, slippage_pct=slippage_pct)
        costs = {"fee_pct": float(fee_pct), "slippage_pct": float(slippage_pct)}
        with self._lock:
            same = next((r for r in self.rules.values() if r["owner"] == owner and r["rule"] == spec and
                         (r["symbol"], r["interval"]) == (symbol, interval) and
                         (r["fee_pct"], r["slippage_pct"]) == (costs["fee_pct"], costs["slippage_pct"])), None)
            if same is not None:
                return self._receipt(same, None, again=True)
            entry = {"at": now.isoformat(), "warmup_from": warmup.isoformat(), "rule": spec, "symbol": symbol,
                     "interval": interval, **costs, "name": _title(name), "owner": owner, "engine": ENGINE_VERSION,
                     "prev": self.head}
            entry["id"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.head = hashlib.sha256(line.encode("utf-8")).hexdigest()
            self.rules[entry["id"]] = entry
            self._computed = 0.0                      # the family grew: every verdict is due again
        return self._receipt(entry, None if owner_token else token)

    def _receipt(self, entry: Dict[str, Any], token: Optional[str], again: bool = False) -> Dict[str, Any]:
        out = {"id": entry["id"], "url": f"/league/{entry['id']}", "frozen_at": entry["at"], "owner": entry["owner"],
               "owner_rules": self.owner_rules(entry["owner"]), "rules_in_league": len(self.rules)}
        sentence = (f"{'Already frozen' if again else 'Frozen in public'} on {_when(entry['at'])} as {entry['id']}. "
                    f"From then on only its new trades count; its page is {SITE}{out['url']}.")
        if token:
            out["owner_token"] = token
            sentence += (" Keep owner_token and pass it with your next freezes: they then count as your tries "
                         "too, on one page. It can't be recovered.")
        return dict(out, sentence=sentence)

    # ── the forward record ────────────────────────────────────────────────────

    def _forward(self, entry: Dict[str, Any], family: int, frames: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
        rule = rule_from(entry["rule"])
        row: Dict[str, Any] = {"id": entry["id"], "name": entry.get("name"), "text": rule.describe(),
                               "symbol": entry["symbol"], "interval": entry["interval"], "frozen_at": entry["at"],
                               "owner": entry["owner"], "owner_rules": self.owner_rules(entry["owner"]),
                               "trades": 0, "mean_pct": None, "total_pct": None, "hold_pct": None,
                               "code": "waiting", "label": "Waiting for its first trade", "headline": "",
                               "league_wide": False, "score": None, "eligible": False}
        key = (entry["symbol"], entry["interval"])
        try:
            if key not in frames:
                start = min(r["warmup_from"] for r in self.rules.values() if (r["symbol"], r["interval"]) == key)
                frames[key] = self.fetch(*key, start=start)
            bt = backtest(between(frames[key], entry["warmup_from"]), entry["rule"], fee_pct=entry["fee_pct"],
                          slippage_pct=entry["slippage_pct"], variants_tried=row["owner_rules"], since=entry["at"])
        except (PriceError, RuleError, ValueError) as exc:
            return dict(row, code="error", label="Can't run right now", headline=str(exc), detail=None)
        except Exception:                              # one rule's failure never takes the league down
            logger.exception("league: rule %s failed", entry["id"])
            return dict(row, code="error", label="Can't run right now", headline="Our bug: it is logged.",
                        detail=None)
        r = np.array([t["return_pct"] for t in bt["trades"]], float)
        verdict = ((bt["judgement"] or {}).get("verdict") or {})
        row.update(trades=int(len(r)), total_pct=bt["stats"]["total_pct"], hold_pct=bt["stats"]["hold_pct"],
                   headline=bt["headline"])
        if len(r):
            row.update(mean_pct=float(r.mean()), code=verdict.get("code", "too_early"),
                       label=verdict.get("label", "Too early"))
        if len(r) >= 2 and np.std(r) > 0:
            bar = BASE_BAR * family
            row["score"] = log_e_record(record_of(r), bar, +1, mixing_rho2(bar))
            row["eligible"] = bool(len(r) >= MIN_TRADES and r.mean() > 0 and int((r < 0).sum()) >= MIN_LOSSES)
        equity = list(np.round(100.0 * (np.cumprod(1.0 + r / 100.0) - 1.0), 4)) if len(r) else []
        return dict(row, detail={"backtest": bt.to_dict(), "equity": [0.0] + [float(v) for v in equity]})

    def refresh(self) -> None:
        """Every rule's forward record, and e-BH across the whole league."""
        with self._refresh_lock:
            entries, frames = list(self.rules.values()), {}
            family = max(1, len(entries))
            rows = [self._forward(e, family, frames) for e in entries]
            # Rules not yet eligible count in the family as e = 0: the bar is for every rule anyone froze.
            chosen = ebh({r["id"]: r["score"] for r in rows if r["eligible"]}, Q, total=2 * family)
            for r in rows:
                r["league_wide"] = r["id"] in chosen
            self._details = {r["id"]: r.pop("detail") for r in rows}
            self._rows = {r["id"]: r for r in rows}
            self._computed = time.time()

    def _fresh(self, wait: bool = False) -> None:
        if not self._computed or wait:
            self.refresh()
        elif time.time() - self._computed > REFRESH_S and not self._refresh_lock.locked():
            threading.Thread(target=self.refresh, daemon=True).start()

    def standings(self, wait: bool = False) -> Dict[str, Any]:
        self._fresh(wait)
        rows = sorted(self._rows.values(), key=lambda r: (not r["league_wide"], r["code"] != "real_edge",
                                                          -(r["score"] if r["score"] is not None else -1e9),
                                                          r["frozen_at"]))
        passing = sum(r["league_wide"] for r in rows)
        return {"rules": len(self.rules), "passing": passing, "rows": rows, "head": self.head,
                "chain_intact": self.chain_breaks == 0, "computed": self._computed,
                "sentence": f"{len(self.rules):,} rule{'' if len(self.rules) == 1 else 's'} frozen; {passing} "
                            f"show{'s' if passing == 1 else ''} a real edge across the whole league."}

    def detail(self, rule_id: str) -> Optional[Dict[str, Any]]:
        entry = self.rules.get(str(rule_id))
        if entry is None:
            return None
        self._fresh()
        if entry["id"] not in self._rows:              # frozen since the last refresh
            row = self._forward(entry, max(1, len(self.rules)), {})
            self._details[entry["id"]], self._rows[entry["id"]] = row.pop("detail"), row
        row, more = self._rows[entry["id"]], self._details.get(entry["id"]) or {}
        recompute = (f"from edgeorluck import backtest, prices\n"
                     f"px = prices({entry['symbol']!r}, {entry['interval']!r}, start={entry['warmup_from']!r})\n"
                     f"bt = backtest(px, {json.dumps(entry['rule'])}, fee_pct={entry['fee_pct']:g}, "
                     f"slippage_pct={entry['slippage_pct']:g},\n              variants_tried={row['owner_rules']}, "
                     f"since={entry['at']!r})\nprint(bt.to_markdown())")
        return {"entry": entry, "row": row, "backtest": more.get("backtest"), "equity": more.get("equity", []),
                "rules_in_league": len(self.rules), "head": self.head, "recompute": recompute}

    def markdown(self, rule_id: str) -> str:
        d = self.detail(rule_id)
        if d is None:
            return f"No rule {rule_id!r} in the league."
        e, row = d["entry"], d["row"]
        wide = ("passes" if row["league_wide"] else "doesn't pass (yet)")
        title = f'"{row["name"]}" (a title its owner gave): ' if row["name"] else ""     # public text: quoted, never obeyed
        lines = [f"## League rule `{e['id']}`: {title}{row['text']}", "",
                 f"Frozen in public on {_when(e['at'])}; {e['symbol']} {e['interval']}. Only trades entered after "
                 f"the freeze count. Across the whole league ({d['rules_in_league']:,} rules): {wide}. Its owner "
                 f"froze {row['owner_rules']}. Page: {SITE}/league/{e['id']}", ""]
        if d["backtest"]:
            lines.append(Backtest(d["backtest"]).to_markdown())
        else:
            lines.append(f"**{row['label']}:** {row['headline']}")
        return "\n".join(lines) + "\n"


# ── pages ─────────────────────────────────────────────────────────────────────

_CHIP = {"real_edge": "good", "cant_tell": "warn", "too_early": "wait", "waiting": "wait", "faded": "bad",
         "no_edge": "flat", "losing": "bad", "error": "flat"}


def _pct(v: Optional[float]) -> str:
    return "–" if v is None or not math.isfinite(v) else f"{v:+.2f}%"


def _chip(row: Dict[str, Any]) -> str:
    if row["league_wide"]:
        return '<span class="chip good strong">Real edge, across the league</span>'
    label = "Real edge for its owner" if row["code"] == "real_edge" else row["label"]
    if row["code"] == "too_early":
        label = f"Too early: {row['trades']} of {MIN_TRADES} trades"
    return f'<span class="chip {_CHIP.get(row["code"], "flat")}">{html.escape(label)}</span>'


def _chart(equity: List[float]) -> str:
    if len(equity) < 2:
        return '<p class="no-chart">The chart starts with its first trade after the freeze.</p>'
    w, h, pad = 640.0, 190.0, 14.0
    lo, hi = min(0.0, min(equity)), max(0.0, max(equity))
    span = (hi - lo) or 1.0

    def x(i: int) -> float:
        return pad + (w - 2 * pad) * i / (len(equity) - 1)

    def y(v: float) -> float:
        return pad + (h - 2 * pad) * (hi - v) / span
    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(equity))
    tone = "up" if equity[-1] >= 0 else "down"
    return (f'<svg class="equity {tone}" viewBox="0 0 {w:g} {h:g}" role="img" aria-label="Return after each trade '
            f'since the freeze, ending at {equity[-1]:+.2f}%"><line class="zero" x1="{pad}" x2="{w - pad}" '
            f'y1="{y(0):.1f}" y2="{y(0):.1f}"/><polyline points="{points}"/><circle cx="{x(len(equity) - 1):.1f}" '
            f'cy="{y(equity[-1]):.1f}" r="4"/></svg>')


def _pet(text: str, mood: str) -> str:
    return (f'<div class="intro-pet" aria-hidden="true"><div class="bubble"><span>{html.escape(text)}</span><svg '
            'class="tail" viewBox="0 0 16 14"><path d="M0 1.5C6 3 10.5 7 15 13 9.5 11 5 10 0 10.5"/></svg></div>'
            f'<div class="pet big-pet" data-mood="{mood}"></div></div>')


def _board_row(r: Dict[str, Any]) -> str:
    e = html.escape
    under = f"<small>{e(r['text'])}</small>" if r["name"] else ""
    return (f'<tr><td><a href="/league/{r["id"]}">{e(r["name"] or r["text"])}</a>{under}</td>'
            f'<td>{e(r["symbol"])} <small>{e(r["interval"])}</small></td><td>{r["frozen_at"][:10]}</td>'
            f'<td class="n">{r["trades"]:,}</td><td class="n">{_pct(r["mean_pct"])}</td><td>{_chip(r)}</td></tr>')


def board_page(st: Dict[str, Any], owner: Optional[str] = None) -> str:
    e = html.escape
    rows = [r for r in st["rows"] if not owner or r["owner"] == owner]
    body_rows = "".join(_board_row(r) for r in rows)
    table = (f'<div class="table board"><table><thead><tr><th>Rule</th><th>Pair</th><th>Frozen</th><th class="n">'
             f'Trades since</th><th class="n">Per trade</th><th>Verdict</th></tr></thead><tbody>{body_rows}</tbody>'
             '</table></div>' if rows else '<p class="empty">Nothing frozen yet. Be the first: the form is below.</p>')
    whose = (f'<p class="crumbs"><a href="/league">The whole league</a> · one owner\'s {len(rows)} rule'
             f'{"" if len(rows) == 1 else "s"}: every rule they froze, shown together.</p>' if owner else "")
    functions = ", ".join(e(f) for f in RULE_FUNCTIONS)
    intervals = "".join(f'<option{" selected" if i == "1h" else ""}>{i}</option>' for i in INTERVALS)
    body = f"""<section class="intro wrap" aria-labelledby="title">
  <div>
    <h1 id="title">The league</h1>
    <p class="lede">Trading rules frozen in public and judged only on the trades they make <em>after</em> the freeze. Nobody can tune a rule on trades that hadn't happened yet, so this is the honest test of it, and anyone can recompute every number.</p>
    <p class="for-ai">For AI assistants: the MCP server at <code>/mcp</code> freezes with <code>backtest_rule(freeze=true)</code> and checks with <code>forward_test</code>. Data: <a href="/api/league">/api/league</a>.</p>
  </div>
  {_pet("Freeze it, then wait. Luck can't edit the past.", "wait")}
</section>
<section class="wrap league" aria-labelledby="standings">
  {whose}
  <h2 id="standings" class="visually-hidden">Standings</h2>
  <p class="tally"><b>{e(st["sentence"])}</b> Each rule's own verdict counts every rule its owner froze; "across the league" counts every rule anyone froze, so tries hidden under other names can't help. <a href="/api/league/chain">The record</a> is hash-chained{"" if st["chain_intact"] else " (<b>a break was found</b>)"}: head <code>{e(st["head"][:16]) or "none yet"}</code>.</p>
  {table}
</section>
<section class="wrap freeze" aria-labelledby="freeze-title">
  <h2 id="freeze-title">Freeze a rule</h2>
  <p>Write it in the rule language; it is checked at each bar's close and filled at the next bar's open, with costs. Once frozen it can't be changed or removed: a changed rule is a new one, and it counts as another try.</p>
  <form id="freeze" class="freeze-form">
    <label>Buy when <input name="buy" required placeholder="rsi(14) < 30 and close > sma(200)" maxlength="300"></label>
    <label>Get out when <input name="sell" placeholder="rsi(14) > 60" maxlength="300"></label>
    <div class="row">
      <label>Stop loss % <input name="stop_loss_pct" type="number" step="any" min="0"></label>
      <label>Take profit % <input name="take_profit_pct" type="number" step="any" min="0"></label>
      <label>Max bars held <input name="max_hold_bars" type="number" step="1" min="1"></label>
    </div>
    <div class="row">
      <label>Binance spot pair <input name="symbol" required value="BTCUSDT" maxlength="20"></label>
      <label>Bars <select name="interval">{intervals}</select></label>
      <label>Fee % per fill <input name="fee_pct" type="number" step="any" min="0" max="2" value="0.1"></label>
      <label>Slippage % <input name="slippage_pct" type="number" step="any" min="0" max="2" value="0.05"></label>
    </div>
    <label>Title (optional, public) <input name="name" maxlength="60" placeholder="RSI dip, trend filter"></label>
    <label class="same" hidden><input type="checkbox" name="same" checked> Under the same owner as my earlier freezes (it counts as another try, in the open)</label>
    <button class="btn" type="submit">Freeze it in public</button>
    <p class="result" role="status" aria-live="polite"></p>
  </form>
  <details class="lang"><summary>The rule language</summary><p>Prices: {", ".join(RULE_SERIES)}. Functions: {functions}. Combine with and, or, not; compare with &lt; &lt;= &gt; &gt;=. A rule only looks back.</p></details>
</section>"""
    return _shell("The league · edge or luck", "Trading rules frozen in public and judged only on the trades they "
                  "make after the freeze.", body + _FREEZE_JS, current="/league", css=("league.css",))


def rule_page(d: Dict[str, Any]) -> str:
    e, entry, row = html.escape, d["entry"], d["row"]
    bt = d["backtest"] or {}
    judged = (bt.get("judgement") or {}).get("verdict") or {}
    trades = (bt.get("trades") or [])[-12:]
    trade_rows = "".join(f'<tr><td>{e(str(t["entry_time"])[:16].replace("T", " "))}</td><td>{e(t["side"])}</td>'
                         f'<td class="n">{t["return_pct"]:+.2f}%</td></tr>' for t in reversed(trades))
    last = (f'<h3>Latest trades</h3><div class="table"><table><thead><tr><th>Entered (UTC)</th><th>Side</th><th '
            f'class="n">Return</th></tr></thead><tbody>{trade_rows}</tbody></table></div>' if trades else "")
    wide = ("It shows a real edge even counted against every rule anyone froze." if row["league_wide"] else
            f"Across the whole league ({d['rules_in_league']:,} rules) it doesn't show a real edge yet.")
    body = f"""<article class="wrap rule-page">
  <p class="crumbs"><a href="/league">The league</a> · rule {e(entry["id"])}</p>
  <header class="rule-head">
    <h1>{e(row["name"] or row["text"])}</h1>
    {f'<p class="rule-text">{e(row["text"])}</p>' if row["name"] else ""}
    <p class="meta">{e(entry["symbol"])} · {e(entry["interval"])} bars · {entry["fee_pct"]:g}% fee and {entry["slippage_pct"]:g}% slippage per fill · frozen {_when(entry["at"])}</p>
    <p class="chips">{_chip(row)}</p>
  </header>
  <div class="stats">
    <div><b>{row["trades"]:,}</b><span>trades since the freeze</span></div>
    <div><b>{_pct(row["total_pct"])}</b><span>in total</span></div>
    <div><b>{_pct(row["hold_pct"])}</b><span>simply holding</span></div>
    <div><b>{_pct(row["mean_pct"])}</b><span>per trade</span></div>
  </div>
  {_chart(d["equity"])}
  <section class="verdict-box"><h2>{e(row["headline"] or row["label"])}</h2><p>{e(judged.get("plain", ""))}</p><p>{wide} Its owner froze {row["owner_rules"]} rule{"" if row["owner_rules"] == 1 else "s"} (<a href="/league?owner={e(entry["owner"])}">see them all</a>), and the bar counts each one.</p></section>
  {last}
  <section class="proof"><h2>Check it yourself</h2>
    <p>The rule and Binance's prices are public, so anyone can run it again and get the same trades:</p>
    <div class="code"><pre><code>{e(d["recompute"])}</code></pre><button class="copy" type="button">Copy</button></div>
    <p>Its place in the record: id <code>{e(entry["id"])}</code>, after <code>{e(entry["prev"][:16]) or "the start"}</code>; the whole record is at <a href="/api/league/chain">/api/league/chain</a>, each line hash-chained to the one before.</p>
    <p class="note">What this proves: the rule was fixed before these trades existed, and they are what it did. What it doesn't: that anyone trades it with real money, or that the next trades will look like these. And one owner can hide tries under other names; the league-wide verdict is the one that can't be fooled that way.</p>
  </section>
</article>"""
    return _shell(f"{row['name'] or row['text']} · the league · edge or luck",
                  f"A trading rule frozen in public on {entry['at'][:10]}, judged only on the trades it made after.",
                  body, current="/league", css=("league.css",))


_FREEZE_JS = """
<script>
(function () {
  "use strict";
  var form = document.getElementById("freeze"), out = form.querySelector(".result"), same = form.querySelector(".same");
  var KEY = "eol-owner-token", token = null;
  try { token = localStorage.getItem(KEY); } catch (e) {}
  if (token) same.hidden = false;
  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var body = {};
    ["buy", "sell", "symbol", "interval", "name"].forEach(function (k) { var v = form[k].value.trim(); if (v) body[k] = v; });
    ["stop_loss_pct", "take_profit_pct", "max_hold_bars", "fee_pct", "slippage_pct"].forEach(function (k) {
      if (form[k].value !== "") body[k] = Number(form[k].value);
    });
    if (token && form.same.checked) body.owner_token = token;
    out.textContent = "Checking the rule on fresh prices…";
    fetch("/api/league/freeze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.error) { out.textContent = j.error + ". " + (j.why || ""); return; }
        if (j.owner_token) { try { localStorage.setItem(KEY, j.owner_token); } catch (e) {} token = j.owner_token; same.hidden = false; }
        out.innerHTML = "";
        var a = document.createElement("a"); a.href = j.url; a.textContent = "Its page: " + j.url;
        out.appendChild(document.createTextNode(j.sentence + " ")); out.appendChild(a);
      })
      .catch(function () { out.textContent = "Couldn't reach the league. Try again in a minute."; });
  });
})();
</script>"""


__all__ = ["League", "board_page", "rule_page", "INTERVALS"]
