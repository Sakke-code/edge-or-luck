"""Popular strategies, judged: well-known trading rules tested honestly on standard markets.

Past prices only, computed once per release by tools/write_strategies.py into data/strategies.json (a handful of
price requests, none per visitor). Every strategy on every market is one try, and each verdict counts all of them:
a catalogue that tests fifty ideas and shows the best would be the best-of-many trap it warns about. There is no
"now" here: no live entries, no alerts, nothing to copy. It shows what the honest test says about ideas people repeat.
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .guides import _shell

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data" / "strategies.json"
END = "2026-09-30"

MARKETS = [
    {"symbol": "SPY", "name": "S&P 500 (SPY)", "start": "2005-01-01", "fee_pct": 0.0, "slippage_pct": 0.05},
    {"symbol": "QQQ", "name": "Nasdaq 100 (QQQ)", "start": "2005-01-01", "fee_pct": 0.0, "slippage_pct": 0.05},
    {"symbol": "GLD", "name": "Gold (GLD)", "start": "2005-01-01", "fee_pct": 0.0, "slippage_pct": 0.05},
    {"symbol": "EURUSD", "name": "EUR/USD", "start": "2005-01-01", "fee_pct": 0.0, "slippage_pct": 0.02},
    {"symbol": "BTCUSDT", "name": "Bitcoin (BTC/USDT)", "start": "2018-01-01", "fee_pct": 0.1, "slippage_pct": 0.05},
]

STRATEGIES = [
    {"slug": "golden-cross", "title": "The golden cross",
     "claim": "Buy when the 50-day average crosses above the 200-day; sell when it crosses back.",
     "rule": {"buy": "crosses_above(sma(50), sma(200))", "sell": "crosses_below(sma(50), sma(200))"},
     "lesson": "Slow averages cross rarely, so even twenty years of daily prices give only a handful of trades: too "
               "few to tell an edge from luck, however good the total looks."},
    {"slug": "200-day-trend", "title": "Above the 200-day average",
     "claim": "Own it only while the price is above its 200-day average.",
     "rule": {"buy": "close > sma(200)", "sell": "close < sma(200)"},
     "lesson": "A trend filter changes when you are in the market more than how much each trade makes. It can skip "
               "the worst falls and still trail simply holding, because it also misses part of every rebound."},
    {"slug": "rsi-30-70", "title": "RSI 30/70",
     "claim": "Buy when the 14-day RSI drops under 30 (oversold); sell above 70 (overbought).",
     "rule": {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"},
     "lesson": "In a rising market the daily RSI seldom reaches 30, so the rule waits a long time and sits out most "
               "of the rise. Few trades again, and a long wait between them."},
    {"slug": "rsi2-dip", "title": "RSI(2) dip in an uptrend",
     "claim": "In an uptrend (above the 200-day average), buy a 2-day RSI under 10; sell when the price closes above "
              "its 5-day average.",
     "rule": {"buy": "rsi(2) < 10 and close > sma(200)", "sell": "close > sma(5)"},
     "lesson": "Short dips in an uptrend are a pattern traders have known for decades. Once a pattern is famous, the "
               "question is whether anything is left after costs, and many trades make costs add up."},
    {"slug": "bollinger-bounce", "title": "Bollinger band bounce",
     "claim": "Buy two standard deviations under the 20-day average; sell back at the average.",
     "rule": {"buy": "close < sma(20) - 2 * std(20)", "sell": "close > sma(20)"},
     "lesson": "Buying far under the average is a bet that prices snap back. In a market that keeps falling they "
               "don't, and those few trades decide the result."},
    {"slug": "macd-cross", "title": "MACD cross",
     "claim": "Buy when the MACD line (12/26) crosses above its 9-day signal line; sell when it crosses below.",
     "rule": {"buy": "crosses_above(ema(12) - ema(26), ema(ema(12) - ema(26), 9))",
              "sell": "crosses_below(ema(12) - ema(26), ema(ema(12) - ema(26), 9))"},
     "lesson": "MACD is a moving-average cross in disguise. It trades often, every trade pays costs, and an edge has "
               "to beat all of them."},
    {"slug": "ema-9-21", "title": "EMA 9/21 cross",
     "claim": "Buy when the 9-day EMA crosses above the 21-day EMA; sell when it crosses below.",
     "rule": {"buy": "crosses_above(ema(9), ema(21))", "sell": "crosses_below(ema(9), ema(21))"},
     "lesson": "Fast crosses trade often. Each extra trade is another cost and another chance for luck, so a big "
               "total on one market says little until the trades clear the bar."},
    {"slug": "breakout-20", "title": "20-day breakout",
     "claim": "Buy a close above the highest close of the last 20 days; sell under the lowest of the last 10.",
     "rule": {"buy": "close > prev(highest(20))", "sell": "close < prev(lowest(10))"},
     "lesson": "Breakouts are the classic trend-following entry: they win big rarely and lose small often, so the "
               "result hangs on a few trades, and a short record says little."},
    {"slug": "buy-the-dip", "title": "Buy a 5% dip",
     "claim": "Buy after a 5% drop in five days; sell ten days later.",
     "rule": {"buy": "pct(5) < -5", "max_hold_bars": 10},
     "lesson": "Buying a drop works until the drop keeps going. A few large losses decide the result, which is why "
               "the judge raises the bar for records whose losses are rarer but bigger."},
    {"slug": "under-the-average", "title": "5% under the 20-day average",
     "claim": "Buy when the price is 5% under its 20-day average; sell back at the average.",
     "rule": {"buy": "close < sma(20) * 0.95", "sell": "close > sma(20)"},
     "lesson": "A rule that waits for big gaps trades rarely in calm markets and too early in a crash: it has few "
               "trades to judge and its worst ones come together."},
]


def compute(end: str = END) -> Dict[str, Any]:
    """Every strategy on every market, judged with the whole catalogue as its family (needs internet)."""
    from ..judge import ENGINE_VERSION
    from ..market import prices
    from ..rulebook import rule_from
    from ..simulate import backtest
    tries = len(STRATEGIES) * len(MARKETS)
    frames = {m["symbol"]: prices(m["symbol"], "1d", start=m["start"], end=end) for m in MARKETS}
    out: List[Dict[str, Any]] = []
    for s in STRATEGIES:
        results = []
        for m in MARKETS:
            bt = backtest(frames[m["symbol"]], s["rule"], fee_pct=m["fee_pct"], slippage_pct=m["slippage_pct"],
                          variants_tried=tries)
            verdict = (bt["judgement"] or {}).get("verdict") or {}
            # The same trades judged as if this were the only rule ever tested: is it the counting that sinks it?
            alone = backtest(frames[m["symbol"]], s["rule"], fee_pct=m["fee_pct"], slippage_pct=m["slippage_pct"],
                             variants_tried=1)["judgement"] or {}
            e_alone = ((((alone.get("strategies") or [{}])[0].get("numbers") or {}).get("evidence") or {})
                       .get("e_value") or 0.0)
            r = np.array([t["return_pct"] for t in bt["trades"]], float)
            curve = 100.0 * (np.cumprod(1.0 + r / 100.0) - 1.0) if len(r) else np.zeros(0)
            keep = np.unique(np.linspace(0, len(curve) - 1, min(len(curve), 60)).astype(int)) if len(curve) else []
            card = ((bt["judgement"] or {}).get("strategies") or [{}])[0]
            nums = card.get("numbers") or {}
            results.append({"symbol": m["symbol"], "trades": bt["stats"]["trades"],
                            "in_market": round(bt["stats"]["time_in_market"], 3),
                            "worst_drop": _worst_drop(frames[m["symbol"]], bt["trades"]),
                            "hold_worst_drop": _worst_drop(frames[m["symbol"]], None),
                            "coin_flips_above": (nums.get("coin_flip") or {}).get("ended_higher"),
                            "top3_share": (nums.get("top_3") or {}).get("share_of_profit"),
                            "trades_needed": nums.get("trades_needed"),
                            "total_pct": round(bt["stats"]["total_pct"], 2), "hold_pct": round(bt["stats"]["hold_pct"], 2),
                            "mean_pct": None if not len(r) else round(float(r.mean()), 4),
                            "code": verdict.get("code", "no_trades"), "label": verdict.get("label", "No trades"),
                            "headline": bt["headline"], "plain": verdict.get("plain", ""),
                            "alone_code": (alone.get("verdict") or {}).get("code", "no_trades"),
                            "e_alone": round(float(e_alone), 2),
                            "curve": [0.0] + [round(float(curve[i]), 2) for i in keep]})
        out.append({**{k: s[k] for k in ("slug", "title", "claim", "rule", "lesson")},
                    "text": rule_from(s["rule"]).describe(), "results": results})
    return {"engine": ENGINE_VERSION, "computed": date.today().isoformat(), "end": end, "tries": tries,
            "markets": [dict(m, first=str(frames[m["symbol"]].index[0].date()), source=frames[m["symbol"]].attrs["source"])
                        for m in MARKETS],
            "strategies": out}


def _worst_drop(frame: Any, trades: Optional[List[Dict[str, Any]]]) -> float:
    """The worst fall from a high, in percent, marked to the close each day: holding (trades None), or the rule
    (in the market only between its trades' fills). Fills are at opens, so the rule's figure is approximate."""
    close = frame["close"].to_numpy(float)
    daily = np.r_[0.0, close[1:] / close[:-1] - 1.0]
    if trades is not None:
        held = np.zeros(len(close))
        for t in trades:
            held[t["entry_bar"] + 1:t["exit_bar"] + 1] = 1.0
        daily = daily * held
    equity = np.cumprod(1.0 + daily)
    return round(float(100.0 * (equity / np.maximum.accumulate(equity) - 1.0).min()), 1)


def category(r: Dict[str, Any]) -> Tuple[str, str]:
    """(code, plain label) for one result: the question a trader asks first is whether it beat simply holding."""
    if r["code"] == "real_edge":
        return "real", "Real edge"
    if r["trades"] < 30:
        return "few", "Too few trades"
    if r["total_pct"] < 0:
        return "lost", "Lost money"
    if r["total_pct"] <= r["hold_pct"]:
        return "less", "Less than holding"
    return "luck", "Beat holding, could be luck"


def story(data: Dict[str, Any], r: Dict[str, Any]) -> str:
    """One result in plain words, from its own numbers."""
    code, _ = category(r)
    made, hold = _pct(r["total_pct"]), _pct(r["hold_pct"])
    beat = r["total_pct"] > r["hold_pct"]
    drop = (f" Its worst drop was {r['worst_drop']:.0f}% against {r['hold_worst_drop']:.0f}% for holding"
            + ("." if beat else ": less risk, not more money.") if r["worst_drop"] > r["hold_worst_drop"] + 5 else "")
    if code == "real":
        return f"It made {made} (holding: {hold}), and luck explains that too rarely, even counting all {data['tries']} tries."
    if code == "few":
        did = f"made {made}" if r["total_pct"] >= 0 else f"lost {abs(r['total_pct']):.0f}%"
        return (f"It {did} ({'better' if beat else 'worse'} than holding: {hold}), but from only {r['trades']} "
                "trades: too few to tell skill from luck. A verdict needs at least 30." + drop)
    if code == "lost":
        return f"It lost money: {made}, while holding made {hold}." + drop
    if code == "less":
        return (f"It made {made}, but simply holding made {hold}. Out of the market {100 - 100 * r['in_market']:.0f}% of "
                "the time, it missed more of the rise than it avoided falls." + drop)
    parts = [f"It made {made}, more than holding ({hold}). But"]
    if r.get("top3_share") and r["top3_share"] > 1:
        parts.append("without its 3 best trades it would have lost money,")
    elif r.get("top3_share") and r["top3_share"] > 0.5:
        parts.append(f"its 3 best trades made {100 * r['top3_share']:.0f}% of the profit,")
    if r.get("coin_flips_above") is not None:
        parts.append(f"{r['coin_flips_above']} of 1,000 coin-flip copies of its own trades did even better,")
    parts.append(f"and it is one of {data['tries']} tries: somewhere among them, luck makes a result like this.")
    if r.get("trades_needed"):
        parts.append(f"About {r['trades_needed']:,} more trades at this pace would tell.")
    return " ".join(parts) + drop


def load() -> Optional[Dict[str, Any]]:
    return json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else None


def find(data: Dict[str, Any], slug: str) -> Optional[Dict[str, Any]]:
    return next((s for s in data["strategies"] if s["slug"] == slug), None)


def why_none(data: Dict[str, Any]) -> List[str]:
    """The answers to "isn't it suspicious that nothing passes?", from the numbers themselves."""
    from ..evidence import BASE_BAR
    names = {m["symbol"]: m["name"] for m in data["markets"]}
    rows = [(r["e_alone"], s["title"], names[r["symbol"]], r["alone_code"]) for s in data["strategies"]
            for r in s["results"]]
    passing_alone = sum(code == "real_edge" for _, _, _, code in rows)
    e, title, market, _ = max(rows)
    out = ["Making money isn't the same as having an edge. Stocks, gold and bitcoin all rose a lot over these years, so "
           "almost any rule that owned them made money. A rule has an edge when it adds something you couldn't get by "
           "simply holding, by more than luck explains. Most of these made less than holding; the ones that made more "
           "did it with a handful of big trades, the kind of result luck produces somewhere among many tries.",
           f"It isn't the counting. Judged as if each were the only rule ever tested, {passing_alone} of "
           f"{len(rows)} pass{'es' if passing_alone == 1 else ''}. The closest, {title} on {market}, reaches evidence of "
           f"{e:.0f} where {BASE_BAR:.0f} is needed."]
    path = PACKAGE / "data" / "validation.json"
    if path.exists():
        power = json.loads(path.read_text(encoding="utf-8")).get("power", {})
        found = (power.get("0.3") or {}).get("300", {}).get("rate")
        if found is not None:
            out.append(f"The judge does say yes when an edge is there: on test records with a real edge of 0.3 "
                       f"standard deviations per trade, it finds it {found:.0%} of the time at 300 trades. These rules "
                       "don't have edges that size on these markets.")
    out.append("Research on technical trading rules finds the same once the many rules tried and the costs are "
               "counted (Sullivan, Timmermann & White 1999; Bajgrowicz & Scaillet 2012). That is the trap this "
               "catalogue shows: a rule that looks great somewhere is usually the luckiest of many.")
    return out


def summary(data: Dict[str, Any], s: Dict[str, Any]) -> str:
    n = len(s["results"])
    real = sum(r["code"] == "real_edge" for r in s["results"])
    beat = sum(r["total_pct"] > r["hold_pct"] for r in s["results"])
    return (f"Beat simply holding on {beat} of {n} markets. A real edge, more than luck explains counting all "
            f"{data['tries']} tries, on {real}.")


# ── pages and Markdown ────────────────────────────────────────────────────────

_CHIP = {"real": "good", "luck": "warn", "few": "wait", "less": "flat", "lost": "bad"}


def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:+,.0f}%"


def _chip(r: Dict[str, Any]) -> str:
    code, label = category(r)
    return f'<span class="chip {_CHIP[code]}">{html.escape(label)}</span>'


def _spark(curve: List[float]) -> str:
    if len(curve) < 2:
        return ""
    w, h = 120.0, 32.0
    lo, hi = min(0.0, min(curve)), max(0.0, max(curve))
    span = (hi - lo) or 1.0
    pts = " ".join(f"{w * i / (len(curve) - 1):.1f},{h - 2 - (h - 4) * (v - lo) / span:.1f}" for i, v in enumerate(curve))
    zero = h - 2 - (h - 4) * (0 - lo) / span
    return (f'<svg class="spark {"up" if curve[-1] >= 0 else "down"}" viewBox="0 0 {w:g} {h:g}" aria-hidden="true">'
            f'<line x1="0" x2="{w:g}" y1="{zero:.1f}" y2="{zero:.1f}"/><polyline points="{pts}"/></svg>')


def _notes(data: Dict[str, Any]) -> str:
    e = html.escape
    costs = "; ".join(f"{e(m['name'])}: {m['fee_pct']:g}% fee and {m['slippage_pct']:g}% slippage per fill, daily bars "
                      f"from {m['first']}" for m in data["markets"])
    return (f'<p class="fine">Past prices only, to {e(data["end"])}. {costs}. Each decision is made at a day\'s close '
            f"and filled at the next day's open (EUR/USD: the next day's rate). Every strategy on every market is one "
            f"of {data['tries']} tries, and each verdict counts all of them. There is no \"now\" on this page: nothing "
            "here says what to trade. Not financial advice.</p>")


def hub_page(data: Dict[str, Any]) -> str:
    e = html.escape
    head = "".join(f"<th>{e(m['name'])}</th>" for m in data["markets"])
    rows = "".join(
        f'<tr><th scope="row"><a href="/strategies/{e(s["slug"])}">{e(s["title"])}</a><small>{e(s["claim"])}</small>'
        "</th>" + "".join(f'<td>{_chip(r)}<span class="vs">made {_pct(r["total_pct"])} <small>holding '
                          f'{_pct(r["hold_pct"])}</small></span></td>' for r in s["results"]) + "</tr>"
        for s in data["strategies"])
    body = f"""<section class="intro wrap" aria-labelledby="title">
  <div>
    <h1 id="title">Popular strategies, judged</h1>
    <p class="lede">Ten trading rules you will hear again and again, tested honestly on stocks, gold, a currency and bitcoin: decided at the close, filled at the next open, with costs, and judged like any record.</p>
    <p class="watch"><a href="/">Test your own idea the same way →</a></p>
    <p class="for-ai">For AI assistants: every strategy is also plain Markdown (add <code>.md</code> to its address), and <a href="/api/strategies">/api/strategies</a> has the numbers.</p>
  </div>
  <div class="intro-pet" aria-hidden="true">
    <div class="bubble"><span>Famous isn't the same as real. Let's look.</span><svg class="tail" viewBox="0 0 16 14"><path d="M0 1.5C6 3 10.5 7 15 13 9.5 11 5 10 0 10.5"/></svg></div>
    <div class="pet big-pet" data-mood="excited"></div>
  </div>
</section>
<section class="wrap catalogue" aria-label="The results">
  <div class="table matrix"><table><thead><tr><th>Strategy</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>
  <section class="why-none" aria-labelledby="why"><h2 id="why">Big returns, but no edge?</h2>{"".join(f"<p>{e(p)}</p>" for p in why_none(data))}</section>
  {_notes(data)}
</section>"""
    return _shell("Popular strategies, judged · edge or luck",
                  "Ten popular trading strategies (golden cross, RSI, MACD, breakouts...) tested honestly on stocks, "
                  "gold, a currency and bitcoin, with costs, and judged.", body, alternate="/strategies.md",
                  current="/strategies", css=("strategies.css",))


def strategy_page(data: Dict[str, Any], s: Dict[str, Any]) -> str:
    e = html.escape
    names = {m["symbol"]: m["name"] for m in data["markets"]}
    cards = "".join(
        f'<li class="result"><h3>{e(names[r["symbol"]])}</h3>{_chip(r)}<p class="nums">made <b>{_pct(r["total_pct"])}</b> · '
        f'holding {_pct(r["hold_pct"])}<br><b>{r["trades"]:,}</b> trades · worst drop {r["worst_drop"]:.0f}% (holding '
        f'{r["hold_worst_drop"]:.0f}%)</p>{_spark(r["curve"])}<p class="head">{e(story(data, r))}</p></li>'
        for r in s["results"])
    rule = json.dumps(s["rule"], indent=4)
    body = f"""<article class="wrap strategy">
  <p class="crumbs"><a href="/strategies">Popular strategies</a> · {e(s["title"])}</p>
  <header class="guide-head">
    <div>
      <h1>{e(s["title"])}</h1>
      <p class="lede">{e(s["claim"])}</p>
      <p class="meta">{e(summary(data, s))} · <a href="/strategies/{e(s["slug"])}.md">Markdown for your AI</a></p>
    </div>
  </header>
  <ul class="results">{cards}</ul>
  <section class="prose"><h2>What to know</h2><p>{e(s["lesson"])}</p>
  <h2>Test your own version</h2><p>Change the numbers, add a filter, try another market: each change is another version, and the judge counts them. You have already seen the {data["tries"]} tries on this page, so count them too. Ask your AI to use edge or luck, or run it yourself:</p>
  <div class="code"><pre><code>import edgeorluck as eol
rule = {e(rule)}
px = eol.prices("SPY", "1d", start="2005-01-01")      # or AAPL, ^GSPC, EURUSD, BTCUSDT ...
print(eol.backtest(px, rule, variants_tried={data["tries"] + 1}).to_markdown())   # the {data["tries"]} here, plus yours</code></pre><button class="copy" type="button">Copy</button></div>
  {_notes(data)}</section>
</article>"""
    return _shell(f"{s['title']}, judged · edge or luck", f"{s['claim']} Tested honestly on five markets.", body,
                  alternate=f"/strategies/{s['slug']}.md", current="/strategies", css=("strategies.css",))


def markdown(data: Dict[str, Any], s: Optional[Dict[str, Any]] = None) -> str:
    names = {m["symbol"]: m["name"] for m in data["markets"]}
    picked = [s] if s else data["strategies"]
    lines = ["# Popular strategies, judged" if not s else f"# {s['title']}, judged", ""]
    for one in picked:
        if not s:
            lines += [f"## {one['title']}", ""]
        lines += [f"**The claim:** {one['claim']}", "", f"**The rule:** {one['text']}", "", summary(data, one), "",
                  "| Market | Made | Holding | Trades | Worst drop (holding) | In short |", "|---|---|---|---|---|---|"]
        lines += [f"| {names[r['symbol']]} | {_pct(r['total_pct'])} | {_pct(r['hold_pct'])} | {r['trades']:,} | "
                  f"{r['worst_drop']:.0f}% ({r['hold_worst_drop']:.0f}%) | {category(r)[1]} |" for r in one["results"]]
        lines += [""] + [f"- **{names[r['symbol']]}:** {story(data, r)}" for r in one["results"]]
        lines += ["", f"**What to know:** {one['lesson']}", ""]
    if not s:
        lines += ["## Big returns, but no edge?", ""] + [p + "\n" for p in why_none(data)]
    lines += [f"_Past prices only, to {data['end']}; daily bars; costs on every fill; each verdict counts all "
              f"{data['tries']} tries. Not a signal and not financial advice._", ""]
    return "\n".join(lines)


__all__ = ["STRATEGIES", "MARKETS", "compute", "load", "find", "summary", "hub_page", "strategy_page", "markdown"]
