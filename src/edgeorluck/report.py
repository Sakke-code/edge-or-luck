"""The result as Markdown: what an assistant pastes, or a trader reads."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .words import fmt, share

STATUS_MARK = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def _num_rows(numbers: Dict[str, Any], card: Dict[str, Any]) -> List[str]:
    unit = numbers["unit"]
    rows = []
    span = ""
    if numbers.get("first_trade"):
        span = f", {numbers['first_trade']} to {numbers['last_trade']}"
    rows.append(("Trades", f"{numbers['trades']:,}{span}"))
    money = card.get("money") or {}
    if money.get("total") is not None:
        text = fmt(money["total"], money.get("unit") or unit) + " in total"
        if money.get("percent_mean") is not None and money.get("unit") != "%":
            text += f" (average trade {fmt(money['percent_mean'], '%')})"
        if money.get("disagree"):
            text += " **(percent and money disagree)**"
        rows.append(("Result", text))
    mpt = numbers["mean_per_trade"]
    lo, hi = mpt["range_95"]
    rows.append(("Average per trade", f"{fmt(mpt['value'], unit)} (95% range {fmt(lo, unit)} to {fmt(hi, unit)})"))
    adj = numbers.get("adjusted")
    if adj:
        what = (f"{adj['versions_tried']} versions tried" if adj.get("versions_tried")
                else f"{adj['strategies']} strategies")
        rows.append((f"Range adjusted for {what}", f"{fmt(adj['range'][0], unit)} to {fmt(adj['range'][1], unit)}"))
    wr = numbers["win_rate"]
    rows.append(("Win rate", f"{share(wr['value'])} ({share(wr['range_95'][0])} to {share(wr['range_95'][1])})"))
    coin = numbers.get("coin_flip") or {}
    if coin:
        need = coin.get("needed_at_most")
        need_text = f"{need:,} or fewer" if need else "none"
        rows.append(("Coin-flip copies", f"{coin['ended_higher']:,} of {coin['copies']:,} ended higher "
                                         f"(a pass needs {need_text})"))
    top3 = numbers.get("top_3")
    if top3:
        rows.append(("Best 3 trades", f"{share(top3['share_of_profit'])} of the profit; without them "
                                      f"{fmt(top3['mean_without'], unit)} per trade"))
    fees = numbers.get("fees")
    if fees:
        text = f"{fmt(fees['paid'], unit, False)} paid"
        if fees.get("share_of_gross") is not None:
            text += f" ({share(fees['share_of_gross'])} of the gross)"
        rows.append(("Fees", text))
    dd = numbers.get("drawdown") or {}
    if dd:
        rows.append(("Drop from peak", f"worst {fmt(dd.get('max'), unit, False)}, now {fmt(dd.get('current'), unit, False)}"))
    recent = card.get("recent")
    if recent:
        rows.append((f"Last {recent['window']} trades", recent["sentence"]))
    bench = card.get("benchmark")
    if bench:
        rows.append(("Against holding", bench["sentence"]))
    out = ["| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def _why(text: Optional[str], indent: str = "") -> List[str]:
    return [f"{indent}_Why: {text}_"] if text else []


def to_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    status = result.get("status")
    if status != "ok":
        need = result.get("need") or {}
        lines += ["## Edge or luck: need the trades", "", f"**We couldn't judge this yet.** {need.get('problem', '')}",
                  "", f"**What we need:** {need.get('needs', '')}", "",
                  f"**Ask the trader:** {need.get('question_for_trader', '')}", ""]
        lines += _footer(result)
        return "\n".join(lines)
    verdict = result["verdict"]
    lines += [f"## Edge or luck: {verdict['label']}", "", f"**{verdict['headline']}**", "", verdict["plain"], ""]
    lines += _why(verdict.get("why")) + [""]
    step = result.get("next_step")
    if step:
        lines += [f"**Next step:** {step['text']}", ""] + _why(step.get("why")) + [""]
    compare = result.get("compare")
    if compare:
        lines += ["### Live vs backtest", "", f"**{compare['headline']}**", "", compare.get("plain", ""), ""]
        lines += _why(compare.get("why")) + [""]
    cards = result.get("strategies") or []
    read_as = (result.get("input") or {}).get("read_as")
    if len(cards) == 1 and result.get("numbers"):
        lines += ["### The numbers", ""] + ([f"_{read_as}_", ""] if read_as else [])
        lines += _num_rows(result["numbers"], cards[0]) + [""]
    elif len(cards) > 1:
        lines += ["### Each strategy", "", "| Strategy | Verdict | Trades | Average per trade | 95% range |",
                  "|---|---|---|---|---|"]
        for c in cards:
            num = c.get("numbers") or {}
            mpt = num.get("mean_per_trade") or {}
            rng = mpt.get("range_95") or [None, None]
            lines.append(f"| {c['name']} | {c['verdict']['label']} | {c['n']:,} | {fmt(mpt.get('value'), c['unit'])} "
                         f"| {fmt(rng[0], c['unit'])} to {fmt(rng[1], c['unit'])} |")
        lines.append("")
        for c in cards:
            lines += [f"- **{c['name']}:** {c['verdict']['headline']}"]
        lines.append("")
    changes = result.get("what_would_change_it") or []
    if changes:
        lines += ["### What would change it", ""] + [f"- {c['text']}" for c in changes] + [""]
    checks = result.get("checks") or []
    if checks:
        lines += ["### Checks", ""]
        for c in checks:
            lines.append(f"- **{STATUS_MARK.get(c['status'], c['status'])}** {c['text']}")
            if c.get("fix") and c["status"] != "pass":
                lines.append(f"  Fix: {c['fix']}")
            lines += _why(c.get("why"), "  ")
        lines.append("")
    questions = result.get("ask_the_user") or []
    if questions:
        lines += ["### Questions for the trader", ""]
        for i, q in enumerate(questions, start=1):
            lines.append(f"{i}. {q['question']} (answer with `{q['answer_with']}`)")
            lines += _why(q.get("why"), "   ")
        lines.append("")
    audit = result.get("ai_audit_prompt")
    if audit:
        lines += ["### Prompt to audit the bot's code", "", "Give this to the AI that wrote the bot:", "", "```text",
                  audit, "```", ""]
    lines += ["<details><summary>How we know</summary>", ""]
    lines += [f"- {h}" for h in result.get("how_we_know", [])]
    lines += ["", "Limits:"] + [f"- {h}" for h in result.get("limitations", [])] + ["", "</details>", ""]
    lines += ["### For the assistant", ""] + [f"- {r}" for r in result.get("for_the_assistant", [])] + [""]
    lines += _footer(result)
    return "\n".join(lines)


def _footer(result: Dict[str, Any]) -> List[str]:
    fp = result.get("input_fingerprint") or "n/a"
    return [f"_edge or luck {result.get('engine_version')} (schema {result.get('schema_version')}), input {fp}. "
            "The same input always gets the same answer. Not financial advice: a statistical check of past trades._"]
