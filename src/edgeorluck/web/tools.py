"""The assistant tools for callers over the web, shared by the MCP server at /mcp and the JSON API at /api/tools/
(for GPTs and anything that calls HTTP): the same tools as the local MCP server, minus files on anyone's computer.

* Versions tried are counted per study: the web keeps no conversation, so each answer names a study id and the
  caller passes it back. Every record and rule judged under one study raises the bar for the next.
* A frozen rule comes back as a receipt the server signs (the rule, its market, the time). forward_test takes the
  receipt and judges only the trades the rule entered after that time. Nothing is stored, nothing is public, and a
  changed receipt fails the signature. Set EDGEORLUCK_SECRET so receipts outlive a restart.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Union

from .. import mcp_server as local

_STUDY = re.compile(r"^[A-Za-z0-9_-]{6,40}$")
_SECRET = (os.environ.get("EDGEORLUCK_SECRET") or secrets.token_hex(32)).encode("utf-8")


@contextlib.contextmanager
def studied(study: Optional[str]) -> Iterator[str]:
    """Count what is judged inside under this study (a new one when none or a malformed one is given). Inside,
    no string names a file on this server: every web tool is called through here."""
    key = study if isinstance(study, str) and _STUDY.match(study) else secrets.token_hex(6)
    token, files = local._CURRENT.set(key), local._FILES.set(False)
    try:
        yield key
    finally:
        local._FILES.reset(files)
        local._CURRENT.reset(token)


def _sign(body: str) -> str:
    return hmac.new(_SECRET, body.encode("ascii"), hashlib.sha256).hexdigest()[:32]


def receipt_for(spec: Dict[str, Any]) -> str:
    """The signed receipt of a rule frozen now."""
    data = dict(spec, since=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    body = base64.urlsafe_b64encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).decode()
    return body.rstrip("=") + "." + _sign(body.rstrip("="))


def open_receipt(receipt: Any) -> Optional[Dict[str, Any]]:
    """The frozen rule in a receipt this server signed; None for anything else."""
    body, _, mac = str(receipt or "").strip().strip("`\"").partition(".")
    if not body or not hmac.compare_digest(mac.encode(), _sign(body).encode()):
        return None
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def judge(csv_text: Optional[str] = None, pnl: Optional[List[float]] = None, variants_tried: Optional[int] = None,
          benchmark_return_pct: Optional[float] = None, fees_included: Optional[bool] = None,
          source: Optional[str] = None, unit: Optional[str] = None, kind: Optional[str] = None,
          backtest_csv_text: Optional[str] = None, backtest_pnl: Optional[List[float]] = None,
          study: Optional[str] = None) -> Dict[str, Any]:
    with studied(study) as key:
        return {"markdown": local.judge_trades(
            csv_text=csv_text, pnl=pnl, variants_tried=variants_tried, benchmark_return_pct=benchmark_return_pct,
            fees_included=fees_included, source=source, unit=unit, kind=kind, backtest_csv_text=backtest_csv_text,
            backtest_pnl=backtest_pnl), "study": key}


def backtest(buy: str, sell: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h",
             start: Optional[str] = None, end: Optional[str] = None, stop_loss_pct: Optional[float] = None,
             take_profit_pct: Optional[float] = None, max_hold_bars: Optional[int] = None, side: str = "long",
             fee_pct: float = 0.1, slippage_pct: float = 0.05, variants_tried: Optional[int] = None,
             freeze: bool = False, study: Optional[str] = None) -> Dict[str, Any]:
    with studied(study) as key:
        md = local.backtest_rule(buy=buy, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
                                 stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                                 max_hold_bars=max_hold_bars, side=side, fee_pct=fee_pct, slippage_pct=slippage_pct,
                                 variants_tried=variants_tried)
    out: Dict[str, Any] = {"markdown": md, "study": key}
    if freeze and not md.startswith("Couldn't"):
        out["receipt"] = receipt_for({"rule": local._rule_dict(buy, sell, stop_loss_pct, take_profit_pct,
                                                               max_hold_bars, side), "symbol": symbol,
                                      "interval": interval, "start": start, "fee_pct": fee_pct,
                                      "slippage_pct": slippage_pct})
    return out


def forward(receipt: str, variants_tried: Optional[int] = None) -> Dict[str, Any]:
    from ..simulate import backtest as run
    spec = open_receipt(receipt)
    if spec is None:
        return {"markdown": "That isn't a receipt this server signed, or it was changed. Freeze the rule again "
                            "(backtest_rule with freeze=true): its trades count from then on."}
    with studied(None):                               # a forward test is no new version: it counts nowhere
        md = local._run("Couldn't run the forward test", lambda: run(
            local._prices_for(spec["symbol"], spec["interval"], spec.get("start"), None, None), spec["rule"],
            fee_pct=spec["fee_pct"], slippage_pct=spec["slippage_pct"], variants_tried=variants_tried or 1,
            since=spec["since"]))
    return {"markdown": f"_Frozen on {spec['since'][:16].replace('T', ' ')} UTC ({spec['symbol']} "
                        f"{spec['interval']}): only its trades after that count._\n\n" + md}


def tune(buy: str, grid: Union[Dict[str, Any], str], sell: Optional[str] = None, symbol: str = "BTCUSDT",
         interval: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
         stop_loss_pct: Optional[Union[float, str]] = None, take_profit_pct: Optional[Union[float, str]] = None,
         max_hold_bars: Optional[Union[int, str]] = None, side: str = "long", holdout: float = 0.3,
         fee_pct: float = 0.1, slippage_pct: float = 0.05, study: Optional[str] = None) -> Dict[str, Any]:
    with studied(study) as key:
        return {"markdown": local.tune_rule(
            buy=buy, grid=grid, sell=sell, symbol=symbol, interval=interval, start=start, end=end,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, max_hold_bars=max_hold_bars, side=side,
            holdout=holdout, fee_pct=fee_pct, slippage_pct=slippage_pct), "study": key}


def as_text(answer: Dict[str, Any]) -> str:
    """One answer as the Markdown an assistant reads: the result, then its study and receipt."""
    text = answer["markdown"]
    if answer.get("receipt"):
        text += (f"\n**Frozen.** Its receipt: `{answer['receipt']}`\n\nGive the receipt to the user to keep. Later, "
                 "forward_test(receipt) judges only the trades the rule makes from now on: the honest test, since "
                 "nobody can tune a rule on trades that haven't happened. Change the rule and it is a new one.\n")
    if answer.get("study"):
        text += (f"\n_Study `{answer['study']}`: pass study=\"{answer['study']}\" in every later call about this "
                 "idea, so each version tried counts._\n")
    return text


TOOLS = {"judge": judge, "backtest": backtest, "forward": forward, "tune": tune}

__all__ = ["studied", "receipt_for", "open_receipt", "judge", "backtest", "forward", "tune", "as_text", "TOOLS"]
