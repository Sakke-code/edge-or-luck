"""The public call and its result object."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterator, Mapping, Optional

from .judge import judge_dict
from .report import to_markdown


class Result(Mapping):
    """A judged record: read it like a dict, or call .to_dict(), .to_json(),
    .to_markdown(). Start with result["verdict"]["headline"]."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def verdict(self) -> Optional[Dict[str, Any]]:
        return self._data.get("verdict")

    @property
    def headline(self) -> str:
        v = self._data.get("verdict")
        return v["headline"] if v else self._data["summary"]["headline"]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self._data, indent=indent, ensure_ascii=False, allow_nan=False)

    def to_markdown(self) -> str:
        return to_markdown(self._data)

    def _repr_markdown_(self) -> str:  # Jupyter
        return self.to_markdown()

    def __repr__(self) -> str:
        code = (self._data.get("verdict") or {}).get("code", self.status)
        return f"<edgeorluck.Result {code}: {self.headline!r}>"

    def __str__(self) -> str:
        return self.to_markdown()


def judge(data: Any = None, *, variants_tried: Any = None, benchmark_return_pct: Any = None,
          kind: Optional[str] = None, unit: Optional[str] = None, fees_included: Any = None, source: Any = None,
          backtest: Any = None, name: Optional[str] = None, include_chart: bool = False) -> Result:
    """Judge a trading record: a real edge, or luck?

    data: a file path (CSV/TSV/XLSX/freqtrade JSON), CSV text or bytes, a
        pandas DataFrame, a list of results per trade, a list of dicts, a dict
        of lists, a backtesting.py stats Series (or its _trades), a vectorbt
        Portfolio (or trades.records_readable), or a freqtrade backtest result.
    variants_tried: how many versions/settings were tried in all (raises the
        bar: the best of many looks good by luck).
    benchmark_return_pct: what buying and holding returned over the same
        dates, in percent (34.5 means +34.5%).
    kind: "auto" (default), "trades", "fills", "tradingview" or "pnl".
    unit: "%" or a currency like "USDT", when the input doesn't say.
    fees_included: True/False when known.
    source: "backtest", "paper" or "live".
    backtest: the backtest's trades (same forms as data), to check whether
        live trades behave like the backtest promised.
    include_chart: add equity and coin-flip curves (for drawing).

    Never raises on messy input: status "need_input" says what is missing.
    """
    return Result(judge_dict(data, variants_tried=variants_tried, benchmark_return_pct=benchmark_return_pct,
                             kind=kind, unit=unit, fees_included=fees_included, source=source, backtest=backtest,
                             name=name, include_chart=include_chart))
