"""The web app's own calls, graded: how the judge keeps score on itself.

Every judged upload (30+ trades) gets a call: the chance that its next HORIZON
trades end positive, written down before those trades exist. When the same
history comes back with at least HORIZON more trades, the call is graded.

The call pulls the record's edge toward zero by n / (n + k). Every call is
scored under every candidate k, so once MIN_GRADED calls are graded the app
uses the k whose calls scored best (lowest Brier). The coin flip (k = inf)
is always a candidate, so it can learn that records predict nothing.

Two append-only files, each line chained to the one before by its hash, so
the record cannot be quietly edited. Kept per call: a one-way fingerprint of
the first 30 trades and of the whole list, the number of trades, the unit,
the forecast and the time. Kept per grade: whether those 30 trades ended up
or down, and the scores. No trades, no files, no addresses.

Only the web app writes this; the library itself writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..judge import ENGINE_VERSION, MIN_TRADES
from ..readers import Trades

HORIZON = 30
SHRINKAGE = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, math.inf)
DEFAULT_K = 200.0
MIN_GRADED = 50
CALIBRATION_BINS = 5
STALE_DAYS = 60


def default_root() -> Path:
    env = os.environ.get("EDGEORLUCK_DATA_DIR")
    return Path(env) if env else Path.cwd() / "edgeorluck-data"


def fingerprint(x: np.ndarray, times: Optional[list], n: Optional[int] = None) -> str:
    n = len(x) if n is None else n
    body: List[Any] = [round(float(v), 6) for v in x[:n]]
    if times:
        body += [t.strftime("%Y-%m-%dT%H:%M") for t in times[:n]]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:32]


def forward_probabilities(x: np.ndarray, shrinkage: Sequence[float], horizon: int = HORIZON) -> List[float]:
    """P(the next ``horizon`` trades end positive) per shrinkage weight k."""
    n, mean = len(x), float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    out = []
    for k in shrinkage:
        if not math.isfinite(k) or not math.isfinite(sd) or sd <= 0:
            out.append(0.5)
            continue
        drift = horizon * mean * n / (n + k)
        spread = math.sqrt(horizon) * sd * math.sqrt(1.0 + horizon / n)
        out.append(0.5 * math.erfc(-drift / (spread * math.sqrt(2.0))))
    return out


def _k_label(k: float) -> Any:
    return "coin_flip" if math.isinf(k) else k


class Ledger:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.root / "calls.jsonl"
        self.grades_path = self.root / "grades.jsonl"
        self._lock = threading.Lock()
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.open: Dict[str, str] = {}
        self.grades: List[Dict[str, Any]] = []
        self._heads = {self.calls_path: "", self.grades_path: ""}
        for path in (self.calls_path, self.grades_path):
            for line in self._lines(path):
                record = json.loads(line)
                self._heads[path] = hashlib.sha256(line.encode("utf-8")).hexdigest()
                if path == self.calls_path:
                    self.calls[record["id"]] = record
                    self.open[record["identity"]] = record["id"]
                else:
                    self.grades.append(record)
                    if self.open.get(self.calls.get(record["call"], {}).get("identity")) == record["call"]:
                        del self.open[self.calls[record["call"]]["identity"]]

    @staticmethod
    def _lines(path: Path) -> List[str]:
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        record = dict(record, prev=self._heads[path])
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._heads[path] = hashlib.sha256(line.encode("utf-8")).hexdigest()

    def chosen_k(self) -> float:
        if len(self.grades) < MIN_GRADED:
            return DEFAULT_K
        scores = np.mean([g["brier"] for g in self.grades], axis=0)
        best = min(range(len(SHRINKAGE)), key=lambda i: (round(float(scores[i]), 12), -i))
        return SHRINKAGE[best]

    def observe(self, st: Trades, *, source: str = "upload", now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        x, times, n = st.pnl, st.times, len(st.pnl)
        if n < MIN_TRADES:
            return None
        now = now or datetime.now(timezone.utc)
        identity = fingerprint(x, times, MIN_TRADES)
        graded = None
        with self._lock:
            call = self.calls.get(self.open.get(identity, ""))
            if call is not None:
                n0 = call["n"]
                same = n >= n0 and fingerprint(x, times, n0) == call["prefix"]
                if same and n >= n0 + HORIZON:
                    outcome = int(float(np.sum(x[n0:n0 + HORIZON])) > 0)
                    graded = {"call": call["id"], "at": now.isoformat(), "outcome": outcome, "source": source,
                              "probability": call["probability"],
                              "brier": [round((p - outcome) ** 2, 8) for p in call["probabilities"]]}
                    self._append(self.grades_path, graded)
                    self.grades.append(graded)
                    del self.open[identity]
                else:
                    return self._view(call, None, kept=True)
            if times and (now - times[-1].to_pydatetime()).days > STALE_DAYS:
                last = times[-1]
                return {"probability": None, "horizon": HORIZON, "made": None, "graded_now": graded is not None,
                        "sentence": f"No forecast: it hasn't traded since {last:%b} {last.day}, {last.year}, so its "
                                    f"next {HORIZON} trades may never come."}
            probabilities = [round(p, 6) for p in forward_probabilities(x, SHRINKAGE)]
            k = self.chosen_k()
            call = {"id": uuid.uuid4().hex[:16], "at": now.isoformat(), "source": source, "identity": identity,
                    "prefix": fingerprint(x, times, n), "n": n, "horizon": HORIZON, "unit": st.unit,
                    "k": _k_label(k), "probability": probabilities[SHRINKAGE.index(k)],
                    "probabilities": probabilities, "engine": ENGINE_VERSION}
            self._append(self.calls_path, call)
            self.calls[call["id"]] = call
            self.open[identity] = call["id"]
            return self._view(call, graded)

    def _view(self, call: Dict[str, Any], graded: Optional[Dict[str, Any]], *, kept: bool = False) -> Dict[str, Any]:
        p = call["probability"]
        sentence = (f"Our forecast: {p:.0%} that your next {HORIZON} trades end positive. Upload again after "
                    f"{HORIZON} more trades and we grade ourselves.")
        if kept:
            made = datetime.fromisoformat(call["at"])
            sentence = (f"Our forecast: {p:.0%} that the {HORIZON} trades after your first {call['n']} end positive "
                        f"(made {made:%b} {made.day}). Come back with {HORIZON} more and we grade it.")
        if graded is not None:
            sentence = (f"Last time we said {graded['probability']:.0%}; those {HORIZON} trades ended "
                        f"{'positive' if graded['outcome'] else 'negative'}. Graded. " + sentence)
        return {"probability": p, "horizon": HORIZON, "made": call["at"], "graded_now": graded is not None,
                "sentence": sentence}

    def track_record(self) -> Dict[str, Any]:
        k = self.chosen_k()
        index = SHRINKAGE.index(k)
        graded = len(self.grades)
        brier = float(np.mean([g["brier"][index] for g in self.grades])) if graded else None
        bins = []
        if graded:
            pairs = [(self.calls[g["call"]]["probabilities"][index], g["outcome"]) for g in self.grades
                     if g["call"] in self.calls]
            predicted = np.array([p for p, _ in pairs])
            outcome = np.array([o for _, o in pairs])
            edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
            for lo, hi in zip(edges[:-1], edges[1:]):
                inside = (predicted >= lo) & ((predicted < hi) | (hi == 1.0))
                if inside.any():
                    bins.append({"from": round(float(lo), 2), "to": round(float(hi), 2), "calls": int(inside.sum()),
                                 "said": round(float(predicted[inside].mean()), 3),
                                 "happened": round(float(outcome[inside].mean()), 3)})
        base_rate = base_brier = None
        if graded == 0:
            made = len(self.calls)
            sentence = (f"We have made {made:,} forecast{'' if made == 1 else 's'} and graded none yet. Each is "
                        "graded when its trader comes back with 30 more trades.")
        else:
            # Beating a coin flip (0.25) is too easy: always saying the usual rate does that whenever outcomes are
            # not 50/50 (proofs/Selection.lean, brier_base_rate_beats_coin). The bar is that constant guess, using
            # the rate seen in the graded outcomes, which flatters the guess, not us.
            outcomes = np.array([float(g["outcome"]) for g in self.grades])
            base_rate = float(outcomes.mean())
            base_brier = float(np.mean((base_rate - outcomes) ** 2))
            verdict = ("better than always guessing the usual rate" if brier < base_brier
                       else "not better than always guessing the usual rate yet")
            sentence = (f"Graded {graded:,} of our {len(self.calls):,} forecasts: Brier score {brier:.3f}. Always "
                        f"guessing the usual rate ({base_rate:.0%} up) would score {base_brier:.3f}, a coin flip "
                        f"0.250. {verdict.capitalize()}.")
        return {"predictions": len(self.calls), "graded": graded,
                "brier": None if brier is None else round(brier, 4), "coin_flip_brier": 0.25,
                "base_rate": None if base_rate is None else round(base_rate, 4),
                "base_rate_brier": None if base_brier is None else round(base_brier, 4),
                "shrinkage": _k_label(k), "learning_from": MIN_GRADED, "calibration": bins, "sentence": sentence}
