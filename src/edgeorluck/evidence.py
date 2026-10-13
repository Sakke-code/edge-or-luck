"""The evidence maths: anytime-valid ranges, e-values and e-BH.

Derived from the "pooled evidence" module of the trading bot this project
grew out of (its commit 14ef300): one-sided normal-mixture e-values and e-BH
(Wang & Ramdas 2022), and the time-uniform ideas of Waudby-Smith, Arbour,
Sinha, Kennedy & Ramdas (2021). Stdlib only.

What changed here, and why:

* The range is now the inversion of the same one-sided mixture e-value the
  verdict uses (one side at alpha/2 each), so "the whole 95% range is above
  zero" and "the verdict is real_edge" are the same statement for a single
  strategy. The earlier version reported a two-sided Robbins-mixture range
  next to an e-value bar it did not match, so the two could disagree.
* Records whose losses are rarer but bigger than their wins (negative skew:
  grids, martingales, option selling) are the classic false "edge": at a few
  dozen trades the big loss has often not happened yet. The claim "above
  zero" must clear a bar raised by the Edgeworth term of a studentized mean
  (Hall 1987) when the trades are skewed against it, and only then: skew is
  never allowed to lower a bar. The same holds for "below zero" and positive
  skew. The range is widened on the matching side.
* The false discovery rate is q = 0.05 (it was 0.10): with both directions of
  each strategy in one family, a single strategy needs e >= 2/q = 40, which is
  exactly the 95% range clearing zero. With N versions tried the family is 2N
  hypotheses (the versions not shown count as e = 0, valid because e-BH holds
  under any dependence), so one strategy needs e >= 40 N: the Bonferroni
  95%/N range clearing zero. One rule, one picture.

Honest limits, stated wherever results are shown:

* trades are treated as independent draws;
* the e-value is a Gaussian likelihood ratio with a plug-in standard
  deviation: valid as the number of trades grows (asymptotically), not
  exactly at every sample size;
* the mixture is tuned to decide soonest around T_STAR trades.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Set, Tuple

import numpy as np

Q = 0.05                 # e-BH false discovery rate, both directions in one family
ALPHA = 0.05             # the range is a two-sided 95% range: alpha/2 on each side
T_STAR = 100             # the mixture decides soonest around this many trades
BASE_BAR = 2.0 / Q       # one strategy, one version: e >= 40
SERIAL_ADJUST = True     # widen the spread when neighbouring trades move together (Newey-West)
SKEW_ADJUST = True       # raise a claim's bar when the trades are skewed against it (Edgeworth)
SKEW_CAP = 6.0


def mixing_rho2(bar: float, t_star: int = T_STAR) -> float:
    """The mixing scale that makes an e-value cross ``bar`` soonest near
    ``t_star`` observations."""
    return 2.0 * math.log(max(bar, 1.0 + 1e-9)) / t_star


def _log_two_phi(z: float) -> float:
    """log(2 * Phi(z)), stable for very negative z."""
    if z > -8.0:
        return math.log(2.0) + math.log(max(1e-300, 0.5 * math.erfc(-z / math.sqrt(2.0))))
    return math.log(2.0) - 0.5 * z * z - math.log(-z) - 0.5 * math.log(2.0 * math.pi)


def log_e_standardized(n: int, s: float, rho2: float) -> float:
    """log of the one-sided normal-mixture e-value for a standardized sum
    ``s`` (sum / sd) after ``n`` observations: a half-normal mixture of
    Gaussian likelihood ratios against "the mean is 0"."""
    if n < 1 or rho2 <= 0:
        return 0.0
    v = 1.0 + rho2 * n
    z = math.sqrt(rho2) * s / math.sqrt(v)
    return -0.5 * math.log(v) + rho2 * s * s / (2.0 * v) + _log_two_phi(z)


def log_e_one_sided(n: int, total: float, sd: float, rho2: float, sign: int) -> float:
    """log e-value against "the mean is 0", for a drift in direction ``sign``
    (+1: above zero, -1: below zero)."""
    if n < 1 or not math.isfinite(sd) or sd <= 0:
        return 0.0
    return log_e_standardized(n, sign * total / sd, rho2)


@functools.lru_cache(maxsize=8192)
def boundary(n: int, bar: float, rho2: float) -> float:
    """The standardized sum ``s`` at which the e-value reaches ``bar`` after
    ``n`` observations (the e-value grows with ``s``)."""
    if n < 1:
        return math.inf
    target = math.log(bar)
    v = 1.0 + rho2 * n
    hi = math.sqrt(2.0 * v / rho2 * (target + 0.5 * math.log(v) + 1.0)) + 1.0
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if log_e_standardized(n, mid, rho2) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def half_width(n: int, sd: float, bar: float, rho2: Optional[float] = None) -> float:
    """Half-width of the anytime-valid range whose ends are where the
    one-sided e-value against that mean reaches ``bar``: a two-sided range at
    level 1 - 2 / bar (bar 40 is a 95% range)."""
    if n < 2 or not math.isfinite(sd):
        return math.inf
    rho2 = mixing_rho2(bar) if rho2 is None else rho2
    return boundary(n, bar, rho2) * sd / n


def z_equivalent(n: int, bar: float, rho2: Optional[float] = None) -> float:
    """The one-look z-score a mean would need to clear the anytime bar now."""
    rho2 = mixing_rho2(bar) if rho2 is None else rho2
    return boundary(n, bar, rho2) / math.sqrt(max(n, 1))


def normal_tail(z: float) -> float:
    """P(Z > z) for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass(frozen=True)
class Record:
    """What the evidence needs from a list of trade results."""
    n: int
    total: float
    sd: float            # the effective spread: sample SD, widened for serial correlation
    skew: float          # sample skewness (adjusted Fisher-Pearson), 0 when unknown
    serial: float = 1.0  # the variance factor the spread was widened by

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")


def sample_skew(x: "np.ndarray") -> float:
    n = len(x)
    if n < 3:
        return 0.0
    d = x - x.mean()
    m2 = float(np.dot(d, d)) / n
    if m2 <= 0:
        return 0.0
    g1 = float(np.mean(d ** 3)) / m2 ** 1.5
    return float(np.clip(g1 * math.sqrt(n * (n - 1)) / (n - 2), -SKEW_CAP, SKEW_CAP))


def record_of(x: "np.ndarray") -> Record:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return Record(n, float(x.sum()) if n else 0.0, float("nan"), 0.0)
    sd = float(np.std(x, ddof=1))
    scale = float(np.mean(np.abs(x)))
    sd = max(sd, 1e-9 * (scale + 1.0))
    factor = serial_factor(x)
    return Record(n, float(x.sum()), sd * math.sqrt(factor), sample_skew(x), factor)


def skew_shift(skew: float, n: int, bar: float, rho2: float, sign: int) -> float:
    """How much more standardized sum a claim in direction ``sign`` must show
    when the trades are skewed against it: the Edgeworth term of a studentized
    mean at the bar's z, g (2 z^2 + 1) / 6 with g the skew against the claim.
    Zero when the skew is on the claim's side."""
    against = -skew if sign > 0 else skew
    if not SKEW_ADJUST or against <= 0 or n < 3:
        return 0.0
    z = boundary(n, bar, rho2) / math.sqrt(n)
    return min(against, SKEW_CAP) * (2.0 * z * z + 1.0) / 6.0


def log_e_record(rec: Record, bar: float, sign: int, rho2: Optional[float] = None) -> float:
    """log e-value for "the average is above (sign +1) / below (-1) zero",
    with the bar's skew correction."""
    if rec.n < 2 or not math.isfinite(rec.sd) or rec.sd <= 0:
        return 0.0
    rho2 = mixing_rho2(bar) if rho2 is None else rho2
    s = sign * rec.total / rec.sd - skew_shift(rec.skew, rec.n, bar, rho2, sign)
    return log_e_standardized(rec.n, s, rho2)


def range_of(rec: Record, bar: float, rho2: Optional[float] = None) -> Tuple[float, float]:
    """The anytime-valid range for the average at level 1 - 2 / bar: where the
    one-sided e-value against that average reaches ``bar``, widened on the
    side the skew argues against."""
    if rec.n < 2 or not math.isfinite(rec.sd):
        return float("-inf"), float("inf")
    rho2 = mixing_rho2(bar) if rho2 is None else rho2
    b = boundary(rec.n, bar, rho2)
    up = b + skew_shift(rec.skew, rec.n, bar, rho2, +1)
    down = b + skew_shift(rec.skew, rec.n, bar, rho2, -1)
    return rec.mean - up * rec.sd / rec.n, rec.mean + down * rec.sd / rec.n


def ebh(log_e: Mapping[str, float], q: float = Q, total: Optional[int] = None) -> Set[str]:
    """e-BH: the keys selected at false discovery rate ``q`` (Wang & Ramdas
    2022), valid under any dependence. ``total`` is the size of the whole
    family; hypotheses not listed (versions tried but not shown) count as
    e = 0 and are never selected."""
    k_total = max(len(log_e), int(total or 0))
    if not k_total or not log_e:
        return set()
    ranked = sorted(log_e.items(), key=lambda kv: kv[1], reverse=True)
    chosen = 0
    for k, (_, value) in enumerate(ranked, start=1):
        if value >= math.log(k_total / (k * q)):
            chosen = k
    return {key for key, _ in ranked[:chosen]}


def trades_needed(rec: Record, bar: float, rho2: Optional[float] = None) -> Optional[int]:
    """More trades until the range would clear zero if the average, spread
    and skew stay what they are now; 0 when it already does, None when the
    average is not positive or the answer is out of reach."""
    mean, sd, n = rec.mean, rec.sd, rec.n
    if not (n >= 2 and mean > 0 and math.isfinite(sd) and sd > 0):
        return None
    rho2 = mixing_rho2(bar) if rho2 is None else rho2

    def clears(m: int) -> bool:
        return mean * m / sd > boundary(m, bar, rho2) + skew_shift(rec.skew, m, bar, rho2, +1)

    if clears(n):
        return 0
    lo, hi = n, max(n, 2) * 2
    while not clears(hi):
        lo, hi = hi, hi * 2
        if hi > 5_000_000:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if clears(mid):
            hi = mid
        else:
            lo = mid
    return hi - n


def max_versions_passing(rec: Record, *, strategies: int = 1, cap: int = 1_000_000) -> int:
    """The largest number of versions tried for which this record would still
    pass on its own (e >= 40 N with the mixture tuned for that bar); 0 when it
    does not pass even as the only version."""
    def passes(m: int) -> bool:
        bar = BASE_BAR * max(m, strategies)
        return log_e_record(rec, bar, +1) >= math.log(bar)

    if not passes(1):
        return 0
    lo, hi = 1, 2
    while passes(hi):
        lo, hi = hi, hi * 2
        if hi > cap:
            return cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return lo


def serial_factor(x: "np.ndarray") -> float:
    """How much neighbouring trades moving together widens the spread of the
    average: the Newey-West long-run variance over the plain variance, never
    below 1 (a negative correlation is not allowed to narrow it)."""
    n = len(x)
    if not SERIAL_ADJUST or n < 20:
        return 1.0
    d = x - x.mean()
    var = float(np.dot(d, d)) / n
    if var <= 0:
        return 1.0
    lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    total = 1.0
    for lag in range(1, max(lags, 1) + 1):
        rho = float(np.dot(d[lag:], d[:-lag])) / n / var
        total += 2.0 * (1.0 - lag / (lags + 1.0)) * rho
    return float(min(10.0, max(1.0, total)))



def wilson_bounds_count(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion (Wilson 1927), from counts."""
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, wins / n))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def describe_family(versions: Optional[int], strategies: int) -> Dict[str, int]:
    """The e-BH family: every version tried or strategy shown, both directions."""
    m = max(int(versions or 1), int(strategies or 1), 1)
    return {"versions_or_strategies": m, "hypotheses": 2 * m, "bar_alone": int(round(BASE_BAR * m))}
