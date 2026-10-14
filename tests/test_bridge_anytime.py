"""Bridge: proofs/EdgeOrLuck/Anytime.lean <-> src/edgeorluck/evidence.py.

The Lean file proves, for independent Gaussian trades with a KNOWN spread, that checking again
and again does not fool the judge:

* ``evalue_ville``: the one-sided e-value ever reaches the bar B with probability <= 1/B;
* ``two_sided_ville`` / ``anytime_range`` / ``anytime_range_95``: either side does with
  probability <= 2/B, so the range ever misses the true mean, however often it is checked, with
  probability <= 2/B (5% at BASE_BAR = 40), for every true mean and spread;
* ``above_zero_false_alarm``: with a true mean <= 0, "above zero" ever reaches B with
  probability <= 1/B (the composite null);
* ``martingale_mix`` / ``supermartingale_mix`` / ``integral_lr``: E[e_n] = 1 at every n under
  mean 0, and E[e_n] = integral of exp(n lambda theta) pi(d lambda) <= 1 under a mean theta <= 0.

Where the Lean objects meet the Python ones: ``mixG (halfNormal rho2) n s`` is
``exp(log_e_standardized(n, s, rho2))``; ``eValue pi sigma theta dir X n`` is
``exp(log_e_one_sided(n, total - n * theta, sigma, rho2, dir))``; ``anytimeRange`` is
``range_of`` with skew 0. Not covered by the proof and not tested here: the plug-in spread,
non-Gaussian or dependent trades (VALIDATION.md measures those by simulation).

Simulations use fixed seeds; the Monte Carlo slack is 4 binomial standard errors.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import edgeorluck
from edgeorluck.evidence import (BASE_BAR, Record, boundary, half_width, log_e_one_sided,
                                 log_e_standardized, mixing_rho2, range_of)

B = BASE_BAR                 # 40: one strategy, one version
RHO2 = mixing_rho2(B)        # the code's half-normal mixing scale for that bar
LOG_B = math.log(B)
STEPS = 1000                 # trades per simulated record, checked after EVERY trade
PATHS = 40_000               # simulated records (vectorized through `boundary`)
DIRECT = 1_000               # of them, also judged trade by trade by the Python functions


def _slack(p: float, n: int) -> float:
    return 4.0 * math.sqrt(p * (1.0 - p) / n)


def _phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


@pytest.fixture(scope="module")
def bars() -> np.ndarray:
    """b_n, n = 1..STEPS: the standardized sum at which the e-value reaches B."""
    return np.array([boundary(n, B, RHO2) for n in range(1, STEPS + 1)])


def _simulate(seed: int, mean: float, sd: float, bars: np.ndarray, chunk: int = 8_000):
    """Records of STEPS trades N(mean, sd^2); after every trade, the e-values against mean 0
    with the spread sd known. Returns the share of records where "above", "below", and either
    ever reached B. The first DIRECT records are also judged trade by trade with
    ``log_e_standardized`` itself (at s = +-total / sd, the code's ``mixing_rho2(B)``) and must
    agree with the vectorized check through ``boundary``."""
    rng = np.random.default_rng(seed)
    up = down = either = 0
    for start in range(0, PATHS, chunk):
        totals = np.cumsum(rng.normal(mean, sd, size=(min(chunk, PATHS - start), STEPS)), axis=1)
        s = totals / sd
        cu = (s >= bars).any(axis=1)
        cd = (-s >= bars).any(axis=1)
        if start == 0:
            for p in range(DIRECT):
                du = dd = False
                for n in range(1, STEPS + 1):
                    t = float(totals[p, n - 1])
                    du = du or log_e_standardized(n, t / sd, RHO2) >= LOG_B
                    dd = dd or log_e_standardized(n, -t / sd, RHO2) >= LOG_B
                assert (du, dd) == (bool(cu[p]), bool(cd[p])), p
        up += int(cu.sum())
        down += int(cd.sum())
        either += int((cu | cd).sum())
    return up / PATHS, down / PATHS, either / PATHS


def _half_normal_mixture(n: int, s: float, rho2: float) -> float:
    """mixG (halfNormal rho2) n s: the integral over lambda >= 0 of exp(lambda s - lambda^2 n / 2)
    against the half-normal |N(0, rho2)| density, by the trapezoid rule."""
    a = n + 1.0 / rho2
    top = max(0.0, s / a) + 40.0 / math.sqrt(a)
    lam = np.linspace(0.0, top, 400_001)
    dens = math.sqrt(2.0 / (math.pi * rho2)) * np.exp(lam * s - lam ** 2 * a / 2.0)
    return float(np.trapezoid(dens, lam))


def _mean_e_value(n: int, theta: float) -> float:
    """E[e_n] when the n standardized trades are N(theta, 1): S_n ~ N(n theta, n)."""
    v = 1.0 + RHO2 * n
    width = 14.0 * math.sqrt(n * v)
    s = np.linspace(n * theta - width, n * theta + width, 120_001)
    log_e = np.array([log_e_standardized(n, float(x), RHO2) for x in s])
    log_dens = -((s - n * theta) ** 2) / (2.0 * n) - 0.5 * math.log(2.0 * math.pi * n)
    return float(np.trapezoid(np.exp(log_e + log_dens), s))


def test_e_value_starts_at_one():
    """``mix_zero`` / ``lr_zero``: before any trade the e-value is exactly 1."""
    assert log_e_standardized(0, 0.0, RHO2) == 0.0
    assert log_e_one_sided(0, 0.0, 1.0, RHO2, +1) == 0.0


@pytest.mark.parametrize("n,s", [(1, 0.0), (1, 2.5), (10, -3.0), (30, 9.0), (100, 12.0),
                                 (1000, 80.0), (1000, -40.0)])
def test_code_e_value_is_the_half_normal_mixture(n, s):
    """The Lean e-value ``mixG (halfNormal rho2) n s`` is the code's closed form (glue only;
    the closed form itself is proved in another file)."""
    assert math.exp(log_e_standardized(n, s, RHO2)) == pytest.approx(
        _half_normal_mixture(n, s, RHO2), rel=1e-7)


@pytest.mark.parametrize("n", [1, 5, 30, 100, 1000])
def test_mean_e_value_is_one_under_no_edge(n):
    """``martingale_mix`` (via ``integral_lr`` and Fubini): E[e_n] = E[e_0] = 1 at every n."""
    assert _mean_e_value(n, 0.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n,theta", [(10, -0.05), (100, -0.05), (100, -0.2), (30, 0.1)])
def test_mean_e_value_under_a_shifted_mean(n, theta):
    """``integral_lr`` + Fubini: E[e_n] = integral of exp(lambda theta)^n over the half-normal
    = 2 exp(n^2 theta^2 rho2 / 2) Phi(n theta sqrt(rho2)); at most 1 when theta <= 0
    (``supermartingale_mix``), above 1 when there is an edge."""
    exact = 2.0 * math.exp((n * theta) ** 2 * RHO2 / 2.0) * _phi(n * theta * math.sqrt(RHO2))
    assert _mean_e_value(n, theta) == pytest.approx(exact, rel=1e-6)
    assert (exact <= 1.0) == (theta <= 0.0)


def test_checking_after_every_trade_one_and_two_sided(bars):
    """``evalue_ville`` and ``two_sided_ville``: no edge, standardized trades N(0, 1), the
    e-values checked after every one of 1000 trades. "Above zero" ever reached B in at most
    1/B of the records, "below zero" too, and either of them in at most 2/B."""
    up, down, either = _simulate(20261014, 0.0, 1.0, bars)
    assert up <= 1.0 / B + _slack(1.0 / B, PATHS)
    assert down <= 1.0 / B + _slack(1.0 / B, PATHS)
    assert either <= 2.0 / B + _slack(2.0 / B, PATHS)
    assert either > 0.0          # the e-values do move: the check is not vacuous


def test_no_false_edge_when_the_strategy_is_losing(bars):
    """``above_zero_false_alarm``: trades N(-0.1, 1.7^2) with the spread known; "above zero"
    (against mean 0) ever reached B in at most 1/B of the records."""
    up, _, _ = _simulate(20261015, -0.1, 1.7, bars)
    assert up <= 1.0 / B + _slack(1.0 / B, PATHS)


def test_range_is_anytime_valid_for_any_true_mean():
    """``anytime_range`` / ``anytime_range_95``: trades N(0.37, 2.5^2), spread known, the 95%
    range (``range_of``, BASE_BAR = 40) read after every one of 1000 trades. It ever left out
    the true mean 0.37 in at most 2/B = 5% of the records."""
    theta, sd = 0.37, 2.5
    hw = np.array([half_width(n, sd, B) for n in range(1, STEPS + 1)])
    n = np.arange(1, STEPS + 1)
    rng = np.random.default_rng(20261016)
    missed = 0
    for start in range(0, PATHS, 8_000):
        totals = np.cumsum(rng.normal(theta, sd, size=(8_000, STEPS)), axis=1)
        mean = totals / n
        miss = ((theta < mean - hw) | (theta > mean + hw)).any(axis=1)
        if start == 0:
            for p in range(DIRECT):
                direct = False
                for k in range(1, STEPS + 1):
                    lo, hi = range_of(Record(k, float(totals[p, k - 1]), sd, 0.0), B)
                    direct = direct or not (lo <= theta <= hi)
                assert direct == bool(miss[p]), p
        missed += int(miss.sum())
    rate = missed / PATHS
    assert rate <= 2.0 / B + _slack(2.0 / B, PATHS)
    assert rate > 0.0


def test_range_leaves_out_a_mean_exactly_when_an_e_value_reaches_the_bar():
    """``anytimeRange`` is ``range_of``: a mean theta is outside the code's range iff one of the
    two one-sided e-values against theta reaches B (off a hair-thin band at the ends, where the
    open set and the closed interval differ only by the end points)."""
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(4_000):
        n = int(rng.integers(2, 3_000))
        sd = float(np.exp(rng.uniform(math.log(0.05), math.log(20.0))))
        total = float(rng.normal(0.0, sd * math.sqrt(n)) + rng.normal(0.0, 0.3) * sd * n)
        lo, hi = range_of(Record(n, total, sd, 0.0), B)
        mid, hw = (lo + hi) / 2.0, (hi - lo) / 2.0
        u = float(rng.uniform(-0.5, 0.5))
        if abs(u) < 1e-6:
            continue
        theta = mid + float(rng.choice([-1.0, 1.0])) * hw * (1.0 + u)
        rejected = max(log_e_one_sided(n, total - n * theta, sd, RHO2, +1),
                       log_e_one_sided(n, total - n * theta, sd, RHO2, -1)) >= LOG_B
        assert (lo <= theta <= hi) == (not rejected), (n, sd, total, theta)
        checked += 1
    assert checked > 3_900


def test_a_fixed_95_interval_checked_after_every_trade_is_fooled():
    """Why the time-uniform range is needed (README: "A normal confidence interval checked
    every day is wrong far more often than it says"): a fixed 95% z-interval read after every
    one of 1000 no-edge trades leaves out the true mean in far more than 5% of the records,
    while the anytime range stays under 2/B (the tests above)."""
    rng = np.random.default_rng(11)
    s = np.cumsum(rng.standard_normal((8_000, STEPS)), axis=1)
    fooled = (np.abs(s) / np.sqrt(np.arange(1, STEPS + 1)) > 1.959964).any(axis=1).mean()
    assert fooled > 0.25


def test_range_shown_from_two_trades_keeps_the_95_promise():
    """The page: "It stays honest even if you check after every trade". Gaussian trades
    N(0.37, 2.5^2); a range the judge shows at 2 trades must leave out the true mean in at most
    2/B = 5% of the records.

    This test found a false claim (fixed 2026-10-14): judge() used to show range_95 from 2 trades,
    with the PLUG-IN spread. At 2 trades the spread is |x1 - x2| / sqrt(2), so the range missed
    exactly when a standard Cauchy variable exceeds b_2 / sqrt(2): (2/pi) atan(sqrt(2) / b_2) = 9.1%
    on a single look, and about 12.6% for a trader checking after every trade from 2 to 300 trades
    (the Lean guarantee assumes a KNOWN spread and does not cover this). The judge now shows no
    range before MIN_TRADES = 30, where verdicts start; from 30 on it is about 2.2%."""
    theta, sd, paths = 0.37, 2.5, 6_000
    rng = np.random.default_rng(12)
    missed = 0
    for _ in range(paths):
        x = [float(v) for v in rng.normal(theta, sd, 2)]
        card = edgeorluck.judge(data=x, include_chart=False)["strategies"][0]
        lo, hi = card["numbers"]["mean_per_trade"]["range_95"]
        assert lo is None and hi is None, "no range is promised below 30 trades"
        missed += lo is not None and hi is not None and not (lo <= theta <= hi)
    assert missed / paths <= 2.0 / B + _slack(2.0 / B, paths)
