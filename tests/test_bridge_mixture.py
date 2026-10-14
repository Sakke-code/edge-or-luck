"""Bridge tests: the Lean proofs in proofs/EdgeOrLuck/Mixture.lean against evidence.py.

Lean proves statements about real numbers (exact arithmetic). These tests check that the Python
code computes the same objects in floating point, and they measure what Lean does not prove
(the tuning heuristic). Each test names the Lean theorems it bridges.

The mixture integral is computed here by quadrature straight from its definition,
e(n, s) = int_{l >= 0} exp(l s - l^2 n / 2) * 2 / sqrt(2 pi rho2) * exp(-l^2 / (2 rho2)) dl,
never from the closed form the code uses.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from edgeorluck.evidence import (BASE_BAR, SKEW_CAP, T_STAR, Record, boundary, half_width,
                                 log_e_record, log_e_standardized, mixing_rho2, range_of,
                                 record_of, skew_shift)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)
SWITCH_Z = -37.0         # _log_two_phi switches from erfc to its four-term series at z <= -37
RHO2S = [mixing_rho2(BASE_BAR), mixing_rho2(400.0), mixing_rho2(40_000.0), 1e-3, 0.5, 2.0]
NS = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 10_000]


def _gauss_legendre(lo: float, hi: float, panels: int):
    edges = np.linspace(lo, hi, panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    return (mid + half * _GL_X[None, :]).ravel(), (half * _GL_W[None, :]).ravel()


def log_mixture_quad(n: float, s: float, rho2: float, panels: int = 60) -> float:
    """log of the mixture integral by composite Gauss-Legendre. The window is read off the
    exponent l s - a l^2 / 2 (a = n + 1 / rho2) of the integrand, not from the closed form."""
    a = n + 1.0 / rho2
    peak = max(0.0, s / a)                      # the exponent's maximum on l >= 0
    width = 40.0 / math.sqrt(a)
    if s < 0:
        width = min(width, 60.0 / -s)           # exp(l s) alone is down by e^-60 there
    l, w = _gauss_legendre(max(0.0, peak - width), peak + width, panels)
    top = peak * s - a * peak * peak / 2.0
    total = float(np.sum(w * np.exp(l * s - a * l * l / 2.0 - top)))
    return top + math.log(total) + math.log(2.0 / math.sqrt(2.0 * math.pi * rho2))


def z_of(n: float, s: float, rho2: float) -> float:
    return math.sqrt(rho2) * s / math.sqrt(1.0 + rho2 * n)


def s_of(n: float, z: float, rho2: float) -> float:
    return z * math.sqrt(1.0 + rho2 * n) / math.sqrt(rho2)


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ── 1. closed form: eMix_closed_form, eMix_eq_exp_logEStd ─────────────────────────────

def _closed_form_cases():
    for n in NS:
        for rho2 in RHO2S:
            zs = list(np.linspace(-7.9, 12.0, 23))
            ss = [s_of(n, z, rho2) for z in zs] + [k * math.sqrt(n) for k in (-3, -1, -0.1, 0, 0.1, 1, 3, 10)]
            for s in ss:
                if z_of(n, s, rho2) > SWITCH_Z:
                    yield n, s, rho2


def test_closed_form_matches_the_mixture_integral():
    """log_e_standardized(n, s, rho2) = log of the half-normal mixture integral (1e-9 relative)."""
    worst, checked = 0.0, 0
    for n, s, rho2 in _closed_form_cases():
        code = log_e_standardized(n, s, rho2)
        quad = log_mixture_quad(n, s, rho2)
        worst = max(worst, abs(code - quad))
        checked += 1
        assert abs(code - quad) <= 1e-9, (n, s, rho2, code, quad)
    assert checked > 1500 and worst < 1e-9


def test_closed_form_far_below_zero():
    """Found a defect (fixed 2026-10-14): for z <= -8 _log_two_phi kept only the leading Mills-ratio
    term phi(z)/|z|, which overstated Phi(z) (log error 0.0151 at z = -8). Decision-irrelevant, since
    e < 1 there; now erfc down to z = -37 and a four-term series below."""
    for n in (10, 100, 1000):
        rho2 = mixing_rho2(BASE_BAR)
        for z in (-8.0, -9.0, -12.0, -20.0):
            s = s_of(n, z, rho2)
            assert abs(log_e_standardized(n, s, rho2) - log_mixture_quad(n, s, rho2)) <= 1e-9


def test_far_below_zero_is_exact_and_never_decides():
    """Far below zero the e-value is now exact too (it used to be overstated by up to 1.6% past
    z = -8), and it stays far below every bar."""
    for n in (1, 10, 100, 1000, 10_000):
        for rho2 in RHO2S:
            for z in (-8.0, -8.5, -10.0, -15.0, -25.0, -35.0):
                s = s_of(n, z, rho2)
                code, quad = log_e_standardized(n, s, rho2), log_mixture_quad(n, s, rho2)
                assert abs(code - quad) <= 1e-7 * max(1.0, abs(quad))
                assert code < math.log(1.0)               # e < 1 < any bar (bars are >= 40)


# ── 3. monotone in s: eMix_strictMono, logEStd_strictMono ─────────────────────────────

def test_strictly_increasing_in_s():
    for n in NS:
        for rho2 in RHO2S:
            s = np.linspace(s_of(n, -7.99, rho2), s_of(n, 12.0, rho2), 4001)
            vals = np.array([log_e_standardized(n, float(x), rho2) for x in s])
            assert np.all(np.diff(vals) > 0), (n, rho2)


def test_strictly_increasing_across_the_branch_switch():
    """Found a defect (fixed 2026-10-14): log_e_standardized dropped by 0.0151 as s crossed z = -8,
    where _log_two_phi used to switch to a one-term approximation. It now switches at z = -37, where
    erfc and the four-term series agree to about 1e-10, so it keeps rising across both points."""
    n, rho2 = 100, mixing_rho2(BASE_BAR)
    for zs in (-8.0, SWITCH_Z):
        s0 = s_of(n, zs, rho2)                                # negative
        lower_s, higher_s = s0 * (1 + 1e-6), s0 * (1 - 1e-6)  # z just under / just over the point
        assert z_of(n, lower_s, rho2) < zs < z_of(n, higher_s, rho2)
        assert log_e_standardized(n, lower_s, rho2) < log_e_standardized(n, higher_s, rho2)


# ── 4. the value at zero and the down side: eMix_zero, eMix_zero_le_one, eMix_neg_le_one ──

def test_value_at_zero_and_down_side():
    for n in NS:
        for rho2 in RHO2S:
            v = 1.0 + rho2 * n
            assert log_e_standardized(n, 0.0, rho2) == pytest.approx(-0.5 * math.log(v), abs=1e-12)
            assert log_e_standardized(n, 0.0, rho2) <= 0.0
            for s in (0.0, 0.5, 1.0, 3.0, 10.0, 100.0, 1e4) + tuple(np.sqrt(n) * k for k in (1, 3, 30)):
                assert log_e_standardized(n, -float(s), rho2) <= 0.0


# ── 5. boundary: boundary_bracket, boundary_exists_unique, boundaryCode_spec ─────────────

def _root_by_quadrature(n, bar, rho2, hi):
    lo, top, target = 0.0, hi, math.log(bar)
    for _ in range(200):
        mid = 0.5 * (lo + top)
        if log_mixture_quad(n, mid, rho2) >= target:
            top = mid
        else:
            lo = mid
    return top


@pytest.mark.parametrize("bar", [1.5, 40.0, 400.0, 40_000.0, 1e6])
def test_bisection_bracket_and_root(bar):
    for n in (1, 2, 5, 20, 100, 1000, 10_000):
        for rho2 in (mixing_rho2(bar), 1e-3, 2.0):
            v = 1.0 + rho2 * n
            target = math.log(bar)
            hi = math.sqrt(2.0 * v / rho2 * (target + 0.5 * math.log(v) + 1.0)) + 1.0
            assert log_e_standardized(n, 0.0, rho2) < target                  # e(lo) < B
            assert log_e_standardized(n, hi, rho2) >= target + 1.0            # e(hi) >= e * B
            b = boundary(n, bar, rho2)
            assert 0.0 < b < hi
            assert log_e_standardized(n, b, rho2) >= target                   # the code keeps e(hi_k) >= B
            truth = _root_by_quadrature(n, bar, rho2, hi)
            assert abs(b - truth) <= 1e-9 * b, (n, bar, rho2, b, truth)


# ── 6. range <=> verdict: verdict_iff_range_clears_zero, down_verdict_..., range_eq_not_rejected,
#       above_rejects_iff, below_rejects_iff, pass_implies_total_pos ──────────────────────

def _random_trades(rng, kind, n):
    if kind == "normal":
        x = rng.normal(0.0, 1.0, n)
    elif kind == "martingale":          # small wins, rare big losses: negative skew
        x = np.where(rng.random(n) < 0.97, 1.0, -30.0) + rng.normal(0.0, 0.05, n)
    elif kind == "lottery":             # small losses, rare big wins: positive skew
        x = np.where(rng.random(n) < 0.95, -1.0, 25.0) + rng.normal(0.0, 0.05, n)
    elif kind == "lognormal":
        x = rng.lognormal(0.0, 1.0, n) - math.exp(0.5)
    elif kind == "t3":
        x = rng.standard_t(3, n)
    else:                               # AR(1) 0.5: neighbouring trades move together
        e = rng.normal(0.0, 1.0, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + e[i]
    x = x - x.mean() + rng.normal(0.0, 0.4) * x.std()      # a random true average
    return x * 10.0 ** rng.uniform(-3, 3)


def _records(seed=20261014, count=600):
    rng = np.random.default_rng(seed)
    kinds = ["normal", "martingale", "lottery", "lognormal", "t3", "ar1"]
    for i in range(count):
        n = int(rng.choice([2, 3, 5, 8, 12, 20, 30, 50, 100, 250, 600]))
        rec = record_of(_random_trades(rng, kinds[i % len(kinds)], n))
        if math.isfinite(rec.sd) and rec.sd > 0:
            yield rec


@pytest.mark.parametrize("bar", [BASE_BAR, 80.0, 400.0])
def test_range_and_verdict_are_the_same_statement(bar):
    rho2 = mixing_rho2(bar)
    target = math.log(bar)
    agree = ties = passes = 0
    for rec in _records():
        n, sd = rec.n, rec.sd
        b = boundary(n, bar, rho2)
        su, sdn = skew_shift(rec.skew, n, bar, rho2, +1), skew_shift(rec.skew, n, bar, rho2, -1)
        lo, hi = range_of(rec, bar, rho2)
        assert lo == pytest.approx(rec.mean - (b + su) * sd / n, rel=1e-12, abs=1e-300)
        assert hi == pytest.approx(rec.mean + (b + sdn) * sd / n, rel=1e-12, abs=1e-300)
        up = log_e_record(rec, bar, +1, rho2) >= target
        down = log_e_record(rec, bar, -1, rho2) >= target
        band_lo = 1e-9 * (b + su) * sd / n
        band_hi = 1e-9 * (b + sdn) * sd / n
        if abs(lo) > band_lo and abs(hi) > band_hi:
            assert up == (lo >= 0), (rec, lo)
            assert down == (hi <= 0), (rec, hi)
            agree += 1
        else:
            ties += 1
        if up:
            passes += 1
            assert rec.total > 0                          # pass_implies_total_pos
        assert not (up and down)
        # right at the boundary: move the total so the lower end sits just above / below zero
        for eps, expect in ((1e-7, True), (-1e-7, False)):
            moved = dataclasses.replace(rec, total=rec.total - n * lo + eps * n * (b + su) * sd / n)
            assert (log_e_record(moved, bar, +1, rho2) >= target) == expect
    assert agree > 500 and ties == 0 and passes > 30


def test_range_is_the_set_of_means_not_rejected():
    bar = BASE_BAR
    rho2 = mixing_rho2(bar)
    target = math.log(bar)
    for rec in list(_records(seed=7, count=120)):
        lo, hi = range_of(rec, bar, rho2)
        width = hi - lo
        for t in np.linspace(-0.5, 1.5, 41):
            mu0 = lo + t * width
            shifted = dataclasses.replace(rec, total=rec.total - rec.n * mu0)   # the test against mu0
            above = log_e_record(shifted, bar, +1, rho2) >= target
            below = log_e_record(shifted, bar, -1, rho2) >= target
            if min(abs(mu0 - lo), abs(mu0 - hi)) <= 1e-9 * width:
                continue
            assert above == (mu0 <= lo)                   # above_rejects_iff
            assert below == (mu0 >= hi)                   # below_rejects_iff
            assert (lo < mu0 < hi) == (not above and not below)   # range_eq_not_rejected


def test_no_skew_range_is_mean_plus_minus_half_width():
    for rec in list(_records(seed=11, count=60)):
        flat = dataclasses.replace(rec, skew=0.0)
        hw = half_width(rec.n, rec.sd, BASE_BAR)
        lo, hi = range_of(flat, BASE_BAR)
        assert lo == pytest.approx(rec.mean - hw, rel=1e-12, abs=1e-12 * hw)
        assert hi == pytest.approx(rec.mean + hw, rel=1e-12, abs=1e-12 * hw)


# ── 7. skew never lowers a bar: skewShift_nonneg, skewShift_zero_skew, skew_never_raises_e,
#       range_with_skew_contains ───────────────────────────────────────────────────────

def test_skew_shift_is_never_negative():
    rng = np.random.default_rng(3)
    skews = list(rng.uniform(-12, 12, 300)) + [0.0, SKEW_CAP, -SKEW_CAP, 1e-12, -1e-12, 50.0, -50.0]
    for skew in skews:
        for n in (1, 2, 3, 4, 10, 100, 5000):
            for bar in (1.5, BASE_BAR, 400.0, 1e6):
                rho2 = mixing_rho2(bar)
                for sign in (+1, -1):
                    shift = skew_shift(float(skew), n, bar, rho2, sign)
                    assert shift >= 0.0
                    if skew == 0.0 or skew * sign > 0 or n < 3:
                        assert shift == 0.0


def test_skew_only_widens_the_range_and_lowers_the_e_value():
    for bar in (BASE_BAR, 400.0):
        rho2 = mixing_rho2(bar)
        for rec in _records(seed=5, count=300):
            flat = dataclasses.replace(rec, skew=0.0)
            lo, hi = range_of(rec, bar, rho2)
            flo, fhi = range_of(flat, bar, rho2)
            assert lo <= flo and fhi <= hi
            for sign in (+1, -1):
                assert log_e_record(rec, bar, sign, rho2) <= log_e_record(flat, bar, sign, rho2)


# ── 2. it is an e-value: eMix_expectation_eq_one, eMix_expectation_le_one ─────────────────

def _mean_e_by_quadrature(n, mu, rho2, panels=200):
    """E[e(n, S)] for S ~ N(mu, n), by quadrature over s (the code's e-value inside)."""
    v = 1.0 + rho2 * n
    centre, spread = mu * v, math.sqrt(n * v)        # where e(n, s) * density(s) lives
    s, w = _gauss_legendre(centre - 40 * spread, centre + 40 * spread, panels)
    log_e = np.array([log_e_standardized(n, float(x), rho2) for x in s])
    log_density = -((s - mu) ** 2) / (2 * n) - 0.5 * math.log(2 * math.pi * n)
    return float(np.sum(w * np.exp(log_e + log_density)))


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_mean_e_value_is_one_with_no_edge_and_below_one_when_losing(n):
    rho2 = mixing_rho2(BASE_BAR)
    for k in (0.0, -0.1, -1.0, -3.0, 0.5):
        mu = k * math.sqrt(n)
        # E[e] = int_{l >= 0} exp(l mu) pi(l) dl = 2 exp(mu^2 rho2 / 2) Phi(mu sqrt(rho2))  (Tonelli + MGF)
        exact = 2.0 * math.exp(mu * mu * rho2 / 2.0) * normal_cdf(mu * math.sqrt(rho2))
        got = _mean_e_by_quadrature(n, mu, rho2)
        assert got == pytest.approx(exact, rel=1e-8, abs=1e-12)
        if k == 0.0:
            assert got == pytest.approx(1.0, abs=1e-8)
        elif k < 0:
            assert got < 1.0
        else:
            assert got > 1.0                          # an edge is rewarded


@pytest.mark.parametrize("n", [1, 5, 10])
def test_mean_e_value_monte_carlo(n):
    """Monte Carlo sanity check. Below n = 1 / rho2 (13.6 at bar 40) e has a finite variance:
    E[e^2] = (1 + (2/pi) asin(r)) / sqrt(1 - r^2) with r = rho2 n."""
    rho2 = mixing_rho2(BASE_BAR)
    rng = np.random.default_rng(20261014 + n)
    draws = 200_000
    s = rng.normal(0.0, math.sqrt(n), draws)
    e = np.exp([log_e_standardized(n, float(x), rho2) for x in s])
    r = rho2 * n
    second = (1.0 + 2.0 / math.pi * math.asin(r)) / math.sqrt(1.0 - r * r)
    se = math.sqrt((second - 1.0) / draws)
    assert abs(e.mean() - 1.0) <= 6.0 * se, (e.mean(), se)


# ── 8. the tuning: logEStd_scale_free (proved) and "decides soonest near T_STAR" (measured) ──

def _z_boundary(r, bar, t=T_STAR):
    return boundary(t, bar, r / t) / math.sqrt(t)


def _best_r(bar):
    a, b = math.log(1.0), math.log(60.0)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _z_boundary(math.exp(c), bar), _z_boundary(math.exp(d), bar)
    for _ in range(120):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _z_boundary(math.exp(c), bar)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _z_boundary(math.exp(d), bar)
    return math.exp(0.5 * (a + b))


def test_only_rho2_times_n_matters():
    for bar in (BASE_BAR, 400.0):
        for r in (0.5, 2 * math.log(bar), 20.0):
            zs = [boundary(n, bar, r / n) / math.sqrt(n) for n in (3, 10, 100, 1000, 10_000)]
            assert max(zs) - min(zs) <= 1e-9 * zs[0]
            for n in (3, 10, 100, 1000):
                for s in (-2.0, 0.0, 1.0, 4.0):
                    assert log_e_standardized(n, s * math.sqrt(n), r / n) == pytest.approx(
                        log_e_standardized(1, s, r), abs=1e-11)


@pytest.mark.parametrize("bar, best, excess_max, n_best", [
    (BASE_BAR, 8.24, 3.5e-4, (110, 114)),
    (400.0, 13.26, 2.0e-4, (109, 113)),
    (4000.0, 18.15, 1.2e-4, (108, 112)),
])
def test_tuning_is_a_close_heuristic(bar, best, excess_max, n_best):
    """mixing_rho2 = 2 log(bar) / T_STAR is NOT the rho2 that decides soonest at T_STAR (it is the
    leading-order approximation), but the boundary it gives at T_STAR is within 0.035% of the
    best, and it is the exact optimum a little later (about 1.1 T_STAR)."""
    r_code = mixing_rho2(bar) * T_STAR
    assert r_code == pytest.approx(2.0 * math.log(bar))
    r_best = _best_r(bar)
    assert r_best == pytest.approx(best, abs=0.02)
    assert 1.05 < r_best / r_code < 1.15                     # not the optimum ...
    excess = _z_boundary(r_code, bar) / _z_boundary(r_best, bar) - 1.0
    assert 0.0 < excess < excess_max                         # ... but a close one
    n_opt = T_STAR * r_best / r_code                         # where the code's rho2 IS optimal
    assert n_best[0] <= n_opt <= n_best[1]
