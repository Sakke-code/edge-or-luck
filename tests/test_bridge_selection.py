"""Bridge tests for proofs/EdgeOrLuck/Selection.lean: the Python code satisfies exactly the
hypotheses and conclusions of the Lean theorems on selection, e-value combination and the small
algebraic claims.

The Lean statements are about real numbers. These tests run the float code on many random
inputs (ties, zeros, unseen hypotheses, knife-edge thresholds) and check it against the theorems
and their closed forms. Each test names the theorem it bridges. Two tests are marked xfail: they
pin false sentences of judge._money_card found while proving section 7.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from edgeorluck import compare, evidence
from edgeorluck.compare import compare_runs
from edgeorluck.evidence import (ALPHA, BASE_BAR, Q, describe_family, ebh, half_width, log_e_record,
                                 max_versions_passing, normal_tail, record_of, wilson_bounds_count)
from edgeorluck.judge import MIN_TRADES, _measure, _money_card, judge_dict
from edgeorluck.readers import ReadResult, Trades
from edgeorluck.web.ledger import SHRINKAGE, Ledger, forward_probabilities
from edgeorluck.words import rnd

NEG_INF = float("-inf")


# ── 1. Markov for e-values (evalue_markov, base_bar_levels) ───────────────────

def test_base_bar_is_two_over_q_and_its_markov_level_is_half_of_alpha():
    # base_bar_levels: 2 / (1/20) = 40 and 1/40 = (1/20)/2
    assert BASE_BAR == 2.0 / Q == 40.0
    assert 1.0 / BASE_BAR == pytest.approx(ALPHA / 2, abs=1e-17)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_markov_bound_holds_for_gaussian_likelihood_ratio_evalues(theta):
    # E = exp(theta Z - theta^2/2), Z ~ N(0,1), has E[E] = 1 exactly. evalue_markov says
    # P(E >= 1/alpha) <= alpha; the closed form is P(Z >= (log(1/alpha) + theta^2/2) / theta).
    for alpha in (0.5, 0.2, 0.1, 0.05, 1 / 40, 1 / 400, 1e-6):
        p = normal_tail((math.log(1 / alpha) + theta * theta / 2) / theta)
        assert p <= alpha


# ── 2. Averages and bests (average_isEValue, max_div_le_average, best_*) ──────

def test_describe_family_is_two_hypotheses_and_bar_40_per_version():
    for versions in [None, 0, 1, 2, 3, 7, 10, 40, 100, 1000]:
        for strategies in [0, 1, 2, 5, 12]:
            fam = describe_family(versions, strategies)
            m = max(int(versions or 1), int(strategies or 1), 1)
            assert fam["versions_or_strategies"] == m
            assert fam["hypotheses"] == 2 * m
            assert fam["bar_alone"] == 40 * m
            assert fam["bar_alone"] == pytest.approx(2 * m / Q, rel=1e-15)  # = 2N/q, the e-BH bar


def test_the_best_divided_by_n_is_at_most_the_average_pointwise():
    rng = np.random.default_rng(11)
    for _ in range(2000):
        n = int(rng.integers(1, 60))
        e = rng.lognormal(0.0, 2.0, n) * (rng.random(n) < 0.8)          # zeros included
        if rng.random() < 0.3:
            e[: max(1, n // 2)] = e[0]                                   # ties
        assert e.max() / n <= e.sum() / n


@pytest.mark.parametrize("theta", [0.5, 1.0, 2.0, 4.0])
def test_best_of_n_must_clear_n_over_alpha(theta):
    # best_of_N_markov: P(max_i E_i >= N/alpha) <= alpha under any dependence. Closed forms for
    # independent versions, 1 - (1 - p)^N, and for identical versions, p, with p = P(E >= N/alpha).
    for n_versions in (1, 2, 5, 10, 40, 100, 1000):
        for alpha in (1 / 40, 0.05, 0.2):
            c = n_versions / alpha
            p = normal_tail((math.log(c) + theta * theta / 2) / theta)
            assert 1 - (1 - p) ** n_versions <= alpha + 1e-15
            assert p <= alpha


def test_max_versions_passing_uses_the_bar_40n():
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(120):
        x = rng.normal(float(rng.uniform(0.2, 0.6)), 1.0, int(rng.integers(60, 400)))
        rec = record_of(x)
        top = max_versions_passing(rec)
        if top == 0 or top >= 1_000_000:
            continue
        bar = BASE_BAR * top
        assert log_e_record(rec, bar, +1) >= math.log(bar)
        bar_next = BASE_BAR * (top + 1)
        assert log_e_record(rec, bar_next, +1) < math.log(bar_next)
        checked += 1
    assert checked >= 20


# ── 3 and 4. e-BH (ebh_fdr, card_ebhSet, ebhSet_selfConsistent, pyTop_eq_ebhSet, ...) ──

def lean_ebh(log_e, q, total):
    """The Lean definitions, in log space with the code's own thresholds log(m / (k q)):
    k* = max{k in [1, m] : #{j : e_j >= m/(q k)} >= k}, R = {j : e_j >= m/(q k*)}.
    Unseen hypotheses (total > len) are e = 0 (log -inf) and never counted."""
    m = max(len(log_e), int(total or 0))
    if not m or not log_e:
        return set(), 0

    def thr(k):
        return math.log(m / (k * q))

    def count(k):
        t = thr(k)
        return sum(1 for v in log_e.values() if v >= t)

    kstar = max((k for k in range(1, m + 1) if count(k) >= k), default=0)
    chosen = {key for key, v in log_e.items() if kstar > 0 and v >= thr(kstar)}
    return chosen, kstar


POOL = [NEG_INF, -3.0, 0.0, math.log(0.5), math.log(5), math.log(19.9), math.log(20), math.log(20.1),
        math.log(40), math.log(80), math.log(100), math.log(400), math.log(1e4), 50.0]


def random_family(rng, ties=True):
    size = rng.randint(1, 14)
    total = rng.choice([None, 0, size, size + rng.randint(1, 30)])
    m = max(size, int(total or 0))
    values = []
    for _ in range(size):
        u = rng.random()
        if ties and u < 0.45:
            values.append(rng.choice(POOL))
        elif u < 0.65:
            k = rng.randint(1, m)
            values.append(math.log(m / (k * Q)))        # exactly at a threshold the code computes
        else:
            values.append(rng.uniform(-5.0, 12.0))
    return {f"h{i}": v for i, v in enumerate(values)}, total


def test_python_ebh_is_the_lean_ebh_set_with_size_kstar():
    rng = random.Random(2026)
    for _ in range(5000):
        log_e, total = random_family(rng)
        q = rng.choice([Q, 0.1, 0.01, 0.2])
        got = ebh(log_e, q, total)
        want, kstar = lean_ebh(log_e, q, total)
        assert got == want                                   # pyTop_eq_ebhSet, pyChosen_listed
        assert len(got) == kstar                             # card_ebhSet


def test_ebh_selection_is_self_consistent():
    # ebhSet_selfConsistent: every selected j has e_j q |R| >= m, i.e. log e_j >= log(m / (|R| q)).
    rng = random.Random(7)
    for _ in range(5000):
        log_e, total = random_family(rng)
        got = ebh(log_e, Q, total)
        m = max(len(log_e), int(total or 0))
        for key in got:
            assert log_e[key] >= math.log(m / (len(got) * Q))


def test_zero_and_small_evalues_are_never_selected():
    # notMem_ebhSet_of_nonpos / notMem_ebhSet_of_lt: e = 0 (log -inf), or e < 1/q, is never selected.
    rng = random.Random(3)
    for _ in range(5000):
        log_e, total = random_family(rng)
        got = ebh(log_e, Q, total)
        for key, v in log_e.items():
            if v == NEG_INF or v < math.log(1 / Q) - 1e-12:
                assert key not in got


def test_ties_never_change_the_selection():
    # pyTop_eq_ebhSet holds for every descending ranking: Python's stable sort breaks ties by
    # insertion order, so any insertion order must give the same set.
    rng = random.Random(9)
    for _ in range(1500):
        log_e, total = random_family(rng)
        base = ebh(log_e, Q, total)
        items = list(log_e.items())
        for _ in range(4):
            rng.shuffle(items)
            assert ebh(dict(items), Q, total) == base


def test_no_tie_straddles_the_cut():
    # ebh_no_tie_at_cut: if 0 < k* < (listed), the value ranked k*+1 is strictly below the k*-th.
    rng = random.Random(13)
    for _ in range(5000):
        log_e, total = random_family(rng)
        _, kstar = lean_ebh(log_e, Q, total)
        ranked = sorted(log_e.values(), reverse=True)
        if 0 < kstar < len(ranked):
            assert ranked[kstar] < ranked[kstar - 1]


def test_unseen_hypotheses_only_raise_the_bar():
    # pyChosen_listed: unseen versions (total > len) are e = 0; they are never selected and the
    # selection equals the Lean set over the whole family of size total.
    rng = random.Random(21)
    for _ in range(2000):
        size = rng.randint(1, 6)
        log_e = {f"h{i}": rng.uniform(0.0, 12.0) for i in range(size)}
        total = size + rng.randint(0, 50)
        got = ebh(log_e, Q, total)
        assert got <= set(log_e)
        assert got == lean_ebh(log_e, Q, total)[0]


def test_log_comparison_is_evalue_comparison():
    # python_log_compare: for e > 0 and a positive threshold, log t <= log e iff t <= e.
    rng = random.Random(1)
    for _ in range(20000):
        e = math.exp(rng.uniform(-20, 20))
        t = math.exp(rng.uniform(-20, 20))
        if abs(e - t) > 1e-12 * max(e, t):
            assert (math.log(t) <= math.log(e)) == (t <= e)


def test_fdp_of_the_ebh_selection_is_bounded_pointwise():
    # fdp_le_sum / indicator_div_le: FDP(R) <= sum_{j in H0} q e_j / m for any H0.
    rng = random.Random(17)
    for _ in range(3000):
        log_e, total = random_family(rng)
        keys = list(log_e)
        got = ebh(log_e, Q, total)
        m = max(len(log_e), int(total or 0))
        for _ in range(3):
            h0 = {k for k in keys if rng.random() < 0.6}
            fdp = len(got & h0) / max(len(got), 1)
            bound = sum(Q * math.exp(min(log_e[k], 700)) / m for k in h0)
            assert fdp <= bound * (1 + 1e-12) + 1e-15


@pytest.mark.parametrize("theta", [1.0, 2.0, 3.0])
def test_fully_dependent_nulls_reach_at_most_q(theta):
    # ebh_fdr with all m nulls equal to one e-value E = exp(theta Z - theta^2/2): e-BH selects all
    # of them iff E >= 1/q, so FDR = P(E >= 1/q), which ebh_fdr bounds by q |H0| / m = q.
    m = 10
    threshold_z = (math.log(m / (m * Q)) + theta * theta / 2) / theta
    for z in np.linspace(-4, 6, 401):
        lv = theta * z - theta * theta / 2
        if abs(z - threshold_z) < 1e-9:
            continue
        got = ebh({f"h{i}": lv for i in range(m)}, Q, m)
        assert got == (set(f"h{i}" for i in range(m)) if z > threshold_z else set())
    assert normal_tail(threshold_z) <= Q


def test_ebh_false_discovery_rate_by_simulation():
    # ebh_fdr / ebhSet_fdr: empirical FDR below q |H0| / m, with dependent nulls (a shared factor).
    rng = np.random.default_rng(4)
    m, nulls, reps, theta = 20, 15, 4000, 2.0
    fdps = np.empty(reps)
    for r in range(reps):
        shared = rng.normal()
        z = 0.7 * shared + math.sqrt(1 - 0.49) * rng.normal(size=m)      # correlated standard normals
        z[nulls:] += 3.5                                                 # the real effects
        log_e = {i: theta * z[i] - theta * theta / 2 for i in range(m)}
        got = ebh(log_e, Q, m)
        fdps[r] = sum(1 for i in got if i < nulls) / max(len(got), 1)
    se = fdps.std(ddof=1) / math.sqrt(reps)
    assert fdps.mean() <= Q * nulls / m + 4 * se


# ── 5. One strategy, N versions (single_strategy_rule', single_strategy_rule_q05) ──

def test_one_strategy_is_selected_iff_e_up_reaches_2n_over_q():
    rng = random.Random(5)
    for _ in range(20000):
        n_versions = rng.choice([1, 2, 3, 7, 10, 40, 100, 1000, rng.randint(1, 5000)])
        bar_log = math.log(2 * n_versions / Q)
        log_up = rng.choice([rng.uniform(-10, 15), bar_log, bar_log + rng.uniform(-1e-9, 1e-9)])
        log_down = rng.uniform(-30, 0.0)                     # e_down <= 1
        if rng.random() < 0.5:
            log_up, log_down = min(log_up, 0.0), rng.uniform(-10, 15)   # or e_up <= 1
        got = ebh({"0#up": log_up, "0#down": log_down}, Q, 2 * n_versions)
        assert ("0#up" in got) == (log_up >= bar_log)
        assert 2 * n_versions / Q == pytest.approx(40 * n_versions, rel=1e-15)


def test_the_hypothesis_e_down_at_most_one_is_needed():
    # Without min(e_up, e_down) <= 1 the rule fails: e_up = 30, e_down = 25, one version (m = 2):
    # k = 2 qualifies (both >= 2/(2 q) = 20), so "up" is selected although e_up < 40.
    got = ebh({"0#up": math.log(30), "0#down": math.log(25)}, Q, 2)
    assert got == {"0#up", "0#down"}


def records(rng, count):
    for _ in range(count):
        n = int(rng.integers(MIN_TRADES, 400))
        mu = float(rng.uniform(-0.3, 0.6))
        yield rng.normal(mu, 1.0, n) if rng.random() < 0.7 else rng.standard_t(4, n) + mu


def test_the_judge_selects_one_strategy_iff_e_up_reaches_40n():
    # judge.analyze's own path: _measure at the family bar, then ebh(log_e, Q, total=2*family).
    rng = np.random.default_rng(8)
    seen = {True: 0, False: 0}
    for x in records(rng, 300):
        n_versions = int(rng.choice([1, 2, 5, 10, 40, 200]))
        mm = _measure(Trades("s", "%", x), BASE_BAR * n_versions)
        assert min(mm["log_e_up"], mm["log_e_down"]) <= 0.0       # the premise (mixture lane)
        got = ebh({"0#up": mm["log_e_up"], "0#down": mm["log_e_down"]}, Q, 2 * n_versions)
        up = "0#up" in got
        assert up == (mm["log_e_up"] >= math.log(2 * n_versions / Q))
        seen[up] += 1
    assert seen[True] >= 20 and seen[False] >= 20


def test_the_reported_evidence_passes_iff_e_reaches_the_bar():
    rng = np.random.default_rng(12)
    seen = {True: 0, False: 0}
    for x in records(rng, 40):
        n_versions = int(rng.choice([1, 3, 10]))
        result = judge_dict([float(v) for v in x], variants_tried=n_versions, details=False)
        ev = result["strategies"][0]["numbers"]["evidence"]
        assert ev["bar"] == pytest.approx(40 * n_versions, rel=1e-15)
        if ev["e_value"] is None or abs(math.log(max(ev["e_value"], 1e-300)) - math.log(ev["bar"])) < 1e-9:
            continue
        assert ev["passes"] == (ev["e_value"] >= ev["bar"] and float(np.mean(x)) > 0)
        seen[ev["passes"]] += 1
    assert seen[True] >= 3 and seen[False] >= 3


# ── 6. Brier is proper (brier_bernoulli, brier_proper, brier_coin_flip, ...) ──

def test_expected_brier_closed_form_and_its_unique_minimum():
    grid = np.linspace(0.0, 1.0, 1001)
    for pi in np.linspace(0.0, 1.0, 41):
        expected = pi * (grid - 1) ** 2 + (1 - pi) * grid ** 2
        assert np.allclose(expected, (grid - pi) ** 2 + pi * (1 - pi), atol=1e-14)
        assert abs(grid[int(np.argmin(expected))] - pi) <= 0.0005 + 1e-12


def test_the_ledger_grades_each_call_by_its_squared_error(tmp_path):
    book = Ledger(tmp_path)
    rng = np.random.default_rng(3)
    x = rng.normal(0.3, 2.0, 45)
    # dated trades: a forecast made just after trade 45, graded by trades 46-75, which close after it
    times = list(pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.arange(75) * 6, unit="h"))
    made, later = (times[44] + pd.Timedelta(hours=1)).to_pydatetime(), (times[74] + pd.Timedelta(hours=1)).to_pydatetime()
    book.observe(Trades("s", "%", x, times[:45]), now=made)
    for tail_mean in (-1.0, 1.0):
        longer = np.concatenate((x, np.full(30, tail_mean)))
        call = book.calls[book.open[next(iter(book.open))]]
        book.observe(Trades("s", "%", longer, times), now=later)
        grade = book.grades[-1]
        outcome = int(float(np.sum(longer[45:75])) > 0)
        assert grade["outcome"] == outcome
        assert grade["brier"] == [round((p - outcome) ** 2, 8) for p in call["probabilities"]]
        coin = SHRINKAGE.index(math.inf)
        assert call["probabilities"][coin] == 0.5 and grade["brier"][coin] == 0.25   # brier_coin_flip
        book = Ledger(tmp_path / f"again{tail_mean}")
        book.observe(Trades("s", "%", x, times[:45]), now=made)


def test_the_coin_flip_forecast_is_one_half():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.normal(float(rng.uniform(-1, 1)), 2.0, int(rng.integers(30, 300)))
        assert forward_probabilities(x, [math.inf]) == [0.5]


def test_choosing_the_lowest_mean_brier_finds_the_calibrated_forecast(tmp_path):
    # brier_proper in action: Ledger.chosen_k keeps the candidate with the lowest mean Brier; with
    # outcomes of chance 0.62 and constant candidate forecasts, that is the candidate saying 0.62.
    rng = np.random.default_rng(6)
    forecasts = [0.2, 0.35, 0.45, 0.5, 0.55, 0.62, 0.7, 0.8, 0.5]
    assert len(forecasts) == len(SHRINKAGE)
    book = Ledger(tmp_path)
    outcomes = rng.random(4000) < 0.62
    book.grades = [{"brier": [(p - int(y)) ** 2 for p in forecasts]} for y in outcomes]
    assert book.chosen_k() == SHRINKAGE[forecasts.index(0.62)]


def test_a_base_rate_forecast_beats_the_coin_flip_without_skill():
    # brier_base_rate_beats_coin: track_record calls Brier < 0.25 "better than guessing"; a
    # constant forecast at a base rate pi != 1/2 already scores pi (1 - pi) < 0.25.
    for pi in (0.3, 0.55, 0.6, 0.8):
        assert pi * (1 - pi) < 0.25
        assert pi * (pi - 1) ** 2 + (1 - pi) * pi ** 2 == pytest.approx(pi * (1 - pi), abs=1e-15)


# ── 7. Money vs percent (losers_bigger, winners_bigger, counterexamples) ──────

def weighted_sizes(w, r):
    win, lose = r > 0, r < 0
    return float(np.sum(w[win] * r[win]) / np.sum(r[win])), float(np.sum(w[lose] * -r[lose]) / np.sum(-r[lose]))


def test_losers_are_bigger_in_the_weighted_sense_whenever_percent_and_money_disagree():
    rng = np.random.default_rng(10)
    hits = {"losers": 0, "winners": 0}
    for _ in range(20000):
        n = int(rng.integers(2, 40))
        r = rng.normal(float(rng.uniform(-1, 1)), 2.0, n)
        r[rng.random(n) < 0.1] = 0.0                                        # flat trades
        w = rng.lognormal(0.0, 1.5, n)
        money = w * r
        if r.sum() > 0 and money.sum() < 0:                                 # losers_bigger
            assert (r < 0).any() and (r > 0).any()
            win_size, lose_size = weighted_sizes(w, r)
            assert lose_size > win_size
            hits["losers"] += 1
        elif r.sum() < 0 and money.sum() > 0:                               # winners_bigger
            assert (r < 0).any() and (r > 0).any()
            win_size, lose_size = weighted_sizes(w, r)
            assert win_size > lose_size
            hits["winners"] += 1
    assert hits["losers"] > 100 and hits["winners"] > 100


def test_every_money_card_sentence_is_true_in_the_weighted_sense():
    rng = np.random.default_rng(14)
    said = {"losing": 0, "winning": 0}
    for _ in range(3000):
        n = int(rng.integers(30, 200))
        r = rng.normal(float(rng.uniform(0.0, 1.5)), 2.0, n) * rng.choice([1, -1])
        r[r == 0] = 0.01
        w = rng.lognormal(0.0, 0.5, n) * np.where(r < 0, float(rng.uniform(1, 6)), 1.0)
        if rng.random() < 0.5:
            w = rng.lognormal(0.0, 0.5, n) * np.where(r > 0, float(rng.uniform(1, 6)), 1.0)
        money = w * r / 100.0                                              # money = size x percent
        for card in (_money_card(None, money, "USDT", r, None), _money_card(None, r, "%", None, money)):
            sentence = card["sentence"]
            win_size, lose_size = weighted_sizes(w, r)
            if "losing trades were" in sentence:
                assert r.sum() > 0 > money.sum() and lose_size > win_size
                said["losing"] += 1
            if "winning trades were" in sentence:
                assert r.sum() < 0 < money.sum() and win_size > lose_size
                said["winning"] += 1
    assert said["losing"] > 50 and said["winning"] > 50


def test_the_two_other_readings_of_bigger_can_fail():
    # plain_average_size_can_disagree and money_per_trade_can_disagree, the Lean witnesses.
    w, r = np.array([1.0, 1000.0, 10.0]), np.array([10.0, 0.01, -5.0])
    assert r.sum() > 0 > (w * r).sum() and w[2] < (w[0] + w[1]) / 2
    win_size, lose_size = weighted_sizes(w, r)
    assert lose_size > win_size
    w, r = np.array([1.0, 30.0, 30.0]), np.array([50.0, -1.0, -1.0])
    assert r.sum() > 0 > (w * r).sum() and (w[1] * 1 + w[2] * 1) / 2 < w[0] * r[0]
    win_size, lose_size = weighted_sizes(w, r)
    assert lose_size > win_size


def test_a_blank_percent_cell_does_not_invent_a_disagreement():
    """Found a bug (fixed 2026-10-14): pmean = np.mean(pct) over a percent column that readers keep when
    >= 90% of it is finite. One blank cell made pmean NaN, so a record whose percent and money both say
    'up' was told 'Your average trade is n/a, but you made X in total: your winning trades were the
    bigger ones.' Only trades with both numbers count now."""
    rng = np.random.default_rng(1)
    pct = rng.normal(1.0, 0.5, 40)
    rows = ["Date,Profit USDT,Profit %"]
    for i, p in enumerate(pct):
        rows.append(f"2025-01-{1 + i % 28:02d} {i % 24:02d}:00,{p:.4f},{'' if i == 5 else f'{p:.4f}'}")
    result = judge_dict("\n".join(rows) + "\n")
    money = result["strategies"][0]["money"]
    assert money["total"] > 0 and not money["disagree"], money["sentence"]


def test_a_blank_percent_cell_does_not_hide_a_real_disagreement():
    """Found a bug (fixed 2026-10-14): with a blank percent cell the NaN mean also hid a real
    disagreement (percent mean clearly positive, money clearly negative)."""
    rng = np.random.default_rng(2)
    pct = rng.normal(1.0, 0.5, 60)
    size = np.ones(60)
    pct[::5] = -2.0                                    # every fifth trade loses, on a big position
    size[::5] = 10.0
    money = size * pct
    assert pct.mean() > 0 > money.sum()
    pct_blank = pct.copy()
    pct_blank[7] = np.nan
    assert _money_card(None, money, "USDT", pct, None)["disagree"]
    assert _money_card(None, money, "USDT", pct_blank, None)["disagree"]


def test_the_bigger_losers_sentence_needs_money_and_percent_to_agree_trade_by_trade():
    """Found a false claim (fixed 2026-10-14): 'your losing trades were bigger than your winning ones'
    needs money = size x percent with size > 0 trade by trade (losers_bigger). When the percent is before
    fees and the money after them, every trade can be up in percent and down in money; the card blamed
    trade sizes, but no trade lost in percent: fees did it. Now it says so."""
    rng = np.random.default_rng(0)
    pct = rng.normal(0.05, 0.01, 50)                   # every trade +0.03% .. +0.08% before fees
    money = rng.normal(-0.5, 0.1, 50)                  # every trade loses money after fees
    assert (pct > 0).all() and (money < 0).all()
    card = _money_card(None, money, "USDT", pct, None)
    assert card["disagree"]
    assert "losing trades were bigger" not in card["sentence"], card["sentence"]
    assert "fees" in card["sentence"] and "50 trades" in card["sentence"], card["sentence"]


def test_the_bigger_losers_sentence_stays_when_it_is_a_theorem():
    """When every trade's money and percent agree in sign (money = size x percent, size > 0), Σr > 0 and
    Σm < 0 imply the losers' |r|-weighted size beats the winners' (losers_bigger), and the card says it."""
    rng = np.random.default_rng(3)
    pct = np.where(np.arange(60) % 5 == 0, -2.0, rng.uniform(0.5, 1.5, 60))
    size = np.where(np.arange(60) % 5 == 0, 10.0, 1.0)
    card = _money_card(None, size * pct, "USDT", pct, None)
    assert card["disagree"] and "losing trades were bigger" in card["sentence"]


# ── 8. Wilson (wilson_roots, wilson_ends_mem_unit, wilson_score_inversion) ────

def unclipped_wilson(wins, n, z):
    p = min(1.0, max(0.0, wins / n))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return p, centre - spread, centre + spread


def score_gap(phat, p, z, n):
    return (phat - p) ** 2 - z * z * p * (1 - p) / n


def wilson_cases():
    rng = random.Random(4)
    cases = [(0, 1, 1.96), (1, 1, 1.96), (0, 30, 1.96), (30, 30, 1.96), (15, 30, 0.0), (7, 9, 3.5)]
    for _ in range(3000):
        n = rng.choice([1, 2, 5, 30, 100, 1000, rng.randint(1, 100000)])
        cases.append((rng.randint(0, n), n, rng.choice([1.0, 1.645, 1.96, 2.576, rng.uniform(0.01, 5.0)])))
    return cases


def test_wilson_ends_are_the_roots_of_the_score_equation():
    for wins, n, z in wilson_cases():
        lo, hi = wilson_bounds_count(wins, n, z)
        phat = wins / n
        a, b, c = 1 + z * z / n, -(2 * phat + z * z / n), phat * phat
        disc = max(b * b - 4 * a * c, 0.0)
        r1, r2 = (-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)
        assert lo == pytest.approx(r1, abs=1e-12) and hi == pytest.approx(r2, abs=1e-12)
        assert abs(score_gap(phat, lo, z, n)) < 1e-12 and abs(score_gap(phat, hi, z, n)) < 1e-12


def test_wilson_clipping_never_changes_the_ends():
    # wilson_ends_mem_unit: the unclipped ends already lie in [0, 1] (up to float rounding).
    for wins, n, z in wilson_cases():
        _, lo_raw, hi_raw = unclipped_wilson(wins, n, z)
        lo, hi = wilson_bounds_count(wins, n, z)
        assert -1e-15 <= lo_raw and hi_raw <= 1 + 1e-15
        assert lo == pytest.approx(lo_raw, abs=1e-15) and hi == pytest.approx(hi_raw, abs=1e-15)


def test_wilson_interval_is_the_score_test_inversion():
    grid = np.linspace(0.0, 1.0, 2001)
    for wins, n, z in wilson_cases()[:400]:
        phat = wins / n
        lo, hi = wilson_bounds_count(wins, n, z)
        for p in grid:
            if min(abs(p - lo), abs(p - hi)) < 1e-9:
                continue
            assert (score_gap(phat, p, z, n) <= 0) == (lo <= p <= hi)


# ── 9. Compare's range (diff_range, diff_range_union_bound, compare_levels) ───

def test_the_backtest_z_is_the_98_75_percent_normal_quantile():
    assert NormalDist().inv_cdf(1 - 0.0125) == pytest.approx(compare.Z_BACKTEST, abs=5e-6)
    assert normal_tail(compare.Z_BACKTEST) == pytest.approx(0.0125, abs=1e-6)


def test_the_levels_add_up_to_a_95_percent_range():
    # compare_levels: the live side misses 2/80 = 2.5% (a 97.5% range), the backtest side
    # 2 x 1.25% = 2.5%; together 5%.
    assert 1 - 2 / compare.BAR == pytest.approx(0.975, abs=1e-15)
    live_miss = 2 / compare.BAR
    backtest_miss = 2 * normal_tail(compare.Z_BACKTEST)
    assert live_miss + backtest_miss == pytest.approx(0.05, abs=1e-6)
    assert 1 - (live_miss + backtest_miss) == pytest.approx(0.95, abs=1e-6)


def read_of(x):
    return ReadResult([Trades("s", "%", np.asarray(x, dtype=float))], "list")


def test_compare_runs_range_is_the_sum_of_both_half_widths():
    # diff_range: range_95 = [d - h_live - h_bt, d + h_live + h_bt], symmetric (two-sided on both
    # sides), with h_live at bar 80 and h_bt = Z_BACKTEST sd / sqrt(n); "worse" iff its top < 0.
    rng = np.random.default_rng(15)
    codes = set()
    for _ in range(60):
        n, nb = int(rng.integers(compare.MIN_LIVE, 300)), int(rng.integers(30, 500))
        x = rng.normal(float(rng.uniform(-1.0, 0.5)), 1.0, n)
        y = rng.normal(0.3, 1.0, nb)
        out = compare_runs(read_of(x), read_of(y))
        h_live = half_width(n, compare._live_sd(x), compare.BAR)
        h_bt = compare.Z_BACKTEST * compare._live_sd(y) / math.sqrt(nb)
        diff = float(np.mean(x)) - float(np.mean(y))
        lo, hi = diff - h_live - h_bt, diff + h_live + h_bt
        assert out["difference"]["range_95"] == [rnd(lo), rnd(hi)]
        assert (hi - diff) == pytest.approx(diff - lo, rel=1e-12)
        assert (out["code"] == "worse_than_backtest") == (hi < 0)
        codes.add(out["code"])
    assert codes == {"worse_than_backtest", "consistent"}
