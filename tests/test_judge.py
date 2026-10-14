"""The judge's known answers, its honesty rules and its result contract."""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

import edgeorluck
from edgeorluck import evidence, samples
from edgeorluck.judge import MIN_LOSSES, judge_dict

from helpers import tradingview_old

jsonschema = pytest.importorskip("jsonschema")


def run(sample_id, **kw):
    args = samples.sample_input(sample_id)
    args.update(kw)
    return edgeorluck.judge(**args)


def codes(result):
    return {c["name"]: c["verdict"]["code"] for c in result["strategies"]}


# ── the samples' known answers ────────────────────────────────────────────────

def test_the_samples_get_their_known_answers():
    assert codes(run("lucky")) == {"Lucky streak": "cant_tell"}
    assert codes(run("real")) == {"Steady edge": "real_edge"}
    assert codes(run("faded")) == {"Faded breakout": "faded"}
    trader = run("trader")
    assert codes(trader) == {"Breakout BTC": "real_edge", "Trend ETH": "cant_tell", "Mean-revert SOL": "cant_tell",
                             "News scalper": "faded"}
    [pair] = trader["portfolio"]["pairs"]
    assert {pair["a"], pair["b"]} == {"Breakout BTC", "Trend ETH"} and pair["reading"] == "same_bet"
    assert trader["summary"]["headline"] == "1 of 4 strategies shows a real edge: Breakout BTC."
    assert trader["numbers"] is None and trader["verdict"]["about"] == "Breakout BTC"
    swbl = run("small_wins_big_losses")
    card = swbl["strategies"][0]
    assert card["verdict"]["code"] == "losing" and card["unit"] == "USDT"
    assert card["money"]["disagree"] and card["money"]["percent_mean"] > 1.5 and card["money"]["total"] < 0
    assert "small_wins_big_losses" in card["red_flags"] and swbl["ai_audit_prompt"]
    assert swbl["verdict"]["plain"].startswith("Your average trade is +1.")
    fills = run("fills")
    assert fills["input"]["format"] == "fills" and fills["strategies"][0]["n"] == 70
    assert len(fills["warnings"]) == 2
    compare = run("compare")
    assert compare["compare"]["code"] == "worse_than_backtest"
    assert compare["next_step"] == compare["compare"]["next_step"]


def test_more_versions_tried_raise_the_bar_and_say_so():
    alone = run("real")
    tried = run("real", variants_tried=10_000)
    assert alone["verdict"]["code"] == "real_edge" and tried["verdict"]["code"] == "cant_tell"
    survives = alone["numbers"]["versions_it_survives"]
    assert survives >= 2
    assert run("real", variants_tried=survives)["verdict"]["code"] == "real_edge"
    assert run("real", variants_tried=survives + 1)["verdict"]["code"] == "cant_tell"
    assert "if this is the only version you tried" in alone["verdict"]["headline"]
    assert any(q["id"] == "variants_tried" for q in alone["ask_the_user"])
    assert tried["numbers"]["adjusted"]["versions_tried"] == 10_000
    assert "when you tried 10000 versions" in tried["verdict"]["headline"]
    assert not any(q["id"] == "variants_tried" for q in tried["ask_the_user"])


def test_lucky_reads_consistently_with_its_coin_flips():
    r = run("lucky")
    coin = r["numbers"]["coin_flip"]
    assert coin["ended_higher"] > coin["needed_at_most"]                  # can't tell, and the bar says why
    sentence = r["strategies"][0]["luck"]["sentence"]
    assert "stricter test" in sentence and "check again at any time" in sentence
    assert "rarer than about 1 in" in sentence and "needs" not in sentence   # the bar as odds, not a copy count
    real = run("real")
    assert real["numbers"]["coin_flip"]["ended_higher"] <= real["numbers"]["coin_flip"]["needed_at_most"]
    assert real["strategies"][0]["luck"]["sentence"].endswith("It passes.")


# ── honesty on noise ──────────────────────────────────────────────────────────

def test_on_pure_noise_the_judge_rarely_calls_anything():
    labelled = 0
    for seed in range(300):
        x = np.random.default_rng(seed).normal(0.0, 2.0, 300)
        labelled += judge_dict(list(x), unit="%", details=False)["verdict"]["code"] in ("real_edge", "losing")
    assert labelled <= 6                                                   # target 5%; measured far below


def test_best_of_n_needs_variants_tried():
    """Pick the best of N pure-noise strategies and judge it alone: without
    variants_tried the false 'real edge' rate is many times higher than with
    it, and with it the rate stays under q."""
    reps, n_versions, n_trades = 150, 300, 200
    without = with_ = 0
    for r in range(reps):
        noise = np.random.default_rng(50_000 + r).normal(0, 1, (n_versions, n_trades))
        best = list(noise[np.argmax(noise.sum(axis=1))])
        without += judge_dict(best, details=False)["verdict"]["code"] == "real_edge"
        with_ += judge_dict(best, variants_tried=n_versions, details=False)["verdict"]["code"] == "real_edge"
    assert with_ / reps <= evidence.Q
    assert without / reps >= 0.10 and without >= 5 * max(with_, 1)


def test_ebh_counts_unshown_versions_as_zero():
    log_e = {"a#up": math.log(1000.0), "a#down": 0.0}
    assert evidence.ebh(log_e, 0.05) == {"a#up"}
    assert evidence.ebh(log_e, 0.05, total=2 * 25) == {"a#up"}            # bar 40 x 25 = 1000
    assert evidence.ebh(log_e, 0.05, total=2 * 26) == set()


def test_the_range_and_the_verdict_agree():
    """One strategy, one version: real_edge exactly when the whole 95% range
    is above zero."""
    for seed in range(60):
        x = np.random.default_rng(seed).normal(0.25, 2.0, 200)
        r = judge_dict(list(x), unit="%", details=False)
        lo, _ = r["numbers"]["mean_per_trade"]["range_95"]
        if r["verdict"]["code"] in ("real_edge", "cant_tell"):
            assert (r["verdict"]["code"] == "real_edge") == (lo > 0), seed


def test_skewed_records_need_more_and_martingales_need_losses():
    one_loss = [1.0, 1.1, 0.9] * 13 + [-5.0]                               # 39 wins, 1 loss: looks perfect
    r = judge_dict(one_loss)
    assert r["verdict"]["code"] == "cant_tell" and "only 1 losing trade" in r["verdict"]["headline"]
    assert any(c["code"] == "few_losses" for c in r["checks"])
    assert "losing trades" in r["next_step"]["text"]                       # many small wins, one big loss
    false = 0
    for seed in range(300):
        g = np.random.default_rng(seed)
        x = np.where(g.random(100) < 0.97, 1.0, -32.333) + g.normal(0, 0.05, 100)
        false += judge_dict(list(x), details=False)["verdict"]["code"] == "real_edge"
    assert false <= 3
    same = np.random.default_rng(9).normal(0.4, 1.0, 120)
    rec_sym, rec_neg = evidence.record_of(same), evidence.record_of(np.sort(same))
    assert evidence.skew_shift(-2.0, 120, 40.0, evidence.mixing_rho2(40.0), +1) > 0
    assert evidence.skew_shift(+2.0, 120, 40.0, evidence.mixing_rho2(40.0), +1) == 0
    assert rec_sym.skew == pytest.approx(rec_neg.skew)


def test_serial_correlation_widens_the_spread():
    e = np.random.default_rng(1).normal(0, 1, 600)
    ar = np.zeros(600)
    for i in range(1, 600):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    assert evidence.serial_factor(ar) > 2.0
    assert evidence.serial_factor(e) < 1.3
    assert evidence.serial_factor(-np.cumsum(np.ones(10))) == 1.0


def test_faded_is_conservative_on_a_steady_edge():
    faded = 0
    for seed in range(150):
        x = np.random.default_rng(seed).normal(0.4, 2.0, 400)
        faded += judge_dict(list(x), unit="%", details=False)["verdict"]["code"] == "faded"
    assert faded <= 3


# ── broken, or a bad streak? ──────────────────────────────────────────────────

def test_recent_trades_get_plain_odds():
    rng = np.random.default_rng(4)
    calm = np.concatenate((rng.normal(0.5, 2.0, 300), rng.normal(0.5, 2.0, 30)))
    r = edgeorluck.judge(list(calm), unit="USDT")
    recent = r["strategies"][0]["recent"]
    assert recent["reading"] in ("in_line", "normal", "bad_streak") and recent["why"]
    broken = np.concatenate((rng.normal(0.5, 2.0, 300), rng.normal(-2.5, 2.0, 30)))
    r = edgeorluck.judge(list(broken), unit="USDT")
    recent = r["strategies"][0]["recent"]
    assert recent["reading"] == "likely_broken"
    assert re.search(r"a drop that deep within 30 trades happens only (fewer than|about) 1 in [\d,]+ times",
                     recent["sentence"])
    assert "USDT" in recent["sentence"]


# ── percent, money, fees, the market ─────────────────────────────────────────

def test_warnings_for_a_few_big_trades_and_percent_that_money_contradicts():
    x = np.concatenate((np.full(97, -0.2), [30.0, 25.0, 20.0]))
    r = edgeorluck.judge(list(x), unit="%")
    assert any(c["code"] == "few_big_trades" for c in r["checks"])
    rows = pd.DataFrame({"pnl %": x, "pnl usdt": np.where(x > 0, 0.5, -3.0)})
    r = edgeorluck.judge(rows)
    card = r["strategies"][0]
    assert card["unit"] == "USDT" and card["money"]["disagree"]
    assert any(c["code"] == "percent_vs_money" for c in r["checks"])


def test_fees_answered_and_unanswered():
    x = list(np.random.default_rng(2).normal(0.3, 2.0, 200))
    asked = edgeorluck.judge(x, unit="%")
    assert any(q["id"] == "fees_included" for q in asked["ask_the_user"])
    before = edgeorluck.judge(x, unit="%", fees_included="no")
    fee_check = next(c for c in before["checks"] if c["code"] == "fees")
    assert fee_check["status"] == "fail" and fee_check["fix"]
    assert not any(q["id"] == "fees_included" for q in before["ask_the_user"])


def test_long_only_in_a_rising_market_is_caught():
    """Every trade only rides the market up: the raw record passes, the
    market-adjusted one does not."""
    rng = np.random.default_rng(8)
    returns = rng.normal(1.2, 1.0, 120)
    text = tradingview_old(returns, rising=0.0005, seed=3)
    r = edgeorluck.judge(text, variants_tried=1)
    card = r["strategies"][0]
    assert card["benchmark"] is not None and card["benchmark"]["hold_pct"] > 0
    assert card["benchmark"]["market_adjusted"] is not None
    assert card["benchmark"]["downgraded"] and card["verdict"]["code"] in ("cant_tell", "no_edge")
    assert any(c["code"] == "beats_the_market" and c["status"] == "fail" for c in r["checks"])
    raw = edgeorluck.judge(list(returns), unit="%", variants_tried=1)
    assert raw["verdict"]["code"] == "real_edge"                             # without the market check it passes


def test_a_given_benchmark_is_compared_and_answers_the_question():
    x = list(np.random.default_rng(3).normal(0.3, 2.0, 150))
    asked = edgeorluck.judge(x, unit="%")
    assert any(q["id"] == "benchmark" for q in asked["ask_the_user"])
    given = edgeorluck.judge(x, unit="%", benchmark_return_pct="200%")
    assert given["strategies"][0]["benchmark"]["hold_pct"] == 200.0
    assert not any(q["id"] == "benchmark" for q in given["ask_the_user"])
    assert any(c["code"] == "beats_the_market" for c in given["checks"])


# ── red flags ─────────────────────────────────────────────────────────────────

def test_too_good_to_be_true_raises_flags_and_a_tailored_audit_prompt():
    rng = np.random.default_rng(6)
    x = np.where(rng.random(200) < 0.93, rng.normal(1.0, 0.2, 200), rng.normal(-0.6, 0.1, 200))
    r = edgeorluck.judge(list(x), unit="%", source="backtest")
    flags = r["strategies"][0]["red_flags"]
    assert "high_win_rate" in flags and "huge_profit_factor" in flags
    prompt = r["ai_audit_prompt"]
    assert prompt and "Lookahead" in prompt and "fees" in prompt.lower()
    assert "audit prompt" in r["next_step"]["text"]
    clean = run("real")
    assert clean["strategies"][0]["red_flags"] == []


# ── determinism and the contract ──────────────────────────────────────────────

def test_same_input_same_output():
    a = run("trader", include_chart=True).to_json()
    b = run("trader", include_chart=True).to_json()
    assert a == b
    x = list(np.random.default_rng(5).normal(0.2, 1.5, 250))
    r1, r2 = edgeorluck.judge(x).to_dict(), edgeorluck.judge(list(x)).to_dict()
    assert r1 == r2 and r1["input_fingerprint"].startswith("sha256:")


def _all_results():
    out = [run(s["id"]) for s in samples.SAMPLES]
    out += [run("lucky", variants_tried=200), run("real", fees_included=False, source="backtest"),
            edgeorluck.judge([1, -2, 3]), edgeorluck.judge("a,b\n1,2"), edgeorluck.judge([1.0] * 40),
            run("small_wins_big_losses", benchmark_return_pct=10)]
    return out


def test_every_result_follows_the_schema_and_has_a_why_everywhere():
    schema = edgeorluck.schema()
    validator = jsonschema.Draft202012Validator(schema)
    for r in _all_results():
        d = r.to_dict()
        errors = sorted(validator.iter_errors(d), key=lambda e: e.path)
        assert not errors, errors[0].message
        json.dumps(d, allow_nan=False)
        items = list(d["checks"]) + list(d["ask_the_user"])
        for card in d["strategies"]:
            items += card["checks"] + card["ask_the_user"] + [card["verdict"], card["next_step"]]
        if d["verdict"]:
            items += [d["verdict"], d["next_step"]]
        for item in items:
            assert item["why"] and len(item["why"].split()) >= 4, item


def test_next_steps_are_about_testing_never_trading_advice():
    banned = re.compile(r"\b(buy more|sell (it|now|everything)|go live|size up|increase (the )?(size|position)|"
                        r"double|put more money|bet more|all in)\b", re.I)
    for r in _all_results():
        d = r.to_dict()
        steps = [d["next_step"]] + [c["next_step"] for c in d["strategies"]]
        for step in steps:
            if step:
                assert not banned.search(step["text"]), step["text"]


def test_plain_words_on_the_main_view():
    jargon = re.compile(r"e-value|e-BH|confidence sequence|Newey|Edgeworth|p-value|martingale bar|bootstrap", re.I)
    for r in _all_results():
        d = r.to_dict()
        if not d["verdict"]:
            continue
        texts = [d["verdict"]["headline"], d["verdict"]["plain"], d["next_step"]["text"]]
        texts += [c["text"] for c in d["checks"]] + [q["question"] for q in d["ask_the_user"]]
        for t in texts:
            assert not jargon.search(t), t
        assert any("e-BH" in h for h in d["how_we_know"])


def test_markdown_carries_the_why_and_the_fingerprint():
    r = run("lucky", variants_tried=40)
    md = r.to_markdown()
    assert md.startswith("## Edge or luck: Can't tell yet")
    assert r["verdict"]["headline"] in md and "_Why:" in md and r["input_fingerprint"] in md
    assert "### Questions for the trader" not in md or "answer with" in md
    assert "Not financial advice" in md


def test_too_few_trades_and_the_reply_draft():
    r = edgeorluck.judge([1.0, -0.5, 2.0])
    assert r["verdict"]["code"] == "too_early" and "at least 30" in r["verdict"]["headline"]
    full = run("lucky")
    assert full["reply_draft"].startswith(full["verdict"]["headline"])
    assert "Next step:" in full["reply_draft"]


def test_settings_are_forgiving():
    r = edgeorluck.judge(list(np.random.default_rng(1).normal(0, 1, 50)), variants_tried="about 40",
                         benchmark_return_pct="+34.5%", fees_included="yes", source="Real money")
    s = r["settings"]
    assert s["variants_tried"] == 40 and s["benchmark_return_pct"] == 34.5 and s["fees_included"] is True
    assert s["source"] == "live"
    odd = edgeorluck.judge([1, 2, 3], variants_tried="lots", source="moon")
    assert any("variants_tried" in w for w in odd["warnings"]) and any("source" in w for w in odd["warnings"])


# ── backtest vs live ──────────────────────────────────────────────────────────

def test_backtest_vs_live_codes():
    rng = np.random.default_rng(12)
    backtest = list(rng.normal(0.4, 1.5, 400))
    same = edgeorluck.judge(list(rng.normal(0.4, 1.5, 80)), backtest=backtest)
    assert same["compare"]["code"] == "consistent" and same["compare"]["why"]
    worse = edgeorluck.judge(list(rng.normal(-0.6, 1.5, 150)), backtest=backtest)
    assert worse["compare"]["code"] == "worse_than_backtest"
    assert worse["compare"]["difference"]["range_95"][1] < 0
    assert worse["compare"]["e_value_worse"] >= worse["compare"]["bar"]
    early = edgeorluck.judge([0.1, -0.2, 0.3], backtest=backtest)
    assert early["compare"]["code"] == "too_early"
    units = edgeorluck.judge(list(rng.normal(0.4, 1.5, 50)), unit="USDT", backtest=pd.DataFrame(
        {"pnl %": rng.normal(0.4, 1.5, 100)}))
    assert units["compare"]["code"] == "units_differ"


def test_backtest_vs_live_stays_honest_when_checked_often():
    """Live trades drawn from the backtest's own distribution, checked every
    10 trades up to 200: 'worse' should almost never be called."""
    false = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        backtest = rng.normal(0.3, 1.5, 300)
        live = rng.normal(0.3, 1.5, 200)
        for n in range(10, 201, 10):
            if edgeorluck.judge(list(live[:n]), backtest=list(backtest))["compare"]["code"] == "worse_than_backtest":
                false += 1
                break
    assert false <= 3
