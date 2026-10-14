"""The honest backtester: fills, costs, look-ahead, the rule language, tuning,
prices, and its MCP tools and command line."""
from __future__ import annotations

import io
import json
import math
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

import edgeorluck as eol
from edgeorluck import cli, market
from edgeorluck.rulebook import RuleError, compile_rule, rule_from


def frame(closes, opens=None, start="2024-01-01"):
    closes = np.asarray(closes, float)
    opens = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.001,
                         "low": np.minimum(opens, closes) * 0.999, "close": closes, "volume": 1.0},
                        index=pd.date_range(start, periods=len(closes), freq="h", tz="UTC"))


def walk(n=2000, seed=0, drift=0.0, vol=0.01):
    rng = np.random.default_rng(seed)
    return frame(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def by_position(positions):
    """A Python rule holding after the closes of the given bars (it can't peek: it ignores prices)."""
    def rule(f):
        return pd.Series([1.0 if i in positions else 0.0 for i in range(len(f))], index=f.index)
    return rule


# ── fills and costs ───────────────────────────────────────────────────────────

def test_a_decision_is_filled_at_the_next_open_with_fees_and_slippage_on_both_fills():
    n = 40
    closes = np.linspace(100, 140, n)
    opens = closes - 0.5
    px = frame(closes, opens)
    # a rule on bar numbers ignores the prices, so the look-ahead test would (rightly) refuse it: fills only here
    bt = eol.backtest(px, by_position({2, 3, 4, 5}), fee_pct=0.1, slippage_pct=0.05, check_peeking=False)
    (t,) = bt["trades"]
    # in after bar 2's close (filled at bar 3's open); out after bar 6's close says flat (bar 7's open)
    assert (t["entry_bar"], t["exit_bar"]) == (3, 7)
    f, s = 0.001, 0.0005
    want = opens[7] * (1 - s) * (1 - f) / (opens[3] * (1 + s) * (1 + f)) - 1
    assert t["return_pct"] == pytest.approx(100 * want, rel=1e-12)
    assert t["entry_price"] == opens[3] and t["exit_price"] == opens[7]
    assert bt["costs"]["round_trip_pct"] == pytest.approx(0.3)


def test_a_short_pays_costs_on_both_fills_too():
    n = 40
    closes = np.linspace(140, 100, n)
    opens = closes + 0.5
    rule = lambda f: pd.Series([-1.0 if 2 <= i <= 5 else 0.0 for i in range(len(f))], index=f.index)
    bt = eol.backtest(frame(closes, opens), rule, allow_short=True, fee_pct=0.1, slippage_pct=0.05,
                      check_peeking=False)
    (t,) = bt["trades"]
    sell, buy = opens[3] * (1 - 0.0005), opens[7] * (1 + 0.0005)
    assert t["side"] == "short"
    assert t["return_pct"] == pytest.approx(100 * (sell * 0.999 - buy * 1.001) / sell, rel=1e-12)


def test_without_allow_short_a_minus_one_means_flat():
    n = 40
    rule = lambda f: pd.Series([-1.0] * len(f), index=f.index)
    assert eol.backtest(frame(np.linspace(100, 120, n)), rule).status == "no_trades"


def test_the_rule_language_trades_on_the_bar_after_the_signal():
    closes = np.full(80, 100.0)
    closes[40:50] = 110.0
    opens = np.r_[closes[0], closes[:-1]] + 0.25
    bt = eol.backtest(frame(closes, opens), {"buy": "close > 105", "sell": "close < 105"})
    (t,) = bt["trades"]
    assert (t["entry_bar"], t["exit_bar"]) == (41, 51)
    assert t["entry_price"] == opens[41] and t["exit_price"] == opens[51]


def test_stops_targets_and_max_hold_are_checked_at_closes_against_the_fill():
    def shape(after):
        closes = np.full(60, 100.0)
        closes[35] = 106.0
        closes[36:36 + len(after)] = after
        return frame(closes)
    stop = eol.backtest(shape([106, 104, 99, 99]), {"buy": "close > 105", "stop_loss_pct": 5})
    assert [(t["entry_bar"], t["exit_bar"]) for t in stop["trades"]] == [(36, 39)]
    take = eol.backtest(shape([106, 109, 112, 100]), {"buy": "close > 105 and close < 107", "take_profit_pct": 5})
    assert [(t["entry_bar"], t["exit_bar"]) for t in take["trades"]] == [(36, 39)]
    hold = eol.backtest(shape([106, 104, 104, 104]), {"buy": "close > 105", "max_hold_bars": 3})
    assert [(t["entry_bar"], t["bars"]) for t in hold["trades"]] == [(36, 3)]


def test_a_trade_still_open_at_the_end_is_closed_at_the_last_close_and_flagged():
    closes = np.linspace(100, 150, 60)
    bt = eol.backtest(frame(closes), {"buy": "close > 140", "sell": "close < 0"})
    assert bt["trades"][-1]["end_of_data"] and bt["trades"][-1]["exit_price"] == closes[-1]
    assert any(c["code"] == "open_at_end" and c["status"] == "warn" for c in bt["checks"])


def test_backtest_hands_the_trades_to_the_judge_with_fees_and_the_benchmark():
    bt = eol.backtest(walk(3000, seed=4), {"buy": "rsi(3) < 15", "sell": "rsi(3) > 60"}, variants_tried=7)
    assert bt.status == "ok" and bt["stats"]["trades"] >= 30
    settings = bt["judgement"]["settings"]
    assert settings["fees_included"] is True and settings["source"] == "backtest" and settings["variants_tried"] == 7
    assert settings["benchmark_return_pct"] == pytest.approx(bt["stats"]["hold_pct"], abs=1e-5)
    assert bt.judgement.headline == bt["headline"]
    assert bt["judgement"]["numbers"]["trades"] == len(bt["trades"])
    assert json.loads(bt.to_json())["status"] == "ok"
    assert "## Honest backtest:" in bt.to_markdown() and "## Edge or luck" in bt.to_markdown()


# ── look-ahead ────────────────────────────────────────────────────────────────

PEEKERS = {
    "next close": lambda f: (f["close"].shift(-1) > f["close"]).astype(float),
    "centred window": lambda f: (f["close"] < f["close"].rolling(9, center=True).mean()).astype(float),
    "whole-series z-score": lambda f: ((f["close"] - f["close"].mean()) / f["close"].std() < -0.5).astype(float),
}


@pytest.mark.parametrize("name", list(PEEKERS))
def test_a_rule_that_peeks_is_stopped_not_judged(name):
    bt = eol.backtest(walk(1500, seed=2), PEEKERS[name])
    assert bt.status == "rule_peeks", name
    assert bt["judgement"] is None and bt["trades"] == []
    check = next(c for c in bt["checks"] if c["code"] == "no_look_ahead")
    assert check["status"] == "fail" and "shift(-1)" in check["fix"]
    assert "looks into the future" in bt.to_markdown()


def test_an_honest_python_rule_passes_the_look_ahead_test():
    rule = lambda f: (f["close"] < f["close"].rolling(20).mean()).astype(float)
    bt = eol.backtest(walk(1500, seed=2), rule)
    assert bt.status == "ok"
    assert next(c for c in bt["checks"] if c["code"] == "no_look_ahead")["status"] == "pass"


def test_a_python_rule_must_return_one_position_per_bar():
    with pytest.raises(RuleError):
        eol.backtest(walk(200), lambda f: f)
    with pytest.raises(RuleError):
        eol.backtest(walk(200), lambda f: [1, 0])


def test_one_bar_later_catches_an_edge_that_lives_on_instant_fills():
    zigzag = 100 * (1 + 0.02 * (-1.0) ** np.arange(400)) + np.random.default_rng(1).normal(0, 0.05, 400)
    bt = eol.backtest(frame(zigzag), {"buy": "change(1) < 0", "sell": "change(1) > 0"})
    assert bt["stats"]["mean_pct"] > 0
    assert next(c for c in bt["checks"] if c["code"] == "one_bar_later")["status"] == "fail"


def test_one_bar_later_passes_a_slow_edge():
    bt = eol.backtest(walk(1500, seed=5, drift=0.002, vol=0.004), {"buy": "close > 0", "max_hold_bars": 24})
    assert next(c for c in bt["checks"] if c["code"] == "one_bar_later")["status"] == "pass"


# ── the rule language ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "__import__('os').system('echo hi') > 0", "close.__class__ > 0", "open[0] > 1", "(lambda: 1)() > 0",
    "sma(close, n=20) > 1", "'text' > 1", "close if open else high", "close ** 2 > 1", "[1, 2] > 1",
    "os.system('x') > 0", "True", "eval('1') > 0", "getattr(close, 'x') > 0", "close > sma", "x > 1",
    "close; import os", "sma(20)",
])
def test_the_rule_language_refuses_everything_but_itself(text):
    with pytest.raises(RuleError):
        compile_rule(text)


def test_a_placeholder_outside_tuning_says_what_to_do():
    with pytest.raises(RuleError, match="tune"):
        eol.backtest(walk(200), {"buy": "rsi(14) < {low}", "sell": "rsi(14) > 70"})


def test_rule_from_needs_an_exit_and_numbers_for_settings():
    with pytest.raises(RuleError, match="get out"):
        rule_from({"buy": "close > 1"})
    with pytest.raises(RuleError, match="number"):
        rule_from({"buy": "close > 1", "stop_loss_pct": "five"})
    with pytest.raises(RuleError, match="grid"):
        rule_from({"buy": "close > 1", "stop_loss_pct": "{s}"}, {"low": 1})
    with pytest.raises(RuleError, match="unknown"):
        rule_from({"buy": "close > 1", "sell": "close < 1", "leverage": 10})
    r = rule_from({"buy": "rsi(14) < {low}", "sell": "close < 1", "max_hold_bars": "{h}"}, {"low": 27.5, "h": 12})
    assert r.buy == "rsi(14) < 27.5" and r.max_hold_bars == 12


def _tv_average(x, n, alpha):
    out, avg, seen = [], None, []
    for v in x:
        if avg is None:
            if not math.isnan(v):
                seen.append(v)
            if len(seen) == n:
                avg = sum(seen) / n
            out.append(avg if avg is not None else math.nan)
        else:
            avg = alpha * v + (1 - alpha) * avg
            out.append(avg)
    return np.array(out)


def test_rsi_ema_and_atr_match_tradingview_from_the_first_bar():
    px = walk(300, seed=9)
    c, h, l = px["close"].to_numpy(), px["high"].to_numpy(), px["low"].to_numpy()
    d = np.r_[math.nan, np.diff(c)]
    up = _tv_average(np.where(np.isnan(d), math.nan, np.maximum(d, 0)), 14, 1 / 14)
    down = _tv_average(np.where(np.isnan(d), math.nan, np.maximum(-d, 0)), 14, 1 / 14)
    rsi = np.where(down == 0, 100, np.where(up == 0, 0, 100 - 100 / (1 + up / down)))
    tr = np.r_[h[0] - l[0], np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])]
    from edgeorluck.rulebook import _rb_call
    got = {"rsi": _rb_call("rsi", [14.0], px), "ema": _rb_call("ema", [10.0], px), "atr": _rb_call("atr", [14.0], px),
           "std": _rb_call("std", [20.0], px)}
    stdev = np.array([np.std(c[i - 19:i + 1]) if i >= 19 else math.nan for i in range(len(c))])  # ta.stdev: population
    want = {"rsi": rsi, "ema": _tv_average(c, 10, 2 / 11), "atr": _tv_average(tr, 14, 1 / 14), "std": stdev}
    for name in got:
        g, w = got[name].to_numpy(), want[name]
        assert np.array_equal(np.isnan(g), np.isnan(w)), name
        assert np.allclose(g[~np.isnan(g)], w[~np.isnan(w)], rtol=1e-10, atol=1e-10), name
    assert int(np.flatnonzero(~np.isnan(got["rsi"].to_numpy()))[0]) == 14


def test_rsi_of_a_flat_or_one_way_market():
    from edgeorluck.rulebook import _rb_call
    flat = frame(np.full(40, 100.0))
    rising = frame(np.linspace(100, 140, 40))
    falling = frame(np.linspace(140, 100, 40))
    assert _rb_call("rsi", [14.0], flat).dropna().eq(100).all()
    assert _rb_call("rsi", [14.0], rising).dropna().eq(100).all()
    assert _rb_call("rsi", [14.0], falling).dropna().eq(0).all()


def test_every_function_in_the_language_runs_and_only_looks_back():
    px = walk(400, seed=3)
    texts = ["sma(20) > 0", "ema(close, 20) > 0", "rsi(14) > 0", "highest(high, 20) > 0", "lowest(20) > 0",
             "std(20) > 0", "atr(14) > 0", "change(5) > -1000", "pct(volume, 3) > -1000", "prev(close) > 0",
             "crosses_above(ema(5), ema(20))", "crosses_below(close, 100)", "abs(change(1)) >= 0",
             "min(open, close) > 0", "max(high, 1) > 0", "not close > 1000 or volume > 0"]
    for text in texts:
        full = compile_rule(text)(px).to_numpy()
        for cut in (60, 150, 399):
            assert np.array_equal(compile_rule(text)(px.iloc[: cut + 1]).to_numpy(), full[: cut + 1]), text


# ── tuning ────────────────────────────────────────────────────────────────────

def test_tune_chooses_on_the_first_part_and_judges_on_the_rest():
    px = walk(3000, seed=4)
    grid = {"low": [10, 15, 20], "high": [55, 60, 65]}
    tu = eol.tune(px, {"buy": "rsi(3) < {low}", "sell": "rsi(3) > {high}"}, grid, holdout=0.3)
    assert tu.status == "ok" and tu["tried"] == 9
    cut = tu["first_part"]["bars"]
    assert cut == round(3000 * 0.7) and tu["unseen_part"]["bars"] == 3000 - cut
    assert tu["in_sample"]["settings"]["variants_tried"] == 9
    assert tu["out_of_sample"]["settings"]["variants_tried"] == 1
    best = next(r for r in tu["table"] if r["settings"] == tu.best)
    assert best["in_sample_t"] == max(r["in_sample_t"] for r in tu["table"] if r["in_sample_t"] is not None)
    assert tu["out_of_sample"]["numbers"]["trades"] == best["unseen_trades"]
    assert tu.judgement.headline == tu["headline"]
    md = tu.to_markdown()
    assert "## Honest tuning:" in md and "Chosen settings" in md and "unseen part" in md


def test_tuning_on_random_walks_rarely_calls_an_edge():
    calls = []
    for seed in range(12):
        tu = eol.tune(walk(2500, seed=100 + seed), {"buy": "rsi(3) < {low}", "sell": "rsi(3) > {high}"},
                      {"low": [10, 20, 30], "high": [50, 60, 70]})
        calls.append(tu["out_of_sample"]["verdict"]["code"] if tu["out_of_sample"] else tu.status)
    assert calls.count("real_edge") <= 1, calls


def test_tune_refuses_silly_grids_and_holdouts():
    px = walk(500)
    rule = {"buy": "rsi(14) < {low}", "sell": "rsi(14) > 70"}
    with pytest.raises(RuleError):
        eol.tune(px, rule, {})
    with pytest.raises(RuleError):
        eol.tune(px, rule, {"low": list(range(2000))})
    with pytest.raises(RuleError):
        eol.tune(px, rule, {"low": [20]}, holdout=0.9)


def test_tune_with_too_few_trades_says_so():
    tu = eol.tune(walk(400), {"buy": "rsi(14) < {low}", "sell": "rsi(14) > 70"}, {"low": [1, 2]})
    assert tu.status == "too_few_trades" and tu.best is None and "nothing to choose from" in tu.to_markdown()


# ── prices ────────────────────────────────────────────────────────────────────

class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_prices_pages_through_binance_past_gaps_and_drops_the_unfinished_bar(monkeypatch):
    step = market.INTERVAL_MS["1h"]
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    gap = (now_ms // step - 8) * step  # the exchange was down for this bar
    calls = []

    def fake_urlopen(request, timeout=30):
        q = parse_qs(urlparse(request.full_url).query)
        calls.append(q)
        start, end, limit = int(q["startTime"][0]), int(q["endTime"][0]), int(q["limit"][0])
        rows = []
        for t in range(-(-start // step) * step, end + 1, step):
            if t == gap:
                continue
            if len(rows) == limit:
                break
            p = 100 + (t // step) % 7
            rows.append([t, str(p), str(p + 1), str(p - 1), str(p + 0.5), "10", t + step - 1, "0", 0, "0", "0", "0"])
        return _Response(json.dumps(rows).encode())

    monkeypatch.setattr(market.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(market, "KLINE_LIMIT", 4)
    market._mk_cache.clear()
    px = market.prices("eth/usdt", "1h", start=pd.Timestamp(now_ms - 10 * step, unit="ms", tz="UTC"))
    assert len(calls) == 3 and all(q["symbol"] == ["ETHUSDT"] for q in calls)
    assert list(px.columns) == ["open", "high", "low", "close", "volume"]
    assert px.index.is_monotonic_increasing and px.index.tz is not None
    open_ms = (px.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    assert (open_ms + step - 1 < now_ms).all()               # finished bars only
    assert open_ms[-1] == (now_ms // step - 1) * step        # ... right up to the last finished one
    assert gap not in set(open_ms) and len(px) in (8, 9)
    assert px.attrs["source"] == "Binance spot ETHUSDT 1h"
    n = len(calls)
    market.prices("ETHUSDT", "1h", start=pd.Timestamp(now_ms - 10 * step, unit="ms", tz="UTC"))
    assert len(calls) == n  # cached


def test_prices_explains_bad_input(monkeypatch):
    with pytest.raises(market.PriceError, match="interval"):
        market.prices("BTCUSDT", "7h")
    with pytest.raises(market.PriceError, match="date"):
        market.prices("BTCUSDT", "1h", start="last tuesday-ish")
    import urllib.error

    def refuse(request, timeout=30):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"code":-1121,"msg":"Invalid symbol."}'))
    monkeypatch.setattr(market.urllib.request, "urlopen", refuse)
    market._mk_cache.clear()
    with pytest.raises(market.PriceError, match="Invalid symbol"):
        market.prices("NOPEUSDT", "1h", start="2024-01-01", end="2024-01-02")


def test_load_prices_reads_csv_text_files_and_lists(tmp_path):
    text = "Date,Open,High,Low,Close,Volume\n" + "\n".join(
        f"2024-01-{d:02d},{100 + d},{101 + d},{99 + d},{100.5 + d},5" for d in range(1, 11))
    from_text = eol.load_prices(text)
    path = tmp_path / "px.csv"
    path.write_text(text, encoding="utf-8")
    from_file = eol.load_prices(str(path))
    pd.testing.assert_frame_equal(from_text, from_file)
    assert list(from_text.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(from_text.index, pd.DatetimeIndex)
    assert list(eol.load_prices([1, 2, 3])["close"]) == [1, 2, 3]
    one_column = tmp_path / "closes.csv"
    one_column.write_text("close\n100.5\n101\n99.25\n", encoding="utf-8")
    assert list(eol.load_prices(str(one_column))["close"]) == [100.5, 101, 99.25]
    european = "Datum;Close\n2024-01-01;100,5\n2024-01-02;101,25\n2024-01-03;99\n"
    assert list(eol.load_prices(european)["close"]) == [100.5, 101.25, 99]
    with pytest.raises(eol.PriceError, match="No price file"):
        eol.load_prices("missing_prices.csv")
    with pytest.raises(eol.PriceError, match="close"):
        eol.load_prices("a,b\n1,2\n3,4\n5,6\n")


# ── the MCP tools and the command line ────────────────────────────────────────

@pytest.fixture()
def price_csv(tmp_path):
    px = walk(2500, seed=4)
    path = tmp_path / "prices.csv"
    px.reset_index().to_csv(path, index=False)
    return path


def test_mcp_backtest_and_tune_tools_called_directly(price_csv):
    from edgeorluck.mcp_server import backtest_rule, rule_language, tune_rule
    md = backtest_rule(buy="rsi(3) < 15", sell="rsi(3) > 60", prices_file=str(price_csv), variants_tried=3)
    assert md.startswith("## Honest backtest:") and "## Edge or luck" in md
    for grid in ({"low": [10, 20], "high": [60, 70]}, '{"low": [10, 20], "high": [60, 70]}'):
        md = tune_rule(buy="rsi(3) < {low}", sell="rsi(3) > {high}", grid=grid, prices_file=str(price_csv))
        assert md.startswith("## Honest tuning:") and "Tried 4 settings" in md
    assert backtest_rule(buy="close.__class__ > 1", prices_file=str(price_csv)).startswith("Couldn't run")
    assert backtest_rule(buy="close > 1", sell="close < 1", prices_file="nope.csv").startswith("Couldn't run")
    assert tune_rule(buy="rsi(3) < {low}", sell="rsi(3) > 70", grid="not json",
                     prices_file=str(price_csv)).startswith("Couldn't tune")
    doc = rule_language()
    for name in eol.RULE_FUNCTIONS:
        assert f"{name}(" in doc


def test_cli_backtest_and_tune(price_csv, capsys):
    code = cli.main(["backtest", "--buy", "rsi(3) < 15", "--sell", "rsi(3) > 60", "--prices", str(price_csv),
                     "--tried", "2"])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("## Honest backtest:")
    code = cli.main(["tune", "--buy", "rsi(3) < {low}", "--sell", "rsi(3) > {high}", "--prices", str(price_csv),
                     "--grid", "low=10,20 high=60,70", "--json"])
    got = json.loads(capsys.readouterr().out)
    assert code == 0 and got["tried"] == 4 and got["out_of_sample"]["settings"]["variants_tried"] == 1
    code = cli.main(["backtest", "--buy", "import os", "--sell", "close < 1", "--prices", str(price_csv)])
    assert code == 1 and "edgeorluck backtest:" in capsys.readouterr().err
    code = cli.main(["backtest", "--buy", "close > 1e9", "--sell", "close < 1", "--prices", str(price_csv)])
    assert code == 2 and "never traded" in capsys.readouterr().out


def test_cli_grid_reads_lists_and_json():
    assert cli._grid(["low=20,25.5 high=70", '{"h": [1, 2]}']) == {"low": [20, 25.5], "high": [70], "h": [1, 2]}
    with pytest.raises(ValueError):
        cli._grid(["low"])


def test_the_one_file_build_backtests_like_the_package(tmp_path):
    import importlib.util
    import sys
    from edgeorluck import single
    target = single.write(str(tmp_path / "edgeorluck_single.py"))
    spec = importlib.util.spec_from_file_location("eol_single_bt", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eol_single_bt"] = mod
    try:
        spec.loader.exec_module(mod)
        px = walk(1500, seed=4)
        rule = {"buy": "rsi(3) < 15", "sell": "rsi(3) > 60"}
        assert mod.backtest(px, rule).to_dict() == eol.backtest(px, rule).to_dict()
        grid = {"low": [10, 20]}
        tuned = {"buy": "rsi(3) < {low}", "sell": "rsi(3) > 60"}
        assert mod.tune(px, tuned, grid).to_dict() == eol.tune(px, tuned, grid).to_dict()
    finally:
        sys.modules.pop("eol_single_bt", None)
