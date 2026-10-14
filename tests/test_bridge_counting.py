"""Bridge tests: the Python code against the Lean models in proofs/EdgeOrLuck/Counting.lean.

Each Lean theorem is about a model written to follow one Python function line by line. These
tests check, on many inputs, that the Python function computes what its model computes, and
that the identities Lean proves (in exact arithmetic) hold for the Python numbers within float
tolerance:

* `_p_streak_at_most` / `_max_losing_streak`  <->  `pStreakAtMost_eq`, `pStreakAtMost_eq_noRun`
* `_coin_flips` (the coin-flip test)          <->  `orbit_count_le`, `signFlip_mean`,
                                                   `signFlip_second_moment`
* `serial_factor`                             <->  `pairSum_eq`, `tendsto_ar1Ratio`,
                                                   `ar1Ratio_le_limit`, `serialFactorOf_mem`, ...
* `round_trips` (FIFO)                        <->  `fifo_coins`, `fifo_sold`, `fifo_cost`,
                                                   `fifo_pnl_pieces`, `fifo_flat`
* `_drawdown`                                 <->  `worstDrop_spec`, `lastPeak_is_last_high`
* `_exposure`                                 <->  `sweep_eq_volume`, `sweep_le`
"""
from __future__ import annotations

import importlib
import itertools
import math
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# `edgeorluck.judge` the attribute is the public function; the module is in sys.modules.
judge = importlib.import_module("edgeorluck.judge")
evidence = importlib.import_module("edgeorluck.evidence")
fills = importlib.import_module("edgeorluck.fills")


# ── 1. losing streaks ─────────────────────────────────────────────────────────

def _longest_run(bits) -> int:
    """Longest run of 1s, written independently of the code (regex over the string)."""
    s = "".join("1" if b else "0" for b in bits)
    runs = re.findall(r"1+", s)
    return max((len(r) for r in runs), default=0)


def _brute_p_streak(n: int, p: float, L: int) -> Tuple[float, float]:
    """Sum of p^losses (1-p)^wins over all 2^n sequences, two ways: longest run <= L, and
    'no L+1 losses in a row' (the form of pStreakAtMost_eq_noRun)."""
    by_run = by_window = 0.0
    window = "1" * (L + 1)
    for bits in itertools.product((0, 1), repeat=n):
        k = sum(bits)
        w = p ** k * (1.0 - p) ** (n - k)
        if _longest_run(bits) <= L:
            by_run += w
        if window not in "".join(map(str, bits)):
            by_window += w
    return by_run, by_window


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 11, 14])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.37, 0.5, 0.83, 1.0])
def test_streak_dp_equals_brute_force(n, p):
    for L in sorted({0, 1, 2, 3, max(n - 1, 0), n, n + 2}):
        by_run, by_window = _brute_p_streak(n, p, L)
        got = judge._p_streak_at_most(n, p, L)
        assert by_run == pytest.approx(by_window, abs=1e-12)
        assert got == pytest.approx(by_run, abs=1e-12), (n, p, L)


def test_streak_dp_without_its_shortcut_also_returns_one():
    """dpTotal_eq_one_of_le: for n <= L the DP itself gives 1, so the shortcut changes nothing."""
    for n in range(0, 9):
        for L in range(n, n + 3):
            for p in (0.0, 0.2, 0.5, 0.9, 1.0):
                state = [0.0] * (L + 1)
                state[0] = 1.0
                for _ in range(n):
                    nxt = [0.0] * (L + 1)
                    nxt[0] = sum(state) * (1.0 - p)
                    for k in range(L):
                        nxt[k + 1] = state[k] * p
                    state = nxt
                assert sum(state) == pytest.approx(1.0, abs=1e-12)


def test_max_losing_streak_is_the_longest_run_of_negative_trades():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        n = int(rng.integers(0, 40))
        x = rng.choice([-2.0, -1.0, 0.0, 1.0, 3.0], size=n)
        assert judge._max_losing_streak(x) == _longest_run(x < 0)


# ── 2. coin flips ─────────────────────────────────────────────────────────────

def _all_signs(n: int) -> np.ndarray:
    return np.array(list(itertools.product((1.0, -1.0), repeat=n))).reshape(2 ** n, n)


def _orbit_check(x: np.ndarray, T) -> None:
    """orbit_count_le, exhaustively: for y = g*x, p(y) = #{h : T(h*y) >= T(y)} / |G|, and for every
    u >= 0, #{g : p(g*x) <= u} / |G| <= u. T is evaluated once per point (a function of the point,
    as in the lemma; batched floating point could give one point two values)."""
    G = _all_signs(len(x))
    N = len(G)
    cache: Dict[tuple, float] = {}

    def t(y: np.ndarray) -> float:
        key = tuple(float(v) for v in y)
        if key not in cache:
            cache[key] = T(key)
        return cache[key]

    pvals = np.empty(N)
    for gi, g in enumerate(G):
        y = g * x
        ty = t(y)
        pvals[gi] = sum(1 for h in G if t(h * y) >= ty) / N
    for u in np.unique(np.concatenate((pvals, np.linspace(0, 1, 41)))):
        assert np.count_nonzero(pvals <= u) / N <= u + 1e-12


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_orbit_lemma_exhaustive(n):
    rng = np.random.default_rng(100 + n)
    w = [float(v) for v in rng.normal(size=n)]
    stats = [
        lambda y: math.fsum(y),                                          # the judge's statistic: the total
        lambda y: max(y),
        lambda y: math.sin(3.0 * math.fsum(a * b for a, b in zip(y, w))),  # an arbitrary statistic
        lambda y: float(round(math.fsum(y))),                            # many ties
    ]
    for _ in range(6):
        x = rng.choice([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], size=n) if rng.random() < 0.5 else rng.normal(size=n)
        for T in stats:
            _orbit_check(x, T)


@pytest.mark.parametrize("n", [3, 6, 10, 12])
def test_sign_flip_total_has_mean_zero_and_variance_sum_of_squares(n):
    """signFlip_mean, signFlip_second_moment: over all 2^n flips, the total has mean 0 and mean square sum(x^2), the
    centre and scale the normal approximation (above SIM_CELLS) uses."""
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) * rng.choice([0.1, 1.0, 10.0], size=n)
    S = _all_signs(n) @ x
    assert S.mean() == pytest.approx(0.0, abs=1e-9 * np.abs(x).sum())
    assert (S ** 2).mean() == pytest.approx(float(x @ x), rel=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_coin_flip_count_is_a_monte_carlo_draw_of_the_exact_share(seed):
    """The code draws 1000 random flips; its count should look like Binomial(1000, s) for the exact
    share s of flips that end strictly higher (it uses >, the exact p-value uses >=)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 13))
    x = np.round(rng.normal(0.2, 1.0, size=n), 2)
    S = _all_signs(n) @ x
    s = np.count_nonzero(S > x.sum() + 1e-12) / len(S)
    above, _ = judge._coin_flips(x, fan=False)
    sd = math.sqrt(judge.COIN_FLIPS * s * (1 - s))
    assert abs(above - judge.COIN_FLIPS * s) <= 6 * sd + 1


def test_a_passing_card_never_contradicts_its_coin_flips():
    """Found a display contradiction (fixed 2026-10-14): 'needed_at_most' comes from the e-value bar (a
    one-look z for the serial- and skew-adjusted t statistic), but 'ended_higher' counts sign flips of the
    raw trades (z = total/sqrt(sum x^2), smaller than t, plus Monte Carlo noise). A real_edge card could say
    '3 ended higher ... a pass here needs 1 or fewer ... It passes.' The sentence now gives the bar as odds,
    which the coin count can't contradict; the verdict always came from the e-values."""
    rng = np.random.default_rng(9)
    n = int(rng.choice([60, 100, 200]))
    m = float(rng.uniform(0.2, 0.6))
    x = np.round(rng.normal(m, 1.0, size=n), 3)
    card = judge.judge_dict([float(v) for v in x], unit="USDT")["strategies"][0]
    assert card["verdict"]["code"] == "real_edge"
    sentence = card["luck"]["sentence"]
    assert sentence.endswith("It passes.") and "needs" not in sentence and "or fewer" not in sentence


# ── 3. serial correlation ─────────────────────────────────────────────────────

def _ar1_ratio(r: float, n: int) -> float:
    return 1.0 + 2.0 * sum((1.0 - k / n) * r ** k for k in range(1, n))


def test_pair_sum_identity():
    """pairSum_eq: sum_{i,j<n} r^|i-j| = n + 2 sum_{k=1}^{n-1} (n-k) r^k."""
    for n in range(0, 30):
        for r in (-0.9, -0.3, 0.0, 0.25, 0.6, 0.95, 1.0, 1.3):
            i = np.arange(n)
            lhs = float((r ** np.abs(i[:, None] - i[None, :])).sum()) if n else 0.0
            rhs = n + 2.0 * sum((n - k) * r ** k for k in range(1, n))
            assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_ar1_ratio_limit_and_bound():
    """tendsto_ar1Ratio and ar1Ratio_le_limit."""
    for r in (-0.8, -0.3, 0.0, 0.2, 0.5, 0.8, 0.95):
        limit = (1 + r) / (1 - r)
        assert _ar1_ratio(r, 200_000) == pytest.approx(limit, rel=2e-3 if r < 0.9 else 1e-2)
        if r >= 0:
            assert all(_ar1_ratio(r, n) <= limit + 1e-12 for n in range(1, 400))


def test_ar1_ratio_matches_simulated_variance_of_the_mean():
    """variance_mean_ar1: n * Var(mean) / sigma^2 = ar1Ratio r n for a stationary AR(1)."""
    rng = np.random.default_rng(2026)
    reps = 40_000
    for r, n in ((0.3, 20), (0.6, 30), (-0.4, 25)):
        x = np.empty((reps, n))
        x[:, 0] = rng.normal(size=reps)
        eps = rng.normal(size=(reps, n)) * math.sqrt(1 - r * r)
        for t in range(1, n):
            x[:, t] = r * x[:, t - 1] + eps[:, t]
        est = n * x.mean(axis=1).var()
        assert est == pytest.approx(_ar1_ratio(r, n), rel=0.03)


def _serial_model(x: np.ndarray) -> float:
    """serialFactorOf nw (ar1Floor (kendallCapped r1 n)), computed independently."""
    n = len(x)
    if n < 20:
        return 1.0
    d = x - x.mean()
    var = float(d @ d) / n
    if var <= 0:
        return 1.0
    lags = max(int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), 1)
    acf = [float(d[k:] @ d[:-k]) / n / var for k in range(1, lags + 1)]
    nw = 1.0 + 2.0 * sum((1.0 - k / (lags + 1.0)) * acf[k - 1] for k in range(1, lags + 1))
    r = min(0.8, acf[0] + (1.0 + 3.0 * acf[0]) / n)
    ar1 = (1.0 + r) / (1.0 - r) if r > 0 else 1.0
    assert 1.0 <= ar1 <= 9.0 + 1e-12                          # ar1Floor_mem
    return min(10.0, max(1.0, nw, ar1))


def test_serial_factor_follows_its_model_and_stays_in_one_to_ten():
    rng = np.random.default_rng(11)
    for trial in range(600):
        n = int(rng.integers(2, 400))
        kind = trial % 4
        if kind == 0:
            x = rng.normal(size=n)
        elif kind == 1:                                       # strong positive AR
            phi = rng.uniform(0.5, 0.999)
            x = np.zeros(n)
            for t in range(1, n):
                x[t] = phi * x[t - 1] + rng.normal()
        elif kind == 2:                                       # alternating: negative correlation
            x = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) + 0.1 * rng.normal(size=n)
        else:                                                 # trending: correlation near 1
            x = np.arange(n, dtype=float) + rng.normal(size=n)
        got = evidence.serial_factor(x)
        assert 1.0 <= got <= 10.0                             # serialFactorOf_mem
        assert got == pytest.approx(_serial_model(x), rel=1e-12)


def test_ar1_factor_facts():
    """one_le_ar1Factor, ar1Factor_strictMonoOn, ar1Factor_le_nine, at their edges."""
    rs = np.linspace(0.0, 0.8, 801)
    f = (1 + rs) / (1 - rs)
    assert (f >= 1).all() and (np.diff(f) > 0).all() and f[-1] == pytest.approx(9.0)


# ── 4. FIFO accounting ────────────────────────────────────────────────────────

def _take(lots: List[List[Fraction]], q: Fraction):
    """takeFifo: take q coins from the head; returns (lots left, pieces taken)."""
    left, pieces = [list(l) for l in lots], []
    while q > 0 and left:
        qty, cost = left[0]
        if q < qty:
            pieces.append([q, cost])
            left[0][0] = qty - q
            q = Fraction(0)
        else:
            pieces.append([qty, cost])
            left.pop(0)
            q -= qty
    return left, pieces


def _lean_book(rows, dust: float):
    """Book.step over the fills, exactly (Fractions), with the drop-the-rest decision taken
    the way round_trips takes it (left <= DUST * peak since the episode began)."""
    lots: List[List[Fraction]] = []
    b = dict(matched=Fraction(0), unmatched=Fraction(0), unmatched_cash=Fraction(0), basis=Fraction(0),
             proceeds=Fraction(0), piece_pnl=Fraction(0), dust=Fraction(0), dust_cost=Fraction(0))
    sells: List[Fraction] = []          # per-sell realized P&L of sells that matched something
    b["short_sells"] = 0                # sells larger than what is held, with the code's display tolerance
    peak, open_episode = Fraction(0), False
    D = Fraction(dust)
    for r in rows:
        q, p, fee = Fraction(r["qty"]), Fraction(r["price"]), Fraction(r["fee"])
        if r["side"] > 0:
            if not open_episode:
                open_episode, peak = True, Fraction(0)
            lots.append([q, (q * p + fee) / q])
            peak = max(peak, sum(l[0] for l in lots))
            continue
        held = sum((l[0] for l in lots), Fraction(0))
        sold = min(q, held)
        if held <= 0 or q > held * (1 + Fraction(1e-9)) + Fraction(1e-12):
            b["short_sells"] += 1
        left, pieces = _take(lots, sold)
        b["matched"] += sum((x[0] for x in pieces), Fraction(0))
        b["unmatched"] += q - sold
        b["unmatched_cash"] += (p - fee / q) * (q - sold)
        basis = sum((x[0] * x[1] for x in pieces), Fraction(0))
        b["basis"] += basis
        proceeds = sold * p - fee * (sold / q)
        b["proceeds"] += proceeds
        b["piece_pnl"] += sum(((p - x[1]) * x[0] for x in pieces), Fraction(0)) - fee * (sold / q)
        drop = held > 0 and sum((l[0] for l in left), Fraction(0)) <= D * peak
        if held > 0:
            sells.append(proceeds - basis)
        if drop:
            b["dust"] += sum((l[0] for l in left), Fraction(0))
            b["dust_cost"] += sum((l[0] * l[1] for l in left), Fraction(0))
            left, open_episode = [], False
        lots = left
    b["lots"] = lots
    return b, sells


def _random_fills(rng, n: int, never_flat: bool = False) -> List[Dict]:
    rows, held, t0 = [], 0.0, pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(n):
        price = float(np.round(rng.uniform(50, 150), 2))
        buy = (i == 0) or rng.random() < 0.5
        if never_flat and not buy and held < 2.0:
            buy = True
        if buy:
            qty = float(np.round(rng.uniform(0.1, 3.0), 3))
            held += qty
        else:
            if never_flat:
                qty = float(np.round(rng.uniform(0.01, held - 1.0), 3))
            else:
                qty = float(np.round(rng.choice([rng.uniform(0.05, max(held, 0.1)), held, held * 1.3, 0.7]), 3))
            if qty <= 0:
                continue
            held = max(held - qty, 0.0)
        fee = float(np.round(price * qty * rng.choice([0.0, 0.001, 0.002]), 6))
        rows.append({"time": t0 + pd.Timedelta(minutes=i), "pair": "BTCUSDT", "side": 1 if buy else -1,
                     "price": price, "qty": qty, "fee": fee, "fee_coin": rng.choice(["USDT", None]),
                     "quote_total": float("nan")})
    return rows


def _sum_by_side(rows, side, f):
    return sum((f(r) for r in rows if r["side"] == side), Fraction(0))


def _fifo_identities(rows, b):
    """The Lean identities, checked in exact arithmetic on the model."""
    F = lambda v: Fraction(v)
    bought = _sum_by_side(rows, 1, lambda r: F(r["qty"]))
    sold = _sum_by_side(rows, -1, lambda r: F(r["qty"]))
    buy_cost = _sum_by_side(rows, 1, lambda r: F(r["qty"]) * F(r["price"]) + F(r["fee"]))
    sell_cash = _sum_by_side(rows, -1, lambda r: F(r["qty"]) * F(r["price"]) - F(r["fee"]))
    held = sum((l[0] for l in b["lots"]), Fraction(0))
    open_cost = sum((l[0] * l[1] for l in b["lots"]), Fraction(0))
    realized = b["proceeds"] - b["basis"]
    assert bought == b["matched"] + held + b["dust"]                          # fifo_coins
    assert sold == b["matched"] + b["unmatched"]                              # fifo_sold
    assert buy_cost == b["basis"] + open_cost + b["dust_cost"]                # fifo_cost
    assert realized == b["piece_pnl"]                                         # fifo_pnl_pieces
    assert sell_cash == b["proceeds"] + b["unmatched_cash"]                   # fifo_cash
    assert realized == b["proceeds"] - buy_cost + open_cost + b["dust_cost"]  # fifo_pnl_cash
    if not b["lots"] and b["dust_cost"] == 0 and b["unmatched"] == 0:         # fifo_flat
        assert realized == sell_cash - buy_cost
    return realized, buy_cost, sell_cash, open_cost, held


def _python_round_trips(rows):
    frame, warnings, facts = fills.round_trips([dict(r) for r in rows], float)
    open_qty = sum(p["quantity"] for p in facts["open_positions"])
    return frame, facts, open_qty


@pytest.mark.parametrize("dust", [0.0, fills.DUST])
@pytest.mark.parametrize("seed", range(40))
def test_round_trips_follow_the_fifo_model(monkeypatch, dust, seed):
    monkeypatch.setattr(fills, "DUST", dust)
    rng = np.random.default_rng(seed)
    rows = _random_fills(rng, int(rng.integers(1, 60)))
    b, sells = _lean_book(rows, dust)
    realized, buy_cost, sell_cash, open_cost, held = _fifo_identities(rows, b)
    frame, facts, open_qty = _python_round_trips(rows)
    scale = 1.0 + float(buy_cost) + float(sell_cash)
    assert float(frame["pnl"].sum()) == pytest.approx(float(realized), abs=1e-9 * scale)
    assert float(frame["size"].sum()) == pytest.approx(float(b["basis"]), abs=1e-9 * scale)
    assert open_qty == pytest.approx(float(held), abs=1e-8)
    assert facts["unmatched_sells"] == b["short_sells"]


@pytest.mark.parametrize("seed", range(30))
def test_round_trips_match_fifo_sell_by_sell(seed):
    """A position never closed: round_trips reports each sell on its own, so the per-sell P&L
    shows the matching order. FIFO (the model) must agree; LIFO would not."""
    rng = np.random.default_rng(1000 + seed)
    rows = _random_fills(rng, int(rng.integers(5, 50)), never_flat=True)
    b, sells = _lean_book(rows, fills.DUST)
    _fifo_identities(rows, b)
    frame, facts, open_qty = _python_round_trips(rows)
    if len(frame) == 0:
        assert not sells
        return
    assert len(frame) == len(sells)
    got = frame["pnl"].to_numpy(dtype=float)
    want = np.array([float(s) for s in sells])
    assert got == pytest.approx(want, abs=1e-9 * (1.0 + np.abs(want).max()))


def test_round_trips_flat_book_is_cash_in_minus_cash_out():
    """fifo_flat on a hand-made book: two buys, two sells, flat at the end."""
    t = pd.Timestamp("2026-03-01", tz="UTC")
    rows = [
        {"time": t, "pair": "ETHUSDT", "side": 1, "price": 100.0, "qty": 2.0, "fee": 0.2, "fee_coin": "USDT",
         "quote_total": float("nan")},
        {"time": t + pd.Timedelta(hours=1), "pair": "ETHUSDT", "side": 1, "price": 110.0, "qty": 1.0, "fee": 0.11,
         "fee_coin": "USDT", "quote_total": float("nan")},
        {"time": t + pd.Timedelta(hours=2), "pair": "ETHUSDT", "side": -1, "price": 120.0, "qty": 2.5, "fee": 0.3,
         "fee_coin": "USDT", "quote_total": float("nan")},
        {"time": t + pd.Timedelta(hours=3), "pair": "ETHUSDT", "side": -1, "price": 90.0, "qty": 0.5, "fee": 0.045,
         "fee_coin": "USDT", "quote_total": float("nan")},
    ]
    frame, facts, open_qty = _python_round_trips(rows)
    cash_in = 2.5 * 120 - 0.3 + 0.5 * 90 - 0.045
    cash_out = 2 * 100 + 0.2 + 110 + 0.11
    assert len(frame) == 1 and open_qty == 0
    assert float(frame["pnl"].iloc[0]) == pytest.approx(cash_in - cash_out, abs=1e-9)


# ── 5. drawdown ───────────────────────────────────────────────────────────────

def _drawdown_by_definition(x: np.ndarray):
    """worstDrop_spec and lastPeak_is_last_high, by brute force over pairs."""
    C = np.concatenate(([0.0], np.cumsum(x)))
    n = len(C) - 1
    worst = max(C[s] - C[t] for t in range(n + 1) for s in range(t + 1))
    current = max(C[: n + 1]) - C[n]
    last_high = max(i for i in range(n + 1) if C[i] == C.max())          # last time at the overall high
    last_peak = max(i for i in range(n + 1) if C[i] == C[: i + 1].max())  # last time at the running peak
    assert last_high == last_peak
    return worst, current, n - last_peak


def test_drawdown_matches_the_pairwise_definition():
    rng = np.random.default_rng(5)
    for _ in range(400):
        n = int(rng.integers(1, 120))
        x = rng.integers(-5, 6, size=n).astype(float) / rng.choice([1.0, 8.0])   # exact in floats
        assert judge._drawdown(x) == _drawdown_by_definition(x)


def test_a_curve_exactly_back_at_its_peak_is_at_its_peak():
    """Found a defect (fixed 2026-10-14): _drawdown tested 'at the peak' as peak - curve <= 1e-12, an
    absolute tolerance. With large amounts the float cumsum misses the peak by ~1e-10 when the curve is
    exactly back at it, so the card said 'You are 0.000000 USDT below your peak, 2 trades ago.' The
    tolerance is now relative to the curve's scale."""
    x = np.array([628040.6, -72356.44, 72356.44])      # in exact arithmetic C_3 = C_1, the high
    worst, current, since = judge._drawdown(x)
    assert since == 0 and current == 0.0


# ── 6. time in the market ─────────────────────────────────────────────────────

def _exposure_by_grid(spans: List[Tuple[int, int]]) -> Optional[float]:
    spans = [(s, e) for s, e in spans if e >= s]
    if not spans:
        return None
    lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
    if hi <= lo:
        return None
    covered = np.zeros(hi - lo, dtype=bool)          # unit cells [k, k+1)
    for s, e in spans:
        covered[s - lo:e - lo] = True
    return covered.sum() / (hi - lo)


def test_exposure_is_the_union_length_over_the_span():
    rng = np.random.default_rng(9)
    base = pd.Timestamp("2026-01-01", tz="UTC").value
    for _ in range(500):
        k = int(rng.integers(1, 12))
        starts = rng.integers(0, 200, size=k)
        lengths = rng.integers(-10, 60, size=k)            # some exits before entries: dropped
        spans = [(int(s), int(s + l)) for s, l in zip(starts, lengths)]
        entries = [pd.Timestamp(base + s, tz="UTC") for s, _ in spans]
        exits = [pd.Timestamp(base + e, tz="UTC") for _, e in spans]
        got = judge._exposure(entries, exits)
        want = _exposure_by_grid(spans)
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-12)
            assert 0.0 <= got <= 1.0
