"""How often is the judge wrong? Measure it, and write VALIDATION.md plus the
small JSON the page reads (src/edgeorluck/data/validation.json).

    python tools/validate.py            # the full run (a few minutes)
    python tools/validate.py --quick    # a smoke run with few repetitions

Every number comes from simulated trade records whose truth is known, fixed
seeds, and the same judge() code path the package uses (without the
pictures, which don't change verdicts). Rates carry 95% Wilson ranges, so
simulation noise is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import edgeorluck  # noqa: E402
from edgeorluck import evidence, samples  # noqa: E402
from edgeorluck.evidence import wilson_bounds_count  # noqa: E402
from edgeorluck.judge import judge_dict  # noqa: E402

COUNTS = (30, 100, 300, 1000)
TARGET_REAL = evidence.Q / 2          # one direction of the family: 2.5%


# ── trade generators (all with mean zero unless an edge is added) ─────────────

def normal(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, n)


def heavy(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_t(3, n) / math.sqrt(3.0)


def autocorrelated(n: int, rng: np.random.Generator, phi: float = 0.3) -> np.ndarray:
    e = rng.normal(0.0, 1.0, n + 50)
    x = np.zeros(n + 50)
    for i in range(1, n + 50):
        x[i] = phi * x[i - 1] + e[i]
    return x[50:] * math.sqrt(1 - phi * phi)


def small_wins_big_losses(n: int, rng: np.random.Generator) -> np.ndarray:
    """Negatively skewed: most trades win a little, a few lose a lot."""
    y = rng.lognormal(0.0, 1.0, n)
    return -(y - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)


def martingale(n: int, rng: np.random.Generator) -> np.ndarray:
    """97% of trades win 1, 3% lose 32.3: zero on average."""
    return np.where(rng.random(n) < 0.97, 1.0, -0.97 / 0.03) + rng.normal(0, 0.05, n)


SHAPES: Dict[str, Callable] = {"normal": normal, "heavy_tailed": heavy, "autocorrelated": autocorrelated,
                               "small_wins_big_losses": small_wins_big_losses, "martingale": martingale}
SHAPE_WORDS = {"normal": "Normal", "heavy_tailed": "Heavy tails (Student t, 3 d.f.)",
               "autocorrelated": "Autocorrelated (AR 0.3)",
               "small_wins_big_losses": "Small wins, big losses (skewed)", "martingale": "Martingale (3% big losses)"}


def verdict(x, **kw) -> str:
    return judge_dict(list(x), details=False, **kw)["verdict"]["code"]


def rate(hits: int, reps: int) -> Dict[str, float]:
    lo, hi = wilson_bounds_count(hits, reps)
    return {"rate": hits / reps, "low": lo, "high": hi, "hits": hits, "reps": reps}


def pct(r: Dict[str, float]) -> str:
    return f"{100 * r['rate']:.1f}% ({100 * r['low']:.1f}-{100 * r['high']:.1f})"


# ── the measurements ──────────────────────────────────────────────────────────

def false_rates(reps: int) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for name, gen in SHAPES.items():
        out[name] = {}
        for n in COUNTS:
            real = any_label = 0
            for r in range(reps):
                code = verdict(gen(n, np.random.default_rng(1_000_003 * n + r)))
                real += code == "real_edge"
                any_label += code in ("real_edge", "losing")
            out[name][str(n)] = {"real_edge": rate(real, reps), "any_label": rate(any_label, reps)}
    return out


def without_fixes(reps: int) -> Dict[str, Dict[str, Dict]]:
    """The same no-edge records with the serial and skew corrections off and
    no five-losses rule: what the fixes buy."""
    judge_module = sys.modules["edgeorluck.judge"]      # the module; edgeorluck.judge is the function
    saved = (evidence.SERIAL_ADJUST, evidence.SKEW_ADJUST, judge_module.MIN_LOSSES)
    evidence.SERIAL_ADJUST, evidence.SKEW_ADJUST, judge_module.MIN_LOSSES = False, False, 0
    evidence.boundary.cache_clear()
    try:
        out: Dict[str, Dict[str, Dict]] = {}
        for name in ("autocorrelated", "small_wins_big_losses", "martingale"):
            out[name] = {}
            for n in (30, 100, 300):
                real = sum(verdict(SHAPES[name](n, np.random.default_rng(1_000_003 * n + r))) == "real_edge"
                           for r in range(reps))
                out[name][str(n)] = rate(real, reps)
        return out
    finally:
        evidence.SERIAL_ADJUST, evidence.SKEW_ADJUST, judge_module.MIN_LOSSES = saved


def checked_often(reps: int, last: int = 300, every: int = 10) -> Dict[str, Dict]:
    """A trader who checks after every 10 trades, from 30 to 300: how often
    is a no-edge record EVER called real?"""
    out = {}
    for name in ("normal", "autocorrelated"):
        ever = 0
        for r in range(reps):
            x = SHAPES[name](last, np.random.default_rng(7_000_001 + r))
            for n in range(30, last + 1, every):
                if verdict(x[:n]) == "real_edge":
                    ever += 1
                    break
        out[name] = rate(ever, reps)
    return out


def checked_often_newey_west_only(reps: int) -> Dict[str, float]:
    """The autocorrelated row of checked_often with Newey-West alone (no AR(1)
    floor): the cell that failed before the floor was added."""
    evidence.SERIAL_AR1_FLOOR = False
    try:
        return {"autocorrelated": checked_often(reps)["autocorrelated"]}
    finally:
        evidence.SERIAL_AR1_FLOOR = True


def best_of_n(reps: int, n_trades: int = 200) -> Dict[str, Dict]:
    out = {}
    for versions in (10, 100, 1000):
        without = with_ = 0
        for r in range(reps):
            noise = np.random.default_rng(9_000_011 * versions + r).normal(0, 1, (versions, n_trades))
            best = noise[np.argmax(noise.sum(axis=1))]
            without += verdict(best) == "real_edge"
            with_ += verdict(best, variants_tried=versions) == "real_edge"
        out[str(versions)] = {"without": rate(without, reps), "with": rate(with_, reps)}
    return out


def power(reps: int) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for edge in (0.1, 0.2, 0.3):
        out[str(edge)] = {}
        for n in COUNTS:
            hits = sum(verdict(np.random.default_rng(3_000_017 * n + r).normal(edge, 1.0, n)) == "real_edge"
                       for r in range(reps))
            out[str(edge)][str(n)] = rate(hits, reps)
    return out


def faded(reps: int) -> Dict[str, Dict]:
    steady = sum(verdict(np.random.default_rng(5_000_000 + r).normal(0.25, 1.0, 400)) == "faded" for r in range(reps))
    caught = 0
    for r in range(reps):
        rng = np.random.default_rng(6_000_000 + r)
        x = np.concatenate((rng.normal(0.25, 1.0, 340), rng.normal(-0.4, 1.0, 60)))
        caught += verdict(x) == "faded"
    return {"steady_edge_called_faded": rate(steady, reps), "real_fade_caught": rate(caught, reps)}


def compare_runs(reps: int) -> Dict[str, Dict]:
    false = caught = 0
    for r in range(reps):
        rng = np.random.default_rng(8_000_000 + r)
        backtest = list(rng.normal(0.25, 1.0, 300))
        same, worse = rng.normal(0.25, 1.0, 200), rng.normal(-0.15, 1.0, 200)
        for n in range(10, 201, 10):
            if judge_dict(list(same[:n]), backtest=backtest, details=False)["compare"]["code"] == \
                    "worse_than_backtest":
                false += 1
                break
        for n in range(10, 201, 10):
            if judge_dict(list(worse[:n]), backtest=backtest, details=False)["compare"]["code"] == \
                    "worse_than_backtest":
                caught += 1
                break
    return {"false_worse_checked_every_10": rate(false, reps), "worse_caught_by_200": rate(caught, reps)}


def determinism() -> Dict[str, object]:
    digests = {}
    for s in samples.SAMPLES:
        a = edgeorluck.judge(**samples.sample_input(s["id"]), include_chart=True).to_json()
        b = edgeorluck.judge(**samples.sample_input(s["id"]), include_chart=True).to_json()
        assert a == b, s["id"]
        digests[s["id"]] = hashlib.sha256(a.encode("utf-8")).hexdigest()
    code = ("import sys, json, hashlib; sys.path.insert(0, %r); import edgeorluck; from edgeorluck import samples; "
            "print(json.dumps({s['id']: hashlib.sha256(edgeorluck.judge(**samples.sample_input(s['id']), "
            "include_chart=True).to_json().encode('utf-8')).hexdigest() for s in samples.SAMPLES}))"
            % str(ROOT / "src"))
    fresh = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                      check=True).stdout)
    return {"samples": len(digests), "same_in_process": True, "same_in_a_fresh_process": fresh == digests}


def runtime() -> Dict[str, float]:
    out = {}
    for n in (100, 1000):
        x = list(np.random.default_rng(1).normal(0.1, 1.0, n))
        edgeorluck.judge(x)
        times = []
        for _ in range(7):
            t = time.perf_counter()
            edgeorluck.judge(x)
            times.append(time.perf_counter() - t)
        out[str(n)] = 1000 * float(np.median(times))
    return out


# ── the report ────────────────────────────────────────────────────────────────

def write(results: Dict, quick: bool) -> None:
    fr, wf, often = results["false_real_edge"], results["without_fixes"], results["checked_often"]
    bon, pw, fd, cmp_ = results["best_of_n"], results["power"], results["faded"], results["compare"]
    worst = max(fr[s][str(n)]["real_edge"]["rate"] for s in fr for n in COUNTS)
    lines = [
        "# How often is the judge wrong?",
        "",
        f"Measured by `tools/validate.py` on simulated trade records whose truth is known (engine "
        f"{edgeorluck.__version__}, {results['date']}{', QUICK RUN: few repetitions' if quick else ''}). Rerun it "
        "to check every number. Rates show a 95% range for simulation noise in brackets.",
        "",
        "## 1. Records with no edge: how often are they called a \"real edge\"?",
        "",
        f"The target is at most {100 * TARGET_REAL:.1f}% (one side of a 5% error budget shared by \"real edge\" "
        "and \"losing\"). One look at the record, no versions tried.",
        "",
        "| Trades look like | " + " | ".join(f"{n} trades" for n in COUNTS) + " |",
        "|---|" + "---|" * len(COUNTS),
    ]
    for s, label in SHAPE_WORDS.items():
        lines.append(f"| {label} | " + " | ".join(pct(fr[s][str(n)]["real_edge"]) for n in COUNTS) + " |")
    lines += [
        "",
        f"Worst cell: {100 * worst:.1f}%. \"Any label\" (real edge or losing, the whole 5% budget):",
        "",
        "| Trades look like | " + " | ".join(f"{n} trades" for n in COUNTS) + " |",
        "|---|" + "---|" * len(COUNTS),
    ]
    for s, label in SHAPE_WORDS.items():
        lines.append(f"| {label} | " + " | ".join(pct(fr[s][str(n)]["any_label"]) for n in COUNTS) + " |")
    lines += [
        "",
        "### What the corrections buy",
        "",
        "Before these corrections, skewed and martingale records broke the target on a single look, and "
        "autocorrelated records broke it when checked again and again (section 2). The same records with the "
        "corrections switched off (no widening for autocorrelation, no skew-raised bar, no five-losses rule):",
        "",
        "| Trades look like | 30 trades | 100 trades | 300 trades |",
        "|---|---|---|---|",
    ]
    for s in wf:
        lines.append(f"| {SHAPE_WORDS[s]} | " + " | ".join(pct(wf[s][n]) for n in ("30", "100", "300")) + " |")
    lines += [
        "",
        "Honest remainder: autocorrelated and skewed records are still called real more often than normal ones, "
        "and every guarantee here is asymptotic (normal approximation, estimated spread). Records with a few "
        "dozen trades and rare, huge losses remain the hardest case; the judge says \"can't tell\" until at least "
        "five losses have happened.",
        "",
        "## 2. Checking again and again",
        "",
        "A trader who re-checks after every 10 trades, from 30 to 300 trades, on a record with no edge. How often "
        "is it EVER called a real edge? A normal confidence interval read this way fails far more often than its "
        "stated rate; the anytime-valid range is built for it.",
        "",
        f"- Normal trades: {pct(often['normal'])}",
        f"- Autocorrelated trades: {pct(often['autocorrelated'])} (with Newey-West alone it was "
        f"{pct(results['checked_often_newey_west_only']['autocorrelated'])}: Newey-West under-corrects at a few "
        "dozen trades, so the spread is now widened by at least a bias-corrected AR(1) factor, with an allowance "
        "of 12/n for the noise in that factor's estimate)",
        "",
        "## 3. The best of many versions",
        "",
        "Try N versions of a strategy with no edge (200 trades each), keep the best one, and judge it. Without "
        "`variants_tried` the judge can only assume it was the only version; with `variants_tried=N` it raises "
        "the bar to match.",
        "",
        "| Versions tried | Called real, not told | Called real, told `variants_tried=N` |",
        "|---|---|---|",
    ]
    for v in ("10", "100", "1000"):
        lines.append(f"| {v} | {pct(bon[v]['without'])} | {pct(bon[v]['with'])} |")
    lines += [
        "",
        "## 4. Power: how often is a real edge found?",
        "",
        "Normal trades with a true edge of 0.1, 0.2 or 0.3 standard deviations per trade (for a strategy whose "
        "trades swing about 2%, that is +0.2%, +0.4% and +0.6% per trade).",
        "",
        "| Edge per trade | " + " | ".join(f"{n} trades" for n in COUNTS) + " |",
        "|---|" + "---|" * len(COUNTS),
    ]
    for e in ("0.1", "0.2", "0.3"):
        lines.append(f"| {e} SD | " + " | ".join(pct(pw[e][str(n)]) for n in COUNTS) + " |")
    lines += [
        "",
        "This is the price of an answer that stays valid however often you check, and of a strict error budget: "
        "small edges need many trades. The judge says so (\"about N more trades would settle it\") instead of "
        "guessing.",
        "",
        "## 5. Faded, and live vs backtest",
        "",
        f"- A steady edge (0.25 SD, 400 trades) wrongly called \"faded\": {pct(fd['steady_edge_called_faded'])}.",
        f"- A real fade (340 good trades, then 60 at -0.4 SD) called \"faded\": {pct(fd['real_fade_caught'])}.",
        f"- Live trades as good as the backtest, compared every 10 trades up to 200, ever called \"worse than "
        f"the backtest\": {pct(cmp_['false_worse_checked_every_10'])}.",
        f"- Live trades 0.4 SD worse than the backtest, caught within 200 trades: "
        f"{pct(cmp_['worse_caught_by_200'])}.",
        "",
        "## 6. Same file, same answer",
        "",
        f"Every sample was judged twice in one process and once in a fresh process: identical JSON every time "
        f"({'yes' if results['determinism']['same_in_a_fresh_process'] else 'NO'}). Random pictures (coin-flip "
        "copies, resampled histories) are seeded from the trades themselves; the result carries "
        "`input_fingerprint` so anyone can check they judged the same trades.",
        "",
        "## 7. Speed",
        "",
        f"Median `judge()` time with every check and picture: {results['runtime_ms']['100']:.0f} ms for 100 trades, "
        f"{results['runtime_ms']['1000']:.0f} ms for 1,000 (on the machine that ran this).",
        "",
        "## Method in one paragraph",
        "",
        "The range is an anytime-valid 95% range (a confidence sequence; Waudby-Smith, Arbour, Sinha, Kennedy & "
        "Ramdas 2021; Howard, Ramdas, McAuliffe & Sekhon 2021) from a one-sided normal-mixture e-value. Every "
        "version tried and every strategy shown joins one e-BH family (Wang & Ramdas 2022) at a 5% false "
        "discovery rate; versions tried but not shown count as e = 0. The spread is widened for autocorrelation "
        "(the larger of Newey-West and a bias-corrected AR(1) factor), the bar is raised for skew against the "
        "claim (Hall 1987), and \"real edge\" needs five losing trades. See `src/edgeorluck/evidence.py`.",
        "",
    ]
    (ROOT / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def headline(results: Dict) -> Dict:
    fr = results["false_real_edge"]
    cells = [fr[s][str(n)]["real_edge"]["rate"] for s in fr for n in COUNTS]
    normal_cells = [fr["normal"][str(n)]["real_edge"]["rate"] for n in COUNTS]
    return {
        "false_real_edge_typical": max(normal_cells),
        "false_real_edge_worst": max(cells),
        "false_real_edge_target": TARGET_REAL,
        "checked_every_10_trades": results["checked_often"]["normal"]["rate"],
        "best_of_100_without": results["best_of_n"]["100"]["without"]["rate"],
        "best_of_100_with": results["best_of_n"]["100"]["with"]["rate"],
        "power_0.2sd_300_trades": results["power"]["0.2"]["300"]["rate"],
        "power_0.2sd_1000_trades": results["power"]["0.2"]["1000"]["rate"],
        "same_file_same_answer": results["determinism"]["same_in_a_fresh_process"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--render", action="store_true", help="rewrite VALIDATION.md from the saved JSON")
    args = parser.parse_args()
    saved = ROOT / "src" / "edgeorluck" / "data" / "validation.json"
    if args.render:
        results = json.loads(saved.read_text(encoding="utf-8"))
        write(results, bool(results.get("quick")))
        print("rewrote VALIDATION.md from", saved.relative_to(ROOT))
        return 0
    reps = 60 if args.quick else 2000
    started = time.time()
    results: Dict = {"engine_version": edgeorluck.__version__, "date": date.today().isoformat(),
                     "quick": args.quick, "reps": {}}
    steps = [("false_real_edge", false_rates, reps), ("without_fixes", without_fixes, max(60, reps // 2)),
             ("checked_often", checked_often, max(60, reps // 2)),
             ("checked_often_newey_west_only", checked_often_newey_west_only, max(60, reps // 2)),
             ("best_of_n", best_of_n, max(60, reps // 4)),
             ("power", power, max(60, reps // 2)), ("faded", faded, max(60, reps // 4)),
             ("compare", compare_runs, max(40, reps // 8))]
    for key, fn, n in steps:
        t = time.time()
        results[key] = fn(n)
        results["reps"][key] = n
        print(f"{key}: {time.time() - t:.0f}s", flush=True)
    results["determinism"] = determinism()
    results["runtime_ms"] = runtime()
    results["headline"] = headline(results)
    results["target"] = {"false_real_edge": TARGET_REAL, "false_discovery_rate": evidence.Q}
    write(results, args.quick)
    out = ROOT / "src" / "edgeorluck" / "data" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote VALIDATION.md and {out.relative_to(ROOT)} in {time.time() - started:.0f}s")
    print(json.dumps(results["headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
