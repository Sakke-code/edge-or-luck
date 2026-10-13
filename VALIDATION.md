# How often is the judge wrong?

Measured by `tools/validate.py` on simulated trade records whose truth is known (engine 2.0.0, 2026-10-14). Rerun it to check every number. Rates show a 95% range for simulation noise in brackets.

## 1. Records with no edge: how often are they called a "real edge"?

The target is at most 2.5% (one side of a 5% error budget shared by "real edge" and "losing"). One look at the record, no versions tried.

| Trades look like | 30 trades | 100 trades | 300 trades | 1000 trades |
|---|---|---|---|---|
| Normal | 0.1% (0.1-0.4) | 0.1% (0.0-0.4) | 0.1% (0.0-0.4) | 0.0% (0.0-0.2) |
| Heavy tails (Student t, 3 d.f.) | 0.0% (0.0-0.2) | 0.0% (0.0-0.2) | 0.1% (0.0-0.3) | 0.1% (0.0-0.4) |
| Autocorrelated (AR 0.3) | 0.3% (0.1-0.7) | 0.3% (0.1-0.7) | 0.1% (0.1-0.4) | 0.0% (0.0-0.2) |
| Small wins, big losses (skewed) | 0.8% (0.5-1.2) | 0.4% (0.2-0.7) | 0.5% (0.3-0.9) | 0.1% (0.0-0.4) |
| Martingale (3% big losses) | 0.0% (0.0-0.2) | 0.0% (0.0-0.2) | 0.0% (0.0-0.2) | 0.1% (0.1-0.4) |

Worst cell: 0.8%. "Any label" (real edge or losing, the whole 5% budget):

| Trades look like | 30 trades | 100 trades | 300 trades | 1000 trades |
|---|---|---|---|---|
| Normal | 0.2% (0.1-0.5) | 0.1% (0.1-0.4) | 0.2% (0.1-0.5) | 0.1% (0.1-0.4) |
| Heavy tails (Student t, 3 d.f.) | 0.1% (0.0-0.3) | 0.1% (0.0-0.3) | 0.1% (0.1-0.4) | 0.1% (0.1-0.4) |
| Autocorrelated (AR 0.3) | 0.7% (0.4-1.1) | 0.5% (0.3-0.9) | 0.2% (0.1-0.6) | 0.1% (0.1-0.4) |
| Small wins, big losses (skewed) | 0.8% (0.5-1.2) | 0.4% (0.2-0.8) | 0.5% (0.3-0.9) | 0.1% (0.0-0.4) |
| Martingale (3% big losses) | 0.0% (0.0-0.2) | 0.0% (0.0-0.2) | 0.0% (0.0-0.2) | 0.2% (0.1-0.5) |

### What the corrections buy

Before these corrections, skewed and martingale records broke the target on a single look, and autocorrelated records broke it when checked again and again (section 2). The same records with the corrections switched off (no widening for autocorrelation, no skew-raised bar, no five-losses rule):

| Trades look like | 30 trades | 100 trades | 300 trades |
|---|---|---|---|
| Autocorrelated (AR 0.3) | 1.1% (0.6-2.0) | 2.1% (1.4-3.2) | 1.4% (0.8-2.3) |
| Small wins, big losses (skewed) | 3.4% (2.4-4.7) | 2.0% (1.3-3.1) | 0.9% (0.5-1.7) |
| Martingale (3% big losses) | 37.1% (34.2-40.1) | 4.4% (3.3-5.9) | 1.6% (1.0-2.6) |

Honest remainder: autocorrelated and skewed records are still called real more often than normal ones, and every guarantee here is asymptotic (normal approximation, estimated spread). Records with a few dozen trades and rare, huge losses remain the hardest case; the judge says "can't tell" until at least five losses have happened.

## 2. Checking again and again

A trader who re-checks after every 10 trades, from 30 to 300 trades, on a record with no edge. How often is it EVER called a real edge? A normal confidence interval read this way fails far more often than its stated rate; the anytime-valid range is built for it.

- Normal trades: 0.5% (0.2-1.2)
- Autocorrelated trades: 1.9% (1.2-2.9) (with Newey-West alone it was 3.4% (2.4-4.7): Newey-West under-corrects at a few dozen trades, so the spread is now widened by at least a bias-corrected AR(1) factor)

## 3. The best of many versions

Try N versions of a strategy with no edge (200 trades each), keep the best one, and judge it. Without `variants_tried` the judge can only assume it was the only version; with `variants_tried=N` it raises the bar to match.

| Versions tried | Called real, not told | Called real, told `variants_tried=N` |
|---|---|---|
| 10 | 0.4% (0.1-1.4) | 0.0% (0.0-0.8) |
| 100 | 8.8% (6.6-11.6) | 0.0% (0.0-0.8) |
| 1000 | 51.2% (46.8-55.6) | 0.2% (0.0-1.1) |

## 4. Power: how often is a real edge found?

Normal trades with a true edge of 0.1, 0.2 or 0.3 standard deviations per trade (for a strategy whose trades swing about 2%, that is +0.2%, +0.4% and +0.6% per trade).

| Edge per trade | 30 trades | 100 trades | 300 trades | 1000 trades |
|---|---|---|---|---|
| 0.1 SD | 0.4% (0.2-1.0) | 1.5% (0.9-2.5) | 7.9% (6.4-9.7) | 44.8% (41.7-47.9) |
| 0.2 SD | 1.1% (0.6-2.0) | 12.1% (10.2-14.3) | 60.3% (57.2-63.3) | 99.7% (99.1-99.9) |
| 0.3 SD | 4.3% (3.2-5.7) | 42.9% (39.9-46.0) | 97.0% (95.7-97.9) | 99.5% (98.8-99.8) |

This is the price of an answer that stays valid however often you check, and of a strict error budget: small edges need many trades. The judge says so ("about N more trades would settle it") instead of guessing.

## 5. Faded, and live vs backtest

- A steady edge (0.25 SD, 400 trades) wrongly called "faded": 1.4% (0.7-2.9).
- A real fade (340 good trades, then 60 at -0.4 SD) called "faded": 62.0% (57.7-66.1).
- Live trades as good as the backtest, compared every 10 trades up to 200, ever called "worse than the backtest": 0.4% (0.1-2.2).
- Live trades 0.4 SD worse than the backtest, caught within 200 trades: 66.0% (59.9-71.6).

## 6. Same file, same answer

Every sample was judged twice in one process and once in a fresh process: identical JSON every time (yes). Random pictures (coin-flip copies, resampled histories) are seeded from the trades themselves; the result carries `input_fingerprint` so anyone can check they judged the same trades.

## 7. Speed

Median `judge()` time with every check and picture: 11 ms for 100 trades, 55 ms for 1,000 (on the machine that ran this).

## Method in one paragraph

The range is an anytime-valid 95% range (a confidence sequence; Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas 2021; Howard, Ramdas, McAuliffe & Sekhon 2021) from a one-sided normal-mixture e-value. Every version tried and every strategy shown joins one e-BH family (Wang & Ramdas 2022) at a 5% false discovery rate; versions tried but not shown count as e = 0. The spread is widened for autocorrelation (the larger of Newey-West and a bias-corrected AR(1) factor), the bar is raised for skew against the claim (Hall 1987), and "real edge" needs five losing trades. See `src/edgeorluck/evidence.py`.
