import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Group.Action.Units
import Mathlib.Algebra.Group.Int.Units
import Mathlib.Algebra.GroupWithZero.Units.Fintype
import Mathlib.Algebra.Module.Pi
import Mathlib.Algebra.Ring.Int.Units
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Finset.Max
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.List.Induction
import Mathlib.Data.Nat.Dist
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Order.Interval.Set.LinearOrder
import Mathlib.Probability.Moments.Variance
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Counting, coin flips, serial correlation and bookkeeping

edge or luck, `src/edgeorluck/judge.py`, `evidence.py` and `fills.py`. Every statement is about
real numbers (exact arithmetic), not IEEE doubles. Each model follows one Python function line by
line; `tests/test_bridge_counting.py` checks that the Python function computes what its model
computes, on many inputs, within float tolerance.

## Results (machine-checked; no unfinished proofs, no new axioms)

1. **The losing-streak odds are exact** (`_p_streak_at_most`, `_max_losing_streak`).
   `pStreakAtMost_eq`, `pStreakAtMost_eq_noRun`: the dynamic programme over the current losing
   run returns exactly `Σ p^losses (1-p)^wins` over the win/loss sequences of length `n` with no
   `L + 1` losses in a row, i.e. `P(longest losing streak ≤ L)` for `n` independent trades that
   each lose with probability `p`. The code's shortcut `longest >= n -> 1.0` is the same number
   (`dpTotal_eq_one_of_le`). The weights are a probability distribution (`sum_weight`,
   `weight_nonneg`), and the fold in `_max_losing_streak` is the longest run (`maxRun_le_iff`).
2. **The coin-flip test** (`_coin_flips`). `orbit_count_le`: for any finite group acting on the
   data and ANY statistic, at most a share `u` of each orbit has randomization p-value `≤ u`.
   `randP_valid`, `coinFlip_valid`: so if the law of the data is unchanged by every move,
   `P(p ≤ u) ≤ u` (Hoeffding 1952). `coinFlip_valid_of_indep`: this holds for sign flips of
   independent trades each symmetric about zero. `signFlip_mean`, `signFlip_second_moment`: over
   all `2^n` flips the total has mean 0 and mean square `Σ x²`, the centre and scale of the normal
   approximation the code uses above `SIM_CELLS`.
3. **The serial-correlation factor** (`serial_factor`). `pairSum_eq`, `variance_mean_ar1`: with
   AR(1) correlations `r^|i-j|`, `n Var(mean) / σ² = 1 + 2 Σ_{k<n} (1 - k/n) r^k`;
   `tendsto_ar1Ratio`: this tends to `(1 + r) / (1 - r)` for `|r| < 1`; `ar1Ratio_le_limit`: for
   `0 ≤ r < 1` the limit is never below the finite-`n` value. Safety: `serialFactorOf_mem`
   (the factor is in `[1, 10]`: it never narrows), `one_le_ar1Factor`, `ar1Factor_strictMonoOn`,
   `ar1Factor_le_nine`, `ar1Floor_mem` (the AR(1) floor is in `[1, 9]` because of the cap
   `r ≤ 0.8`).
4. **FIFO accounting** (`round_trips`). `fifo_coins`, `fifo_sold`, `fifo_cost`: coins and cost are
   conserved (bought = matched + still held + dust; sold = matched + unmatched; buy cost = basis
   of what was sold + cost still held + cost of dust). `fifo_pnl_pieces`: realized P&L =
   `Σ (sell price − lot cost) · qty` over the matched pieces, minus the sells' fee shares.
   `fifo_pnl_cash`, `fifo_flat`: realized P&L = cash in − cash out (+ what is still held).
5. **Drawdown** (`_drawdown`). `worstDrop_spec`: the running-peak formula is the largest fall
   `C_s − C_t` over all `s ≤ t`. `lastPeak_is_last_high`: "trades since the peak" counts from
   the last time the curve stood at its high.
6. **Time in the market** (`_exposure`). `sweep_eq_volume`: the sweep returns the Lebesgue
   measure of the union of the trades' time spans; `sweep_le`: so the share is in `[0, 1]`.

## What this does NOT cover (and how the code differs from the models)

* `_p_streak_at_most` is called with `p` = the observed share of losing trades: a plug-in, not a
  known probability. Trades are modelled as independent.
* The code draws 1,000 random sign patterns (seeded by the trades) instead of all `2^n`, counts
  copies that end strictly higher (`>`; the exact p-value counts `≥`, the identity included), and
  above 20 M cells uses `round(1000 · P(Z > total / √Σx²))`. The Monte Carlo error and the
  accuracy of the normal approximation are not proved. The page only DISPLAYS this count; the
  verdict comes from the e-values. The null "the trades' law is unchanged by sign flips" (e.g.
  independent, each symmetric about zero) is stronger than "the mean is zero".
* Kendall's bias correction `r + (1 + 3r)/n` and the Newey-West estimate with Bartlett weights
  are cited approximations, and the noise allowance `12/n` is set by measurement; only the safety
  facts about the returned factor are proved.
* FIFO: quote-currency fees only (a base-coin fee shrinks the lot; a third-coin fee is ignored by
  the code), and the code's float tolerances (`1e-15`, `1e-9`) are not modelled. The dust rule
  (`left <= DUST * peak`) is modelled as an arbitrary per-sell decision, so the identities hold
  for whatever it decides. That every sell lands in exactly one reported trade is checked by the
  bridge tests, not proved.
* Drawdown: the code tests "at the peak" with a `1e-12` tolerance; the model uses equality.

Toolchain: Lean 4 v4.35.0-rc4 and Mathlib v4.35.0-rc4. To check, copy this file to
`Lawcheck/EdgeOrLuck/Counting.lean` in a Lean project with that Mathlib and run
`lake env lean Lawcheck/EdgeOrLuck/Counting.lean`.
-/

noncomputable section

namespace EdgeOrLuck.Counting

/-! ## 1. The losing-streak odds: `_p_streak_at_most` and `_max_losing_streak` -/

section Streak

open Finset

/-- One pass of the loop in `_max_losing_streak`: the state is (longest run so far, current run),
and `loss` is `v < 0`. -/
def streakStep (s : ℕ × ℕ) (loss : Bool) : ℕ × ℕ :=
  if loss then (max s.1 (s.2 + 1), s.2 + 1) else (s.1, 0)

def streakState (l : List Bool) : ℕ × ℕ := l.foldl streakStep (0, 0)

/-- `_max_losing_streak`: the longest run of losses (`true`). -/
def maxRun (l : List Bool) : ℕ := (streakState l).1

/-- The current losing run at the end of the list. -/
def trailRun (l : List Bool) : ℕ := (streakState l).2

theorem streakState_concat (l : List Bool) (b : Bool) :
    streakState (l ++ [b]) = streakStep (streakState l) b := by
  simp [streakState, List.foldl_append]

theorem foldl_trail_le (l : List Bool) : ∀ s : ℕ × ℕ, s.2 ≤ s.1 →
    (l.foldl streakStep s).2 ≤ (l.foldl streakStep s).1 := by
  induction l with
  | nil => intro s hs; simpa using hs
  | cons b l ih =>
    intro s hs
    simp only [List.foldl_cons]
    apply ih
    unfold streakStep
    split_ifs <;> simp

theorem trailRun_le_maxRun (l : List Bool) : trailRun l ≤ maxRun l :=
  foldl_trail_le l (0, 0) le_rfl

theorem ofFn_snoc {n : ℕ} (v : Fin n → Bool) (b : Bool) :
    List.ofFn (Fin.snoc v b : Fin (n + 1) → Bool) = List.ofFn v ++ [b] := by
  rw [List.ofFn_succ']
  simp [List.concat_eq_append]

/-- The probability of one win/loss sequence when every trade is a loss with probability `p`,
independently: a product of `p` (loss) and `1 - p` (win). -/
def weight {n : ℕ} (p : ℝ) (v : Fin n → Bool) : ℝ := ∏ i, if v i then p else 1 - p

theorem weight_snoc {n : ℕ} (p : ℝ) (v : Fin n → Bool) (b : Bool) :
    weight p (Fin.snoc v b : Fin (n + 1) → Bool) = weight p v * (if b then p else 1 - p) := by
  unfold weight
  rw [Fin.prod_univ_castSucc]
  simp

theorem sum_snoc {n : ℕ} (F : (Fin (n + 1) → Bool) → ℝ) :
    ∑ v, F v = ∑ v : Fin n → Bool, ∑ b : Bool, F (Fin.snoc v b) := by
  rw [← (Fin.snocEquiv (fun _ => Bool)).sum_comp, Fintype.sum_prod_type, Finset.sum_comm]
  rfl

/-- One step of the state vector in `_p_streak_at_most`: all mass moves to run 0 with
probability `1 - p`; the mass at run `k` moves to `k + 1` with probability `p`; the mass that
would reach `L + 1` is dropped. -/
def dpStep (p : ℝ) (L : ℕ) (s : ℕ → ℝ) : ℕ → ℝ := fun k =>
  if k = 0 then (1 - p) * ∑ j ∈ range (L + 1), s j
  else if k ≤ L then p * s (k - 1) else 0

/-- The state vector after `n` trades, starting from all mass at run 0. -/
def dp (p : ℝ) (L : ℕ) : ℕ → ℕ → ℝ
  | 0 => fun k => if k = 0 then 1 else 0
  | n + 1 => dpStep p L (dp p L n)

/-- `float(sum(state))` at the end of the loop. -/
def dpTotal (p : ℝ) (L n : ℕ) : ℝ := ∑ k ∈ range (L + 1), dp p L n k

/-- `_p_streak_at_most(n, p, L)`, including its shortcut `if longest >= n: return 1.0`. -/
def pStreakAtMost (n : ℕ) (p : ℝ) (L : ℕ) : ℝ := if n ≤ L then 1 else dpTotal p L n

/-- The mass of the sequences whose longest run is `≤ L` and whose current run is `k`. -/
def runMass (p : ℝ) (L n k : ℕ) : ℝ :=
  ∑ v : Fin n → Bool,
    if maxRun (List.ofFn v) ≤ L ∧ trailRun (List.ofFn v) = k then weight p v else 0

theorem maxRun_concat (l : List Bool) (b : Bool) :
    maxRun (l ++ [b]) = if b then max (maxRun l) (trailRun l + 1) else maxRun l := by
  unfold maxRun trailRun
  rw [streakState_concat]
  cases b <;> simp [streakStep]

theorem trailRun_concat (l : List Bool) (b : Bool) :
    trailRun (l ++ [b]) = if b then trailRun l + 1 else 0 := by
  unfold trailRun
  rw [streakState_concat]
  cases b <;> simp [streakStep]

theorem sum_bool (f : Bool → ℝ) : ∑ b : Bool, f b = f false + f true := by
  simp [add_comm]

theorem step_zero (L m t : ℕ) (p w : ℝ) :
    (1 - p) * (if m ≤ L then w else 0) =
      (if m ≤ L ∧ (0 : ℕ) = 0 then w * (1 - p) else 0) +
        (if max m (t + 1) ≤ L ∧ t + 1 = 0 then w * p else 0) := by
  by_cases hm : m ≤ L <;> simp [hm, mul_comm]

theorem step_pos (L m t k : ℕ) (p w : ℝ) (hk : k ≤ L) (hkpos : 0 < k) :
    p * (if m ≤ L ∧ t = k - 1 then w else 0) =
      (if m ≤ L ∧ (0 : ℕ) = k then w * (1 - p) else 0) +
        (if max m (t + 1) ≤ L ∧ t + 1 = k then w * p else 0) := by
  have h0 : ¬ ((0 : ℕ) = k) := by omega
  simp only [h0, and_false, ite_false, zero_add, max_le_iff]
  by_cases h : m ≤ L ∧ t = k - 1
  · have h' : (m ≤ L ∧ t + 1 ≤ L) ∧ t + 1 = k := by omega
    rw [ite_eq_left h, ite_eq_left h', mul_comm]
  · have h' : ¬ ((m ≤ L ∧ t + 1 ≤ L) ∧ t + 1 = k) := by omega
    rw [ite_eq_right h, ite_eq_right h', mul_zero]

/-- The invariant of the dynamic programme: the mass it keeps at run `k ≤ L` after `n` trades is
the probability of the sequences with no run longer than `L` that end on a run of `k` losses. -/
theorem dp_eq_runMass (p : ℝ) (L : ℕ) : ∀ n k, k ≤ L → dp p L n k = runMass p L n k := by
  intro n
  induction n with
  | zero =>
    intro k hk
    simp only [dp, runMass]
    rw [Fintype.sum_unique]
    by_cases hk0 : k = 0
    · subst hk0; simp [maxRun, trailRun, streakState, weight]
    · simp [maxRun, trailRun, streakState, hk0, Ne.symm hk0]
  | succ n ih =>
    intro k hk
    have hsum : ∑ j ∈ range (L + 1), dp p L n j =
        ∑ v : Fin n → Bool, if maxRun (List.ofFn v) ≤ L then weight p v else 0 := by
      rw [Finset.sum_congr rfl (fun j hj => ih j (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)))]
      unfold runMass
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl (fun v _ => ?_)
      by_cases hv : maxRun (List.ofFn v) ≤ L
      · have ht : trailRun (List.ofFn v) ∈ range (L + 1) :=
          Finset.mem_range.mpr (Nat.lt_succ_of_le ((trailRun_le_maxRun _).trans hv))
        rw [Finset.sum_eq_single_of_mem _ ht]
        · simp [hv]
        · intro j _ hj
          simp [Ne.symm hj]
      · simp [hv]
    simp only [dp, dpStep, runMass]
    rw [sum_snoc]
    simp only [ofFn_snoc, maxRun_concat, trailRun_concat, weight_snoc, sum_bool,
      Bool.false_eq_true, ite_false, ite_true]
    rcases Nat.eq_zero_or_pos k with rfl | hkpos
    · rw [ite_eq_left rfl, hsum, Finset.mul_sum]
      exact Finset.sum_congr rfl (fun v _ => step_zero L _ _ p _)
    · rw [ite_eq_right (Nat.pos_iff_ne_zero.mp hkpos), ite_eq_left hk, ih (k - 1) (by omega),
        runMass, Finset.mul_sum]
      exact Finset.sum_congr rfl (fun v _ => step_pos L _ _ k p _ hk hkpos)

/-- The DP's total is the mass of the sequences whose longest losing run is at most `L`. -/
theorem dpTotal_eq_sum (p : ℝ) (L n : ℕ) :
    dpTotal p L n = ∑ v : Fin n → Bool, if maxRun (List.ofFn v) ≤ L then weight p v else 0 := by
  unfold dpTotal
  rw [Finset.sum_congr rfl (fun j hj => dp_eq_runMass p L n j
    (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)))]
  unfold runMass
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun v _ => ?_)
  by_cases hv : maxRun (List.ofFn v) ≤ L
  · have ht : trailRun (List.ofFn v) ∈ range (L + 1) :=
      Finset.mem_range.mpr (Nat.lt_succ_of_le ((trailRun_le_maxRun _).trans hv))
    rw [Finset.sum_eq_single_of_mem _ ht]
    · simp [hv]
    · intro j _ hj
      simp [Ne.symm hj]
  · simp [hv]

/-- **The sequence weights are a probability distribution**: they sum to one (for every real
`p`; they are also nonnegative when `0 ≤ p ≤ 1`, `weight_nonneg`). So a sum of weights over a set
of sequences is the probability of that set for `n` independent trades that each lose with
probability `p`. -/
theorem sum_weight (p : ℝ) (n : ℕ) : ∑ v : Fin n → Bool, weight p v = 1 := by
  unfold weight
  rw [show (∑ v : Fin n → Bool, ∏ i, if v i = true then p else 1 - p) =
      ∏ _i : Fin n, ∑ b : Bool, (if b = true then p else 1 - p) from
    (Fintype.prod_sum (fun (_ : Fin n) (b : Bool) => if b = true then p else 1 - p)).symm]
  simp

theorem weight_nonneg {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (v : Fin n → Bool) :
    0 ≤ weight p v := by
  unfold weight
  refine Finset.prod_nonneg (fun i _ => ?_)
  split_ifs <;> linarith

/-- The weight is `p ^ (number of losses) * (1 - p) ^ (number of wins)`. -/
theorem weight_eq_pow {n : ℕ} (p : ℝ) (v : Fin n → Bool) :
    weight p v = p ^ (univ.filter (fun i => v i = true)).card *
      (1 - p) ^ (univ.filter (fun i => ¬ v i = true)).card := by
  unfold weight
  rw [Finset.prod_ite, Finset.prod_const, Finset.prod_const]

theorem foldl_best_le (l : List Bool) : ∀ s : ℕ × ℕ, s.2 ≤ s.1 →
    (l.foldl streakStep s).1 ≤ s.1 + l.length := by
  induction l with
  | nil => intro s _; simp
  | cons b l ih =>
    intro s hs
    simp only [List.foldl_cons, List.length_cons]
    have hs' : (streakStep s b).2 ≤ (streakStep s b).1 := by
      unfold streakStep; split_ifs <;> simp
    have h1 : (streakStep s b).1 ≤ s.1 + 1 := by
      unfold streakStep; split_ifs <;> simp; omega
    have := ih (streakStep s b) hs'
    omega

theorem maxRun_le_length (l : List Bool) : maxRun l ≤ l.length := by
  have := foldl_best_le l (0, 0) le_rfl
  simpa [maxRun, streakState] using this

/-- **`_p_streak_at_most` is exact.** For every `n`, `p` and `L`,
`_p_streak_at_most(n, p, L) = Σ p^losses (1-p)^wins` over the win/loss sequences of length `n`
whose longest losing run (`_max_losing_streak`) is at most `L`. With `0 ≤ p ≤ 1` this is
`P(longest losing streak ≤ L)` for `n` independent trades that each lose with probability `p`
(`sum_weight`, `weight_nonneg`). Both branches of the code are covered: the shortcut
`longest >= n -> 1.0` and the dynamic programme. The code passes the observed share of losing
trades as `p` (a plug-in). Backs `judge._p_streak_at_most` (the "no losing streaks" red flag). -/
theorem pStreakAtMost_eq (n : ℕ) (p : ℝ) (L : ℕ) :
    pStreakAtMost n p L =
      ∑ v : Fin n → Bool, if maxRun (List.ofFn v) ≤ L then weight p v else 0 := by
  unfold pStreakAtMost
  split_ifs with h
  · have : ∀ v : Fin n → Bool, maxRun (List.ofFn v) ≤ L := fun v =>
      (maxRun_le_length _).trans (by simpa using h)
    simp only [this, ite_true]
    exact (sum_weight p n).symm
  · exact dpTotal_eq_sum p L n

/-- **The shortcut changes nothing**: when `n ≤ L` the dynamic programme itself returns 1, the
value of the code's shortcut `if longest >= n: return 1.0`. -/
theorem dpTotal_eq_one_of_le {n L : ℕ} (p : ℝ) (h : n ≤ L) : dpTotal p L n = 1 := by
  rw [dpTotal_eq_sum]
  have : ∀ v : Fin n → Bool, maxRun (List.ofFn v) ≤ L := fun v =>
    (maxRun_le_length _).trans (by simpa using h)
  simp only [this, ite_true]
  exact sum_weight p n

theorem maxRun_concat' (l : List Bool) (b : Bool) :
    maxRun (l ++ [b]) = max (maxRun l) (trailRun (l ++ [b])) := by
  rw [maxRun_concat, trailRun_concat]
  cases b <;> simp

/-- The current run is the longest all-loss suffix. -/
theorem replicate_suffix_iff (l : List Bool) : ∀ k : ℕ,
    List.replicate k true <:+ l ↔ k ≤ trailRun l := by
  induction l using List.reverseRecOn with
  | nil =>
    intro k
    cases k with
    | zero => simp [trailRun, streakState]
    | succ k => simp [trailRun, streakState]
  | append_singleton l b ih =>
    intro k
    rw [trailRun_concat]
    cases k with
    | zero => simp
    | succ k =>
      rw [List.replicate_succ', List.suffix_concat_iff]
      simp only [List.append_eq_nil_iff, List.cons_ne_nil, and_false, false_or,
        List.append_singleton_inj]
      constructor
      · rintro ⟨t, ⟨rfl, rfl⟩, ht⟩
        simp only [ite_true]
        have := (ih k).mp ht
        omega
      · intro hk
        cases b with
        | false => simp at hk
        | true =>
          refine ⟨List.replicate k true, ⟨rfl, rfl⟩, (ih k).mpr ?_⟩
          simp at hk
          omega

/-- `_max_losing_streak` is the longest block of consecutive losses. -/
theorem replicate_infix_iff (l : List Bool) : ∀ k : ℕ,
    List.replicate k true <:+: l ↔ k ≤ maxRun l := by
  induction l using List.reverseRecOn with
  | nil =>
    intro k
    cases k with
    | zero => simp [maxRun, streakState]
    | succ k => simp [maxRun, streakState]
  | append_singleton l b ih =>
    intro k
    rw [List.infix_concat_iff, maxRun_concat', replicate_suffix_iff, ih, le_max_iff, or_comm]

/-- **The fold in `_max_losing_streak` measures the longest losing run**: it is at most `L`
exactly when no `L + 1` trades in a row all lose. -/
theorem maxRun_le_iff (l : List Bool) (L : ℕ) :
    maxRun l ≤ L ↔ ¬ List.replicate (L + 1) true <:+: l := by
  rw [replicate_infix_iff]
  omega

/-- **`_p_streak_at_most` is exact**, stated without the fold: it is the probability that no
`L + 1` trades in a row are all losses, for `n` independent trades that each lose with
probability `p`. Backs `judge._p_streak_at_most` and `judge._max_losing_streak`. -/
theorem pStreakAtMost_eq_noRun (n : ℕ) (p : ℝ) (L : ℕ) :
    pStreakAtMost n p L =
      ∑ v : Fin n → Bool,
        if ¬ List.replicate (L + 1) true <:+: List.ofFn v then weight p v else 0 := by
  rw [pStreakAtMost_eq]
  refine Finset.sum_congr rfl (fun v _ => ?_)
  simp only [maxRun_le_iff]

end Streak

/-! ## 2. The coin-flip test: `_coin_flips` -/

section CoinFlip

open Finset MeasureTheory

/-- The heart of every randomization test: among finitely many values, those whose count of
values "at least as large" is at most `c` number at most `c` (`c ≥ 0`). -/
theorem card_rank_le {ι : Type*} (s : Finset ι) (t : ι → ℝ) (c : ℝ) (hc : 0 ≤ c) :
    ((s.filter (fun i => ((s.filter (fun j => t i ≤ t j)).card : ℝ) ≤ c)).card : ℝ) ≤ c := by
  classical
  set S := s.filter (fun i => ((s.filter (fun j => t i ≤ t j)).card : ℝ) ≤ c) with hS
  rcases S.eq_empty_or_nonempty with hE | hne
  · rw [hE]; simpa using hc
  · obtain ⟨i, hiS, hmin⟩ := S.exists_min_image t hne
    have hsub : S ⊆ s.filter (fun j => t i ≤ t j) := by
      intro j hj
      have hj' := hj
      rw [hS, Finset.mem_filter] at hj'
      exact Finset.mem_filter.mpr ⟨hj'.1, hmin j hj⟩
    have hi : ((s.filter (fun j => t i ≤ t j)).card : ℝ) ≤ c := by
      have := hiS
      rw [hS, Finset.mem_filter] at this
      exact this.2
    exact le_trans (by exact_mod_cast Finset.card_le_card hsub) hi

variable (G : Type*) {X : Type*} [Group G] [Fintype G] [MulAction G X]

/-- The exact randomization p-value of a statistic `T` at `x`: the share of the group whose move
gives a statistic at least as large as the observed one (the identity included). -/
def randP (T : X → ℝ) (x : X) : ℝ :=
  ((univ.filter (fun g : G => T x ≤ T (g • x))).card : ℝ) / Fintype.card G

variable {G}

theorem randP_smul (T : X → ℝ) (x : X) (g : G) :
    randP G T (g • x) =
      ((univ.filter (fun h : G => T (g • x) ≤ T (h • x))).card : ℝ) / Fintype.card G := by
  unfold randP
  congr 2
  rw [Finset.card_filter, Finset.card_filter]
  exact Fintype.sum_equiv (Equiv.mulRight g) _ _ (fun h => by simp [mul_smul])

/-- **The orbit-counting lemma.** For every finite group `G` acting on the data, every statistic
`T` (no condition at all), every point `x` and every `u ≥ 0`: at most a share `u` of the group
moves `x` to a point whose randomization p-value is `≤ u`, i.e.
`#{g : p(g·x) ≤ u} / |G| ≤ u`. Deterministic; for `u < 0` the left side is `0 > u`, hence the
hypothesis. Backs the coin-flip copies of `judge._coin_flips` (with `G = {±1}^n`). -/
theorem orbit_count_le (T : X → ℝ) (x : X) (u : ℝ) (hu : 0 ≤ u) :
    ((univ.filter (fun g : G => randP G T (g • x) ≤ u)).card : ℝ) / Fintype.card G ≤ u := by
  have hN : (0 : ℝ) < Fintype.card G := by exact_mod_cast Fintype.card_pos
  have key := card_rank_le (univ : Finset G) (fun g => T (g • x)) (u * Fintype.card G)
    (mul_nonneg hu hN.le)
  have hfilt : univ.filter (fun g : G => randP G T (g • x) ≤ u) =
      univ.filter (fun i : G => ((univ.filter (fun j : G => T (i • x) ≤ T (j • x))).card : ℝ) ≤
        u * Fintype.card G) := by
    refine Finset.filter_congr (fun g _ => ?_)
    rw [randP_smul, div_le_iff₀ hN]
  rw [hfilt, div_le_iff₀ hN]
  exact key

theorem randP_eq_sum (T : X → ℝ) (x : X) :
    randP G T x = (∑ h : G, (if T x ≤ T (h • x) then (1 : ℝ) else 0)) / Fintype.card G := by
  unfold randP
  rw [Finset.sum_boole]

theorem measurable_randP [MeasurableSpace X] (T : X → ℝ) (hT : Measurable T)
    (hact : ∀ g : G, Measurable (fun x : X => g • x)) : Measurable (randP G T) := by
  have : randP G T = fun x => (∑ h : G, (if T x ≤ T (h • x) then (1 : ℝ) else 0)) /
      Fintype.card G := funext (randP_eq_sum T)
  rw [this]
  refine Measurable.div_const (Finset.measurable_sum _ (fun h _ => ?_)) _
  exact Measurable.ite (measurableSet_le hT (hT.comp (hact h))) measurable_const measurable_const

/-- **The randomization test is valid** (Hoeffding 1952). If the law `μ` of the data is
unchanged by every group move (`μ.map (g • ·) = μ`), then `P(p ≤ u) ≤ u` for every `u ≥ 0`.
Assumptions: `μ` a probability measure, `T` measurable, each move measurable. -/
theorem randP_valid [MeasurableSpace X] (μ : Measure X) [IsProbabilityMeasure μ]
    (T : X → ℝ) (hT : Measurable T) (hact : ∀ g : G, Measurable (fun x : X => g • x))
    (hinv : ∀ g : G, μ.map (fun x => g • x) = μ) (u : ℝ) (hu : 0 ≤ u) :
    μ {x | randP G T x ≤ u} ≤ ENNReal.ofReal u := by
  set A := {x | randP G T x ≤ u} with hA
  have hAm : MeasurableSet A := measurableSet_le (measurable_randP T hT hact) measurable_const
  set N := Fintype.card G with hNdef
  have hNpos : 0 < N := Fintype.card_pos
  have hmove : ∀ g : G, μ ((fun x => g • x) ⁻¹' A) = μ A := by
    intro g
    rw [← Measure.map_apply (hact g) hAm, hinv g]
  have hAg : ∀ g : G, MeasurableSet ((fun x => g • x) ⁻¹' A) := fun g => (hact g) hAm
  have hsum : (N : ENNReal) * μ A = ∫⁻ x, ∑ g : G, ((fun x => g • x) ⁻¹' A).indicator 1 x ∂μ := by
    rw [lintegral_finsetSum (s := univ) (f := fun g x => ((fun x => g • x) ⁻¹' A).indicator 1 x)
      (fun g _ => (measurable_const.indicator (hAg g)))]
    simp_rw [lintegral_indicator_one (hAg _), hmove]
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hpt : ∀ x, ∑ g : G, ((fun x => g • x) ⁻¹' A).indicator 1 x ≤
      ENNReal.ofReal (u * N) := by
    intro x
    classical
    have hcount := orbit_count_le (G := G) T x u hu
    have hNr : (0 : ℝ) < N := by exact_mod_cast hNpos
    rw [div_le_iff₀ hNr] at hcount
    simp only [Set.indicator_apply, Set.mem_preimage, hA, Set.mem_ofPred_eq, Pi.one_apply]
    rw [Finset.sum_boole, ← ENNReal.ofReal_natCast]
    exact ENNReal.ofReal_le_ofReal hcount
  have hle : (N : ENNReal) * μ A ≤ ENNReal.ofReal (u * N) := by
    rw [hsum]
    calc ∫⁻ x, ∑ g : G, ((fun x => g • x) ⁻¹' A).indicator 1 x ∂μ
        ≤ ∫⁻ _x, ENNReal.ofReal (u * N) ∂μ := lintegral_mono hpt
      _ = ENNReal.ofReal (u * N) := by simp
  rw [ENNReal.ofReal_mul hu, ENNReal.ofReal_natCast, mul_comm (ENNReal.ofReal u)] at hle
  have h0 : (N : ENNReal) ≠ 0 := by exact_mod_cast hNpos.ne'
  have hinf : (N : ENNReal) ≠ ⊤ := ENNReal.natCast_ne_top N
  exact (ENNReal.mul_le_mul_iff_right h0 hinf).mp hle

/-! ### The group `{±1}^n` of coin flips -/

/-- A sign pattern `g : Fin n → ℤˣ` flips the sign of the trades where `g i = -1`. -/
theorem signFlip_apply {n : ℕ} (g : Fin n → ℤˣ) (x : Fin n → ℝ) (i : Fin n) :
    (g • x) i = ((g i : ℤ) : ℝ) * x i := by
  rw [Pi.smul_apply', Units.smul_def, zsmul_eq_mul]

theorem card_signFlip (n : ℕ) : Fintype.card (Fin n → ℤˣ) = 2 ^ n := by
  simp

theorem measurable_signFlip {n : ℕ} (g : Fin n → ℤˣ) :
    Measurable (fun x : Fin n → ℝ => g • x) := by
  have : (fun x : Fin n → ℝ => g • x) = fun x i => ((g i : ℤ) : ℝ) * x i := by
    funext x i; exact signFlip_apply g x i
  rw [this]
  exact Measurable.of_eval (fun i => (measurable_pi_apply i).const_mul _)

/-- **The coin-flip test is valid**: with `G = {±1}^n` acting coordinatewise, if the law of the
trades is unchanged when any of them changes sign, then `P(p ≤ u) ≤ u` for the exact coin-flip
p-value of any measurable statistic (the code's statistic is the total, `measurable_total`).
Backs `judge._coin_flips`, which estimates the same count from 1,000 random flips (a Monte Carlo
version, error not proved), counts strictly higher copies, and above 20 M cells uses a normal
approximation. The page only displays the count; the verdict comes from the e-values. -/
theorem coinFlip_valid {n : ℕ} (μ : Measure (Fin n → ℝ)) [IsProbabilityMeasure μ]
    (T : (Fin n → ℝ) → ℝ) (hT : Measurable T)
    (hsym : ∀ g : Fin n → ℤˣ, μ.map (fun x => g • x) = μ) (u : ℝ) (hu : 0 ≤ u) :
    μ {x | randP (Fin n → ℤˣ) T x ≤ u} ≤ ENNReal.ofReal u :=
  randP_valid μ T hT measurable_signFlip hsym u hu

/-- Independent trades, each with a law symmetric about zero, have a law unchanged by sign
flips. -/
theorem pi_symm_invariant {n : ℕ} (ν : Fin n → Measure ℝ) [∀ i, IsProbabilityMeasure (ν i)]
    (hsymm : ∀ i, (ν i).map (fun t => -t) = ν i) (g : Fin n → ℤˣ) :
    (Measure.pi ν).map (fun x => g • x) = Measure.pi ν := by
  have hfun : (fun x : Fin n → ℝ => g • x) = fun x i => ((g i : ℤ) : ℝ) * x i := by
    funext x i; exact signFlip_apply g x i
  have hmap : ∀ i, (ν i).map (fun t => ((g i : ℤ) : ℝ) * t) = ν i := by
    intro i
    rcases Int.units_eq_one_or (g i) with h | h
    · rw [h]; simp
    · rw [h]
      have : (fun t : ℝ => (((-1 : ℤˣ) : ℤ) : ℝ) * t) = fun t => -t := by funext t; simp
      rw [this, hsymm i]
  have : ∀ i, SigmaFinite ((ν i).map (fun t => ((g i : ℤ) : ℝ) * t)) := fun i => by
    rw [hmap i]; infer_instance
  rw [hfun, Measure.pi_map_pi (fun i => (measurable_const_mul _).aemeasurable)]
  simp_rw [hmap]

/-- **The coin-flip test is valid for independent, symmetric trades**: if the trades are
independent and each one's law is symmetric about zero (`ν i` maps to itself under `t ↦ -t`),
then `P(p ≤ u) ≤ u`. This null ("symmetric trades") is stronger than "the mean is zero". -/
theorem coinFlip_valid_of_indep {n : ℕ} (ν : Fin n → Measure ℝ) [∀ i, IsProbabilityMeasure (ν i)]
    (hsymm : ∀ i, (ν i).map (fun t => -t) = ν i)
    (T : (Fin n → ℝ) → ℝ) (hT : Measurable T) (u : ℝ) (hu : 0 ≤ u) :
    Measure.pi ν {x | randP (Fin n → ℤˣ) T x ≤ u} ≤ ENNReal.ofReal u :=
  coinFlip_valid (Measure.pi ν) T hT (pi_symm_invariant ν hsymm) u hu

/-- The judge's statistic, the total of the trades, is measurable. -/
theorem measurable_total (n : ℕ) : Measurable (fun x : Fin n → ℝ => ∑ i, x i) :=
  Finset.measurable_sum _ (fun i _ => measurable_pi_apply i)

theorem unit_sq (u : ℤˣ) : ((u : ℤ) : ℝ) * ((u : ℤ) : ℝ) = 1 := by
  rcases Int.units_eq_one_or u with rfl | rfl <;> simp

/-- Over all sign patterns, each sign averages to zero. -/
theorem sum_sign (n : ℕ) (i : Fin n) : ∑ g : Fin n → ℤˣ, ((g i : ℤ) : ℝ) = 0 := by
  have h := Fintype.sum_equiv (Equiv.mulLeft (fun _ : Fin n => (-1 : ℤˣ)))
    (fun g : Fin n → ℤˣ => ((g i : ℤ) : ℝ)) (fun g => -((g i : ℤ) : ℝ)) (fun g => by simp)
  rw [Finset.sum_neg_distrib] at h
  linarith

/-- Two different signs are uncorrelated over all sign patterns. -/
theorem sum_sign_mul {n : ℕ} {i j : Fin n} (hij : i ≠ j) :
    ∑ g : Fin n → ℤˣ, ((g i : ℤ) : ℝ) * ((g j : ℤ) : ℝ) = 0 := by
  classical
  have h := Fintype.sum_equiv (Equiv.mulLeft (Pi.mulSingle i (-1 : ℤˣ)))
    (fun g : Fin n → ℤˣ => ((g i : ℤ) : ℝ) * ((g j : ℤ) : ℝ))
    (fun g => -(((g i : ℤ) : ℝ) * ((g j : ℤ) : ℝ))) (fun g => by
      simp [Pi.mulSingle_eq_of_ne hij.symm])
  rw [Finset.sum_neg_distrib] at h
  linarith

/-- **The coin-flipped total has mean zero**: summed over all `2^n` sign patterns, the total of
the flipped trades is `0`. The centre of the normal approximation in `judge._coin_flips`. -/
theorem signFlip_mean (n : ℕ) (x : Fin n → ℝ) :
    ∑ g : Fin n → ℤˣ, ∑ i, (g • x) i = 0 := by
  simp_rw [signFlip_apply]
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero (fun i _ => ?_)
  rw [← Finset.sum_mul, sum_sign, zero_mul]

/-- **The coin-flipped total has variance `Σ x²`**: averaged over all `2^n` sign patterns, its
square is `Σ x_i²`. So `z = total / √(Σ x²)`, which `judge._coin_flips` feeds to the normal tail
above `SIM_CELLS`, is the observed total standardized by the exact null mean and spread. -/
theorem signFlip_second_moment (n : ℕ) (x : Fin n → ℝ) :
    ∑ g : Fin n → ℤˣ, (∑ i, (g • x) i) ^ 2 = 2 ^ n * ∑ i, x i ^ 2 := by
  classical
  simp_rw [signFlip_apply, sq, Finset.sum_mul_sum]
  rw [Finset.sum_comm]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Finset.sum_comm]
  rw [Finset.sum_eq_single i]
  · have : ∀ g : Fin n → ℤˣ, ((g i : ℤ) : ℝ) * x i * (((g i : ℤ) : ℝ) * x i) = x i * x i := by
      intro g
      have := unit_sq (g i)
      calc ((g i : ℤ) : ℝ) * x i * (((g i : ℤ) : ℝ) * x i)
          = (((g i : ℤ) : ℝ) * ((g i : ℤ) : ℝ)) * (x i * x i) := by ring
        _ = x i * x i := by rw [this, one_mul]
    simp_rw [this]
    rw [Finset.sum_const, Finset.card_univ]
    simp
  · intro j _ hji
    have h := sum_sign_mul (n := n) (Ne.symm hji)
    calc ∑ g : Fin n → ℤˣ, ((g i : ℤ) : ℝ) * x i * (((g j : ℤ) : ℝ) * x j)
        = (∑ g : Fin n → ℤˣ, ((g i : ℤ) : ℝ) * ((g j : ℤ) : ℝ)) * (x i * x j) := by
          rw [Finset.sum_mul]
          exact Finset.sum_congr rfl (fun g _ => by ring)
      _ = 0 := by rw [h, zero_mul]
  · intro h; exact absurd (Finset.mem_univ i) h

end CoinFlip

/-! ## 3. The serial-correlation factor: `serial_factor` -/

section Serial

open Finset Filter Topology

/-- `n · Var(mean) / σ²` when the trades' correlations are `r ^ |i - j|` (AR(1)):
`1 + 2 Σ_{k=1}^{n-1} (1 - k/n) r^k`. -/
def ar1Ratio (r : ℝ) (n : ℕ) : ℝ :=
  1 + 2 * ∑ k ∈ Ico 1 n, (1 - (k : ℝ) / n) * r ^ k

/-- The sum of all pairwise correlations `r ^ |i - j|` over `n` trades. -/
def pairSum (r : ℝ) (n : ℕ) : ℝ :=
  ∑ i ∈ range n, ∑ j ∈ range n, r ^ Nat.dist i j

theorem pairSum_succ (r : ℝ) (n : ℕ) :
    pairSum r (n + 1) = pairSum r n + 2 * ∑ k ∈ Ico 1 (n + 1), r ^ k + 1 := by
  unfold pairSum
  rw [Finset.sum_range_succ]
  simp_rw [Finset.sum_range_succ]
  rw [Finset.sum_add_distrib]
  have hcol : ∑ i ∈ range n, r ^ Nat.dist i n = ∑ k ∈ Ico 1 (n + 1), r ^ k := by
    rw [← Finset.sum_range_reflect, Finset.sum_Ico_eq_sum_range]
    refine Finset.sum_congr (by simp) (fun i hi => ?_)
    have hi' : i < n := Finset.mem_range.mp hi
    congr 1
    rw [Nat.dist_comm, Nat.dist.eq_def]
    omega
  have hrow : ∑ j ∈ range n, r ^ Nat.dist n j = ∑ k ∈ Ico 1 (n + 1), r ^ k := by
    rw [← hcol]
    exact Finset.sum_congr rfl (fun j _ => by rw [Nat.dist_comm])
  rw [hcol, hrow, Nat.dist_self, pow_zero]
  ring

/-- **The finite identity**: the pairwise correlations `r^|i-j|` of `n` trades add up to
`n + 2 Σ_{k=1}^{n-1} (n - k) r^k` (there are `2 (n - k)` pairs at distance `k ≥ 1`). For every
real `r`. Backs the AR(1) part of `evidence.serial_factor`. -/
theorem pairSum_eq (r : ℝ) (n : ℕ) :
    pairSum r n = n + 2 * ∑ k ∈ Ico 1 n, ((n : ℝ) - k) * r ^ k := by
  induction n with
  | zero => simp [pairSum]
  | succ n ih =>
    rw [pairSum_succ, ih]
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ n), Finset.sum_Ico_succ_top (by omega : 1 ≤ n)]
      have : ∑ k ∈ Ico 1 n, (((n + 1 : ℕ) : ℝ) - k) * r ^ k =
          ∑ k ∈ Ico 1 n, ((n : ℝ) - k) * r ^ k + ∑ k ∈ Ico 1 n, r ^ k := by
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl (fun k _ => ?_)
        push_cast; ring
      rw [this]
      push_cast
      ring

theorem ar1Ratio_eq_pairSum (r : ℝ) {n : ℕ} (hn : 0 < n) :
    ar1Ratio r n = pairSum r n / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [pairSum_eq, ar1Ratio, add_div, div_self hn', mul_div_assoc, Finset.sum_div]
  congr 2
  refine Finset.sum_congr rfl (fun k _ => ?_)
  field_simp

theorem pairSum_eq_mul (r : ℝ) {n : ℕ} (hn : 0 < n) : pairSum r n = n * ar1Ratio r n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [ar1Ratio_eq_pairSum r hn]
  field_simp

open MeasureTheory ProbabilityTheory in
/-- **The variance of the average under AR(1) correlations.** If `n ≥ 1` trades (square
integrable, on a finite measure space) have covariances `cov(X_i, X_j) = σ² r^|i-j|`, then
`n · Var(average) = σ² · (1 + 2 Σ_{k=1}^{n-1} (1 - k/n) r^k)`. Backs the AR(1) factor of
`evidence.serial_factor`. -/
theorem variance_mean_ar1 {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} (hn : 0 < n) (X : Fin n → Ω → ℝ) (hX : ∀ i, MemLp (X i) 2 μ) (σ2 r : ℝ)
    (hcov : ∀ i j, cov[X i, X j; μ] = σ2 * r ^ Nat.dist (i : ℕ) (j : ℕ)) :
    n * Var[fun ω => (∑ i, X i ω) / n; μ] = σ2 * ar1Ratio r n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : Var[fun ω => ∑ i, X i ω; μ] = σ2 * pairSum r n := by
    rw [variance_fun_sum hX]
    simp_rw [hcov]
    rw [pairSum, Finset.mul_sum]
    rw [← Fin.sum_univ_eq_sum_range (fun i => σ2 * ∑ j ∈ range n, r ^ Nat.dist i j)]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Finset.mul_sum, ← Fin.sum_univ_eq_sum_range (fun j => σ2 * r ^ Nat.dist (i : ℕ) j)]
  have hdiv : (fun ω => (∑ i, X i ω) / (n : ℝ)) = fun ω => (∑ i, X i ω) * (n : ℝ)⁻¹ := by
    funext ω; rw [div_eq_mul_inv]
  rw [hdiv, variance_mul_const, hsum, pairSum_eq_mul r hn]
  field_simp

/-- **The limit**: for `|r| < 1`, `1 + 2 Σ_{k=1}^{n-1} (1 - k/n) r^k → (1 + r) / (1 - r)` as
`n → ∞`: the AR(1) factor `(1 + r) / (1 - r)` in `evidence.serial_factor` is the long-run
variance ratio. -/
theorem tendsto_ar1Ratio {r : ℝ} (hr : |r| < 1) :
    Tendsto (ar1Ratio r) atTop (𝓝 ((1 + r) / (1 - r))) := by
  have h1r : (1 : ℝ) - r ≠ 0 := by
    have := (abs_lt.mp hr).2; linarith
  have hgeo : Tendsto (fun n => ∑ k ∈ range n, r ^ k) atTop (𝓝 (1 - r)⁻¹) :=
    (hasSum_geometric_of_abs_lt_one hr).tendsto_sum_nat
  have hnorm : ‖r‖ < 1 := by rwa [Real.norm_eq_abs]
  have hlin : Tendsto (fun n => ∑ k ∈ range n, (k : ℝ) * r ^ k) atTop (𝓝 (r / (1 - r) ^ 2)) :=
    (hasSum_coe_mul_geometric_of_norm_lt_one hnorm).tendsto_sum_nat
  have hdiv : Tendsto (fun n : ℕ => (∑ k ∈ range n, (k : ℝ) * r ^ k) / n) atTop (𝓝 0) :=
    hlin.div_atTop tendsto_natCast_atTop_atTop
  have hlim : Tendsto (fun n : ℕ => 2 * ∑ k ∈ range n, r ^ k - 1 -
      2 * ((∑ k ∈ range n, (k : ℝ) * r ^ k) / n)) atTop (𝓝 (2 * (1 - r)⁻¹ - 1 - 2 * 0)) :=
    ((hgeo.const_mul 2).sub_const 1).sub (hdiv.const_mul 2)
  have hval : 2 * (1 - r)⁻¹ - 1 - 2 * 0 = (1 + r) / (1 - r) := by
    field_simp; ring
  rw [hval] at hlim
  refine hlim.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with n hn
  have e1 : ∑ k ∈ range n, r ^ k = 1 + ∑ k ∈ Ico 1 n, r ^ k := by
    rw [Finset.sum_range_eq_add_Ico _ hn, pow_zero]
  have e2 : ∑ k ∈ range n, (k : ℝ) * r ^ k = ∑ k ∈ Ico 1 n, (k : ℝ) * r ^ k := by
    rw [Finset.sum_range_eq_add_Ico _ hn]; simp
  have hsplit : ∑ k ∈ Ico 1 n, (1 - (k : ℝ) / n) * r ^ k =
      ∑ k ∈ Ico 1 n, r ^ k - (∑ k ∈ Ico 1 n, (k : ℝ) * r ^ k) / n := by
    rw [Finset.sum_div, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun k _ => by ring)
  rw [e1, e2, ar1Ratio, hsplit]
  ring

/-- **The limit never under-widens**: for `0 ≤ r < 1` and every `n`, the exact finite-`n` ratio
is at most `(1 + r) / (1 - r)`; using the limit (as `serial_factor` does) is conservative for
the same `r`. -/
theorem ar1Ratio_le_limit {r : ℝ} (h0 : 0 ≤ r) (h1 : r < 1) (n : ℕ) :
    ar1Ratio r n ≤ (1 + r) / (1 - r) := by
  have h1r : (0 : ℝ) < 1 - r := by linarith
  have hgeo := hasSum_geometric_of_lt_one h0 h1
  have hpart : ∑ k ∈ range n, r ^ k ≤ (1 - r)⁻¹ :=
    sum_le_hasSum (range n) (fun i _ => pow_nonneg h0 i) hgeo
  have hterm : ∑ k ∈ Ico 1 n, (1 - (k : ℝ) / n) * r ^ k ≤ ∑ k ∈ Ico 1 n, r ^ k := by
    refine Finset.sum_le_sum (fun k _ => ?_)
    have hk : 0 ≤ (k : ℝ) / n := by positivity
    have := pow_nonneg h0 k
    nlinarith
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [ar1Ratio, Finset.Ico_eq_empty_of_le (by norm_num : (1 : ℕ) ≥ 0),
      Finset.sum_empty, mul_zero, add_zero]
    rw [le_div_iff₀ h1r]; linarith
  · rw [Finset.sum_range_eq_add_Ico _ hn, pow_zero] at hpart
    have hval : 1 + 2 * ((1 - r)⁻¹ - 1) = (1 + r) / (1 - r) := by
      field_simp; ring
    unfold ar1Ratio
    linarith

/-- `(1 + r) / (1 - r)`, the AR(1) variance factor. -/
def ar1Factor (r : ℝ) : ℝ := (1 + r) / (1 - r)

/-- **The AR(1) factor never narrows**: `(1 + r) / (1 - r) ≥ 1` for `0 ≤ r < 1`. -/
theorem one_le_ar1Factor {r : ℝ} (h0 : 0 ≤ r) (h1 : r < 1) : 1 ≤ ar1Factor r := by
  unfold ar1Factor
  rw [le_div_iff₀ (by linarith)]
  linarith

/-- **The AR(1) factor grows with the correlation**: strictly increasing on `r < 1`. -/
theorem ar1Factor_strictMonoOn : StrictMonoOn ar1Factor (Set.Iio 1) := by
  intro a ha b hb hab
  simp only [Set.mem_Iio] at ha hb
  unfold ar1Factor
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **The cap `r ≤ 0.8` keeps the AR(1) factor at most `1.8 / 0.2 = 9`.** -/
theorem ar1Factor_le_nine {r : ℝ} (h : r ≤ 0.8) : ar1Factor r ≤ 9 := by
  unfold ar1Factor
  rw [div_le_iff₀ (by norm_num at h ⊢; linarith)]
  norm_num at h ⊢
  linarith

/-- The lag-one correlation after Kendall's correction, the allowance `12/n` for its own noise
(`SERIAL_NOISE`) and the cap at 0.8: `min(0.8, r + (1 + 3r + 12) / n)`. -/
def kendallCapped (r : ℝ) (n : ℕ) : ℝ := min 0.8 (r + (1 + 3 * r + 12) / n)

/-- The AR(1) floor exactly as `serial_factor` computes it: `(1 + r)/(1 - r)` when `r > 0`,
else 1. -/
def ar1Floor (r : ℝ) : ℝ := if 0 < r then ar1Factor r else 1

/-- `serial_factor`'s last line: `min(10, max(1, newey_west, ar1))`. -/
def serialFactorOf (nw ar1 : ℝ) : ℝ := min 10 (max 1 (max nw ar1))

/-- **The AR(1) floor is between 1 and 9**, whatever the data: for every estimate `r` of the
lag-one correlation and every `n`, `ar1` in `evidence.serial_factor` is in `[1, 9]`. -/
theorem ar1Floor_mem (r : ℝ) (n : ℕ) :
    1 ≤ ar1Floor (kendallCapped r n) ∧ ar1Floor (kendallCapped r n) ≤ 9 := by
  have hcap : kendallCapped r n ≤ 0.8 := min_le_left _ _
  unfold ar1Floor
  split_ifs with h
  · refine ⟨one_le_ar1Factor h.le (by norm_num at hcap ⊢; linarith), ar1Factor_le_nine hcap⟩
  · norm_num

/-- **`serial_factor` never narrows the spread and widens the variance at most tenfold**: for
any Newey-West value and any AR(1) value, `min(10, max(1, NW, AR1)) ∈ [1, 10]`, so the spread
`sd · √factor` is never below the sample SD. (The early returns of `serial_factor`, `1.0`, are
in `[1, 10]` too.) Backs `evidence.serial_factor` and `evidence.record_of`. -/
theorem serialFactorOf_mem (nw ar1 : ℝ) :
    1 ≤ serialFactorOf nw ar1 ∧ serialFactorOf nw ar1 ≤ 10 :=
  ⟨le_min (by norm_num) (le_max_left _ _), min_le_left _ _⟩

/-- Kendall's correction `r + (1 + 3r)/n` only raises the estimate when `r ≥ -1/3`. -/
theorem kendall_ge {r : ℝ} (hr : -1 / 3 ≤ r) (n : ℕ) : r ≤ r + (1 + 3 * r) / n := by
  have : 0 ≤ (1 + 3 * r) / n := div_nonneg (by linarith) (Nat.cast_nonneg n)
  linarith

end Serial

/-! ## 4. FIFO accounting: `round_trips` -/

section Fifo

/-- A lot of coins bought: how many, and what each cost (quote currency, buy fee included). -/
structure Lot where
  qty : ℝ
  cost : ℝ

/-- Coins held in a list of lots. -/
def held (ls : List Lot) : ℝ := (ls.map Lot.qty).sum

/-- What a list of lots cost. -/
def value (ls : List Lot) : ℝ := (ls.map (fun l => l.qty * l.cost)).sum

@[simp] theorem held_nil : held [] = 0 := rfl
@[simp] theorem value_nil : value [] = 0 := rfl
@[simp] theorem held_cons (l : Lot) (ls : List Lot) : held (l :: ls) = l.qty + held ls := by
  simp [held]
@[simp] theorem value_cons (l : Lot) (ls : List Lot) :
    value (l :: ls) = l.qty * l.cost + value ls := by
  simp [value]
@[simp] theorem held_append (ls ms : List Lot) : held (ls ++ ms) = held ls + held ms := by
  simp [held]
@[simp] theorem value_append (ls ms : List Lot) : value (ls ++ ms) = value ls + value ms := by
  simp [value]

/-- Take `q` coins from the lots, oldest first (the `while remaining > 0 and lots` loop of
`round_trips`): (the lots left, the pieces taken). A lot is split when only part is needed. -/
def takeFifo : List Lot → ℝ → List Lot × List Lot
  | [], _ => ([], [])
  | l :: ls, q =>
    if q ≤ 0 then (l :: ls, [])
    else if q < l.qty then (⟨l.qty - q, l.cost⟩ :: ls, [⟨q, l.cost⟩])
    else ((takeFifo ls (q - l.qty)).1, l :: (takeFifo ls (q - l.qty)).2)

/-- Taking never creates or destroys coins. -/
theorem takeFifo_held (ls : List Lot) (q : ℝ) :
    held (takeFifo ls q).2 + held (takeFifo ls q).1 = held ls := by
  induction ls generalizing q with
  | nil => simp [takeFifo]
  | cons l ls ih =>
    unfold takeFifo
    split_ifs with h1 h2
    · simp
    · simp; ring
    · simp only [held_cons]; linarith [ih (q - l.qty)]

/-- Taking never creates or destroys cost. -/
theorem takeFifo_value (ls : List Lot) (q : ℝ) :
    value (takeFifo ls q).2 + value (takeFifo ls q).1 = value ls := by
  induction ls generalizing q with
  | nil => simp [takeFifo]
  | cons l ls ih =>
    unfold takeFifo
    split_ifs with h1 h2
    · simp
    · simp; ring
    · simp only [value_cons]; linarith [ih (q - l.qty)]

theorem held_nonneg {ls : List Lot} (h : ∀ l ∈ ls, 0 ≤ l.qty) : 0 ≤ held ls := by
  induction ls with
  | nil => simp
  | cons l ls ih =>
    simp only [held_cons]
    have := h l (by simp)
    have := ih (fun m hm => h m (by simp [hm]))
    linarith

theorem takeFifo_nonneg (ls : List Lot) (q : ℝ) (h : ∀ l ∈ ls, 0 ≤ l.qty) :
    ∀ l ∈ (takeFifo ls q).1, 0 ≤ l.qty := by
  induction ls generalizing q with
  | nil => simp [takeFifo]
  | cons l ls ih =>
    unfold takeFifo
    split_ifs with h1 h2
    · exact h
    · intro m hm
      simp only [List.mem_cons] at hm
      rcases hm with rfl | hm
      · simp only; linarith
      · exact h m (by simp [hm])
    · exact ih (q - l.qty) (fun m hm => h m (by simp [hm]))

/-- **FIFO takes exactly `min q held` coins** (lots with nonnegative quantities, `q ≥ 0`). -/
theorem takeFifo_held_taken (ls : List Lot) (q : ℝ) (hq : 0 ≤ q) (h : ∀ l ∈ ls, 0 ≤ l.qty) :
    held (takeFifo ls q).2 = min q (held ls) := by
  induction ls generalizing q with
  | nil => simp [takeFifo, hq]
  | cons l ls ih =>
    have hl := h l (by simp)
    have hls : 0 ≤ held ls := held_nonneg (fun m hm => h m (by simp [hm]))
    unfold takeFifo
    split_ifs with h1 h2
    · have : q = 0 := le_antisymm h1 hq
      subst this
      simp; linarith
    · simp only [held_cons, held_nil, add_zero]
      rw [min_eq_left (by linarith)]
    · have h1' : 0 < q := lt_of_not_ge h1
      have h2' : l.qty ≤ q := le_of_not_gt h2
      simp only [held_cons]
      rw [ih (q - l.qty) (by linarith) (fun m hm => h m (by simp [hm]))]
      rw [← min_add_add_left, add_sub_cancel]

/-- The realized P&L of pieces sold at price `p`, `Σ (p - cost) · qty`, is
`p · (coins) - (their cost)`. -/
theorem pieces_pnl (ps : List Lot) (p : ℝ) :
    (ps.map (fun l => (p - l.cost) * l.qty)).sum = p * held ps - value ps := by
  induction ps with
  | nil => simp
  | cons l ps ih => simp only [List.map_cons, List.sum_cons, ih, held_cons, value_cons]; ring

/-- One exchange fill of one pair: quantity, price and fee (quote currency). A sell carries the
bookkeeping decision whether what is left afterwards is discarded as dust (`round_trips` discards
it when it is under `DUST` of the peak holding; any rule is allowed here). -/
inductive Fill
  | buy (q p fee : ℝ)
  | sell (q p fee : ℝ) (dropRest : Bool)

/-- The book kept while reading fills. -/
structure Book where
  lots : List Lot
  matched : ℝ
  unmatched : ℝ
  unmatchedCash : ℝ
  basis : ℝ
  proceeds : ℝ
  piecePnl : ℝ
  dust : ℝ
  dustCost : ℝ

def Book.empty : Book := ⟨[], 0, 0, 0, 0, 0, 0, 0, 0⟩

/-- One fill, as `round_trips` books it (quote-currency fees). A buy appends a lot at its unit
cost `(q p + fee) / q`. A sell takes `sold = min(q, held)` coins oldest first, gets
`sold · p - fee · sold / q`, and its realized P&L is that minus the cost of the coins taken. -/
def Book.step (b : Book) : Fill → Book
  | .buy q p fee => { b with lots := b.lots ++ [⟨q, (q * p + fee) / q⟩] }
  | .sell q p fee drop =>
    let sold := min q (held b.lots)
    let t := takeFifo b.lots sold
    { lots := if drop then [] else t.1
      matched := b.matched + held t.2
      unmatched := b.unmatched + (q - sold)
      unmatchedCash := b.unmatchedCash + (p - fee / q) * (q - sold)
      basis := b.basis + value t.2
      proceeds := b.proceeds + (sold * p - fee * (sold / q))
      piecePnl := b.piecePnl + ((t.2.map (fun l => (p - l.cost) * l.qty)).sum - fee * (sold / q))
      dust := b.dust + (if drop then held t.1 else 0)
      dustCost := b.dustCost + (if drop then value t.1 else 0) }

/-- The book after reading all fills of one pair, in time order. -/
def run (fs : List Fill) : Book := fs.foldl Book.step Book.empty

/-- Realized P&L: what the matched coins sold for, net of their share of the sell fees, minus
what they cost (`Σ` of the `pnl` of the trades `round_trips` reports). -/
def Book.realized (b : Book) : ℝ := b.proceeds - b.basis

def Fill.bought : Fill → ℝ
  | .buy q _ _ => q
  | .sell .. => 0

def Fill.soldQty : Fill → ℝ
  | .buy .. => 0
  | .sell q _ _ _ => q

/-- Cash out for a buy, fee included. -/
def Fill.buyCost : Fill → ℝ
  | .buy q p fee => q * p + fee
  | .sell .. => 0

/-- Cash in for a sell, net of its fee. -/
def Fill.sellCash : Fill → ℝ
  | .buy .. => 0
  | .sell q p fee _ => q * p - fee

/-- A fill with a positive quantity (`round_trips` skips the others). -/
def Fill.Valid : Fill → Prop
  | .buy q _ _ => 0 < q
  | .sell q _ _ _ => 0 < q

def Book.Inv (b : Book) : Prop :=
  (∀ l ∈ b.lots, 0 ≤ l.qty) ∧ 0 ≤ b.unmatched ∧ (b.unmatched = 0 → b.unmatchedCash = 0)

theorem sold_eq (b : Book) (q : ℝ) (hq : 0 < q) (hb : ∀ l ∈ b.lots, 0 ≤ l.qty) :
    held (takeFifo b.lots (min q (held b.lots))).2 = min q (held b.lots) := by
  have hheld : 0 ≤ held b.lots := held_nonneg hb
  have h := takeFifo_held_taken b.lots (min q (held b.lots)) (le_min hq.le hheld) hb
  rwa [min_eq_left (min_le_right _ _)] at h

theorem step_inv (b : Book) (f : Fill) (hf : f.Valid) (hb : b.Inv) : (b.step f).Inv := by
  obtain ⟨hl, hu, hc⟩ := hb
  cases f with
  | buy q p fee =>
    simp only [Fill.Valid] at hf
    refine ⟨?_, hu, hc⟩
    intro l hl'
    simp only [Book.step, List.mem_append, List.mem_singleton] at hl'
    rcases hl' with hl' | rfl
    · exact hl l hl'
    · exact hf.le
  | sell q p fee drop =>
    simp only [Fill.Valid] at hf
    have hu' : 0 ≤ q - min q (held b.lots) := sub_nonneg.mpr (min_le_left _ _)
    refine ⟨?_, ?_, ?_⟩
    · cases drop
      · exact takeFifo_nonneg _ _ hl
      · intro l hl'; simp [Book.step] at hl'
    · simp only [Book.step]; linarith
    · simp only [Book.step]
      intro h0
      have h1 : b.unmatched = 0 := by linarith
      have h2 : q - min q (held b.lots) = 0 := by linarith
      rw [hc h1, h2]; ring

theorem step_coins (b : Book) (f : Fill) :
    (b.step f).matched + held (b.step f).lots + (b.step f).dust =
      b.matched + held b.lots + b.dust + f.bought := by
  cases f with
  | buy q p fee => simp [Book.step, Fill.bought]; ring
  | sell q p fee drop =>
    have := takeFifo_held b.lots (min q (held b.lots))
    cases drop <;> simp [Book.step, Fill.bought] <;> linarith

theorem step_cost (b : Book) (f : Fill) (hf : f.Valid) :
    (b.step f).basis + value (b.step f).lots + (b.step f).dustCost =
      b.basis + value b.lots + b.dustCost + f.buyCost := by
  cases f with
  | buy q p fee =>
    simp only [Fill.Valid] at hf
    simp only [Book.step, value_append, value_cons, value_nil, Fill.buyCost]
    field_simp
    ring
  | sell q p fee drop =>
    have := takeFifo_value b.lots (min q (held b.lots))
    cases drop <;> simp [Book.step, Fill.buyCost] <;> linarith

theorem step_sold (b : Book) (f : Fill) (hf : f.Valid) (hb : b.Inv) :
    (b.step f).matched + (b.step f).unmatched = b.matched + b.unmatched + f.soldQty := by
  cases f with
  | buy q p fee => simp [Book.step, Fill.soldQty]
  | sell q p fee drop =>
    simp only [Fill.Valid] at hf
    simp only [Book.step, Fill.soldQty]
    rw [sold_eq b q hf hb.1]
    ring

theorem step_cash (b : Book) (f : Fill) (hf : f.Valid) :
    (b.step f).proceeds + (b.step f).unmatchedCash =
      b.proceeds + b.unmatchedCash + f.sellCash := by
  cases f with
  | buy q p fee => simp [Book.step, Fill.sellCash]
  | sell q p fee drop =>
    simp only [Fill.Valid] at hf
    simp only [Book.step, Fill.sellCash]
    field_simp
    ring

theorem step_piece (b : Book) (f : Fill) (hf : f.Valid) (hb : b.Inv) :
    (b.step f).proceeds - (b.step f).basis - (b.step f).piecePnl =
      b.proceeds - b.basis - b.piecePnl := by
  cases f with
  | buy q p fee => simp [Book.step]
  | sell q p fee drop =>
    simp only [Fill.Valid] at hf
    simp only [Book.step]
    rw [pieces_pnl, sold_eq b q hf hb.1]
    ring

theorem foldl_inv (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    ∀ b : Book, b.Inv → (fs.foldl Book.step b).Inv := by
  induction fs with
  | nil => intro b hb; exact hb
  | cons f fs ih =>
    intro b hb
    exact ih (fun g hg => hfs g (by simp [hg])) _ (step_inv b f (hfs f (by simp)) hb)

theorem empty_inv : Book.empty.Inv := ⟨by simp [Book.empty], le_rfl, fun _ => rfl⟩

theorem run_inv (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) : (run fs).Inv :=
  foldl_inv fs hfs _ empty_inv

/-- **Coins are conserved**: every coin bought is matched to a sell, still held, or discarded
as dust: `Σ buy qty = matched + open + dust`. No assumption on the fills; with the dust rule off
(`DUST = 0`, never discarding) the dust term is 0. Backs `fills.round_trips`. -/
theorem fifo_coins (fs : List Fill) :
    (fs.map Fill.bought).sum = (run fs).matched + held (run fs).lots + (run fs).dust := by
  suffices h : ∀ b : Book, (fs.foldl Book.step b).matched + held (fs.foldl Book.step b).lots +
      (fs.foldl Book.step b).dust = b.matched + held b.lots + b.dust + (fs.map Fill.bought).sum by
    have := h Book.empty
    simp only [run]
    rw [this]; simp [Book.empty]
  induction fs with
  | nil => intro b; simp
  | cons f fs ih =>
    intro b
    rw [List.foldl_cons, ih, step_coins, List.map_cons, List.sum_cons]
    ring

/-- **Sells are conserved**: `Σ sell qty = matched + unmatched`; every coin sold is matched to a
lot or counted as unmatched (sold before the export began). Assumes positive quantities.
Backs `fills.round_trips`. -/
theorem fifo_sold (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    (fs.map Fill.soldQty).sum = (run fs).matched + (run fs).unmatched := by
  suffices h : ∀ b : Book, b.Inv → (fs.foldl Book.step b).matched +
      (fs.foldl Book.step b).unmatched = b.matched + b.unmatched + (fs.map Fill.soldQty).sum by
    have := h Book.empty empty_inv
    simp only [run]
    rw [this]; simp [Book.empty]
  induction fs with
  | nil => intro b _; simp
  | cons f fs ih =>
    intro b hb
    have hf := hfs f (by simp)
    rw [List.foldl_cons, ih (fun g hg => hfs g (by simp [hg])) _ (step_inv b f hf hb),
      step_sold b f hf hb, List.map_cons, List.sum_cons]
    ring

/-- **Cost is conserved**: what the buys cost (fees in) is the basis of the coins sold, plus the
cost of the coins still held, plus the cost of the dust discarded. Assumes positive quantities.
Backs `fills.round_trips` (the `size` of its trades is the basis). -/
theorem fifo_cost (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    (fs.map Fill.buyCost).sum = (run fs).basis + value (run fs).lots + (run fs).dustCost := by
  suffices h : ∀ b : Book, (fs.foldl Book.step b).basis + value (fs.foldl Book.step b).lots +
      (fs.foldl Book.step b).dustCost = b.basis + value b.lots + b.dustCost +
        (fs.map Fill.buyCost).sum by
    have := h Book.empty
    simp only [run]
    rw [this]; simp [Book.empty]
  induction fs with
  | nil => intro b; simp
  | cons f fs ih =>
    intro b
    have hf := hfs f (by simp)
    rw [List.foldl_cons, ih (fun g hg => hfs g (by simp [hg])), step_cost b f hf,
      List.map_cons, List.sum_cons]
    ring

/-- **The P&L identity**: the realized P&L is, sell by sell, `Σ (sell price − lot cost) · qty`
over the pieces of lots that sell matched (oldest first), minus the sell's share of its fee
(`fee · sold / q`). Assumes positive quantities. Backs `fills.round_trips`. -/
theorem fifo_pnl_pieces (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    (run fs).realized = (run fs).piecePnl := by
  suffices h : ∀ b : Book, b.Inv → (fs.foldl Book.step b).proceeds -
      (fs.foldl Book.step b).basis - (fs.foldl Book.step b).piecePnl =
        b.proceeds - b.basis - b.piecePnl by
    have := h Book.empty empty_inv
    have h0 : Book.empty.proceeds - Book.empty.basis - Book.empty.piecePnl = 0 := by
      simp [Book.empty]
    rw [h0] at this
    simp only [run, Book.realized]
    linarith
  induction fs with
  | nil => intro b _; simp
  | cons f fs ih =>
    intro b hb
    have hf := hfs f (by simp)
    rw [List.foldl_cons, ih (fun g hg => hfs g (by simp [hg])) _ (step_inv b f hf hb),
      step_piece b f hf hb]

/-- The cash the sells brought in is what the matched coins got plus what the unmatched part
would have got. -/
theorem fifo_cash (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    (fs.map Fill.sellCash).sum = (run fs).proceeds + (run fs).unmatchedCash := by
  suffices h : ∀ b : Book, (fs.foldl Book.step b).proceeds +
      (fs.foldl Book.step b).unmatchedCash = b.proceeds + b.unmatchedCash +
        (fs.map Fill.sellCash).sum by
    have := h Book.empty
    simp only [run]
    rw [this]; simp [Book.empty]
  induction fs with
  | nil => intro b; simp
  | cons f fs ih =>
    intro b
    have hf := hfs f (by simp)
    rw [List.foldl_cons, ih (fun g hg => hfs g (by simp [hg])), step_cash b f hf,
      List.map_cons, List.sum_cons]
    ring

/-- **P&L is cash in minus cash out, plus what is still held**: realized P&L = what the matched
coins sold for − what every buy cost + the cost of the coins still held + the cost of the dust
discarded. Assumes positive quantities. Backs `fills.round_trips`. -/
theorem fifo_pnl_cash (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) :
    (run fs).realized =
      (run fs).proceeds - (fs.map Fill.buyCost).sum + value (run fs).lots + (run fs).dustCost := by
  rw [fifo_cost fs hfs, Book.realized]
  ring

/-- **A flat book**: when nothing is left open, nothing was discarded as dust and no sell
exceeded the coins held, the realized P&L is exactly the cash the sells brought in minus the cash
the buys took out, fees included. Assumes positive quantities. Backs `fills.round_trips`. -/
theorem fifo_flat (fs : List Fill) (hfs : ∀ f ∈ fs, f.Valid) (hflat : (run fs).lots = [])
    (hdust : (run fs).dustCost = 0) (hcover : (run fs).unmatched = 0) :
    (run fs).realized = (fs.map Fill.sellCash).sum - (fs.map Fill.buyCost).sum := by
  have hc : (run fs).unmatchedCash = 0 := (run_inv fs hfs).2.2 hcover
  rw [fifo_pnl_cash fs hfs, fifo_cash fs hfs, hflat, hdust, hc]
  simp

end Fifo

/-! ## 5. Drawdown: `_drawdown` -/

section Drawdown

open Finset

variable (C : ℕ → ℝ)

/-- `np.maximum.accumulate(curve)`: the highest the curve has been so far. -/
def runPeak : ℕ → ℝ
  | 0 => C 0
  | t + 1 => max (runPeak t) (C (t + 1))

theorem runPeak_succ (t : ℕ) : runPeak C (t + 1) = max (runPeak C t) (C (t + 1)) := rfl

theorem runPeak_mono {s t : ℕ} (h : s ≤ t) : runPeak C s ≤ runPeak C t := by
  induction t with
  | zero => rw [Nat.le_zero.mp h]
  | succ t ih =>
    rcases Nat.lt_or_eq_of_le h with h' | rfl
    · exact (ih (Nat.lt_succ_iff.mp h')).trans (le_max_left _ _)
    · exact le_rfl

theorem le_runPeak {s t : ℕ} (h : s ≤ t) : C s ≤ runPeak C t := by
  have : C s ≤ runPeak C s := by
    cases s with
    | zero => exact le_rfl
    | succ s => exact le_max_right _ _
  exact this.trans (runPeak_mono C h)

theorem runPeak_attained (t : ℕ) : ∃ s ≤ t, runPeak C t = C s := by
  induction t with
  | zero => exact ⟨0, le_rfl, rfl⟩
  | succ t ih =>
    obtain ⟨s, hs, he⟩ := ih
    rw [runPeak_succ]
    rcases le_total (runPeak C t) (C (t + 1)) with h | h
    · exact ⟨t + 1, le_rfl, max_eq_right h⟩
    · exact ⟨s, by omega, by rw [max_eq_left h, he]⟩

/-- **The drop at `t` is the largest fall from any earlier point**: `peak_t − C_t` is the maximum
of `C_s − C_t` over `s ≤ t` (it is attained). For any sequence `C`. -/
theorem drop_eq_max_fall (t : ℕ) :
    (∀ s ≤ t, C s - C t ≤ runPeak C t - C t) ∧ ∃ s ≤ t, C s - C t = runPeak C t - C t := by
  refine ⟨fun s hs => by linarith [le_runPeak C hs], ?_⟩
  obtain ⟨s, hs, he⟩ := runPeak_attained C t
  exact ⟨s, hs, by rw [he]⟩

/-- `dd.max()` over `t = 0..T`: the worst drop from a peak. -/
def worstDrop (T : ℕ) : ℝ :=
  (range (T + 1)).sup' nonempty_range_add_one (fun t => runPeak C t - C t)

/-- **The worst drawdown is the largest fall over all pairs**: `max_t (peak_t − C_t)` equals
`max_{s ≤ t ≤ T} (C_s − C_t)`; every fall is at most it, and some fall equals it. For any
sequence `C` (the code's curve is `C_0 = 0`, `C_{t+1} = C_t + x_t`). Backs `judge._drawdown`. -/
theorem worstDrop_spec (T : ℕ) :
    (∀ s t, s ≤ t → t ≤ T → C s - C t ≤ worstDrop C T) ∧
      ∃ s t, s ≤ t ∧ t ≤ T ∧ C s - C t = worstDrop C T := by
  constructor
  · intro s t hst htT
    have h1 := (drop_eq_max_fall C t).1 s hst
    have h2 : runPeak C t - C t ≤ worstDrop C T :=
      Finset.le_sup' (fun t => runPeak C t - C t) (Finset.mem_range.mpr (Nat.lt_succ_of_le htT))
    linarith
  · obtain ⟨t, ht, he⟩ := Finset.exists_mem_eq_sup' (nonempty_range_add_one (n := T))
      (fun t => runPeak C t - C t)
    obtain ⟨s, hs, hs'⟩ := (drop_eq_max_fall C t).2
    refine ⟨s, t, hs, Nat.lt_succ_iff.mp (Finset.mem_range.mp ht), ?_⟩
    rw [hs', worstDrop, he]

/-- The last time `≤ T` the curve stood at its running peak (`at_peak[-1]` in `_drawdown`). -/
def lastPeak (T : ℕ) : ℕ :=
  ((range (T + 1)).filter (fun i => runPeak C i = C i)).max'
    ⟨0, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (Nat.succ_pos T), rfl⟩⟩

theorem lastPeak_spec (T : ℕ) :
    lastPeak C T ≤ T ∧ runPeak C (lastPeak C T) = C (lastPeak C T) ∧
      ∀ i, lastPeak C T < i → i ≤ T → C i < runPeak C i := by
  set S := (range (T + 1)).filter (fun i => runPeak C i = C i) with hS
  have hne : S.Nonempty :=
    ⟨0, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (Nat.succ_pos T), rfl⟩⟩
  have hmem : lastPeak C T ∈ S := Finset.max'_mem S hne
  rw [hS, Finset.mem_filter, Finset.mem_range] at hmem
  refine ⟨Nat.lt_succ_iff.mp hmem.1, hmem.2, ?_⟩
  intro i hi hiT
  have hle : C i ≤ runPeak C i := le_runPeak C le_rfl
  rcases lt_or_eq_of_le hle with h | h
  · exact h
  · exfalso
    have hiS : i ∈ S := by
      rw [hS, Finset.mem_filter, Finset.mem_range]
      exact ⟨Nat.lt_succ_of_le hiT, h.symm⟩
    have := Finset.le_max' S i hiS
    exact absurd (lt_of_lt_of_le hi this) (lt_irrefl _)

/-- **"Trades since the peak" counts from the last time the curve was at its high**: with
`L = lastPeak C T` (the last `i ≤ T` where the curve equals its running peak; `_drawdown`
returns `T − L`), the running peak at `T` is `C_L`, every point up to `T` is at most `C_L`, and
every point after `L` is strictly below it. So the current drop is `C_L − C_T`. Exact equality;
the code tests "at the peak" as `peak − curve ≤ 1e-12`. Backs `judge._drawdown`. -/
theorem lastPeak_is_last_high (T : ℕ) :
    runPeak C T = C (lastPeak C T) ∧ (∀ i ≤ T, C i ≤ C (lastPeak C T)) ∧
      ∀ i, lastPeak C T < i → i ≤ T → C i < C (lastPeak C T) := by
  obtain ⟨hle, hpk, hafter⟩ := lastPeak_spec C T
  set L := lastPeak C T with hL
  have hflat : ∀ k, L + k ≤ T → runPeak C (L + k) = C L := by
    intro k
    induction k with
    | zero => intro _; simpa using hpk
    | succ k ih =>
      intro hk
      have ih' := ih (by omega)
      have hlt := hafter (L + k + 1) (by omega) hk
      rw [show L + (k + 1) = L + k + 1 by ring, runPeak_succ, ih']
      rw [runPeak_succ, ih'] at hlt
      have : C (L + k + 1) < C L := by
        by_contra hc
        rw [max_eq_right (not_lt.mp hc)] at hlt
        exact lt_irrefl _ hlt
      exact max_eq_left this.le
  have hT : runPeak C T = C L := by
    have := hflat (T - L) (by omega)
    rwa [show L + (T - L) = T by omega] at this
  refine ⟨hT, fun i hi => hT ▸ le_runPeak C hi, ?_⟩
  intro i hi hiT
  have h1 := hafter i hi hiT
  have h2 := hflat (i - L) (by omega)
  rw [show L + (i - L) = i by omega] at h2
  rwa [h2] at h1

end Drawdown

/-! ## 6. Time in the market: `_exposure` -/

section Exposure

open MeasureTheory Set

/-- One pass of the loop in `_exposure`, state (covered, current start, current end): a span
that starts after the current block ends closes the block; otherwise it extends it. -/
def sweepStep (st : ℝ × ℝ × ℝ) (sp : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  if st.2.2 < sp.1 then (st.1 + (st.2.2 - st.2.1), sp.1, sp.2)
  else (st.1, st.2.1, max st.2.2 sp.2)

def sweepFinish (st : ℝ × ℝ × ℝ) : ℝ := st.1 + (st.2.2 - st.2.1)

theorem sweepStep_gap {st : ℝ × ℝ × ℝ} {sp : ℝ × ℝ} (h : st.2.2 < sp.1) :
    sweepStep st sp = (st.1 + (st.2.2 - st.2.1), sp.1, sp.2) := by
  simp [sweepStep, h]

theorem sweepStep_merge {st : ℝ × ℝ × ℝ} {sp : ℝ × ℝ} (h : ¬ st.2.2 < sp.1) :
    sweepStep st sp = (st.1, st.2.1, max st.2.2 sp.2) := by
  simp [sweepStep, h]

/-- The time `_exposure` counts as covered, for spans (entry, exit) sorted by entry. -/
def sweep : List (ℝ × ℝ) → ℝ
  | [] => 0
  | sp :: rest => sweepFinish (rest.foldl sweepStep (0, sp.1, sp.2))

/-- The time covered by at least one span. -/
def spanUnion : List (ℝ × ℝ) → Set ℝ
  | [] => ∅
  | sp :: rest => Icc sp.1 sp.2 ∪ spanUnion rest

theorem mem_spanUnion (l : List (ℝ × ℝ)) (t : ℝ) :
    t ∈ spanUnion l ↔ ∃ sp ∈ l, sp.1 ≤ t ∧ t ≤ sp.2 := by
  induction l with
  | nil => simp [spanUnion]
  | cons sp l ih => simp [spanUnion, ih, mem_Icc]

theorem measurableSet_spanUnion (l : List (ℝ × ℝ)) : MeasurableSet (spanUnion l) := by
  induction l with
  | nil => exact MeasurableSet.empty
  | cons sp l ih => exact measurableSet_Icc.union ih

/-- The invariant of the sweep: what is finished (`B`, all before the current start) has measure
`covered`, and the current block is `[a, b]`. -/
theorem sweep_inv (rest : List (ℝ × ℝ)) :
    ∀ (c a b : ℝ) (B : Set ℝ), a ≤ b → 0 ≤ c → MeasurableSet B → B ⊆ Iio a →
      volume B = ENNReal.ofReal c →
      (∀ sp ∈ rest, a ≤ sp.1 ∧ sp.1 ≤ sp.2) → rest.Pairwise (fun x y => x.1 ≤ y.1) →
      0 ≤ sweepFinish (rest.foldl sweepStep (c, a, b)) ∧
        volume (B ∪ Icc a b ∪ spanUnion rest) =
          ENNReal.ofReal (sweepFinish (rest.foldl sweepStep (c, a, b))) := by
  induction rest with
  | nil =>
    intro c a b B hab hc hB hBa hvol _ _
    simp only [List.foldl_nil, sweepFinish, spanUnion, union_empty]
    refine ⟨by linarith, ?_⟩
    have hd : Disjoint B (Icc a b) :=
      Set.disjoint_left.mpr (fun x hxB hxI => absurd (hBa hxB) (not_lt.mpr hxI.1))
    rw [measure_union hd measurableSet_Icc, hvol, Real.volume_Icc,
      ENNReal.ofReal_add hc (by linarith)]
  | cons sp rest ih =>
    intro c a b B hab hc hB hBa hvol hspans hsorted
    obtain ⟨hsp1, hsp2⟩ := hspans sp (by simp)
    rw [List.pairwise_cons] at hsorted
    have hrest_sorted := hsorted.2
    have hrest_ge : ∀ x ∈ rest, sp.1 ≤ x.1 := hsorted.1
    rw [List.foldl_cons]
    by_cases hgap : b < sp.1
    · -- a gap: close the block `[a, b]`, start `[sp.1, sp.2]`
      have hd : Disjoint B (Icc a b) :=
        Set.disjoint_left.mpr (fun x hxB hxI => absurd (hBa hxB) (not_lt.mpr hxI.1))
      have hB' : B ∪ Icc a b ⊆ Iio sp.1 := by
        intro x hx
        rcases hx with hx | hx
        · exact lt_of_lt_of_le (hBa hx) (hab.trans hgap.le)
        · exact lt_of_le_of_lt hx.2 hgap
      have hvol' : volume (B ∪ Icc a b) = ENNReal.ofReal (c + (b - a)) := by
        rw [measure_union hd measurableSet_Icc, hvol, Real.volume_Icc,
          ENNReal.ofReal_add hc (by linarith)]
      rw [sweepStep_gap (st := (c, a, b)) hgap]
      have := ih (c + (b - a)) sp.1 sp.2 (B ∪ Icc a b) hsp2 (by linarith)
        (hB.union measurableSet_Icc) hB' hvol'
        (fun x hx => ⟨hrest_ge x hx, (hspans x (by simp [hx])).2⟩) hrest_sorted
      refine ⟨this.1, ?_⟩
      rw [← this.2]
      congr 1
      simp only [spanUnion]
      ext t; simp only [mem_union]; tauto
    · -- an overlap or a touch: extend the block to `[a, max b sp.2]`
      have hle : sp.1 ≤ b := not_lt.mp hgap
      have hmerge : Icc a b ∪ Icc sp.1 sp.2 = Icc a (max b sp.2) := by
        rw [Icc_union_Icc' hle (hsp1.trans hsp2), min_eq_left hsp1]
      rw [sweepStep_merge (st := (c, a, b)) hgap]
      have := ih c a (max b sp.2) B (hab.trans (le_max_left _ _)) hc hB hBa hvol
        (fun x hx => ⟨hsp1.trans (hrest_ge x hx), (hspans x (by simp [hx])).2⟩) hrest_sorted
      refine ⟨this.1, ?_⟩
      rw [← this.2, ← hmerge]
      congr 1
      simp only [spanUnion]
      ext t; simp only [mem_union]; tauto

/-- **`_exposure`'s sweep measures the union.** For spans sorted by entry (the code sorts them),
each with entry ≤ exit (the code drops the others), the sweep returns the total length (Lebesgue
measure) of the time covered by at least one span: overlapping trades are not counted twice.
Backs `judge._exposure`. -/
theorem sweep_eq_volume (l : List (ℝ × ℝ)) (hspans : ∀ sp ∈ l, sp.1 ≤ sp.2)
    (hsorted : l.Pairwise (fun x y => x.1 ≤ y.1)) :
    0 ≤ sweep l ∧ ENNReal.ofReal (sweep l) = volume (spanUnion l) := by
  cases l with
  | nil => simp [sweep, spanUnion]
  | cons sp rest =>
    rw [List.pairwise_cons] at hsorted
    have h := sweep_inv rest 0 sp.1 sp.2 ∅ (hspans sp (by simp)) le_rfl MeasurableSet.empty
      (empty_subset _) (by simp)
      (fun x hx => ⟨hsorted.1 x hx, hspans x (by simp [hx])⟩) hsorted.2
    refine ⟨h.1, ?_⟩
    rw [sweep, ← h.2, spanUnion, empty_union]

/-- **Time in the market is a share**: when every span lies in `[lo, hi]` (the code divides by
last exit − first entry), the covered time is at most `hi − lo`, so `_exposure` is in
`[0, 1]`. Backs `judge._exposure`. -/
theorem sweep_le (l : List (ℝ × ℝ)) (hspans : ∀ sp ∈ l, sp.1 ≤ sp.2)
    (hsorted : l.Pairwise (fun x y => x.1 ≤ y.1)) {lo hi : ℝ} (hlh : lo ≤ hi)
    (hin : ∀ sp ∈ l, lo ≤ sp.1 ∧ sp.2 ≤ hi) : sweep l ≤ hi - lo := by
  obtain ⟨h0, hv⟩ := sweep_eq_volume l hspans hsorted
  have hsub : spanUnion l ⊆ Icc lo hi := by
    intro t ht
    obtain ⟨sp, hsp, h1, h2⟩ := (mem_spanUnion l t).mp ht
    exact ⟨(hin sp hsp).1.trans h1, h2.trans (hin sp hsp).2⟩
  have := measure_mono (μ := volume) hsub
  rw [← hv, Real.volume_Icc] at this
  exact (ENNReal.ofReal_le_ofReal_iff (by linarith)).mp this

end Exposure

end EdgeOrLuck.Counting

end

#print axioms EdgeOrLuck.Counting.pStreakAtMost_eq
#print axioms EdgeOrLuck.Counting.pStreakAtMost_eq_noRun
#print axioms EdgeOrLuck.Counting.dpTotal_eq_one_of_le
#print axioms EdgeOrLuck.Counting.sum_weight
#print axioms EdgeOrLuck.Counting.weight_nonneg
#print axioms EdgeOrLuck.Counting.maxRun_le_iff
#print axioms EdgeOrLuck.Counting.orbit_count_le
#print axioms EdgeOrLuck.Counting.randP_valid
#print axioms EdgeOrLuck.Counting.coinFlip_valid
#print axioms EdgeOrLuck.Counting.coinFlip_valid_of_indep
#print axioms EdgeOrLuck.Counting.signFlip_mean
#print axioms EdgeOrLuck.Counting.signFlip_second_moment
#print axioms EdgeOrLuck.Counting.pairSum_eq
#print axioms EdgeOrLuck.Counting.variance_mean_ar1
#print axioms EdgeOrLuck.Counting.tendsto_ar1Ratio
#print axioms EdgeOrLuck.Counting.ar1Ratio_le_limit
#print axioms EdgeOrLuck.Counting.one_le_ar1Factor
#print axioms EdgeOrLuck.Counting.ar1Factor_strictMonoOn
#print axioms EdgeOrLuck.Counting.ar1Factor_le_nine
#print axioms EdgeOrLuck.Counting.ar1Floor_mem
#print axioms EdgeOrLuck.Counting.serialFactorOf_mem
#print axioms EdgeOrLuck.Counting.fifo_coins
#print axioms EdgeOrLuck.Counting.fifo_sold
#print axioms EdgeOrLuck.Counting.fifo_cost
#print axioms EdgeOrLuck.Counting.fifo_pnl_pieces
#print axioms EdgeOrLuck.Counting.fifo_pnl_cash
#print axioms EdgeOrLuck.Counting.fifo_flat
#print axioms EdgeOrLuck.Counting.worstDrop_spec
#print axioms EdgeOrLuck.Counting.lastPeak_is_last_high
#print axioms EdgeOrLuck.Counting.sweep_eq_volume
#print axioms EdgeOrLuck.Counting.sweep_le
