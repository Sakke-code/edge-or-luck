import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Edge or luck: selection, e-value combination, and the small algebraic claims

Machine-checked statements behind these parts of `src/edgeorluck`:

1. **Markov for e-values** (`evidence.py: BASE_BAR`): a strategy needs `e ≥ 40`, because an e-value
   reaches `1/α` with chance at most `α` (`1/40 = 2.5%`, one side of the 5% budget).
2. **Averages and bests** (`evidence.py: describe_family, max_versions_passing`): the average of
   e-values is an e-value, the best of `N` divided by `N` is an e-value, so the best of `N`
   versions must clear `40 N` ("versions tried raise the bar").
3. **e-BH controls the false discovery rate under any dependence** (Wang & Ramdas 2022): any
   self-consistent rejection set has `E[|R ∩ H0| / max(|R|,1)] ≤ q |H0| / m ≤ q`.
4. **The e-BH procedure exactly as `evidence.py: ebh` computes it**: `k*` and the selected set,
   its size, self-consistency, zero (and small) e-values never selected, and the Python ranked
   loop (any tie-break, listed hypotheses only, comparisons of logarithms) gives the same set.
5. **One strategy, `N` versions** (`judge.py: analyze`, `ebh(log_e, Q, total=2*family)`): "up" is
   selected iff `e_up ≥ 2N/q` (`= 40 N` at `q = 1/20`).
6. **Brier is proper** (`web/ledger.py: Ledger`): `E[(p - Y)²] = (p - π)² + π(1 - π)`.
7. **Money vs percent** (`judge.py: _money_card`): the exact sense in which "your losing trades
   were bigger than your winning ones" is true, and two senses in which it is not.
8. **Wilson** (`evidence.py: wilson_bounds_count`): the ends are the roots of the score equation,
   the interval is the score-test inversion, and the clipping to `[0, 1]` never changes them.
9. **Compare's range** (`compare.py: compare_runs`): the difference range and the union bound
   (also when the live side is checked at every time), and the false "worse" verdict rate.

Everything is about real numbers (exact arithmetic). The float behaviour of the Python code is the
job of the bridge tests in `tests/test_bridge_selection.py`.
-/

namespace EdgeOrLuck.Selection

open MeasureTheory Finset

/-! ## 1. Markov for e-values -/

section EValues

variable {Ω : Type*} [MeasurableSpace Ω]

/-- An e-value: a nonnegative, integrable random variable with expectation at most 1.
(Integrability is part of the definition on purpose: in Lean the integral of a non-integrable
function is `0`, so `∫ E ≤ 1` alone would say nothing.) -/
structure IsEValue (μ : Measure Ω) (E : Ω → ℝ) : Prop where
  nonneg : ∀ ω, 0 ≤ E ω
  integrable : Integrable E μ
  integral_le_one : ∫ ω, E ω ∂μ ≤ 1

/-- **Markov for e-values.** If `E` is an e-value and `α > 0`, then `P(E ≥ 1/α) ≤ α`.
(The spec's `α ≤ 1` is not needed; the measure need not be a probability measure.)

Backs `evidence.py: BASE_BAR = 2/Q = 40` and the judge's sentence "One strategy alone needs
e >= 40" (`judge.py: HOW_WE_KNOW`): with no edge, the e-value reaches 40 with chance at most
`1/40 = 2.5%`, one side of the 5% budget (`ALPHA / 2`). -/
theorem evalue_markov (μ : Measure Ω) {E : Ω → ℝ} (hE : IsEValue μ E) {α : ℝ} (hα : 0 < α) :
    μ.real {ω | 1 / α ≤ E ω} ≤ α := by
  have h := mul_meas_ge_le_integral_of_nonneg (Filter.Eventually.of_forall hE.nonneg)
    hE.integrable (1 / α)
  have h2 : 1 / α * μ.real {ω | 1 / α ≤ E ω} ≤ 1 := h.trans hE.integral_le_one
  rw [div_mul_eq_mul_div, one_mul, div_le_one hα] at h2
  exact h2

/-- **Markov for e-values, as a measure.** The same bound with the measure itself (in `[0, ∞]`),
so it cannot be read as `0` for an infinite set: `μ {E ≥ 1/α} ≤ α`. Backs the same code as
`evalue_markov`. -/
theorem evalue_markov_measure (μ : Measure Ω) {E : Ω → ℝ} (hE : IsEValue μ E) {α : ℝ}
    (hα : 0 < α) : μ {ω | 1 / α ≤ E ω} ≤ ENNReal.ofReal α := by
  have hint : ∫⁻ ω, ENNReal.ofReal (E ω) ∂μ ≤ 1 := by
    rw [← ofReal_integral_eq_lintegral_ofReal hE.integrable
      (Filter.Eventually.of_forall hE.nonneg)]
    exact ENNReal.ofReal_le_one.2 hE.integral_le_one
  have hm := mul_meas_ge_le_lintegral₀ (μ := μ)
    (hE.integrable.aemeasurable.ennreal_ofReal) (ENNReal.ofReal (1 / α))
  have hset : {ω | ENNReal.ofReal (1 / α) ≤ ENNReal.ofReal (E ω)} = {ω | 1 / α ≤ E ω} := by
    ext ω
    simp only [Set.mem_ofPred_eq]
    exact ENNReal.ofReal_le_ofReal_iff (hE.nonneg ω)
  rw [hset] at hm
  have h3 : ENNReal.ofReal (1 / α) * μ {ω | 1 / α ≤ E ω} ≤ 1 := hm.trans hint
  have hpos : ENNReal.ofReal (1 / α) ≠ 0 := by
    rw [ne_eq, ENNReal.ofReal_eq_zero, not_le]; positivity
  have hfin : ENNReal.ofReal (1 / α) ≠ ⊤ := ENNReal.ofReal_ne_top
  calc μ {ω | 1 / α ≤ E ω}
      = (ENNReal.ofReal (1 / α))⁻¹ * (ENNReal.ofReal (1 / α) * μ {ω | 1 / α ≤ E ω}) := by
        rw [← mul_assoc, ENNReal.inv_mul_cancel hpos hfin, one_mul]
    _ ≤ (ENNReal.ofReal (1 / α))⁻¹ * 1 := by gcongr
    _ = ENNReal.ofReal α := by
        rw [mul_one, ← ENNReal.ofReal_inv_of_pos (by positivity), one_div, inv_inv]

/-! ## 2. Averages and bests -/

/-- **The average of `N` e-values is an e-value** (`N ≥ 1`), whatever their dependence.

Backs the family arithmetic of `evidence.py: describe_family` (one family of `2 m` hypotheses,
bar `40 m`) and the idea that every version tried counts. -/
theorem average_isEValue (μ : Measure Ω) {N : ℕ} (hN : 0 < N) (E : Fin N → Ω → ℝ)
    (hE : ∀ i, IsEValue μ (E i)) : IsEValue μ (fun ω => (∑ i, E i ω) / N) where
  nonneg ω := div_nonneg (sum_nonneg fun i _ => (hE i).nonneg ω) (Nat.cast_nonneg N)
  integrable := (integrable_finsetSum univ fun i _ => (hE i).integrable).div_const _
  integral_le_one := by
    rw [integral_div, integral_finsetSum univ fun i _ => (hE i).integrable]
    rw [div_le_one (by exact_mod_cast hN)]
    calc ∑ i, ∫ ω, E i ω ∂μ ≤ ∑ _i : Fin N, (1 : ℝ) :=
          sum_le_sum fun i _ => (hE i).integral_le_one
      _ = N := by simp

/-- **Pointwise, the best divided by `N` is at most the average**: for nonnegative
`e_1, …, e_N`, `(max_i e_i) / N ≤ (1/N) Σ e_i`.

Backs "versions tried raise the bar" (`judge.py: _checks`, `versions_tried`; page: "we raise the
bar for every other version you tried"). -/
theorem max_div_le_average {N : ℕ} (hN : 0 < N) (e : Fin N → ℝ) (he : ∀ i, 0 ≤ e i) :
    univ.sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) e / N ≤ (∑ i, e i) / N := by
  obtain ⟨i, -, hi⟩ := exists_mem_eq_sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) e
  rw [hi]
  gcongr
  exact single_le_sum (fun j _ => he j) (mem_univ i)

/-- **The best of `N` e-values, divided by `N`, is an e-value**, under any dependence between
the versions.

Backs `evidence.py: max_versions_passing` and `describe_family` (`bar_alone = 40 m`): keeping
the best of `N` versions costs a factor `N` in the bar. -/
theorem best_div_isEValue (μ : Measure Ω) {N : ℕ} (hN : 0 < N) (E : Fin N → Ω → ℝ)
    (hE : ∀ i, IsEValue μ (E i)) :
    IsEValue μ (fun ω => univ.sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) (fun i => E i ω) / N) := by
  have hne : (univ : Finset (Fin N)).Nonempty := univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩
  have havg := average_isEValue μ hN E hE
  have hle : ∀ ω, univ.sup' hne (fun i => E i ω) / N ≤ (∑ i, E i ω) / N :=
    fun ω => max_div_le_average hN (fun i => E i ω) (fun i => (hE i).nonneg ω)
  have hnn : ∀ ω, 0 ≤ univ.sup' hne (fun i => E i ω) / N := by
    intro ω
    apply div_nonneg _ (Nat.cast_nonneg N)
    exact (hE ⟨0, hN⟩).nonneg ω |>.trans (le_sup' (fun i => E i ω) (mem_univ _))
  have hmeas : AEStronglyMeasurable (fun ω => univ.sup' hne (fun i => E i ω) / N) μ := by
    have h1 : AEStronglyMeasurable (univ.sup' hne E) μ :=
      Finset.sup'_induction hne E (p := fun f => AEStronglyMeasurable f μ)
        (fun f hf g hg => hf.sup hg) (fun i _ => (hE i).integrable.aestronglyMeasurable)
    have h2 : (fun ω => univ.sup' hne (fun i => E i ω)) = univ.sup' hne E := by
      funext ω; rw [Finset.sup'_apply]
    have h3 : AEStronglyMeasurable (fun ω => univ.sup' hne (fun i => E i ω)) μ := by
      rw [h2]; exact h1
    simpa only [div_eq_mul_inv] using h3.mul_const ((N : ℝ)⁻¹)
  have hint : Integrable (fun ω => univ.sup' hne (fun i => E i ω) / N) μ := by
    refine havg.integrable.mono' hmeas (Filter.Eventually.of_forall fun ω => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (hnn ω)]
    exact hle ω
  exact
    { nonneg := hnn
      integrable := hint
      integral_le_one := (integral_mono hint havg.integrable hle).trans havg.integral_le_one }

/-- **The best of `N` versions must clear `N/α`.** For e-values `E_1, …, E_N` (any dependence)
and `α > 0`: `P(max_i E_i ≥ N/α) ≤ α`. At `α = 1/40` the bar is `40 N`.

Backs `evidence.py` ("with N versions tried ... one strategy needs e >= 40 N: the Bonferroni
95%/N range clearing zero"), `describe_family` and `max_versions_passing`. -/
theorem best_of_N_markov (μ : Measure Ω) {N : ℕ} (hN : 0 < N) (E : Fin N → Ω → ℝ)
    (hE : ∀ i, IsEValue μ (E i)) {α : ℝ} (hα : 0 < α) :
    μ.real {ω | (N : ℝ) / α ≤ univ.sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) (fun i => E i ω)} ≤ α := by
  have h := evalue_markov μ (best_div_isEValue μ hN E hE) hα
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hset : {ω | (N : ℝ) / α ≤ univ.sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) (fun i => E i ω)} =
      {ω | 1 / α ≤ univ.sup' (univ_nonempty_iff.2 ⟨⟨0, hN⟩⟩) (fun i => E i ω) / N} := by
    ext ω
    simp only [Set.mem_ofPred_eq]
    rw [le_div_iff₀ hNpos, show 1 / α * (N : ℝ) = N / α by ring]
  rw [hset]
  exact h

end EValues

/-! ## 3. e-BH controls the false discovery rate under any dependence -/

variable {m : ℕ}

/-- The false discovery proportion of a selection `R` when `H0` are the true nulls:
`|R ∩ H0| / max(|R|, 1)`. -/
noncomputable def fdp (R H0 : Finset (Fin m)) : ℝ :=
  ((R ∩ H0).card : ℝ) / max (R.card : ℝ) 1

/-- A selection `R` of the `m` hypotheses is self-consistent at level `q` for the e-values `e`
when every selected `j` has `e_j · q · |R| ≥ m` (Wang & Ramdas 2022). -/
def SelfConsistent (q : ℝ) (e : Fin m → ℝ) (R : Finset (Fin m)) : Prop :=
  ∀ j ∈ R, (m : ℝ) ≤ e j * q * R.card

theorem fdp_nonneg (R H0 : Finset (Fin m)) : 0 ≤ fdp R H0 := by
  unfold fdp; positivity

theorem fdp_le_one (R H0 : Finset (Fin m)) : fdp R H0 ≤ 1 := by
  unfold fdp
  rw [div_le_one (lt_of_lt_of_le one_pos (le_max_right _ _))]
  exact le_trans (by exact_mod_cast card_le_card inter_subset_left) (le_max_left _ _)

/-- **The key pointwise inequality of e-BH**: for a self-consistent `R`, `q > 0` and `e_j ≥ 0`,
`1{j ∈ R} / max(|R|, 1) ≤ q e_j / m`.

Backs `evidence.py: ebh` ("valid under any dependence"). -/
theorem indicator_div_le {q : ℝ} (hq : 0 < q) {e : Fin m → ℝ} {R : Finset (Fin m)}
    (hR : SelfConsistent q e R) (j : Fin m) (hej : 0 ≤ e j) :
    (if j ∈ R then (1 : ℝ) else 0) / max (R.card : ℝ) 1 ≤ q * e j / m := by
  have hm : (0 : ℝ) < m := by exact_mod_cast j.pos
  split_ifs with hj
  · have hcard : (1 : ℝ) ≤ R.card := by exact_mod_cast card_pos.2 ⟨j, hj⟩
    rw [max_eq_left hcard, div_le_div_iff₀ (by linarith) hm]
    have := hR j hj
    nlinarith
  · rw [zero_div]; positivity

theorem card_inter_eq_sum (R H0 : Finset (Fin m)) :
    ((R ∩ H0).card : ℝ) = ∑ j ∈ H0, if j ∈ R then (1 : ℝ) else 0 := by
  rw [← natCast_card_filter, filter_mem_eq_inter, inter_comm]

theorem card_eq_sum (R : Finset (Fin m)) :
    (R.card : ℝ) = ∑ j, if j ∈ R then (1 : ℝ) else 0 := by
  rw [← natCast_card_filter, filter_mem_eq_inter, univ_inter]

/-- Pointwise: the false discovery proportion of a self-consistent selection is at most
`Σ_{j ∈ H0} q e_j / m`. Backs `evidence.py: ebh`. -/
theorem fdp_le_sum {q : ℝ} (hq : 0 < q) {e : Fin m → ℝ} (he : ∀ j, 0 ≤ e j)
    {R : Finset (Fin m)} (hR : SelfConsistent q e R) (H0 : Finset (Fin m)) :
    fdp R H0 ≤ ∑ j ∈ H0, q * e j / m := by
  unfold fdp
  rw [card_inter_eq_sum, sum_div]
  exact sum_le_sum fun j _ => indicator_div_le hq hR j (he j)

section FDR

variable {Ω : Type*} [MeasurableSpace Ω]

theorem nullBound_integrable (μ : Measure Ω) (q : ℝ) (e : Ω → Fin m → ℝ) (H0 : Finset (Fin m))
    (hnull : ∀ j ∈ H0, Integrable (fun ω => e ω j) μ) :
    Integrable (fun ω => ∑ j ∈ H0, q * e ω j / m) μ :=
  integrable_finsetSum _ fun j hj => ((hnull j hj).const_mul q).div_const _

theorem nullBound_integral_le (μ : Measure Ω) {q : ℝ} (hq : 0 < q) (e : Ω → Fin m → ℝ)
    (H0 : Finset (Fin m))
    (hnull : ∀ j ∈ H0, Integrable (fun ω => e ω j) μ ∧ ∫ ω, e ω j ∂μ ≤ 1) :
    ∫ ω, ∑ j ∈ H0, q * e ω j / m ∂μ ≤ q * H0.card / m := by
  rw [integral_finsetSum _ fun j hj => ((hnull j hj).1.const_mul q).div_const _]
  calc ∑ j ∈ H0, ∫ ω, q * e ω j / m ∂μ ≤ ∑ _j ∈ H0, q / m := by
        refine sum_le_sum fun j hj => ?_
        rw [integral_div, integral_const_mul]
        gcongr
        calc q * ∫ ω, e ω j ∂μ ≤ q * 1 := by gcongr; exact (hnull j hj).2
          _ = q := mul_one q
    _ = q * H0.card / m := by rw [sum_const, nsmul_eq_mul]; ring

/-- **e-BH false discovery rate, lower-integral form (no measurability of `R` needed).**
`m` hypotheses, `q > 0`, nonnegative e-values `e ω j`; for every true null `j ∈ H0` the e-value
is integrable with `∫ e_j ≤ 1`; nothing at all is assumed about how the e-values depend on each
other. If the selection `R ω` is self-consistent at every `ω`, then
`∫⁻ FDP ≤ q |H0| / m`.

Backs `evidence.py: ebh` ("valid under any dependence") and `judge.py: HOW_WE_KNOW` ("Versions
tried but not shown count as e = 0, which is valid under any dependence"). -/
theorem ebh_fdr_lintegral (μ : Measure Ω) {q : ℝ} (hq : 0 < q)
    (e : Ω → Fin m → ℝ) (he : ∀ ω j, 0 ≤ e ω j) (H0 : Finset (Fin m))
    (hnull : ∀ j ∈ H0, Integrable (fun ω => e ω j) μ ∧ ∫ ω, e ω j ∂μ ≤ 1)
    (R : Ω → Finset (Fin m)) (hR : ∀ ω, SelfConsistent q (e ω) (R ω)) :
    ∫⁻ ω, ENNReal.ofReal (fdp (R ω) H0) ∂μ ≤ ENNReal.ofReal (q * H0.card / m) := by
  have hg := nullBound_integrable μ q e H0 fun j hj => (hnull j hj).1
  have hg0 : 0 ≤ᵐ[μ] fun ω => ∑ j ∈ H0, q * e ω j / m :=
    Filter.Eventually.of_forall fun ω => sum_nonneg fun j _ => by
      have := he ω j; positivity
  calc ∫⁻ ω, ENNReal.ofReal (fdp (R ω) H0) ∂μ
      ≤ ∫⁻ ω, ENNReal.ofReal (∑ j ∈ H0, q * e ω j / m) ∂μ :=
        lintegral_mono fun ω => ENNReal.ofReal_le_ofReal (fdp_le_sum hq (he ω) (hR ω) H0)
    _ = ENNReal.ofReal (∫ ω, ∑ j ∈ H0, q * e ω j / m ∂μ) :=
        (ofReal_integral_eq_lintegral_ofReal hg hg0).symm
    _ ≤ ENNReal.ofReal (q * H0.card / m) :=
        ENNReal.ofReal_le_ofReal (nullBound_integral_le μ hq e H0 hnull)

theorem measurable_fdp (R : Ω → Finset (Fin m)) (hR : ∀ j, MeasurableSet {ω | j ∈ R ω})
    (H0 : Finset (Fin m)) : Measurable fun ω => fdp (R ω) H0 := by
  have hind : ∀ j, Measurable fun ω => if j ∈ R ω then (1 : ℝ) else 0 := fun j =>
    Measurable.ite (hR j) measurable_const measurable_const
  have h : (fun ω => fdp (R ω) H0) = fun ω =>
      (∑ j ∈ H0, if j ∈ R ω then (1 : ℝ) else 0) /
        max (∑ j, if j ∈ R ω then (1 : ℝ) else 0) 1 := by
    funext ω; unfold fdp; rw [card_inter_eq_sum, card_eq_sum]
  rw [h]
  exact (Finset.measurable_sum _ fun j _ => hind j).div
    ((Finset.measurable_sum _ fun j _ => hind j).max measurable_const)

/-- **e-BH controls the false discovery rate under any dependence** (Wang & Ramdas 2022).
Setup: `m` hypotheses, `q > 0`, nonnegative e-values `e ω j`; every true null `j ∈ H0` has an
integrable e-value with `∫ e_j ≤ 1`; the dependence between e-values is arbitrary. `R ω` is a
self-consistent selection (`e_j q |R| ≥ m` for every selected `j`) whose membership events are
measurable. Then the false discovery proportion is integrable and
`E[|R ∩ H0| / max(|R|,1)] ≤ q |H0| / m ≤ q`.

Backs `evidence.py: ebh` and `judge.py: HOW_WE_KNOW` ("e-BH (Wang & Ramdas 2022) selects at a 5%
false discovery rate ... valid under any dependence"). -/
theorem ebh_fdr (μ : Measure Ω) {q : ℝ} (hq : 0 < q)
    (e : Ω → Fin m → ℝ) (he : ∀ ω j, 0 ≤ e ω j) (H0 : Finset (Fin m))
    (hnull : ∀ j ∈ H0, Integrable (fun ω => e ω j) μ ∧ ∫ ω, e ω j ∂μ ≤ 1)
    (R : Ω → Finset (Fin m)) (hR : ∀ ω, SelfConsistent q (e ω) (R ω))
    (hRm : ∀ j, MeasurableSet {ω | j ∈ R ω}) :
    Integrable (fun ω => fdp (R ω) H0) μ ∧
      ∫ ω, fdp (R ω) H0 ∂μ ≤ q * H0.card / m ∧ q * H0.card / m ≤ q := by
  have hg := nullBound_integrable μ q e H0 fun j hj => (hnull j hj).1
  have hle : ∀ ω, fdp (R ω) H0 ≤ ∑ j ∈ H0, q * e ω j / m := fun ω =>
    fdp_le_sum hq (he ω) (hR ω) H0
  have hint : Integrable (fun ω => fdp (R ω) H0) μ := by
    refine hg.mono' (measurable_fdp R hRm H0).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (fdp_nonneg _ _)]
    exact hle ω
  refine ⟨hint, (integral_mono hint hg hle).trans (nullBound_integral_le μ hq e H0 hnull), ?_⟩
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm; simp; exact hq.le
  · rw [mul_div_assoc]
    have hc : (H0.card : ℝ) ≤ m := by
      have := card_le_univ H0; rw [Fintype.card_fin] at this; exact_mod_cast this
    have : (H0.card : ℝ) / m ≤ 1 := by rw [div_le_one (by exact_mod_cast hm)]; exact hc
    nlinarith

end FDR

/-! ## 4. The e-BH procedure as `evidence.py: ebh` computes it -/

/-- The e-BH threshold for `k` discoveries among `m` hypotheses: `m / (q k)`. -/
noncomputable def thr (m : ℕ) (q : ℝ) (k : ℕ) : ℝ := (m : ℝ) / (q * k)

/-- How many of the `m` e-values reach `t`. -/
noncomputable def countAbove (e : Fin m → ℝ) (t : ℝ) : ℕ := (univ.filter fun j => t ≤ e j).card

/-- `k* = max {k ∈ [1, m] : #{j : e_j ≥ m/(q k)} ≥ k}` (`0` when no `k` qualifies). -/
noncomputable def kstar (e : Fin m → ℝ) (q : ℝ) : ℕ :=
  ((Icc 1 m).filter fun k => k ≤ countAbove e (thr m q k)).sup id

/-- The e-BH selection `R = {j : e_j ≥ m/(q k*)}` (empty when `k* = 0`). -/
noncomputable def ebhSet (e : Fin m → ℝ) (q : ℝ) : Finset (Fin m) :=
  univ.filter fun j => 0 < kstar e q ∧ thr m q (kstar e q) ≤ e j

theorem countAbove_le (e : Fin m → ℝ) (t : ℝ) : countAbove e t ≤ m := by
  unfold countAbove
  exact (card_filter_le _ _).trans (by rw [card_univ, Fintype.card_fin])

theorem countAbove_anti (e : Fin m → ℝ) {t t' : ℝ} (h : t ≤ t') :
    countAbove e t' ≤ countAbove e t := by
  unfold countAbove
  apply card_le_card
  intro j
  simp only [mem_filter, mem_univ, true_and]
  intro hj; linarith

theorem thr_anti {q : ℝ} (hq : 0 < q) {k k' : ℕ} (hk : 0 < k) (hkk : k ≤ k') :
    thr m q k' ≤ thr m q k := by
  unfold thr
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  apply div_le_div_of_nonneg_left (Nat.cast_nonneg m) (by positivity)
  exact mul_le_mul_of_nonneg_left (by exact_mod_cast hkk) hq.le

theorem one_div_le_thr {q : ℝ} (hq : 0 < q) {k : ℕ} (hk : 0 < k) (hkm : k ≤ m) :
    1 / q ≤ thr m q k := by
  unfold thr
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have hkm' : (k : ℝ) ≤ m := by exact_mod_cast hkm
  rw [div_le_div_iff₀ hq (by positivity)]
  nlinarith

theorem kstar_le (e : Fin m → ℝ) (q : ℝ) : kstar e q ≤ m :=
  Finset.sup_le fun _k hk => (mem_Icc.1 (mem_filter.1 hk).1).2

theorem le_kstar {e : Fin m → ℝ} {q : ℝ} {k : ℕ} (hk1 : 1 ≤ k) (hkm : k ≤ m)
    (hc : k ≤ countAbove e (thr m q k)) : k ≤ kstar e q :=
  Finset.le_sup (f := id) (mem_filter.2 ⟨mem_Icc.2 ⟨hk1, hkm⟩, hc⟩)

theorem kstar_spec {e : Fin m → ℝ} {q : ℝ} (h : 0 < kstar e q) :
    kstar e q ≤ countAbove e (thr m q (kstar e q)) := by
  have hne : ((Icc 1 m).filter fun k => k ≤ countAbove e (thr m q k)).Nonempty := by
    rw [nonempty_iff_ne_empty]
    intro hempty
    unfold kstar at h
    rw [hempty, sup_empty] at h
    exact lt_irrefl _ h
  obtain ⟨_k, hk, hkeq⟩ := exists_mem_eq_sup _ hne id
  have hk' := (mem_filter.1 hk).2
  unfold kstar
  rw [hkeq]
  exact hk'

theorem ebhSet_eq_filter {e : Fin m → ℝ} {q : ℝ} (h : 0 < kstar e q) :
    ebhSet e q = univ.filter fun j => thr m q (kstar e q) ≤ e j := by
  unfold ebhSet
  exact filter_congr fun j _ => by simp [h]

/-- **The e-BH selection has exactly `k*` members**: `|R| = k*` (`q > 0`; no assumption on the
e-values). Backs `evidence.py: ebh`, which returns the top `chosen` of its ranking. -/
theorem card_ebhSet {q : ℝ} (hq : 0 < q) (e : Fin m → ℝ) : (ebhSet e q).card = kstar e q := by
  rcases Nat.eq_zero_or_pos (kstar e q) with h0 | hpos
  · have : ebhSet e q = ∅ := by
      unfold ebhSet
      rw [filter_eq_empty_iff]
      intro j _ hj
      rw [h0] at hj
      exact lt_irrefl 0 hj.1
    rw [this, h0, card_empty]
  · rw [ebhSet_eq_filter hpos]
    change countAbove e (thr m q (kstar e q)) = kstar e q
    apply le_antisymm
    · by_contra hlt
      push Not at hlt
      have h1 : kstar e q + 1 ≤ m := hlt.trans_le (countAbove_le e _)
      have h2 : kstar e q + 1 ≤ countAbove e (thr m q (kstar e q + 1)) :=
        hlt.trans_le (countAbove_anti e (thr_anti hq hpos (Nat.le_succ _)))
      have := le_kstar (by omega) h1 h2
      omega
    · exact kstar_spec hpos

/-- **The e-BH selection is self-consistent**: every selected `j` has `e_j q |R| ≥ m` (`q > 0`).
So `ebh_fdr` applies to it. Backs `evidence.py: ebh`. -/
theorem ebhSet_selfConsistent {q : ℝ} (hq : 0 < q) (e : Fin m → ℝ) :
    SelfConsistent q e (ebhSet e q) := by
  intro j hj
  rw [card_ebhSet hq]
  simp only [ebhSet, mem_filter, mem_univ, true_and] at hj
  obtain ⟨hk, hthr⟩ := hj
  have hk' : (0 : ℝ) < kstar e q := by exact_mod_cast hk
  unfold thr at hthr
  rw [div_le_iff₀ (by positivity)] at hthr
  linarith

/-- **An e-value below `1/q` is never selected** (`q > 0`): every threshold `m/(q k)` with
`k ≤ m` is at least `1/q` (20 at `q = 0.05`). Backs `evidence.py: ebh`. -/
theorem notMem_ebhSet_of_lt {q : ℝ} (hq : 0 < q) (e : Fin m → ℝ) (j : Fin m)
    (hj : e j < 1 / q) : j ∉ ebhSet e q := by
  intro hmem
  simp only [ebhSet, mem_filter, mem_univ, true_and] at hmem
  obtain ⟨hk, hthr⟩ := hmem
  have := one_div_le_thr hq hk (kstar_le e q) (m := m)
  linarith

/-- **Hypotheses with `e_j = 0` (or below) are never selected** (`q > 0`). Backs
`evidence.py: ebh` ("hypotheses not listed ... count as e = 0 and are never selected"). -/
theorem notMem_ebhSet_of_nonpos {q : ℝ} (hq : 0 < q) (e : Fin m → ℝ) (j : Fin m)
    (hj : e j ≤ 0) : j ∉ ebhSet e q :=
  notMem_ebhSet_of_lt hq e j (lt_of_le_of_lt hj (by positivity))

/-- `σ` ranks the hypotheses from the largest e-value down, as Python's
`sorted(log_e.items(), key=..., reverse=True)` does: `σ i` is the hypothesis at (0-based) rank
`i`. Ties may be broken in any way. -/
structure IsDescRanking (e : Fin m → ℝ) (σ : Fin m → Fin m) : Prop where
  bij : Function.Bijective σ
  desc : ∀ a b : Fin m, a ≤ b → e (σ b) ≤ e (σ a)

/-- Order statistics: along a descending ranking, the e-value at rank `i` (0-based) reaches `t`
exactly when at least `i + 1` e-values reach `t`. -/
theorem rank_le_iff {e : Fin m → ℝ} {σ : Fin m → Fin m} (hσ : IsDescRanking e σ) (i : Fin m)
    (t : ℝ) : t ≤ e (σ i) ↔ (i : ℕ) + 1 ≤ countAbove e t := by
  constructor
  · intro ht
    have hsub : (Iic i).image σ ⊆ univ.filter fun j => t ≤ e j := by
      intro j hj
      simp only [mem_image, mem_Iic] at hj
      obtain ⟨a, ha, rfl⟩ := hj
      simp only [mem_filter, mem_univ, true_and]
      exact ht.trans (hσ.desc a i ha)
    have hcard : ((Iic i).image σ).card = (i : ℕ) + 1 := by
      rw [card_image_of_injective _ hσ.bij.1, Fin.card_Iic]
    unfold countAbove
    rw [← hcard]
    exact card_le_card hsub
  · intro hc
    by_contra hlt
    have hlt' : e (σ i) < t := lt_of_not_ge hlt
    have hsub : (univ.filter fun j => t ≤ e j) ⊆ (Iio i).image σ := by
      intro j hj
      simp only [mem_filter, mem_univ, true_and] at hj
      obtain ⟨a, rfl⟩ := hσ.bij.2 j
      simp only [mem_image, mem_Iio]
      refine ⟨a, ?_, rfl⟩
      by_contra hai
      have := hσ.desc i a (le_of_not_gt hai)
      linarith
    have h1 : countAbove e t ≤ (i : ℕ) := by
      unfold countAbove
      calc _ ≤ ((Iio i).image σ).card := card_le_card hsub
        _ ≤ (Iio i).card := card_image_le
        _ = i := Fin.card_Iio _
    omega

/-- Python's `chosen` in `evidence.py: ebh`: over the ranks `k = i + 1` (1-based), the largest
`k` whose `k`-th largest e-value reaches `k_total / (k * q)`; `0` if none. -/
noncomputable def pyChosen (e : Fin m → ℝ) (σ : Fin m → Fin m) (q : ℝ) : ℕ :=
  (univ.filter fun i : Fin m => (m : ℝ) / ((((i : ℕ) + 1 : ℕ) : ℝ) * q) ≤ e (σ i)).sup
    fun i => (i : ℕ) + 1

/-- Python's result `{key for key, _ in ranked[:chosen]}`: the top `chosen` of the ranking. -/
noncomputable def pyTop (e : Fin m → ℝ) (σ : Fin m → Fin m) (q : ℝ) : Finset (Fin m) :=
  (univ.filter fun i : Fin m => (i : ℕ) < pyChosen e σ q).image σ

/-- **Python's `chosen` is `k*`**, for every descending ranking (any tie-break). Backs
`evidence.py: ebh` (the loop `if value >= math.log(k_total / (k * q)): chosen = k`). -/
theorem pyChosen_eq_kstar (q : ℝ) {e : Fin m → ℝ} {σ : Fin m → Fin m} (hσ : IsDescRanking e σ) :
    pyChosen e σ q = kstar e q := by
  have himg : (univ.filter fun i : Fin m => (m : ℝ) / ((((i : ℕ) + 1 : ℕ) : ℝ) * q) ≤ e (σ i)).image
      (fun i : Fin m => (i : ℕ) + 1) = (Icc 1 m).filter fun k => k ≤ countAbove e (thr m q k) := by
    ext k
    simp only [mem_image, mem_filter, mem_univ, true_and, mem_Icc]
    constructor
    · rintro ⟨i, hi, rfl⟩
      refine ⟨⟨by omega, i.isLt⟩, ?_⟩
      rw [← rank_le_iff hσ i]
      unfold thr
      rwa [mul_comm q]
    · rintro ⟨⟨hk1, hkm⟩, hc⟩
      have hk : ((⟨k - 1, by omega⟩ : Fin m) : ℕ) + 1 = k := by dsimp only; omega
      refine ⟨⟨k - 1, by omega⟩, ?_, hk⟩
      rw [hk, mul_comm]
      exact (rank_le_iff hσ ⟨k - 1, by omega⟩ (thr m q k)).2 (by rw [hk]; exact hc)
  unfold pyChosen kstar
  rw [← himg, sup_image]
  rfl

/-- **Python's selection is the e-BH set** `R = {j : e_j ≥ m/(q k*)}`, for every descending
ranking: ties at the cut cannot change the selected set. Backs `evidence.py: ebh`. -/
theorem pyTop_eq_ebhSet {q : ℝ} (hq : 0 < q) {e : Fin m → ℝ} {σ : Fin m → Fin m}
    (hσ : IsDescRanking e σ) : pyTop e σ q = ebhSet e q := by
  unfold pyTop
  rw [pyChosen_eq_kstar q hσ]
  ext j
  simp only [mem_image, mem_filter, mem_univ, true_and]
  constructor
  · rintro ⟨a, ha, rfl⟩
    have hk : 0 < kstar e q := by omega
    have hkm := kstar_le e q
    rw [ebhSet_eq_filter hk]
    simp only [mem_filter, mem_univ, true_and]
    have h1 : thr m q (kstar e q) ≤ e (σ ⟨kstar e q - 1, by omega⟩) :=
      (rank_le_iff hσ _ _).2 (by dsimp only; have := kstar_spec hk; omega)
    exact h1.trans (hσ.desc a _ (by rw [Fin.le_def]; dsimp only; omega))
  · intro hj
    obtain ⟨a, rfl⟩ := hσ.bij.2 j
    refine ⟨a, ?_, rfl⟩
    have hk : 0 < kstar e q := by
      by_contra h0
      have hc : (ebhSet e q).card = 0 := by rw [card_ebhSet hq]; omega
      rw [card_eq_zero] at hc
      rw [hc] at hj
      exact absurd hj (notMem_empty _)
    rw [ebhSet_eq_filter hk] at hj
    simp only [mem_filter, mem_univ, true_and] at hj
    have h1 := (rank_le_iff hσ a _).1 hj
    have h2 : countAbove e (thr m q (kstar e q)) = kstar e q := by
      have := card_ebhSet hq e
      rw [ebhSet_eq_filter hk] at this
      exact this
    omega

/-- **No tie straddles the cut**: if `0 < k* < m`, the e-value ranked just below the cut is
strictly smaller than the last one selected (if `e_(k*+1) = e_(k*)`, then `k* + 1` would also
qualify). Backs the tie handling of `evidence.py: ebh`. -/
theorem ebh_no_tie_at_cut {q : ℝ} (hq : 0 < q) {e : Fin m → ℝ} {σ : Fin m → Fin m}
    (hσ : IsDescRanking e σ) (hk : 0 < kstar e q) (hkm : kstar e q < m) :
    e (σ ⟨kstar e q, hkm⟩) < e (σ ⟨kstar e q - 1, by omega⟩) := by
  rcases lt_or_eq_of_le (hσ.desc ⟨kstar e q - 1, by omega⟩ ⟨kstar e q, hkm⟩
      (by rw [Fin.le_def]; dsimp only; omega)) with h | h
  · exact h
  · exfalso
    have h1 : thr m q (kstar e q) ≤ e (σ ⟨kstar e q - 1, by omega⟩) :=
      (rank_le_iff hσ _ _).2 (by dsimp only; have := kstar_spec hk; omega)
    rw [← h] at h1
    have h3 := (rank_le_iff hσ _ _).1 h1
    have h4 : kstar e q + 1 ≤ countAbove e (thr m q (kstar e q + 1)) :=
      h3.trans (countAbove_anti e (thr_anti hq hk (Nat.le_succ _)))
    have := le_kstar (by omega) (by omega) h4
    omega

/-- **Python loops only over the listed hypotheses.** `ebh` enumerates the `L` listed keys;
the `m - L` unseen ones (versions tried but not shown) have `e = 0` and sit at the end of the
ranking. Restricting the loop to the first `L` ranks gives the same `chosen` (`q > 0`).
Backs `evidence.py: ebh` (`k_total = max(len(log_e), total)`). -/
theorem pyChosen_listed {q : ℝ} (hq : 0 < q) (e : Fin m → ℝ) (σ : Fin m → Fin m) (L : ℕ)
    (hunseen : ∀ i : Fin m, L ≤ (i : ℕ) → e (σ i) ≤ 0) :
    (univ.filter fun i : Fin m =>
        (i : ℕ) < L ∧ (m : ℝ) / ((((i : ℕ) + 1 : ℕ) : ℝ) * q) ≤ e (σ i)).sup
      (fun i => (i : ℕ) + 1) = pyChosen e σ q := by
  unfold pyChosen
  congr 1
  apply filter_congr
  intro i _
  constructor
  · exact fun h => h.2
  · intro h
    refine ⟨?_, h⟩
    by_contra hL
    push Not at hL
    have h0 := hunseen i hL
    have hpos : 0 < (m : ℝ) / ((((i : ℕ) + 1 : ℕ) : ℝ) * q) := by
      have : (0 : ℝ) < m := by exact_mod_cast i.pos
      positivity
    linarith

/-- **Comparing logarithms is comparing e-values.** Python compares `log e` with
`log(k_total / (k q))`; for `e > 0` and a positive threshold this is `e ≥ threshold`. (For
`e = 0` Python's `log` value is `-inf`, which never passes, as `0 < threshold` never does.)
Backs `evidence.py: ebh`. -/
theorem python_log_compare {e t : ℝ} (he : 0 < e) (ht : 0 < t) :
    Real.log t ≤ Real.log e ↔ t ≤ e :=
  Real.log_le_log_iff ht he

theorem kstar_eq_iff (x : Fin m → ℝ) (q : ℝ) (k : ℕ) :
    kstar x q = k ↔ (k = 0 ∨ (1 ≤ k ∧ k ≤ m ∧ k ≤ countAbove x (thr m q k))) ∧
      ∀ k' : ℕ, k < k' → k' ≤ m → countAbove x (thr m q k') < k' := by
  constructor
  · rintro rfl
    refine ⟨?_, fun k' hk' hk'm => ?_⟩
    · rcases Nat.eq_zero_or_pos (kstar x q) with h | h
      · exact Or.inl h
      · exact Or.inr ⟨h, kstar_le x q, kstar_spec h⟩
    · by_contra hc
      have := le_kstar (by omega) hk'm (not_lt.1 hc)
      omega
  · rintro ⟨h1, h2⟩
    apply le_antisymm
    · apply Finset.sup_le
      intro k' hk'
      simp only [mem_filter, mem_Icc] at hk'
      show k' ≤ k
      by_contra hlt
      have := h2 k' (not_le.1 hlt) hk'.1.2
      omega
    · rcases h1 with h0 | ⟨hk1, hkm, hc⟩
      · rw [h0]; exact Nat.zero_le _
      · exact le_kstar hk1 hkm hc

section Measurability

variable {Ω : Type*} [MeasurableSpace Ω]

theorem measurable_countAbove_ge (e : Ω → Fin m → ℝ) (he : ∀ j, Measurable fun ω => e ω j)
    (t : ℝ) (c : ℕ) : Measurable fun ω => c ≤ countAbove (e ω) t := by
  have hsum : Measurable fun ω => ∑ j, if t ≤ e ω j then (1 : ℝ) else 0 :=
    Finset.measurable_sum _ fun j _ =>
      Measurable.ite (measurableSet_le measurable_const (he j)) measurable_const measurable_const
  have h : (fun ω => c ≤ countAbove (e ω) t) =
      fun ω => (c : ℝ) ≤ ∑ j, if t ≤ e ω j then (1 : ℝ) else 0 := by
    funext ω
    apply propext
    unfold countAbove
    rw [← natCast_card_filter]
    exact Nat.cast_le.symm
  rw [h]
  exact measurableSet_setOfPred.1 (measurableSet_le measurable_const hsum)

theorem measurable_kstar (e : Ω → Fin m → ℝ) (he : ∀ j, Measurable fun ω => e ω j) (q : ℝ) :
    Measurable fun ω => kstar (e ω) q := by
  apply measurable_to_countable'
  intro k
  have h : (fun ω => kstar (e ω) q) ⁻¹' {k} =
      {ω | (k = 0 ∨ (1 ≤ k ∧ k ≤ m ∧ k ≤ countAbove (e ω) (thr m q k))) ∧
        ∀ k' : ℕ, k < k' → k' ≤ m → countAbove (e ω) (thr m q k') < k'} := by
    ext ω
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_ofPred_eq]
    exact kstar_eq_iff _ _ _
  rw [h]
  apply measurableSet_setOfPred.2
  refine Measurable.and (Measurable.or measurable_const (Measurable.and measurable_const
    (Measurable.and measurable_const (measurable_countAbove_ge e he _ _)))) ?_
  refine Measurable.forall fun k' => Measurable.imp measurable_const
    (Measurable.imp measurable_const ?_)
  have h2 : (fun ω => countAbove (e ω) (thr m q k') < k') =
      fun ω => ¬ (k' ≤ countAbove (e ω) (thr m q k')) := by
    funext ω; exact propext not_le.symm
  rw [h2]
  exact (measurable_countAbove_ge e he _ _).not

theorem measurableSet_mem_ebhSet (e : Ω → Fin m → ℝ) (he : ∀ j, Measurable fun ω => e ω j)
    (q : ℝ) (j : Fin m) : MeasurableSet {ω | j ∈ ebhSet (e ω) q} := by
  have hK := measurable_kstar e he q
  have h : {ω | j ∈ ebhSet (e ω) q} =
      {ω | 0 < kstar (e ω) q} ∩ {ω | thr m q (kstar (e ω) q) ≤ e ω j} := by
    ext ω; simp [ebhSet]
  rw [h]
  exact (hK (MeasurableSet.of_discrete (s := {k : ℕ | 0 < k}))).inter
    (measurableSet_le ((measurable_from_nat (f := thr m q)).comp hK) (he j))

/-- **The e-BH procedure, as computed, controls the false discovery rate under any
dependence.** Measurable nonnegative e-values; every true null `j ∈ H0` integrable with
`∫ e_j ≤ 1`; `q > 0`. Then `E[FDP(ebhSet)] ≤ q |H0| / m ≤ q`.

Backs `evidence.py: ebh` and `judge.py: analyze` (`ebh(log_e, Q, total=2*family)`, unseen
versions at `e = 0`). -/
theorem ebhSet_fdr (μ : Measure Ω) {q : ℝ} (hq : 0 < q)
    (e : Ω → Fin m → ℝ) (he : ∀ ω j, 0 ≤ e ω j) (hem : ∀ j, Measurable fun ω => e ω j)
    (H0 : Finset (Fin m))
    (hnull : ∀ j ∈ H0, Integrable (fun ω => e ω j) μ ∧ ∫ ω, e ω j ∂μ ≤ 1) :
    Integrable (fun ω => fdp (ebhSet (e ω) q) H0) μ ∧
      ∫ ω, fdp (ebhSet (e ω) q) H0 ∂μ ≤ q * H0.card / m ∧ q * H0.card / m ≤ q :=
  ebh_fdr μ hq e he H0 hnull (fun ω => ebhSet (e ω) q) (fun ω => ebhSet_selfConsistent hq (e ω))
    (fun j => measurableSet_mem_ebhSet e hem q j)

end Measurability

/-! ## 5. One strategy, `N` versions -/

/-- **One strategy, `N` versions: "up" is selected iff `e_up ≥ 2N/q`.**
The family has `m = 2N` hypotheses: "up" and "down" for the strategy shown, and the `2N - 2`
versions tried but not shown, all at `e = 0`. Assumptions: `e_down ≤ 1` (taken as a hypothesis
here; `EdgeOrLuck.Mixture.eMix_neg_le_one` proves it for the mixture e-value whenever the observed
sum is `≥ 0`), and `0 < q < 1` (weaker than the spec's `q ≤ 1/2`; the judge's `q` is 0.05).
`up` and `down` need not even be distinct for the statement to hold.

Backs `evidence.py` ("one strategy needs e >= 40 N") and `judge.py: analyze`
(`ebh(log_e, Q, total=2 * family)` with the keys `0#up`, `0#down`). -/
theorem single_strategy_rule {N : ℕ} {q : ℝ} (hq : 0 < q) (hq1 : q < 1)
    (e : Fin (2 * N) → ℝ) (up down : Fin (2 * N))
    (hrest : ∀ j, j ≠ up → j ≠ down → e j = 0) (hdown : e down ≤ 1) :
    up ∈ ebhSet e q ↔ 2 * (N : ℝ) / q ≤ e up := by
  have hm : 0 < 2 * N := up.pos
  have hthr1 : thr (2 * N) q 1 = 2 * (N : ℝ) / q := by unfold thr; push_cast; ring
  constructor
  · intro hmem
    simp only [ebhSet, mem_filter, mem_univ, true_and] at hmem
    obtain ⟨hk, hthr⟩ := hmem
    have hk1 : kstar e q = 1 := by
      by_contra hne
      have hspec := kstar_spec hk
      have hdown_in : thr (2 * N) q (kstar e q) ≤ e down := by
        by_contra hnot
        have hsub : (univ.filter fun j => thr (2 * N) q (kstar e q) ≤ e j) ⊆ {up} := by
          intro j hj
          simp only [mem_filter, mem_univ, true_and] at hj
          rw [mem_singleton]
          by_contra hjup
          by_cases hjd : j = down
          · subst hjd; exact hnot hj
          · have hpos : 0 < thr (2 * N) q (kstar e q) := by
              unfold thr
              have : (0 : ℝ) < kstar e q := by exact_mod_cast hk
              have : (0 : ℝ) < ((2 * N : ℕ) : ℝ) := by exact_mod_cast hm
              positivity
            rw [hrest j hjup hjd] at hj; linarith
        have := card_le_card hsub
        rw [card_singleton] at this
        unfold countAbove at hspec
        omega
      have hge := one_div_le_thr (m := 2 * N) hq hk (kstar_le e q)
      have : 1 < 1 / q := by rw [lt_div_iff₀ hq]; linarith
      linarith
    rw [hk1, hthr1] at hthr
    exact hthr
  · intro hup
    have hc : 1 ≤ countAbove e (thr (2 * N) q 1) := by
      unfold countAbove
      apply card_pos.2
      exact ⟨up, by simp only [mem_filter, mem_univ, true_and]; rw [hthr1]; exact hup⟩
    have hk := le_kstar (le_refl 1) hm hc
    rw [ebhSet_eq_filter (by omega)]
    simp only [mem_filter, mem_univ, true_and]
    calc thr (2 * N) q (kstar e q) ≤ thr (2 * N) q 1 := thr_anti hq one_pos hk
      _ = 2 * N / q := hthr1
      _ ≤ e up := hup

/-- **The same rule whatever the sign of the record.** It is enough that one of the two
directions has `e ≤ 1` (`EdgeOrLuck.Mixture.eMix_neg_le_one`: `e_down ≤ 1` when the sum is `≥ 0`,
and by symmetry `e_up ≤ 1` when it is `≤ 0`). If `e_up ≤ 1`, "up" is not selected and
`e_up < 2N/q`.
Backs `judge.py: analyze` and `evidence.py: ebh` for a single strategy. -/
theorem single_strategy_rule' {N : ℕ} {q : ℝ} (hq : 0 < q) (hq1 : q < 1)
    (e : Fin (2 * N) → ℝ) (up down : Fin (2 * N))
    (hrest : ∀ j, j ≠ up → j ≠ down → e j = 0) (hsmall : e up ≤ 1 ∨ e down ≤ 1) :
    up ∈ ebhSet e q ↔ 2 * (N : ℝ) / q ≤ e up := by
  rcases hsmall with hup | hdown
  · have hm : (1 : ℝ) ≤ N := by
      have := up.pos; exact_mod_cast (show 1 ≤ N by omega)
    have h1 : 1 < 1 / q := by rw [lt_div_iff₀ hq]; linarith
    have h2 : 1 / q ≤ 2 * (N : ℝ) / q := by
      apply div_le_div_of_nonneg_right _ hq.le; linarith
    constructor
    · intro hmem
      exact absurd hmem (notMem_ebhSet_of_lt hq e up (by linarith))
    · intro h; linarith
  · exact single_strategy_rule hq hq1 e up down hrest hdown

/-- **At the judge's `q = 1/20`: real edge iff `e ≥ 40 N`.** Backs `evidence.py: BASE_BAR * N`
(`describe_family: bar_alone`) and `judge.py: HOW_WE_KNOW` ("with N versions it needs
e >= 40 N"). (The code's `q` is the float `0.05`; the bridge tests check the float threshold.) -/
theorem single_strategy_rule_q05 {N : ℕ} (e : Fin (2 * N) → ℝ) (up down : Fin (2 * N))
    (hrest : ∀ j, j ≠ up → j ≠ down → e j = 0) (hsmall : e up ≤ 1 ∨ e down ≤ 1) :
    up ∈ ebhSet e (1 / 20) ↔ 40 * (N : ℝ) ≤ e up := by
  rw [single_strategy_rule' (by norm_num) (by norm_num) e up down hrest hsmall]
  have : 2 * (N : ℝ) / (1 / 20) = 40 * N := by ring
  rw [this]

/-- The bar arithmetic: `2 / (1/20) = 40` and `1/40` is half of `1/20`.
Backs `evidence.py: BASE_BAR = 2.0 / Q` and `ALPHA / 2`. -/
theorem base_bar_levels : (2 : ℝ) / (1 / 20) = 40 ∧ (1 : ℝ) / 40 = (1 / 20) / 2 := by norm_num

/-! ## 6. Brier is proper -/

/-- **Brier score of a forecast `p` on a 0/1 outcome with `P(Y = 1) = π`, as a weighted sum**:
`π (p - 1)² + (1 - π) p² = (p - π)² + π (1 - π)`. Backs `web/ledger.py: Ledger.observe`
(`"brier": (p - outcome) ** 2`). -/
theorem brier_bernoulli (π p : ℝ) :
    π * (p - 1) ^ 2 + (1 - π) * (p - 0) ^ 2 = (p - π) ^ 2 + π * (1 - π) := by ring

/-- **The true chance is the unique best forecast** (weighted-sum form): for `p ≠ π` the
expected Brier score is strictly larger than at `p = π`. Backs `web/ledger.py: Ledger.chosen_k`
(the shrinkage with the lowest mean Brier is used). -/
theorem brier_bernoulli_proper {π p : ℝ} (hp : p ≠ π) :
    π * (π - 1) ^ 2 + (1 - π) * (π - 0) ^ 2 < π * (p - 1) ^ 2 + (1 - π) * (p - 0) ^ 2 := by
  rw [brier_bernoulli, brier_bernoulli]
  have : 0 < (p - π) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (sub_ne_zero.2 hp)))
  nlinarith

section Brier

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Expected Brier score** of a fixed forecast `p` for a measurable outcome `Y ∈ {0, 1}` on a
probability space, with `π = P(Y = 1)`: `E[(p - Y)²] = (p - π)² + π (1 - π)`.
Backs `web/ledger.py: Ledger.observe` and `track_record`. -/
theorem brier_expected (μ : Measure Ω) [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hYm : Measurable Y)
    (hY : ∀ ω, Y ω = 0 ∨ Y ω = 1) (p : ℝ) :
    ∫ ω, (p - Y ω) ^ 2 ∂μ = (p - μ.real {ω | Y ω = 1}) ^ 2 +
      μ.real {ω | Y ω = 1} * (1 - μ.real {ω | Y ω = 1}) := by
  have hs : MeasurableSet {ω | Y ω = 1} := hYm (measurableSet_singleton 1)
  have hYeq : ∀ ω, Y ω = Set.indicator {ω | Y ω = 1} (1 : Ω → ℝ) ω := by
    intro ω
    rcases hY ω with h | h
    · rw [Set.indicator_of_notMem (by simp [h]), h]
    · rw [Set.indicator_of_mem (by simp [h]), h, Pi.one_apply]
  have hYint : Integrable Y μ := by
    have : Y = Set.indicator {ω | Y ω = 1} (1 : Ω → ℝ) := funext hYeq
    rw [this]
    exact (integrable_const (μ := μ) (1 : ℝ)).indicator hs
  have hYπ : ∫ ω, Y ω ∂μ = μ.real {ω | Y ω = 1} := by
    rw [integral_congr_ae (Filter.Eventually.of_forall hYeq), integral_indicator_one hs]
  have hpt : ∀ ω, (p - Y ω) ^ 2 = p ^ 2 + (1 - 2 * p) * Y ω := by
    intro ω
    rcases hY ω with h | h <;> rw [h] <;> ring
  rw [integral_congr_ae (Filter.Eventually.of_forall hpt),
    integral_add (integrable_const _) (hYint.const_mul _), integral_const_mul, hYπ]
  simp only [integral_const, probReal_univ, one_smul]
  ring

/-- **Brier is strictly proper**: forecasting the true chance `π = P(Y = 1)` scores strictly
better on average than any other fixed forecast. Backs `web/ledger.py: Ledger.chosen_k` (the
ledger keeps the shrinkage whose calls scored best by Brier). -/
theorem brier_proper (μ : Measure Ω) [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hYm : Measurable Y)
    (hY : ∀ ω, Y ω = 0 ∨ Y ω = 1) {p : ℝ} (hp : p ≠ μ.real {ω | Y ω = 1}) :
    ∫ ω, (μ.real {ω | Y ω = 1} - Y ω) ^ 2 ∂μ < ∫ ω, (p - Y ω) ^ 2 ∂μ := by
  rw [brier_expected μ hYm hY, brier_expected μ hYm hY, sub_self]
  have : 0 < (p - μ.real {ω | Y ω = 1}) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (sub_ne_zero.2 hp)))
  nlinarith

/-- **A coin flip scores exactly 1/4**, whatever the outcome's chance. Backs
`web/ledger.py: track_record` ("against 0.250 for a coin flip"; the `k = inf` forecast is 0.5). -/
theorem brier_coin_flip (μ : Measure Ω) [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hYm : Measurable Y)
    (hY : ∀ ω, Y ω = 0 ∨ Y ω = 1) : ∫ ω, (1 / 2 - Y ω) ^ 2 ∂μ = 1 / 4 := by
  rw [brier_expected μ hYm hY]; ring

/-- **Caveat for "better than guessing"**: a constant forecast equal to the base rate `π` beats
the coin flip's 0.25 whenever `π ≠ 1/2`, without any skill about individual records. So
`track_record`'s "Brier < 0.25: better than guessing" means "better than a coin flip", not
"better than the base rate". -/
theorem brier_base_rate_beats_coin (μ : Measure Ω) [IsProbabilityMeasure μ] {Y : Ω → ℝ}
    (hYm : Measurable Y) (hY : ∀ ω, Y ω = 0 ∨ Y ω = 1) (hπ : μ.real {ω | Y ω = 1} ≠ 1 / 2) :
    ∫ ω, (μ.real {ω | Y ω = 1} - Y ω) ^ 2 ∂μ < 1 / 4 := by
  rw [← brier_coin_flip μ hYm hY]
  exact brier_proper μ hYm hY (p := 1 / 2) (Ne.symm hπ)

end Brier

/-! ## 7. Money vs percent -/

theorem split_r (r : ℝ) : r = (if 0 < r then r else 0) - (if r < 0 then |r| else 0) := by
  split_ifs with h1 h2 h2
  · linarith
  · ring
  · rw [abs_of_neg h2]; ring
  · have : r = 0 := le_antisymm (not_lt.1 h1) (not_lt.1 h2)
    rw [this]; ring

theorem split_wr (w r : ℝ) :
    w * r = (if 0 < r then w * r else 0) - (if r < 0 then w * |r| else 0) := by
  split_ifs with h1 h2 h2
  · linarith
  · ring
  · rw [abs_of_neg h2]; ring
  · have : r = 0 := le_antisymm (not_lt.1 h1) (not_lt.1 h2)
    rw [this]; ring

theorem sum_split_r {ι : Type*} (s : Finset ι) (r : ι → ℝ) :
    ∑ i ∈ s, r i = (∑ i ∈ s.filter (fun i => 0 < r i), r i) -
      ∑ i ∈ s.filter (fun i => r i < 0), |r i| := by
  rw [sum_filter, sum_filter, ← sum_sub_distrib]
  exact sum_congr rfl fun i _ => split_r (r i)

theorem sum_split_wr {ι : Type*} (s : Finset ι) (w r : ι → ℝ) :
    ∑ i ∈ s, w i * r i = (∑ i ∈ s.filter (fun i => 0 < r i), w i * r i) -
      ∑ i ∈ s.filter (fun i => r i < 0), w i * |r i| := by
  rw [sum_filter, sum_filter, ← sum_sub_distrib]
  exact sum_congr rfl fun i _ => split_wr (w i) (r i)

/-- **"Your average trade is +x%, but you lost money: your losing trades were bigger than your
winning ones" — the exact sense in which it is true.** Trades `i ∈ s` with sizes `w_i > 0` and
percent results `r_i`, money `w_i r_i`. If `Σ r_i > 0` and `Σ w_i r_i < 0`, then there is a
losing trade AND a winning trade (the positive percent sum forces a winner, so the sentence
never talks about winners that do not exist), and the `|r|`-weighted average size of the losers
is strictly larger than the `r`-weighted average size of the winners:
`(Σ_{r<0} w|r|) / (Σ_{r<0} |r|) > (Σ_{r>0} w r) / (Σ_{r>0} r)`.

The model `money = size × percent` with `size > 0` is an assumption: it fails when the percent
column is before fees and the money after them (a trade can be up in percent and down in money).

Backs `judge.py: _money_card` (the `disagree and total < 0` sentence). -/
theorem losers_bigger {ι : Type*} (s : Finset ι) (w r : ι → ℝ) (hw : ∀ i ∈ s, 0 < w i)
    (hr : 0 < ∑ i ∈ s, r i) (hm : ∑ i ∈ s, w i * r i < 0) :
    (∃ i ∈ s, r i < 0) ∧ (∃ i ∈ s, 0 < r i) ∧
      (∑ i ∈ s.filter (fun i => 0 < r i), w i * r i) / (∑ i ∈ s.filter (fun i => 0 < r i), r i) <
      (∑ i ∈ s.filter (fun i => r i < 0), w i * |r i|) /
        (∑ i ∈ s.filter (fun i => r i < 0), |r i|) := by
  have hsr := sum_split_r s r
  have hswr := sum_split_wr s w r
  have hB0 : 0 ≤ ∑ i ∈ s.filter (fun i => r i < 0), |r i| := sum_nonneg fun i _ => abs_nonneg _
  have hWA0 : 0 ≤ ∑ i ∈ s.filter (fun i => 0 < r i), w i * r i := by
    refine sum_nonneg fun i hi => ?_
    rw [mem_filter] at hi
    exact (mul_pos (hw i hi.1) hi.2).le
  have hA : 0 < ∑ i ∈ s.filter (fun i => 0 < r i), r i := by linarith
  have hWB : 0 < ∑ i ∈ s.filter (fun i => r i < 0), w i * |r i| := by linarith
  have hLne : (s.filter (fun i => r i < 0)).Nonempty := by
    rw [nonempty_iff_ne_empty]
    intro h
    rw [h, sum_empty] at hWB
    exact lt_irrefl 0 hWB
  have hPne : (s.filter (fun i => 0 < r i)).Nonempty := by
    rw [nonempty_iff_ne_empty]
    intro h
    rw [h, sum_empty] at hA
    exact lt_irrefl 0 hA
  have hB : 0 < ∑ i ∈ s.filter (fun i => r i < 0), |r i| := by
    refine sum_pos (fun i hi => ?_) hLne
    rw [mem_filter] at hi
    exact abs_pos.2 hi.2.ne
  refine ⟨?_, ?_, ?_⟩
  · obtain ⟨i, hi⟩ := hLne
    rw [mem_filter] at hi
    exact ⟨i, hi.1, hi.2⟩
  · obtain ⟨i, hi⟩ := hPne
    rw [mem_filter] at hi
    exact ⟨i, hi.1, hi.2⟩
  · rw [div_lt_div_iff₀ hA hB]
    have h1 : ∑ i ∈ s.filter (fun i => 0 < r i), w i * r i <
        ∑ i ∈ s.filter (fun i => r i < 0), w i * |r i| := by linarith
    have h2 : ∑ i ∈ s.filter (fun i => r i < 0), |r i| <
        ∑ i ∈ s.filter (fun i => 0 < r i), r i := by linarith
    nlinarith [mul_pos (sub_pos.2 h1) hB, mul_pos hWB (sub_pos.2 h2)]

/-- **The mirror sentence: "... but you made money: your winning trades were the bigger
ones."** If `Σ r_i < 0` and `Σ w_i r_i > 0` (sizes `w_i > 0`), there is a winner and a loser, and
the `r`-weighted average size of the winners strictly exceeds the `|r|`-weighted average size of
the losers. Backs `judge.py: _money_card` (the `disagree` sentences with a positive total and the
percent-unit `money_shadow` branch). -/
theorem winners_bigger {ι : Type*} (s : Finset ι) (w r : ι → ℝ) (hw : ∀ i ∈ s, 0 < w i)
    (hr : ∑ i ∈ s, r i < 0) (hm : 0 < ∑ i ∈ s, w i * r i) :
    (∃ i ∈ s, 0 < r i) ∧ (∃ i ∈ s, r i < 0) ∧
      (∑ i ∈ s.filter (fun i => r i < 0), w i * |r i|) /
          (∑ i ∈ s.filter (fun i => r i < 0), |r i|) <
        (∑ i ∈ s.filter (fun i => 0 < r i), w i * r i) /
          (∑ i ∈ s.filter (fun i => 0 < r i), r i) := by
  have hsr := sum_split_r s r
  have hswr := sum_split_wr s w r
  have hA0 : 0 ≤ ∑ i ∈ s.filter (fun i => 0 < r i), r i := by
    refine sum_nonneg fun i hi => ?_
    rw [mem_filter] at hi; exact hi.2.le
  have hWB0 : 0 ≤ ∑ i ∈ s.filter (fun i => r i < 0), w i * |r i| := by
    refine sum_nonneg fun i hi => ?_
    rw [mem_filter] at hi
    exact mul_nonneg (hw i hi.1).le (abs_nonneg _)
  have hB : 0 < ∑ i ∈ s.filter (fun i => r i < 0), |r i| := by linarith
  have hWA : 0 < ∑ i ∈ s.filter (fun i => 0 < r i), w i * r i := by linarith
  have hPne : (s.filter (fun i => 0 < r i)).Nonempty := by
    rw [nonempty_iff_ne_empty]
    intro h
    rw [h, sum_empty] at hWA
    exact lt_irrefl 0 hWA
  have hLne : (s.filter (fun i => r i < 0)).Nonempty := by
    rw [nonempty_iff_ne_empty]
    intro h
    rw [h, sum_empty] at hB
    exact lt_irrefl 0 hB
  have hA : 0 < ∑ i ∈ s.filter (fun i => 0 < r i), r i := by
    refine sum_pos (fun i hi => ?_) hPne
    rw [mem_filter] at hi; exact hi.2
  refine ⟨?_, ?_, ?_⟩
  · obtain ⟨i, hi⟩ := hPne
    rw [mem_filter] at hi
    exact ⟨i, hi.1, hi.2⟩
  · obtain ⟨i, hi⟩ := hLne
    rw [mem_filter] at hi
    exact ⟨i, hi.1, hi.2⟩
  · rw [div_lt_div_iff₀ hB hA]
    have h1 : ∑ i ∈ s.filter (fun i => r i < 0), w i * |r i| <
        ∑ i ∈ s.filter (fun i => 0 < r i), w i * r i := by linarith
    have h2 : ∑ i ∈ s.filter (fun i => 0 < r i), r i <
        ∑ i ∈ s.filter (fun i => r i < 0), |r i| := by linarith
    nlinarith [mul_pos (sub_pos.2 h1) hA, mul_pos hWA (sub_pos.2 h2)]

/-- **"Bigger" does not mean "larger positions on average".** Three trades, sizes 1, 1000, 10,
percent +10, +0.01, -5: the average percent is positive, the money is negative, yet the losing
trade's position (10) is smaller than the winners' average position (500.5). So
`_money_card`'s sentence is only true in the weighted sense of `losers_bigger`. -/
theorem plain_average_size_can_disagree :
    ∃ w r : Fin 3 → ℝ, (∀ i, 0 < w i) ∧ 0 < r 0 ∧ 0 < r 1 ∧ r 2 < 0 ∧
      0 < ∑ i, r i ∧ ∑ i, w i * r i < 0 ∧ w 2 < (w 0 + w 1) / 2 := by
  refine ⟨![1, 1000, 10], ![10, 1 / 100, -5], ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  all_goals simp [Fin.sum_univ_three]
  all_goals norm_num

/-- **"Bigger" does not mean "lost more per trade than a winner made".** One winner (size 1,
+50%) and two losers (size 30, -1% each): the average percent is positive and the money negative,
yet each loser lost 0.3 while the winner made 0.5 (in units of size × percent / 100). -/
theorem money_per_trade_can_disagree :
    ∃ w r : Fin 3 → ℝ, (∀ i, 0 < w i) ∧ 0 < r 0 ∧ r 1 < 0 ∧ r 2 < 0 ∧
      0 < ∑ i, r i ∧ ∑ i, w i * r i < 0 ∧ (w 1 * |r 1| + w 2 * |r 2|) / 2 < w 0 * r 0 := by
  refine ⟨![1, 30, 30], ![50, -1, -1], ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  all_goals simp [Fin.sum_univ_three]
  all_goals norm_num

/-! ## 8. Wilson -/

/-- `evidence.py: wilson_bounds_count`'s `centre = (p + z2 / (2 n)) / denom`. -/
noncomputable def wilsonCentre (phat z n : ℝ) : ℝ := (phat + z ^ 2 / (2 * n)) / (1 + z ^ 2 / n)

/-- `evidence.py: wilson_bounds_count`'s
`spread = z * sqrt((p (1 - p) + z2 / (4 n)) / n) / denom`. -/
noncomputable def wilsonSpread (phat z n : ℝ) : ℝ :=
  z * Real.sqrt ((phat * (1 - phat) + z ^ 2 / (4 * n)) / n) / (1 + z ^ 2 / n)

/-- **The Wilson ends are exactly the roots of the score equation.** For `n > 0` and
`p̂ ∈ [0, 1]` (the code clips `wins / n` to `[0, 1]`), and any `z`:
`(p̂ - p)² = z² p (1 - p) / n` iff `p = centre - spread` or `p = centre + spread`.
Backs `evidence.py: wilson_bounds_count` (before its clipping to `[0, 1]`). -/
theorem wilson_roots {phat z n : ℝ} (hn : 0 < n) (hp0 : 0 ≤ phat) (hp1 : phat ≤ 1) (p : ℝ) :
    (phat - p) ^ 2 = z ^ 2 * p * (1 - p) / n ↔
      p = wilsonCentre phat z n - wilsonSpread phat z n ∨
        p = wilsonCentre phat z n + wilsonSpread phat z n := by
  set D := (phat * (1 - phat) + z ^ 2 / (4 * n)) / n with hDdef
  have hD : 0 ≤ D := by
    apply div_nonneg _ hn.le
    have : 0 ≤ phat * (1 - phat) := mul_nonneg hp0 (by linarith)
    positivity
  set S := Real.sqrt D with hSdef
  have hS : S * S = D := Real.mul_self_sqrt hD
  have ha : (1 + z ^ 2 / n) ≠ 0 := by positivity
  have hdisc : discrim (1 + z ^ 2 / n) (-(2 * phat + z ^ 2 / n)) (phat ^ 2) =
      (2 * z * S) * (2 * z * S) := by
    unfold discrim
    have : (2 * z * S) * (2 * z * S) = 4 * z ^ 2 * (S * S) := by ring
    rw [this, hS, hDdef]
    field_simp
    ring
  have heq : (phat - p) ^ 2 = z ^ 2 * p * (1 - p) / n ↔
      (1 + z ^ 2 / n) * (p * p) + (-(2 * phat + z ^ 2 / n)) * p + phat ^ 2 = 0 := by
    constructor
    · intro h
      have : (1 + z ^ 2 / n) * (p * p) + (-(2 * phat + z ^ 2 / n)) * p + phat ^ 2 =
          (phat - p) ^ 2 - z ^ 2 * p * (1 - p) / n := by ring
      rw [this, h, sub_self]
    · intro h
      have : (phat - p) ^ 2 - z ^ 2 * p * (1 - p) / n = 0 := by
        rw [← h]; ring
      linarith
  rw [heq, quadratic_eq_zero_iff ha hdisc p]
  have hplus : (-(-(2 * phat + z ^ 2 / n)) + 2 * z * S) / (2 * (1 + z ^ 2 / n)) =
      wilsonCentre phat z n + wilsonSpread phat z n := by
    unfold wilsonCentre wilsonSpread
    rw [← hSdef]
    field_simp
  have hminus : (-(-(2 * phat + z ^ 2 / n)) - 2 * z * S) / (2 * (1 + z ^ 2 / n)) =
      wilsonCentre phat z n - wilsonSpread phat z n := by
    unfold wilsonCentre wilsonSpread
    rw [← hSdef]
    field_simp
  rw [hplus, hminus, or_comm]

/-- Any root of the score equation with `z ≠ 0` lies in `[0, 1]`. -/
theorem wilson_root_mem_unit {phat z n p : ℝ} (hn : 0 < n) (hz : z ≠ 0)
    (h : (phat - p) ^ 2 = z ^ 2 * p * (1 - p) / n) : 0 ≤ p ∧ p ≤ 1 := by
  have h1 : 0 ≤ z ^ 2 * p * (1 - p) / n := h ▸ sq_nonneg _
  have h2 : 0 ≤ z ^ 2 * (p * (1 - p)) := by
    rw [le_div_iff₀ hn, zero_mul] at h1; linarith [h1]
  have hz2 : 0 < z ^ 2 := by positivity
  have h3 : 0 ≤ p * (1 - p) := (mul_nonneg_iff_of_pos_left hz2).1 h2
  constructor <;> nlinarith

/-- **The clipping to `[0, 1]` never changes the Wilson ends**: for `n > 0`, `p̂ ∈ [0, 1]` and any
`z`, both ends already lie in `[0, 1]`. Backs `evidence.py: wilson_bounds_count`
(`max(0.0, ...)`, `min(1.0, ...)` act only on float rounding). -/
theorem wilson_ends_mem_unit {phat z n : ℝ} (hn : 0 < n) (hp0 : 0 ≤ phat) (hp1 : phat ≤ 1) :
    (0 ≤ wilsonCentre phat z n - wilsonSpread phat z n ∧
        wilsonCentre phat z n - wilsonSpread phat z n ≤ 1) ∧
      (0 ≤ wilsonCentre phat z n + wilsonSpread phat z n ∧
        wilsonCentre phat z n + wilsonSpread phat z n ≤ 1) := by
  by_cases hz : z = 0
  · subst hz
    have hc : wilsonCentre phat 0 n = phat := by unfold wilsonCentre; simp
    have hs : wilsonSpread phat 0 n = 0 := by unfold wilsonSpread; simp
    rw [hc, hs]
    exact ⟨⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩⟩
  · exact ⟨wilson_root_mem_unit hn hz ((wilson_roots hn hp0 hp1 _).2 (Or.inl rfl)),
      wilson_root_mem_unit hn hz ((wilson_roots hn hp0 hp1 _).2 (Or.inr rfl))⟩

/-- **The Wilson interval is the score-test inversion**: for `n > 0`, `z ≥ 0`, `p̂ ∈ [0, 1]`,
a chance `p` is not rejected by the score test, `(p̂ - p)² ≤ z² p (1 - p) / n`, exactly when
`centre - spread ≤ p ≤ centre + spread`. Backs `evidence.py: wilson_bounds_count` (Wilson 1927)
and the win-rate range in `judge.py`. -/
theorem wilson_score_inversion {phat z n : ℝ} (hn : 0 < n) (hz : 0 ≤ z) (hp0 : 0 ≤ phat)
    (hp1 : phat ≤ 1) (p : ℝ) :
    (phat - p) ^ 2 ≤ z ^ 2 * p * (1 - p) / n ↔
      wilsonCentre phat z n - wilsonSpread phat z n ≤ p ∧
        p ≤ wilsonCentre phat z n + wilsonSpread phat z n := by
  set D := (phat * (1 - phat) + z ^ 2 / (4 * n)) / n with hDdef
  have hD : 0 ≤ D := by
    apply div_nonneg _ hn.le
    have : 0 ≤ phat * (1 - phat) := mul_nonneg hp0 (by linarith)
    positivity
  have ha : 0 < 1 + z ^ 2 / n := by positivity
  have hs2 : wilsonSpread phat z n ^ 2 = z ^ 2 * D / (1 + z ^ 2 / n) ^ 2 := by
    unfold wilsonSpread
    rw [div_pow, mul_pow, Real.sq_sqrt hD]
  have key : (phat - p) ^ 2 - z ^ 2 * p * (1 - p) / n =
      (1 + z ^ 2 / n) * ((p - (wilsonCentre phat z n - wilsonSpread phat z n)) *
        (p - (wilsonCentre phat z n + wilsonSpread phat z n))) := by
    have hfac : (p - (wilsonCentre phat z n - wilsonSpread phat z n)) *
        (p - (wilsonCentre phat z n + wilsonSpread phat z n)) =
        (p - wilsonCentre phat z n) ^ 2 - wilsonSpread phat z n ^ 2 := by ring
    rw [hfac, hs2, hDdef]
    unfold wilsonCentre
    field_simp
    ring
  have hs0 : 0 ≤ wilsonSpread phat z n := by unfold wilsonSpread; positivity
  constructor
  · intro h
    have hle : (1 + z ^ 2 / n) * ((p - (wilsonCentre phat z n - wilsonSpread phat z n)) *
        (p - (wilsonCentre phat z n + wilsonSpread phat z n))) ≤ 0 := by rw [← key]; linarith
    have hprod : (p - (wilsonCentre phat z n - wilsonSpread phat z n)) *
        (p - (wilsonCentre phat z n + wilsonSpread phat z n)) ≤ 0 := by
      by_contra hpos
      push Not at hpos
      have := mul_pos ha hpos
      linarith
    constructor
    · by_contra hlt
      push Not at hlt
      have h1 : p - (wilsonCentre phat z n - wilsonSpread phat z n) < 0 := by linarith
      have h2 : p - (wilsonCentre phat z n + wilsonSpread phat z n) < 0 := by linarith
      have := mul_pos_of_neg_of_neg h1 h2
      linarith
    · by_contra hlt
      push Not at hlt
      have h1 : 0 < p - (wilsonCentre phat z n - wilsonSpread phat z n) := by linarith
      have h2 : 0 < p - (wilsonCentre phat z n + wilsonSpread phat z n) := by linarith
      have := mul_pos h1 h2
      linarith
  · rintro ⟨h1, h2⟩
    have hprod : (p - (wilsonCentre phat z n - wilsonSpread phat z n)) *
        (p - (wilsonCentre phat z n + wilsonSpread phat z n)) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    have : (phat - p) ^ 2 - z ^ 2 * p * (1 - p) / n ≤ 0 := by
      rw [key]; exact mul_nonpos_of_nonneg_of_nonpos ha.le hprod
    linarith

/-! ## 9. Compare's range -/

section Compare

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **The difference range (deterministic).** If the live mean `μ_l` is in
`[m_l - h_l, m_l + h_l]` and the backtest mean `μ_b` in `[m_b - h_b, m_b + h_b]`, then
`μ_l - μ_b ∈ [d - h_l - h_b, d + h_l + h_b]` with `d = m_l - m_b`.
Backs `compare.py: compare_runs` (`lo, hi = diff - h_live - h_bt, diff + h_live + h_bt`). -/
theorem diff_range {μl μb ml mb hl hb : ℝ}
    (hlive : μl ∈ Set.Icc (ml - hl) (ml + hl)) (hbt : μb ∈ Set.Icc (mb - hb) (mb + hb)) :
    μl - μb ∈ Set.Icc (ml - mb - hl - hb) (ml - mb + hl + hb) := by
  obtain ⟨h1, h2⟩ := hlive
  obtain ⟨h3, h4⟩ := hbt
  constructor <;> linarith

/-- **The union bound, with the live side checked at every time `t`.** If the live range misses
`μ_l` at SOME time with probability `≤ a` (anytime-valid), and the backtest range misses `μ_b`
with probability `≤ b`, then the difference range misses `μ_l - μ_b` at some time with
probability `≤ a + b`. No independence and no measurability are needed (outer measure).
With `a = b = 2.5%` this is the 95% of `compare.py: compare_runs` (`range_95`). -/
theorem diff_range_union_bound (μ : Measure Ω) (μl μb : ℝ) (ml hl : ℕ → Ω → ℝ) (mb hb : Ω → ℝ)
    {a b : ENNReal}
    (hlive : μ {ω | ∃ t, μl ∉ Set.Icc (ml t ω - hl t ω) (ml t ω + hl t ω)} ≤ a)
    (hbt : μ {ω | μb ∉ Set.Icc (mb ω - hb ω) (mb ω + hb ω)} ≤ b) :
    μ {ω | ∃ t, μl - μb ∉ Set.Icc (ml t ω - mb ω - hl t ω - hb ω)
      (ml t ω - mb ω + hl t ω + hb ω)} ≤ a + b := by
  calc μ {ω | ∃ t, μl - μb ∉ Set.Icc (ml t ω - mb ω - hl t ω - hb ω)
          (ml t ω - mb ω + hl t ω + hb ω)}
      ≤ μ ({ω | ∃ t, μl ∉ Set.Icc (ml t ω - hl t ω) (ml t ω + hl t ω)} ∪
          {ω | μb ∉ Set.Icc (mb ω - hb ω) (mb ω + hb ω)}) := by
        apply measure_mono
        rintro ω ⟨t, ht⟩
        by_contra hc
        rw [Set.mem_union, not_or] at hc
        simp only [Set.mem_ofPred_eq, not_exists, not_not] at hc
        exact ht (diff_range (hc.1 t) hc.2)
    _ ≤ _ := measure_union_le _ _
    _ ≤ a + b := add_le_add hlive hbt

/-- **The false "worse than the backtest" rate.** When live is in fact as good as the backtest
(`μ_b ≤ μ_l`), the verdict `hi < 0` ever happening needs the live range's upper end to fall
below `μ_l` at some time, or the backtest range's lower end to rise above `μ_b`: one-sided
misses, 1.25% + 1.25% in the code. Backs `compare.py: compare_runs` (`if hi < 0:
worse_than_backtest`). -/
theorem worse_false_alarm (μ : Measure Ω) {μl μb : ℝ} (hge : μb ≤ μl) (ml hl : ℕ → Ω → ℝ)
    (mb hb : Ω → ℝ) :
    μ {ω | ∃ t, ml t ω - mb ω + hl t ω + hb ω < 0} ≤
      μ {ω | ∃ t, ml t ω + hl t ω < μl} + μ {ω | μb < mb ω - hb ω} := by
  calc μ {ω | ∃ t, ml t ω - mb ω + hl t ω + hb ω < 0}
      ≤ μ ({ω | ∃ t, ml t ω + hl t ω < μl} ∪ {ω | μb < mb ω - hb ω}) := by
        apply measure_mono
        rintro ω ⟨t, ht⟩
        by_contra hc
        rw [Set.mem_union, not_or] at hc
        simp only [Set.mem_ofPred_eq, not_exists, not_lt] at hc
        have := hc.1 t
        linarith [hc.2]
    _ ≤ _ := measure_union_le _ _

/-- **The levels add up**: the live side (bar 80) misses with chance `2/80 = 2.5%` (a 97.5%
range), the backtest side (`z` at 1.25% per tail) with `2 × 1.25% = 2.5%`, together `5%`: a 95%
range. Backs `compare.py: BAR = 80.0`, `Z_BACKTEST = 2.2414` (its normal quantile is checked
numerically in the bridge tests) and the key `range_95`. -/
theorem compare_levels :
    (1 : ℝ) - 2 / 80 = 0.975 ∧ (2 : ℝ) / 80 + 2 * 0.0125 = 0.05 ∧ (1 : ℝ) - 0.05 = 0.95 := by
  norm_num

end Compare

end EdgeOrLuck.Selection

/-! ## Axioms used: only `propext`, `Classical.choice` and `Quot.sound` may appear. -/

open EdgeOrLuck.Selection

#print axioms evalue_markov
#print axioms evalue_markov_measure
#print axioms average_isEValue
#print axioms max_div_le_average
#print axioms best_div_isEValue
#print axioms best_of_N_markov
#print axioms indicator_div_le
#print axioms fdp_le_sum
#print axioms ebh_fdr_lintegral
#print axioms ebh_fdr
#print axioms card_ebhSet
#print axioms ebhSet_selfConsistent
#print axioms notMem_ebhSet_of_lt
#print axioms notMem_ebhSet_of_nonpos
#print axioms pyChosen_eq_kstar
#print axioms pyTop_eq_ebhSet
#print axioms ebh_no_tie_at_cut
#print axioms pyChosen_listed
#print axioms python_log_compare
#print axioms ebhSet_fdr
#print axioms single_strategy_rule
#print axioms single_strategy_rule'
#print axioms single_strategy_rule_q05
#print axioms base_bar_levels
#print axioms brier_bernoulli
#print axioms brier_bernoulli_proper
#print axioms brier_expected
#print axioms brier_proper
#print axioms brier_coin_flip
#print axioms brier_base_rate_beats_coin
#print axioms losers_bigger
#print axioms winners_bigger
#print axioms plain_average_size_can_disagree
#print axioms money_per_trade_can_disagree
#print axioms wilson_roots
#print axioms wilson_ends_mem_unit
#print axioms wilson_score_inversion
#print axioms diff_range
#print axioms diff_range_union_bound
#print axioms worse_false_alarm
#print axioms compare_levels
