import Mathlib.Probability.Martingale.OptionalStopping
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Independence.Integration

/-!
# Checking again and again doesn't fool it: the time-uniform guarantee

edge or luck, `src/edgeorluck/evidence.py`. The page (`web/static/index.html`) says: "The range
shown for your average trade is an anytime-valid confidence sequence (Waudby-Smith and Ramdas). It
stays honest even if you check after every trade, so watching it does not fool you." This file
proves that sentence for the exact Gaussian model with a known spread.

The code's one-sided e-value (`log_e_standardized(n, s, rho2)`, `log_e_one_sided`) is the
logarithm of a half-normal mixture of Gaussian likelihood ratios,

  `e_n = ∫_{λ ≥ 0} exp (λ s - λ² n / 2) π(dλ)`,   `s` = the standardized running sum,

with `π = |N(0, rho2)|` and `rho2 = mixing_rho2(bar)`. "Above zero" is called when `e_n ≥ B`
(`B = BASE_BAR = 40` for one strategy), and the range (`range_of`, `half_width`) holds every mean
that neither one-sided e-value rejects: a `1 - 2 / B` range (95% at `B = 40`). That the closed form
in the code equals this integral, and that it grows with `s`, is another file's job (the mixture
e-value's closed form and monotonicity); here the e-value IS the integral (`mixG`, `eValue`).

## Results (machine-checked; no unfinished proofs, no new axioms)

1. `ville_super`, `ville`, `ville_martingale`: **Ville's inequality.** A nonnegative
   supermartingale with `E[M₀] ≤ 1` ever reaches `c > 0` with probability at most `1 / c`.
   Doob's argument (stop at the first crossing, optional stopping) for each finite horizon
   (`maximal_ineq_super`), then continuity of the measure from below.
2. `setIntegral_lr_succ`, `martingale_lr`, `supermartingale_lr`: for independent Gaussian trades
   `N(θ, 1)` the likelihood ratio `L_n(λ) = exp (λ S_n - λ² n / 2)` gains exactly the factor
   `exp (λ θ)` per trade on every event of the past: a martingale at `θ = 0`
   (`E[L₀] = 1`), a supermartingale when `λ θ ≤ 0`.
3. `martingale_mix`, `supermartingale_mix`: the mixture `M_n = ∫ L_n(λ) π(dλ)` over ANY
   probability measure `π` on `ℝ` is a nonnegative martingale with `M₀ = 1` when `θ = 0`, and a
   supermartingale when `λ θ ≤ 0` for `π`-almost every `λ` (the code's half-normal has `λ ≥ 0`,
   so this is every mean `θ ≤ 0`: no edge, or a losing strategy). Tonelli/Fubini, no conditional
   Fubini needed.
4. `evalue_ville`, `two_sided_ville`, `anytime_range`, `anytime_range_95`,
   `above_zero_false_alarm`: the e-value ever reaches `B` with probability `≤ 1 / B`; either
   one-sided e-value does with probability `≤ 2 / B` (union bound); so the range misses the true
   mean at SOME trade count, however often it is checked, with probability `≤ 2 / B` (5% at
   `B = 40`), for every true mean `θ` and known spread `σ > 0` (standardize `(X_i - θ) / σ`).

## What this does NOT cover

* The code plugs in the sample spread (`record_of`: the sample SD, widened for serial
  correlation). Here `σ` is known. With a plug-in spread the guarantee is asymptotic only.
* Trades that are not Gaussian (heavy tails, skew) or not independent (autocorrelation). Those
  are measured by simulation (VALIDATION.md, "checked every 10 trades"), not proved.
* The skew-raised bar and the five-losses rule (both only make a call harder), the e-BH family
  across versions, and the closed form and monotonicity of the e-value: other files.
* Floating point: every statement is about real numbers.
-/

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal NNReal

namespace EdgeOrLuck.Anytime

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

/-! ## 1. Ville's inequality -/

section Ville

variable {ℱ : Filtration ℕ mΩ} {M : ℕ → Ω → ℝ}

/-- Doob's maximal inequality for a NONNEGATIVE SUPERMARTINGALE, finite horizon `N`:
`c · P(M_k ≥ c for some k ≤ N) ≤ E[M₀]`.
Proof: stop at the first `k ≤ N` with `M_k ≥ c` (else at `N`). On the event the stopped value is
`≥ c`, everywhere it is `≥ 0`, and optional stopping for the supermartingale (Mathlib's
`Submartingale.expected_stoppedValue_mono` applied to `-M`) gives `E[M_τ] ≤ E[M₀]`.
Backs: the finite-horizon step of `ville_super`. -/
theorem maximal_ineq_super [IsFiniteMeasure μ] (hM : Supermartingale M ℱ μ)
    (hnn : ∀ n ω, 0 ≤ M n ω) (c : ℝ) (N : ℕ) :
    c * μ.real {ω | ∃ k ≤ N, c ≤ M k ω} ≤ ∫ ω, M 0 ω ∂μ := by
  have hτ : IsStoppingTime ℱ (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ)) :=
    hM.stronglyAdapted.adapted.isStoppingTime_hittingBtwn measurableSet_Ici
  have hτN : ∀ ω, ((hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ) : WithTop ℕ) ≤ N :=
    fun ω ↦ mod_cast hittingBtwn_le ω
  have hint : Integrable
      (stoppedValue M (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ))) μ :=
    integrable_stoppedValue ℕ hτ hM.integrable hτN
  have hmeas : ∀ k, Measurable (M k) := fun k ↦
    (hM.stronglyMeasurable k).measurable.le (ℱ.le k)
  have hA : MeasurableSet {ω | ∃ k ≤ N, c ≤ M k ω} := by
    have : {ω | ∃ k ≤ N, c ≤ M k ω} = ⋃ k ∈ Finset.range (N + 1), {ω | c ≤ M k ω} := by
      ext ω; simp
    rw [this]
    exact Finset.measurableSet_biUnion _ fun k _ ↦ measurableSet_le measurable_const (hmeas k)
  have hge : ∀ ω ∈ {ω | ∃ k ≤ N, c ≤ M k ω},
      c ≤ stoppedValue M (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ)) ω := by
    rintro ω ⟨k, hk, hck⟩
    exact stoppedValue_hittingBtwn_mem ⟨k, ⟨Nat.zero_le _, hk⟩, hck⟩
  have h1 := setIntegral_ge_of_const_le_real hA (measure_ne_top _ _) hge hint.integrableOn
  have h2 : ∫ ω in {ω | ∃ k ≤ N, c ≤ M k ω},
      stoppedValue M (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ)) ω ∂μ ≤
      ∫ ω, stoppedValue M (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ)) ω ∂μ :=
    setIntegral_le_integral hint (ae_of_all _ fun ω ↦ hnn _ _)
  have h3 : ∫ ω, stoppedValue M (fun ω ↦ (hittingBtwn M {y : ℝ | c ≤ y} 0 N ω : ℕ)) ω ∂μ ≤
      ∫ ω, M 0 ω ∂μ := by
    have := hM.neg.expected_stoppedValue_mono (isStoppingTime_const ℱ 0) hτ
      (fun ω ↦ by simp) hτN
    rw [stoppedValue_neg, stoppedValue_neg] at this
    simp only [Pi.neg_apply, integral_neg, neg_le_neg_iff] at this
    simpa [stoppedValue] using this
  linarith

/-- **Ville's inequality** (supermartingale form). A nonnegative supermartingale ever reaches a
level `c > 0`, at ANY time, with probability at most `E[M₀] / c`:
`P(∃ n, M_n ≥ c) ≤ E[M₀] / c`.
Assumptions: `μ` finite, `M` a supermartingale for some filtration, `M_n ≥ 0` everywhere.
Backs: "It stays honest even if you check after every trade" (`web/static/index.html`); a
supermartingale also covers a composite null such as "the mean is at most 0". -/
theorem ville_super [IsFiniteMeasure μ] (hM : Supermartingale M ℱ μ) (hnn : ∀ n ω, 0 ≤ M n ω)
    {c : ℝ} (hc : 0 < c) :
    μ {ω | ∃ n, c ≤ M n ω} ≤ ENNReal.ofReal ((∫ ω, M 0 ω ∂μ) / c) := by
  have hmono : Monotone fun N : ℕ ↦ {ω | ∃ k ≤ N, c ≤ M k ω} :=
    fun a b hab ω ⟨k, hk, h⟩ ↦ ⟨k, hk.trans hab, h⟩
  have hU : {ω | ∃ n, c ≤ M n ω} = ⋃ N, {ω | ∃ k ≤ N, c ≤ M k ω} := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion]
    exact ⟨fun ⟨n, h⟩ ↦ ⟨n, n, le_rfl, h⟩, fun ⟨_, k, _, h⟩ ↦ ⟨k, h⟩⟩
  rw [hU, hmono.measure_iUnion]
  refine iSup_le fun N ↦ ?_
  have h := maximal_ineq_super hM hnn c N
  rw [measureReal_def] at h
  rw [← ENNReal.ofReal_toReal (measure_ne_top μ _)]
  refine ENNReal.ofReal_le_ofReal ?_
  rw [le_div_iff₀ hc, mul_comm]
  exact h

/-- **Ville's inequality**: a nonnegative supermartingale with `E[M₀] ≤ 1` ever reaches `c > 0`
with probability at most `1 / c`. Assumptions: `μ` finite, `M_n ≥ 0`, `E[M₀] ≤ 1`.
Backs: the e-value bar `B` (`BASE_BAR = 40`): an e-value process that is a nonnegative
supermartingale under the null ever reaches `B` with probability `≤ 1 / B`. -/
theorem ville [IsFiniteMeasure μ] (hM : Supermartingale M ℱ μ) (hnn : ∀ n ω, 0 ≤ M n ω)
    (h0 : ∫ ω, M 0 ω ∂μ ≤ 1) {c : ℝ} (hc : 0 < c) :
    μ {ω | ∃ n, c ≤ M n ω} ≤ ENNReal.ofReal (1 / c) :=
  (ville_super hM hnn hc).trans (ENNReal.ofReal_le_ofReal (by gcongr))

/-- **Ville's inequality for nonnegative martingales**: if `M` is a nonnegative martingale with
`E[M₀] ≤ 1`, then `P(∃ n, M_n ≥ c) ≤ 1 / c` for every `c > 0`. (A martingale is a
supermartingale, so this is `ville`.) -/
theorem ville_martingale [IsFiniteMeasure μ] (hM : Martingale M ℱ μ) (hnn : ∀ n ω, 0 ≤ M n ω)
    (h0 : ∫ ω, M 0 ω ∂μ ≤ 1) {c : ℝ} (hc : 0 < c) :
    μ {ω | ∃ n, c ≤ M n ω} ≤ ENNReal.ofReal (1 / c) :=
  ville hM.supermartingale hnn h0 hc

end Ville

/-! ## 2. The likelihood ratio of Gaussian trades -/

/-- The running sum after `n` trades, `S_n = X_0 + ⋯ + X_{n-1}` (`S_0 = 0`). With
standardized trades it is the code's `s = total / sd`. -/
def runSum (X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ := ∑ i ∈ range n, X i ω

/-- The Gaussian likelihood ratio of "mean `λ`" against "mean 0" after `n` standardized
trades: `L_n(λ) = exp (λ S_n - λ² n / 2)`. -/
noncomputable def lr (X : ℕ → Ω → ℝ) (l : ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  Real.exp (l * runSum X n ω - l ^ 2 * n / 2)

/-- What is known after `n` trades: `ℱ_n = σ(X_0, …, X_{n-1})` (`ℱ_0` trivial). -/
def past (X : ℕ → Ω → ℝ) (hX : ∀ i, Measurable (X i)) : Filtration ℕ mΩ where
  seq n := ⨆ i ∈ Set.Iio n, MeasurableSpace.comap (X i) inferInstance
  mono' _ _ hab := biSup_mono fun _ hi ↦ lt_of_lt_of_le hi hab
  le' _ := iSup₂_le fun i _ ↦ (hX i).comap_le

/-- The code's e-value as a function of the standardized sum `s` after `n` trades: the mixture
`∫ exp (λ s - λ² n / 2) π(dλ)`. With `π = halfNormal rho2` this is
`exp (log_e_standardized(n, s, rho2))` in `evidence.py` (closed form: another file). -/
noncomputable def mixG (π : Measure ℝ) (n : ℕ) (s : ℝ) : ℝ :=
  ∫ l, Real.exp (l * s - l ^ 2 * n / 2) ∂π

/-- The mixture of likelihood ratios, `M_n = ∫ L_n(λ) π(dλ) = mixG π n S_n`. -/
noncomputable def mix (X : ℕ → Ω → ℝ) (π : Measure ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∫ l, lr X l n ω ∂π

lemma mix_eq_mixG (X : ℕ → Ω → ℝ) (π : Measure ℝ) (n : ℕ) (ω : Ω) :
    mix X π n ω = mixG π n (runSum X n ω) := rfl

lemma lr_succ (X : ℕ → Ω → ℝ) (l : ℝ) (n : ℕ) (ω : Ω) :
    lr X l (n + 1) ω = lr X l n ω * Real.exp (l * X n ω - l ^ 2 / 2) := by
  unfold lr runSum
  rw [Finset.sum_range_succ, ← Real.exp_add]
  congr 1
  push_cast
  ring

lemma lr_pos (X : ℕ → Ω → ℝ) (l : ℝ) (n : ℕ) (ω : Ω) : 0 < lr X l n ω := Real.exp_pos _

/-- `L_0(λ) = 1`. -/
lemma lr_zero (X : ℕ → Ω → ℝ) (l : ℝ) (ω : Ω) : lr X l 0 ω = 1 := by
  simp [lr, runSum]

lemma mix_nonneg (X : ℕ → Ω → ℝ) (π : Measure ℝ) (n : ℕ) (ω : Ω) : 0 ≤ mix X π n ω :=
  integral_nonneg fun _ ↦ (lr_pos X _ n ω).le

/-- Before any trade the mixture is exactly 1 (`π` is a probability measure). -/
lemma mix_zero (X : ℕ → Ω → ℝ) (π : Measure ℝ) [IsProbabilityMeasure π] (ω : Ω) :
    mix X π 0 ω = 1 := by
  simp [mix, lr, runSum]

lemma mixG_measurable (π : Measure ℝ) [SFinite π] (n : ℕ) : Measurable (mixG π n) := by
  have h : StronglyMeasurable
      (Function.uncurry fun (s l : ℝ) ↦ Real.exp (l * s - l ^ 2 * n / 2)) := by
    apply Measurable.stronglyMeasurable
    fun_prop
  exact h.integral_prod_right.measurable

section Process

variable {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i))
include hX

lemma comap_le_past {i n : ℕ} (h : i < n) :
    MeasurableSpace.comap (X i) inferInstance ≤ past X hX n :=
  le_iSup₂ (f := fun i (_ : i ∈ Set.Iio n) ↦ MeasurableSpace.comap (X i) inferInstance) i h

lemma runSum_measurable_past (n : ℕ) : Measurable[past X hX n] (runSum X n) :=
  Finset.measurable_sum (range n) fun _ hi ↦
    Measurable.of_comap_le (comap_le_past hX (Finset.mem_range.mp hi))

lemma lr_measurable_past (l : ℝ) (n : ℕ) : Measurable[past X hX n] (lr X l n) := by
  have h : Measurable fun x : ℝ ↦ Real.exp (l * x - l ^ 2 * n / 2) := by fun_prop
  exact h.comp (runSum_measurable_past hX n)

lemma mix_measurable_past (π : Measure ℝ) [SFinite π] (n : ℕ) :
    Measurable[past X hX n] (mix X π n) :=
  (mixG_measurable π n).comp (runSum_measurable_past hX n)

lemma measurable_lr_uncurry (n : ℕ) :
    Measurable (Function.uncurry fun (ω : Ω) (l : ℝ) ↦ lr X l n ω) := by
  have hS : Measurable (runSum X n) := Finset.measurable_sum _ fun i _ ↦ hX i
  have e : (Function.uncurry fun (ω : Ω) (l : ℝ) ↦ lr X l n ω) =
      fun p : Ω × ℝ ↦ Real.exp (p.2 * runSum X n p.1 - p.2 ^ 2 * n / 2) := rfl
  rw [e]
  fun_prop

/-- The next trade is independent of everything seen so far. -/
lemma indep_next_past (hind : iIndepFun X μ) (n : ℕ) :
    Indep (MeasurableSpace.comap (X n) inferInstance) (past X hX n) μ := by
  have h := indep_iSup_of_disjoint (m := fun i ↦ MeasurableSpace.comap (X i) inferInstance)
    (fun i ↦ (hX i).comap_le) hind.iIndep (S := {n}) (T := Set.Iio n) (by simp)
  have e : (⨆ i ∈ ({n} : Set ℕ), MeasurableSpace.comap (X i) inferInstance) =
      MeasurableSpace.comap (X n) inferInstance := by simp
  rw [e] at h
  exact h

variable {θ : ℝ} (hind : iIndepFun X μ) (hlaw : ∀ i, HasLaw (X i) (gaussianReal θ 1) μ)
include hind hlaw

/-- **One trade of the likelihood ratio.** For independent Gaussian trades `N(θ, 1)`, on every
event `s` of the past `ℱ_n`: `∫_s L_{n+1}(λ) = exp (λ θ) · ∫_s L_n(λ)`. The new factor
`exp (λ X_n - λ² / 2)` is independent of `ℱ_n` and has mean `exp (λ θ)` (the Gaussian moment
generating function, `mgf_gaussianReal`). -/
lemma setIntegral_lr_succ (l : ℝ) (n : ℕ) {s : Set Ω} (hs : MeasurableSet[past X hX n] s) :
    ∫ ω in s, lr X l (n + 1) ω ∂μ = Real.exp (l * θ) * ∫ ω in s, lr X l n ω ∂μ := by
  have hs' : MeasurableSet s := (past X hX).le n s hs
  set Y : Ω → ℝ := fun ω ↦ Real.exp (l * X n ω - l ^ 2 / 2) with hYdef
  have hY : ∫ ω, Y ω ∂μ = Real.exp (l * θ) := by
    have hm : mgf (X n) μ l = Real.exp (θ * l + ((1 : ℝ≥0) : ℝ) * l ^ 2 / 2) :=
      mgf_gaussianReal (hlaw n) l
    simp only [mgf] at hm
    calc ∫ ω, Y ω ∂μ = (∫ ω, Real.exp (l * X n ω) ∂μ) / Real.exp (l ^ 2 / 2) := by
          simp only [Y, Real.exp_sub, integral_div]
      _ = Real.exp (l * θ) := by
          rw [hm, ← Real.exp_sub]
          congr 1
          push_cast
          ring
  have hmI : Measurable[past X hX n] (s.indicator (lr X l n)) :=
    (lr_measurable_past hX l n).indicator hs
  have hindep : IndepFun (s.indicator (lr X l n)) Y μ := by
    rw [IndepFun_iff_Indep]
    refine indep_of_indep_of_le (indep_next_past hX hind n).symm hmI.comap_le ?_
    exact MeasurableSpace.comap_le_comap_of_eq_comp (fun x ↦ Real.exp (l * x - l ^ 2 / 2))
      (by fun_prop) rfl
  have hYm : Measurable Y := by
    have : Measurable fun x : ℝ ↦ Real.exp (l * x - l ^ 2 / 2) := by fun_prop
    exact this.comp (hX n)
  calc ∫ ω in s, lr X l (n + 1) ω ∂μ = ∫ ω, (s.indicator (lr X l n) * Y) ω ∂μ := by
        rw [← integral_indicator hs']
        congr 1
        ext ω
        by_cases hω : ω ∈ s <;> simp [hω, lr_succ, Y]
    _ = (∫ ω, s.indicator (lr X l n) ω ∂μ) * ∫ ω, Y ω ∂μ :=
        hindep.integral_mul_eq_mul_integral
          (hmI.mono ((past X hX).le n) le_rfl).aestronglyMeasurable hYm.aestronglyMeasurable
    _ = Real.exp (l * θ) * ∫ ω in s, lr X l n ω ∂μ := by
        rw [hY, mul_comm, integral_indicator hs']

variable [IsProbabilityMeasure μ]

/-- `E[L_n(λ)] = exp (λ θ)ⁿ` (so `= 1` at `θ = 0`). -/
lemma integral_lr (l : ℝ) (n : ℕ) : ∫ ω, lr X l n ω ∂μ = Real.exp (l * θ) ^ n := by
  induction n with
  | zero => simp [lr, runSum]
  | succ n ih =>
    rw [← setIntegral_univ, setIntegral_lr_succ hX hind hlaw l n MeasurableSet.univ,
      setIntegral_univ, ih, pow_succ, mul_comm]

lemma integrable_lr (l : ℝ) (n : ℕ) : Integrable (lr X l n) μ :=
  Integrable.of_integral_ne_zero (by rw [integral_lr hX hind hlaw l n]; positivity)

/-- **The likelihood ratio is a supermartingale** for Gaussian trades `N(θ, 1)` whenever
`λ θ ≤ 0` (for instance `λ ≥ 0` and a mean `θ ≤ 0`: no edge or worse).
Assumptions: `X_i` measurable, independent (`iIndepFun`), each `N(θ, 1)`; `μ` a probability.
Filtration: `past X hX` (the trades seen so far). -/
theorem supermartingale_lr {l : ℝ} (hl : l * θ ≤ 0) :
    Supermartingale (lr X l) (past X hX) μ := by
  refine supermartingale_of_setIntegral_succ_le
    (fun n ↦ (lr_measurable_past hX l n).stronglyMeasurable) (integrable_lr hX hind hlaw l)
    fun n s hs ↦ ?_
  rw [setIntegral_lr_succ hX hind hlaw l n hs]
  exact mul_le_of_le_one_left (integral_nonneg fun ω ↦ (lr_pos X l n ω).le)
    (Real.exp_le_one_iff.mpr hl)

variable (π : Measure ℝ) [IsProbabilityMeasure π] (hπ : ∀ᵐ l ∂π, l * θ ≤ 0)
include hπ

/-- Tonelli's bound that makes Fubini legal: `(ω, λ) ↦ L_n(λ)(ω)` is integrable on
`(μ restricted to s) × π`, because for `π`-almost every `λ`, `∫_s L_n(λ) ≤ E[L_n(λ)] ≤ 1`. -/
lemma integrable_lr_prod (n : ℕ) (s : Set Ω) :
    Integrable (Function.uncurry fun (ω : Ω) (l : ℝ) ↦ lr X l n ω) ((μ.restrict s).prod π) := by
  have hm := measurable_lr_uncurry hX n
  rw [integrable_prod_iff' hm.aestronglyMeasurable]
  refine ⟨ae_of_all _ fun l ↦ (integrable_lr hX hind hlaw l n).restrict, ?_⟩
  refine Integrable.of_bound ?_ 1 ?_
  · exact (StronglyMeasurable.integral_prod_left (f := fun ω l ↦ ‖lr X l n ω‖)
      hm.norm.stronglyMeasurable).aestronglyMeasurable
  · filter_upwards [hπ] with l hl
    simp only [Function.uncurry_apply_pair]
    have hpos : ∀ ω, ‖lr X l n ω‖ = lr X l n ω := fun ω ↦
      Real.norm_of_nonneg (lr_pos X l n ω).le
    simp_rw [hpos]
    have h0 : 0 ≤ ∫ ω in s, lr X l n ω ∂μ := integral_nonneg fun ω ↦ (lr_pos X l n ω).le
    rw [Real.norm_of_nonneg h0]
    calc ∫ ω in s, lr X l n ω ∂μ ≤ ∫ ω, lr X l n ω ∂μ :=
          setIntegral_le_integral (integrable_lr hX hind hlaw l n)
            (ae_of_all _ fun ω ↦ (lr_pos X l n ω).le)
      _ = Real.exp (l * θ) ^ n := integral_lr hX hind hlaw l n
      _ ≤ 1 := pow_le_one₀ (Real.exp_pos _).le (Real.exp_le_one_iff.mpr hl)

/-- Fubini: `∫_s M_n dμ = ∫ (∫_s L_n(λ) dμ) π(dλ)`. -/
lemma setIntegral_mix (n : ℕ) (s : Set Ω) :
    ∫ ω in s, mix X π n ω ∂μ = ∫ l, ∫ ω in s, lr X l n ω ∂μ ∂π :=
  integral_integral_swap (integrable_lr_prod hX hind hlaw π hπ n s)

lemma integrable_mix (n : ℕ) : Integrable (mix X π n) μ := by
  have h := (integrable_lr_prod hX hind hlaw π hπ n Set.univ).integral_prod_left
  rw [Measure.restrict_univ] at h
  exact h

lemma setIntegral_mix_succ (n : ℕ) {s : Set Ω} (hs : MeasurableSet[past X hX n] s) :
    ∫ ω in s, mix X π (n + 1) ω ∂μ =
      ∫ l, Real.exp (l * θ) * ∫ ω in s, lr X l n ω ∂μ ∂π := by
  rw [setIntegral_mix hX hind hlaw π hπ (n + 1) s]
  congr 1
  ext l
  exact setIntegral_lr_succ hX hind hlaw l n hs

/-- **The mixture is a supermartingale** for Gaussian trades `N(θ, 1)` whenever `λ θ ≤ 0` for
`π`-almost every `λ`. With the code's half-normal `π` (`λ ≥ 0`) that is every mean `θ ≤ 0`.
Assumptions: `X_i` measurable, independent, each `N(θ, 1)`; `μ`, `π` probability measures. -/
theorem supermartingale_mix : Supermartingale (mix X π) (past X hX) μ := by
  refine supermartingale_of_setIntegral_succ_le
    (fun n ↦ (mix_measurable_past hX π n).stronglyMeasurable)
    (integrable_mix hX hind hlaw π hπ) fun n s hs ↦ ?_
  rw [setIntegral_mix_succ hX hind hlaw π hπ n hs, setIntegral_mix hX hind hlaw π hπ n s]
  refine integral_mono_of_nonneg ?_
    (integrable_lr_prod hX hind hlaw π hπ n s).integral_prod_right ?_
  · exact ae_of_all _ fun l ↦ mul_nonneg (Real.exp_pos _).le
      (integral_nonneg fun ω ↦ (lr_pos X l n ω).le)
  · filter_upwards [hπ] with l hl
    exact mul_le_of_le_one_left (integral_nonneg fun ω ↦ (lr_pos X l n ω).le)
      (Real.exp_le_one_iff.mpr hl)

/-- **Ville for the e-value.** For independent Gaussian trades `N(θ, 1)` and a mixing
probability `π` with `λ θ ≤ 0` almost surely, the e-value `M_n = ∫ L_n(λ) π(dλ)` ever reaches
`B > 0`, at ANY trade count, with probability at most `1 / B`.
Backs: "above zero" is called when `exp(log_e_one_sided(...)) ≥ B`; under the null this happens,
however often one checks, with probability `≤ 1 / B`. -/
theorem evalue_ville {B : ℝ} (hB : 0 < B) :
    μ {ω | ∃ n, B ≤ mix X π n ω} ≤ ENNReal.ofReal (1 / B) :=
  ville (supermartingale_mix hX hind hlaw π hπ) (fun n ω ↦ mix_nonneg X π n ω)
    (by simp [mix_zero]) hB

end Process

/-- **The likelihood ratio is a martingale** with `E[L₀] = 1` (`lr_zero`): for independent
standard Gaussian trades `X_i ~ N(0, 1)` (measurable, `iIndepFun`), the natural filtration of the
trades (`past`), and any fixed `λ`, `L_n(λ) = exp (λ S_n - λ² n / 2)` is a martingale. -/
theorem martingale_lr {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) [IsProbabilityMeasure μ]
    (hind : iIndepFun X μ) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ) (l : ℝ) :
    Martingale (lr X l) (past X hX) μ :=
  martingale_of_setIntegral_eq_succ (fun n ↦ (lr_measurable_past hX l n).stronglyMeasurable)
    (integrable_lr hX hind hlaw l) fun n s hs ↦ by
      rw [setIntegral_lr_succ hX hind hlaw l n hs]; simp

/-! ## 3. The mixture is a martingale -/

/-- **The mixture is a martingale** with `M₀ = 1` (`mix_zero`) and `M_n ≥ 0` (`mix_nonneg`):
for independent standard Gaussian trades and ANY probability measure `π` on `ℝ` (in particular
the code's half-normal, `halfNormal rho2`), `M_n = ∫ exp (λ S_n - λ² n / 2) π(dλ)` is a
martingale for the trades seen so far. Proof by Tonelli/Fubini on each event of the past; no
conditional Fubini.
Backs: `log_e_standardized` (the e-value is a test martingale under "mean 0"). -/
theorem martingale_mix {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) [IsProbabilityMeasure μ]
    (hind : iIndepFun X μ) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ)
    (π : Measure ℝ) [IsProbabilityMeasure π] :
    Martingale (mix X π) (past X hX) μ := by
  have hπ : ∀ᵐ l ∂π, l * (0 : ℝ) ≤ 0 := ae_of_all _ fun l ↦ by simp
  refine martingale_of_setIntegral_eq_succ
    (fun n ↦ (mix_measurable_past hX π n).stronglyMeasurable)
    (integrable_mix hX hind hlaw π hπ) fun n s hs ↦ ?_
  rw [setIntegral_mix_succ hX hind hlaw π hπ n hs, setIntegral_mix hX hind hlaw π hπ n s]
  simp

/-! ## 4. The conclusion: the e-value, both sides, the range -/

/-- The code's mixing distribution: the half-normal `|N(0, rho2)|`, a probability measure that
lives on `λ ≥ 0` (`halfNormal_ae_nonneg`). In `evidence.py`, `rho2 = mixing_rho2(bar)`. -/
noncomputable def halfNormal (rho2 : ℝ≥0) : Measure ℝ :=
  (gaussianReal 0 rho2).map (fun x ↦ |x|)

instance (rho2 : ℝ≥0) : IsProbabilityMeasure (halfNormal rho2) := by
  unfold halfNormal
  infer_instance

lemma halfNormal_ae_nonneg (rho2 : ℝ≥0) : ∀ᵐ l ∂(halfNormal rho2), 0 ≤ l := by
  unfold halfNormal
  exact (ae_map_iff (by fun_prop) measurableSet_Ici).mpr (ae_of_all _ fun x ↦ abs_nonneg x)

/-- The code's one-sided e-value after `n` trades against "the mean is `θ`", for a drift in
direction `dir` (`1`: above, `-1`: below), with a KNOWN spread `σ`: the mixture at the
standardized sum `dir · (S_n - n θ) / σ`. In `evidence.py`:
`exp(log_e_one_sided(n, total - n * θ, σ, rho2, dir))` with `π = halfNormal rho2`. -/
noncomputable def eValue (π : Measure ℝ) (σ θ dir : ℝ) (X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  mixG π n (dir * (runSum X n ω - n * θ) / σ)

/-- The anytime range after `n` trades at bar `B`: every mean `θ` that neither one-sided
e-value rejects (a rejection is `e ≥ B`, as in the code). `range_of` (no skew) inverts the same
e-values with `boundary`, and its closed interval contains this set; that identification uses the
closed form and monotonicity (another file) and is checked numerically in
`tests/test_bridge_anytime.py`. -/
def anytimeRange (π : Measure ℝ) (σ B : ℝ) (X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : Set ℝ :=
  {θ | eValue π σ θ 1 X n ω < B ∧ eValue π σ θ (-1) X n ω < B}

lemma runSum_standardize (X : ℕ → Ω → ℝ) (m c : ℝ) (n : ℕ) (ω : Ω) :
    runSum (fun i ω ↦ (X i ω - m) / c) n ω = (runSum X n ω - n * m) / c := by
  unfold runSum
  rw [← Finset.sum_div, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range,
    nsmul_eq_mul]

/-- Standardizing a Gaussian trade with its known spread: `(Y - m) / σ ~ N((θ - m) / σ, 1)`. -/
lemma hasLaw_standardize {Y : Ω → ℝ} {θ m : ℝ} {σ : ℝ≥0} (hσ : σ ≠ 0)
    (h : HasLaw Y (gaussianReal θ (σ ^ 2)) μ) :
    HasLaw (fun ω ↦ (Y ω - m) / σ) (gaussianReal ((θ - m) / σ) 1) μ := by
  have h3 := gaussianReal_div_const (gaussianReal_sub_const h m) (σ : ℝ)
  convert h3 using 2
  rw [← NNReal.coe_inj]
  have : (σ : ℝ) ≠ 0 := NNReal.coe_ne_zero.mpr hσ
  push_cast
  field_simp

/-- **Both sides.** For independent standard Gaussian trades and any mixing probability `π`,
the probability that EITHER one-sided e-value (`s = S_n` for "above", `s = -S_n` for "below")
ever reaches `B > 0` is at most `2 / B`: Ville on each side, then a union bound. -/
theorem two_sided_ville {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) [IsProbabilityMeasure μ]
    (hind : iIndepFun X μ) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ)
    (π : Measure ℝ) [IsProbabilityMeasure π] {B : ℝ} (hB : 0 < B) :
    μ {ω | ∃ n, B ≤ mixG π n (runSum X n ω) ∨ B ≤ mixG π n (-runSum X n ω)} ≤
      ENNReal.ofReal (2 / B) := by
  set Y : ℕ → Ω → ℝ := fun i ω ↦ -X i ω with hYdef
  have hY : ∀ i, Measurable (Y i) := fun i ↦ (hX i).neg
  have hindY : iIndepFun Y μ := hind.comp (fun _ x ↦ -x) (fun _ ↦ measurable_neg)
  have hlawY : ∀ i, HasLaw (Y i) (gaussianReal 0 1) μ := fun i ↦ by
    have h := gaussianReal_neg (hlaw i)
    rw [neg_zero] at h
    exact h
  have hπ : ∀ᵐ l ∂π, l * (0 : ℝ) ≤ 0 := ae_of_all _ fun l ↦ by simp
  have hsumY : ∀ n ω, runSum Y n ω = -runSum X n ω := fun n ω ↦ by
    simp [Y, runSum, Finset.sum_neg_distrib]
  have h1 := evalue_ville hX hind hlaw π hπ hB
  have h2 := evalue_ville hY hindY hlawY π hπ hB
  calc μ {ω | ∃ n, B ≤ mixG π n (runSum X n ω) ∨ B ≤ mixG π n (-runSum X n ω)}
      ≤ μ ({ω | ∃ n, B ≤ mix X π n ω} ∪ {ω | ∃ n, B ≤ mix Y π n ω}) := by
        apply measure_mono
        rintro ω ⟨n, h | h⟩
        · exact Or.inl ⟨n, h⟩
        · refine Or.inr ⟨n, ?_⟩
          rw [mix_eq_mixG, hsumY]
          exact h
    _ ≤ μ {ω | ∃ n, B ≤ mix X π n ω} + μ {ω | ∃ n, B ≤ mix Y π n ω} := measure_union_le _ _
    _ ≤ ENNReal.ofReal (1 / B) + ENNReal.ofReal (1 / B) := add_le_add h1 h2
    _ = ENNReal.ofReal (2 / B) := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        ring

/-- **The range is anytime-valid (a confidence sequence).** Trades `X_i` independent,
each `N(θ, σ²)` with the spread `σ > 0` KNOWN, any mixing probability `π` (the code's is
`halfNormal rho2`), any bar `B > 0`. Then the probability that the range ever, at ANY trade
count, leaves out the true mean `θ` is at most `2 / B`: the range covers `θ` at every trade
count at once with probability `≥ 1 - 2 / B`.
Backs: `range_of` / `half_width` and "It stays honest even if you check after every trade, so
watching it does not fool you." Holds for every true mean `θ` (by translation). -/
theorem anytime_range {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) [IsProbabilityMeasure μ]
    {θ : ℝ} {σ : ℝ≥0} (hσ : σ ≠ 0) (hind : iIndepFun X μ)
    (hlaw : ∀ i, HasLaw (X i) (gaussianReal θ (σ ^ 2)) μ)
    (π : Measure ℝ) [IsProbabilityMeasure π] {B : ℝ} (hB : 0 < B) :
    μ {ω | ∃ n, θ ∉ anytimeRange π σ B X n ω} ≤ ENNReal.ofReal (2 / B) := by
  set Z : ℕ → Ω → ℝ := fun i ω ↦ (X i ω - θ) / σ with hZdef
  have hZ : ∀ i, Measurable (Z i) := fun i ↦ ((hX i).sub_const θ).div_const _
  have hindZ : iIndepFun Z μ := hind.comp (fun _ x ↦ (x - θ) / σ) (fun _ ↦ by fun_prop)
  have hlawZ : ∀ i, HasLaw (Z i) (gaussianReal 0 1) μ := fun i ↦ by
    simpa using hasLaw_standardize (m := θ) hσ (hlaw i)
  refine le_trans (measure_mono ?_) (two_sided_ville hZ hindZ hlawZ π hB)
  rintro ω ⟨n, hn⟩
  refine ⟨n, ?_⟩
  simp only [anytimeRange, Set.mem_ofPred_eq, not_and_or, not_lt] at hn
  rw [hZdef, runSum_standardize]
  rcases hn with h | h
  · refine Or.inl ?_
    have e : eValue π σ θ 1 X n ω = mixG π n ((runSum X n ω - n * θ) / σ) := by
      simp only [eValue, one_mul]
    rwa [e] at h
  · refine Or.inr ?_
    have e : eValue π σ θ (-1) X n ω = mixG π n (-((runSum X n ω - n * θ) / σ)) := by
      simp only [eValue]
      congr 1
      ring
    rwa [e] at h

/-- **The page's 95%.** At the code's bar `BASE_BAR = 40` the anytime range misses the true mean
at some trade count with probability at most `2 / 40 = 1 / 20` (5%), for the code's half-normal
mixture with ANY `rho2` (in particular `mixing_rho2(40)`).
Assumptions: as `anytime_range` (independent Gaussian trades, known spread `σ > 0`). -/
theorem anytime_range_95 {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) [IsProbabilityMeasure μ]
    {θ : ℝ} {σ : ℝ≥0} (hσ : σ ≠ 0) (hind : iIndepFun X μ)
    (hlaw : ∀ i, HasLaw (X i) (gaussianReal θ (σ ^ 2)) μ) (rho2 : ℝ≥0) :
    μ {ω | ∃ n, θ ∉ anytimeRange (halfNormal rho2) σ 40 X n ω} ≤ ENNReal.ofReal (1 / 20) := by
  have h := anytime_range hX hσ hind hlaw (halfNormal rho2) (B := 40) (by norm_num)
  convert h using 2
  norm_num

/-- **No false "above zero" under no edge or worse (composite null).** Trades `X_i`
independent, each `N(θ, σ²)` with `θ ≤ 0` and the spread `σ > 0` known. The code's "above zero"
e-value against mean 0 (half-normal mixture, any `rho2`) ever reaches `B > 0` with probability at
most `1 / B`. Proof: `(X_i / σ)` are `N(θ / σ, 1)` with `θ / σ ≤ 0`, the half-normal has
`λ ≥ 0`, so the mixture is a nonnegative supermartingale (`supermartingale_mix`); then Ville.
Backs: `log_e_record(..., +1) >= log(bar)` ("real edge" alone, `judge._passes_alone`), up to the
plug-in spread and the skew correction. -/
theorem above_zero_false_alarm {X : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i))
    [IsProbabilityMeasure μ] {θ : ℝ} (hθ : θ ≤ 0) {σ : ℝ≥0} (hσ : σ ≠ 0)
    (hind : iIndepFun X μ) (hlaw : ∀ i, HasLaw (X i) (gaussianReal θ (σ ^ 2)) μ)
    (rho2 : ℝ≥0) {B : ℝ} (hB : 0 < B) :
    μ {ω | ∃ n, B ≤ eValue (halfNormal rho2) σ 0 1 X n ω} ≤ ENNReal.ofReal (1 / B) := by
  set Z : ℕ → Ω → ℝ := fun i ω ↦ (X i ω - 0) / σ with hZdef
  have hZ : ∀ i, Measurable (Z i) := fun i ↦ ((hX i).sub_const 0).div_const _
  have hindZ : iIndepFun Z μ := hind.comp (fun _ x ↦ (x - 0) / σ) (fun _ ↦ by fun_prop)
  have hlawZ : ∀ i, HasLaw (Z i) (gaussianReal ((θ - 0) / σ) 1) μ := fun i ↦
    hasLaw_standardize (m := 0) hσ (hlaw i)
  have hπ : ∀ᵐ l ∂(halfNormal rho2), l * ((θ - 0) / σ) ≤ 0 := by
    filter_upwards [halfNormal_ae_nonneg rho2] with l hl
    exact mul_nonpos_of_nonneg_of_nonpos hl (div_nonpos_of_nonpos_of_nonneg (by simpa using hθ)
      σ.coe_nonneg)
  refine le_trans (measure_mono ?_) (evalue_ville hZ hindZ hlawZ (halfNormal rho2) hπ hB)
  rintro ω ⟨n, hn⟩
  refine ⟨n, ?_⟩
  rw [mix_eq_mixG, hZdef, runSum_standardize]
  simpa [eValue] using hn

end EdgeOrLuck.Anytime

#print axioms EdgeOrLuck.Anytime.maximal_ineq_super
#print axioms EdgeOrLuck.Anytime.ville_super
#print axioms EdgeOrLuck.Anytime.ville
#print axioms EdgeOrLuck.Anytime.ville_martingale
#print axioms EdgeOrLuck.Anytime.setIntegral_lr_succ
#print axioms EdgeOrLuck.Anytime.martingale_lr
#print axioms EdgeOrLuck.Anytime.supermartingale_lr
#print axioms EdgeOrLuck.Anytime.martingale_mix
#print axioms EdgeOrLuck.Anytime.supermartingale_mix
#print axioms EdgeOrLuck.Anytime.evalue_ville
#print axioms EdgeOrLuck.Anytime.two_sided_ville
#print axioms EdgeOrLuck.Anytime.anytime_range
#print axioms EdgeOrLuck.Anytime.anytime_range_95
#print axioms EdgeOrLuck.Anytime.above_zero_false_alarm
