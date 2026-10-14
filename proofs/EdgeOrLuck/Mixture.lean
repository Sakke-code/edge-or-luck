import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.CDF
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# The one-sided normal-mixture e-value and the range it gives

edge or luck, `src/edgeorluck/evidence.py`: `log_e_standardized`, `boundary`, `half_width`,
`range_of`, `log_e_record`, `skew_shift`, `mixing_rho2`, `BASE_BAR`. Bridge tests:
`tests/test_bridge_mixture.py`.

## Notation

* `phi x = exp (-x²/2) / √(2π)` and `Phi z = ∫_{x ≤ z} phi x` (`Phi_eq_cdf`: it is the CDF of
  Mathlib's `gaussianReal 0 1`).
* `ρ2 > 0` is the mixing variance (`rho2`), `n ≥ 0` the number of trades, `v = 1 + ρ2 n`.
* `mixDensity ρ2 l = 2 / √(2π ρ2) · exp (-l² / (2 ρ2))` on `l ≥ 0`: the half-normal mixing density.
* `eMix ρ2 n s = ∫_{l ≥ 0} exp (l s - l² n / 2) · mixDensity ρ2 l`: the e-value against
  "the mean is 0" for a standardized sum `s` (sum / spread) after `n` trades.

## Results (machine-checked; no unfinished proofs, no new axioms)

1. `eMix_closed_form`, `eMix_eq_exp_logEStd`: **closed form.**
   `eMix ρ2 n s = v^(-1/2) · exp (ρ2 s² / (2v)) · 2 Phi (√ρ2 s / √v)`, which is
   `exp (log_e_standardized(n, s, rho2))` with the code's `_log_two_phi(z) = log 2 + log Phi(z)`
   (its `z > -8` branch; see below for `z ≤ -8`).
2. `eMix_expectation_eq_one`, `eMix_expectation_le_one`: **it is an e-value.** If the standardized
   sum `S` has law `N(μ, n)` then `E[eMix ρ2 n S] = 1` for `μ = 0` and `≤ 1` for every `μ ≤ 0`
   (Tonelli, then the Gaussian moment generating function: `E[exp (l S - l² n/2)] = exp (l μ)`).
3. `eMix_strictMono`, `logEStd_strictMono`: **strictly increasing in `s`.**
4. `eMix_zero`, `eMix_zero_le_one`, `eMix_neg_le_one`: `eMix ρ2 n 0 = v^(-1/2) ≤ 1`, and
   `eMix ρ2 n (-s) ≤ 1` for `s ≥ 0`: a sum on the up side is never evidence for the down side.
5. `boundary_exists_unique`, `boundary_pos`, `boundary_bracket`, `exp_one_mul_le_eMix_bisectHi`,
   `bisect_invariant`, `boundary_bisection`, `boundaryCode_spec`: **the boundary.** For `B > 1`
   exactly one `b ≥ 0` has `eMix ρ2 n b = B` (and `b > 0`). The code's bracket `lo = 0`,
   `hi = √(2v/ρ2 · (log B + ½ log v + 1)) + 1` satisfies `e(lo) < B < e(hi)` (in fact
   `e(hi) ≥ e · B`), so after `k` halvings the code's answer `hi_k` satisfies
   `b ≤ hi_k ≤ b + hi / 2^k` (`k = 80` in `boundary`).
6. `above_rejects_iff`, `below_rejects_iff`, `range_eq_not_rejected`,
   `verdict_iff_range_clears_zero`, `down_verdict_iff_range_below_zero`, `pass_implies_total_pos`:
   **range ⟺ verdict.** With `range_of`'s ends `T/n - (b + shiftUp)·sd/n` and
   `T/n + (b + shiftDown)·sd/n`: the "above zero" e-value (`log_e_record(rec, bar, +1)`) reaches the
   bar iff the lower end is `≥ 0`; more generally the open range is exactly the set of means `μ0`
   that neither one-sided test (each with its own skew shift) rejects at bar `B`.
7. `skewShift_nonneg`, `skewShift_zero_skew`, `skew_never_raises_e`, `range_with_skew_contains`:
   **skew never lowers a bar**: `skew_shift ≥ 0`, so the skew-adjusted e-value is never larger and
   the skew-adjusted range always contains the unadjusted one.
8. `logEStd_scale_free`: the e-value depends on `n` and `ρ2` only through `ρ2·n` and `s/√n`, so the
   best mixing scale for a target trade count `t*` is exactly of the form `c(B)/t*`.
   **The constant is a heuristic, not an optimum** (`mixing_rho2`: `c = 2 log B`). Numerically
   (bridge test `test_tuning_is_a_close_heuristic`): the `ρ2` that minimizes the boundary at
   `t* = 100` is `8.24/t*` for `B = 40` (the code uses `7.38/t*`); the code's boundary at `t*` is
   0.030% above the best possible (0.017% at `B = 400`, 0.010% at `B = 4000`), and the code's
   `ρ2` is the exact optimum for `n ≈ 112`, not 100. "Decides soonest near T_STAR" is true to
   within these margins and is not a theorem.
9. `range_mono_sd`, `range_covers_fixed_n`: at a FIXED number of trades, Gaussian trades with
   known spread `σ`, any nonnegative (data-dependent) skew shifts and any spread `sd ≥ σ`:
   the range misses the true mean with probability `≤ 2 / B` (5% at `B = 40`). The stronger
   every-`n`-at-once statement is `EdgeOrLuck.Anytime.anytime_range` (Anytime.lean).

## Exact vs asymptotic, and what is NOT covered

* Everything is exact real arithmetic, not IEEE floats; the float behaviour is the bridge tests'.
* Items 2 and 9 are exact only for Gaussian trades with KNOWN spread (the standardized sum is
  `total / σ`). `evidence.py` plugs in the sample SD (times the serial-correlation factor): its
  e-value and range are then valid only as the number of trades grows (asymptotically).
* `_log_two_phi` switches at `z ≤ -8` to the leading Mills-ratio term `φ(z)/|z|`, which
  overstates `Phi z` (by 1.5% at `z = -8`). There the code's `log_e_standardized` is not the closed
  form (`logEStd`) and drops by 0.0151 as `s` crosses `z = -8`, so it is not monotone at that
  point. It never changes a decision (there `e < 1`, and every bar is `≥ 40`); xfail tests in the
  bridge file record it.
* Item 6 in exact arithmetic: "the range clears zero" must be read as "lower end `≥ 0`" (at a tie
  `e = B` passes). In floats `boundary` matches `b` to about `1e-15` relative; the bridge test
  checks that the two statements agree outside a `1e-9` band and on both sides of the tie.
* Item 8 is measured, not proved. The e-BH family, and the fact that e-BH with one strategy selects
  "above zero" exactly when `e ≥ 40 N`, are other files.
-/

open Real MeasureTheory ProbabilityTheory Set

namespace EdgeOrLuck.Mixture

/-! ## The standard normal density and CDF -/

/-- The standard normal density `φ(x) = exp (-x²/2) / √(2π)`. -/
noncomputable def phi (x : ℝ) : ℝ := exp (-x ^ 2 / 2) / √(2 * π)

/-- The standard normal CDF `Φ(z) = ∫_{x ≤ z} φ(x) dx`. The code computes it as
`0.5 * erfc(-z / √2)` (`_log_two_phi`, for `z > -8`). -/
noncomputable def Phi (z : ℝ) : ℝ := ∫ x in Iic z, phi x

lemma sqrt_two_pi_pos : 0 < √(2 * π) := Real.sqrt_pos.2 (by positivity)

lemma phi_pos (x : ℝ) : 0 < phi x := div_pos (exp_pos _) sqrt_two_pi_pos

lemma phi_neg (x : ℝ) : phi (-x) = phi x := by simp [phi]

lemma phi_continuous : Continuous phi := by unfold phi; fun_prop

lemma phi_integrable : Integrable phi := by
  have h := (integrable_exp_neg_mul_sq (by norm_num : (0:ℝ) < 1 / 2)).div_const (√(2 * π))
  refine h.congr (Filter.Eventually.of_forall fun x => ?_)
  simp only [phi]
  congr 2
  ring

/-- `Phi` is the CDF of Mathlib's standard Gaussian `gaussianReal 0 1`. -/
theorem Phi_eq_cdf (z : ℝ) : Phi z = cdf (gaussianReal 0 1) z := by
  rw [cdf_eq_real, measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero,
    ENNReal.toReal_ofReal (setIntegral_nonneg measurableSet_Iic
      fun x _ => gaussianPDFReal_nonneg 0 1 x)]
  unfold Phi
  congr 1
  ext x
  simp only [phi, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
  ring

lemma Phi_pos (z : ℝ) : 0 < Phi z := by
  unfold Phi
  rw [setIntegral_pos_iff_support_of_nonneg_ae
    (Filter.Eventually.of_forall fun x => (phi_pos x).le) phi_integrable.integrableOn]
  have hs : Function.support phi = univ :=
    Function.support_eq_univ fun x => (phi_pos x).ne'
  rw [hs, univ_inter, Real.volume_Iic]
  exact ENNReal.zero_lt_top

/-- The upper tail: `∫_{x > c} φ = Φ(-c)` (reflection `x ↦ -x`). -/
lemma integral_Ioi_phi (c : ℝ) : ∫ x in Ioi c, phi x = Phi (-c) := by
  unfold Phi
  rw [← integral_comp_neg_Ioi]
  simp only [phi_neg]

/-- `Φ(0) = 1/2`. -/
theorem Phi_zero : Phi 0 = 1 / 2 := by
  have h1 : Phi 0 = ∫ x in Ioi 0, phi x := by rw [integral_Ioi_phi, neg_zero]
  rw [h1]
  have h2 : ∫ x in Ioi (0:ℝ), phi x = (∫ x in Ioi (0:ℝ), exp (-(1/2) * x ^ 2)) / √(2 * π) := by
    rw [← integral_div]
    congr 1
    ext x
    simp only [phi]
    congr 2
    ring
  rw [h2, integral_gaussian_Ioi]
  have h3 : π / (1 / 2) = 2 * π := by ring
  rw [h3]
  field_simp [sqrt_two_pi_pos.ne']

lemma Phi_eq_add (z : ℝ) : Phi z = Phi 0 + ∫ x in (0:ℝ)..z, phi x := by
  rw [← intervalIntegral.integral_Iic_sub_Iic phi_integrable.integrableOn
    phi_integrable.integrableOn]
  unfold Phi
  ring

lemma Phi_hasDerivAt (z : ℝ) : HasDerivAt Phi (phi z) z := by
  have h : Phi = fun u => Phi 0 + ∫ x in (0:ℝ)..u, phi x := funext Phi_eq_add
  rw [h]
  exact ((phi_continuous.integral_hasStrictDerivAt 0 z).hasDerivAt).const_add _

lemma Phi_continuous : Continuous Phi :=
  continuous_iff_continuousAt.2 fun z => (Phi_hasDerivAt z).continuousAt

lemma Phi_strictMono : StrictMono Phi :=
  strictMono_of_deriv_pos fun z => by rw [(Phi_hasDerivAt z).deriv]; exact phi_pos z

/-- `2Φ(z) ≥ 1` for `z ≥ 0` (used for the bisection bracket). -/
lemma one_le_two_Phi {z : ℝ} (hz : 0 ≤ z) : 1 ≤ 2 * Phi z := by
  have := Phi_strictMono.monotone hz
  rw [Phi_zero] at this
  linarith

/-- Translation on a half-line. -/
lemma integral_comp_sub_right_Ioi (f : ℝ → ℝ) (a d : ℝ) :
    ∫ x in Ioi a, f (x - d) = ∫ x in Ioi (a - d), f x := by
  rw [← integral_indicator measurableSet_Ioi, ← integral_indicator measurableSet_Ioi]
  rw [← integral_sub_right_eq_self (μ := volume) (fun x => (Ioi (a - d)).indicator f x) d]
  congr 1
  ext x
  simp only [indicator, mem_Ioi, sub_lt_sub_iff_right]

/-- The half-line Gaussian integral: `∫_{l > 0} exp (-(k l - k c)²/2) dl = √(2π)/k · Φ(k c)`. -/
lemma halfline_gauss {k : ℝ} (hk : 0 < k) (c : ℝ) :
    ∫ l in Ioi 0, exp (-(k * l - k * c) ^ 2 / 2) = √(2 * π) / k * Phi (k * c) := by
  have h1 := integral_comp_mul_left_Ioi (fun w => exp (-(w - k * c) ^ 2 / 2)) 0 hk
  simp only [mul_zero] at h1
  rw [h1, integral_comp_sub_right_Ioi (fun u => exp (-u ^ 2 / 2)) 0 (k * c), zero_sub]
  have h2 : ∫ u in Ioi (-(k * c)), exp (-u ^ 2 / 2)
      = √(2 * π) * ∫ u in Ioi (-(k * c)), phi u := by
    rw [← integral_const_mul]
    congr 1
    ext u
    simp only [phi]
    field_simp [sqrt_two_pi_pos.ne']
  rw [h2, integral_Ioi_phi, neg_neg, smul_eq_mul]
  field_simp

/-! ## 1. The mixture e-value and its closed form -/

/-- The half-normal mixing density `π(l) = 2/√(2π ρ2) · exp (-l²/(2 ρ2))` (used on `l ≥ 0`). -/
noncomputable def mixDensity (ρ2 l : ℝ) : ℝ := 2 / √(2 * π * ρ2) * exp (-l ^ 2 / (2 * ρ2))

/-- The Gaussian likelihood ratio of drift `l` against drift 0 for a standardized sum `s`
after `n` observations: `exp (l s - l² n / 2)`. -/
noncomputable def likelihoodRatio (n s l : ℝ) : ℝ := exp (l * s - l ^ 2 * n / 2)

/-- The one-sided normal-mixture e-value `e(n, s) = ∫_{l ≥ 0} exp (l s - l² n/2) π(l) dl`. -/
noncomputable def eMix (ρ2 n s : ℝ) : ℝ := ∫ l in Ici 0, likelihoodRatio n s l * mixDensity ρ2 l

/-- **Closed form** (Python: `log_e_standardized`). For `ρ2 > 0`, `n ≥ 0` and any real `s`,
with `v = 1 + ρ2 n`:
`e(n, s) = v^(-1/2) · exp (ρ2 s² / (2v)) · 2Φ(√ρ2 · s / √v)`.
Proof: complete the square in `l`, then the half-line Gaussian integral. -/
theorem eMix_closed_form {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (s : ℝ) :
    eMix ρ2 n s = (√(1 + ρ2 * n))⁻¹ * exp (ρ2 * s ^ 2 / (2 * (1 + ρ2 * n)))
      * (2 * Phi (√ρ2 * s / √(1 + ρ2 * n))) := by
  set v := 1 + ρ2 * n with hv_def
  have hv : 0 < v := by positivity
  have hsv : 0 < √v := sqrt_pos.2 hv
  have hsρ : 0 < √ρ2 := sqrt_pos.2 hρ
  set k := √v / √ρ2 with hk_def
  have hk : 0 < k := div_pos hsv hsρ
  set c := ρ2 * s / v with hc_def
  have hk2 : k ^ 2 = v / ρ2 := by rw [hk_def, div_pow, sq_sqrt hv.le, sq_sqrt hρ.le]
  have hexp : ∀ l : ℝ, l * s - l ^ 2 * n / 2 + -l ^ 2 / (2 * ρ2)
      = ρ2 * s ^ 2 / (2 * v) + -(k * l - k * c) ^ 2 / 2 := by
    intro l
    have : (k * l - k * c) ^ 2 = k ^ 2 * (l - c) ^ 2 := by ring
    rw [this, hk2, hc_def, hv_def]
    field_simp
    ring
  have hpt : ∀ l : ℝ, likelihoodRatio n s l * mixDensity ρ2 l
      = (2 / √(2 * π * ρ2) * exp (ρ2 * s ^ 2 / (2 * v))) * exp (-(k * l - k * c) ^ 2 / 2) := by
    intro l
    simp only [likelihoodRatio, mixDensity]
    calc exp (l * s - l ^ 2 * n / 2) * (2 / √(2 * π * ρ2) * exp (-l ^ 2 / (2 * ρ2)))
        = 2 / √(2 * π * ρ2) * exp (l * s - l ^ 2 * n / 2 + -l ^ 2 / (2 * ρ2)) := by
          rw [exp_add]; ring
      _ = 2 / √(2 * π * ρ2) * exp (ρ2 * s ^ 2 / (2 * v) + -(k * l - k * c) ^ 2 / 2) := by
          rw [hexp l]
      _ = _ := by rw [exp_add]; ring
  have hkc : k * c = √ρ2 * s / √v := by
    rw [hk_def, hc_def]
    have e1 : ρ2 = √ρ2 * √ρ2 := (mul_self_sqrt hρ.le).symm
    have e2 : v = √v * √v := (mul_self_sqrt hv.le).symm
    field_simp
    nth_rewrite 1 [e1]
    nth_rewrite 2 [e2]
    ring
  unfold eMix
  rw [integral_Ici_eq_integral_Ioi]
  simp_rw [hpt]
  rw [integral_const_mul, halfline_gauss hk c, hkc,
    sqrt_mul (by positivity : (0:ℝ) ≤ 2 * π) ρ2, hk_def]
  field_simp

/-- The code's `log_e_standardized(n, s, rho2)` in exact arithmetic:
`-0.5 log v + rho2 s² / (2v) + _log_two_phi(z)` with `z = √rho2 · s / √v` and
`_log_two_phi(z) = log 2 + log Φ(z)` (its `z > -8` branch; the `z ≤ -8` branch is an
approximation, see the module docstring). -/
noncomputable def logEStd (n s ρ2 : ℝ) : ℝ :=
  -(1 / 2) * log (1 + ρ2 * n) + ρ2 * s ^ 2 / (2 * (1 + ρ2 * n))
    + (log 2 + log (Phi (√ρ2 * s / √(1 + ρ2 * n))))

lemma sqrt_eq_exp_half_log {v : ℝ} (hv : 0 < v) : √v = exp (1 / 2 * log v) := by
  rw [sqrt_eq_rpow, rpow_def_of_pos hv, mul_comm]

/-- **The code computes the logarithm of the mixture integral** (Python: `log_e_standardized`):
for `ρ2 > 0`, `n ≥ 0`, `e(n, s) = exp (log_e_standardized(n, s, rho2))`. -/
theorem eMix_eq_exp_logEStd {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (s : ℝ) :
    eMix ρ2 n s = exp (logEStd n s ρ2) := by
  have hv : 0 < 1 + ρ2 * n := by positivity
  rw [eMix_closed_form hρ hn, logEStd, exp_add, exp_add, exp_add,
    exp_log two_pos, exp_log (Phi_pos _), neg_mul, exp_neg, ← sqrt_eq_exp_half_log hv]

lemma eMix_pos {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (s : ℝ) : 0 < eMix ρ2 n s := by
  rw [eMix_eq_exp_logEStd hρ hn]; exact exp_pos _

lemma integrableOn_integrand {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (s : ℝ) :
    IntegrableOn (fun l => likelihoodRatio n s l * mixDensity ρ2 l) (Ici 0) :=
  Integrable.of_integral_ne_zero (eMix_pos hρ hn s).ne'

lemma mixDensity_nonneg {ρ2 : ℝ} (hρ : 0 < ρ2) (l : ℝ) : 0 ≤ mixDensity ρ2 l := by
  unfold mixDensity; positivity

lemma integrand_nonneg {ρ2 : ℝ} (hρ : 0 < ρ2) (n s l : ℝ) :
    0 ≤ likelihoodRatio n s l * mixDensity ρ2 l :=
  mul_nonneg (exp_pos _).le (mixDensity_nonneg hρ l)

/-! ## 3–4. Monotone in `s`; the value at zero; the down side -/

/-- **Strictly increasing in `s`** (Python: `log_e_standardized`; `boundary`'s docstring "the
e-value grows with `s`"). For `ρ2 > 0`, `n ≥ 0`: `s < t → e(n, s) < e(n, t)`. For `l > 0` the
integrand grows strictly with `s`. -/
theorem eMix_strictMono {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) : StrictMono (eMix ρ2 n) := by
  intro s t hst
  have hs := integrableOn_integrand hρ hn s
  have ht := integrableOn_integrand hρ hn t
  have hdiff : eMix ρ2 n t - eMix ρ2 n s = ∫ l in Ici 0,
      (likelihoodRatio n t l * mixDensity ρ2 l - likelihoodRatio n s l * mixDensity ρ2 l) :=
    (integral_sub ht hs).symm
  have hle : ∀ l, 0 ≤ l → likelihoodRatio n s l * mixDensity ρ2 l
      ≤ likelihoodRatio n t l * mixDensity ρ2 l := by
    intro l hl
    apply mul_le_mul_of_nonneg_right _ (mixDensity_nonneg hρ l)
    unfold likelihoodRatio
    apply exp_le_exp.2
    nlinarith
  have hlt : ∀ l, 0 < l → likelihoodRatio n s l * mixDensity ρ2 l
      < likelihoodRatio n t l * mixDensity ρ2 l := by
    intro l hl
    have hd : 0 < mixDensity ρ2 l := by unfold mixDensity; positivity
    apply mul_lt_mul_of_pos_right _ hd
    unfold likelihoodRatio
    apply exp_lt_exp.2
    nlinarith
  have hpos : 0 < ∫ l in Ici 0,
      (likelihoodRatio n t l * mixDensity ρ2 l - likelihoodRatio n s l * mixDensity ρ2 l) := by
    have hint : IntegrableOn (fun l => likelihoodRatio n t l * mixDensity ρ2 l
        - likelihoodRatio n s l * mixDensity ρ2 l) (Ici 0) := ht.sub hs
    rw [setIntegral_pos_iff_support_of_nonneg_ae _ hint]
    · have hsub : Ioi (0:ℝ) ⊆ Function.support (fun l =>
          likelihoodRatio n t l * mixDensity ρ2 l - likelihoodRatio n s l * mixDensity ρ2 l)
          ∩ Ici 0 := by
        intro l hl
        refine ⟨?_, mem_Ici.2 (le_of_lt hl)⟩
        rw [Function.mem_support]
        exact (sub_pos.2 (hlt l hl)).ne'
      have := measure_mono (μ := volume) hsub
      rw [Real.volume_Ioi] at this
      exact lt_of_lt_of_le ENNReal.zero_lt_top this
    · filter_upwards [ae_restrict_mem measurableSet_Ici] with l hl
      exact sub_nonneg.2 (hle l hl)
  linarith

/-- **The code's `log_e_standardized` is strictly increasing in `s`** (exact arithmetic, the
`z > -8` formula everywhere). -/
theorem logEStd_strictMono {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) :
    StrictMono (fun s => logEStd n s ρ2) := by
  intro s t hst
  have := eMix_strictMono hρ hn hst
  rw [eMix_eq_exp_logEStd hρ hn, eMix_eq_exp_logEStd hρ hn] at this
  exact exp_lt_exp.1 this

/-- **No sum, no evidence** (Python: `log_e_standardized(n, 0, rho2) = -0.5 log v`):
`e(n, 0) = v^(-1/2)`. -/
theorem eMix_zero {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) :
    eMix ρ2 n 0 = (√(1 + ρ2 * n))⁻¹ := by
  rw [eMix_closed_form hρ hn]
  simp [Phi_zero]

/-- `e(n, 0) = v^(-1/2) ≤ 1`. -/
theorem eMix_zero_le_one {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) : eMix ρ2 n 0 ≤ 1 := by
  rw [eMix_zero hρ hn]
  apply inv_le_one_of_one_le₀
  rw [one_le_sqrt]
  nlinarith [mul_nonneg hρ.le hn]

/-- **The down side is small when the sum is up** (Python: `log_e_record(rec, bar, -1)` when the
total is `≥ 0`): for `s ≥ 0`, `e(n, -s) ≤ 1`. -/
theorem eMix_neg_le_one {ρ2 n s : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hs : 0 ≤ s) :
    eMix ρ2 n (-s) ≤ 1 :=
  ((eMix_strictMono hρ hn).monotone (neg_nonpos.2 hs)).trans (eMix_zero_le_one hρ hn)

lemma eMix_continuous {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) : Continuous (eMix ρ2 n) := by
  have h : eMix ρ2 n = fun s => (√(1 + ρ2 * n))⁻¹ * exp (ρ2 * s ^ 2 / (2 * (1 + ρ2 * n)))
      * (2 * Phi (√ρ2 * s / √(1 + ρ2 * n))) := funext (eMix_closed_form hρ hn)
  rw [h]
  have hP := Phi_continuous
  fun_prop

lemma eMix_ge {ρ2 n s : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hs : 0 ≤ s) :
    (√(1 + ρ2 * n))⁻¹ * exp (ρ2 * s ^ 2 / (2 * (1 + ρ2 * n))) ≤ eMix ρ2 n s := by
  rw [eMix_closed_form hρ hn]
  have hz : 0 ≤ √ρ2 * s / √(1 + ρ2 * n) := by positivity
  have h1 := one_le_two_Phi hz
  have h2 : 0 ≤ (√(1 + ρ2 * n))⁻¹ * exp (ρ2 * s ^ 2 / (2 * (1 + ρ2 * n))) := by positivity
  nlinarith

/-! ## 2. It is an e-value (fixed `n`, known spread) -/

/-- The mixing density integrates to 1 on `l ≥ 0`. -/
theorem integral_mixDensity {ρ2 : ℝ} (hρ : 0 < ρ2) : ∫ l in Ici 0, mixDensity ρ2 l = 1 := by
  have h := eMix_zero hρ (le_refl 0)
  simp only [mul_zero, add_zero, sqrt_one, inv_one] at h
  unfold eMix likelihoodRatio at h
  simpa using h

lemma integrableOn_mixDensity {ρ2 : ℝ} (hρ : 0 < ρ2) : IntegrableOn (mixDensity ρ2) (Ici 0) :=
  Integrable.of_integral_ne_zero (by rw [integral_mixDensity hρ]; exact one_ne_zero)

lemma lintegral_mixDensity {ρ2 : ℝ} (hρ : 0 < ρ2) :
    ∫⁻ l in Ici 0, ENNReal.ofReal (mixDensity ρ2 l) = 1 := by
  rw [← ofReal_integral_eq_lintegral_ofReal (integrableOn_mixDensity hρ)
    (Filter.Eventually.of_forall fun l => mixDensity_nonneg hρ l), integral_mixDensity hρ,
    ENNReal.ofReal_one]

/-- Tonelli and the Gaussian moment generating function: under `N(μ, n)` the average e-value is
`∫_{l ≥ 0} exp (l μ) π(l) dl`. -/
theorem lintegral_eMix_gaussianReal {ρ2 : ℝ} (hρ : 0 < ρ2) (μ : ℝ) (n : NNReal) :
    ∫⁻ x, ENNReal.ofReal (eMix ρ2 n x) ∂(gaussianReal μ n)
      = ∫⁻ l in Ici 0, ENNReal.ofReal (exp (μ * l) * mixDensity ρ2 l) := by
  have hn : (0:ℝ) ≤ n := n.2
  calc ∫⁻ x, ENNReal.ofReal (eMix ρ2 n x) ∂(gaussianReal μ n)
      = ∫⁻ x, (∫⁻ l in Ici 0, ENNReal.ofReal (likelihoodRatio n x l * mixDensity ρ2 l))
          ∂(gaussianReal μ n) := by
        refine lintegral_congr fun x => ?_
        exact ofReal_integral_eq_lintegral_ofReal (integrableOn_integrand hρ hn x)
          (Filter.Eventually.of_forall fun l => integrand_nonneg hρ n x l)
    _ = ∫⁻ l in Ici 0, ∫⁻ x, ENNReal.ofReal (likelihoodRatio n x l * mixDensity ρ2 l)
          ∂(gaussianReal μ n) := by
        have hmeas : Measurable (Function.uncurry fun (x l : ℝ) =>
            ENNReal.ofReal (likelihoodRatio n x l * mixDensity ρ2 l)) := by
          apply Measurable.ennreal_ofReal
          apply Continuous.measurable
          unfold likelihoodRatio mixDensity
          fun_prop
        exact lintegral_lintegral_swap
          (f := fun (x l : ℝ) => ENNReal.ofReal (likelihoodRatio n x l * mixDensity ρ2 l))
          (μ := gaussianReal μ n) (ν := volume.restrict (Ici 0)) hmeas.aemeasurable
    _ = ∫⁻ l in Ici 0, ENNReal.ofReal (exp (μ * l) * mixDensity ρ2 l) := by
        refine lintegral_congr fun l => ?_
        have h1 : (fun x => likelihoodRatio n x l * mixDensity ρ2 l)
            = fun x => exp (l * x) * (exp (-(l ^ 2 * n / 2)) * mixDensity ρ2 l) := by
          ext x; unfold likelihoodRatio; rw [sub_eq_add_neg, exp_add]; ring
        have hint : Integrable (fun x => likelihoodRatio n x l * mixDensity ρ2 l)
            (gaussianReal μ n) := by
          rw [h1]; exact (integrable_exp_mul_gaussianReal l).mul_const _
        rw [← ofReal_integral_eq_lintegral_ofReal hint
          (Filter.Eventually.of_forall fun x => integrand_nonneg hρ n x l)]
        congr 1
        rw [h1, integral_mul_const]
        have h2 : ∫ x, exp (l * x) ∂(gaussianReal μ n) = exp (μ * l + n * l ^ 2 / 2) := by
          have := congrFun (mgf_id_gaussianReal (μ := μ) (v := n)) l
          simpa [mgf] using this
        rw [h2, ← mul_assoc, ← exp_add]
        congr 2
        ring

/-- **It is an e-value under the composite null** (Python: `log_e_standardized`,
`log_e_one_sided`). If the standardized sum `S` has law `N(μ, n)` with `μ ≤ 0` (no edge or a
losing strategy, `n ≥ 0` any variance) then `E[e(n, S)] ≤ 1`.

Exact at a fixed number of trades for Gaussian trades with KNOWN spread (`S = total / σ`). The
code plugs in the sample SD: there it holds only asymptotically. For every `n` at once see
`EdgeOrLuck.Anytime`. -/
theorem eMix_expectation_le_one {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    {ρ2 : ℝ} (hρ : 0 < ρ2) {n : NNReal} {μ : ℝ} (hμ : μ ≤ 0) {S : Ω → ℝ}
    (hS : HasLaw S (gaussianReal μ n) P) :
    ∫ ω, eMix ρ2 n (S ω) ∂P ≤ 1 := by
  have hn : (0:ℝ) ≤ n := n.2
  have hcont := eMix_continuous hρ hn
  have h1 : ∫ ω, eMix ρ2 n (S ω) ∂P = ∫ x, eMix ρ2 n x ∂(gaussianReal μ n) :=
    hS.integral_comp hcont.aestronglyMeasurable
  rw [h1, integral_eq_lintegral_of_nonneg_ae
    (Filter.Eventually.of_forall fun x => (eMix_pos hρ hn x).le) hcont.aestronglyMeasurable,
    lintegral_eMix_gaussianReal hρ μ n]
  apply ENNReal.toReal_le_of_le_ofReal zero_le_one
  rw [ENNReal.ofReal_one, ← lintegral_mixDensity hρ]
  apply setLIntegral_mono' measurableSet_Ici
  intro l hl
  apply ENNReal.ofReal_le_ofReal
  have hl0 : (0:ℝ) ≤ l := hl
  have : exp (μ * l) ≤ 1 := exp_le_one_iff.2 (mul_nonpos_of_nonpos_of_nonneg hμ hl0)
  nlinarith [mixDensity_nonneg hρ l]

/-- **A fair bet when there is no edge** (Python: `log_e_standardized`). If the standardized sum
`S` has law `N(0, n)` then `E[e(n, S)] = 1` exactly (fixed `n`, known spread; see
`eMix_expectation_le_one`). -/
theorem eMix_expectation_eq_one {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    {ρ2 : ℝ} (hρ : 0 < ρ2) {n : NNReal} {S : Ω → ℝ}
    (hS : HasLaw S (gaussianReal 0 n) P) :
    ∫ ω, eMix ρ2 n (S ω) ∂P = 1 := by
  have hn : (0:ℝ) ≤ n := n.2
  have hcont := eMix_continuous hρ hn
  have h1 : ∫ ω, eMix ρ2 n (S ω) ∂P = ∫ x, eMix ρ2 n x ∂(gaussianReal 0 n) :=
    hS.integral_comp hcont.aestronglyMeasurable
  rw [h1, integral_eq_lintegral_of_nonneg_ae
    (Filter.Eventually.of_forall fun x => (eMix_pos hρ hn x).le) hcont.aestronglyMeasurable,
    lintegral_eMix_gaussianReal hρ 0 n]
  simp only [zero_mul, exp_zero, one_mul]
  rw [lintegral_mixDensity hρ, ENNReal.toReal_one]

/-! ## 5. The boundary and the code's bisection -/

/-- The code's upper bracket in `boundary`:
`hi = sqrt(2 v / rho2 * (log(bar) + 0.5 log v + 1)) + 1`. -/
noncomputable def bisectHi (ρ2 n B : ℝ) : ℝ :=
  √(2 * (1 + ρ2 * n) / ρ2 * (log B + 0.5 * log (1 + ρ2 * n) + 1)) + 1

/-- The upper bracket clears the bar with room to spare: `e · B ≤ e(n, hi)` (Python: `boundary`).
Uses `2Φ(z) ≥ 1` for `z ≥ 0`, so `e(n, s) ≥ v^(-1/2) exp (ρ2 s² / (2v))` for `s ≥ 0`. -/
theorem exp_one_mul_le_eMix_bisectHi {ρ2 n B : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B) :
    B * exp 1 ≤ eMix ρ2 n (bisectHi ρ2 n B) := by
  set v := 1 + ρ2 * n with hv_def
  have hv1 : 1 ≤ v := by nlinarith [mul_nonneg hρ.le hn]
  have hv : 0 < v := by linarith
  have hL : 0 < log B + 0.5 * log v + 1 := by
    have := log_pos hB
    have := log_nonneg hv1
    linarith
  set h0 := √(2 * v / ρ2 * (log B + 0.5 * log v + 1)) with hh0
  have h0nn : 0 ≤ h0 := sqrt_nonneg _
  have h0sq : h0 ^ 2 = 2 * v / ρ2 * (log B + 0.5 * log v + 1) :=
    sq_sqrt (by positivity)
  have hhi : bisectHi ρ2 n B = h0 + 1 := rfl
  rw [hhi]
  refine le_trans ?_ (eMix_ge hρ hn (by linarith))
  have hexp_arg : ρ2 * h0 ^ 2 / (2 * v) = log B + 0.5 * log v + 1 := by
    rw [h0sq]; field_simp
  have hmono : ρ2 * h0 ^ 2 / (2 * v) < ρ2 * (h0 + 1) ^ 2 / (2 * v) := by
    apply div_lt_div_of_pos_right _ (by positivity)
    apply mul_lt_mul_of_pos_left _ hρ
    nlinarith
  have key : (√v)⁻¹ * exp (ρ2 * h0 ^ 2 / (2 * v)) = B * exp 1 := by
    rw [hexp_arg, exp_add, exp_add, exp_log (by linarith), sqrt_eq_exp_half_log hv]
    have : (0.5 : ℝ) * log v = 1 / 2 * log v := by norm_num
    rw [this]
    field_simp
  calc B * exp 1 = (√v)⁻¹ * exp (ρ2 * h0 ^ 2 / (2 * v)) := key.symm
    _ ≤ (√v)⁻¹ * exp (ρ2 * (h0 + 1) ^ 2 / (2 * v)) := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          exact exp_le_exp.2 hmono.le

lemma lt_eMix_bisectHi {ρ2 n B : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B) :
    B < eMix ρ2 n (bisectHi ρ2 n B) := by
  have h := exp_one_mul_le_eMix_bisectHi hρ hn hB
  have he : 1 < exp (1:ℝ) := by have := add_one_lt_exp (one_ne_zero (α := ℝ)); linarith
  have hB0 : 0 < B := by linarith
  nlinarith

lemma eMix_zero_lt {ρ2 n B : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B) : eMix ρ2 n 0 < B :=
  lt_of_le_of_lt (eMix_zero_le_one hρ hn) hB

/-- **The code's bisection bracket is valid** (Python: `boundary`): for `B > 1`,
`e(n, lo) < B < e(n, hi)` with `lo = 0` and `hi = bisectHi`. -/
theorem boundary_bracket {ρ2 n B : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B) :
    eMix ρ2 n 0 < B ∧ B < eMix ρ2 n (bisectHi ρ2 n B) :=
  ⟨eMix_zero_lt hρ hn hB, lt_eMix_bisectHi hρ hn hB⟩

/-- **The boundary exists and is unique** (Python: `boundary`, `half_width`): for `B > 1`
there is exactly one `b ≥ 0` with `e(n, b) = B` (continuity, intermediate values on
`[0, hi]`, strict monotonicity). -/
theorem boundary_exists_unique {ρ2 n B : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B) :
    ∃! b : ℝ, 0 ≤ b ∧ eMix ρ2 n b = B := by
  have hhi : 0 ≤ bisectHi ρ2 n B := by unfold bisectHi; positivity
  obtain ⟨b, hb, hbB⟩ := intermediate_value_Icc hhi (eMix_continuous hρ hn).continuousOn
    ⟨(eMix_zero_lt hρ hn hB).le, (lt_eMix_bisectHi hρ hn hB).le⟩
  refine ⟨b, ⟨hb.1, hbB⟩, fun b' hb' => ?_⟩
  exact (eMix_strictMono hρ hn).injective (hb'.2.trans hbB.symm)

/-- The boundary is positive: any `b` with `e(n, b) = B > 1` has `b > 0`. -/
theorem boundary_pos {ρ2 n B b : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B)
    (hb : eMix ρ2 n b = B) : 0 < b := by
  by_contra h
  have := (eMix_strictMono hρ hn).monotone (not_lt.1 h)
  have := eMix_zero_lt hρ hn hB
  linarith

/-- One step of the code's bisection: keep the half where `f(mid) ≥ target` changes. -/
noncomputable def bisectStep (f : ℝ → ℝ) (target : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  if target ≤ f ((p.1 + p.2) / 2) then (p.1, (p.1 + p.2) / 2) else ((p.1 + p.2) / 2, p.2)

/-- `k` steps of the code's bisection from `(lo, hi)`. -/
noncomputable def bisect (f : ℝ → ℝ) (target lo hi : ℝ) : ℕ → ℝ × ℝ
  | 0 => (lo, hi)
  | k + 1 => bisectStep f target (bisect f target lo hi k)

/-- The bisection invariant (Python: `boundary`'s loop): if `f lo < target ≤ f hi` then after
every step `f lo_k < target ≤ f hi_k` and the bracket has width `(hi - lo) / 2^k`. -/
theorem bisect_invariant (f : ℝ → ℝ) (target lo hi : ℝ) (hlo : f lo < target)
    (hhi : target ≤ f hi) (k : ℕ) :
    f (bisect f target lo hi k).1 < target ∧ target ≤ f (bisect f target lo hi k).2 ∧
      (bisect f target lo hi k).2 - (bisect f target lo hi k).1 = (hi - lo) / 2 ^ k := by
  induction k with
  | zero => simp [bisect, hlo, hhi]
  | succ k ih =>
    obtain ⟨h1, h2, h3⟩ := ih
    simp only [bisect, bisectStep]
    split_ifs with h
    · refine ⟨h1, h, ?_⟩
      simp only
      rw [pow_succ, ← div_div, ← h3]
      ring
    · refine ⟨lt_of_not_ge h, h2, ?_⟩
      simp only
      rw [pow_succ, ← div_div, ← h3]
      ring

/-- **The code's bisection converges to the boundary** (Python: `boundary`): with
`f = log_e_standardized(n, ·, rho2)`, `target = log B`, `lo = 0`, `hi = bisectHi`, the upper end
after `k` steps lies in `[b, b + hi / 2^k]`, where `b` is the boundary (`e(n, b) = B`, `B > 1`). -/
theorem boundary_bisection {ρ2 n B b : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B)
    (hb : eMix ρ2 n b = B) (k : ℕ) :
    b ≤ (bisect (fun s => logEStd n s ρ2) (log B) 0 (bisectHi ρ2 n B) k).2 ∧
    (bisect (fun s => logEStd n s ρ2) (log B) 0 (bisectHi ρ2 n B) k).2
      ≤ b + bisectHi ρ2 n B / 2 ^ k := by
  have hB0 : 0 < B := by linarith
  have hlogb : logEStd n b ρ2 = log B := by
    rw [← hb, eMix_eq_exp_logEStd hρ hn, log_exp]
  have hlo : logEStd n 0 ρ2 < log B := by
    have := eMix_zero_lt hρ hn hB
    rw [eMix_eq_exp_logEStd hρ hn] at this
    exact (lt_log_iff_exp_lt hB0).2 this
  have hhi : log B ≤ logEStd n (bisectHi ρ2 n B) ρ2 := by
    have := (lt_eMix_bisectHi hρ hn hB).le
    rw [eMix_eq_exp_logEStd hρ hn] at this
    exact (log_le_iff_le_exp hB0).2 this
  obtain ⟨h1, h2, h3⟩ := bisect_invariant (fun s => logEStd n s ρ2) (log B) 0
    (bisectHi ρ2 n B) hlo hhi k
  have hmono := logEStd_strictMono hρ hn
  have hlb : (bisect (fun s => logEStd n s ρ2) (log B) 0 (bisectHi ρ2 n B) k).1 < b :=
    hmono.lt_iff_lt.1 (h1.trans_eq hlogb.symm)
  have hub : b ≤ (bisect (fun s => logEStd n s ρ2) (log B) 0 (bisectHi ρ2 n B) k).2 :=
    hmono.le_iff_le.1 (hlogb.trans_le h2)
  refine ⟨hub, ?_⟩
  rw [sub_zero] at h3
  linarith

/-- What `boundary(n, bar, rho2)` returns, in exact arithmetic: the upper end after 80 steps. -/
noncomputable def boundaryCode (ρ2 n B : ℝ) : ℝ :=
  (bisect (fun s => logEStd n s ρ2) (log B) 0 (bisectHi ρ2 n B) 80).2

/-- **`boundary` returns the boundary to within `hi / 2^80`** (Python: `boundary`):
`b ≤ boundary(n, B, rho2) ≤ b + hi / 2^80`. -/
theorem boundaryCode_spec {ρ2 n B b : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (hB : 1 < B)
    (hb : eMix ρ2 n b = B) :
    b ≤ boundaryCode ρ2 n B ∧ boundaryCode ρ2 n B ≤ b + bisectHi ρ2 n B / 2 ^ 80 :=
  boundary_bisection hρ hn hB hb 80

/-! ## 6. The range is the inversion of the two one-sided tests -/

/-- The lower end of `range_of`: `mean - up * sd / n` with `up = b + shiftUp`. -/
noncomputable def rangeLower (n T sd b shiftUp : ℝ) : ℝ := T / n - (b + shiftUp) * sd / n

/-- The upper end of `range_of`: `mean + down * sd / n` with `down = b + shiftDown`. -/
noncomputable def rangeUpper (n T sd b shiftDown : ℝ) : ℝ := T / n + (b + shiftDown) * sd / n

/-- **"Above `μ0`" is called exactly when `μ0` is at or under the range** (Python: `range_of`,
`log_e_record(+1)`): the e-value for "the average is above `μ0`", with standardized sum
`(T - n μ0)/sd - shift`, reaches `B` iff `μ0 ≤ T/n - (b + shift)·sd/n`. Needs `n > 0`, `sd > 0`
and `b` the boundary (`e(n, b) = B`). -/
theorem above_rejects_iff {ρ2 n sd b B : ℝ} (hρ : 0 < ρ2) (hn : 0 < n) (hsd : 0 < sd)
    (hb : eMix ρ2 n b = B) (T shift μ0 : ℝ) :
    B ≤ eMix ρ2 n ((T - n * μ0) / sd - shift) ↔ μ0 ≤ rangeLower n T sd b shift := by
  rw [← hb, (eMix_strictMono hρ hn.le).le_iff_le]
  have e1 : b ≤ (T - n * μ0) / sd - shift ↔ (b + shift) * sd ≤ T - n * μ0 := by
    rw [le_sub_iff_add_le, le_div_iff₀ hsd]
  have e2 : μ0 ≤ rangeLower n T sd b shift ↔ μ0 * n ≤ T - (b + shift) * sd := by
    rw [rangeLower, show T / n - (b + shift) * sd / n = (T - (b + shift) * sd) / n by ring,
      le_div_iff₀ hn]
  rw [e1, e2]
  constructor <;> intro h <;> linarith

/-- **"Below `μ0`" is called exactly when `μ0` is at or over the range** (Python: `range_of`,
`log_e_record(-1)`): with standardized sum `(n μ0 - T)/sd - shift`, the e-value reaches `B` iff
`T/n + (b + shift)·sd/n ≤ μ0`. -/
theorem below_rejects_iff {ρ2 n sd b B : ℝ} (hρ : 0 < ρ2) (hn : 0 < n) (hsd : 0 < sd)
    (hb : eMix ρ2 n b = B) (T shift μ0 : ℝ) :
    B ≤ eMix ρ2 n ((n * μ0 - T) / sd - shift) ↔ rangeUpper n T sd b shift ≤ μ0 := by
  rw [← hb, (eMix_strictMono hρ hn.le).le_iff_le]
  have e1 : b ≤ (n * μ0 - T) / sd - shift ↔ (b + shift) * sd ≤ n * μ0 - T := by
    rw [le_sub_iff_add_le, le_div_iff₀ hsd]
  have e2 : rangeUpper n T sd b shift ≤ μ0 ↔ T + (b + shift) * sd ≤ μ0 * n := by
    rw [rangeUpper, show T / n + (b + shift) * sd / n = (T + (b + shift) * sd) / n by ring,
      div_le_iff₀ hn]
  rw [e1, e2]
  constructor <;> intro h <;> linarith

/-- **The range is the set of means neither one-sided test rejects** (Python: `range_of`,
docstring "where the one-sided e-value against that average reaches `bar`"): `μ0` lies strictly
inside `range_of` iff both one-sided e-values against `μ0` (each with its side's skew shift) stay
below `B`. -/
theorem range_eq_not_rejected {ρ2 n sd b B : ℝ} (hρ : 0 < ρ2) (hn : 0 < n) (hsd : 0 < sd)
    (hb : eMix ρ2 n b = B) (T shiftUp shiftDown μ0 : ℝ) :
    μ0 ∈ Ioo (rangeLower n T sd b shiftUp) (rangeUpper n T sd b shiftDown) ↔
      eMix ρ2 n ((T - n * μ0) / sd - shiftUp) < B ∧
        eMix ρ2 n ((n * μ0 - T) / sd - shiftDown) < B := by
  have hA := above_rejects_iff hρ hn hsd hb T shiftUp μ0
  have hBl := below_rejects_iff hρ hn hsd hb T shiftDown μ0
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨lt_of_not_ge fun h => (not_le.2 h1) (hA.1 h),
      lt_of_not_ge fun h => (not_le.2 h2) (hBl.1 h)⟩
  · rintro ⟨h1, h2⟩
    exact ⟨lt_of_not_ge fun h => (not_le.2 h1) (hA.2 h),
      lt_of_not_ge fun h => (not_le.2 h2) (hBl.2 h)⟩

/-- **Range ⟺ verdict** (Python: the module docstring "the whole 95% range is above zero and
the verdict real_edge are the same statement"; `log_e_record(rec, bar, +1) >= log(bar)` vs
`range_of(rec, bar)[0]`): for `n > 0` trades, total `T`, spread `sd > 0`, any shift and `b` the
boundary: `e(n, T/sd - shift) ≥ B ⟺ T/n - (b + shift)·sd/n ≥ 0`. -/
theorem verdict_iff_range_clears_zero {ρ2 n sd b B : ℝ} (hρ : 0 < ρ2) (hn : 0 < n)
    (hsd : 0 < sd) (hb : eMix ρ2 n b = B) (T shift : ℝ) :
    B ≤ eMix ρ2 n (T / sd - shift) ↔ 0 ≤ rangeLower n T sd b shift := by
  have := above_rejects_iff hρ hn hsd hb T shift 0
  rwa [mul_zero, sub_zero] at this

/-- The same for "below zero" (Python: `log_e_record(rec, bar, -1)` vs `range_of(rec, bar)[1]`):
`e(n, -T/sd - shift) ≥ B ⟺ T/n + (b + shift)·sd/n ≤ 0`. -/
theorem down_verdict_iff_range_below_zero {ρ2 n sd b B : ℝ} (hρ : 0 < ρ2) (hn : 0 < n)
    (hsd : 0 < sd) (hb : eMix ρ2 n b = B) (T shift : ℝ) :
    B ≤ eMix ρ2 n (-T / sd - shift) ↔ rangeUpper n T sd b shift ≤ 0 := by
  have := below_rejects_iff hρ hn hsd hb T shift 0
  rwa [mul_zero, zero_sub] at this

/-- **A pass needs a positive total** (Python: `passes_up = up and mean > 0` in `judge.py`; the
`mean > 0` is implied): with `B > 1` and a shift `≥ 0`, `e(n, T/sd - shift) ≥ B` forces `T > 0`. -/
theorem pass_implies_total_pos {ρ2 n sd b B T shift : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n)
    (hsd : 0 < sd) (hB : 1 < B) (hb : eMix ρ2 n b = B) (hshift : 0 ≤ shift)
    (hpass : B ≤ eMix ρ2 n (T / sd - shift)) : 0 < T := by
  have hbpos := boundary_pos hρ hn hB hb
  have h1 : b ≤ T / sd - shift := (eMix_strictMono hρ hn).le_iff_le.1 (hb.trans_le hpass)
  have h2 : 0 < T / sd := by linarith
  exact (div_pos_iff_of_pos_right hsd).1 h2

/-! ## 7. Skew never lowers a bar -/

/-- The skew against a claim in direction `sign` (Python: `against = -skew if sign > 0 else skew`). -/
def against (skew : ℝ) (sign : ℤ) : ℝ := if 0 < sign then -skew else skew

/-- The code's `skew_shift(skew, n, bar, rho2, sign)` with `SKEW_ADJUST` on and `SKEW_CAP = 6`;
`z` stands for the bar's z, `boundary(n, bar, rho2) / sqrt(n)` (any real number here). -/
noncomputable def skewShift (skew : ℝ) (n : ℕ) (z : ℝ) (sign : ℤ) : ℝ :=
  if against skew sign ≤ 0 ∨ n < 3 then 0 else min (against skew sign) 6 * (2 * z ^ 2 + 1) / 6

/-- **`skew_shift ≥ 0`** (Python: `skew_shift`) for every skew, trade count, bar and side. -/
theorem skewShift_nonneg (skew : ℝ) (n : ℕ) (z : ℝ) (sign : ℤ) :
    0 ≤ skewShift skew n z sign := by
  unfold skewShift
  split_ifs with h
  · exact le_rfl
  · have ha : 0 < against skew sign := lt_of_not_ge fun h' => h (Or.inl h')
    have hm : 0 < min (against skew sign) 6 := lt_min ha (by norm_num)
    have hz : 0 < 2 * z ^ 2 + 1 := by positivity
    exact div_nonneg (mul_nonneg hm.le hz.le) (by norm_num)

/-- Without skew there is no shift (Python: `skew_shift(0, ...) = 0`). -/
theorem skewShift_zero_skew (n : ℕ) (z : ℝ) (sign : ℤ) : skewShift 0 n z sign = 0 := by
  unfold skewShift against
  split_ifs <;> simp_all

/-- **Skew never raises an e-value** (Python: `log_e_record`): the skew-adjusted e-value
`e(n, s - skew_shift)` is never above the unadjusted `e(n, s)`; a claim that passes with the skew
correction passes without it. -/
theorem skew_never_raises_e {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 ≤ n) (s skew : ℝ) (m : ℕ)
    (z : ℝ) (sign : ℤ) : eMix ρ2 n (s - skewShift skew m z sign) ≤ eMix ρ2 n s :=
  (eMix_strictMono hρ hn).monotone (by linarith [skewShift_nonneg skew m z sign])

/-- **The skew-adjusted range contains the unadjusted one** (Python: `range_of`, widened "on the
side the skew argues against"): for shifts `≥ 0`, `n > 0`, `sd > 0`. -/
theorem range_with_skew_contains {n sd : ℝ} (hn : 0 < n) (hsd : 0 < sd) {shiftUp shiftDown : ℝ}
    (hu : 0 ≤ shiftUp) (hd : 0 ≤ shiftDown) (T b : ℝ) :
    Icc (rangeLower n T sd b 0) (rangeUpper n T sd b 0) ⊆
      Icc (rangeLower n T sd b shiftUp) (rangeUpper n T sd b shiftDown) := by
  apply Icc_subset_Icc
  · unfold rangeLower
    have : (b + 0) * sd / n ≤ (b + shiftUp) * sd / n :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (by linarith) hsd.le) hn.le
    linarith
  · unfold rangeUpper
    have : (b + 0) * sd / n ≤ (b + shiftDown) * sd / n :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (by linarith) hsd.le) hn.le
    linarith

/-! ## 8. The tuning: exact scale-free form, heuristic constant -/

/-- **Only `ρ2·n` and `s/√n` matter** (Python: `log_e_standardized`, `mixing_rho2`): for `n > 0`,
`log_e_standardized(n, s, rho2) = log_e_standardized(1, s/√n, rho2·n)`. So the boundary in
z units, `b/√n`, is a function of `ρ2·n` and `B` alone, and the `ρ2` that minimizes it at `n = t*`
is exactly `c(B)/t*`. That `c(B) = 2 log B` (`mixing_rho2`) is a heuristic; see the module
docstring for how close it is. -/
theorem logEStd_scale_free {ρ2 n : ℝ} (hρ : 0 < ρ2) (hn : 0 < n) (s : ℝ) :
    logEStd n s ρ2 = logEStd 1 (s / √n) (ρ2 * n) := by
  unfold logEStd
  have hsn : 0 < √n := sqrt_pos.2 hn
  have h1 : ρ2 * n * (s / √n) ^ 2 = ρ2 * s ^ 2 := by
    rw [div_pow, sq_sqrt hn.le]; field_simp
  have h2 : √(ρ2 * n) * (s / √n) = √ρ2 * s := by
    rw [sqrt_mul hρ.le]; field_simp
  rw [mul_one, h1, h2]

/-! ## 9. The range at a fixed number of trades -/

/-- A wider spread gives a wider range (Python: `record_of` widens the SD by the serial factor
`≥ 1`; `range_of` scales with `sd`). -/
theorem range_mono_sd {n σ sd' b sU sD : ℝ} (hn : 0 < n) (hσ : σ ≤ sd') (hU : 0 ≤ b + sU)
    (hD : 0 ≤ b + sD) (T : ℝ) :
    Ioo (rangeLower n T σ b sU) (rangeUpper n T σ b sD) ⊆
      Ioo (rangeLower n T sd' b sU) (rangeUpper n T sd' b sD) := by
  apply Ioo_subset_Ioo
  · unfold rangeLower
    have : (b + sU) * σ / n ≤ (b + sU) * sd' / n :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hσ hU) hn.le
    linarith
  · unfold rangeUpper
    have : (b + sD) * σ / n ≤ (b + sD) * sd' / n :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hσ hD) hn.le
    linarith

/-- **The range covers the true mean with probability `≥ 1 - 2/B` at a fixed trade count**
(Python: `range_of`, `half_width` "a two-sided range at level 1 - 2 / bar"). Assumptions: `n > 0`
trades whose standardized total `(T - n m)/σ` has law `N(0, n)` (Gaussian trades with mean `m`
and KNOWN spread `σ > 0`), `b` the boundary for `B > 1`, any spread `sd ≥ σ` and any skew shifts
`≥ 0` (all may depend on the data). The code uses the sample SD, which is not `≥ σ` in general:
there the statement holds only asymptotically. For every `n` at once see
`EdgeOrLuck.Anytime.anytime_range`. -/
theorem range_covers_fixed_n {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {ρ2 B b : ℝ} (hρ : 0 < ρ2) (hB : 1 < B) {n : NNReal}
    (hn : 0 < (n : ℝ)) (hb : eMix ρ2 n b = B) {σ m : ℝ} (hσ : 0 < σ) {T : Ω → ℝ}
    (hT : HasLaw (fun ω => (T ω - n * m) / σ) (gaussianReal 0 n) P)
    (sd : Ω → ℝ) (hsd : ∀ ω, σ ≤ sd ω)
    (shiftUp shiftDown : Ω → ℝ) (hu : ∀ ω, 0 ≤ shiftUp ω) (hd : ∀ ω, 0 ≤ shiftDown ω) :
    P.real {ω | m ∉ Ioo (rangeLower n (T ω) (sd ω) b (shiftUp ω))
      (rangeUpper n (T ω) (sd ω) b (shiftDown ω))} ≤ 2 / B := by
  set S0 : Ω → ℝ := fun ω => (T ω - n * m) / σ with hS0
  have hB0 : 0 < B := by linarith
  have hbpos := boundary_pos hρ hn.le hB hb
  have hmono := (eMix_strictMono hρ hn.le).monotone
  have hsub : {ω | m ∉ Ioo (rangeLower n (T ω) (sd ω) b (shiftUp ω))
      (rangeUpper n (T ω) (sd ω) b (shiftDown ω))} ⊆
      {ω | B ≤ eMix ρ2 n (S0 ω)} ∪ {ω | B ≤ eMix ρ2 n ((-S0) ω)} := by
    intro ω hω
    have hω' : m ∉ Ioo (rangeLower n (T ω) (sd ω) b (shiftUp ω))
        (rangeUpper n (T ω) (sd ω) b (shiftDown ω)) := hω
    have hnarrow : m ∉ Ioo (rangeLower n (T ω) σ b (shiftUp ω))
        (rangeUpper n (T ω) σ b (shiftDown ω)) := fun h =>
      hω' (range_mono_sd hn (hsd ω) (by linarith [hu ω]) (by linarith [hd ω]) (T ω) h)
    rw [range_eq_not_rejected hρ hn hσ hb, not_and_or, not_lt, not_lt] at hnarrow
    rcases hnarrow with h | h
    · left
      exact h.trans (hmono (by linarith [hu ω]))
    · right
      refine h.trans (hmono ?_)
      have : (n * m - T ω) / σ = (-S0) ω := by
        simp only [hS0, Pi.neg_apply]; ring
      linarith [hd ω]
  have hmarkov : ∀ S : Ω → ℝ, HasLaw S (gaussianReal 0 n) P →
      P.real {ω | B ≤ eMix ρ2 n (S ω)} ≤ 1 / B := by
    intro S hS
    have hE := eMix_expectation_eq_one hρ hS
    have hint : Integrable (fun ω => eMix ρ2 n (S ω)) P :=
      Integrable.of_integral_ne_zero (by rw [hE]; exact one_ne_zero)
    have := mul_meas_ge_le_integral_of_nonneg
      (Filter.Eventually.of_forall fun ω => (eMix_pos hρ hn.le (S ω)).le) hint B
    rw [hE] at this
    rw [le_div_iff₀ hB0]
    linarith
  have hneg : HasLaw (-S0) (gaussianReal 0 n) P := by
    have := gaussianReal_neg hT
    rwa [neg_zero] at this
  calc P.real {ω | m ∉ Ioo (rangeLower n (T ω) (sd ω) b (shiftUp ω))
        (rangeUpper n (T ω) (sd ω) b (shiftDown ω))}
      ≤ P.real ({ω | B ≤ eMix ρ2 n (S0 ω)} ∪ {ω | B ≤ eMix ρ2 n ((-S0) ω)}) :=
        measureReal_mono hsub
    _ ≤ P.real {ω | B ≤ eMix ρ2 n (S0 ω)} + P.real {ω | B ≤ eMix ρ2 n ((-S0) ω)} :=
        measureReal_union_le _ _
    _ ≤ 1 / B + 1 / B := add_le_add (hmarkov S0 hT) (hmarkov (-S0) hneg)
    _ = 2 / B := by ring

end EdgeOrLuck.Mixture

#print axioms EdgeOrLuck.Mixture.Phi_eq_cdf
#print axioms EdgeOrLuck.Mixture.Phi_zero
#print axioms EdgeOrLuck.Mixture.eMix_closed_form
#print axioms EdgeOrLuck.Mixture.eMix_eq_exp_logEStd
#print axioms EdgeOrLuck.Mixture.eMix_expectation_le_one
#print axioms EdgeOrLuck.Mixture.eMix_expectation_eq_one
#print axioms EdgeOrLuck.Mixture.eMix_strictMono
#print axioms EdgeOrLuck.Mixture.logEStd_strictMono
#print axioms EdgeOrLuck.Mixture.eMix_zero
#print axioms EdgeOrLuck.Mixture.eMix_zero_le_one
#print axioms EdgeOrLuck.Mixture.eMix_neg_le_one
#print axioms EdgeOrLuck.Mixture.boundary_bracket
#print axioms EdgeOrLuck.Mixture.exp_one_mul_le_eMix_bisectHi
#print axioms EdgeOrLuck.Mixture.boundary_exists_unique
#print axioms EdgeOrLuck.Mixture.boundary_pos
#print axioms EdgeOrLuck.Mixture.bisect_invariant
#print axioms EdgeOrLuck.Mixture.boundary_bisection
#print axioms EdgeOrLuck.Mixture.boundaryCode_spec
#print axioms EdgeOrLuck.Mixture.above_rejects_iff
#print axioms EdgeOrLuck.Mixture.below_rejects_iff
#print axioms EdgeOrLuck.Mixture.range_eq_not_rejected
#print axioms EdgeOrLuck.Mixture.verdict_iff_range_clears_zero
#print axioms EdgeOrLuck.Mixture.down_verdict_iff_range_below_zero
#print axioms EdgeOrLuck.Mixture.pass_implies_total_pos
#print axioms EdgeOrLuck.Mixture.skewShift_nonneg
#print axioms EdgeOrLuck.Mixture.skewShift_zero_skew
#print axioms EdgeOrLuck.Mixture.skew_never_raises_e
#print axioms EdgeOrLuck.Mixture.range_with_skew_contains
#print axioms EdgeOrLuck.Mixture.logEStd_scale_free
#print axioms EdgeOrLuck.Mixture.range_mono_sd
#print axioms EdgeOrLuck.Mixture.range_covers_fixed_n
