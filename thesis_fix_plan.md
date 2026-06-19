# Thesis Revision Plan — Scout & Refine Framework

Status of the committee-style critique. Two tracks:
- **Track A** — text-only fixes
- **Track B** — code experiments

---

## Status Summary

**Track A: ALL 14 items closed.** ✅

**Track B:**
- ✅ B1 (R-metric figure)
- ✅ B2 (Phase 1 wall-clock — merged into A7 table)
- ✅ B3 (matched-budget control — the decisive experiment)
- ⏳ B4, B5, B6, B7, B8, B9 — all optional; the thesis is defense-ready without them, but each closes an additional examiner question.

**Next-up recommendation:** **B5** (small injection campaign at 150 Mpc). It is the single highest-leverage remaining experiment — converts the existence claim into a population-level statement and produces a PP-plot-style figure that pre-empts the "single-injection generalization" question.

---

## Track A — Text-Only Fixes (all closed)

For audit purposes, here's the disposition of each item.

| # | Item | Status | Notes |
|---|---|---|---|
| A1 | Reframe motivating premise | ✅ | Rewritten around low-latency-to-catalog duplication; Chaudhary 2024, Sun 2020, Biscoveanu 2020 cited |
| A2 | Formal reweighted-evidence estimator | ✅ | Eq. (truncated_prior) + Eq. (reweighted_Z); Geyer 1992, Owen 2013; Berger 2006 rebuttal of "use the data twice" |
| A3 | 3σ_scout justification | ✅ | Chebyshev/Cantelli distribution-free bound + empirical skewness/kurtosis + adaptive-quantile alternative |
| A4 | Evidence-threshold fallback rule | ✅ | Boxed rule with `ln Z_ref(ρ) = ½ρ² − C(ρ)`, τ = 5, FP ~ 10⁻³, FN bound |
| A5 | Novelty vs. `bilby_pipe` | ✅ | Four-axis comparison + "what is genuinely new" paragraph |
| A6 | Multi-messenger qualification | ✅ | §7.5 explicitly conjectural; Conclusion paragraph also hedged |
| A7 | Phase 1 wall-clock table | ✅ | Table 5 populated with real per-distance Scout times; CPU vs. wall-clock distinction made explicit |
| A8 | 7× scaling derivation | ✅ | 3.5× from ln(V_prior/V_posterior) growth × 2× from precessing waveform cost; Skilling 2006, Higson 2019, Romero-Shaw 2020, Pratten 2020 |
| A9 | Noise injection clarification | ✅ | Zero-noise injection stated explicitly; ln(BF) ≈ ρ²/2 sanity check |
| A10 | Gradient-descent scope note | ✅ | §4 "Scope note (added at submission)" line |
| A11 | Transition-boundary metric | ✅ | R = \|Δ ln BF\| / σ_pooled replaces relative error throughout |
| A12 | Literature comparison | ✅ | Quantitative speedups per method + operational regime statements |
| A13 | Bilby version caveat | ✅ | Appendix A note; version-independence of relative savings |
| A14 | N=3 statistics | ✅ | t-statistic, 90% CI, fallback sign-test (one-tailed p ≈ 10⁻³) |

---

## Track B — Remaining Code Experiments (all optional)

Sorted shortest run-time → longest. The thesis is defense-ready without any of these; each one closes an additional examiner question.

### B4. [~2–3 days] Sky-localization free run at one distance
**Run-time:** 10D run, ≈ 8–16h × 3 iterations × 2 methods = 50–100 CPU-hours.
**What:** At 150 Mpc, repeat Baseline + Phase 2 with RA, Dec, ψ free.
**Why:** Lets you remove the "this claim is conjectural" disclaimer from §7.5 and the Conclusion. Quantifies the sky-area shrinkage in deg² at 90% confidence for both methods.
**Setup:** Add RA (uniform), Dec (cos), ψ (uniform) to the free parameters; `IMRPhenomXPHM`, 150 Mpc injection; N=3 each.
**Deliverable:** New §6.4.x with sky-area numbers and an extended corner plot.

### B5. [~3–5 days] **Small injection campaign at 150 Mpc (RECOMMENDED NEXT)**
**Run-time:** 20 injections × 2 methods × ~4h ≈ 160 CPU-hours.
**What:** Sample 20 injections at 150 Mpc varying q ∈ [0.3, 1.0] and χ_eff ∈ [–0.5, 0.5] (aligned spins only). Run Baseline + Phase 2.
**Why:** This is the single highest-leverage remaining experiment. Converts the existence claim into a population-level statement. The committee question "does this generalize beyond your one source?" becomes "here's the Δ_param distribution over 20 sources."
**Setup:** Quasi-random grid in (q, χ_eff); `IMRPhenomD` or `IMRPhenomXAS` (aligned-spin, cheap); per-injection Δ_param computation; box-plot output.
**Deliverable:** New §6.5 with a Δ_param box-plot per parameter — turns the defense from "well-defended single point" into "characterized across the BBH population."

### B6. [~5–7 days] Full 15-parameter demonstration at one distance
**Run-time:** 15D with `IMRPhenomXPHM` ≈ 20–30h per run × 2 methods × 3 iterations = ~150 CPU-hours.
**What:** At 150 Mpc, run Baseline + Phase 2 with all 15 parameters free (including 3D spins for both BHs).
**Why:** Tests whether single-Gaussian truncation survives in the production dimensionality and the q–χ_eff degeneracy region. De-conjectures the 6D-to-15D extrapolation in §6.1.
**Setup:** All 15 parameters free with standard priors; extend Phase 1 settings as needed (e.g., `nlive=750` for Scout); watch for posterior multimodality.
**Deliverable:** New §6.6.

### B7. [~5–7 days] Real-event validation (GW150914 or GW170817)
**Run-time:** Similar to B6 if full 15D; lower if aligned-spin.
**What:** Public LIGO data + PSDs from GWOSC; Phase 1 on preliminary PSD, Phase 2 on final PSD; compare to published GWTC posteriors.
**Why:** Most rigorous unbiasedness test — does it work on real noise with real glitches?
**Setup:** GW150914 strain + PSD from GWOSC; both preliminary and final calibration versions where available; posterior comparison against GWTC-1/3.
**Deliverable:** New §6.7. Substantial credibility upgrade if it works.

### B8. [optional, ~3–5 days] Glitch / non-Gaussian noise robustness
**Run-time:** ~50–100 CPU-hours.
**What:** Inject a known glitch (e.g., blip glitch from gravity-spy) near the signal; re-run Phase 1 + Phase 2 at 150 Mpc.
**Why:** Tests the §7.6.1 catastrophic-mis-characterization concern and exercises the §7.6.1 fallback rule defined in A4.
**Setup:** BayesWave or Bilby glitch models; compare Phase 1 posterior under glitch vs. clean noise.
**Deliverable:** Empirical paragraph in §7.6.1.

### B9. [optional, ~5+ days] Hybrid waveform approximant demo
**Run-time:** Mostly engineering + one full run set.
**What:** Scout with `IMRPhenomD` (aligned-spin, cheap); Refine with `IMRPhenomXPHM` (precessing, expensive).
**Why:** Demonstrates the §7.3 hybrid-approximant claim quantitatively.
**Setup:** Modify Phase 1 to use cheap approximant; verify posterior overlap across approximants.
**Deliverable:** New §6.8.

---

## Suggested Order (if you have time)

1. **B5** — population generalization. Best return on compute time.
2. **B4** — sky-free run. Lets you remove the §7.5 conjectural disclaimer.
3. **B6** — full 15D run. Closes the dimensionality-extrapolation question.
4. **B7** — real event. Strongest possible vindication, but requires more setup (data acquisition, PSD versioning).
5. **B8, B9** — only if you want to overshoot.

## If you do nothing else

The thesis is defense-ready as-is. The remaining items are all about **extending** the claims, not **defending** the ones already made. The matched-budget control experiment (B3) closed the single most consequential examiner question. Everything else is a victory-lap addition.
