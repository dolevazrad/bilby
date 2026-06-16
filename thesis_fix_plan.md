# Thesis Revision Plan — Scout & Refine Framework

Address the critique from the committee-style review. Two parallel tracks:
- **Track A** — text-only fixes (no new compute needed; do while code runs)
- **Track B** — code experiments (sorted shortest → longest run time)

Each item is self-contained so it can be opened as its own chat.

---

## Track A — Text-Only Fixes (do in parallel with code)

Ordered by impact on the defense, not by time. Each is ~1–4 hours of writing/citation work.

### A1. Reframe the motivating premise (HIGH PRIORITY)
The "6-month calibration delay" framing is the single biggest exposure. Current text implies LIGO embargoes PE until final calibration; in reality low-latency pipelines run within hours of detection on preliminary PSDs.

**Action:**
- Read Biscoveanu et al. 2020 (PRD 102, 023008) carefully and engage with it explicitly.
- Read LIGO calibration uncertainty papers (e.g., Sun et al. 2020 / Vitale et al. 2021) and quote actual preliminary-vs-final PSD differences (typically <5% amplitude, <few degrees phase).
- Rewrite §1 and §3 to either (a) reframe around "early-data PE with subsequent PSD refinement" — a real and defensible problem — or (b) explicitly limit claims to the catalog re-analysis use case where calibration *does* change before re-release.
- Remove or weaken the "18-month catalog delay = calibration-limited" assertion.

### A2. Write down the formal reweighted-evidence estimator
The Bayesian rigor is currently hand-waved ("Importance Re-weighting of the Phase 2 evidence").

**Action:**
- Add one paragraph in §6.2 / §5 with the explicit equation: how Phase 2 evidence is corrected for the prior swap, what the bias is in the limit of vanishing Scout-posterior tails, and the formal claim of consistency with the blind posterior.
- Cite the importance-sampling literature (e.g., Geyer 1992; Owen 2013 *Monte Carlo*).
- Address the "use the data twice" concern explicitly.

### A3. Justify the 3σ_scout truncation
"3σ captures 99.7% under Gaussian" undercuts itself — the whole reason to sample is non-Gaussianity.

**Action:**
- Argue empirically: show distribution of Phase 1 posterior shapes (Gaussianity check via skew/kurtosis or KS test).
- Define an adaptive criterion: e.g., truncate at the 99th percentile of Scout samples rather than 3σ. Even if you don't re-run, frame this as a robustness check.
- State the failure mode quantitatively: "if true posterior extends beyond 3σ_scout by Δ, clipping bias is approximately ε".

### A4. Specify the evidence-threshold fallback explicitly
§7.6.1 says "automated Evidence-threshold trigger" without defining it.

**Action:**
- Write the actual rule: "if log Z_Phase2 < log Z_threshold(SNR), revert to blind run." Define the threshold function.
- Discuss what false-positive / false-negative rate it would have.

### A5. Sharpen novelty claim against bilby_pipe pilot-tuning
The thesis does not differentiate against `bilby_pipe`'s existing prior-tuning workflow.

**Action:**
- Add 1 paragraph in §7.2 explicitly contrasting Scout-and-Refine with the pilot/restart features already in `bilby_pipe` (Smith et al. 2020).
- Identify what is actually new: the institutional framing? The systematic SNR sweep? The control experiment? Be precise.

### A6. Remove or qualify multi-messenger / sky-localization claims
§7.5 sells faster sky maps. Sky was fixed in every systematic run.

**Action:**
- Either remove §7.5, or replace with "the framework is *expected to* yield benefits for sky localization, pending the validation reported in §B4 / B5" — and keep it only if you run B4.

### A7. Quantify Phase 1 wall-clock from existing log files
Phase 1 cost is not reported anywhere. This is text-only if logs already exist.

**Action:**
- Pull Phase 1 wall-clock from existing run directories at each of the 10 distances.
- Add a column to Table 3 (or new table): Phase 1 time, Phase 2 time, total CPU-hours, wall-clock-with-calibration-overlap.
- Address the "free during calibration" claim with the actual numbers.

### A8. Justify the 7× scaling factor (6D → 15D)
Currently asserted without derivation.

**Action:**
- Cite Romero-Shaw 2020 or other empirical scaling references.
- Or derive: nested sampling cost ~ `nlive × walks × ln(V_prior/V_posterior)` plus per-likelihood waveform cost. Show the back-of-envelope.

### A9. Specify noise injection: zero-noise vs. realized noise
Currently ambiguous. ln(BF) ≈ 3×10⁵ at 10 Mpc suggests zero-noise injection, which has known pathologies.

**Action:**
- State explicitly in §6.3 whether each Baseline/Phase2/Scout run used a sampled noise realization or zero noise.
- If zero-noise: discuss the known limitations and why claims are still valid.
- If sampled-noise: confirm that the *same* noise realization was reused across iterations (otherwise σ values comingle two different sources of variance).

### A10. Address the gradient-descent hypothesis explicitly
Promised in §4 hypotheses; relegated to §7.7 future work without comment.

**Action:**
- Add 1 paragraph stating that the gradient-descent comparison was scoped out of this thesis and why (limited time, single-supervisor MSc, focus on nested-sampling framework). Keep it honest.

### A11. Improve the 1300 Mpc relative-error explanation
Currently called a "mathematical artifact" — true but uninformative.

**Action:**
- Replace relative error with a noise-floor-normalized metric: e.g., `|Δ ln Z| / σ_intra-run` or `|Δ ln Z| / ln(BF_threshold)`. Show this metric is small everywhere.

### A12. Tighten literature comparison §7.2
Currently defensive ("not directly competitive") rather than analytical.

**Action:**
- For each compared method (ROQ, RB, RIFT, ML, IS, bilby_pipe parallel), give a quantitative speedup factor from the cited paper, then state precisely the operational regime where Scout-and-Refine wins.
- Address: "Why would a practitioner use this over relative binning (10³ speedup)?"

### A13. Acknowledge Bilby 1.1.3 limitation
You're on a pre-2.x version. Current `bilby` has improved samplers and JIT-compiled likelihoods.

**Action:**
- One sentence in Appendix A: state the version, note that newer versions may compress the absolute timing baseline, but the *relative* savings argument is version-agnostic.

### A14. Statistical caveats on N=3
σ from N=3 is extremely noisy. Either run more (Track B) or document.

**Action:**
- Add 1 paragraph in §6.3 with the t-statistic for the "44% savings ≠ 0" claim. Document the wide CI.
- If you do B2/B3, this becomes moot.

---

## Track B — Code Experiments (sorted shortest run-time → longest)

Run-time estimates assume your Core i7-10700KF (8 cores), based on the Phase 2 ≈4.3h-at-40-Mpc reference. Adjust if you have access to a cluster.

### B1. [~½ day] Re-analyze existing data for the transition-boundary metric
**Run-time:** No new sampling. Post-processing of existing log files.
**What:** Compute a noise-floor-normalized BF metric (see A11) from existing posteriors. Replace Figure 9 bottom panel.
**Deliverable:** Updated Figure with new metric; 1-paragraph caption.

### B2. [~1 day] Phase 1 cost extraction + reporting
**Run-time:** None — read existing run directories.
**What:** Parse `result.json` / log files for Phase 1 wall-clock at all 10 distances. Build a 4-column table (Phase 1, Phase 2, Total CPU, Wall-clock-with-overlap). See A7.
**Deliverable:** New Table in §6.3.2 + 1 paragraph commentary.

### B3. [~2–3 days] **Matched-budget control experiment (THE critical experiment)**
**Run-time:** 4 conditions × 3 iterations × ~5–8h each ≈ 60–100 CPU-hours.
**What:** Run Refine sampler with `nlive=2048, walks=100` (matched to Baseline) AND Scout-compressed priors, at d_L = 40 and 150 Mpc.
**Why:** Decisively decomposes "savings from prior compression" vs "savings from reduced sampler budget." This is the experiment a committee will demand.
**Setup:**
- Re-use existing Phase 1 Scout posteriors at 40 and 150 Mpc (no new Scout runs).
- Configure dynesty: `nlive=2048, walks=100, dlogz=0.1` + truncated-Gaussian priors from Scout.
- N=3 iterations each, same noise realization as systematic study.
**Deliverable:** New subsection §6.4.4 with 2×3 results table. Update the headline "44%" claim to attribute fractions to prior-compression vs budget-reduction.

### B4. [~2–3 days] Sky-localization free run (single distance)
**Run-time:** 10D (add RA, Dec, ψ, $\iota$ untied from current θ_JN treatment) costs ~2× the 6D run. ≈ 8–16h × 3 × 2 methods = 50–100h.
**What:** At 150 Mpc, repeat Baseline + Phase 2 with sky position free.
**Why:** Validates (or kills) the multi-messenger claims in §7.5.
**Setup:**
- Add RA (uniform), Dec (cos), ψ (uniform) to free parameters.
- Use same waveform `IMRPhenomXPHM`, 150 Mpc injection.
- N=3 each.
**Deliverable:** New subsection in §6.4 with sky-area at 90% confidence, plus an updated corner plot. Either confirms §7.5 or forces you to remove it.

### B5. [~3–5 days] Small injection campaign at 150 Mpc (statistical generalization)
**Run-time:** 20 injections × 2 methods × ~4h ≈ 160 CPU-hours. With N=1 per condition this fits in 4–5 days.
**What:** Sample 20 injections from your BBH prior at d_L = 150 Mpc, varying q ∈ [0.3, 1.0] and χ_eff ∈ [–0.5, 0.5], aligned spins only. Run both Baseline and Phase 2.
**Why:** First evidence that the framework generalizes off the single test point. Enables a PP-plot-style coverage statement.
**Setup:**
- Sample (q, χ_eff) on a 4×5 grid or quasi-random sequence.
- Use aligned-spin waveform (`IMRPhenomD` or `IMRPhenomXAS`) to keep cost down.
- Compute Δ_param per injection per parameter; report as a distribution, not a single number.
**Deliverable:** New §6.5 — "Statistical robustness across the BBH population." Box-plot of Δ_param per parameter. If clean, this is the result that saves the thesis.

### B6. [~5–7 days] Full 15-parameter demonstration run (single distance)
**Run-time:** 15D with `IMRPhenomXPHM` ≈ 20–30h per run × 2 methods × 3 iterations = ~150 CPU-hours.
**What:** At d_L = 150 Mpc, run Baseline + Phase 2 with **all 15 parameters free** (including 3D spins for both BHs).
**Why:** Tests whether the single-Gaussian truncation survives in the actual production-dimensional space and the q–χ_eff degeneracy region.
**Setup:**
- Free all 15 parameters with standard priors.
- May need to extend Phase 1 settings to handle 15D (e.g., `nlive=750` for Scout).
- Watch for posterior multimodality.
**Deliverable:** New §6.6. Either validates §7.2.1 extrapolation or surfaces a failure mode you must discuss honestly.

### B7. [~5–7 days] Real-event validation (GW150914 or GW170817)
**Run-time:** Similar to B6 if using full 15D, lower if aligned-spin.
**What:** Pull public LIGO O1/O3 data and PSDs (preliminary + final calibration releases). Run Phase 1 on preliminary PSD, Phase 2 on final PSD. Compare to published GWTC posteriors.
**Why:** Most rigorous unbiasedness test. Directly answers "does this work on real noise with real glitches and real calibration uncertainty?"
**Setup:**
- GW150914 strain + PSD from GWOSC.
- Get both initial and final calibration versions if available (LIGO DCC).
- Compare posteriors to GWTC-1/3 release.
**Deliverable:** New §6.7. Substantial credibility upgrade if it works.

### B8. [optional, ~3–5 days] Glitch / non-Gaussian noise robustness
**Run-time:** ~50–100 CPU-hours.
**What:** Inject a known glitch (e.g., blip glitch from gravity-spy catalog) near the signal and rerun Phase 1 + Phase 2 at 150 Mpc.
**Why:** Tests the §7.6.1 catastrophic-mis-characterization concern empirically. Also exercises the evidence-threshold fallback you defined in A4.
**Setup:**
- Use BayesWave or Bilby glitch models.
- Compare Phase 1 posterior under glitch vs glitch-free noise.
**Deliverable:** New paragraph in §7.6.1 with empirical evidence.

### B9. [optional, ~5+ days] Hybrid waveform approximant demo
**Run-time:** Mostly engineering time + 1 full set of runs.
**What:** Run Scout with `IMRPhenomD` (aligned-spin, cheap); run Refine with `IMRPhenomXPHM` (precessing, expensive).
**Why:** Demonstrates the §7.3 claim quantitatively rather than as conjecture.
**Setup:**
- Modify Phase 1 to use cheaper approximant.
- Confirm posterior overlap is acceptable when crossing approximants.
**Deliverable:** New §6.8.

---

## Suggested Order of Operations

**Week 1 (now):**
- Start B3 (matched-budget control) running in background — most decisive experiment.
- In parallel write A1 (reframe motivation), A2 (formal estimator), A7 (Phase 1 cost — from logs).

**Week 2:**
- B3 completes → write new §6.4.4.
- Start B5 (injection campaign) running.
- Write A3, A5, A6, A8, A9, A11, A12.

**Week 3:**
- B5 completes → write §6.5.
- Start B6 (15-parameter run) or B4 (sky-free) depending on what's more defensible.
- Finalize A4, A10, A13, A14.

**Week 4 (stretch):**
- B7 (real event) if time allows.
- Integration pass over the whole document.

---

## Minimum Acceptable Defense Set

If you only do **one** code experiment, do **B3** (matched-budget control). It is the single most consequential test for the credibility of the headline result.

If you can do **two**, add **B5** (injection campaign).

If you can do **three**, add **B6** (15D) or **B4** (sky-free) — whichever your supervisor weighs more heavily.

The remaining text-only fixes (Track A) are non-negotiable regardless of how much code you run.
