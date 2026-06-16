# Thesis Fix Plan — Scout and Refine

A work plan for closing the gaps identified in the committee-style review. Each item is self-contained: open a fresh chat for each, paste the **Kickoff prompt**, and work through it.

Items are ordered by impact on the defense. Tackle P1 in order; P2/P3 can be parallelized.

---

## Status legend
- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

# P1 — Must fix before defense

These are the items where an examiner has the strongest weapons. None can be skipped.

---

## P1.1 — Reframe the motivating premise (the "calibration delay")

**Status:** `[ ]`
**Effort:** ~2 hours of reading + 2 hours of writing. No new compute.
**Dependencies:** none. Do this first.

### What's broken
The thesis frames the 6-month calibration window as a forced compute embargo. In reality LIGO already runs low-latency PE on preliminary PSDs within hours of detection, and the strain reconstruction between preliminary and final calibration differs by only ~1–2 % in amplitude. The "20-minute ASD vs 1152-hour ASD" you actually test is a PSD-stationarity / Welch-averaging question, not a calibration-version question. If the premise wobbles, the whole headline number ("free Phase 1 during calibration") wobbles with it.

### What you need to do
1. Read carefully and cite: Sun et al. 2020/2021 (LIGO calibration papers for O3), Davis et al. 2021 (you already cite — re-read with this lens), Biscoveanu+ 2020 (PSD uncertainty on PE — already cited).
2. Quantify: what is the actual amplitude/phase difference between preliminary and final O3 calibration? Numbers, not adjectives.
3. Reframe Section 1 (Introduction) and Section 3 (Research Objective) around **one** of these defensible motivations — pick whichever your data supports:
   - **A.** Robustness to PSD updates within a single observing run (drift, glitches, line-feature changes).
   - **B.** Acceleration of catalog *re-analysis* with updated waveform models, where Phase 1 is a previous PE result reused as a prior.
   - **C.** Hierarchical-population priors: the Phase 1 distribution is informed by *other events* in the same run, not by a preliminary noise curve.
4. Explicitly differentiate from `bilby_pipe`'s built-in pilot-run prior tuning. Cite Smith+ 2020 (already in your bibliography) and state what is new.

### Kickoff prompt for a new chat
> I'm writing the MSc thesis on a "Scout and Refine" two-phase Bayesian inference framework for gravitational-wave PE. My current motivation — that the 6-month LIGO calibration delay forces idle time that Phase 1 can productively use — was challenged by a committee-style review as factually weak: low-latency PE already runs on preliminary PSDs, and the actual PSD shift between preliminary and final calibration is small. Help me reframe the motivation around a defensible premise. I need you to: (1) summarize what the preliminary-vs-final O3 calibration update actually changes in $h(t)$ and $S_n(f)$ with numbers and citations; (2) help me pick between three reframing options (PSD-update robustness, catalog re-analysis with new waveforms, hierarchical-population priors) based on what my data already shows; (3) draft the revised Introduction and Research Objective paragraphs. I will paste the current Section 1 and Section 3 text when we start.

### Done when
- Introduction no longer claims PE waits 6 months for calibration.
- The "Phase 1 is free" argument is grounded in a defensible operational scenario.
- One paragraph explicitly compares against `bilby_pipe` pilot-tuning and names the novel contribution.

---

## P1.2 — Run a small injection campaign and produce a PP plot

**Status:** `[ ]`
**Effort:** ~3–5 days of compute (20 injections × 2 methods × ~5 h each, parallel where possible). 1 day of writing.
**Dependencies:** none — start the compute on day 1.

### What's broken
All systematic results come from **one** injection at **one** noise realization with **N=3** sampler seeds. You cannot claim a 44 % saving "for the catalog" from a single event. The field standard for unbiasedness is a probability-probability (PP) plot drawn from a population of injections. You currently have nothing of the sort.

### What you need to do
1. Draw 20 injections from a realistic BBH prior:
   - Chirp mass: log-uniform in [15, 60] $M_\odot$
   - Mass ratio: uniform in [0.3, 1.0]
   - At least one spin component free: $\chi_{1,z}, \chi_{2,z}$ uniform in [-0.5, 0.5]
   - Fix the rest if you must, but document
   - Fix luminosity distance at 150 Mpc for this campaign (single-distance is acceptable for a PP test)
2. Run Baseline and Phase 2 on each injection.
3. Build the PP plot: for each parameter, plot the cumulative distribution of the quantile at which the injected truth falls in the recovered posterior. A correctly calibrated PE method should produce a diagonal line; deviations from the diagonal indicate bias.
4. Compute the Kolmogorov–Smirnov p-value of the PP curve against the uniform distribution for each parameter and report it.
5. Plot the savings distribution (a histogram of saving % across the 20 injections), not just a mean.

### Kickoff prompt for a new chat
> I need to run a 20-injection PP-plot validation campaign for my Scout-and-Refine two-phase GW PE framework. I already have a working Bilby pipeline that runs Baseline and Phase 2 on a single injection at 6 free parameters (chirp mass, mass ratio, distance, theta_jn, phase, geocent_time). I need to extend it to draw 20 injections from a BBH prior at 150 Mpc with at least the aligned spins free, run both methods on each, and produce: (a) a PP plot per free parameter; (b) a histogram of percent savings across the 20 injections; (c) KS p-values per parameter. I will paste my existing single-injection script. Help me modify it for the campaign, advise on smart parallelization given I have an 8-core i7-10700KF, and write the analysis code for the PP plot and savings histogram.

### Done when
- PP plot exists for at least 6 free parameters and is included in Section 6.
- Savings number is reported as `mean ± σ` over 20 injections, not over 3 sampler seeds on one injection.
- KS p-values are tabulated.
- The headline number ("44 %") is either confirmed, revised, or replaced.

---

## P1.3 — Run the matched-budget control experiment

**Status:** `[ ]`
**Effort:** ~1–2 days of compute. Half a day of writing.
**Dependencies:** none. Can run in parallel with P1.2.

### What's broken
Your control experiment in §6.4.3 compares Baseline (nlive=2048, broad priors) against a budget-reduced control (nlive=1024, broad priors) and against Phase 2 (nlive=1024, compressed priors). This conflates two effects. The decisive experiment is **nlive=2048 with Scout-compressed priors**. If that still shows 30–40 % saving, your method's contribution is real. If it shows only 10–15 %, most of your "44 %" was just sampler-budget reduction enabled by the compression — a much weaker claim.

### What you need to do
1. Reuse the Phase 1 Scout posteriors you already have at 40 Mpc and 150 Mpc (no need to rerun Scout).
2. Run Phase 2 with `nlive=2048`, `walks=100`, `dlogz=0.1` (= Baseline settings, but with Scout-compressed priors). Three iterations each at 40 Mpc and 150 Mpc.
3. Tabulate four cells per distance:
   - Baseline (2048, broad)
   - Budget-reduced (1024, broad) — already have
   - Scout-compressed (1024, compressed) = current Phase 2 — already have
   - **Scout-compressed full budget (2048, compressed)** — NEW
4. Decompose the savings: how much came from compression alone, how much from budget reduction, and is the combination superadditive?
5. Update §6.4.3 with the decomposition table and rewrite the conclusion.

### Kickoff prompt for a new chat
> I need to run a matched-budget control to decompose the 44 % time savings in my Scout-and-Refine GW PE framework into (a) prior compression and (b) sampler-budget reduction. I have existing Phase 1 Scout posteriors at 40 Mpc and 150 Mpc. I need to launch Phase 2 with nlive=2048, walks=100, dlogz=0.1 (matching the Baseline budget, not the reduced Refine budget) but using the Scout-compressed truncated Gaussian priors. Run 3 independent iterations per distance. Help me modify my Phase 2 runner script for this configuration, then write the decomposition analysis that produces a 4-cell table per distance and a clear narrative of which factor contributes how much.

### Done when
- §6.4.3 has a 4-row × 2-distance decomposition table.
- The thesis honestly attributes the saving to compression vs budget reduction.
- The headline saving for the framework's *unique contribution* (compression alone, holding budget constant) is stated explicitly.

---

## P1.4 — Account properly for Phase 1 compute cost

**Status:** `[ ]`
**Effort:** ~2 hours of bookkeeping (re-running a log analysis on existing runs). 2 hours writing.
**Dependencies:** none. Pure accounting.

### What's broken
Phase 1 wall-clock is nowhere reported. Every savings number in the thesis is Baseline-vs-Refine, not Baseline-vs-(Scout+Refine). The "free during calibration" argument is institutional, not computational. If Phase 1 takes 2 h and Phase 2 takes 4.35 h, the total compute is 6.35 h — close to Baseline's 7.78 h, so the *real* CPU saving at 40 Mpc may be ~18 % rather than ~44 %.

### What you need to do
1. Recover Phase 1 wall-clock from your existing run logs at every distance. Add a column to Table 6 (`tab:time_savings_summary`).
2. Produce two figures: wall-clock saving (current Figure 5) and **total CPU-hour saving** (new).
3. Explicitly state the operational regime under which the wall-clock saving is the relevant metric, and the regime under which total CPU is the relevant metric.
4. Rewrite the 4,200-hours-to-2,100-hours appendix calculation honestly: it currently asserts Phase 1 "doesn't count" without quantifying it.

### Kickoff prompt for a new chat
> I need to do honest cost accounting for my Scout-and-Refine GW PE framework. My current results report wall-clock for Phase 2 only and treat Phase 1 as free because it runs during the calibration period. I need to recover Phase 1 wall-clock from my existing run logs (located at PATH — I will provide), add it to Table 6 (`tab:time_savings_summary`), and produce both: (1) wall-clock saving figure as currently shown, and (2) a new total-CPU-hour saving figure that includes Phase 1. Then help me rewrite Appendix A.2 ("Computational Time Savings Analysis") to honestly state both the wall-clock framing (savings = 44 % if Phase 1 is free) and the total-CPU framing (savings = X % regardless).

### Done when
- Phase 1 wall-clock appears in Table 6 as its own column.
- A second figure shows the total-CPU-hour comparison.
- Appendix A.2 quotes both numbers and names the operational assumption each requires.

---

## P1.5 — Increase iteration count from N=3 to a defensible N

**Status:** `[ ]`
**Effort:** Either ~1 week of compute (extend N to 10 at every distance), or ~2 hours of writing (acknowledge the N=3 limit and reframe what you can claim).
**Dependencies:** P1.3 (so you only re-extend the runs that matter).

### What's broken
N=3 with denominator $N-1=2$ gives σ that is itself enormously noisy. A 44 % saving at 40 Mpc has a Baseline σ of 19 % — the savings differ from zero by ~2σ at best. A t-test would not reject the null.

### What you need to do
Choose one of two paths:

**Path A (more credible):** Extend N=3 → N=10 at the four most important distances (40, 150, 300, 600 Mpc). That's ~28 additional Baseline + Phase 2 runs. Report mean ± standard error, and report a paired t-test of saving vs zero.

**Path B (faster):** Keep N=3 but explicitly note in the thesis that the sample standard deviation is on $N-1=2$ degrees of freedom and the confidence intervals are wide. Reframe the empirical claim as illustrative rather than statistical, and lean on P1.2's PP-plot campaign for the rigorous validation.

### Kickoff prompt for a new chat
> I have N=3 sampler iterations per distance for my Scout-and-Refine GW PE results. A committee-style review pointed out that with denominator N-1=2 my standard deviations are unreliable and my "44 % saving" is only ~2σ from zero. I need to either extend to N=10 at the four most important distances (40, 150, 300, 600 Mpc) or restrict my claims and lean on a parallel PP-plot study for rigor. Help me: (1) estimate compute time for the N=10 extension on my hardware; (2) draft the analysis update (paired t-test, mean ± SE, possibly a Welch's t per condition); (3) write a paragraph for §6 that honestly characterises uncertainty.

### Done when
- Either you have N ≥ 10 at the priority distances, **or** the thesis prominently disclaims the N=3 statistic and points the reader to the PP campaign.
- A paired t-test result (savings ≠ 0) is reported.

---

## P1.6 — Formalize the Bayesian validity of the prior-construction step

**Status:** `[ ]`
**Effort:** ~1 day of derivation + ~1 day of writing. No new compute.
**Dependencies:** none.

### What's broken
You build the Phase 2 prior from a posterior computed on the same data with a different noise model, and call the resulting Phase 2 evidence "importance reweighted" without giving the math. An examiner will ask: write down the estimator. Under what conditions is it unbiased? Why $3\sigma$? What happens if the true posterior has heavier-than-Gaussian tails?

### What you need to do
1. Write down the formal expression for the Phase 2 evidence under truncated Gaussian priors and the data-driven prior choice. Identify the change-of-prior correction explicitly.
2. State the assumptions (e.g., the preliminary-PSD posterior is sufficiently close to the final-PSD posterior that the truncated Gaussian envelope covers the true support; the true posterior is sufficiently unimodal in 6D).
3. Justify the choice of $3\sigma$ — show empirically what fraction of the Baseline posterior mass falls within the Scout's $3\sigma$ envelope, at each of the four representative distances. If <99 %, you have a problem.
4. Add a half-page subsection — call it "Formal construction and consistency conditions" — under §6.2 or §7.1.

### Kickoff prompt for a new chat
> I need to write a formally rigorous half-page in my MSc thesis on the Bayesian validity of constructing Phase 2 priors from Phase 1 posteriors, with importance reweighting. Specifically: (1) the explicit estimator for the Phase 2 evidence $\ln Z_2$ under the truncated Gaussian priors that are themselves derived from Phase 1; (2) the consistency conditions under which the Phase 2 posterior is identical in distribution to a single-shot blind run; (3) empirical evidence that the Scout $3\sigma$ envelope captures ≥99 % of the Baseline posterior mass at every tested distance. I have the Baseline posteriors stored locally. Help me derive the estimator, identify the assumptions, and write a script that computes the envelope-coverage fraction.

### Done when
- A new subsection contains the explicit estimator and named assumptions.
- A small table (4 distances × 6 parameters) reports the fraction of Baseline mass inside the Scout $3\sigma$ envelope.
- A defensible justification for "$3\sigma$" (rather than $2\sigma$ or $4\sigma$) is in the text.

---

# P2 — Should fix; sharpens the work substantially

---

## P2.1 — Demonstrate the 6D → 15D scaling claim (or pull back from it)

**Status:** `[ ]`
**Effort:** ~3–5 days of compute (one 15D injection, Baseline + Phase 2, N=3). Half a day of writing.
**Dependencies:** P1.3 (so you know the right sampler config).

### What's broken
Every catalog-scale claim depends on the assertion that the 6D result generalizes to 15D, with a "7×" scaling factor that is not derived. The most degenerate sub-manifolds (q–χ_eff, distance–inclination, sky reflections) only appear in 15D, and they are exactly the regime where single-Gaussian truncation might fail.

### What you need to do
1. Run **one** representative injection at 40 Mpc in the full 15D (or at minimum 11D: add the four spin components and sky location). Baseline and Phase 2, N=3.
2. Report: ratio of Phase 2 saving in 6D vs Nd; comparison of Scout-compressed envelope coverage; per-parameter $\Delta_\text{param}$.
3. Replace the asserted "7×" scaling factor with either a measured value or a derivation grounded in nested-sampling theory.

### Kickoff prompt for a new chat
> My MSc thesis on Scout-and-Refine GW PE validated everything in 6D. I now need to demonstrate that the framework scales to the full 15D (or at least 11D with spins free). Help me extend my existing Bilby pipeline to a higher-dimensional configuration, run one injection at 40 Mpc with Baseline and Phase 2 (N=3 each), and compare: (a) percent savings vs 6D; (b) Scout envelope coverage of the Baseline posterior; (c) per-parameter $\Delta_\text{param}$. I will paste my 6D injection script. Advise on which extra parameters are most important to free first given limited compute.

### Done when
- A single high-dimensional result is in the thesis as a "scaling check."
- The "7×" scaling factor is either replaced or removed.
- The Discussion's catalog-scale extrapolation is explicitly conditioned on this scaling check.

---

## P2.2 — Remove or substantiate the sky-localization / multi-messenger claims

**Status:** `[ ]`
**Effort:** ~2 hours.
**Dependencies:** ideally after P2.1, where sky is free.

### What's broken
Section 7.5 ("Enhancement of Multi-Messenger Astrophysics") claims faster sky localization. But sky was fixed in every systematic run. The claim is unsupported by any data in the thesis.

### What you need to do
Two options:

**Option A:** Free RA, Dec, ψ in one or two runs (would fit naturally in the P2.1 dimension-scaling exercise) and quantify sky-area-at-90 %-confidence saving between Baseline and Phase 2.

**Option B:** Remove §7.5 entirely or convert it into a "Future Work" paragraph naming the experiment that would justify the multi-messenger claim.

### Kickoff prompt for a new chat
> Section 7.5 of my thesis ("Enhancement of Multi-Messenger Astrophysics") claims faster sky localization with my Scout-and-Refine framework, but RA, Dec, and polarization were fixed in all systematic runs, so the claim is unsupported. Help me: either (a) design a small targeted run with sky free at 40 Mpc and 150 Mpc and quantify the 90 % credible sky area difference, or (b) rewrite §7.5 as a future-work paragraph that names what would have to be done. I will paste the current §7.5 and let me know which path looks more defensible given my compute constraints.

### Done when
- Either §7.5 has empirical support, or the section is recast as future work.

---

## P2.3 — Quantitative comparison against at least one competing method

**Status:** `[ ]`
**Effort:** Depends on which method. Relative binning is cheapest (~1 week to wrap and run via bilby's built-in heterodyne likelihood, if your bilby version supports it).
**Dependencies:** none, but does best after P1.3.

### What's broken
Section 7.2 reviews ROQ, normalizing flows, RIFT, parallel `bilby_pipe`, relative binning, and importance sampling. It declares them "complementary" or "orthogonal" without one quantitative comparison. Relative binning gives ~10³ speedup; you give ~2×. The literature comparison reads as a defense.

### What you need to do
Pick **one** competitor and benchmark properly. Recommended target: **relative binning / heterodyne likelihood** (cite Zackay+ 2018). Run a Baseline-with-heterodyne against Phase 2 at 40 Mpc and 150 Mpc. Report wall-clock and posterior consistency.

### Kickoff prompt for a new chat
> My thesis on Scout-and-Refine GW PE qualitatively compares against ROQ, normalizing flows, RIFT, parallel bilby_pipe, relative binning, and importance sampling, but never benchmarks against any of them. I want to add one quantitative head-to-head against the strongest competitor in my regime: relative binning / heterodyne likelihood (Zackay 2018). I am using Bilby 1.1.3. Help me: (1) check whether my Bilby version exposes heterodyne likelihood and if not what the upgrade path is; (2) run Baseline-with-heterodyne against Phase 2 at 40 Mpc and 150 Mpc; (3) write the comparison subsection honestly even if relative binning wins on wall-clock.

### Done when
- One competitor is empirically benchmarked.
- Section 7.2 cites the result.
- The thesis is honest about which regimes the framework wins in.

---

# P3 — Nice to have; fix any you have time for

---

## P3.1 — Resolve the gradient-descent vs MCMC hypothesis

**Status:** `[ ]`
**Effort:** ~3 days of code + 1 day of writing.

### What's broken
Sections 4 and 5.3 frame gradient descent vs MCMC as a hypothesis to be tested empirically. In §7.7 (Future Work) it has quietly become "wasn't done." Committees notice.

### What you need to do
Either run a minimal gradient-descent comparison on one injection (a JAX/Numpy gradient over the analytic chirp-mass/distance subspace is enough), **or** remove the hypothesis from §4 and §5 entirely and frame it from the start as out of scope.

### Kickoff prompt for a new chat
> Section 4 and 5.3 of my thesis hypothesize that gradient descent will outperform MCMC at extreme SNR but I never tested it; it appears in Section 7.7 as future work. Help me either (a) run a minimal gradient-descent comparison in 2D (chirp mass and luminosity distance) on one 40 Mpc injection so the hypothesis is at least demonstrated, or (b) edit Sections 4 and 5.3 so the hypothesis is removed cleanly and the gradient-descent discussion is reframed as background. I will paste the current text of both sections.

### Done when
- The hypothesis is either tested or removed from the front matter.

---

## P3.2 — Test against a non-Gaussian noise realization (glitch injection)

**Status:** `[ ]`
**Effort:** ~2 days.

### What's broken
The framework's premise — Phase 1 runs on "noisy preliminary data" — is most interesting precisely when noise is non-stationary or contains glitches. You cite Powell 2018 on glitch contamination but never test against it.

### What you need to do
Inject one of the standard Bilby glitch templates (or a synthetic blip glitch from gravityspy) on top of your 40 Mpc BBH signal and rerun Baseline and Phase 2. Document whether Phase 1 mis-characterizes the posterior and whether the evidence-threshold fallback (§7.6.1) would have caught it.

### Kickoff prompt for a new chat
> My Scout-and-Refine GW PE thesis claims robustness to "preliminary, noisy data" but only ever tested Gaussian noise drawn from a known PSD. Help me design and run a glitch-injection test: inject a blip glitch on top of a 40 Mpc BBH signal, run Baseline and Phase 2, and check whether (a) Phase 1 still recovers the correct posterior mode and (b) the evidence ratio between Phase 2 and Baseline could have flagged the issue as a fallback trigger. I will paste my existing injection script and the Phase 1 prior boundaries.

### Done when
- One glitch-contaminated run is documented in §7.6.
- The fallback-trigger threshold (§7.6.1) is calibrated against this case.

---

## P3.3 — Address smaller textual issues flagged in the review

**Status:** `[ ]`
**Effort:** ~2 hours.

### Items
- Specify whether injections used zero noise or a noise realization. The $\ln(\mathrm{BF}) \approx 3 \times 10^5$ at 10 Mpc strongly suggests zero noise — say so.
- Replace "Bilby 1.1.3" with a sentence acknowledging the current 2.x version exists and stating why your results are not version-specific.
- The "transition boundary" at 1300 Mpc with -5 % relative error: replace the division-by-near-zero artefact with a noise-floor-normalized metric, e.g. $|\Delta \ln Z| / \sqrt{\mathrm{Var}(\ln Z)}$.
- Remove the placeholder title page text in `\title{...}`.
- Fix the duplicate `\subsection*{Implementation}` patterns / repeated noise-refinement figures between §5 and §6.

### Kickoff prompt for a new chat
> I have a punch-list of smaller textual fixes in my thesis: (1) specify zero-noise vs sampled-noise in injections; (2) acknowledge the Bilby version; (3) replace the divide-by-near-zero relative-error metric at 1300 Mpc with $|\Delta \ln Z| / \sqrt{\mathrm{Var}(\ln Z)}$; (4) remove placeholder title text; (5) deduplicate the noise-refinement subsection that repeats between §5 and §6. Please walk through them with me one at a time. I will paste the relevant LaTeX section as we go.

### Done when
- All five items are checked off.

---

# Suggested order of operations

```
Week 1: P1.1 (reframe motivation) + start P1.2 compute
Week 2: P1.2 finish + P1.4 + P1.5 path-decision
Week 3: P1.3 (matched-budget control) + P1.6 (formal Bayes)
Week 4: P2.1 (15D scaling) + P2.2 (sky claims)
Week 5: P2.3 (relative-binning benchmark) + P3 cleanup
```

Items P1.1, P1.4, P1.5, P1.6, P2.2, P3.3 are writing-only or analysis-only and can fill time while compute runs.

---

# A defense-prep companion

When everything above is closed, dry-run the five committee questions from the review:

1. "What, specifically, did you do that `bilby_pipe`'s pilot-run prior tuning doesn't already do?" → Answer comes from P1.1.
2. "Walk me through the preliminary-vs-final O3 calibration update with numbers." → P1.1.
3. "How much of the 44 % is prior compression vs sampler budget reduction?" → P1.3.
4. "Show me a PP plot." → P1.2.
5. "At what dimensionality does the single-Gaussian truncation start to clip the posterior?" → P2.1.

You will know you are defense-ready when you can answer each in under two minutes without looking at the slides.
