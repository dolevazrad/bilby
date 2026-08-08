# Thesis revision — master fix list

Source of truth: your canonical `main.tex` (the version you compile in
`...\MyStuff\Phase_2`). All line numbers below refer to that file. Figure
numbers refer to compiled order (Fig. 10 = `Time_vs_Distance_Scientific.png`,
Fig. 12 = `Posterior_Overlap_150Mpc.png` — both confirmed to match the
mentor's references).

Priority tiers: **Tier 1** = quick text fixes (done). **Tier 2** = script you
run yourself. **Tier 3** = structural/scientific changes (explained here, NOT
edited, per your instruction).

---

## TIER 1 — Quick text fixes (DONE this pass)

### 1. Remove the word "catastrophic" everywhere (mentor note 5)
8 occurrences removed. Examples (before → after):

- §7.9.1 heading: `\subsubsection{Catastrophic Phase 1 Mis-characterization}`
  → `\subsubsection{Severe Phase 1 Mis-characterization}`
- 4× `$1\sigma$ catastrophic-bias threshold` → `$1\sigma$ bias threshold`
  (lines 977, 982, 1000, 1263)
- `a genuinely catastrophic Scout failure` → `a genuinely severe Scout failure` (1219)
- `vulnerable to catastrophic mis-characterization` → `vulnerable to severe mis-characterization` (1267)
- `no curve falls catastrophically outside the bands` → `...falls grossly outside...` (1424)

### 2. Tone down hyperbolic language (mentor note 5, "exaggerated words")

- `seamlessly reducing an ~8-hour blind run` → `reducing an ~8-hour blind run` (783)
- `Yielded aggressive time savings` → `Yielded time savings` (784)
- `wandering aimlessly through the prior volume` → `exploring the full prior volume` (784)
- `the Baseline squanders significant evaluations` → `the Baseline spends significant evaluations` (787)
- `exhibit exceptional precision` → `exhibit high precision` (929)
- `(red) perfectly envelope ... do not artifically truncate` → `(red) closely envelop ... do not artificially truncate` (935; also fixed the "artifically" typo)
- `(red) perfectly overlap ... fully preserved` → `(red) closely overlap ... preserved` (940)
- `maintains the absolute statistical rigor` → `maintains the statistical rigor` (1157)
- `is computationally exhaustive` → `is computationally expensive` (1183)

(Left untouched deliberately: "perfectly calibrated sampler" at 1005 — that is
standard statistics terminology, not hyperbole.)

### 3. Reword "asking the wrong question" (mentor note 6, p.51)
Line 1159, before:
> A practitioner asking "why use Scout-and-Refine instead of relative binning?"
> is therefore asking the wrong question — the operationally correct
> configuration in catalog deployment is to use both.

After:
> Scout-and-Refine and relative binning are therefore not competing
> alternatives: the operationally correct configuration in catalog deployment
> is to use both, with relative binning accelerating each individual
> likelihood evaluation inside the prior-compressed Phase 2 sampler.

### Bonus — pre-existing compile bug fixed
Line 1163 had `\bilby-based` (undefined command → "Undefined control sequence").
Changed to `\texttt{bilby}-based`. This was already in your file; pdflatex only
survived it because nonstopmode pushes past errors.

> Not touched (pre-existing, not on the mentor's list): the several
> `\bfseries invalid in math mode` warnings from `\textbf{$...$}` in tables,
> and the natbib author-year notice. Neither stops the PDF. Flag me if you want
> them cleaned.

---

## TIER 2 — Standalone script (delivered: `make_overlap_corners.py`)

Regenerates all three corner plots (40 / 150 / 300 Mpc) from one code path so
they are mutually consistent (same params, colours, labels, truths). Copy it by
hand into `...\MyStuff\Phase_2`, edit the CONFIG block (paths to your Baseline +
Refine `result.json` files at each distance), then:

```bash
python make_overlap_corners.py
```

It also prints a **posterior-width table** (Baseline vs Phase 2, ratio P2/B) —
use that to answer the Figure 12 question below with numbers rather than by eye.

---

## TIER 3 — Structural / scientific changes (NOT edited — how-to only)

### 4. Future Work placed mid-document (mentor note 1) — HIGH priority, easy
**Where:** §5.5 "Future Work and Directions" (line ~659) sits inside
*Preliminary Results*, before the *Results* section (§6, line 670).
**Why it's wrong:** future-of-the-past. Once real results exist, a Future Work
block in the middle reads as unfinished.
**How to fix:** cut the whole §5.5 subsection. Anything in it that is still a
genuine open direction, fold into the final `\section{Future Work}`
(`sec:future_work`, ~line 1236), which already exists. Most of §5.5 is
superseded by that section, so this is mostly a deletion.

### 5. Preliminary Results vs Results split (mentor note 1) — HIGH priority
**Where:** entire §5 "Preliminary Results" (line 594) vs §6 "Results" (line 670).
**Why:** the "at proposal-submission time" framing is meaningless now that the
full results are in.
**How to fix:** dissolve §5. Its genuinely unique content is (a) the noise-
refinement description and (b) the 6-parameter runtime/scaling numbers. Move (a)
into Methods §4 (where the same description ALREADY appears — see item 6), and
move (b) into §6.1 Baseline (which already restates the 2.66 h / 10,517-sample
numbers). After that, §5 can be deleted entirely. Net effect: one Results
section, no proposal-era framing, and a few pages shorter.

### 6. Duplicated noise-refinement text and figures — HIGH priority, easy win
**Where:** the moving-average noise-refinement paragraph appears twice
(Methods §4, line ~476; and Preliminary Results §5.1, line ~596). The four
`H1_noise_ASD_*` ASD panels appear twice as well: lines 486–507 (Methods) and
605–620 (Preliminary Results) — same four images, same caption idea, both
labelled `fig:noise_refinement_comparison` (a duplicate-label bug too).
**How to fix:** keep ONE copy (in Methods), delete the duplicate figure block
and paragraph in §5. This removes a full page and fixes the duplicate label.
Folds naturally into item 5.

### 7. Add an algorithm diagram: which data + which noise, where (mentor note 2)
**Where:** best placed at the top of §6.2 "The Two-Phase Parameter Estimation
Framework" (~line 691), or end of Methods.
**What it must show:** Baseline = single run on the *finalized* PSD with broad
priors. Scout-and-Refine = Phase 1 on the *preliminary* PSD (broad priors) →
extract posterior → build truncated-Gaussian prior → Phase 2 on the *finalized*
PSD. This visual is what settles the "using the data twice?" question: the two
phases use two *different* PSDs, and Phase 1 only sets the prior *support*, not
likelihood information that re-enters Phase 2.
**How to fix (recommended, no image file):** do it as inline **TikZ** so there
is no external PNG to manage. Minimal skeleton to drop in (add
`\usepackage{tikz}` and `\usetikzlibrary{arrows.meta,positioning}` to the
preamble):

```latex
\begin{figure}[htbp]\centering
\begin{tikzpicture}[node distance=6mm and 12mm,
  box/.style={draw,rounded corners,align=center,inner sep=4pt,minimum height=9mm},
  data/.style={box,fill=blue!8}, noise/.style={box,fill=orange!12},
  ->,>={Stealth}]
% Baseline row
\node[data] (bdat) {Strain data};
\node[noise,right=of bdat] (bpsd) {Finalized PSD};
\node[box,right=of bpsd] (brun) {Baseline run\\broad priors};
\node[box,right=of brun] (bpost){Posterior + $\ln Z$};
\draw (bdat)--(bpsd); \draw (bpsd)--(brun); \draw (brun)--(bpost);
% Two-phase row (below)
\node[data,below=14mm of bdat] (p1dat){Strain data};
\node[noise,right=of p1dat] (p1psd){\textbf{Preliminary} PSD};
\node[box,right=of p1psd] (p1run){Phase 1 Scout\\broad priors};
\node[box,right=of p1run] (p1post){Scout posterior};
\node[box,below=of p1post] (prior){Truncated-Gaussian\\prior $3\sigma_{\rm scout}$};
\node[box,left=of prior] (p2run){Phase 2 Refine\\on \textbf{Finalized} PSD};
\node[box,left=of p2run] (p2post){Final posterior + $\ln Z$};
\draw (p1dat)--(p1psd);\draw(p1psd)--(p1run);\draw(p1run)--(p1post);
\draw (p1post)--(prior);\draw(prior)--(p2run);\draw(p2run)--(p2post);
\end{tikzpicture}
\caption{Data/noise flow: Baseline (top) vs.\ Scout-and-Refine (bottom).
Phase 1 uses the preliminary PSD only to define the prior support for Phase 2,
which evaluates the likelihood against the finalized PSD.}
\label{fig:algorithm}
\end{figure}
```

Tune spacing to taste; the point is the structure, not the exact layout.

### 8. Figure 10 (`Time_vs_Distance_Scientific.png`) — clarify + consolidate (mentor note 3)
**Where:** Fig. 10 (line 775); Table 3 (`tab:time_savings_summary`) and Table 4
(`tab:time_decomposition`), plus the "cost decomposition" paragraph (~line 814).
**Mentor's point:** is the red curve Phase 2 = Refine-only (stage 2) or Scout +
Refine? Show Phase 1 and/or the combined total, to make the argument explicit:
we save time *where it is expensive* (the final run), possibly at a higher
*total* CPU cost. Right now that story is split across Tables 3 and 4.
**How to fix:** replot Fig. 10 with **three** curves vs distance — Baseline,
Phase 1 (Scout), and Phase 2 total (Scout+Refine) — or a stacked bar (Scout part
+ Refine part) against the Baseline bar. All numbers already exist in Table 4
(`Phase 1`, `Refine only`, `Total CPU`, `Wall-clock w/ calibration overlap`).
Then state in the caption which quantity the "savings %" refers to (you use
total-CPU for the headline 44 %, and Refine-only for the wall-clock-to-result
64 %). I can write this plotting script the same way as the corner script if you
send/point me at the timing table as a CSV — the values are already in Table 4,
so it needs no cluster access.

### 9. Figure 12 (`Posterior_Overlap_150Mpc.png`) — width claim + double-use (mentor note 4)
**Where:** Fig. 12 (line 948) and the surrounding text (935, 944).
**Mentor's point:** the text implies Phase 2 (red) is the wider/consistent one,
but blue (Baseline) looks wider — and if Phase 2 is actually *narrower*, is that
the "using the data twice" effect showing up as over-confidence?
**How to fix, in order:**
1. Run `make_overlap_corners.py` — its width table gives the ratio P2/B per
   parameter at each distance. That is the factual answer.
2. If ratios ≈ 1 (within sampler scatter): just correct the wording to say the
   two are the *same* width, not that one envelops the other.
3. If Phase 2 is systematically narrower (ratio < ~0.9): that is exactly the
   over-confidence signature the mentor suspects. Address it head-on — connect
   to the existing "using the data twice" paragraph (§6.3, ~line 742) and to the
   importance-reweighting correction (Eq. `reweighted_Z`), and state whether the
   narrowing survives after reweighting. This is a real scientific point, not
   just wording — worth a short paragraph.

### 10. Appendix bulleted lists → tables (mentor note 7)
**Where:** Appendix A.4.2 "Injected Signal Parameters" (~line 1391), A.4.3
"Recovery Parameters and Prior Distributions" (~line 1403), and the
time-savings justification list (A.3).
**How to fix:** each is a set of (parameter, value/prior) pairs — convert to a
2–3 column `tabular`. Purely mechanical; shortens the appendix and reads better.

### 11. Add physical meaning + the O4/O5 citation (mentor note 8)
**Where:** intro and/or discussion (catalog §7.6, multi-messenger §7.7).
**Mentor's point:** give the physics, not just the numbers — how many events,
at what SNRs, projected future rates, why fast turnaround matters, why it
matters for Tests of GR (TGR).
**How to fix:** add 2–3 short sentences: e.g. GWTC-3 has 90 events; O4/O5 are
projected to yield of order one confident detection every few days; faster
catalog PE shortens the latency for TGR and population studies. **Citation is
ready:** use the LIGO/Virgo/KAGRA observing-scenarios paper —

```latex
\bibitem{ObservingScenarios}
  B. P. Abbott et al. (KAGRA, LIGO Scientific, and Virgo Collaborations),
  ``Prospects for Observing and Localizing Gravitational-Wave Transients with
  Advanced LIGO, Advanced Virgo and KAGRA'',
  Living Reviews in Relativity, vol. 23, no. 1, article 3, 2020,
  DOI:10.1007/s41114-020-00026-9, arXiv:1304.0670.
```

and cite it with `\cite{ObservingScenarios}` on the rate sentence.

---

### Suggested order to tackle Tier 3
6 → 5 → 4 (these three are one connected cleanup and remove the most pages),
then 11 and 10 (quick), then 7 (diagram), then 8 and 9 (need replots; the
scripts make these cheap). Length (mentor's global note) largely takes care of
itself once 4–6 and 10 are done.
