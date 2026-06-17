# Plan: B3 Matched-Budget Refine Experiment

This is the implementation of **B3** as written in `thesis_fix_plan.md`:

> Run Refine sampler with `nlive=2048, walks=100` (matched to Baseline) AND
> Scout-compressed priors, at d_L = 40 and 150 Mpc.

It is the *opposite* of the existing `run_control_experiment.py`. Together the
two experiments form a clean orthogonal decomposition:

| Configuration            | nlive | walks | dlogz | Priors      | What it isolates                       |
|--------------------------|------:|------:|------:|-------------|----------------------------------------|
| Baseline                 |  2048 |   100 |   0.1 | Broad       | reference                              |
| **Matched-budget Refine**|  2048 |   100 |   0.1 | Compressed  | **compression-only saving** vs Baseline|
| Phase 2 (Refine)         |  1024 |    50 |   0.1 | Compressed  | full Phase 2 saving                    |
| Control (existing)       |  1024 |    50 |   0.1 | Broad       | budget-only saving vs Baseline         |

The "Matched-budget Refine -> Phase 2" delta is the residual budget effect once
the compression has already done its work, while "Baseline -> Matched-budget
Refine" is the pure compression effect. If both control and matched-budget runs
behave as expected, the total Phase 2 saving (~44 %) decomposes cleanly into a
compression component and a budget component.

---

## Files produced

1. `MyStuff/Phase_2/all_in_one_correct_phase2.py` — already extended:
   a new `matched_budget` branch in `run_pe` applies `nlive=2048, walks=100,
   dlogz=0.1` whenever the label contains `matched_budget` AND informed priors
   are supplied. No other path is changed.
2. `MyStuff/Phase_2/run_matched_budget_experiment.py` — new driver.

---

## What the driver does

For each distance in `[40 Mpc, 150 Mpc]` and each iteration `i in {1, 2, 3}`:

1. Loads `dist_<d>_iter_<i>_scout_result.json` from
   `MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/`.
   No new Scout runs.
2. Calls `create_informed_priors(scout_res)` — same Truncated-Gaussian
   constructor used by the original Phase 2 study.
3. Calls `run_pe(full_files, 'dist_<d>_iter_<i>_matched_budget', ...,
   informed_priors=informed_priors)` so the new `matched_budget` branch in
   `run_pe` applies the Baseline budget.
4. Importance-reweights the resulting evidence back to the broad prior via
   `Re_Weight_Posterior`, and computes `|delta median| / sigma_baseline` for
   `chirp_mass` and `mass_ratio` against the saved Baseline iteration `i`.
5. Checkpoints JSON after every iteration so an interrupted run can resume
   without re-doing completed iterations.

After both distances finish, the driver pulls `avg_baseline_time_h` and
`avg_phase2_time_h` from the existing
`systematic_snr_variance_results.json` to compute, per distance:

- `compression_savings_pct` = (Baseline - Matched) / Baseline x 100
- `budget_savings_pct`      = (Matched - Phase 2) / Matched x 100
- `total_savings_pct`       = (Baseline - Phase 2) / Baseline x 100
  (sanity check; should reproduce the ~44 % figure at 40 Mpc.)

JSON output structure (per distance key like `40_Mpc`):

```json
{
  "matched_times_s": [...],
  "matched_bfs": [...],
  "matched_deltas": [{"chirp_mass": ..., "mass_ratio": ...}, ...],
  "iterations_completed": 3,
  "matched_mean_time_h": ...,
  "matched_std_time_h":  ...,
  "matched_mean_bf":     ...,
  "baseline_mean_time_h": ...,
  "phase2_mean_time_h":   ...,
  "compression_savings_pct": ...,
  "budget_savings_pct":      ...,
  "total_savings_pct":       ...
}
```

---

## Execution

Run on the compute machine where the ASD files at
`/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window_270226`
exist:

```bash
cd /home/useradd/projects/bilby
python MyStuff/Phase_2/run_matched_budget_experiment.py
```

Expected wall-clock per iteration ≈ Baseline wall-clock (~5–8 h at 40 Mpc,
slightly less at 150 Mpc) minus whatever compression actually buys. Total
expected runtime ≈ 30–48 h. Resumable: re-running after an interruption picks
the next free `matched_budget_run_*` timestamp; existing completed iterations
in a previous directory remain intact and can be merged manually if needed.

---

## Acceptance criteria

- 6 matched-budget result JSONs written
  (`dist_40_iter_{1,2,3}_matched_budget_result.json`,
  `dist_150_iter_{1,2,3}_matched_budget_result.json`).
- `matched_budget_experiment_results.json` contains
  `compression_savings_pct`, `budget_savings_pct`, `total_savings_pct` for
  both `40_Mpc` and `150_Mpc`.
- `total_savings_pct` at 40 Mpc lies within ±5 % of the existing ~44 % thesis
  result (validates the experimental setup is identical).
- `compression_savings_pct + budget_savings_pct ≈ total_savings_pct`
  (additivity check; small mismatch is expected because the budget-savings
  ratio is computed relative to Matched-budget rather than Baseline).

---

## How this updates the thesis

Once results are in, §6.4.4 gains:

> "A matched-budget Refine run (`nlive=2048, walks=100`, Scout-compressed
> priors) at 40 Mpc averaged X.XX h, compared with Y.YY h for the Baseline —
> a compression-only saving of ZZ %. The remaining WW % of the headline ~44 %
> Phase 2 saving is attributable to the reduced sampler budget, as confirmed
> by the budget-only control (`nlive=1024, walks=50`, broad priors)."

The two control experiments together convert §6.4.4 from a single "44 %"
number into a fully decomposed claim.
