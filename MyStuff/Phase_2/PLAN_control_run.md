# Plan: Sampler Budget Control Run

## Purpose
This experiment isolates the contribution of **prior compression** from the contribution of
**reduced sampler budget** (nlive=1024, walks=50) to the observed time savings.

Three configurations are compared at 40 Mpc (and 150 Mpc):

| Run          | nlive | walks | dlogz | Priors     |
|--------------|-------|-------|-------|------------|
| Baseline     | 2048  | 100   | 0.1   | Broad      |
| Phase 2      | 1024  | 50    | 0.1   | Compressed |
| **Control**  | 1024  | 50    | 0.1   | **Broad**  |

The control uses identical sampler settings to Phase 2 but with the original broad priors —
no Scout, no prior compression. If the control is as fast as Phase 2, the savings come from
the budget reduction alone. If Phase 2 is meaningfully faster than the control, the prior
compression is contributing independently.

Expected result: Phase 2 faster than control (saves an additional ~15-25%), AND
control Δ_param and BF variance worse than Phase 2, proving the compression maintains
quality that the budget reduction alone cannot.

---

## Output location
Save all result JSONs and the summary report to:
```
MyStuff/my_outdir/phase_2/control_run_YYYYMMDD_HHMMSS/
```

---

## Steps for Claude Code

### Step 1 — Read and understand the existing infrastructure
Read these two files fully before writing any code:
- `MyStuff/Phase_2/all_in_one_correct_phase2.py` — contains `run_pe()`, `create_injection_parameters()`, `find_asd_scenarios()`, `Re_Weight_Posterior()`, `Get_Original_Priors()` equivalent, `OUTPUT_BASE`
- `MyStuff/Phase_2/comprehensive_snr_variance_test.py` — shows how the systematic runs loop

### Step 2 — Create `MyStuff/Phase_2/run_control_experiment.py`

The script must:

1. Import from `all_in_one_correct_phase2`: `find_asd_scenarios`, `run_pe`, `create_injection_parameters`, `Re_Weight_Posterior`, `OUTPUT_BASE`

2. Define broad priors identical to how `Get_Original_Priors()` works in `comprehensive_snr_variance_test.py`

3. Add a new function `run_pe_control(asd_files, label, outdir, injection_params)` that calls
   `run_pe` exactly as the Baseline does, EXCEPT it overrides the sampler settings
   to nlive=1024, walks=50, dlogz=0.1 INSIDE the function. The cleanest way to do this
   is to add a `control=True` flag to a wrapper, or directly modify a copy of `run_pe`
   with the reduced settings hardcoded. Do NOT pass `informed_priors` (that would trigger
   Phase 2 logic). Instead, patch the sampler settings after the label check.

   The simplest approach: in the control wrapper, call `bilby.run_sampler` directly with
   the same setup as `run_pe` but with `npoints=1024, walks=50, dlogz=0.1` and no
   `informed_priors`.

4. Run **3 iterations** at **40 Mpc** and **3 iterations** at **150 Mpc** (6 total control runs).
   Use the same `full_files` ASD as the Baseline (not the half-time scout ASD).

5. For each iteration, load the corresponding existing Baseline and Phase 2 result JSONs from:
   ```
   MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/
   ```
   Files follow the pattern:
   - `dist_40_iter_1_baseline_result.json`
   - `dist_40_iter_1_refine_result.json`
   (and same for iter_2, iter_3, dist_150)

6. Compute for each distance:
   - `control_mean_time_h` — mean wall-clock hours across 3 control iterations
   - `phase2_mean_time_h` — mean wall-clock hours from existing refine results (load from JSON `sampling_time` or re-use the timing already in `systematic_snr_variance_results.json`)
   - `baseline_mean_time_h` — from existing baseline results
   - `compression_savings_pct` — (control_time - phase2_time) / control_time * 100
     (this is the savings from prior compression ALONE, with budget held fixed)
   - `budget_savings_pct` — (baseline_time - control_time) / baseline_time * 100
     (this is the savings from budget reduction ALONE, with priors held fixed)
   - `total_savings_pct` — (baseline_time - phase2_time) / baseline_time * 100
     (this should match the existing thesis result ~44% at 40 Mpc)
   - `control_delta_chirp_mass` — |control_median - baseline_median| / baseline_std
   - `control_delta_mass_ratio` — same formula for mass_ratio
   - `control_bf` — log Bayes Factor from control run

7. Save a JSON summary:
   ```
   control_run_YYYYMMDD/control_experiment_results.json
   ```
   with structure:
   ```json
   {
     "40_Mpc": {
       "baseline_mean_time_h": ...,
       "control_mean_time_h": ...,
       "phase2_mean_time_h": ...,
       "budget_savings_pct": ...,
       "compression_savings_pct": ...,
       "total_savings_pct": ...,
       "control_delta_chirp_mass": ...,
       "control_delta_mass_ratio": ...,
       "control_mean_bf": ...
     },
     "150_Mpc": { ... }
   }
   ```

8. Print a clean summary table to stdout at the end:
   ```
   ============================================================
   CONTROL EXPERIMENT SUMMARY
   ============================================================
   Distance | Baseline | Control | Phase 2 | Budget Savings | Compression Savings
   40 Mpc   |  7.78 h  |  X.XX h |  4.35 h |     XX.X%     |       XX.X%
   150 Mpc  |  5.22 h  |  X.XX h |  3.26 h |     XX.X%     |       XX.X%
   ============================================================
   ```

### Step 3 — Run the script
```bash
cd /path/to/bilby/repo/root
python MyStuff/Phase_2/run_control_experiment.py
```
Each control run at 40 Mpc takes approximately 3-5 hours (nlive=1024 on broad priors).
Total expected runtime: 18-30 hours. You can interrupt and resume — the script should
checkpoint after each completed iteration.

### Step 4 — Verify output
```bash
ls -lh MyStuff/my_outdir/phase_2/control_run_*/
cat MyStuff/my_outdir/phase_2/control_run_*/control_experiment_results.json
```

### Step 5 — Report back
Report the final summary table values so the thesis can be updated with actual numbers.
Specifically report:
- `control_mean_time_h` at 40 Mpc and 150 Mpc
- `budget_savings_pct` (savings from nlive+walks reduction alone)
- `compression_savings_pct` (additional savings from prior compression)
- Whether control Δ_param values are larger than Phase 2 Δ_param values
  (expected: yes, control quality is worse)

---

## Acceptance criteria
- 6 control result JSONs saved (3 × 40 Mpc + 3 × 150 Mpc)
- Summary JSON exists and contains all fields above
- `total_savings_pct` at 40 Mpc is within ±5% of the existing 44% thesis result
  (sanity check that the experimental setup is identical)
- `compression_savings_pct` > 0 (i.e., Phase 2 is faster than budget-only control)

---

## How this updates the thesis
Once the numbers are in, the thesis will be updated to add one sentence to the Results
section and one row to the time savings table:

> "A matched control run (nlive=1024, walks=50, broad priors) at 40 Mpc averaged
> X.XX hours — faster than the Baseline (budget savings: XX%), but slower than
> Phase 2 (compression savings: XX%). This demonstrates that prior compression
> contributes an independent XX% reduction beyond the sampler parameter choices."

The "Sampler budget control run" item in Future Work will be removed and replaced by
a statement that the experiment was performed.
