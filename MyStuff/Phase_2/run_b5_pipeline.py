#!/usr/bin/env python3
"""
B5 pipeline orchestrator (VS Code-friendly).

Runs the three B5 steps in order, with per-step skip/resume:

  1. ASD regeneration via ``Analyzing_GW_Noise_window.py`` (skipped if the
     80-day Gold-Standard pickles already exist).
  2. Injection campaign via ``run_injection_campaign_b5.py`` (resumes the
     latest incomplete campaign directory, or starts a fresh one).
  3. Box-plot + LaTeX table via ``plot_b5_box.py``.

Just hit Run in VS Code -- each substep runs as its own subprocess so stdout
streams to the integrated terminal in real time and Ctrl+C cleanly stops the
current step. Re-running the file resumes wherever you left off.
"""

import glob
import json
import os
import subprocess
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Paths -- adjust BILBY_ROOT if you've cloned to a different location.
# ---------------------------------------------------------------------------
BILBY_ROOT = '/home/useradd/projects/bilby'
MYSTUFF_DIR = os.path.join(BILBY_ROOT, 'MyStuff')
PHASE2_DIR = os.path.join(MYSTUFF_DIR, 'Phase_2')
ASD_DIR = os.path.join(MYSTUFF_DIR, 'my_outdir', 'GW_Noise_H1_L1_window_270226')
CAMPAIGN_ROOT = os.path.join(MYSTUFF_DIR, 'my_outdir', 'phase_2')

NOISE_SCRIPT = os.path.join(MYSTUFF_DIR, 'Analyzing_GW_Noise_window.py')
B5_SCRIPT = os.path.join(PHASE2_DIR, 'run_injection_campaign_b5.py')
PLOT_SCRIPT = os.path.join(PHASE2_DIR, 'plot_b5_box.py')

# 80 d window (Gold) -- needed by B5 Baseline and Refine
GOLD_H1 = os.path.join(ASD_DIR, 'H1_asd_win6912000.pkl')
GOLD_L1 = os.path.join(ASD_DIR, 'L1_asd_win6912000.pkl')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg: str) -> None:
    print()
    print('=' * 78)
    print(f'  {msg}')
    print('=' * 78, flush=True)


def run_step(script: str, cwd: str, *extra_args: str) -> None:
    """Run a script in its own subprocess, inheriting stdout/stderr."""
    cmd = [sys.executable, '-u', script, *extra_args]
    print(f'$ cd {cwd}')
    print(f'$ {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f'Step failed (exit {result.returncode}): {script}')


def latest_campaign_dir() -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(CAMPAIGN_ROOT, 'b5_injection_campaign_*')))
    return matches[-1] if matches else None


def campaign_progress(run_dir: str) -> tuple[int, int]:
    """Return (n_complete, n_total) for an existing campaign directory."""
    json_path = os.path.join(run_dir, 'b5_campaign_results.json')
    if not os.path.exists(json_path):
        return 0, 0
    with open(json_path, 'r') as f:
        data = json.load(f)
    n_total = data.get('config', {}).get('n_injections', 0)
    n_done = sum(1 for r in data.get('injections', {}).values() if r.get('complete'))
    return n_done, n_total


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step1_asds() -> None:
    banner('Step 1/3  ASD files')
    if os.path.exists(GOLD_H1) and os.path.exists(GOLD_L1):
        print('  Gold-standard (80 d) ASDs already on disk. Skipping noise regen.')
        return
    print('  Missing 80 d ASDs -- running Analyzing_GW_Noise_window.py.')
    print('  This fetches LIGO open data from GWOSC; expect ~1-3 days wall-clock.')
    run_step(NOISE_SCRIPT, MYSTUFF_DIR)
    if not (os.path.exists(GOLD_H1) and os.path.exists(GOLD_L1)):
        raise SystemExit(
            'Noise script finished but 80 d ASDs still missing. '
            'Check data_generation_optimized.log.'
        )


def step2_campaign() -> str:
    banner('Step 2/3  B5 injection campaign')
    latest = latest_campaign_dir()
    if latest is not None:
        n_done, n_total = campaign_progress(latest)
        if 0 < n_done < n_total:
            print(f'  Resuming existing run: {latest}  ({n_done}/{n_total} complete)')
            run_step(B5_SCRIPT, PHASE2_DIR, latest)
            return latest
        if n_done >= n_total and n_total > 0:
            print(f'  Existing run {latest} already at {n_done}/{n_total}. Skipping.')
            return latest
        print(f'  Existing run {latest} has no progress; starting a fresh campaign.')
    else:
        print('  No prior campaign directory found. Starting fresh.')
    run_step(B5_SCRIPT, PHASE2_DIR)
    latest = latest_campaign_dir()
    if latest is None:
        raise SystemExit('B5 campaign finished but no output directory was created.')
    return latest


def step3_plot(run_dir: str) -> None:
    banner('Step 3/3  Box-plot + LaTeX table')
    run_step(PLOT_SCRIPT, PHASE2_DIR, run_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    banner('B5 pipeline')
    print(f'  BILBY_ROOT    : {BILBY_ROOT}')
    print(f'  ASD_DIR       : {ASD_DIR}')
    print(f'  CAMPAIGN_ROOT : {CAMPAIGN_ROOT}')

    step1_asds()
    run_dir = step2_campaign()
    step3_plot(run_dir)

    banner('DONE')
    print(f'  Outputs in: {run_dir}')
    print('    - b5_delta_param_boxplot.pdf  (Section 6.5 figure)')
    print('    - b5_summary_table.tex        (LaTeX summary table)')
    print('    - b5_q_chi_eff_scatter.pdf    (grid coverage)')
    print('    - b5_campaign_results.json    (full numeric results)')


if __name__ == '__main__':
    main()
