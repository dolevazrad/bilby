#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# B5 pipeline orchestrator
#
#   1) Regenerate the H1/L1 ASD pickles if the 80-day window is missing
#   2) Run the 20-injection campaign (resumable; auto-skips completed inj.)
#   3) Generate the section 6.5 box-plot + LaTeX table from the latest run
#
# Safe to interrupt at any point and rerun: each step checks for outputs
# already on disk and only does the missing work.
# ---------------------------------------------------------------------------
set -euo pipefail

BILBY_ROOT="/home/useradd/projects/bilby"
ASD_DIR="${BILBY_ROOT}/MyStuff/my_outdir/GW_Noise_H1_L1_window_270226"
PHASE2_DIR="${BILBY_ROOT}/MyStuff/Phase_2"
NOISE_SCRIPT="${BILBY_ROOT}/MyStuff/Analyzing_GW_Noise_window.py"

GOLD_H1="${ASD_DIR}/H1_asd_win6912000.pkl"   # 80 d (Baseline + Refine PSD)
GOLD_L1="${ASD_DIR}/L1_asd_win6912000.pkl"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ---------------------------------------------------------------------------
# Step 1: ASD files
# ---------------------------------------------------------------------------
log "Step 1/3  Checking ASD files..."
if [[ -f "$GOLD_H1" && -f "$GOLD_L1" ]]; then
    log "  Gold-standard (80 d) ASDs present. Skipping noise regen."
else
    log "  Missing 80 d ASDs - running Analyzing_GW_Noise_window.py."
    log "  This fetches LIGO open data and can take ~1-3 days wall-clock."
    cd "${BILBY_ROOT}/MyStuff"
    python Analyzing_GW_Noise_window.py
    if [[ ! -f "$GOLD_H1" || ! -f "$GOLD_L1" ]]; then
        log "  ERROR: noise script finished but 80 d ASDs still missing."
        log "  Inspect data_generation_optimized.log in ${BILBY_ROOT}/MyStuff."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: B5 campaign (Baseline + Scout + Refine for every injection)
# ---------------------------------------------------------------------------
log "Step 2/3  Running B5 injection campaign..."
cd "$PHASE2_DIR"

# Resume into the most recent campaign dir if it's not already complete;
# otherwise start a fresh one. The Python script handles per-injection
# resume on its own once we point it at a directory.
LATEST_RUN=$(ls -td "${BILBY_ROOT}/MyStuff/my_outdir/phase_2/b5_injection_campaign_"* 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_RUN" && -f "${LATEST_RUN}/b5_campaign_results.json" ]]; then
    N_DONE=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sum(1 for r in d['injections'].values() if r.get('complete')))" "${LATEST_RUN}/b5_campaign_results.json")
    N_TOTAL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config']['n_injections'])" "${LATEST_RUN}/b5_campaign_results.json")
    if [[ "$N_DONE" -lt "$N_TOTAL" ]]; then
        log "  Resuming existing run: $LATEST_RUN  ($N_DONE / $N_TOTAL complete)"
        python run_injection_campaign_b5.py "$LATEST_RUN"
    else
        log "  Existing run $LATEST_RUN already at $N_DONE / $N_TOTAL. Skipping."
    fi
else
    log "  No prior run found. Starting fresh."
    python run_injection_campaign_b5.py
    LATEST_RUN=$(ls -td "${BILBY_ROOT}/MyStuff/my_outdir/phase_2/b5_injection_campaign_"* | head -1)
fi

# ---------------------------------------------------------------------------
# Step 3: Box-plot + LaTeX table
# ---------------------------------------------------------------------------
log "Step 3/3  Generating box-plot and summary table..."
python plot_b5_box.py "$LATEST_RUN"

log "DONE."
log "Outputs in: $LATEST_RUN"
log "  - b5_delta_param_boxplot.pdf  (Section 6.5 figure)"
log "  - b5_summary_table.tex        (LaTeX summary table)"
log "  - b5_q_chi_eff_scatter.pdf    (grid coverage)"
log "  - b5_campaign_results.json    (full numeric results)"
