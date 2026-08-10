#!/usr/bin/env python3
"""
@brief Compute Baseline vs Phase 2 posterior-width ratios for the Figure 12 claim.

Auto-discovers bilby *result.json files under a root directory instead of relying
on hardcoded filenames, prints what it found so the classification can be checked,
then reports per-parameter posterior widths and the Phase 2 / Baseline ratio.

Run:
    python Posterior_Widths.py
    python Posterior_Widths.py --root . --dry-run
    python Posterior_Widths.py --baseline-key blind --phase2-key final
"""

import argparse
import json
import os
import re
import sys

# parameters of the 6-free-parameter configuration; missing ones are skipped
TARGET_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "theta_jn",
    "phase",
    "geocent_time",
]

BASELINE_MARKERS = ["baseline", "blind", "_base", "base_"]
PHASE2_MARKERS = ["phase2", "phase_2", "refine", "_p2", "p2_"]
PHASE1_MARKERS = ["phase1", "phase_1", "scout", "_p1", "p1_"]

NARROW_RATIO_THRESHOLD = 0.90   # below this, Phase 2 is materially narrower
MATCH_RATIO_TOLERANCE = 0.10    # within this of 1.0 counts as "same width"


def Find_Result_Files(root):
    """
    @brief Recursively collect bilby result JSON files.
    @param root directory to search.
    @return list of absolute paths, sorted.
    """
    found = []

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith("result.json") or filename.endswith("_result.json"):
                found.append(os.path.abspath(os.path.join(dirpath, filename)))

    return sorted(found)


def Extract_Distance(path):
    """
    @brief Parse the luminosity distance label out of a result path.
    @param path file path to inspect.
    @return int distance in Mpc, or None when absent.
    """
    match = re.search(r"(\d+)\s*mpc", path, flags=re.IGNORECASE)

    if match is None:
        return None

    return int(match.group(1))


def Classify_Arm(path, baseline_key, phase2_key):
    """
    @brief Decide whether a result file is the Baseline, Phase 2, Phase 1, or unknown.
    @param path file path to inspect.
    @param baseline_key extra substring that forces a BASELINE match.
    @param phase2_key extra substring that forces a PHASE2 match.
    @return str one of "BASELINE", "PHASE2", "PHASE1", "UNKNOWN".
    """
    lowered = path.lower()

    if baseline_key and baseline_key.lower() in lowered:
        return "BASELINE"

    if phase2_key and phase2_key.lower() in lowered:
        return "PHASE2"

    # check Phase 2 before Phase 1: a "scout_refine" path is the Refine output
    if any(marker in lowered for marker in PHASE2_MARKERS):
        return "PHASE2"

    if any(marker in lowered for marker in PHASE1_MARKERS):
        return "PHASE1"

    if any(marker in lowered for marker in BASELINE_MARKERS):
        return "BASELINE"

    return "UNKNOWN"


def Load_Posterior(path):
    """
    @brief Read a bilby result file and return its posterior samples.
    @param path result JSON path.
    @return pandas.DataFrame posterior samples.
    """
    try:
        import bilby

        result = bilby.core.result.read_in_result(filename=path)
        return result.posterior
    except Exception:
        pass  # fall through to raw JSON

    import pandas as pd

    with open(path, "r") as handle:
        payload = json.load(handle)

    posterior = payload.get("posterior")

    if posterior is None:
        raise ValueError(f"no 'posterior' key in {path}")

    if isinstance(posterior, dict) and "content" in posterior:
        posterior = posterior["content"]

    return pd.DataFrame(posterior)


def Measure_Widths(frame, params):
    """
    @brief Compute standard deviation and 90% credible-interval width per parameter.
    @param frame posterior samples.
    @param params parameter names to measure.
    @return dict mapping parameter to (std, ci90) tuples.
    """
    widths = {}

    for param in params:
        if param not in frame.columns:
            continue

        column = frame[param].dropna()

        if len(column) < 2:
            continue

        low, high = column.quantile(0.05), column.quantile(0.95)
        widths[param] = (float(column.std()), float(high - low))

    return widths


def Report_Discovery(records):
    """
    @brief Print the discovered files and their inferred arm/distance.
    @param records list of (path, arm, distance) tuples.
    """
    print(f"\nDiscovered {len(records)} result file(s):\n")
    print(f"  {'ARM':<9} {'d_L':>6}   PATH")
    print("  " + "-" * 74)

    for path, arm, distance in records:
        label = f"{distance}" if distance is not None else "?"
        print(f"  {arm:<9} {label:>6}   {path}")

    unknown = [r for r in records if r[1] == "UNKNOWN" or r[2] is None]

    if unknown:
        print(f"\n  WARNING: {len(unknown)} file(s) could not be classified by "
              f"arm and/or distance.")
        print("  Use --baseline-key / --phase2-key with a substring unique to each "
              "arm's paths.")


def Report_Widths(distance, baseline_widths, phase2_widths):
    """
    @brief Print the width comparison for one distance.
    @param distance luminosity distance in Mpc.
    @param baseline_widths Baseline width dict.
    @param phase2_widths Phase 2 width dict.
    @return list of Phase 2 / Baseline std ratios.
    """
    shared = [p for p in TARGET_PARAMS
              if p in baseline_widths and p in phase2_widths]

    if not shared:
        print(f"\n{distance} Mpc: no shared parameters between the two arms.")
        return []

    print(f"\n{distance} Mpc")
    print(f"  {'parameter':<21} {'B std':>11} {'P2 std':>11} {'ratio':>7}"
          f" {'B 90%':>11} {'P2 90%':>11} {'ratio':>7}")
    print("  " + "-" * 84)

    ratios = []

    for param in shared:
        base_std, base_ci = baseline_widths[param]
        p2_std, p2_ci = phase2_widths[param]

        std_ratio = p2_std / base_std if base_std else float("nan")
        ci_ratio = p2_ci / base_ci if base_ci else float("nan")
        ratios.append(std_ratio)

        print(f"  {param:<21} {base_std:11.4g} {p2_std:11.4g} {std_ratio:7.3f}"
              f" {base_ci:11.4g} {p2_ci:11.4g} {ci_ratio:7.3f}")

    median_ratio = sorted(ratios)[len(ratios) // 2]
    print(f"  {'median std ratio':<21} {'':>11} {'':>11} {median_ratio:7.3f}")

    return ratios


def Report_Verdict(all_ratios):
    """
    @brief State which of the item-9 branches the numbers support.
    @param all_ratios every Phase 2 / Baseline std ratio collected.
    """
    if not all_ratios:
        print("\nNo ratios computed — nothing to conclude.")
        return

    ordered = sorted(all_ratios)
    median_ratio = ordered[len(ordered) // 2]
    narrow_count = sum(1 for r in ordered if r < NARROW_RATIO_THRESHOLD)

    print("\n" + "=" * 70)
    print(f"Overall median Phase 2 / Baseline std ratio: {median_ratio:.3f}")
    print(f"Parameters with ratio < {NARROW_RATIO_THRESHOLD}: "
          f"{narrow_count} / {len(ordered)}")

    if abs(median_ratio - 1.0) <= MATCH_RATIO_TOLERANCE:
        print("\nVERDICT: widths agree within sampler scatter.")
        print("  -> Figure 12 wording only needs correcting: the two posteriors are")
        print("     the SAME width; neither envelops the other.")
    elif median_ratio < NARROW_RATIO_THRESHOLD:
        print("\nVERDICT: Phase 2 is systematically NARROWER.")
        print("  -> This is the over-confidence signature. Figure 12 needs a short")
        print("     paragraph tying it to the 'using the data twice' discussion and")
        print("     the importance-reweighting correction.")
    else:
        print("\nVERDICT: Phase 2 is systematically WIDER.")
        print("  -> Not an over-confidence problem; correct the wording to match.")

    print("=" * 70)


def Main():
    """
    @brief Discover result files, then report posterior-width ratios.
    @return int exit status.
    """
    parser = argparse.ArgumentParser(
        description="Baseline vs Phase 2 posterior-width comparison.")
    parser.add_argument("--root", default=".",
                        help="directory to search recursively (default: cwd)")
    parser.add_argument("--baseline-key", default="",
                        help="substring that identifies Baseline result paths")
    parser.add_argument("--phase2-key", default="",
                        help="substring that identifies Phase 2 result paths")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list discovered files and their classification")
    args = parser.parse_args()

    paths = Find_Result_Files(args.root)

    if not paths:
        print(f"No '*result.json' files found under {os.path.abspath(args.root)}")
        print("Point --root at the directory holding your bilby outdir(s).")
        return 1

    records = [(p, Classify_Arm(p, args.baseline_key, args.phase2_key),
                Extract_Distance(p)) for p in paths]
    Report_Discovery(records)

    if args.dry_run:
        return 0

    by_distance = {}

    for path, arm, distance in records:
        if arm not in ("BASELINE", "PHASE2") or distance is None:
            continue
        by_distance.setdefault(distance, {})[arm] = path

    all_ratios = []

    for distance in sorted(by_distance):
        arms = by_distance[distance]

        if "BASELINE" not in arms or "PHASE2" not in arms:
            present = ", ".join(sorted(arms))
            print(f"\n{distance} Mpc: skipped, only {present} present.")
            continue

        try:
            base_widths = Measure_Widths(Load_Posterior(arms["BASELINE"]),
                                         TARGET_PARAMS)
            p2_widths = Measure_Widths(Load_Posterior(arms["PHASE2"]),
                                       TARGET_PARAMS)
        except Exception as error:
            print(f"\n{distance} Mpc: failed to load ({error})")
            continue

        all_ratios.extend(Report_Widths(distance, base_widths, p2_widths))

    Report_Verdict(all_ratios)
    return 0


if __name__ == "__main__":
    sys.exit(Main())