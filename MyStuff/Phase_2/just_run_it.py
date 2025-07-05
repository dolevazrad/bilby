#!/usr/bin/env python3
# FILENAME: just_run_it.py
"""
Simple solution that just runs whatever is in phase2_complete.py
No complicated imports, no assumptions about function names
"""

import os
import sys
import time
from datetime import datetime

# Configuration
BASE_DIR = '/home/useradd/projects/bilby/MyStuff/Phase_2'
OUTPUT_DIR = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2'
ASD_FILES = {
    'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
    'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
}

print("="*70)
print("PHASE 2 - SIMPLE RUNNER")
print("="*70)

# Step 1: Change to correct directory
print(f"\nChanging to: {BASE_DIR}")
os.chdir(BASE_DIR)
print(f"Current directory: {os.getcwd()}")

# Step 2: Check files exist
print("\nChecking files...")
for detector, path in ASD_FILES.items():
    if os.path.exists(path):
        print(f"✓ {detector} ASD file found")
    else:
        print(f"✗ {detector} ASD file NOT FOUND")
        sys.exit(1)

if not os.path.exists('phase2_complete.py'):
    print("✗ phase2_complete.py NOT FOUND")
    sys.exit(1)
else:
    print("✓ phase2_complete.py found")

# Step 3: Import and find the function
print("\nImporting phase2_complete...")
try:
    import phase2_complete
    
    # Find any function with 'run' in the name
    run_functions = []
    for name in dir(phase2_complete):
        if 'run' in name.lower() and callable(getattr(phase2_complete, name)):
            if not name.startswith('_'):
                run_functions.append(name)
    
    if not run_functions:
        print("✗ No run function found in phase2_complete.py!")
        print("\nFunctions found:")
        for name in dir(phase2_complete):
            if callable(getattr(phase2_complete, name)) and not name.startswith('_'):
                print(f"  - {name}")
        sys.exit(1)
    
    # Use the first run function found
    func_name = run_functions[0]
    print(f"\n✓ Found function: {func_name}")
    
    if len(run_functions) > 1:
        print(f"  (Also found: {', '.join(run_functions[1:])})")
    
    func = getattr(phase2_complete, func_name)
    
except Exception as e:
    print(f"✗ Error importing phase2_complete: {e}")
    sys.exit(1)

# Step 4: Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
outdir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
os.makedirs(outdir, exist_ok=True)
print(f"\nOutput directory: {outdir}")

# Step 5: Run the function
print(f"\nRunning {func_name}...")
print("This may take 1-3 hours...\n")

start_time = time.time()

try:
    # Try with outdir parameter
    result = func(ASD_FILES, outdir=outdir)
    print("\n✓ Function completed with outdir parameter")
except TypeError as e:
    if 'outdir' in str(e):
        print("Function doesn't accept 'outdir', trying without...")
        try:
            result = func(ASD_FILES)
            print("\n✓ Function completed without outdir parameter")
            print(f"  Note: Results may be in default location, not {outdir}")
        except Exception as e2:
            print(f"\n✗ Function failed: {e2}")
            sys.exit(1)
    else:
        print(f"\n✗ Function failed: {e}")
        sys.exit(1)
except Exception as e:
    print(f"\n✗ Function failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

runtime = time.time() - start_time
print(f"\nRuntime: {runtime/3600:.2f} hours")

# Step 6: Run analysis (if enhanced_analysis.py exists)
if os.path.exists('enhanced_analysis.py'):
    print("\n" + "-"*50)
    print("Running analysis...")
    
    try:
        from enhanced_analysis import analyze_refinement_results
        analyze_refinement_results(outdir)
        print("✓ Analysis completed")
    except Exception as e:
        print(f"⚠ Analysis failed: {e}")
        print("  You can run it manually later")

# Step 7: Summary
print("\n" + "="*70)
print("COMPLETED!")
print("="*70)
print(f"\nTotal runtime: {runtime/3600:.2f} hours")
print(f"Output directory: {outdir}")

# List files created
print("\nFiles in output directory:")
try:
    for f in os.listdir(outdir):
        fpath = os.path.join(outdir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size:.1f} KB)")
except:
    print("  (Unable to list files)")

print("\n✓ All done!")