#!/usr/bin/env python3
# FILENAME: simple_run_all.py
"""
Simple script to run everything from anywhere
Just run: python simple_run_all.py
"""

import os
import subprocess
import sys

# Set the correct directory
PHASE2_DIR = '/home/useradd/projects/bilby/MyStuff/Phase_2/'

print("="*70)
print("PHASE 2 WORKFLOW - SIMPLE RUNNER")
print("="*70)

# Change to the correct directory
print(f"\nChanging to Phase_2 directory: {PHASE2_DIR}")
os.chdir(PHASE2_DIR)
print(f"Current directory: {os.getcwd()}")

# Check if files exist
print("\nChecking for required files...")
required_files = ['phase2_complete.py', 'add_baseline_to_phase2.py', 'master_run_all.py', 'enhanced_analysis.py']
all_present = True
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file} found")
    else:
        print(f"  ✗ {file} NOT FOUND")
        all_present = False

if not all_present:
    print("\n✗ Some required files are missing!")
    sys.exit(1)

# Step 1: Add baseline comparison
print("\n" + "-"*50)
print("STEP 1: Adding baseline comparison to phase2_complete.py")
print("-"*50)

result = subprocess.run([sys.executable, 'add_baseline_to_phase2.py'], capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Baseline comparison added successfully!")
else:
    print("✗ Failed to add baseline comparison")
    print("Error:", result.stderr)
    print("\nTrying to continue anyway...")

# Step 2: Run the master workflow
print("\n" + "-"*50)
print("STEP 2: Running complete workflow")
print("-"*50)
print("This will take 1-3 hours...")
print("Output will be saved to: /home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/")

result = subprocess.run([sys.executable, 'master_run_all.py'])

if result.returncode == 0:
    print("\n✓ WORKFLOW COMPLETED SUCCESSFULLY!")
else:
    print("\n✗ Workflow failed. Check the error messages above.")

print("\n" + "="*70)