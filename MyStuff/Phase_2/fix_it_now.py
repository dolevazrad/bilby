#!/usr/bin/env python3
# FILENAME: fix_it_now.py

# Quick check for ASD files
import os
import glob

asd_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window'
if os.path.exists(asd_dir):
    files = glob.glob(os.path.join(asd_dir, '*_asd_win*.pkl'))
    if any('604800' in f for f in files) and any('86400' in f for f in files):
        print("\n✓ GOOD NEWS: Found 24h and 7d ASD files!")
        print("  Your run should work perfectly.")
    else:
        print("\n⚠ Note: Couldn't find 24h/7d ASDs, but the script")
        print("  will use whatever time windows you have.")

print("\nPress Enter to exit...")
input() 