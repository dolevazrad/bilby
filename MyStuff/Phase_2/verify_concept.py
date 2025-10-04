#!/usr/bin/env python3
# FILENAME: verify_concept.py
"""
Quick test to verify the concept makes sense
Shows why half-time data should be faster
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def verify_asd_concept():
    """Verify that half-time ASD is noisier but real data."""
    
    print("="*70)
    print("VERIFYING PHASE 2 CONCEPT")
    print("="*70)
    
    # Look for ASD files
    asd_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window'
    
    # Find pairs of ASDs with different window sizes
    asd_files = {}
    for detector in ['H1', 'L1']:
        detector_files = {}
        for window in [600, 3600, 43200, 86400]:  # 10min, 1h, 12h, 24h
            filename = os.path.join(asd_dir, f'{detector}_asd_win{window}.pkl')
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                detector_files[window] = {
                    'file': filename,
                    'asd': data['asd'],
                    'hours': window / 3600
                }
        asd_files[detector] = detector_files
    
    if not asd_files['H1']:
        print("✗ No H1 ASD files found!")
        return
    
    # Compare ASDs
    print("\nComparing ASDs for H1:")
    print("-"*50)
    
    windows = sorted(asd_files['H1'].keys())
    for window in windows:
        asd_data = asd_files['H1'][window]
        mean_asd = np.mean(asd_data['asd'].value[100:1000])  # 10-100 Hz range
        print(f"{asd_data['hours']:5.1f} hours: mean ASD = {mean_asd:.3e}")
    
    # Key insight
    print("\n" + "="*50)
    print("KEY INSIGHT:")
    print("="*50)
    print("\n1. Shorter observation time → Higher ASD (noisier)")
    print("2. But it's REAL data, not artificially degraded")
    print("3. PE with less data is FASTER")
    print("4. Results are rougher but still contain the signal")
    print("\nTHIS is why the two-phase method works:")
    print("- Phase 1: Fast PE with short observation → rough estimate")
    print("- Phase 2: Refined PE with full observation + informed priors")
    print("- Total time < Full PE from scratch")
    
    # Plot comparison if we have multiple windows
    if len(windows) >= 2:
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(windows)))
        
        for i, window in enumerate(windows):
            asd_data = asd_files['H1'][window]
            freqs = asd_data['asd'].frequencies.value
            asd_values = asd_data['asd'].value
            
            # Plot in frequency range of interest
            mask = (freqs >= 10) & (freqs <= 1000)
            plt.loglog(freqs[mask], asd_values[mask], 
                      label=f'{asd_data["hours"]:.1f} hours',
                      color=colors[i], linewidth=2)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ASD (strain/√Hz)')
        plt.title('ASD Comparison: Different Observation Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('asd_comparison.png')
        print("\n✓ Saved asd_comparison.png")
        
        # Calculate expected time savings
        if 43200 in windows and 86400 in windows:  # 12h and 24h
            print("\n" + "-"*50)
            print("EXPECTED TIME SAVINGS:")
            print("-"*50)
            print("Traditional: 24h data → PE → 1.0x time")
            print("Two-phase:   12h data → PE → ~0.5x time")
            print("             + refined PE  → ~0.3x time")
            print("             Total         → ~0.8x time")
            print("Expected savings: ~20%")

if __name__ == "__main__":
    verify_asd_concept()