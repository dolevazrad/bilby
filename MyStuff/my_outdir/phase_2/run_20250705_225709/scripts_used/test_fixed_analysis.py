#!/usr/bin/env python3
"""
Test the fixed enhanced analysis script
"""

import os
import sys
import traceback

try:
    # Try importing the enhanced analysis
    from enhanced_analysis import  Run_Complete_Analysis
    print("✓ Enhanced analysis module imported successfully")
    
    # Find results directory
    results_dir = None
    possible_dirs = [
        'simple_refinement_test',
        'refinement_results',
        'refinement_test',
        'minimal_test_results',
        'full_comparison_results'
    ]
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            results_dir = dir_name
            print(f"✓ Found results directory: {results_dir}")
            break
    
    if results_dir:
        # Try running the analysis
        print("\nRunning enhanced analysis...")
        Run_Complete_Analysis(results_dir)
        print("\n✓ Analysis completed successfully!")
        
    else:
        print("\n✗ No results directory found!")
        print("Available directories in current location:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
                
except Exception as e:
    print(f"\n✗ Error occurred: {type(e).__name__}: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Additional debugging info
    print("\n\nDebugging information:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Check if specific modules are available
    try:
        import bilby
        print(f"Bilby version: {bilby.__version__ if hasattr(bilby, '__version__') else 'Unknown'}")
    except:
        print("Bilby not installed or not importable")