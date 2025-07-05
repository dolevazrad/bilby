#!/bin/bash
# FILENAME: run_add_baseline.sh
# Simple script to run add_baseline_to_phase2.py from the correct directory

echo "Changing to Phase_2 directory..."
cd /home/useradd/projects/bilby/MyStuff/Phase_2/

echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la *.py

echo ""
echo "Running add_baseline_to_phase2.py..."
python add_baseline_to_phase2.py

echo ""
echo "Done!"