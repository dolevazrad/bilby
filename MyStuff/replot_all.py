import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import logging
from astropy.time import Time
import glob
import re

# --- CONFIGURATION ---
PARENT_LABEL = "GW_Noise_H1_L1_window_201225"
# ---------------------

# Define output directory
user = os.environ.get('USER', 'default_user')
if user == 'useradd':
    BASE_OUTDIR = f'/home/{user}/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
elif user == 'dolev':
    BASE_OUTDIR = f'/home/{user}/code/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
else:
    BASE_OUTDIR = f'./{PARENT_LABEL}'

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def format_duration(duration_hours):
    if duration_hours >= 1:
        return f"{duration_hours:.1f} hours"
    else:
        return f"{duration_hours*60:.1f} minutes"

def plot_asd(filename):
    """Load a pickle and re-plot the PNG."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        cumulative_asd = data['asd']
        win_length = data['window_length']
        # Detect detector from filename (H1_asd... or L1_asd...)
        detector = os.path.basename(filename).split('_')[0] 
        processed_time = data['processed_time']
        event_time = data['event_time']
        
        # Calculate times for the title
        start_time = event_time - win_length/2
        end_time = event_time + win_length/2
        
        # Format strings
        window_str = format_duration(win_length/3600)
        total_processed_str = format_duration(processed_time/3600)
        
        plt.figure(figsize=(12, 8))
        
        if cumulative_asd is not None and np.any(cumulative_asd.value > 0):
            frequencies = cumulative_asd.frequencies.value
            asd_values = cumulative_asd.value
            f_min, f_max = 10, 1000
            mask = (frequencies >= f_min) & (frequencies <= f_max)
            plt.loglog(frequencies[mask], asd_values[mask], label='ASD')
            
            # --- THE CORRECTED TITLE FORMAT ---
            title = (f'Noise ASD for {detector}\n'
                     f'Window size: {window_str}\n'
                     f'Total processed: {total_processed_str}\n'
                     f'Event time: {Time(event_time, format="gps").iso} (GPS: {event_time})\n'
                     f'Window: {Time(start_time, format="gps").iso} to {Time(end_time, format="gps").iso}')
            plt.title(title)
            # ---------------------------------
            
            plt.ylabel('ASD (strain/√Hz)')
            plt.xlabel('Frequency (Hz)')
            plt.grid(True, which='both', alpha=0.5)
            plt.legend()
            plt.xlim(f_min, f_max)
            
            # Save overwrite
            png_filename = filename.replace('.pkl', '.png')
            # Handle special naming format if your original code used "_hours.png"
            # We will force the standard name here to be safe:
            plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_ASD_win{win_length/3600:.3f}_hours.png")
            
            plt.savefig(plot_filename)
            plt.close()
            logging.info(f"✓ Re-plotted: {os.path.basename(plot_filename)}")
            
    except Exception as e:
        logging.error(f"✗ Failed to plot {filename}: {e}")

def main():
    print(f"Scanning {BASE_OUTDIR} for .pkl files...")
    pkl_files = glob.glob(os.path.join(BASE_OUTDIR, "*_asd_win*.pkl"))
    
    if not pkl_files:
        print("No files found! Check the directory path.")
        return

    print(f"Found {len(pkl_files)} files. Updating plots...")
    
    for pkl_file in sorted(pkl_files):
        plot_asd(pkl_file)
        
    print("\nAll plots updated successfully.")

if __name__ == "__main__":
    main()