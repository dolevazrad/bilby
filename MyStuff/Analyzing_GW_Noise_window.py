import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import logging
import time
import pickle
import socket
import gc
# Assuming Analyzing_GW_Noise is in the same folder
from Analyzing_GW_Noise import find_data_gaps, combine_asds
import urllib3.util.connection as urllib3_cn

# --- CONFIGURATION ---
PARENT_LABEL = "GW_Noise_H1_L1_window_201225"
MERGER_TIME = 1238303719

# Change this list to ['H1', 'L1'] when you want to run both!
detectors = ['H1', 'L1']
detectors = ['L1']
# ---------------------

# FORCE IPv4
def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

# Define output directory
user = os.environ.get('USER', 'default_user')
if user == 'useradd':
    BASE_OUTDIR = f'/home/{user}/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
elif user == 'dolev':
    BASE_OUTDIR = f'/home/{user}/code/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
else:
    BASE_OUTDIR = f'./{PARENT_LABEL}'

if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)

print(f"Data will be saved in: {BASE_OUTDIR}")

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_generation_optimized.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def save_checkpoint(cumulative_asd, win_length, detector, asd_count, processed_time):
    """Save the current ASD to disk and plot it."""
    filename = os.path.join(BASE_OUTDIR, f"{detector}_asd_win{win_length}.pkl")
    
    start_time = MERGER_TIME - win_length/2
    end_time = MERGER_TIME + win_length/2

    data = {
        'asd': cumulative_asd,
        'event_time': MERGER_TIME,
        'window_length': win_length,
        'start_time': start_time,
        'end_time': end_time,
        'asd_count': asd_count,
        'processed_time': processed_time
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"✓ CHECKPOINT SAVED: {filename}")
    
    # Create the plot
    try:
        plot_asd(cumulative_asd, win_length, detector, processed_time)
    except Exception as e:
        logging.warning(f"Plotting failed (non-critical): {e}")

def plot_asd(cumulative_asd, win_length, detector, processed_time):
    """Helper to plot the ASD."""
    plt.figure(figsize=(12, 8))
    if cumulative_asd and np.any(cumulative_asd.value > 0):
        frequencies = cumulative_asd.frequencies.value
        asd_values = cumulative_asd.value
        f_min, f_max = 10, 1000
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        plt.loglog(frequencies[mask], asd_values[mask], label='ASD')
        
        # Formatting hours for title
        hours = win_length/3600
        proc_hours = processed_time/3600
        plt.title(f'Noise ASD for {detector}\nWindow: {hours:.2f}h | Processed: {proc_hours:.2f}h')
        plt.ylabel('ASD (strain/√Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True, which='both', alpha=0.5)
        plt.legend()
        plt.xlim(f_min, f_max)
        
    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_ASD_win{win_length/3600:.3f}_hours.png")
    plt.savefig(plot_filename)
    plt.close()

def load_checkpoint(detector, window_sizes):
    """
    Smart Resume: Find the largest existing file for THIS detector 
    and load it so we don't re-download data we already have.
    """
    best_idx = -1
    best_data = None

    # Check files in reverse order (Largest -> Smallest)
    for i in range(len(window_sizes) - 1, -1, -1):
        win_length = window_sizes[i]
        filename = os.path.join(BASE_OUTDIR, f"{detector}_asd_win{win_length}.pkl")
        if os.path.exists(filename):
            try:
                logging.info(f"Found existing checkpoint for {detector}: {filename}")
                with open(filename, 'rb') as f:
                    best_data = pickle.load(f)
                best_idx = i
                break
            except Exception as e:
                logging.warning(f"Corrupt checkpoint {filename}: {e}")
    
    if best_data:
        logging.info(f"Resuming {detector} from Window Index {best_idx} ({window_sizes[best_idx]/3600:.1f}h)")
        return best_data['asd'], best_data['asd_count'], best_data['processed_time'], best_idx
    else:
        logging.info(f"No checkpoints found for {detector}. Starting from scratch.")
        return None, 0, 0, -1

def process_interval(detector, start, end, cumulative_asd, asd_count, processed_time, fftlength=600):
    """
    Process a specific time chunk (The Flanks).
    This contains all your original detailed logic.
    """
    current_time = start
    FETCH_CHUNK = 4096 
    
    while current_time < end:
        this_chunk_size = min(FETCH_CHUNK, end - current_time)
        interval_end = current_time + this_chunk_size
        
        logging.info(f"   Fetching {detector}: {current_time:.0f} .. {interval_end:.0f} (Chunk: {this_chunk_size}s)")
        
        strain = None
        # Robust Fetch Loop
        for attempt in range(3):
            try:
                strain = TimeSeries.fetch_open_data(
                    detector, current_time, interval_end, cache=False, verbose=False
                )
                break
            except ValueError:
                # GWOSC says no data here
                break
            except Exception as e:
                logging.warning(f"   Network hiccup: {e}. Retrying...")
                time.sleep(2)
        
        # If fetch failed or empty, skip
        if strain is None or strain.duration.value < 15:
            current_time += this_chunk_size
            continue

        # Process Segment logic (Your original robust checks)
        try:
            # 1. Check for gaps
            if find_data_gaps(strain):
                current_time += this_chunk_size
                continue

            # 2. Calculate ASD
            actual_fftlength = min(fftlength, strain.duration.value/2)
            segment_asd = strain.asd(fftlength=actual_fftlength, 
                                     overlap=actual_fftlength/2, 
                                     method='median')
            
            # 3. Check for NaNs/Inf
            if np.any(~np.isfinite(segment_asd.value)):
                current_time += this_chunk_size
                continue

            # 4. Combine (The core accumulation)
            if cumulative_asd is None:
                cumulative_asd = segment_asd
                asd_count = 1
            else:
                cumulative_asd = combine_asds(cumulative_asd, segment_asd, asd_count)
                asd_count += 1
            
            processed_time += strain.duration.value

        except Exception as e:
            logging.error(f"   Math error processing segment: {e}")

        current_time += this_chunk_size
        del strain
        
        # Memory cleanup
        if asd_count % 50 == 0:
            gc.collect()

    return cumulative_asd, asd_count, processed_time

def main():
    window_sizes = [
        600,      # 10m
        3600,     # 1h
        86400,    # 1d
        604800,   # 1w
        2592000,  # 30d
        5184000,  # 60d
        6912000  # 80d
        #8640000   # 100d
    ]

    # LOOP THROUGH DETECTORS (H1, then L1, etc.)
    for detector in detectors:
        logging.info(f"\n{'='*40}")
        logging.info(f"STARTING GENERATION FOR: {detector}")
        logging.info(f"{'='*40}")

        # 1. smart Resume: Load progress specifically for this detector
        curr_asd, curr_count, curr_processed, start_idx = load_checkpoint(detector, window_sizes)
        
        # 2. Expanding Window Loop
        for i in range(start_idx + 1, len(window_sizes)):
            target_win = window_sizes[i]
            logging.info(f"\n>>> EXPANDING TO WINDOW: {target_win/3600:.2f} Hours ({detector})")
            
            # Define the geometry
            t_center = MERGER_TIME
            
            # If this is the very first window (start from scratch)
            if i == 0:
                # Simple center block
                w_start = t_center - target_win/2
                w_end = t_center + target_win/2
                logging.info(f"Processing Center Block: {w_start} to {w_end}")
                curr_asd, curr_count, curr_processed = process_interval(
                    detector, w_start, w_end, curr_asd, curr_count, curr_processed
                )
            else:
                # THE DONUT STRATEGY
                # We already have data for window[i-1]. We only need the flanks.
                prev_win = window_sizes[i-1]
                
                # Left Flank: [New_Start, Old_Start]
                left_start = t_center - target_win/2
                left_end = t_center - prev_win/2
                
                # Right Flank: [Old_End, New_End]
                right_start = t_center + prev_win/2
                right_end = t_center + target_win/2
                
                logging.info(f"1. Processing Left Flank ({left_start} to {left_end})")
                curr_asd, curr_count, curr_processed = process_interval(
                    detector, left_start, left_end, curr_asd, curr_count, curr_processed
                )
                
                logging.info(f"2. Processing Right Flank ({right_start} to {right_end})")
                curr_asd, curr_count, curr_processed = process_interval(
                    detector, right_start, right_end, curr_asd, curr_count, curr_processed
                )

            # Save result for this level
            save_checkpoint(curr_asd, target_win, detector, curr_count, curr_processed)
            gc.collect()

    print("\n✓ ALL DETECTORS GENERATED SUCCESSFULLY")

if __name__ == "__main__":
    main()