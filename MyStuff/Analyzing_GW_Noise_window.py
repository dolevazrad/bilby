
import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import logging
import time
from json.decoder import JSONDecodeError
import requests
from datetime import date
import shutil
from astropy.utils.data import get_cached_urls, clear_download_cache
from astropy.time import Time
import glob
from astropy import units as u
import pickle
from scipy.interpolate import interp1d
from gwpy.detector import Channel
from gwpy.frequencyseries import FrequencySeries
from Analyzing_GW_Noise import check_and_clear_space, find_data_gaps, save_asd_and_plot, combine_asds
# Define output directory
PARENT_LABEL = "GW_Noise_H1_L1_window"
user = os.environ.get('USER', 'default_user')
if user == 'useradd':
    BASE_OUTDIR = f'/home/{user}/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
elif user == 'dolev':
    BASE_OUTDIR = f'/home/{user}/code/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)
def save_asd_and_plot_window(cumulative_asd, win_length, detector, total_processed_duration, 
                      event_time, asd_count):
    """
    Save and plot ASD data for a window centered around an event.
    
    Args:
        cumulative_asd: The calculated ASD
        win_length: Length of the window
        detector: Detector name ('H1' or 'L1')
        total_processed_duration: Total duration of processed data
        event_time: Time of the event
        asd_count: Number of ASDs averaged
    """
    start_time = event_time - win_length/2
    end_time = event_time + win_length/2
    
    filename = os.path.join(BASE_OUTDIR, f"{detector}_asd_win{win_length}.pkl")
    data = {
        'asd': cumulative_asd,
        'event_time': event_time,
        'window_length': win_length,
        'start_time': start_time,
        'end_time': end_time,
        'asd_count': asd_count,
        'processed_time': total_processed_duration
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Saved ASD data to {filename}")

    def format_duration(duration_hours):
        if duration_hours >= 1:
            return f"{duration_hours:.1f} hours"
        else:
            return f"{duration_hours*60:.1f} minutes"

    window_str = format_duration(win_length/3600)
    total_processed_str = format_duration(total_processed_duration/3600)

    plt.figure(figsize=(12, 8))
    if np.any(cumulative_asd.value > 0):
        frequencies = cumulative_asd.frequencies.value
        asd_values = cumulative_asd.value
        
        # Frequency range of interest
        f_min, f_max = 10, 1000  # Hz
        freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
        
        plt.loglog(frequencies[freq_mask], asd_values[freq_mask], label='ASD')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ASD (strain/√Hz)')
        plt.title(f'Noise ASD for {detector}\n'
                  f'Window size: {window_str}\n'
                  f'Total processed: {total_processed_str}\n'
                  f'Event time: {Time(event_time, format="gps").iso} (GPS: {event_time})\n'
                  f'Window: {Time(start_time, format="gps").iso} to {Time(end_time, format="gps").iso}')
        plt.grid(True)
        plt.legend()
        plt.xlim(f_min, f_max)
        
    else:
        plt.text(0.5, 0.5, 'No valid ASD data', ha='center', va='center', transform=plt.gcf().transFigure)
        plt.title(f'No valid ASD data for {detector}\n'
                  f'Window size: {window_str}\n'
                  f'Event time: {Time(event_time, format="gps").iso} (GPS: {event_time})\n'
                  f'Window: {Time(start_time, format="gps").iso} to {Time(end_time, format="gps").iso}')

    plt.tight_layout()
    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_ASD_win{win_length/3600:.3f}_hours.png")
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Plot saved to {plot_filename}")
    
def process_segment_window(segment, cumulative_asd, asd_count, processed_time, failed_time, 
                        event_time, detector, win_length, fftlength, overlap):
    """Process a segment of strain data for window-based analysis."""
    
    if segment.duration.value < 15:  # Minimum duration for a reliable calculation
        logging.info(f"Skipping segment shorter than 15 seconds: {segment.duration.value} seconds")
        failed_time += segment.duration.value
        return cumulative_asd, asd_count, processed_time, failed_time

    # Check for NaNs using find_data_gaps
    gaps = find_data_gaps(segment)
    if gaps:
        for start, end, duration, gap_type in gaps:
            logging.warning(f"{gap_type} from {start} to {end}, duration: {duration} seconds")
            failed_time += duration
        return cumulative_asd, asd_count, processed_time, failed_time

    try:
        # Adjust fftlength to be appropriate for segment length
        actual_fftlength = min(fftlength, segment.duration.value/2)  # Use at most half the segment length
        actual_overlap = actual_fftlength/2  # 50% overlap
        
        # Calculate ASD for this segment
        segment_asd = segment.asd(fftlength=actual_fftlength, 
                                overlap=actual_overlap,
                                method='median')

        if np.any(~np.isfinite(segment_asd.value)):
            logging.warning(f"Non-finite values in ASD for segment from {segment.t0.value} to {segment.t0.value + segment.duration.value}")
            failed_time += segment.duration.value
            return cumulative_asd, asd_count, processed_time, failed_time

        if cumulative_asd is None:
            cumulative_asd = segment_asd
            asd_count = 1
        else:
            # Combine ASDs using your existing combine_asds function
            cumulative_asd = combine_asds(cumulative_asd, segment_asd, asd_count)
            asd_count += 1
        
        processed_time += segment.duration.value
        logging.info(f"ASD calculated for segment: {segment.t0.value} to {segment.t0.value + segment.duration.value}, "
                    f"duration: {segment.duration.value}s, fftlength: {actual_fftlength}s. Total ASDs: {asd_count}")
        
        # Save intermediate results periodically
        if asd_count % 10 == 0:
            save_asd_and_plot_window(cumulative_asd, win_length, detector, processed_time, 
                                   event_time, asd_count)
        
    except Exception as e:
        logging.error(f"Error in ASD calculation: {str(e)}")
        logging.error(f"Segment duration: {segment.duration.value}s, FFT length: {actual_fftlength}s, Overlap: {actual_overlap}s")
        failed_time += segment.duration.value

    return cumulative_asd, asd_count, processed_time, failed_time
def fetch_and_process_strain_window(detector, event_time, win_length, fftlength, overlap):
    """Process strain data for a time window centered around an event."""
    start_time = event_time - win_length/2
    end_time = event_time + win_length/2
    
    if not check_and_clear_space():
        logging.error("Insufficient disk space. Aborting.")
        return 0, 0, 0

    # Initialize variables
    cumulative_asd = None
    asd_count = 0
    current_time = start_time
    processed_time = 0
    failed_time = 0
    consecutive_errors = 0
    max_consecutive_errors = 500000

    while current_time < end_time:
        if not check_and_clear_space():
            logging.error("Ran out of disk space during processing. Aborting.")
            break

        try:
            # Always move forward by at least 1 second
            chunk_size = max(min(fftlength, end_time - current_time), 1.0)
            interval_end = current_time + chunk_size
            
            logging.info(f"Fetching data from {current_time} to {interval_end}, chunk size: {chunk_size}")
            strain = TimeSeries.fetch_open_data(detector, current_time, interval_end, 
                                              cache=True, verbose=True)
            
            if strain is None or strain.duration.value <= 0:
                logging.info(f"No data available from {current_time} to {interval_end}")
                failed_time += chunk_size
                current_time += chunk_size
                continue

            # Process the strain data
            gaps = find_data_gaps(strain)
            if gaps:
                for start, end, duration, gap_type in gaps:
                    logging.info(f"{gap_type.capitalize()} from {start} to {end}, duration: {duration}s")
                    failed_time += duration

            # Process the valid data segment
            cumulative_asd, asd_count, processed_time, failed_time = process_segment_window(
                strain, cumulative_asd, asd_count, processed_time, failed_time,
                event_time, detector, win_length, fftlength, overlap
            )
            
            # Move forward by the actual data duration or at least 1 second
            time_advance = max(strain.duration.value, 1.0)
            current_time += time_advance
            
            logging.info(f'Progress - Current time: {current_time}, End time: {end_time}')
            logging.info(f'Processed: {processed_time:.2f}s, Failed: {failed_time:.2f}s')

        except Exception as e:
            logging.error(f"Error processing data from {current_time} to {interval_end}: {str(e)}")
            # Ensure we move forward even on error
            current_time += max(chunk_size, 1.0)
            failed_time += chunk_size
            
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logging.error(f"Too many consecutive errors ({max_consecutive_errors}). continue.")
                
        
        # Save intermediate results periodically
        if asd_count > 0 and asd_count % 10 == 0:
            save_asd_and_plot_window(cumulative_asd, win_length, detector, processed_time, 
                                   event_time, asd_count)
    
    # Save final results
    if cumulative_asd is not None:
        save_asd_and_plot_window(cumulative_asd, win_length, detector, processed_time, 
                                event_time, asd_count)
    
    return processed_time, failed_time, processed_time + failed_time
    
if __name__ == "__main__":
    merger_time = 1238303719  # 2019-04-03 05:15:19 UTC for GW190403_051519
    
    # Define window sizes to analyze
    window_sizes = [
        600,      # 10 minutes
        3600,     # 1 hour
        86400,    # 1 day
        604800,   # 1 week
        2592000,  # 30 days
        5184000   # 60 days
    ]

    # Other parameters
    fftlength = 600
    overlap = 300
    detectors = ['H1', 'L1']
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='strain_analysis.log'  # Save to file
    )
    
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    # Results dictionary to store all results
    results = {}
    
    for detector in detectors:
        results[detector] = {}
        logging.info(f"\nStarting analysis for detector {detector}")
        
        for win_length in window_sizes:
            logging.info(f"\nProcessing window size: {win_length/3600:.2f} hours")
            
            # Process the strain data
            processed_time, failed_time, total_duration = fetch_and_process_strain_window(
                detector=detector,
                event_time=merger_time,
                win_length=win_length,
                fftlength=fftlength,
                overlap=overlap
            )
            
            # Store results
            results[detector][win_length] = {
                'processed': processed_time,
                'failed': failed_time,
                'total': total_duration
            }
            
            # Log results for this window
            logging.info(
                f"Window {win_length/3600:.2f} hours completed:\n"
                f"  Success: {processed_time/3600:.2f} hours\n"
                f"  Failures: {failed_time/3600:.2f} hours\n"
                f"  Total: {total_duration/3600:.2f} hours"
            )
    
    # Print summary of all results
    logging.info("\n=== FINAL SUMMARY ===")
    for detector in detectors:
        logging.info(f"\nResults for {detector}:")
        for win_length, data in results[detector].items():
            logging.info(
                f"Window {win_length/3600:.2f} hours:\n"
                f"  Success: {data['processed']/3600:.2f} hours\n"
                f"  Failures: {data['failed']/3600:.2f} hours\n"
                f"  Total: {data['total']/3600:.2f} hours"
            )
            
    # Save results to a file
    import json
    from datetime import datetime
    
    # Convert results to a more readable format
    save_results = {
        detector: {
            f"{win_length/3600:.2f}h": {
                'processed_hours': data['processed']/3600,
                'failed_hours': data['failed']/3600,
                'total_hours': data['total']/3600
            }
            for win_length, data in detector_data.items()
        }
        for detector, detector_data in results.items()
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'strain_analysis_results_{timestamp}.json', 'w') as f:
        json.dump(save_results, f, indent=4)