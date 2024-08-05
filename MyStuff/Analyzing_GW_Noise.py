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

def check_disk_space(min_space_gb=10):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    return free_gb >= min_space_gb

def clear_astropy_cache(age_in_days=30, specific_url=None):
    """
    Clear Astropy cache files.
    
    Parameters:
    age_in_days (int): Clear files older than this many days. Default is 30.
    specific_url (str): If provided, clear only the cache for this specific URL.
    """
    if specific_url:
        try:
            clear_download_cache(specific_url)
            logging.info(f"Cleared cache for specific URL: {specific_url}")
        except Exception as e:
            logging.error(f"Failed to clear cache for URL {specific_url}: {e}")
    else:
        try:
            cached_urls = get_cached_urls()
            for url in cached_urls:
                try:
                    clear_download_cache(url)
                    logging.info(f"Cleared cache for URL: {url}")
                except Exception as e:
                    logging.error(f"Failed to clear cache for URL {url}: {e}")
            logging.info(f"Finished clearing Astropy cache files older than {age_in_days} days")
        except Exception as e:
            logging.error(f"Failed to clear Astropy cache: {e}")
    # logging.warning("Attempting fallback method to clear cache...")
    # try:
    #     cache_dir = os.path.expanduser('~/.astropy/cache')
    #     for root, dirs, files in os.walk(cache_dir):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             file_age = now - datetime.fromtimestamp(os.path.getmtime(file_path))
    #             if file_age > timedelta(days=age_in_days):
    #                 os.remove(file_path)
    #                 logging.info(f"Removed cached file: {file_path}")
    #     logging.info("Fallback cache clearing completed")
    # except Exception as e:
    #     logging.error(f"Fallback cache clearing failed: {e}")

def check_and_clear_space(min_space_gb=10, cache_age_days=30):
    """Check disk space and clear cache if necessary."""
    if not check_disk_space(min_space_gb):
        logging.warning("Low disk space. Attempting to clear Astropy cache.")
        clear_astropy_cache(age_in_days=cache_age_days)
        if not check_disk_space(min_space_gb):
            logging.error("Still low on disk space after clearing cache.")
            return False
    return True
# Function to fetch GPS time with robust error handling
def fetch_event_gps(event, max_retries=3, base_sleep=2):
    retry_count = 0
    while retry_count < max_retries:
        try:
            return event_gps(event)
        except (requests.exceptions.RequestException, JSONDecodeError) as e:
            logging.error(f"Error fetching GPS time for {event}: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise
            else:
                time.sleep(base_sleep * (2 ** retry_count))



# Define output directory
PARENT_LABEL = "Analyzing_GW_Noise"
user = os.environ.get('USER', 'default_user')
if user == 'useradd':
    BASE_OUTDIR = f'/home/{user}/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
elif user == 'dolev':
    BASE_OUTDIR = f'/home/{user}/code/bilby/MyStuff/my_outdir/{PARENT_LABEL}'
if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)
# Setup logging

today = date.today().strftime("%Y-%m-%d")
log_filename = f'process_log_{today}.log'

logging.basicConfig(filename=os.path.join(BASE_OUTDIR, log_filename), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print(f"Data will be saved in: {BASE_OUTDIR}")
#  time increments for loading data: 30 minutes, 1 hour, 48 hours, and 96 days

# Create output directory if it does not exist
if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)
# Fetch GPS time for the event



def save_psd_and_plot(cumulative_psd, increment, detector, total_processed_duration):
    filename = os.path.join(BASE_OUTDIR, f"{detector}_psd_{increment}.npy")
    np.save(filename, cumulative_psd.value)
    logging.info(f"Saved PSD to {filename} at {increment} seconds")

    plt.figure(figsize=(10, 6))
    if np.any(cumulative_psd.value > 0):
        plt.loglog(cumulative_psd.frequencies.value, cumulative_psd.value)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD ((strain^2)/Hz)')
        plt.title(f'Noise PSD after {increment/3600:.1f} hours for {detector}\nTotal processed: {total_processed_duration/3600:.1f} hours')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No valid PSD data', ha='center', va='center')
        plt.title(f'No valid PSD data for {detector} at {increment/3600:.1f} hours')

    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_PSD_{increment/3600:.1f}_hours.png")
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Plot saved to {plot_filename}")

def find_data_gaps(strain, gap_threshold=0.1, nan_threshold=1):
    """
    Identify gaps in the strain data, including NaN stretches and individual NaN values.
    gap_threshold is the minimum time gap size in seconds to be considered.
    nan_threshold is the minimum number of consecutive NaN values to be considered a gap.
    """
    time_array = strain.times.value
    data_array = strain.value
    dt = strain.dt.value
    gaps = []
    nan_start = None
    nan_count = 0

    for i in range(len(time_array)):
        if i > 0:
            # Check for time gaps
            gap_size = time_array[i] - time_array[i-1] - dt
            if gap_size > gap_threshold:
                gaps.append((time_array[i-1], time_array[i], gap_size, "time gap"))
        
        # Check for NaN values
        if np.isnan(data_array[i]):
            if nan_start is None:
                nan_start = time_array[i]
            nan_count += 1
        else:
            if nan_count > 0:
                gaps.append((nan_start, time_array[i], (time_array[i] - nan_start), f"NaN stretch ({nan_count} values)"))
            nan_start = None
            nan_count = 0

    # Check if there's a NaN stretch at the end of the data
    if nan_count > 0:
        gaps.append((nan_start, time_array[-1], (time_array[-1] - nan_start), f"NaN stretch ({nan_count} values)"))

    return gaps

def check_for_nans(data, data_type):
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values out of {data.size} values in {data_type}")
    return nan_count > 0

def fetch_and_process_strain(detector, start_time, increments, fftlength, max_retries=3, base_sleep=2):
    psd_fftlength = 15  # 15 seconds for PSD calculation
    
    if not check_and_clear_space():
        logging.error("Insufficient disk space. Aborting.")
        return 0, 0, 0

    max_duration = max(increments)
    end_time = start_time + max_duration
    cumulative_psd = None
    psd_count = 0
    processed_time = failed_time = 0
    current_time = start_time

    def process_segment(segment, is_last_segment=False):
        nonlocal cumulative_psd, psd_count, processed_time, failed_time

        if segment.duration.value < psd_fftlength:
            logging.info(f"Skipping segment shorter than PSD fftlength: {segment.duration.value} seconds")
            failed_time += segment.duration.value
            return

        # Check for NaNs using find_data_gaps
        gaps = find_data_gaps(segment)
        if gaps:
            for start, end, duration, gap_type in gaps:
                logging.warning(f"{gap_type} from {start} to {end}, duration: {duration} seconds")
                failed_time += duration
            return

        try:
            psd = segment.psd(fftlength=psd_fftlength, overlap=psd_fftlength/2, window='hann')
            if check_for_nans(psd.value, "PSD"):
                logging.warning(f"NaN values in PSD for segment from {segment.t0.value} to {segment.t0.value + segment.duration.value}")
                failed_time += segment.duration.value
                return

            if cumulative_psd is None:
                cumulative_psd = psd
                psd_count = 1
            else:
                if not np.allclose(cumulative_psd.frequencies.value, psd.frequencies.value):
                    logging.warning("Frequency mismatch detected. Resampling new PSD.")
                    psd = psd.interpolate(cumulative_psd.frequencies)
                
                cumulative_psd = cumulative_psd * (psd_count / (psd_count + 1)) + psd * (1 / (psd_count + 1))
                psd_count += 1
            
            processed_time += segment.duration.value
            logging.info(f"PSD calculated successfully for {'last ' if is_last_segment else ''}segment from {segment.t0.value} to {segment.t0.value + segment.duration.value}, duration: {segment.duration.value} seconds. Cumulative PSD count: {psd_count}")
            logging.info(f"Total processed duration: {processed_time} seconds")
            
            increment = segment.t0.value - start_time + segment.duration.value
            if any(inc <= increment < inc + fftlength for inc in increments):
                logging.info(f"Plotting for increment: {increment}")
                save_psd_and_plot(cumulative_psd, increment, detector, processed_time)

        except ValueError as ve:
            logging.error(f"Error in PSD calculation for {'last ' if is_last_segment else ''}segment: {ve}")
            failed_time += segment.duration.value

    while current_time < end_time:
        if not check_and_clear_space():
            logging.error("Ran out of disk space during processing. Aborting.")
            break

        try:
            interval_end = min(current_time + fftlength, end_time)
            strain = TimeSeries.fetch_open_data(detector, current_time, interval_end, cache=True, verbose=True)
            
            if strain.duration.value > 0:
                gaps = find_data_gaps(strain)
                if gaps:
                    logging.info(f"Found {len(gaps)} gaps in data from {strain.t0.value} to {strain.t0.value + strain.duration.value}")
                    for start, end, duration, gap_type in gaps:
                        logging.info(f"{gap_type.capitalize()} from {start} to {end}, duration: {duration} seconds")
                        failed_time += duration

                last_end = strain.times.value[0]
                for i, (gap_start, gap_end, _, _) in enumerate(gaps + [(min(strain.times.value[-1], end_time), None, None, None)]):
                    if gap_start > last_end:
                        segment_duration = min(gap_start, end_time) - last_end
                        logging.info(f'Processing segment from {last_end} to {min(gap_start, end_time)} at size {segment_duration}')
                        if segment_duration >= psd_fftlength:
                            segment = strain.crop(last_end, min(gap_start, end_time))
                            process_segment(segment)
                        else:
                            logging.info(f"Skipping segment shorter than PSD fftlength: {segment_duration} seconds")
                            failed_time += segment_duration
                    last_end = gap_end if gap_end is not None else min(gap_start, end_time)

                current_time = min(strain.times.value[-1], end_time)
            else:
                logging.info(f"No data available from {current_time} to {interval_end}")
                failed_time += min(fftlength, end_time - current_time)
                current_time += min(fftlength, end_time - current_time)

            logging.info(f'Processed time: {processed_time:.2f} seconds, Failed time: {failed_time:.2f} seconds')

            if current_time >= end_time:
                logging.info(f"Reached or exceeded end time {end_time}. Stopping processing.")
                break

        except Exception as e:
            logging.error(f"Error processing data from {current_time} to {interval_end}: {e}")
            failed_time += min(fftlength, end_time - current_time)
            current_time += min(fftlength, end_time - current_time)
            logging.info(f'Processed time: {processed_time:.2f} seconds, Failed time: {failed_time:.2f} seconds')

    return processed_time, failed_time, processed_time + failed_time
if __name__ == "__main__":
    increments = [600, 1800, 3600, 86400, 604800, 172800, 8294400]
    fftlength = 600
    overlap = 300
    start_time = 1238166018
    detector = 'H1'      

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    processed_time, failed_time, total_duration = fetch_and_process_strain('H1', start_time, increments, fftlength)
    print(f"All analysis completed. Success: {processed_time/3600:.2f} hours, Failures: {failed_time/3600:.2f} hours, Total Duration: {total_duration/3600:.2f} hours")
    logging.info(f"All analysis completed. Success: {processed_time/3600:.2f} hours, Failures: {failed_time/3600:.2f} hours, Total Duration: {total_duration/3600:.2f} hours")