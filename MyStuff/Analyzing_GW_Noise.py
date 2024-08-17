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

def get_representative_asd(detector, start_time, duration=4096):
    """
    Get a representative ASD for the detector using gwpy and open data.
    
    Parameters:
    detector (str): Detector name ('H1' or 'L1')
    start_time (float): GPS start time of the data
    duration (int): Duration of data to fetch in seconds
    
    Returns:
    FrequencySeries: Representative ASD
    """
    try:
        # Fetch strain data
        strain = TimeSeries.fetch_open_data(detector, start_time, start_time + duration)
        
        # Calculate ASD
        asd = strain.asd(fftlength=4, overlap=2)
        return asd
    except Exception as e:
        logging.error(f"Error fetching open data: {e}")
        return None


def interpolate_asd(source_freqs, source_asd, target_freqs):
    """
    Interpolate the ASD to match the target frequency range.
    
    Parameters:
    source_freqs (array): Original frequency array
    source_asd (array): Original ASD values
    target_freqs (array): Target frequency array
    
    Returns:
    array: Interpolated ASD values
    """
    interpolator = interp1d(source_freqs, source_asd, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interpolator(target_freqs)
    
# Define output directory
PARENT_LABEL = "Analyzing_GW_Noise_v3"
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

def load_latest_psd(base_dir, detector):
    psd_files = glob.glob(os.path.join(base_dir, f"{detector}_psd_*.pkl"))
    if not psd_files:
        return None, None, None, None, None
    
    latest_file = max(psd_files, key=os.path.getctime)
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    return (data['psd'], data['time'], data['failed_time'], 
            data['psd_count'], data['processed_time'])
def get_next_increment_index(current_time, start_time, increments):
    passed_time = current_time - start_time
    for i, inc in enumerate(increments):
        if passed_time <= inc:
            return i
    return len(increments)  # If we've passed all increments


def save_psd_and_plot(cumulative_psd, increment, detector, total_processed_duration, 
                      start_time, increment_name, psd_count, increments):
    filename = os.path.join(BASE_OUTDIR, f"{detector}_psd_{increment}.pkl")
    data = {
        'psd': cumulative_psd,
        'time': start_time + increment,
        'failed_time': 0,  # You may want to track this separately
        'psd_count': psd_count,
        'processed_time': total_processed_duration
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Saved PSD data to {filename} at {increment} seconds")#TODO:no need to save every incremnt

    end_time = start_time + increment
    start_date = Time(start_time, format='gps').iso
    end_date = Time(end_time, format='gps').iso

    def format_duration(duration_hours):
        if duration_hours >= 1:
            return f"{duration_hours:.1f} hours"
        else:
            return f"{duration_hours*60:.1f} minutes"

    increment_str = format_duration(increment/3600)
    total_processed_str = format_duration(total_processed_duration/3600)

    plt.figure(figsize=(12, 8))
    if np.any(cumulative_psd.value > 0):
        frequencies = cumulative_psd.frequencies.value
        psd_values = cumulative_psd.value
        
        # Get representative ASD
        try:
            true_asd = get_representative_asd(detector, start_time)
        except Exception as e:
            logging.error(f"Error getting representative ASD: {e}")
            true_asd = None
        
        # Frequency range of interest
        f_min, f_max = 10, 2000  # Hz
        freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
        
        plt.subplot(2, 1, 1)
        plt.loglog(frequencies[freq_mask], np.sqrt(psd_values[freq_mask]), label='Estimated ASD')
        if true_asd is not None:
            true_asd_interp = true_asd.interpolate(frequencies)
            plt.loglog(frequencies[freq_mask], np.sqrt(true_asd_interp.value[freq_mask]), 'r--', label='Open Data ASD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ASD (strain/√Hz)')
        plt.title(f'Noise ASD for {detector}\n'
                  f'Duration: {increment_str}\n'
                  f'Total processed: {total_processed_str}\n'
                  f'Start: {Time(start_time, format="gps").iso} (GPS: {start_time})\n'
                  f'End: {Time(start_time + increment, format="gps").iso} (GPS: {start_time + increment})')
        plt.grid(True)
        plt.legend()
        plt.xlim(f_min, f_max)
        
        if true_asd is not None:
            plt.subplot(2, 1, 2)
            estimated_asd = np.sqrt(psd_values[freq_mask])
            true_asd_values = np.sqrt(true_asd_interp.value[freq_mask])
            # Avoid division by zero and log of zero
            mask = (estimated_asd > 0) & (true_asd_values > 0)
            ratio = np.zeros_like(estimated_asd)
            ratio[mask] = np.abs(np.log(estimated_asd[mask] / true_asd_values[mask]))
            plt.semilogx(frequencies[freq_mask][mask], ratio[mask])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('|Log(estimated/open data)|')
            plt.title(f'Ratio of Estimated to Open Data ASD (for ~{increment/60:.1f} minutes increment)')
            plt.grid(True)
            plt.xlim(f_min, f_max)
        else:
            plt.subplot(2, 1, 2)
            plt.text(0.5, 0.5, 'Open Data ASD not available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No valid PSD data', ha='center', va='center', transform=plt.gcf().transFigure)
        plt.title(f'No valid PSD data for {detector}\n'
                  f'Duration: {increment/60:.1f} minutes\n'
                  f'Start: {Time(start_time, format="gps").iso} (GPS: {start_time})\n'
                  f'End: {Time(start_time + increment, format="gps").iso} (GPS: {start_time + increment})')

    plt.tight_layout()
    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_ASD_{increment/3600:.3f}_hours.png")
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

def process_segment(segment, next_increment_index, cumulative_psd, psd_count, processed_time, failed_time, start_time, detector, increments, is_last_segment=False):
    psd_fftlength = 15  # 15 seconds for PSD calculation

    if segment.duration.value < psd_fftlength:
        logging.info(f"Skipping segment shorter than PSD fftlength: {segment.duration.value} seconds")
        failed_time += segment.duration.value
        return cumulative_psd, psd_count, processed_time, failed_time, next_increment_index

    # Check for NaNs using find_data_gaps
    gaps = find_data_gaps(segment)
    if gaps:
        for start, end, duration, gap_type in gaps:
            logging.warning(f"{gap_type} from {start} to {end}, duration: {duration} seconds")
            failed_time += duration
        return cumulative_psd, psd_count, processed_time, failed_time, next_increment_index

    try:
        segment_psd = segment.psd(fftlength=psd_fftlength, overlap=psd_fftlength/2, window='hann')
        if check_for_nans(segment_psd.value, "PSD"):
            logging.warning(f"NaN values in PSD for segment from {segment.t0.value} to {segment.t0.value + segment.duration.value}")
            failed_time += segment.duration.value
            return cumulative_psd, psd_count, processed_time, failed_time, next_increment_index

        if cumulative_psd is None:
            cumulative_psd = segment_psd
            psd_count = 1
        else:
            # Ensure consistent frequency bins
            if not np.array_equal(cumulative_psd.frequencies, segment_psd.frequencies):
                logging.warning("Frequency mismatch detected. Resampling new PSD.")
                segment_psd = segment_psd.interpolate(cumulative_psd.frequencies)
            
            # Store the unit for later reattachment
            original_unit = cumulative_psd.unit
            
            # Remove units for calculation
            cumulative_value = cumulative_psd.value
            segment_value = segment_psd.value
            
            # Perform the combination
            combined_value = (cumulative_value * psd_count + segment_value) / (psd_count + 1)
            
            # Reattach the unit
            cumulative_psd = FrequencySeries(combined_value, frequencies=cumulative_psd.frequencies, unit=original_unit)
        
        psd_count += 1
        
        processed_time += segment.duration.value
        logging.info(f"PSD calculated successfully for {'last ' if is_last_segment else ''}segment from {segment.t0.value} to {segment.t0.value + segment.duration.value}, duration: {segment.duration.value} seconds. Cumulative PSD count: {psd_count}")
        logging.info(f"Total processed duration: {processed_time} seconds")
        
        increment = processed_time
        while next_increment_index < len(increments) and increment >= increments[next_increment_index]:
            inc = increments[next_increment_index]
            logging.info(f"Plotting for increment: {inc}")
            save_psd_and_plot(cumulative_psd, increment, detector, processed_time, start_time, inc, psd_count, increments)
            next_increment_index += 1
        
    except Exception as e:
        logging.error(f"Error in PSD calculation for {'last ' if is_last_segment else ''}segment: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {e.args}")
        failed_time += segment.duration.value

    return cumulative_psd, psd_count, processed_time, failed_time, next_increment_index
def fetch_and_process_strain(detector, start_time, end_time, increments, fftlength):
    if not check_and_clear_space():
        logging.error("Insufficient disk space. Aborting.")
        return 0, 0, 0

    latest_psd, processed_time, failed_time, psd_count, latest_time = load_latest_psd(BASE_OUTDIR, detector)
    if latest_psd is not None and latest_time is not None:
        cumulative_psd = latest_psd
        current_time = latest_time
        next_increment_index = get_next_increment_index(current_time - start_time, 0, increments)
        logging.info(f"Continuing from time {current_time}, with {psd_count} PSDs. Next increment index: {next_increment_index}")
    else:
        cumulative_psd = None
        psd_count = 0
        current_time = start_time
        processed_time = 0
        failed_time = 0
        next_increment_index = 0

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
                        if segment_duration >= 15:  # psd_fftlength
                            segment = strain.crop(last_end, min(gap_start, end_time))
                            cumulative_psd, psd_count, processed_time, failed_time, next_increment_index = process_segment(
                                segment, next_increment_index, cumulative_psd, psd_count, processed_time, failed_time,
                                start_time, detector, increments
                            )
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
            with open(os.path.join(BASE_OUTDIR, f"{detector}_failed_time.txt"), 'w') as f:
                f.write(str(failed_time))
            if current_time >= end_time:
                logging.info(f"Reached or exceeded end time {end_time}. Stopping processing.")
                break

        except Exception as e:
            logging.error(f"Error processing data from {current_time} to {interval_end}: {str(e)}")
            failed_time += min(fftlength, end_time - current_time)
            current_time += min(fftlength, end_time - current_time)
            logging.info(f'Processed time: {processed_time:.2f} seconds, Failed time: {failed_time:.2f} seconds')
    
    if cumulative_psd is not None:
        save_psd_and_plot(cumulative_psd, processed_time, detector, processed_time, start_time, processed_time, psd_count, increments)
    
    return processed_time, failed_time, processed_time + failed_time
if __name__ == "__main__":
    increments = [600, 1800, 3600, 86400, 172800, 604800, 4147200, 8294400, 8295000]
    fftlength = 600
    overlap = 300
    start_time = 1238166018
    end_time = start_time + max(increments)  
    detector = 'H1'      
    

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    processed_time, failed_time, total_duration = fetch_and_process_strain(
        detector, start_time, end_time, increments, fftlength)    
    print(f"All analysis completed. Success: {processed_time/3600:.2f} hours, "
          f"Failures: {failed_time/3600:.2f} hours, "
          f"Total Duration: {total_duration/3600:.2f} hours")
    logging.info(f"All analysis completed. Success: {processed_time/3600:.2f} hours, "
                 f"Failures: {failed_time/3600:.2f} hours, "
                 f"Total Duration: {total_duration/3600:.2f} hours")