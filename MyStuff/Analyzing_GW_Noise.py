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


detector = 'H1'      

def save_psd_and_plot(cumulative_psd, increment, detector):
    filename = f"{BASE_OUTDIR}/{detector}_psd_{increment}.npy"
    np.save(filename, cumulative_psd.value)
    logging.info(f"Saved PSD to {filename} at {increment} seconds")

    plt.figure(figsize=(10, 4))
    plt.loglog(cumulative_psd.frequencies.value, cumulative_psd.value, label=f'{increment/3600:.1f} hours')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD ((strain^2)/Hz)')
    plt.title(f'Noise PSD after {increment/3600:.1f} hours for {detector}')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_PSD_{increment/3600:.1f}_hours.png")
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Plot saved to {plot_filename}")

def fetch_and_process_strain(detector, start_time, increments, max_retries=3, base_sleep=2):
    if not check_and_clear_space():
        logging.error("Insufficient disk space. Aborting.")
        return 0, 0
    max_duration = max(increments)
    end_time = start_time + max_duration
    cumulative_psd = None
    
    success_count = 0
    failure_count = 0
    current_time = start_time

    while current_time < end_time:
        if not check_and_clear_space():
            logging.error("Ran out of disk space during processing. Aborting.")
            break
        try:
            # Try to fetch data for the full interval
            interval_end = min(current_time + fftlength, end_time)
            strain = TimeSeries.fetch_open_data(detector, current_time, interval_end, cache=True, verbose=True)
            
            # Process whatever data we received
            if strain.duration.value > 0:
                psd = strain.psd(fftlength=fftlength, overlap=overlap, window='hann')
                cumulative_psd = psd if cumulative_psd is None else cumulative_psd + psd

                increment = current_time - start_time + strain.duration.value
                if any(inc <= increment < inc + fftlength for inc in increments):
                    save_psd_and_plot(cumulative_psd, increment, detector)

                success_count += 1
                logging.info(f"Processed {strain.duration.value} seconds of data from {current_time}")
            else:
                logging.info(f"No data available from {current_time} to {interval_end}")
                failure_count += 1

            # Move to the next time interval
            current_time += strain.duration.value if strain.duration.value > 0 else fftlength
            logging.info(f'success_count:{success_count} out of {failure_count+success_count}')

        except Exception as e:
            logging.error(f"Error processing data from {current_time} to {interval_end}: {e}")
            failure_count += 1
            # Move to the next time interval even if there was an error
            current_time += fftlength
            logging.info(f'success_count:{success_count} out of {failure_count+success_count}')


    return success_count, failure_count

# def fetch_and_process_strain(detector, start_time, increments, max_retries=3, base_sleep=2):
#     """ Fetch and process strain data incrementally and save at specified durations. """
#     max_duration = max(increments)
#     end_time = start_time + max_duration
#     cumulative_psd = None
    
#     success_count = 0
#     failure_count = 0
#     count = 0
#     current_time = start_time
#     while current_time < end_time:
#         retry_count = 0
#         logging.info(f'success_count:{success_count} out of {count}')
#         while retry_count < max_retries:
#             try:
#                 # Fetch data for the current 30-minute interval
#                 interval_end = min(current_time + fftlength, end_time)
#                 strain = TimeSeries.fetch_open_data(detector, current_time, interval_end, cache=True, verbose=True )
                
#                 # Compute the PSD of this interval
#                 psd = strain.psd(fftlength=fftlength, overlap=overlap, window='hann')
#                 #psd = strain.psd(4, 2)

#                 # Sum the PSDs cumulatively
#                 cumulative_psd = psd if cumulative_psd is None else cumulative_psd + psd

#                 # Check if it's time to save data at one of the increments
#                 increment = interval_end - start_time
#                 if increment in increments:
#                     filename = f"{BASE_OUTDIR}/{detector}_psd_{increment}.npy"
#                     np.save(filename, cumulative_psd.value)  # Save the cumulative PSD value array
#                     logging.info(f"Saved PSD to {filename} at {increment} seconds")
#                     plt.figure(figsize=(10, 4))
#                     plt.loglog(cumulative_psd.frequencies.value, cumulative_psd.value, label=f'{increment/3600:.1f} hours')
#                     plt.xlabel('Frequency (Hz)')
#                     plt.ylabel('PSD ((strain^2)/Hz)')
#                     plt.title(f'Noise PSD after {increment/3600:.1f} hours for {detector}')
#                     plt.legend()
#                     plt.grid(True)

#                     # Save plot
#                     plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_PSD_{increment/3600:.1f}_hours.png")
#                     plt.savefig(plot_filename)
#                     plt.close()
#                     logging.info(f"Plot saved to {plot_filename}")
#                 success_count = success_count + 1
#                 count = count + 1
#                 current_time = current_time + fftlength
#                 logging.info(f" fetching data from {current_time} to {interval_end} succeeded ")
#             except Exception as e:
#                 retry_count += 1
#                 logging.error(f"Error fetching data from {current_time} to {interval_end}: {e} retry:{retry_count} out of {max_retries}")
#                 if retry_count >= max_retries:
#                     current_time = current_time + fftlength # Skip to next interval if max retries reached
#                     failure_count = failure_count + 1
#                     count = count + 1
#                 else:
#                     time.sleep(base_sleep * (2 ** retry_count))  # Exponential backoff

#                     logging.error(f"Error fetching data from {current_time} to {interval_end}: {e} in {max_retries} tries")
#     return success_count, failure_count



if __name__ == "__main__":
    increments = [600, 1800, 3600, 86400, 604800, 172800, 8294400]  # 30 mins, 1 hour, 48 hours, 96 days
    fftlength = 600 #1800  # 30 minutes
    overlap = 300#900    # 15 minutes
    start_time = 1238166018  # O3a dates: 1st April 2019 15:00 UTC (GPS 1238166018) to 1st Oct 2019 15:00 UTC (GPS 1253977218)

    success_count, failure_count = fetch_and_process_strain(detector, start_time, increments)
    print(f"All analysis completed. Success: {success_count}, Failures: {failure_count}")
    logging.info(f"All analysis completed. Success: {success_count}, Failures: {failure_count}")