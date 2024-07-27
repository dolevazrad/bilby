import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import logging
import time
# Define the event and get the initial strain data at detection
event = 'GW150914'
gps_time = event_gps(event)
detector = 'H1'  # Using H1 as an example

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
logging.basicConfig(filename=os.path.join(BASE_OUTDIR, 'process_log.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
print(f"Data will be saved in: {BASE_OUTDIR}")
#  time increments for loading data: 30 minutes, 1 hour, 48 hours, and 96 days
increments = [1800, 3600, 172800, 8294400]  # 30 mins, 1 hour, 48 hours, 96 days
fftlength = 60 #1800  # 30 minutes
overlap = 30#900    # 15 minutes
# Create output directory if it does not exist
if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)
    
def fetch_and_process_strain(detector, start_time, increments, max_retries=3):
    """ Fetch and process strain data incrementally and save at specified durations. """
    max_duration = max(increments)
    end_time = start_time + max_duration
    cumulative_psd = None
    
    success_count = 0
    failure_count = 0
    count = 0
    current_time = start_time
    while current_time < end_time:
        retry_count = 0
        logging.info(f'success_count:{success_count} out of {count}')
        while retry_count < max_retries:
            try:
                # Fetch data for the current 30-minute interval
                interval_end = min(current_time + fftlength, end_time)
                strain = TimeSeries.fetch_open_data(detector, current_time, interval_end, cache=True)
                
                # Compute the PSD of this interval
                psd = strain.psd(fftlength=fftlength, overlap=overlap, window='hann')
                
                # Sum the PSDs cumulatively
                cumulative_psd = psd if cumulative_psd is None else cumulative_psd + psd

                # Check if it's time to save data at one of the increments
                increment = interval_end - start_time
                if increment in increments:
                    filename = f"{BASE_OUTDIR}/{detector}_psd_{increment}.npy"
                    np.save(filename, cumulative_psd.value)  # Save the cumulative PSD value array
                    logging.info(f"Saved PSD to {filename} at {increment} seconds")
                    plt.figure(figsize=(10, 4))
                    plt.loglog(cumulative_psd.frequencies.value, cumulative_psd.value, label=f'{increment/3600:.1f} hours')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('PSD ((strain^2)/Hz)')
                    plt.title(f'Noise PSD after {increment/3600:.1f} hours for {detector}')
                    plt.legend()
                    plt.grid(True)

                    # Save plot
                    plot_filename = os.path.join(BASE_OUTDIR, f"{detector}_noise_PSD_{increment/3600:.1f}_hours.png")
                    plt.savefig(plot_filename)
                    plt.close()
                    logging.info(f"Plot saved to {plot_filename}")
                success_count = success_count + 1
                count = count + 1
                current_time = current_time + fftlength
                logging.error(f" fetching data from {current_time} to {interval_end} succeeded ")
            except Exception as e:
                retry_count += 1
                logging.error(f"Error fetching data from {current_time} to {interval_end}: {e} retry:{retry_count} out of {max_retries}")
                if retry_count >= max_retries:
                    current_time = current_time + fftlength # Skip to next interval if max retries reached
                    failure_count = failure_count + 1
                    count = count + 1

                    logging.error(f"Error fetching data from {current_time} to {interval_end}: {e} in {max_retries} tries")



if __name__ == "__main__":
    fetch_and_process_strain(detector, gps_time, increments)
    print("All analysis completed.")
    logging.info("All analysis completed.")
