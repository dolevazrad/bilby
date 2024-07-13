import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

# Define the event and get the initial strain data at detection
event = 'GW150914'
gps_time = event_gps(event)
detector = 'H1'  # Using H1 as an example

# Define durations in seconds (1 day, 5 days, 20 days, 50 days, 100 days)
durations = [86400, 5*86400, 20*86400, 50*86400, 100*86400]

# Function to fetch and process strain data
def fetch_and_process_strain(detector, start_time, duration):
    try:
        print(f"Fetching data for {detector} from {start_time} for {duration} seconds")
        strain = TimeSeries.fetch_open_data(detector, start_time, start_time + duration, cache=True)
        psd = strain.psd(4, 2)
        return psd
    except Exception as e:
        print(f"Error fetching data for {detector} at {start_time}: {e}")
        return None

# Define output directory
PARENT_LABEL = "Analyzing_GW_Noise"
BASE_OUTDIR = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'

# Create output directory if it does not exist
if not os.path.exists(BASE_OUTDIR):
    os.makedirs(BASE_OUTDIR)

# Fetch initial strain data at detection
initial_duration = 86400  # 1 day
initial_psd = fetch_and_process_strain(detector, gps_time, initial_duration)

# Fetch strain data for increasing durations to simulate improvement
improvement_psds = []
for duration in durations:
    psd = fetch_and_process_strain(detector, gps_time, duration)
    if psd is not None:
        improvement_psds.append((duration, psd))

# Plot initial noise strain
plt.figure(figsize=(10, 6))
plt.plot(initial_psd.frequencies, np.sqrt(initial_psd), label='Initial (1 day)')

# Plot improved noise strains
for duration, psd in improvement_psds:
    plt.plot(psd.frequencies, np.sqrt(psd), label=f'{duration//86400} days')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Strain noise')
plt.legend()
plt.title(f'Noise Strain Improvement over Time for {event}')
plt.grid(True)

# Save the plot
output_file = os.path.join(BASE_OUTDIR, 'noise_strain_improvement.png')
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
