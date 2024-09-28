import bilby
import matplotlib.pyplot as plt
import os
import pickle
import logging
import numpy as np
from datetime import date

def load_noise_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    asd = data['asd']
    df = asd.df.value  # assuming df is in Hz
    n_points = len(asd)
    frequency = np.arange(n_points) * df
    
    return frequency, asd

def run_parameter_estimation(noise_file_path, start_time, injection_parameters):
    duration = 4
    sampling_frequency = 2048
    
    # Setup output directory
    base_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir'
    label = f"GW190403_051519_{os.path.splitext(os.path.basename(noise_file_path))[0]}"
    outdir = os.path.join(base_dir, "Analyzing_GW_Noise_v3", "corner_plots")
    
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Setup logging
    today = date.today().strftime("%Y-%m-%d")
    log_filename = f'process_log_{today}.log'
    logging.basicConfig(filename=os.path.join(outdir, log_filename), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP",
        reference_frequency=50.0,
    )

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    
    # Load noise from PKL file
    freq, asd = load_noise_from_pkl(noise_file_path)
    for ifo in ifos:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freq, asd_array=asd.value
        )
    
    ifos.set_strain_data_from_power_spectral_densities(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
    )

    _ = ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    priors = bilby.gw.prior.BBHPriorDict(injection_parameters.copy())
    priors["mass_1"] = bilby.core.prior.Uniform(70, 100, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(70, 100, "mass_2")
    priors["luminosity_distance"] = bilby.core.prior.Uniform(
        7000, 9500, "luminosity_distance"
    )

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )

    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=1000,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        sample="unif",
    )

    return result

# Start time for GW190403_051519
start_time = 1238166919  # GPS time for 2019-04-03 05:15:19 UTC

# Parameter values for GW190403_051519
injection_parameters = dict(
    mass_1=85.0,
    mass_2=85.0,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=8300.0,
    theta_jn=0.68,
    phase=0.0,
    ra=1.77,
    dec=0.91,
    geocent_time=start_time,
    psi=0.0,
)

noise_files = [
    "H1_asd_1199.9995171875.pkl",
    "H1_asd_2399.9990234375.pkl",
    "H1_asd_4199.9982910625.pkl",
    "H1_asd_86999.9649990938.pkl",
    "H1_asd_173999.9294943938.pkl",
    "H1_asd_603999.7356621094.pkl",
    "H1_asd_4147798.3122558594.pkl"
]
# Define the base directory and the specific subdirectory
base_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir'
noise_dir = "Analyzing_GW_Noise_v3"

# Get the full path to the noise directory
full_noise_dir = os.path.join(base_dir, noise_dir)

# Get all .pkl files in the noise directory
noise_files = [f for f in os.listdir(full_noise_dir) if f.endswith('.pkl') and f.startswith('H1_asd_')]

# Sort the files to ensure consistent ordering
noise_files.sort()

# Print the files found (for debugging)
print("Noise files found:")
for file in noise_files:
    print(file)

results = []
for noise_file in noise_files:
    # Construct the full path to the noise file
    noise_file_path = os.path.join(full_noise_dir, noise_file)
    result = run_parameter_estimation(noise_file_path, start_time, injection_parameters)
    results.append(result)

# Compare results
for i, result in enumerate(results):
    plt.figure(figsize=(10, 8))
    result.plot_corner(parameters=['mass_1', 'mass_2', 'luminosity_distance'])
    plt.title(f"Results for GW190403_051519 using {noise_files[i]}")
    plt.savefig(os.path.join(full_noise_dir, f"GW190403_051519_corner_plot_{os.path.splitext(noise_files[i])[0]}.png"))
    plt.close()

print("Analysis complete. Corner plots have been saved.")