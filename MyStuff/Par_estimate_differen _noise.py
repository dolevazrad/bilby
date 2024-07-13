import os
import time

import numpy as np
import bilby
import matplotlib.pyplot as plt
from bilby.gw.detector import InterferometerList

# Define the sampling frequency and duration of the data segment
sampling_frequency = 2048
duration = 4

# Define the waveform generator arguments
waveform_arguments = {
    'waveform_approximant': 'IMRPhenomXP',
    'reference_frequency': 50.0,
}

# Define the injection parameters for the signal
injection_parameters = {
    'mass_1': 36.0,
    'mass_2': 29.0,
    'a_1': 0.4,
    'a_2': 0.3,
    'tilt_1': 0.5,
    'tilt_2': 1.0,
    'phi_12': 1.7,
    'phi_jl': 0.3,
    'luminosity_distance': 1000.0,
    'theta_jn': 0.4,
    'phase': 1.3,
    'ra': 1.375,
    'dec': -1.2108,
    'geocent_time': 1126259642.413,
    'psi': 2.659,
}

parent_label = "Same_noise"

def generate_or_load_noise(ifos, noise_time, sampling_frequency, duration, outdir):
    # Path for noise file
    noise_file_path = os.path.join(outdir, f'noise_{noise_time}.npy')
    if not os.path.isfile(noise_file_path):
        # Generate noise manually and save it
        noise_data = []
        for ifo in ifos:
            noise_td = np.random.normal(0, 1, int(sampling_frequency * duration))
            noise_fd = np.fft.rfft(noise_td)
            if noise_time != 0:
                noise_fd /= np.sqrt(noise_time)  # Example modification to the noise
            ifo.strain_data.set_from_frequency_domain_strain(noise_fd,sampling_frequency = sampling_frequency, duration = duration)
            noise_data.append(noise_fd)
        np.save(noise_file_path, noise_data)
    else:
        # Load the noise from file
        noise_data = np.load(noise_file_path, allow_pickle=True)
        for ifo, data in zip(ifos, noise_data):
            ifo.strain_data.set_from_frequency_domain_strain(data, duration = duration, sampling_frequency = sampling_frequency)

def timed_run_with_noise_time(noise_time, label):
    start_time = time.time()  # Start time
    run_with_noise_time(noise_time=noise_time, label=label)
    end_time = time.time()  # End time
    print(f"Function run_with_noise_time with label='{label}' took {end_time - start_time:.4f} seconds.")

def run_with_noise_time(noise_time, label='BaseLine'):
    outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{parent_label}/{label}'
    os.makedirs(outdir, exist_ok=True)
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    ifos = InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - 2,
    )
    generate_or_load_noise(ifos, noise_time, sampling_frequency, duration, outdir)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments=waveform_arguments
    )
        

    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)


    priors = bilby.gw.prior.BBHPriorDict(injection_parameters.copy())
    priors["mass_1"] = bilby.core.prior.Uniform(25, 40, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(25, 40, "mass_2")
    priors["luminosity_distance"] = bilby.core.prior.Uniform(400, 2000, "luminosity_distance")


    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator, priors=priors
    )


    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=100,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        sample="unif"
    )

    tru = {'mass_1': 36.0, 'mass_2': 29.0, 'luminosity_distance': 1000.0}
    result.plot_corner(truths=tru, filename=f'{outdir}/{label}_corner.png')


    plt.close()

if __name__ == "__main__":
    noise_times = [0.0, 250.0, 500.0, 750.0, 1000.0]
    for noise_time in noise_times:
        timed_run_with_noise_time(noise_time=noise_time, label=f'noise_time_{noise_time}')