import os
import numpy as np
import bilby
from scipy.interpolate import interp1d
from bilby.gw.detector import InterferometerList
from scipy.signal import gaussian, convolve
import matplotlib.pyplot as plt
from HelperFunction import plot_noise_comparison_subplots
from scipy.signal import fftconvolve

# Define constants
SAMPLING_FREQUENCY = 2048
DURATION = 4
START_TIME = 1126259642.413 - 2
WAVEFORM_ARGUMENTS = {
    'waveform_approximant': 'IMRPhenomXP',
    'reference_frequency': 50.0
}
INJECTION_PARAMETERS = {
    'mass_1': 36.0, 'mass_2': 29.0, 'a_1': 0.4, 'a_2': 0.3,
    'tilt_1': 0.5, 'tilt_2': 1.0, 'phi_12': 1.7, 'phi_jl': 0.3,
    'luminosity_distance': 1000.0, 'theta_jn': 0.4, 'phase': 1.3,
    'ra': 1.375, 'dec': -1.2108, 'geocent_time': START_TIME, 'psi': 2.659
}
PARENT_LABEL = "Noise_Study"
BASE_OUTDIR = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{PARENT_LABEL}'


def setup_interferometers():
    ifos = InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=INJECTION_PARAMETERS['geocent_time']
    )
    return ifos
def generate_high_resolution_noise(outdir, force_regenerate=False):
    """Generate or load high-resolution noise."""
    noise_path = os.path.join(outdir, 'high_res_noise.npy')
    if not os.path.exists(noise_path) or force_regenerate:
        # Generate noise
        ifos = setup_interferometers()
        noise = generate_noise(ifos)
        np.save(noise_path, noise)
    else:
        # Load existing noise
        noise = np.load(noise_path)
    return noise
def generate_noise(ifo):
    """ Generate frequency domain noise directly matching the interferometer's frequency array. """
    freqs = ifo.frequency_array
    psd_values = ifo.power_spectral_density.psd_array

    # Interpolate PSD to the interferometer's frequency array if needed
    psd_interp = interp1d(ifo.power_spectral_density.frequency_array, psd_values, kind='linear', fill_value="extrapolate")
    interpolated_psd_values = psd_interp(freqs)

    noise_fd = (np.random.normal(size=len(interpolated_psd_values)) + 1j * np.random.normal(size=len(interpolated_psd_values))) * np.sqrt(interpolated_psd_values / 2)
    return noise_fd, freqs
def generate_initial_noise(ifos, outdir):
    """ Generate and save initial high-resolution noise for all interferometers. """
    noise_path = os.path.join(outdir, 'initial_high_res_noise.npy')
    if not os.path.exists(noise_path):
        noise_data = []
        for ifo in ifos:
            freqs = ifo.frequency_array
            psd = ifo.power_spectral_density
            # Interpolate PSD to match the frequency bins
            psd_interp_func = interp1d(psd.frequency_array, psd.psd_array, bounds_error=False, fill_value="extrapolate")
            interpolated_psd_values = psd_interp_func(freqs)

            # Generate complex noise in the frequency domain
            noise_fd = (np.random.normal(size=len(interpolated_psd_values)) + 1j * np.random.normal(size=len(interpolated_psd_values))) * np.sqrt(interpolated_psd_values / 2)
            noise_data.append(noise_fd)
        np.save(noise_path, noise_data)
    else:
        noise_data = np.load(noise_path, allow_pickle=True)
    return noise_data
def generate_time_domain_noise_from_psd(psd, duration, sampling_frequency):
    """ Generate time-domain noise from a given PSD. """
    N = int(duration * sampling_frequency)
    freqs = np.fft.rfftfreq(N, d=1/sampling_frequency)
    
    # Interpolate the PSD to match the frequency bins
    psd_interp_func = interp1d(psd.frequency_array, psd.psd_array, bounds_error=False, fill_value="extrapolate")
    psd_values = psd_interp_func(freqs)
    
    # Generate complex noise in the frequency domain
    noise_fd = (np.random.normal(size=len(psd_values)) + 1j * np.random.normal(size=len(psd_values))) * np.sqrt(psd_values / 2)
    
    return noise_fd  # Returning frequency domain noise

def refine_noise_in_frequency_domain(noise_fd, sampling_frequency, duration, noise_time_factor):
    """ Refine noise using a moving average in the frequency domain. """
    window_size = int((sampling_frequency * duration) / noise_time_factor) if noise_time_factor != 0 else len(noise_fd)

    # Create a Gaussian window for the moving average
    kernel = gaussian(len(noise_fd), std=window_size)
    refined_amplitude = fftconvolve(np.abs(noise_fd), kernel, mode='same')
    
    # Preserve the phase of the original noise and reconstruct the refined noise
    phase = np.angle(noise_fd)
    refined_noise_fd = refined_amplitude * np.exp(1j * phase)
    
    return refined_noise_fd

def degrade_noise(noise_fd, days, max_noise_days):
    """ Apply a degradation filter to simulate the effect of observation time on noise. """
    noise_time_factor = 1 + (max_noise_days - days) / max_noise_days * (SAMPLING_FREQUENCY*DURATION)
    return refine_noise_in_frequency_domain(noise_fd, SAMPLING_FREQUENCY, DURATION, noise_time_factor)


def simulate_signal(ifos, noise_fd, label, outdir):
    """ Set noise and run the parameter estimation. """
    os.makedirs(outdir, exist_ok=True)
    
    # Setting up the waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=INJECTION_PARAMETERS,
        waveform_arguments=WAVEFORM_ARGUMENTS
    )
    
    priors = bilby.gw.prior.BBHPriorDict(INJECTION_PARAMETERS.copy())
    priors["mass_1"] = bilby.core.prior.Uniform(25, 40, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(25, 40, "mass_2")
    priors["luminosity_distance"] = bilby.core.prior.Uniform(400, 2000, "luminosity_distance")

    # Setting the strain data from noise
    for index, ifo in enumerate(ifos):
        ifo.strain_data.set_from_frequency_domain_strain(noise_fd[index], duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY)

    # Initializing the likelihood
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, 
        waveform_generator=waveform_generator, 
        priors=priors
    )

    # Running the sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        npoints=500,
        outdir=outdir,
        label=label,
        sample="unif"  # Ensures uniform sampling across the parameter space
    )

    return result

def main():
    os.makedirs(BASE_OUTDIR, exist_ok=True)
    ifos = setup_interferometers()
    initial_noise_fds = generate_initial_noise(ifos, BASE_OUTDIR)
    results = {}
    noise_days = [1, 20, 30, 50, 100]
    noise_days = [50, 100]

    max_noise_days = max(noise_days)
    results = {}
    for days in noise_days:
        label = f'noise_day_{days}'
        outdir = os.path.join(BASE_OUTDIR, label)
        os.makedirs(outdir, exist_ok=True)

        degraded_noises = [degrade_noise(noise_fd, days, max_noise_days) for noise_fd in initial_noise_fds]
        
        labels=[str(max_noise_days), label]

        plot_noise_comparison_subplots(outdir = outdir, initial_noise_fds = initial_noise_fds, degraded_noises_list = degraded_noises, sampling_frequency = SAMPLING_FREQUENCY, duration=DURATION, labels = labels )
        # result = simulate_signal(ifos, degraded_noises, label, outdir)
        # results[days] = result
        # tru = {'mass_1': 36.0, 'mass_2': 29.0, 'luminosity_distance': 1000.0}
        # result.plot_corner(truths=tru, filename=f'{outdir}/{label}_corner.png')
        # plt.close()
    return results

if __name__ == "__main__":
    results = main()
    a =1
