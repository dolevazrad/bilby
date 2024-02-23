import numpy as np
import bilby
from generate_noise_from_power_spectral_density import generate_noise_from_power_spectral_density
from bilby.gw.detector import PowerSpectralDensity
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

def calculate_chirp_mass_and_ratio(m1, m2):
    chirp_mass = (m1 * m2) ** (3./5.) / (m1 + m2) ** (1./5.)
    mass_ratio = m2 / m1
    return chirp_mass, mass_ratio
def calculate_m1_m2_from_chirp_mass_and_ratio(chirp_mass, mass_ratio):
    # Since (m1 * m2) = (chirp_mass * (m1 + m2)^(1/5))^5 / (m1 * m2)^(3/5)
    # and mass_ratio = m2 / m1, we can solve the quadratic equation for m1:
    # m1^2 - (chirp_mass * (1 + mass_ratio)^(1/5))^5 / mass_ratio^(3/5) * m1 + chirp_mass^5 = 0
    A = 1
    B = -(chirp_mass * (1 + mass_ratio) ** (1./5.)) ** 5 / mass_ratio ** (3./5.)
    C = chirp_mass ** 5
    m1 = (-B + (B**2 - 4 * A * C)**0.5) / (2 * A)
    m2 = mass_ratio * m1
    return m1, m2
def generate_noise_curve(ifo, duration, sampling_frequency, noise_time):
    """
    Generate a noise curve for an interferometer, scaled by an effective noise_time factor.
    
    Parameters:
    ifo -- bilby.gw.detector.Interferometer object
    duration -- Duration of the data (in seconds)
    sampling_frequency -- Sampling frequency (in Hz)
    noise_time -- The time factor to scale the noise
    """
    
    # Generate the noise timeseries for the interferometer
    noise_data = ifo.get_detector_noise(duration=duration, sampling_frequency=sampling_frequency)

    # For simulating a more accurate noise curve, we can generate more noise and then take a mean
    # Essentially simulating longer observation time
    noise_data_long = np.tile(noise_data, int(noise_time))
    noise_psd_long = np.abs(np.fft.rfft(noise_data_long))**2
    noise_psd_long /= noise_time  # Scaling by the noise_time to normalize
    
    # The new PSD will be the averaged one
    new_psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=ifo.power_spectral_density.frequency_array,
                                                     psd_array=noise_psd_long[:len(ifo.power_spectral_density.frequency_array)])
    
    # Set the new PSD to the interferometer
    ifo.power_spectral_density = new_psd

    return ifo
def scale_psd(ifo, scale_factor):
    """
    Scales the PSD of an interferometer to simulate different noise levels.
    
    Parameters:
    ifo -- bilby.gw.detector.Interferometer object
    scale_factor -- Factor by which to scale the PSD values.
    """
    # Access the existing PSD of the interferometer
    existing_psd = ifo.power_spectral_density.psd_array
    
    # Scale the PSD by the given factor
    new_psd_array = existing_psd * scale_factor
    
    # Update the interferometer's PSD
    ifo.power_spectral_density.psd_array = new_psd_array
def apply_moving_average(data, window_size):
    """
    Apply a moving average to a data series.

    Parameters
    ----------
    data : array_like
        Input data series.
    window_size : int
        The size of the moving average window.

    Returns
    -------
    array_like
        The smoothed data series.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
def refine_and_set_noise(ifos, duration, sampling_frequency, noise_refinement_factor, window_size_factor):
    """
    Generate, refine, and set noise for a list of interferometers based on a given PSD and noise refinement factor.

    Parameters
    ----------
    ifos : list of bilby.gw.detector.Interferometer objects
        The list of interferometer objects for which to generate and set noise.
    duration : float
        Duration of the noise realization (in seconds).
    sampling_frequency : float
        Sampling frequency of the noise realization (in Hz).
    noise_refinement_factor : float
        Factor by which to extend the duration for noise generation to refine the noise estimate.
    window_size_factor : float
        Factor to determine the window size of the moving average based on the sampling frequency.
    """
    extended_duration = duration * noise_refinement_factor
    window_size = int(sampling_frequency / window_size_factor)

    for ifo in ifos:
        # Generate extended duration noise from PSD
        extended_noise = generate_noise_from_power_spectral_density(ifo.power_spectral_density, extended_duration, sampling_frequency)
        
        # Apply moving average to refine the noise
        refined_noise = apply_moving_average(extended_noise, window_size)
        
        # Create a new PSD based on the refined noise
        refined_noise_ft = np.fft.rfft(refined_noise)
        refined_psd_values = np.abs(refined_noise_ft)**2 / (sampling_frequency * extended_duration)
        refined_freqs = np.fft.rfftfreq(len(refined_noise), 1/sampling_frequency)
        
        # Update the interferometer's PSD with the refined PSD
        ifo.power_spectral_density = PowerSpectralDensity(frequency_array=refined_freqs, psd_array=refined_psd_values)

        # Set the strain data directly from the refined noise
        ifo.set_strain_data_from_frequency_domain_strain(refined_noise_ft, sampling_frequency=sampling_frequency)
def generate_refined_noise(ifo, duration, sampling_frequency, noise_refinement_time):
    """
    Generates refined noise for an interferometer based on an extended duration and applies a moving average.
    """
    # Calculate the extended duration based on the refinement time
    extended_duration = duration * noise_refinement_time

    # Generate the extended noise realization
    noise_series = ifo.power_spectral_density.sample_noise(extended_duration, sampling_frequency)

    # Apply a moving average to the extended noise to refine it
    window_size = int(sampling_frequency * duration / noise_refinement_time)  # Window size for the moving average
    refined_noise = convolve(noise_series, np.ones(window_size)/window_size, mode='same', method='auto')

    # Trim the refined noise to match the original duration
    start_index = len(refined_noise) // 2 - int(sampling_frequency * duration / 2)
    end_index = start_index + int(sampling_frequency * duration)
    trimmed_refined_noise = refined_noise[start_index:end_index]

    return trimmed_refined_noise
def generate_time_domain_noise_from_psd(psd, duration, sampling_frequency):
    N = int(duration * sampling_frequency)
    freqs = np.fft.rfftfreq(N, d=1/sampling_frequency)
    
    # Interpolate PSD to match the frequency bins
    psd_interp_func = interp1d(psd.frequency_array, psd.psd_array, bounds_error=False, fill_value="extrapolate")
    psd_values = psd_interp_func(freqs)
    
    # Generate complex noise in the frequency domain
    noise_fd = (np.random.normal(size=len(psd_values)) + 1j * np.random.normal(size=len(psd_values))) * np.sqrt(psd_values / 2)
    
    # Inverse FFT to obtain time-domain noise
    noise_td = np.fft.irfft(noise_fd, n=N)
    
    return noise_td

def refine_noise_with_moving_average(noise, window_size):
    kernel = np.ones(window_size) / window_size
    refined_noise = fftconvolve(noise, kernel, mode='same')
    return refined_noise

import numpy as np
from gwpy.frequencyseries import FrequencySeries

def apply_refined_noise(ifos, noise_time_factor, sampling_frequency, duration, outdir, label):
    window_size = int(sampling_frequency / noise_time_factor)
    
    for ifo in ifos:
        # Generate and refine noise in the time domain
        noise_td = generate_time_domain_noise_from_psd(ifo.power_spectral_density, duration, sampling_frequency)
        refined_noise_td = refine_noise_with_moving_average(noise_td, window_size)
        
        # Convert the original and refined noise back to the frequency domain
        original_noise_fd = np.fft.rfft(noise_td)
        refined_noise_fd = np.fft.rfft(refined_noise_td)
        freqs = np.fft.rfftfreq(len(noise_td), d=1.0/sampling_frequency)

        # Plot and save the noise graph
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs, np.abs(original_noise_fd), label='Original Noise', alpha=0.7)
        plt.loglog(freqs, np.abs(refined_noise_fd), label='Refined Noise', linestyle='--')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Strain amplitude')
        plt.title(f'Noise Spectrum Comparison for {ifo.name}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"{outdir}/{ifo.name}_{label}_noise_comparison.png")
        plt.close()

        # Create a FrequencySeries object for the refined noise
        refined_noise_series = FrequencySeries(refined_noise_fd, frequencies=freqs, df=freqs[1] - freqs[0])

        # Manually update the strain data of the interferometer
        ifo.strain_data.set_from_frequency_domain_strain(refined_noise_series.value, sampling_frequency=sampling_frequency, duration=duration)
