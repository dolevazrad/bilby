# numpy
import numpy as np

# pandas
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# sampler
import bilby
from bilby.core.prior import Uniform

# misc
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
label = 'GW150914_show_duplicate'      # identifier to apply to output files
outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{label}'

H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")
event_start_time = 1126259462.4    # event GPS time at which first GW discovered
post_event = 2                     # time after event
pre_event = 2                      # time before event
duration = pre_event + post_event  # total time range containing event
analysis_begin = event_start_time - pre_event  # GPS time at which to start analysis
H1_data = TimeSeries.fetch_open_data("H1", analysis_begin, 
                                     analysis_begin + duration,
                                     sample_rate=4096, cache=True)

L1_data = TimeSeries.fetch_open_data("L1", analysis_begin, 
                                     analysis_begin + duration,
                                     sample_rate=4096, cache=True)

H1_data.plot()
plt.savefig("H1_data")
L1_data.plot()
plt.savefig("L1_data")
# duration typically multiplied by 32 for psd
psd_duration = duration * 32                
psd_begin = analysis_begin - psd_duration

# fetch data for psd
H1psd = TimeSeries.fetch_open_data(
    "H1", psd_begin, psd_begin + psd_duration,
    sample_rate=4096, cache=True)
L1psd = TimeSeries.fetch_open_data(
    "L1", psd_begin, psd_begin + psd_duration,
    sample_rate=4096, cache=True)

# set interferometers with psd data
psd_alpha = 2 * H1.strain_data.roll_off / duration
H1psd_data = H1psd.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")
L1psd_data = L1psd.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")
H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=H1psd_data.frequencies.value, psd_array=H1psd_data.value)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=L1psd_data.frequencies.value, psd_array=L1psd_data.value)
H1.maximum_frequency = 1024
L1.maximum_frequency = 1024

prior = bilby.core.prior.PriorDict()
# uniform priors for variable parameters
prior['chirp_mass'] = Uniform(name='chirp_mass', minimum=10.0,maximum=100.0)
prior['mass_ratio'] = Uniform(name='mass_ratio', minimum=0.5, maximum=1)
prior['phase'] = Uniform(name="phase", minimum=0, maximum=2*np.pi)
prior['geocent_time'] = Uniform(name="geocent_time", minimum=event_start_time-0.1,
                                maximum=event_start_time+0.1)
# fixed values for all other parameters
prior['a_1'] =  0.0     
prior['a_2'] =  0.0
prior['tilt_1'] =  0.0
prior['tilt_2'] =  0.0
prior['phi_12'] =  0.0
prior['phi_jl'] =  0.0
prior['dec'] =  -1.2232
prior['ra'] =  2.19432
prior['theta_jn'] =  1.89694
prior['psi'] =  0.532268
prior['luminosity_distance'] = 412.066
interferometers = [H1, L1]

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

# set sampling frequency of data
sampling_frequency = 2048.

# generate waveforms
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator)
nlive = 1000          # live points
stop = 0.1            # stopping criterion
method = "unif"       # method of sampling
sampler = "dynesty"   # sampler to use

result = bilby.run_sampler(
    likelihood, prior, sampler=sampler, outdir=outdir, label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    sample=method, nlive=nlive, dlogz=stop) 
result.plot_corner()
