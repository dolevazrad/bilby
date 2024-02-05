# numpy
import numpy as np

# pandas
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# sampler
import bilby
from bilby.core.prior import Uniform,Constraint

# misc
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
label = 'GW150914_show_duplicate_update_sampler'      # identifier to apply to output files
outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{label}'


trigger_time = 1126259462.4    # event GPS time at which first GW discovered


roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration




psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1"]:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration,
        overlap=0,
        window=("tukey", psd_alpha),
        method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)


prior = bilby.core.prior.PriorDict()
# uniform priors for variable parameters
prior['chirp_mass'] = Uniform(name='chirp_mass', minimum=30.0,maximum=100.0)
prior['mass_ratio'] = Uniform(name='mass_ratio', minimum=0.5, maximum=1)
prior['phase'] = Uniform(name="phase", minimum=0, maximum=2*np.pi)
prior['geocent_time'] = Uniform(name="geocent_time", minimum=start_time-0.1,
                                maximum=start_time+0.1)
# prior['mass_1'] = Constraint(name='mass_1', minimum=10, maximum=80)
# prior['mass_2'] = Constraint(name='mass_2', minimum=10, maximum=80)
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

waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                        'reference_frequency': 50})

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=prior, time_marginalization=True,
    phase_marginalization=True, distance_marginalization=False)
result = bilby.run_sampler(
    likelihood, prior, sampler='dynesty', outdir=outdir, label=label,
    nlive=1000, walks=100, n_check_point=10000, check_point_plot=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)
result.plot_corner()
result.plot_corner(parameters=['mass_ratio', 'chirp_mass', 'a_1', 'a_2'], filename='{}/subset.png'.format(outdir))
