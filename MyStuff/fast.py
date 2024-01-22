import numpy as np
import bilby
from gwpy.timeseries import TimeSeries
from bilby.core.prior import Uniform, Constraint



logger = bilby.core.utils.logger
label = 'fast'
outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{label}'
# Set a random seed for result reproducibility
np.random.seed(88170235)

from bilby.core.prior import Uniform, Constraint

# Priors for GW150914, estimating only masses and spins
priors = bilby.gw.prior.BBHPriorDict()

# Estimating parameters
priors['mass_ratio'] = Uniform(name='mass_ratio', minimum=0.125, maximum=1, boundary='reflective')
priors['chirp_mass'] = Uniform(name='chirp_mass', minimum=25, maximum=35, unit='$M_{\odot}$', boundary='reflective')
priors['mass_1'] = Uniform(name='mass_1', minimum=10, maximum=80, unit='$M_{\odot}$')
priors['mass_2'] = Uniform(name='mass_2', minimum=10, maximum=80, unit='$M_{\odot}$')
priors['a_1'] = Uniform(name='a_1', minimum=0, maximum=0.99, boundary='reflective')
priors['a_2'] = Uniform(name='a_2', minimum=0, maximum=0.99, boundary='reflective')

# Fixed parameters to injected values
priors['tilt_1'] = 0.5  # Fixed value from injection
priors['tilt_2'] = 1.0  # Fixed value from injection
priors['phi_12'] = 1.7  # Fixed value from injection
priors['phi_jl'] = 0.3  # Fixed value from injection
priors['luminosity_distance'] = Uniform(name='luminosity_distance', minimum=100, maximum=5000, unit='Mpc')  # Estimating parameter within the range
priors['dec'] = -1.2108  # Fixed value from injection
priors['ra'] = 1.375  # Fixed value from injection
priors['theta_jn'] = 0.4  # Fixed value from injection
priors['psi'] = 2.659  # Fixed value from injection
priors['phase'] = 1.3  # Fixed value from injection
priors['geocent_time'] = 1126259642.413  # Fixed value from injection




# Set the duration and sampling frequency of the data segment
duration = 4.
outdir = 'outdir'
label = 'fast_tutorial'
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# Create the waveform_generator using the same waveform_arguments
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                        'reference_frequency': 50})

# Set up interferometers
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
start_time = 1126259642.413 - 2  # 2 seconds before the trigger time
end_time = start_time + duration
for det in ifos:
    data = TimeSeries.fetch_open_data(det.name, start_time, end_time)
    det.set_strain_data_from_gwpy_timeseries(data)


# Initialise the likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform_generator)

# Run the sampler
result = bilby.run_sampler(
    likelihood, priors, sampler='dynesty', outdir=outdir, label=label,
    nlive=500, walks=50, n_check_point=5000, check_point_plot=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)
# Generate and save a corner plot
result.plot_corner()
