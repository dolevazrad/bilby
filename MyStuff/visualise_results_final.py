import bilby
import matplotlib.pyplot as plt
duration = 4
sampling_frequency = 2048
label = "visualise_results_final_O3"
outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/{label}'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Start time for GW190403_051519
start_time = 1238166919#1238166919  # GPS time for 2019-04-03 05:15:19 UTC

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
# specify waveform arguments
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXP",  # waveform approximant name
    reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
)

# set up the waveform generator
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=duration,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=injection_parameters,
    waveform_arguments=waveform_arguments,
)
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=injection_parameters["geocent_time"] - 2,
)
_ = ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)
# first, set up all priors to be equal to a delta function at their designated value
priors = bilby.gw.prior.BBHPriorDict(injection_parameters.copy())
priors["mass_1"] = bilby.core.prior.Uniform(70, 100, "mass_1")
priors["mass_2"] = bilby.core.prior.Uniform(70, 100, "mass_2")
priors["luminosity_distance"] = bilby.core.prior.Uniform(
    7000, 9500, "luminosity_distance"
)
# compute the likelihoods
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
result.plot_corner()
plt.show()
plt.close()