#  This code should estimate paramters with diffrent noise levels
# At first this code will work only for 4 paramters
import bilby
import numpy as np
from HelperFunction import apply_refined_noise
import matplotlib.pyplot as plt
from bilby.gw.detector import InterferometerList, PowerSpectralDensity

# Define the sampling frequency and duration of the data segment
sampling_frequency = 2048
duration = 4

# Define the waveform generator arguments
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXP",
    reference_frequency=50.0,
)

# Define the injection parameters for the signal
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    phase=1.3,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659,
)



def run_with_noise_time(noise_time, label='BaseLine'):
    outdir = f'/home/useradd/projects/bilby/MyStuff/my_outdir/Noise_Levels/{label}'
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - 2,
    )

    # Apply refined noise directly to the existing interferometer objects
    apply_refined_noise(ifos, noise_time_factor=noise_time, sampling_frequency=sampling_frequency, duration=duration,outdir = outdir, label=label)

    # Ensure the maximum frequency is correctly set for all interferometers
    for ifo in ifos:
        ifo.maximum_frequency = 1024.0

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=4.0,
        sampling_frequency=2048.0,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments={'waveform_approximant': 'IMRPhenomXP', 'reference_frequency': 50.0}
    )
        
    # Inject the signal into the interferometers
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    # Set up priors for the search
    priors = bilby.gw.prior.BBHPriorDict(injection_parameters.copy())
    priors["mass_1"] = bilby.core.prior.Uniform(25, 40, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(25, 40, "mass_2")
    priors["luminosity_distance"] = bilby.core.prior.Uniform(400, 2000, "luminosity_distance")

    # Set up the likelihood for parameter estimation
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator, priors=priors
    )

    # Run the sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        npoints=100,
        outdir=outdir,
        label=label,
    )

    # Plot and save the corner plot
  
    result.plot_corner(truths=injection_parameters,filename='{}/_corner.png'.format(outdir))
    plt.savefig(f"{outdir}/{label}_corner.png")
    plt.close()

if __name__ == "__main__":
    run_with_noise_time(noise_time=1.0, label='noise_time_1')
    run_with_noise_time(noise_time=10, label='noise_time_10')