import bilby
import numpy as np
import pickle
import os
from bilby.gw.detector import PowerSpectralDensity, InterferometerList
from bilby.core.prior import Uniform, Cosine, Sine, PowerLaw
import glob
from scipy.interpolate import interp1d

# Set up the output directory
BASE_DIR = r'/home/useradd/projects/bilby/MyStuff/my_outdir/Analyzing_GW_Noise_v3'
outdir = os.path.join(BASE_DIR, 'pe_results')
label = 'bbh_comparison'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Fixed parameters for the injection
injection_parameters = dict(
    chirp_mass=21.75,  # This gives m1=m2=25.0
    mass_ratio=1.0,
    luminosity_distance=400.0,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    theta_jn=0.4,
    phase=0.0,
    ra=1.375,
    dec=-1.2108,
    psi=2.659,
    geocent_time=1238167519
)

# Duration and sampling parameters
duration = 4  # seconds
sampling_frequency = 2048
minimum_frequency = 20

# Adjust start_time to ensure merger is within segment
geocent_time = injection_parameters['geocent_time']
start_time = geocent_time - duration + 0.5

# Load ASD files
asd_files = sorted(glob.glob(os.path.join(BASE_DIR, 'H1_asd_*.pkl')))
if not asd_files:
    raise FileNotFoundError(f"No ASD files found in {BASE_DIR}")

print(f"Found {len(asd_files)} ASD files")
results = []

# Dynesty settings for faster convergence
dynesty_settings = {
    'npoints': 500,  # Reduced from 1000
    'walks': 25,    # Reduced from 100 
    'dlogz': 0.1,
    'sample': 'rwalk',
    'bound': 'multi',
    'checkpoint_file': None  # Disable checkpointing to speed up
}

for asd_file in asd_files:
    print(f"\nProcessing {os.path.basename(asd_file)}")
    
    try:
        # Create interferometer objects
        interferometers = bilby.gw.detector.InterferometerList(['H1'])  # Using only H1 for speed
        
        # Load ASD data
        with open(asd_file, 'rb') as f:
            asd_data = pickle.load(f)
        
        # Calculate frequency array for PSD
        n_freq = int(duration * sampling_frequency / 2) + 1
        frequencies = np.linspace(0, sampling_frequency/2, n_freq)
        
        # Interpolate ASD to match frequency array
        asd_frequencies = asd_data['asd'].frequencies.value
        asd_values = asd_data['asd'].value
        asd_interpolator = interp1d(asd_frequencies, asd_values, 
                                  bounds_error=False, fill_value='extrapolate')
        interpolated_asd = asd_interpolator(frequencies)
        
        # Set up interferometer
        for ifo in interferometers:
            ifo.minimum_frequency = minimum_frequency
            ifo.maximum_frequency = sampling_frequency/2
            ifo.sampling_frequency = sampling_frequency
            ifo.duration = duration
            ifo.start_time = start_time
            
            # Set PSD
            psd_array = interpolated_asd ** 2
            ifo.power_spectral_density = PowerSpectralDensity(
                frequency_array=frequencies,
                psd_array=psd_array
            )
            
            # Initialize strain data
            ifo.strain_data.roll_off = 0.2
            ifo.strain_data.set_from_frequency_domain_strain(
                sampling_frequency=sampling_frequency,
                duration=duration,
                frequency_domain_strain=np.zeros(n_freq, dtype=complex)
            )
        
        # Set up waveform generator with fewer options
        waveform_arguments = dict(
            waveform_approximant='IMRPhenomD',  # Simpler approximant
            reference_frequency=50.0,
            minimum_frequency=minimum_frequency
        )
        
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments
        )
        
        # Inject signal
        for ifo in interferometers:
            ifo.inject_signal(
                parameters=injection_parameters,
                waveform_generator=waveform_generator
            )
            
            print(f"Signal injected successfully in {ifo.name}")
            print(f"Max strain amplitude: {np.max(abs(ifo.strain_data.frequency_domain_strain))}")
        
        # Set up priors - using narrower ranges
        priors = dict()
        priors['chirp_mass'] = Uniform(
            name='chirp_mass',
            minimum=20.0,
            maximum=23.0,  # Narrowed range around true value (21.75)
            latex_label='$\\mathcal{M}$'
        )
        priors['mass_ratio'] = Uniform(
            name='mass_ratio',
            minimum=0.8,
            maximum=1.0,  # Narrowed range
            latex_label='$q$'
        )
        priors['luminosity_distance'] = Uniform(
            name='luminosity_distance',
            minimum=300,
            maximum=500,  # Narrowed range around true value (400)
            latex_label='$d_L$'
        )
        priors['theta_jn'] = Sine(name='theta_jn')
        priors['phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi)
        priors['geocent_time'] = Uniform(
            minimum=injection_parameters['geocent_time'] - 0.1,
            maximum=injection_parameters['geocent_time'] + 0.1,
            name='geocent_time'
        )
        
        # Fix other parameters
        for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
            priors[key] = injection_parameters[key]
        
        # Set up likelihood
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator
        )
        
        # Run the analysis with optimized settings
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            outdir=outdir,
            label=f'{label}_{os.path.basename(asd_file).split(".")[0]}',
            injection_parameters=injection_parameters,
            save=True,
            **dynesty_settings
        )
        
        results.append(result)
        print(f"Completed analysis for {os.path.basename(asd_file)}")
        
    except Exception as e:
        print(f"Error processing {asd_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Plot comparisons
if results:
    print("\nGenerating comparison plots...")
    try:
        bilby.core.result.plot_multiple(
            results,
            parameters=['chirp_mass', 'mass_ratio', 'luminosity_distance'],
            labels=[f'ASD {i+1}' for i in range(len(results))],
            outdir=outdir,
            filename='comparison_plot.png'
        )
        print("Analysis complete!")
    except Exception as e:
        print(f"Error creating comparison plot: {str(e)}")
else:
    print("No successful analyses to plot.")