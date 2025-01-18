import bilby
import numpy as np
import pickle
import os
from bilby.gw.detector import PowerSpectralDensity, InterferometerList
from bilby.core.prior import Uniform, Cosine, Sine, PowerLaw
import glob
from scipy.interpolate import interp1d
import re
import traceback  
import corner
import matplotlib.pyplot as plt 
# Set up the output directory
BASE_DIR = r'/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window'
outdir = os.path.join(BASE_DIR, 'pe_results')
label = 'bbh_comparison'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Fixed parameters for the injection
injection_parameters = dict(
    chirp_mass=21.75,
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
    geocent_time=1238303719  # GW190403_051519
)

# Duration and sampling parameters
duration = 4  # seconds
sampling_frequency = 2048
minimum_frequency = 20

# Adjust start_time to ensure merger is within segment
geocent_time = injection_parameters['geocent_time']
start_time = geocent_time - duration + 0.5

# Load both H1 and L1 window files
h1_files = sorted(glob.glob(os.path.join(BASE_DIR, 'H1_asd_win*.pkl')))
l1_files = sorted(glob.glob(os.path.join(BASE_DIR, 'L1_asd_win*.pkl')))

if not h1_files:
    raise FileNotFoundError(f"No H1 window files found in {BASE_DIR}")
if not l1_files:
    raise FileNotFoundError(f"No L1 window files found in {BASE_DIR}")

print(f"Found {len(h1_files)} H1 files and {len(l1_files)} L1 files")

# Match H1 and L1 files by window size
matched_files = []
for h1_file in h1_files:
    h1_window = re.search(r'win(\d+)', h1_file).group(1)
    l1_match = next((f for f in l1_files if f'win{h1_window}' in f), None)
    if l1_match:
        matched_files.append((h1_file, l1_match))

print(f"Found {len(matched_files)} matching H1-L1 pairs")

# Dynesty settings
dynesty_settings = {
    'npoints': 500,
    'walks': 25,
    'dlogz': 0.1,
    'sample': 'rwalk',
    'bound': 'multi',
    'checkpoint_file': None
}

results = []

for h1_file, l1_file in matched_files:
    print(f"\nProcessing window pair:")
    print(f"H1: {os.path.basename(h1_file)}")
    print(f"L1: {os.path.basename(l1_file)}")
    
    try:
        # Load both ASDs
        with open(h1_file, 'rb') as f:
            h1_data = pickle.load(f)
        with open(l1_file, 'rb') as f:
            l1_data = pickle.load(f)

        # Create interferometer objects
        interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])

        # Calculate frequency array for PSD
        n_freq = int(duration * sampling_frequency / 2) + 1
        frequencies = np.linspace(0, sampling_frequency/2, n_freq)

        # Set up each interferometer
        for ifo in interferometers:
            # Select correct ASD data
            current_data = h1_data if ifo.name == 'H1' else l1_data
            
            # Interpolate ASD
            asd_frequencies = current_data['asd'].frequencies.value
            asd_values = current_data['asd'].value
            asd_interpolator = interp1d(asd_frequencies, asd_values,
                                      bounds_error=False, fill_value='extrapolate')
            interpolated_asd = asd_interpolator(frequencies)

            # Set interferometer properties
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

        # Set up waveform generator
        waveform_arguments = dict(
            waveform_approximant='IMRPhenomD',
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

        # Set up priors
        priors = bilby.gw.prior.BBHPriorDict()
        priors['chirp_mass'] = Uniform(20.0, 23.0, latex_label='$\\mathcal{M}$')
        priors['mass_ratio'] = Uniform(0.8, 1.0, latex_label='$q$')
        priors['luminosity_distance'] = Uniform(300, 500, latex_label='$d_L$')
        priors['theta_jn'] = Sine(latex_label='$\\theta_{JN}$')
        priors['phase'] = Uniform(0, 2 * np.pi, latex_label='$\\phi$')
        priors['geocent_time'] = Uniform(
            injection_parameters['geocent_time'] - 0.1,
            injection_parameters['geocent_time'] + 0.1,
            latex_label='$t_c$'
        )

        # Fix other parameters
        for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
            priors[key] = injection_parameters[key]

        # Set up likelihood
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator
        )

        # Extract window size for label
        window_size = re.search(r'win(\d+)', h1_file).group(1)
        
        # Run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            outdir=outdir,
            label=f'{label}_win{window_size}',
            injection_parameters=injection_parameters,
            save=True,
            **dynesty_settings
        )

        results.append(result)
        print(f"Completed analysis for window size {window_size}")

    except Exception as e:
        print(f"Error processing window {window_size}: {str(e)}")
        traceback.print_exc()
        continue

if results:
    print("\nGenerating comparison and corner plots...")
    try:
        # Ensure output directory exists
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print(f"\nSaving plots to: {outdir}")

        # Create labels for the plots
        labels = []
        for result in results:
            window_size = re.search('win(\d+)', result.label).group(1)
            labels.append(f'Window {window_size}')

        # Parameters for corner plots
        parameters = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 
                     'theta_jn', 'phase', 'geocent_time']

        # Individual corner plots using corner package directly
        for i, result in enumerate(results):
            try:
                window_size = re.search('win(\d+)', result.label).group(1)
                corner_plot_path = os.path.join(outdir, f'corner_plot_win{window_size}.png')
                
                # Create figure
                fig = plt.figure(figsize=(15, 15))
                
                # Prepare data
                data = np.zeros((len(result.posterior), len(parameters)))
                for j, param in enumerate(parameters):
                    data[:, j] = result.posterior[param].values
                
                # Create corner plot
                corner.corner(
                    data,
                    labels=parameters,
                    fig=fig,
                    color='blue',
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                    plot_datapoints=True,
                    plot_density=True,
                    levels=(0.39, 0.86, 0.99),
                    bins=30,
                )
                
                plt.suptitle(f'Corner Plot for Window {window_size}', y=1.02, fontsize=14)
                
                # Save and close
                plt.savefig(corner_plot_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                
                # Verify file was created
                if os.path.exists(corner_plot_path):
                    print(f"Successfully saved corner plot for window {window_size}")
                    print(f"File size: {os.path.getsize(corner_plot_path)} bytes")
                else:
                    print(f"Failed to save corner plot for window {window_size}")
                    
            except Exception as e:
                print(f"Error creating corner plot for window {window_size}: {str(e)}")
                traceback.print_exc()

        # Combined corner plot
        if len(results) > 1:
            try:
                combined_plot_path = os.path.join(outdir, 'combined_corner_plot.png')
                
                # Set up the figure
                fig = plt.figure(figsize=(15, 15))
                
                # Plot with different colors
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                
                # Add datasets
                for i, result in enumerate(results):
                    data = np.zeros((len(result.posterior), len(parameters)))
                    for j, param in enumerate(parameters):
                        data[:, j] = result.posterior[param].values
                    
                    if i == 0:
                        fig = corner.corner(
                            data,
                            labels=parameters,
                            fig=fig,
                            color=colors[i],
                            show_titles=True,
                            title_kwargs={"fontsize": 12},
                            plot_datapoints=True,
                            plot_density=True,
                            levels=(0.39, 0.86, 0.99),
                            bins=30
                        )
                    else:
                        corner.corner(
                            data,
                            fig=fig,
                            color=colors[i % len(colors)],
                            plot_datapoints=True,
                            plot_density=True,
                            levels=(0.39, 0.86, 0.99),
                            bins=30
                        )
                
                # Add legend
                axes = np.array(fig.axes)
                legend_elements = [plt.Line2D([0], [0], color=colors[i], label=label)
                                 for i, label in enumerate(labels)]
                axes[0].legend(handles=legend_elements, 
                             loc='upper right',
                             fontsize=8)

                plt.savefig(combined_plot_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                
                if os.path.exists(combined_plot_path):
                    print(f"Successfully saved combined corner plot")
                    print(f"File size: {os.path.getsize(combined_plot_path)} bytes")
                else:
                    print("Failed to save combined corner plot")
                
            except Exception as e:
                print(f"Error creating combined corner plot: {str(e)}")
                traceback.print_exc()

        # List actual files in directory
        print("\nActual files in output directory:")
        for file in os.listdir(outdir):
            if file.endswith('.png'):
                file_path = os.path.join(outdir, file)
                print(f"{file_path} - Size: {os.path.getsize(file_path)} bytes")

    except Exception as e:
        print(f"Error in plotting routine: {str(e)}")
        traceback.print_exc()
else:
    print("No successful analyses to plot.")