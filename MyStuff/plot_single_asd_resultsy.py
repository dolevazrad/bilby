import bilby
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
import re
from scipy import stats

def plot_single_asd_results_bilby(base_dir, asd_selector=None, outdir=None):
    """
    Plot all parameters for a single ASD analysis using bilby's corner plot.
    
    Args:
        base_dir (str): Base directory containing result files
        asd_selector: Can be either:
            - int: The ASD number to plot
            - str: Part of the filename or full filename
            - None: Will list available ASDs
        outdir (str): Output directory. If None, uses base_dir
    """
    # Set up output directory
    if outdir is None:
        outdir = os.path.join(base_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    
    # Get all result files
    result_files = sorted(glob(os.path.join(base_dir, '*result.json')))
    if not result_files:
        print(f"No result files found in {base_dir}")
        return
        
    # List available ASDs if no selector provided
    if asd_selector is None:
        print("\nAvailable ASD files:")
        for i, f in enumerate(result_files, 1):
            print(f"{i}. {os.path.basename(f)}")
        return
    
    # Select the appropriate file
    selected_file = None
    if isinstance(asd_selector, int):
        # Try to find file by ASD number
        for file in result_files:
            match = re.search(r'asd_(\d+\.?\d*)', file)
            if match:
                if int(float(match.group(1))) == asd_selector:
                    selected_file = file
                    break
    elif isinstance(asd_selector, str):
        matching_files = [f for f in result_files if asd_selector in f]
        if matching_files:
            selected_file = matching_files[0]
    
    if not selected_file:
        print(f"No matching ASD file found for selector: {asd_selector}")
        return
    
    # Load the result
    try:
        result = bilby.result.read_in_result(selected_file)
    except Exception as e:
        print(f"Error reading {selected_file}: {e}")
        return
    
    # Extract ASD value from filename
    asd_value = re.search(r'asd_(\d+\.?\d*)', selected_file)
    asd_value = float(asd_value.group(1)) if asd_value else "Unknown"
    
    try:
        fig = result.plot_corner(
            parameters=['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn'],
            title=f'ASD {asd_value} Analysis',
            save=False,
            show=False,
            truth=True,
            quantiles=[0.16, 0.84],  # 1-sigma (68%) confidence interval
            hist_kwargs={'density': True},
            label_kwargs={'fontsize': 12},
            title_kwargs={'fontsize': 14, 'y': 1.02},
            truth_color='red',
            maximum_posterior=True,
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            figsize=(10, 10)
        )
        
        # Save the plot
        outfile = os.path.join(outdir, f'asd_{asd_value}_corner.png')
        fig.savefig(outfile, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {outfile}")
        
    except Exception as e:
        print(f"Error creating corner plot: {e}")
        import traceback
        traceback.print_exc()

def plot_single_asd_results(base_dir, asd_selector=None, outdir=None):
    """
    Plot all parameters for a single ASD analysis.
    
    Args:
        base_dir (str): Base directory containing result files
        asd_selector: Can be either:
            - int: The ASD number to plot
            - str: Part of the filename or full filename
            - None: Will list available ASDs
        outdir (str): Output directory. If None, uses base_dir
    """
    # Set up output directory
    if outdir is None:
        outdir = os.path.join(base_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    
    # Get all result files
    result_files = sorted(glob(os.path.join(base_dir, '*result.json')))
    if not result_files:
        print(f"No result files found in {base_dir}")
        return
        
    # List available ASDs if no selector provided
    if asd_selector is None:
        print("\nAvailable ASD files:")
        for i, f in enumerate(result_files, 1):
            print(f"{i}. {os.path.basename(f)}")
        return
    
    # Select the appropriate file
    selected_file = None
    if isinstance(asd_selector, int):
        # Try to find file by ASD number
        for file in result_files:
            # Extract ASD value from filename
            match = re.search(r'asd_(\d+\.?\d*)', file)
            if match:
                if int(float(match.group(1))) == asd_selector:
                    selected_file = file
                    break
    elif isinstance(asd_selector, str):
        # Try to find file by name match
        matching_files = [f for f in result_files if asd_selector in f]
        if matching_files:
            selected_file = matching_files[0]
    
    if not selected_file:
        print(f"No matching ASD file found for selector: {asd_selector}")
        return
    
    # Load the result
    try:
        result = bilby.result.read_in_result(selected_file)
    except Exception as e:
        print(f"Error reading {selected_file}: {e}")
        return
    
    # Extract ASD value from filename
    asd_value = re.search(r'asd_(\d+\.?\d*)', selected_file)
    asd_value = float(asd_value.group(1)) if asd_value else "Unknown"
    
    # Set up the plot
    fig = plt.figure(figsize=(15, 10))
    
    # Parameters to plot with their labels and ranges
    params = {
        'chirp_mass': {
            'label': 'Chirp Mass ($M_\\odot$)',
            'range': [20, 23]
        },
        'mass_ratio': {
            'label': 'Mass Ratio',
            'range': [0.8, 1.0]
        },
        'luminosity_distance': {
            'label': 'Luminosity Distance (Mpc)',
            'range': [300, 500]
        },
        'theta_jn': {
            'label': 'Inclination (rad)',
            'range': [0, np.pi]
        }
    }
    
    # Plot each parameter
    for i, (param, info) in enumerate(params.items(), 1):
        ax = fig.add_subplot(2, 2, i)
        
        # Get posterior samples
        samples = result.posterior[param].values
        
        # Plot histogram
        ax.hist(samples, bins=50, density=True, alpha=0.6, color='blue')
        
        # Plot true value if available
        if 'injection_parameters' in result.__dict__:
            true_val = result.injection_parameters.get(param)
            if true_val is not None:
                ax.axvline(true_val, color='r', linestyle='--', 
                          label=f'True: {true_val:.2f}')
                
        # Calculate and plot median and 90% credible interval
        median = np.median(samples)
        lower, upper = np.percentile(samples, [5, 95])
        ax.axvline(median, color='k', linestyle='-', 
                  label=f'Median: {median:.2f}')
        ax.axvspan(lower, upper, alpha=0.2, color='gray',
                  label=f'90% CI: [{lower:.2f}, {upper:.2f}]')
        
        # Customize plot
        ax.set_xlabel(info['label'])
        ax.set_ylabel('Probability Density')
        ax.set_xlim(info['range'])
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Parameter Estimation Results for ASD {asd_value}', 
                fontsize=14, y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    outfile = os.path.join(outdir, f'asd_{asd_value}_analysis.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved as: {outfile}")
def plot_multiple_windows_corner(base_dir, window_selectors, outdir=None):
    """
    Create a clearer corner plot comparing PE results from different time windows.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import corner
    import numpy as np
    
    if outdir is None:
        outdir = os.path.join(base_dir, 'window_plots')
    os.makedirs(outdir, exist_ok=True)
    
    # Parameters to plot with better labels
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn']
    labels = {
        'chirp_mass': 'Chirp Mass $\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': 'Mass Ratio $q$',
        'luminosity_distance': 'Luminosity Distance $d_L$ (Mpc)',
        'theta_jn': 'Inclination $\\theta_{JN}$ (rad)'
    }
    
    # Load results
    results = {}
    for window in window_selectors:
        selected_file = None
        pattern = f'bbh_comparison_win{window}_result.json'
        potential_files = glob(os.path.join(base_dir, pattern))
        
        if potential_files:
            try:
                result = bilby.result.read_in_result(potential_files[0])
                window_hours = window/3600
                results[window_hours] = result
            except Exception as e:
                print(f"Error reading file: {e}")
    
    if not results:
        print("No results loaded")
        return
        
    # Create figure with larger size
    fig = plt.figure(figsize=(20, 20))
    
    # Prepare data
    samples_dict = {}
    for window_hours, result in results.items():
        samples = []
        for param in params:
            samples.append(result.posterior[param].values)
        samples_dict[f'{window_hours:.1f}h'] = np.array(samples).T
    
    # Use a better color scheme with alpha for transparency
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']  # More distinct colors
    
    # Plot settings for better readability
    corner_kwargs = dict(
        show_titles=True,
        title_kwargs={"fontsize": 14, "pad": 10},
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.39, 0.86),
        bins=50,
        smooth=1.0,
        label_kwargs={"fontsize": 14},
        hist_kwargs={
            "density": True,
            "linewidth": 2,
            "alpha": 0.6  # Add transparency
        },
        contour_kwargs={
            "linewidths": 2,
            "alpha": 0.8  # Add transparency
        },
    )
    
    # Create base corner plot without data first
    figure = corner.corner(
        samples_dict[list(samples_dict.keys())[0]],
        labels=[labels[param] for param in params],
        fig=fig,
        **corner_kwargs,
    )
    
    # Get the axes
    axes = np.array(figure.axes).reshape((len(params), len(params)))
    
    # Plot each dataset
    for idx, (label, samples) in enumerate(samples_dict.items()):
        color = colors[idx % len(colors)]
        
        # Plot diagonal (histograms)
        for i in range(len(params)):
            ax = axes[i, i]
            # Plot histograms with transparency
            ax.hist(samples[:, i], bins=50, density=True, color=color, 
                   alpha=0.6, histtype='step', linewidth=2,
                   label=f"Window: {label}")
        
        # Plot contours
        for i in range(len(params)):
            for j in range(i):
                ax = axes[i, j]
                corner.hist2d(samples[:, j], samples[:, i], 
                            ax=ax, color=color, 
                            contour_kwargs={'alpha': 0.8},
                            plot_datapoints=False,
                            levels=(0.39, 0.86),
                            fill_contours=True,
                            smooth=1.0)
    
    # Add legend to the top-left plot
    axes[0, 0].legend(fontsize=12, frameon=True, 
                     facecolor='white', edgecolor='black',
                     bbox_to_anchor=(0.95, 0.95))
    
    # Add true values if available
    if 'injection_parameters' in results[list(results.keys())[0]].__dict__:
        true_params = results[list(results.keys())[0]].injection_parameters
        for i, param in enumerate(params):
            if param in true_params:
                true_val = true_params[param]
                for ax in figure.axes:
                    if ax.get_xlabel() == labels[param]:
                        ax.axvline(true_val, color='black', linestyle='--', 
                                 linewidth=2, label='True Value')
                    if ax.get_ylabel() == labels[param]:
                        ax.axhline(true_val, color='black', linestyle='--', 
                                 linewidth=2)
    
    # Add overall title
    plt.suptitle('Parameter Estimation Results Comparison Across Time Windows',
                y=1.02, fontsize=16, fontweight='bold')
    
    # Save with high DPI
    outfile = os.path.join(outdir, 'window_comparison_corner.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Enhanced corner plot saved to {outfile}")

def plot_multiple_windows_separate(base_dir, window_selectors, outdir=None):
    """
    Create separate plots for each parameter, comparing multiple windows.
    """
    if outdir is None:
        outdir = os.path.join(base_dir, 'window_plots_separate')
    os.makedirs(outdir, exist_ok=True)
    
    params = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # Load results
    results = {}
    for window in window_selectors:
        selected_file = None
        for file in glob(os.path.join(base_dir, '*result.json')):  # Use glob directly
            match = re.search(r'win(\d+)', file)
            if match and (isinstance(window, str) and window in file or 
                         isinstance(window, (int, float)) and 
                         int(float(match.group(1))) == int(window)):
                selected_file = file
                break
                
        if selected_file:
            try:
                result = bilby.result.read_in_result(selected_file)
                window_size = re.search(r'win(\d+)', selected_file).group(1)
                window_hours = float(window_size)/3600
                results[window_hours] = result
            except Exception as e:
                print(f"Error reading {selected_file}: {e}")
    
    if not results:
        print("No results loaded")
        return
    
    # Create separate plots for each parameter
    for param, label in params.items():
        plt.figure(figsize=(12, 8))
        
        for window_hours, result in results.items():
            samples = result.posterior[param].values
            plt.hist(samples, bins=50, density=True, alpha=0.6, 
                    label=f'{window_hours:.1f}h window')
            
            # Add true value if available
            if 'injection_parameters' in result.__dict__:
                true_val = result.injection_parameters.get(param)
                if true_val is not None and not plt.gca().lines:
                    plt.axvline(true_val, color='red', linestyle='--', 
                              label='True Value')
        
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Parameter Estimation Results: {label}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        outfile = os.path.join(outdir, f'comparison_{param}.png')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {param} comparison plot")

def plot_multiple_windows_contours(base_dir, window_selectors, outdir=None):
    """
    Create separate contour plots for each parameter, comparing multiple windows.
    """
    if outdir is None:
        outdir = os.path.join(base_dir, 'window_plots_contours')
    os.makedirs(outdir, exist_ok=True)
    
    params = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # Load results
    results = {}
    for window in window_selectors:
        selected_file = None
        for file in glob(os.path.join(base_dir, '*result.json')):  # Use glob directly
            match = re.search(r'win(\d+)', file)
            if match and (isinstance(window, str) and window in file or 
                         isinstance(window, (int, float)) and 
                         int(float(match.group(1))) == int(window)):
                selected_file = file
                break
                
        if selected_file:
            try:
                result = bilby.result.read_in_result(selected_file)
                window_size = re.search(r'win(\d+)', selected_file).group(1)
                window_hours = float(window_size)/3600
                results[window_hours] = result
            except Exception as e:
                print(f"Error reading {selected_file}: {e}")
    
    if not results:
        print("No results loaded")
        return
        
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for param, label in params.items():
        plt.figure(figsize=(10, 8))
        
        for (window_hours, result), color in zip(results.items(), colors):
            samples = result.posterior[param].values
            
            # Create KDE
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(min(samples), max(samples), 1000)
            density = kde(x_range)
            
            # Plot filled contours
            plt.fill_between(x_range, density, alpha=0.3, color=color)
            plt.plot(x_range, density, color=color, 
                    label=f'{window_hours:.1f}h window')
            
            # Add true value if available
            if 'injection_parameters' in result.__dict__:
                true_val = result.injection_parameters.get(param)
                if true_val is not None and not plt.gca().lines:
                    plt.axvline(true_val, color='red', linestyle='--', 
                              label='True Value')
        
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Parameter Estimation Results: {label}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        outfile = os.path.join(outdir, f'contour_{param}.png')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {param} contour plot")
if __name__ == "__main__":
    base_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/pe_results'
    
    # Convert window sizes from seconds to hours for readability
    window_list = [
        600,     
        #3600,    
        86400,   
        #604800,
        2592000,
        5184000,
        
    ]
    
    print("Creating corner plots...")
    plot_multiple_windows_corner(base_dir, window_list)
    
    print("\nCreating separate parameter plots...")
    plot_multiple_windows_separate(base_dir, window_list)
    
    print("\nCreating contour plots...")
    plot_multiple_windows_contours(base_dir, window_list)
    
    # List available ASDs
    plot_single_asd_results(base_dir)
    
    # Plot by ASD number
    plot_single_asd_results(base_dir, asd_selector=1199)
    plot_single_asd_results_bilby(base_dir, asd_selector=1199)
    
    plot_single_asd_results(base_dir, asd_selector=173399)
    plot_single_asd_results_bilby(base_dir, asd_selector=173399)
    
    plot_single_asd_results(base_dir, asd_selector=2399)
    plot_single_asd_results_bilby(base_dir, asd_selector=2399) 
    
    plot_single_asd_results(base_dir, asd_selector='414')
    plot_single_asd_results_bilby(base_dir, asd_selector='414')
    
    plot_single_asd_results(base_dir, asd_selector=4199)
    plot_single_asd_results_bilby(base_dir, asd_selector=4199) 
    
    plot_single_asd_results(base_dir, asd_selector='605')
    plot_single_asd_results_bilby(base_dir, asd_selector='605')
    
    plot_single_asd_results(base_dir, asd_selector=86999)
    plot_single_asd_results_bilby(base_dir, asd_selector=86999)

    

