import bilby
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import re

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
def plot_multiple_asds_separate(base_dir, asd_selectors, outdir=None):
    """
    Create separate plots for each parameter, comparing multiple ASDs.
    
    Args:
        base_dir (str): Base directory containing result files
        asd_selectors (list): List of ASD values to compare
        outdir (str): Output directory. If None, uses base_dir/plots
    """
    # Set up output directory
    if outdir is None:
        outdir = os.path.join(base_dir, 'plots_separate')
    os.makedirs(outdir, exist_ok=True)
    
    # Parameters to plot
    params = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # Load results for each ASD
    results = {}
    for asd in asd_selectors:
        selected_file = None
        for file in glob(os.path.join(base_dir, '*result.json')):
            match = re.search(r'asd_(\d+\.?\d*)', file)
            if match and (isinstance(asd, str) and asd in file or 
                         isinstance(asd, (int, float)) and 
                         int(float(match.group(1))) == int(asd)):
                selected_file = file
                break
                
        if selected_file:
            try:
                result = bilby.result.read_in_result(selected_file)
                asd_value = re.search(r'asd_(\d+\.?\d*)', selected_file)
                asd_value = float(asd_value.group(1)) if asd_value else "Unknown"
                results[asd_value] = result
            except Exception as e:
                print(f"Error reading {selected_file}: {e}")
    
    if not results:
        print("No results loaded")
        return
        
    # Create a plot for each parameter
    for param, label in params.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each ASD result
        for asd_value, result in results.items():
            # Get posterior samples
            samples = result.posterior[param].values
            
            # Create KDE plot
            from scipy import stats
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(min(samples), max(samples), 1000)
            plt.plot(x_range, kde(x_range), label=f'ASD {asd_value}', alpha=0.7)
            
            # Add vertical line for injection value if available
            if 'injection_parameters' in result.__dict__:
                true_val = result.injection_parameters.get(param)
                if true_val is not None and not plt.gca().lines:  # Only plot once
                    plt.axvline(true_val, color='red', linestyle='--', 
                              label='True Value', alpha=0.8)
        
        # Customize plot
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Parameter Estimation: {label}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        outfile = os.path.join(outdir, f'comparison_{param}.png')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {param} comparison plot to: {outfile}")
def plot_multiple_asds_separate_contours(base_dir, asd_selectors, outdir=None):
    """
    Create separate contour plots for each parameter, comparing multiple ASDs.
    
    Args:
        base_dir (str): Base directory containing result files
        asd_selectors (list): List of ASD values to compare
        outdir (str): Output directory. If None, uses base_dir/plots
    """
    # Set up output directory
    if outdir is None:
        outdir = os.path.join(base_dir, 'plots_separate')
    os.makedirs(outdir, exist_ok=True)
    
    # Parameters to plot
    params = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # Load results for each ASD
    results = {}
    for asd in asd_selectors:
        selected_file = None
        for file in glob(os.path.join(base_dir, '*result.json')):
            match = re.search(r'asd_(\d+\.?\d*)', file)
            if match and (isinstance(asd, str) and asd in file or 
                         isinstance(asd, (int, float)) and 
                         int(float(match.group(1))) == int(asd)):
                selected_file = file
                break
                
        if selected_file:
            try:
                result = bilby.result.read_in_result(selected_file)
                asd_value = re.search(r'asd_(\d+\.?\d*)', selected_file)
                asd_value = float(asd_value.group(1)) if asd_value else "Unknown"
                results[asd_value] = result
            except Exception as e:
                print(f"Error reading {selected_file}: {e}")
    
    if not results:
        print("No results loaded")
        return
        
    # Create a contour plot for each parameter
    for param, label in params.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color map for different ASDs
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        true_val = None
        for (asd_value, result), color in zip(results.items(), colors):
            # Get posterior samples
            samples = result.posterior[param].values
            
            # Create 2D histogram
            hist, bin_edges = np.histogram(samples, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Smooth the histogram for contours
            from scipy.ndimage import gaussian_filter
            hist_smooth = gaussian_filter(hist, sigma=1.0)
            
            # Calculate levels for 1σ and 2σ contours (68% and 95%)
            sorted_hist = np.sort(hist_smooth)[::-1]
            cumsum = np.cumsum(sorted_hist) / np.sum(sorted_hist)
            level_68 = sorted_hist[np.argmin(np.abs(cumsum - 0.68))]
            level_95 = sorted_hist[np.argmin(np.abs(cumsum - 0.95))]
            
            # Plot contours
            plt.fill_between(bin_centers, hist_smooth, 
                           where=hist_smooth >= level_95,
                           alpha=0.3, color=color)
            plt.fill_between(bin_centers, hist_smooth,
                           where=hist_smooth >= level_68,
                           alpha=0.3, color=color,
                           label=f'ASD {asd_value}')
            
            # Store true value if available
            if 'injection_parameters' in result.__dict__ and true_val is None:
                true_val = result.injection_parameters.get(param)
        
        # Add true value line
        if true_val is not None:
            plt.axvline(true_val, color='red', linestyle='--', 
                       label='True Value', linewidth=2)
        
        # Customize plot
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Parameter Estimation: {label}', fontsize=14)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        outfile = os.path.join(outdir, f'contour_comparison_{param}.png')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {param} contour comparison plot to: {outfile}")
        
def plot_multiple_asds_corner(base_dir, asd_selectors, outdir=None):
    """
    Create corner plots for each parameter with overlapping contours from multiple ASDs.
    
    Args:
        base_dir (str): Base directory containing result files
        asd_selectors (list): List of ASD values to compare
        outdir (str): Output directory. If None, uses base_dir/plots
    """
    import corner
    
    if outdir is None:
        outdir = os.path.join(base_dir, 'plots_separate')
    os.makedirs(outdir, exist_ok=True)
    
    # Parameters to plot
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn']
    labels = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # Define color scheme
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Load results
    results = {}
    for asd in asd_selectors:
        selected_file = None
        for file in glob(os.path.join(base_dir, '*result.json')):
            match = re.search(r'asd_(\d+\.?\d*)', file)
            if match and (isinstance(asd, str) and asd in file or 
                         isinstance(asd, (int, float)) and 
                         int(float(match.group(1))) == int(asd)):
                selected_file = file
                break
                
        if selected_file:
            try:
                result = bilby.result.read_in_result(selected_file)
                asd_value = re.search(r'asd_(\d+\.?\d*)', selected_file)
                asd_value = float(asd_value.group(1)) if asd_value else "Unknown"
                results[asd_value] = result
            except Exception as e:
                print(f"Error reading {selected_file}: {e}")
    
    if not results:
        print("No results loaded")
        return
    
    # Set up the figure
    n_params = len(params)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    # Plot each parameter combination
    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            ax = axes[i, j]
            
            if j > i:  # Upper triangle
                ax.set_visible(False)
                continue
                
            elif i == j:  # Diagonal - 1D histograms
                for k, (asd_value, result) in enumerate(results.items()):
                    samples = result.posterior[param1].values
                    ax.hist(samples, bins=50, density=True, histtype='step',
                           color=colors[k % len(colors)], alpha=0.7, 
                           label=f'ASD {asd_value}')
                    
                    if 'injection_parameters' in result.__dict__:
                        true_val = result.injection_parameters.get(param1)
                        if true_val is not None and not ax.lines:
                            ax.axvline(true_val, color='red', linestyle='--', 
                                     label='True Value')
                
                if i == 0:  # Only show legend on first plot
                    ax.legend(fontsize=8)
                    
            else:  # Lower triangle - 2D contours
                for k, (asd_value, result) in enumerate(results.items()):
                    data = np.vstack([
                        result.posterior[param2].values,
                        result.posterior[param1].values
                    ]).T
                    
                    try:
                        # Create KDE
                        from scipy import stats
                        kde = stats.gaussian_kde(data.T)
                        
                        # Create grid
                        xmin, xmax = np.percentile(data[:, 0], [1, 99])
                        ymin, ymax = np.percentile(data[:, 1], [1, 99])
                        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        
                        # Evaluate KDE
                        z = np.reshape(kde(positions).T, xx.shape)
                        
                        # Plot contours
                        levels = [stats.scoreatpercentile(z.ravel(), 100-p) 
                                for p in [95, 68]]
                        ax.contour(xx, yy, z, levels=levels, colors=[colors[k % len(colors)]],
                                 alpha=0.7, linewidths=1)
                        
                        # Plot injection point if available
                        if 'injection_parameters' in result.__dict__:
                            true_x = result.injection_parameters.get(param2)
                            true_y = result.injection_parameters.get(param1)
                            if true_x is not None and true_y is not None and not ax.lines:
                                ax.plot(true_x, true_y, 'r*', markersize=10)
                                
                    except Exception as e:
                        print(f"Error creating contour for ASD {asd_value}: {e}")
                        continue
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(labels[param2])
            if j == 0:
                ax.set_ylabel(labels[param1])
    
    plt.suptitle('Parameter Estimation Comparison Across ASDs', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save plot
    outfile = os.path.join(outdir, 'asd_comparison_corner.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved corner plot comparison to: {outfile}")
        
if __name__ == "__main__":
    base_dir = '/home/useradd/projects/bilby/MyStuff/my_outdir/Analyzing_GW_Noise_v3/pe_results'
    
    
    asd_list = [1199, 86999, 605399, 4147798]
    plot_multiple_asds_corner(base_dir, asd_list)
    plot_multiple_asds_separate(base_dir, asd_list)
    plot_multiple_asds_separate_contours(base_dir, asd_list)
    
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

    

