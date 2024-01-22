import json
import numpy as np
import bilby




def Create_result_text():
    # Define the outdir and file paths
    outdir = '/home/useradd/projects/bilby/MyStuff/my_outdir/GW150914'
    results_file_path = f'{outdir}/GW150914_result.json'
    summary_file_path = f'{outdir}/results_text.txt'

    # Read the JSON data
    with open(results_file_path, 'r') as file:
        results = json.load(file)

    # Initialize an empty string to store results
    summary_text = ""

    # List of parameters to summarize
    param_names = ['mass_ratio', 'chirp_mass', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
                'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase',
                'geocent_time']

    # Loop through each parameter and calculate statistics
    for param in param_names:
        data = results['posterior']['content'][param]
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        conf_int_val = np.percentile(data, [2.5, 97.5])
        # Append the results to the summary text
        summary_text += f"{param}:\nMean: {mean_val}\nMedian: {median_val}\nStandard Deviation: {std_val}\n" \
                        f"95% Confidence Interval: {conf_int_val}\n\n"

    # Save the summary to a text file
    with open(summary_file_path, 'w') as file:
        file.write(summary_text)

    print(f"Summary statistics written to {summary_file_path}")
if __name__ == '__main__':

    # Load the result from the JSON file
    result = bilby.result.read_in_result(filename='/home/useradd/projects/bilby/MyStuff/my_outdir/GW150914_only4/GW150914_only4_result.json')
    
    # Now you can generate a corner plot for specific parameters
    result.plot_corner(parameters=['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn'])
 