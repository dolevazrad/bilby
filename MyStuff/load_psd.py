import numpy as np
import matplotlib.pyplot as plt

def analyze_psd_file(file_path):
    # Load the data
    data = np.load(file_path)
    
    print(f"Analyzing file: {file_path}")
    print(f"Shape of data: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Any non-zero values: {np.any(data != 0)}")
    print(f"Min value: {np.nanmin(data)}")
    print(f"Max value: {np.nanmax(data)}")
    print(f"Mean value: {np.nanmean(data)}")
    print(f"Median value: {np.nanmedian(data)}")
    print(f"Number of NaN values: {np.isnan(data).sum()}")
    print(f"Number of infinite values: {np.isinf(data).sum()}")
    
    # Remove NaN and inf values for further analysis
    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    non_zero = valid_data[valid_data != 0]
    
    print(f"Number of valid, non-zero values: {len(non_zero)}")
    
    if len(non_zero) > 0:
        print(f"Min non-zero value: {np.min(non_zero)}")
        print(f"Max non-zero value: {np.max(non_zero)}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(np.log10(non_zero), bins=50)
        plt.title(f"Histogram of log10(non-zero values) for {file_path.split('/')[-1]}")
        plt.xlabel("log10(PSD value)")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No valid non-zero values to plot histogram.")
    
    print("\n")

if __name__ == "__main__":

    # Analyze both files
    files = [
        r"/home/useradd/projects/bilby/MyStuff/my_outdir/Analyzing_GW_Noise/H1_psd_86400.0.npy",
        r"/home/useradd/projects/bilby/MyStuff/my_outdir/Analyzing_GW_Noise/H1_psd_172800.0.npy"
    ]

    for file in files:
        analyze_psd_file(file)