import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the function for formatting sizes
def format_size(size):
    if size < 2**10:
        return f"{size} B"
    elif size < 2**20:
        return f"{size / 2**10:.1f} KiB"
    elif size < 2**30:
        return f"{size / 2**20:.1f} MiB"
    else:
        return f"{size / 2**30:.1f} GiB"

# List of different settings for "nreduce"
nreduce_settings = ["nreduce_12", "nreduce_12_even", "nreduce_48"]

data = {}
tensor_sizes = set()

# Load CSV files into pandas DataFrames
for setting in nreduce_settings:
    file_path = f"algo/nreduce/allreduce_bandwidth_results_{setting}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data[setting] = df
        tensor_sizes.update(df['Tensor Size (bytes)'])
    else:
        print(f"Warning: {file_path} not found.")

# Sort tensor sizes for consistent x-axis
tensor_sizes = sorted(tensor_sizes)

# Plot Bandwidth by Message Size
plt.figure(figsize=(10, 5))
for setting, df in data.items():
    plt.plot(df['Tensor Size (bytes)'], df['Average Bandwidth (GiB/s)'], label=setting, marker='o')

plt.xlabel('Message Size')
plt.ylabel('Average Bandwidth (GiB/s)')
plt.title('AllReduce Bandwidth Comparison for nreduce Settings')
plt.legend()
plt.grid(True)
plt.xscale('log')
#plt.yscale('log')

# Apply custom x-axis labels using the format_size function on actual tensor sizes
xticks = tensor_sizes  # Use the measured tensor sizes as x-ticks
xticklabels = [format_size(int(tick)) for tick in xticks]  # Format each tick label
plt.xticks(xticks, xticklabels, rotation=45)  # Rotate the x-axis labels by 45 degrees

# Set the x-axis range from 4 KiB (4096 bytes) to 1.5 GiB (1.5 * 2^30 bytes)
plt.xlim(4096, int(1.5 * 2**30))

plt.tight_layout()
plt.savefig('allreduce_nreduce_comparison.png')  # Save the plot as a PNG file
plt.close()

print("Plot saved as 'allreduce_nreduce_comparison.png'")
