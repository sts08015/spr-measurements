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

# List of algorithms
algorithms = ["2d", "direct", "double_tree", "nreduce", "rabenseifner", "recursive_doubling", "ring"]

# Dictionary to store data from all algorithms
data = {}

tensor_sizes = set()


NPROC=12
# Load CSV files into pandas DataFrames
for algo in algorithms:
    file_path = f"algo/{algo}/allreduce_bandwidth_results_{algo}_{NPROC}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data[algo] = df
        tensor_sizes.update(df['Tensor Size (bytes)'])
    else:
        print(f"Warning: {file_path} not found.")

# Sort tensor sizes for consistent x-axis
tensor_sizes = sorted(tensor_sizes)

# Plot Bandwidth by Message Size
plt.figure(figsize=(10, 5))
for algo, df in data.items():
    plt.plot(df['Tensor Size (bytes)'], df['Average Bandwidth (GiB/s)'], label=algo, marker='o')

plt.xlabel('Message Size')
plt.ylabel('Average Bandwidth (GiB/s)')
plt.title('AllReduce Bandwidth Across Different Algorithms')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Apply custom x-axis labels using the format_size function on actual tensor sizes
xticks = tensor_sizes  # Use the measured tensor sizes as x-ticks
xticklabels = [format_size(int(tick)) for tick in xticks]  # Format each tick label
plt.xticks(xticks, xticklabels, rotation=45)  # Rotate the x-axis labels by 45 degrees

# Set the x-axis range from 4 KiB (4096 bytes) to 1.5 GiB (1.5 * 2^30 bytes)
plt.xlim(4096, int(1.5 * 2**30))

plt.tight_layout()
plt.savefig(f'allreduce_bandwidth_comparison_{NPROC}.png')  # Save the plot as a PNG file
plt.close()

print("Plot saved as 'allreduce_bandwidth_comparison.png'")