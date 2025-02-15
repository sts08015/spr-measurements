import pandas as pd
import matplotlib.pyplot as plt

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

# Load CSV files into pandas DataFrames
df_12 = pd.read_csv('allreduce_bandwidth_results_12.csv')
df_24 = pd.read_csv('allreduce_bandwidth_results_24.csv')
df_48 = pd.read_csv('allreduce_bandwidth_results_48.csv')

# Get the unique tensor sizes from the data (you can also adjust for a finer granularity)
tensor_sizes = sorted(set(df_12['Tensor Size (bytes)']).union(set(df_24['Tensor Size (bytes)'])).union(set(df_48['Tensor Size (bytes)'])))

# Plot Time by Message Size
plt.figure(figsize=(10, 5))
plt.plot(df_12['Tensor Size (bytes)'], df_12['Average Time (seconds)'], label='12 processes', marker='o')
plt.plot(df_24['Tensor Size (bytes)'], df_24['Average Time (seconds)'], label='24 processes', marker='o')
plt.plot(df_48['Tensor Size (bytes)'], df_48['Average Time (seconds)'], label='48 processes', marker='o')
plt.xlabel('Message Size')
plt.ylabel('Average Time (seconds)')
plt.title('Time by Message Size')
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
plt.savefig('time_by_message_size.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to avoid overlap with the next one

# Plot Bandwidth by Message Size
plt.figure(figsize=(10, 5))
plt.plot(df_12['Tensor Size (bytes)'], df_12['Average Bandwidth (GiB/s)'], label='12 processes', marker='o')
plt.plot(df_24['Tensor Size (bytes)'], df_24['Average Bandwidth (GiB/s)'], label='24 processes', marker='o')
plt.plot(df_48['Tensor Size (bytes)'], df_48['Average Bandwidth (GiB/s)'], label='48 processes', marker='o')
plt.xlabel('Message Size')
plt.ylabel('Average Bandwidth (GiB/s)')
plt.title('Bandwidth by Message Size')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Apply custom x-axis labels using the format_size function on actual tensor sizes
plt.xticks(xticks, xticklabels, rotation=45)  # Rotate the x-axis labels by 45 degrees

# Set the x-axis range from 4 KiB (4096 bytes) to 1.5 GiB (1.5 * 2^30 bytes)
plt.xlim(4096, int(1.5 * 2**30))

plt.tight_layout()
plt.savefig('bandwidth_by_message_size.png')  # Save the plot as a PNG file
plt.close()  # Close the plot

print("Plots saved as 'time_by_message_size.png' and 'bandwidth_by_message_size.png'")
