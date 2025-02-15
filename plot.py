import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files into pandas DataFrames
df_12 = pd.read_csv('allreduce_bandwidth_results_12.csv')
df_24 = pd.read_csv('allreduce_bandwidth_results_24.csv')
df_48 = pd.read_csv('allreduce_bandwidth_results_48.csv')

# Plot Time by Message Size
plt.figure(figsize=(10, 5))
plt.plot(df_12['Tensor Size (bytes)'], df_12['Average Time (seconds)'], label='12 processes', marker='o')
plt.plot(df_24['Tensor Size (bytes)'], df_24['Average Time (seconds)'], label='24 processes', marker='o')
plt.plot(df_48['Tensor Size (bytes)'], df_48['Average Time (seconds)'], label='48 processes', marker='o')
plt.xlabel('Message Size (bytes)')
plt.ylabel('Average Time (seconds)')
plt.title('Time by Message Size')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('time_by_message_size.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to avoid overlap with the next one

# Plot Bandwidth by Message Size
plt.figure(figsize=(10, 5))
plt.plot(df_12['Tensor Size (bytes)'], df_12['Average Bandwidth (GiB/s)'], label='12 processes', marker='o')
plt.plot(df_24['Tensor Size (bytes)'], df_24['Average Bandwidth (GiB/s)'], label='24 processes', marker='o')
plt.plot(df_48['Tensor Size (bytes)'], df_48['Average Bandwidth (GiB/s)'], label='48 processes', marker='o')
plt.xlabel('Message Size (bytes)')
plt.ylabel('Average Bandwidth (GiB/s)')
plt.title('Bandwidth by Message Size')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('bandwidth_by_message_size.png')  # Save the plot as a PNG file
plt.close()  # Close the plot

print("Plots saved as 'time_by_message_size.png' and 'bandwidth_by_message_size.png'")
