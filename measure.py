import torch
import torch.nn.parallel
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import os
import time
import matplotlib.pyplot as plt
import csv
import sys

# Setup environment variables for distributed processing
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

# Initialize the process group with oneCCL
backend = 'ccl'
dist.init_process_group(backend)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print(f"Rank {my_rank}/{my_size} initialized")

tensor_sizes = [2**i for i in range(13, 31)]  #13, 31

# Run AllReduce for each tensor size
results = []
times = []
bandwidths = []

def format_size(size):
    if size < 2**10:
        return f"{size} B"
    elif size < 2**20:
        return f"{size / 2**10:.1f} KiB"
    elif size < 2**30:
        return f"{size / 2**20:.1f} MiB"
    else:
        return f"{size / 2**30:.1f} GiB"

def warmup():
    for _ in range(10):  # Perform 10 dummy operations for warmup
        x = torch.ones([8*1024], dtype=torch.float32)  # Dummy tensor (8 KiB)
        dist.all_reduce(x)
        dist.barrier()

for size in tensor_sizes:
    elapsed_times = []
    bandwidths_for_size = []
    
    # Warmup before actual measurements
    warmup()

    # Repeat the experiment 20 times and average the results
    for _ in range(20):
        x = torch.ones([size//4], dtype=torch.float32)  #fp32
        dist.barrier()  # Synchronize before timing

        start_time = time.time()
        dist.all_reduce(x)
        dist.barrier()  # Synchronize after operation, might incur sync overhead
        elapsed_time = time.time() - start_time

        # Compute bandwidth in GiB/s
        tensor_size_bytes = x.numel() * 4  # fp32
        bandwidth_GiBps = (tensor_size_bytes / (2**30)) / elapsed_time  # Convert to GiB and compute bandwidth

        elapsed_times.append(elapsed_time)
        bandwidths_for_size.append(bandwidth_GiBps)
    
    # Calculate the average elapsed time and bandwidth for this tensor size
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    avg_bandwidth_GiBps = sum(bandwidths_for_size) / len(bandwidths_for_size)

    # Store the results
    results.append((size, avg_elapsed_time, avg_bandwidth_GiBps))
    times.append(avg_elapsed_time)
    bandwidths.append(avg_bandwidth_GiBps)

    print(f"Tensor size: {format_size(tensor_size_bytes)} -> Average Time: {avg_elapsed_time:.6f} s, Average Bandwidth: {avg_bandwidth_GiBps:.2f} GiB/s")
    sys.stdout.flush()  #to check progress
    
dist.barrier()  # Final sync before exiting

# Save results to a CSV file
if my_rank == 0:
    csv_file = 'allreduce_bandwidth_results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tensor Size (bytes)', 'Average Time (seconds)', 'Average Bandwidth (GiB/s)'])
        
        # Write each result row
        for size, avg_time, avg_bandwidth in results:
            writer.writerow([size, avg_time, avg_bandwidth])
    
    print(f"Results saved to {csv_file}")
