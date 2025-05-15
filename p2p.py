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

# Message sizes to test
message_sizes = [2**i for i in range(13, 31)]

# Run P2P for each message size
p2p_results = []

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
        x = torch.ones([8*1024], dtype=torch.float32)  # Dummy tensor (32 KiB)
        dist.all_reduce(x)
        dist.barrier()

def measure_p2p_time():
    warmup()  # Warmup phase

    # Loop through different message sizes
    for size in message_sizes:
        elapsed_times = []
        bandwidths_for_size = []

        # Only perform P2P measurements on one process (rank 0 sends data, rank 1 receives data)
        if my_rank == 0:
            x = torch.ones([size], dtype=torch.float32)  # Create tensor to send
            dist.barrier()  # Synchronize

            # Perform the P2P measurement
            start_time = time.time()
            dist.send(tensor=x, dst=1)  # Send tensor to rank 1
            dist.barrier()  #sync
            elapsed_time = time.time() - start_time

            # Compute bandwidth in GiB/s
            tensor_size_bytes = x.numel() * 4  # fp32
            bandwidth_GiBps = (tensor_size_bytes / (2**30)) / elapsed_time  # Convert to GiB and compute bandwidth

            elapsed_times.append(elapsed_time)
            bandwidths_for_size.append(bandwidth_GiBps)

            print(f"Rank {my_rank} -> Sent tensor of size {format_size(size)} to rank 1")
            print(f"Send Time: {elapsed_time:.6f} seconds, Bandwidth: {bandwidth_GiBps:.2f} GiB/s")
            sys.stdout.flush()  #to check progress

        elif my_rank == 1:
            x = torch.zeros([size], dtype=torch.float32)  # Create tensor to receive
            dist.barrier()  # Synchronize

            # Perform the P2P measurement
            start_time = time.time()
            dist.recv(tensor=x, src=0)  # Receive tensor from rank 0
            dist.barrier()  #sync
            elapsed_time = time.time() - start_time

            # Compute bandwidth in GiB/s
            tensor_size_bytes = x.numel() * 4  # fp32
            bandwidth_GiBps = (tensor_size_bytes / (2**30)) / elapsed_time  # Convert to GiB and compute bandwidth

            elapsed_times.append(elapsed_time)
            bandwidths_for_size.append(bandwidth_GiBps)

            print(f"Rank {my_rank} -> Received tensor of size {format_size(size)} from rank 0")
            print(f"Receive Time: {elapsed_time:.6f} seconds, Bandwidth: {bandwidth_GiBps:.2f} GiB/s")
            sys.stdout.flush()  #to check progress

        # Average the results for each size
        avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        avg_bandwidth_GiBps = sum(bandwidths_for_size) / len(bandwidths_for_size)

        # Store the results for this size
        p2p_results.append((size, avg_elapsed_time, avg_bandwidth_GiBps))

# Measure Point-to-Point time
measure_p2p_time()

# Save results to a CSV file
if my_rank == 0:
    csv_file = 'p2p_bandwidth_results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tensor Size (bytes)', 'Average Time (seconds)', 'Average Bandwidth (GiB/s)'])
        
        # Write each result row
        for size, avg_time, avg_bandwidth in p2p_results:
            writer.writerow([size, avg_time, avg_bandwidth])
    
    print(f"Results saved to {csv_file}")

