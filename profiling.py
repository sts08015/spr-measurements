
import torch
import torch.nn.parallel
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import os
import time

os.environ['MASTER_ADDR'] = '127.0.0.1' #local
os.environ['MASTER_PORT'] = '29500' #port
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0)) #rank number
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))   #total # of ranks

backend = 'ccl' #oneCCL
dist.init_process_group(backend)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

size = 128*1024*1024
x = torch.ones([size//4], dtype=torch.float32)

with torch.autograd.profiler.profile(record_shapes=True) as prof:
    start = time.time()
    for _ in range(1):
        dist.all_reduce(x)
        dist.barrier()  #sync
    end = time.time()


print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
if my_rank == 0:
    print('Elapsed Time: ',end - start)
    print("################################\n#########################################\n####################fojfoiqjfjff\n#j3jgoqjfqojfqojfqojfo")