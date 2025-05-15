#!/bin/bash
echo "Sourcing CCL vars..."
source /opt/intel/oneapi/ccl/latest/env/vars.sh
echo "Sourcing OneAPI vars..."
source /opt/intel/oneapi/setvars.sh
echo "Activating virtual environment..."
source venv/bin/activate

#profiling
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
sudo sysctl -w kernel.yama.ptrace_scope=0