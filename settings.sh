#!/bin/bash

# Disable HT
echo "Disabling HT..."
echo off | sudo tee /sys/devices/system/cpu/smt/control

# Disable ASLR
echo "Disabling ASLR..."
sudo sysctl -w kernel.randomize_va_space=0

echo "Script execution completed!"