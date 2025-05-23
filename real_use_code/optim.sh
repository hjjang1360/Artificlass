#!/bin/bash
# This script runs six Python files in groups of two concurrently.
# The use of nohup ensures the processes keep running even after terminal logout.

# Create a directory for log files if it doesn't exist
LOG_DIR="./output_log"
mkdir -p "${LOG_DIR}"

# Group 1: Run cifar10_doubleCNN.py concurrently
#nohup python ./tripCNN.py > "${LOG_DIR}/tripCNN.log" 2>&1 &
#wait  # Wait for both to finish

# nohup python ./doubleCNN.py > "${LOG_DIR}/doubleCNN.log" 2>&1 &
# wait  # Wait for both to finish

# nohup python ./resnet_18.py > "${LOG_DIR}/resnet_18.log" 2>&1 &
# wait  # Wait for both to finish
nohup python ./jpg_resnet50_pre.py > "${LOG_DIR}/jpg_resnet50_pre.log" 2>&1 &
wait  # Wait for both to finish

echo "All scripts have finished."
