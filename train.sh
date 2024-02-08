#!/bin/bash

# Increase tcmalloc report threshold to 36 GB
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=38654705664

# Make sure display manager keeps its grubby mits off of the GPUs
#sudo systemctl stop `cat /etc/X11/default-display-manager`

# Set Tensorflow log level if desired:
# 0 - (all logs shown)
# 1 - filter out INFO logs 
# 2 - filter out WARNING logs
# 3 - filter out ERROR logs

export TF_CPP_MIN_LOG_LEVEL=2

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=1,2

# Prevent tensorflow from automapping all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Clean up output directories
# rm --force training_checkpoints/*
# rm --force gan_output/*

# Start GANN training run
python dc-gan3.py