#!/bin/bash

# Convenience script to generate video of training
# from model checkpoints and latent point

# Which run date and specimen to use
RUN_DATE='2024-02-11'
SPECIMEN_LATEN_POINTS='5695.3_latent_points.pkl'

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/.venv/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/.venv/lib/python3.8/site-packages/tensorrt/

# Increase tcmalloc report threshold to 36 GB
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=38654705664

# Set Tensorflow log level if desired:
# 0 - (all logs shown)
# 1 - filter out INFO logs 
# 2 - filter out WARNING logs
# 3 - filter out ERROR logs
export TF_CPP_MIN_LOG_LEVEL=3

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0

# Prevent tensorflow from automapping all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Make images
python ./skylines/training_sequence.py $RUN_DATE $SPECIMEN_LATEN_POINTS