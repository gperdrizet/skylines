#!/bin/bash

# Convenience script to start DC-GANN training run. 
# Sets some environment variables. See skylines/config.py for
# additional run options.

RESUME=True
RESUME_RUN_DATE='2024-02-11'

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
export TF_CPP_MIN_LOG_LEVEL=2

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=1,2

# Prevent tensorflow from automapping all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Start GANN training run
python ./skylines/train.py $RESUME $RESUME_RUN_DATE