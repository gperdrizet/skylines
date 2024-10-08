#!/bin/bash

# Convenience script to generate video of training
# from model checkpoints and latent point

# Which run date and specimen latent point to use
RUN_DATE='2024-02-17'
SPECIMEN_LATEN_POINT='18218.6'

# Resume or add to a previous frame generation run
RESUME='False'

# Frame number to resume from. Is used as index of model in model 
# paths list and number for frame output. This alows the 
# generation of squentialy numbered frames from non-sequential
# model snapshots
RESUME_FRAME='0'

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
export CUDA_VISIBLE_DEVICES=1

# Prevent tensorflow from automapping all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Make images
python ./skylines/make_training_frames.py $RUN_DATE $SPECIMEN_LATEN_POINT $RESUME $RESUME_FRAME