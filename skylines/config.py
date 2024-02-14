import os
from datetime import datetime

# Project name and run date
PROJECT_NAME='skylines'
CURRENT_DATE=datetime.today().strftime('%Y-%m-%d')

########################################################################
# Option to resume a training run ######################################
########################################################################

RESUME=True
RESUME_RUN_DATE='2024-02-11'

########################################################################
# Paths and directories ################################################
########################################################################

# Get path to this config file, we will use this
# to define other paths to data files etc.
path=os.path.dirname(os.path.realpath(__file__))

# Use current date or resume data in file paths as needed

if RESUME == True:
    path_date=RESUME_RUN_DATE

elif RESUME == False:
    path_date=CURRENT_DATE

IMAGE_DIR=f'{path}/data/image_datasets'
RAW_IMAGE_DIR=f'{IMAGE_DIR}/raw_images'
PROCESSED_IMAGE_DIR=f'{IMAGE_DIR}/training_images'
TRAINING_IMAGE_DIR=PROCESSED_IMAGE_DIR
MODEL_CHECKPOINT_DIR=f'{path}/data/training_checkpoints/{path_date}'
SPECIMEN_DIR=f'{path}/data/specimens/{path_date}'
IMAGE_OUTPUT_DIR=f'{path}/data/gan_output/{path_date}'
BENCHMARK_DATA_DIR=f'{path}/benchmarking'


########################################################################
# Data related parameters ##############################################
########################################################################

MAX_CONCURRENCY=2
IMAGE_DIM=1024
SHUFFLE_BUFFER=50

########################################################################
# dc-gann parameters ###################################################
########################################################################

GPUS=[
    '/job:localhost/replica:0/task:0/device:GPU:0',
    '/job:localhost/replica:0/task:0/device:GPU:1'
]

GPU_PARALLELISM='central storage'
LATENT_DIM=100
DISCRIMINATOR_LEARNING_RATE=0.00001 #0.00005
GENERATOR_LEARNING_RATE=0.00001 #0.00005
GANN_LEARNING_RATE=0.00001 #0.00005
BATCH_SIZE=int(3 * len(GPUS))
EPOCHS=100000

CHECKPOINT_SAVE_FREQUENCY=1
