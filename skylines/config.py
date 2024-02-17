import os
from datetime import datetime

# Project name
PROJECT_NAME='skylines'
CURRENT_DATE=datetime.today().strftime('%Y-%m-%d')

########################################################################
# Paths and directories ################################################
########################################################################

# Get path to this config file, we will use this
# to define other paths to data files etc.
PATH=os.path.dirname(os.path.realpath(__file__))

IMAGE_DIR=f'{PATH}/data/image_datasets'
RAW_IMAGE_DIR=f'{IMAGE_DIR}/raw_images'
PROCESSED_IMAGE_DIR=f'{IMAGE_DIR}/training_images'
TRAINING_IMAGE_DIR=PROCESSED_IMAGE_DIR
# MODEL_CHECKPOINT_DIR=f'{path}/data/{path_date}/training_checkpoints'
# SPECIMEN_DIR=f'{path}/data/{path_date}/specimens'
# IMAGE_OUTPUT_DIR=f'{path}/data/{path_date}/gan_output'
BENCHMARK_DATA_DIR=f'{PATH}/benchmarking'


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
DISCRIMINATOR_LEARNING_RATE=0.00005 #0.00005
GENERATOR_LEARNING_RATE=0.00005 #0.00005
GANN_LEARNING_RATE=0.00005 #0.00005
BATCH_SIZE=int(3 * len(GPUS))
EPOCHS=100000

CHECKPOINT_SAVE_FREQUENCY=1
